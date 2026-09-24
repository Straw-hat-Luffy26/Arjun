//! How a worker reads and writes the task's shared memory.
//!
//! ## One graph service, not a second one
//!
//! Every read and write here goes through [`MemoryGraph`] — the same store the
//! parent's `context.refresh` compiles from and the same one
//! `memory_api::recall_authorized` answers out of. A worker does not get a
//! private side channel, and it does not hand its findings back through the
//! result payload alone: it *commits* them, and what comes back to the parent is
//! an item id and a revision the parent can go and read.
//!
//! That is what makes one worker's output reach another. Agent A publishes a
//! fact; agent B queries the graph and finds it. There is no transcript passed
//! between them, no shared prompt, and nothing that depends on A and B running
//! in the same process.
//!
//! ## Why a revision, and not "just read it"
//!
//! Because A and B are concurrent. B asking "is A's fact there yet" by looking
//! and finding nothing cannot tell "A has not published" from "A published and
//! my read was a moment early". So a worker is handed a revision it must be at
//! or past — [`Requirement`] — and either waits for the changefeed to reach it
//! or refuses. Both are answers; guessing is not.
//!
//! ## Why the refresh is between calls, not during one
//!
//! A worker takes a snapshot, works from it, and takes another when it is ready
//! for one. It never merges a concurrent edit into a request already in flight,
//! because a prompt that changed half way through is a prompt nobody can
//! reproduce from the record — and reproducing it is the whole point of
//! recording the revision it was compiled at.
//!
//! ## Conflicts are recorded, never resolved by deletion
//!
//! Two workers can reach contradictory conclusions about the same thing; that is
//! a normal outcome of asking two specialists, not a bug. The rule is
//! [`may_supersede`]: a receipt may correct a model's guess, a person outranks
//! everything, and a model may not quietly overwrite an established fact. Where
//! it refuses, this records a contradiction so both survive and somebody can
//! decide.

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::agent_runtime::memory::Acl;
use crate::identity::Session;
use crate::knowledge::graph::runtime_memory::{
    item_id, may_supersede, ArtifactRef, ItemStatus, MemoryItem, MemoryKind, MemoryScope,
    Provenance, SourceRef,
};
use crate::knowledge::graph::runtime_store::MemoryGraph;
use crate::policy::Classification;

/// How often the changefeed is re-read while a worker waits for a revision.
///
/// 50 ms: long enough that a worker waiting a few seconds is not spinning, short
/// enough that a fact published by a sibling reaches a waiting worker inside one
/// human-noticeable moment.
const POLL_INTERVAL: std::time::Duration = std::time::Duration::from_millis(50);

/// The graph position a worker's inputs were authorised at.
///
/// Carried on the packet so a child works from the cursor the parent chose for
/// it. Two children given the same requirement read the same world, which is
/// what makes their results comparable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", tag = "mode")]
pub enum Requirement {
    /// Whatever the graph holds when the worker looks. The ordinary case for a
    /// worker with no sibling to wait for, and the default: a dispatch that
    /// says nothing about ordering is one with nothing to wait for.
    #[default]
    Latest,
    /// The worker must see at least this changefeed position, and waits for it.
    ///
    /// This is how B reads A's result: the parent hands B the revision A's
    /// commit landed at, and B does not start until the store has it.
    AtLeast { graph_revision: i64 },
    /// The worker must already be at this position, and is refused if it is not.
    ///
    /// For work that must not proceed on a stale world and must not block
    /// either — a check that is only meaningful against the state the parent
    /// just saw.
    Exactly { graph_revision: i64 },
}

impl Requirement {
    pub fn describe(&self) -> String {
        match self {
            Requirement::Latest => "whatever the graph holds now".to_string(),
            Requirement::AtLeast { graph_revision } => {
                format!("graph revision {graph_revision} or later")
            }
            Requirement::Exactly { graph_revision } => {
                format!("graph revision {graph_revision} exactly")
            }
        }
    }

    /// The revision this names, where it names one.
    pub fn revision(&self) -> Option<i64> {
        match self {
            Requirement::Latest => None,
            Requirement::AtLeast { graph_revision } | Requirement::Exactly { graph_revision } => {
                Some(*graph_revision)
            }
        }
    }
}

/// Why a worker could not be given the world it was asked to work from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NotAvailable {
    /// The graph did not reach the required revision inside the deadline.
    StillBehind {
        wanted: i64,
        reached: i64,
        waited_ms: u64,
    },
    /// The graph is past a revision the work was only meaningful at.
    MovedOn { wanted: i64, now: i64 },
    /// This deployment has no runtime memory graph.
    NoGraph,
    /// The store could not be read.
    Storage { detail: String },
}

impl NotAvailable {
    pub fn explain(&self) -> String {
        match self {
            Self::StillBehind {
                wanted,
                reached,
                waited_ms,
            } => format!(
                "this worker needed the shared memory to be at revision {wanted} before it could \
                 start — that is where the result it depends on was written — and after \
                 {waited_ms}ms it was still at {reached}. Nothing was done. The work it was \
                 waiting for has not finished."
            ),
            Self::MovedOn { wanted, now } => format!(
                "this worker was to check the shared memory exactly as it stood at revision \
                 {wanted}, and it has since moved to {now}. Nothing was done: the check would \
                 have been against a different world from the one it was asked about."
            ),
            Self::NoGraph => "this deployment has no runtime memory graph, so a worker has \
                              nowhere to read shared facts from or publish its own to."
                .to_string(),
            Self::Storage { detail } => format!("the shared memory could not be read: {detail}"),
        }
    }
}

/// What a worker's commit did.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Published {
    pub item_id: String,
    /// Which revision of that item.
    pub revision: u64,
    /// The changefeed position it landed at. This is the number a sibling is
    /// given to wait for.
    pub graph_revision: i64,
    /// `established` or `proposed`, from the admission rules — not from what the
    /// worker claimed.
    pub status: String,
    /// Why it is at that status.
    pub because: String,
    /// True when an identical commit had already been applied and this changed
    /// nothing. A retry after a worker died mid-publish lands here.
    pub duplicate: bool,
    /// Items this was recorded as disagreeing with, where the conflict model
    /// refused a supersede.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub conflicts_with: Vec<String>,
}

impl Published {
    /// One line a parent can read, and a sibling can act on.
    pub fn describe(&self) -> String {
        let conflict = if self.conflicts_with.is_empty() {
            String::new()
        } else {
            format!(
                ", recorded as disagreeing with {} existing item(s)",
                self.conflicts_with.len()
            )
        };
        format!(
            "{} revision {} at graph revision {} ({}){}",
            self.item_id, self.revision, self.graph_revision, self.status, conflict
        )
    }
}

/// What a worker is publishing.
///
/// A struct rather than a dozen arguments, and every field is something the
/// worker actually established: the sources it read, the artifacts it produced,
/// and the key that makes a retry harmless. There is no field for "what the
/// model said" — [`Provenance`] decides that, and it is set by the caller from
/// what happened rather than by the worker describing itself.
#[derive(Debug, Clone)]
pub struct Claim {
    pub kind: MemoryKind,
    pub content: String,
    pub sources: Vec<SourceRef>,
    pub artifacts: Vec<ArtifactRef>,
    pub confidence: Option<f32>,
    /// What this was written in reaction to. Lets a reader reconstruct the order
    /// two workers wrote in without trusting two processes' clocks.
    pub causal_parents: Vec<String>,
    /// Supplied so a retry after a worker died is recognised as the same write
    /// rather than performed twice. Derived from the packet, not generated.
    pub idempotency_key: String,
}

/// One worker's view of the task's shared memory.
///
/// Holds the agent id rather than a model id: memory is owned by the agent, and
/// an agent's model changes without its memory moving — the rule
/// `agents::ModelBinding` documents and `model_transition` enforces.
pub struct TaskMemory {
    graph: Arc<MemoryGraph>,
    task_id: String,
    agent_id: String,
    run_id: String,
    classification: Classification,
    project_id: Option<String>,
}

impl TaskMemory {
    pub fn new(
        graph: Arc<MemoryGraph>,
        task_id: impl Into<String>,
        agent_id: impl Into<String>,
        run_id: impl Into<String>,
        classification: Classification,
        project_id: Option<String>,
    ) -> Self {
        Self {
            graph,
            task_id: task_id.into(),
            agent_id: agent_id.into(),
            run_id: run_id.into(),
            classification,
            project_id,
        }
    }

    fn scope(&self) -> MemoryScope {
        MemoryScope::Task {
            task_id: self.task_id.clone(),
        }
    }

    /// Where the changefeed is now.
    pub fn revision(&self) -> Result<i64, NotAvailable> {
        self.graph
            .graph_revision()
            .map_err(|error| NotAvailable::Storage {
                detail: error.explain(),
            })
    }

    /// Blocks until the graph holds what this worker was told to work from.
    ///
    /// Returns the revision actually reached, which is at or past the one
    /// required — never an approximation of it. A worker that cannot be given
    /// the world it was asked about does not start.
    pub async fn await_requirement(
        &self,
        requirement: Requirement,
        within: std::time::Duration,
    ) -> Result<i64, NotAvailable> {
        let started = std::time::Instant::now();
        match requirement {
            Requirement::Latest => self.revision(),
            Requirement::Exactly { graph_revision } => {
                // Deliberately not a wait. "Exactly" means the check is only
                // meaningful against that state, so a graph that has moved on is
                // a refusal rather than something to wait out.
                let now = self.revision()?;
                if now == graph_revision {
                    Ok(now)
                } else if now < graph_revision {
                    Err(NotAvailable::StillBehind {
                        wanted: graph_revision,
                        reached: now,
                        waited_ms: 0,
                    })
                } else {
                    Err(NotAvailable::MovedOn {
                        wanted: graph_revision,
                        now,
                    })
                }
            }
            Requirement::AtLeast { graph_revision } => loop {
                let now = self.revision()?;
                if now >= graph_revision {
                    return Ok(now);
                }
                if started.elapsed() >= within {
                    return Err(NotAvailable::StillBehind {
                        wanted: graph_revision,
                        reached: now,
                        waited_ms: started.elapsed().as_millis() as u64,
                    });
                }
                // Polled rather than notified, and the reason is honest rather
                // than ideal: the changefeed is a SQLite table with no
                // notification channel, and a worker that registered a listener
                // would be a worker holding a lock across its own wait. The
                // interval is short enough that a sibling's publish reaches this
                // inside one human moment.
                tokio::time::sleep(POLL_INTERVAL).await;
            },
        }
    }

    /// Everything in this task's memory this session may read.
    ///
    /// The authorisation is the graph's, not this module's:
    /// [`MemoryGraph::snapshot`] filters by `readable_by` before anything comes
    /// back, so a worker with a narrower clearance than its parent genuinely
    /// sees less — which is the point of giving it one.
    pub fn read(&self, session: &Session) -> Result<Vec<MemoryItem>, NotAvailable> {
        self.graph
            .snapshot(session, &self.scope(), self.project_id.as_deref())
            .map_err(|error| NotAvailable::Storage {
                detail: error.explain(),
            })
    }

    /// Items of one kind, newest revision first.
    ///
    /// The query a sibling uses to find what another worker published. Kept here
    /// rather than written out at each call site, so "what B reads" is one
    /// expression that can be pointed at.
    pub fn read_kind(
        &self,
        session: &Session,
        kind: MemoryKind,
    ) -> Result<Vec<MemoryItem>, NotAvailable> {
        let mut found: Vec<MemoryItem> = self
            .read(session)?
            .into_iter()
            .filter(|item| item.kind == kind)
            .filter(|item| item.status.usable_as_evidence())
            .collect();
        found.sort_by(|a, b| b.revision.cmp(&a.revision).then(a.item_id.cmp(&b.item_id)));
        Ok(found)
    }

    /// Writes one thing a worker established into the task's shared memory.
    ///
    /// ## Why the provenance is a parameter and not a claim
    ///
    /// Because it decides what the item may do. A `ToolReceipt` may correct a
    /// model's guess; a `Model` proposal may not overwrite an established fact.
    /// A worker that could name its own provenance could name the stronger one,
    /// so the caller sets it from what actually happened — a receipt when the
    /// durable log holds the event, a proposal otherwise.
    pub fn publish(
        &self,
        claim: Claim,
        provenance: Provenance,
        model_id: Option<String>,
        session: &Session,
    ) -> Result<Published, NotAvailable> {
        let now = chrono::Utc::now().to_rfc3339();
        let mut item = MemoryItem {
            item_id: item_id(),
            revision: 1,
            kind: claim.kind,
            agent_id: self.agent_id.clone(),
            scope: self.scope(),
            classification: self.classification,
            acl: Acl::for_classification(self.classification, self.project_id.as_deref()),
            creator_model_id: model_id,
            creator_run_id: Some(self.run_id.clone()),
            provenance,
            content: claim.content,
            sources: claim.sources,
            artifacts: claim.artifacts,
            confidence: claim.confidence,
            // Set by `admit` inside `commit`. Whatever is put here is replaced,
            // which is what stops a worker publishing itself as established.
            status: ItemStatus::Proposed,
            valid_from: now.clone(),
            valid_until: None,
            supersedes: None,
            conflicts_with: Vec::new(),
            causal_parents: claim.causal_parents,
            idempotency_key: Some(claim.idempotency_key),
            applies_to: None,
            created_at: now.clone(),
            updated_at: now,
        };

        // Anything already in this task's memory that says something different
        // about the same thing. A contradiction is recorded rather than
        // resolved, and which of the two may replace the other is the graph's
        // rule.
        let existing = self.read(session)?;
        let contradicted = contradictions(&item, &existing);
        item.conflicts_with = contradicted.clone();

        let committed =
            self.graph
                .commit(item, None, &[])
                .map_err(|error| NotAvailable::Storage {
                    detail: error.explain(),
                })?;

        Ok(Published {
            item_id: committed.item_id,
            revision: committed.revision,
            graph_revision: committed.graph_revision,
            status: committed.status.as_str().to_string(),
            because: committed.because,
            duplicate: committed.duplicate,
            conflicts_with: contradicted,
        })
    }
}

/// Which existing items a new claim disagrees with.
///
/// ## What counts as a contradiction here
///
/// A claim of the same kind, about the same subject, from a *different* agent,
/// whose content differs. Deliberately narrow: two workers reporting the same
/// figure are not in conflict, and a worker revising its own earlier proposal is
/// not either — that is a supersede, and [`may_supersede`] decides whether it is
/// allowed.
///
/// The subject is the leading clause of the content, which is the shape every
/// claim this module writes takes: `"<subject>: <what was found>"`. A crude key,
/// and it is the honest one available without an ontology — a claim that cannot
/// be matched is reported as conflicting with nothing rather than with
/// everything.
fn contradictions(candidate: &MemoryItem, existing: &[MemoryItem]) -> Vec<String> {
    let Some(subject) = subject_of(&candidate.content) else {
        return Vec::new();
    };
    existing
        .iter()
        .filter(|held| held.kind == candidate.kind)
        .filter(|held| held.agent_id != candidate.agent_id)
        .filter(|held| held.status.usable_as_evidence())
        .filter(|held| subject_of(&held.content).as_deref() == Some(subject.as_str()))
        .filter(|held| held.content != candidate.content)
        // Only where the conflict model refuses a clean replacement. Where it
        // permits one — a receipt correcting a model's guess — the newer item
        // stands on its own, and recording a disagreement as well would leave
        // two rows a reader has to reconcile for no reason.
        .filter(|held| may_supersede(candidate, held).is_err())
        .map(|held| held.item_id.clone())
        .collect()
}

/// The leading clause of a claim, lowercased.
fn subject_of(content: &str) -> Option<String> {
    let trimmed = content.trim();
    let subject = trimmed.split(':').next()?.trim();
    if subject.is_empty() || subject.len() == trimmed.len() {
        return None;
    }
    Some(subject.to_ascii_lowercase())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::{Role, User};

    fn session() -> Session {
        Session::open(User::new("priya", "Priya Sharma", vec![Role::Employee]))
    }

    fn memory(graph: Arc<MemoryGraph>, agent: &str) -> TaskMemory {
        TaskMemory::new(
            graph,
            "task-1",
            agent,
            "run-1",
            Classification::Internal,
            None,
        )
    }

    fn claim(content: &str, key: &str) -> Claim {
        Claim {
            kind: MemoryKind::Fact,
            content: content.to_string(),
            sources: vec![SourceRef {
                sha256: "a".repeat(64),
                locator: "page 4".into(),
                extraction_revision: None,
            }],
            artifacts: Vec::new(),
            confidence: Some(0.9),
            causal_parents: Vec::new(),
            idempotency_key: key.to_string(),
        }
    }

    fn receipt() -> Provenance {
        Provenance::ToolReceipt {
            run_id: "run-1".into(),
            tool: "knowledge.search_authorized".into(),
            event_seq: 7,
        }
    }

    /// The acceptance case, in miniature: A publishes, B finds it by querying
    /// the graph. Nothing passed between them but a revision number.
    #[tokio::test]
    async fn one_worker_reads_what_another_published() {
        let graph = Arc::new(MemoryGraph::in_memory().expect("a graph"));
        let session = session();

        let a = memory(Arc::clone(&graph), "ag-retriever");
        let published = a
            .publish(
                claim("commissioning tag: the tag is VX-7741-QRT", "key-a"),
                receipt(),
                Some("model-a".into()),
                &session,
            )
            .expect("A publishes");
        assert!(!published.duplicate);

        // B is a different agent, with its own view of the same task.
        let b = memory(Arc::clone(&graph), "ag-checker");
        let reached = b
            .await_requirement(
                Requirement::AtLeast {
                    graph_revision: published.graph_revision,
                },
                std::time::Duration::from_secs(2),
            )
            .await
            .expect("the graph reaches A's revision");
        assert!(reached >= published.graph_revision);

        let found = b
            .read_kind(&session, MemoryKind::Fact)
            .expect("B queries the graph");
        let theirs = found
            .iter()
            .find(|item| item.content.contains("VX-7741-QRT"))
            .unwrap_or_else(|| panic!("B did not find what A published: {found:?}"));
        // And it is attributed to A, not to B.
        assert_eq!(theirs.agent_id, "ag-retriever");
        assert!(!theirs.sources.is_empty(), "the evidence came with it");
    }

    /// A worker waiting for a revision that never arrives is told that, and does
    /// not start.
    #[tokio::test]
    async fn a_worker_waiting_for_work_that_never_lands_is_told_so() {
        let graph = Arc::new(MemoryGraph::in_memory().expect("a graph"));
        let b = memory(graph, "ag-checker");

        let refusal = b
            .await_requirement(
                Requirement::AtLeast {
                    graph_revision: 9_999,
                },
                std::time::Duration::from_millis(150),
            )
            .await
            .expect_err("it never arrives");
        match &refusal {
            NotAvailable::StillBehind { wanted, .. } => assert_eq!(*wanted, 9_999),
            other => panic!("expected still behind, got {other:?}"),
        }
        assert!(
            refusal.explain().contains("has not finished"),
            "{}",
            refusal.explain()
        );
    }

    /// A check that is only meaningful against one state refuses once the world
    /// moves, rather than waiting for a revision that is already past.
    #[tokio::test]
    async fn a_check_pinned_to_one_state_refuses_once_the_world_moves() {
        let graph = Arc::new(MemoryGraph::in_memory().expect("a graph"));
        let session = session();
        let a = memory(Arc::clone(&graph), "ag-retriever");
        let first = a
            .publish(claim("tag: one", "key-1"), receipt(), None, &session)
            .expect("published");
        a.publish(claim("other: two", "key-2"), receipt(), None, &session)
            .expect("published again");

        let refusal = a
            .await_requirement(
                Requirement::Exactly {
                    graph_revision: first.graph_revision,
                },
                std::time::Duration::from_millis(50),
            )
            .await
            .expect_err("the graph has moved on");
        assert!(matches!(refusal, NotAvailable::MovedOn { .. }));
    }

    /// Two workers disagreeing about the same subject leaves both rows, with the
    /// disagreement recorded — not one row silently replacing the other.
    #[tokio::test]
    async fn two_workers_that_disagree_both_survive_and_the_conflict_is_recorded() {
        let graph = Arc::new(MemoryGraph::in_memory().expect("a graph"));
        let session = session();

        let a = memory(Arc::clone(&graph), "ag-retriever");
        a.publish(
            claim("seal torque: the seal torque is 47.5 N\u{b7}m", "key-a"),
            receipt(),
            None,
            &session,
        )
        .expect("A publishes");

        // B is a model proposal, which may not overwrite a receipt.
        let b = memory(Arc::clone(&graph), "ag-checker");
        let theirs = b
            .publish(
                claim("seal torque: the seal torque is 52.0 N\u{b7}m", "key-b"),
                Provenance::Model {
                    model_id: "model-b".into(),
                    run_id: "run-1".into(),
                },
                Some("model-b".into()),
                &session,
            )
            .expect("B publishes");

        assert_eq!(
            theirs.conflicts_with.len(),
            1,
            "the disagreement was not recorded: {theirs:?}"
        );
        // Both are still there.
        let all = b.read(&session).expect("reads");
        assert_eq!(all.len(), 2, "one row replaced the other: {all:?}");
    }

    /// The same worker republishing the same thing after dying mid-write is one
    /// row, not two.
    #[tokio::test]
    async fn a_retry_after_a_worker_died_is_the_same_write() {
        let graph = Arc::new(MemoryGraph::in_memory().expect("a graph"));
        let session = session();
        let a = memory(Arc::clone(&graph), "ag-retriever");

        let first = a
            .publish(claim("tag: VX-1", "same-key"), receipt(), None, &session)
            .expect("first");
        let again = a
            .publish(claim("tag: VX-1", "same-key"), receipt(), None, &session)
            .expect("retry");

        assert!(again.duplicate, "the retry was written a second time");
        assert_eq!(again.item_id, first.item_id);
        assert_eq!(a.read(&session).expect("reads").len(), 1);
    }

    /// Two workers agreeing about the same subject is not a conflict.
    #[tokio::test]
    async fn two_workers_that_agree_are_not_in_conflict() {
        let graph = Arc::new(MemoryGraph::in_memory().expect("a graph"));
        let session = session();
        memory(Arc::clone(&graph), "ag-a")
            .publish(claim("tag: VX-1", "key-a"), receipt(), None, &session)
            .expect("A");
        let theirs = memory(Arc::clone(&graph), "ag-b")
            .publish(claim("tag: VX-1", "key-b"), receipt(), None, &session)
            .expect("B");
        assert!(theirs.conflicts_with.is_empty(), "{theirs:?}");
    }

    /// A claim with no subject clause conflicts with nothing, rather than with
    /// everything.
    #[test]
    fn an_unmatchable_claim_conflicts_with_nothing() {
        assert_eq!(subject_of("no colon here"), None);
        assert_eq!(subject_of("tag: value"), Some("tag".to_string()));
        assert_eq!(subject_of(":"), None);
    }
}
