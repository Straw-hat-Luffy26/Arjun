//! The workers that actually perform the declared roles.
//!
//! ## What was here before, and why it was not enough
//!
//! Nothing. `SubagentManager` was constructed with no workers at all, so every
//! declared profile was a role this build could not perform: `has_worker`
//! answered false for all five, `lib.rs` logged a warning at every start, and
//! `tool_catalogue` withheld `agent.delegate_readonly` from the model on that
//! basis. The lifecycle, the narrowing, the lanes and the idempotency ledger
//! were all real, and all guarding an empty room.
//!
//! These are the occupants. One type, [`SpecialistWorker`], parameterised by
//! profile — because the profile already declares the role, the tools, the
//! limits and the output shape, and a second place restating any of them would
//! be a second place to get them wrong.
//!
//! ## Two ways a role gets done, and why the findings are the same either way
//!
//! A worker's first choice is a **model loop of its own** — see
//! [`super::child_loop`], which registers a plan for the child's run id and
//! drives `run.start` on the runtime the parent is already using. The child's
//! model then decides which of its tools to call and in what order, inside a
//! plan narrowed to exactly what its profile asked for and its parent held.
//! That is what makes it a worker rather than a subroutine, and it is what lets
//! `code-worker` exist at all: writing a program is not something a fixed
//! routine can do.
//!
//! When there is no runtime — nothing has started one yet — the four mechanical
//! roles fall back to calling their tools directly, in a fixed order. Retrieval
//! is a search and a citation; a calculation check is the deterministic engine
//! in [`crate::orchestrator::calculation`]; extraction is reading pages; review
//! is re-opening a file. `code-worker` has no such fallback and refuses by name.
//!
//! **The findings are built the same way on both paths**, and that is the part
//! that matters. They come from what the gateway *settled* — the tool calls this
//! child actually made and the passages the index actually returned — never from
//! the model's prose. The failure that guards against is the one
//! `certification`'s header describes: a retrieval that returned three passages
//! instead of five reports three passages and reads exactly like success. A
//! model can write a confident summary of a search it did not finish; it cannot
//! manufacture a receipt.
//!
//! The model is chosen by [`super::certification::choose`] and reserved against
//! measured VRAM before any of this starts, so four logically parallel children
//! on one card take turns rather than each getting a fraction of it.
//!
//! ## What a worker is not allowed to do
//!
//! Use a tool its [`EffectivePolicy`] does not grant. Every routine below asks
//! first and refuses by name, so a profile edited to request less genuinely gets
//! less — and a parent whose own plan does not permit a tool cannot hand it to a
//! child, because the policy was filtered from the parent's list before it ever
//! reached here.

use std::sync::Arc;

use async_trait::async_trait;

use crate::agent_runtime::cancellation::CancelToken;
use crate::agent_runtime::events::TaskEventLog;
use crate::identity::Session;
use crate::knowledge::graph::runtime_memory::{ArtifactRef, MemoryKind, Provenance, SourceRef};
use crate::knowledge::graph::runtime_store::MemoryGraph;
use crate::knowledge::KnowledgeIndex;
use crate::orchestrator::tools::ToolName;

use super::graph_io::{Claim, Published, TaskMemory};
use super::inherit::EffectivePolicy;
use super::manager::ChildWorker;
use super::packet::{ChildTaskPacket, InputRef};
use super::result::{ChildResult, ChildStatus, EvidenceRef, Finding};
use super::scheduling::ModelScheduler;

/// How many passages a retrieval worker brings back.
///
/// Six, matching what `LocalToolRunner` clamps a model's request to. A worker
/// that returned forty would be handing the parent a context problem instead of
/// an answer.
pub const RETRIEVAL_LIMIT: usize = 6;

/// Everything the workers need, assembled once at start-up.
///
/// Held behind one `Arc` rather than threaded through each worker separately, so
/// registering a sixth profile is adding a routine rather than rewiring five
/// constructors.
pub struct WorkerServices {
    pub index: Arc<KnowledgeIndex>,
    /// `None` on a deployment whose runtime memory graph could not be opened. A
    /// worker then does its work and says plainly that it could not publish,
    /// rather than reporting a completed handoff that reached nobody.
    pub graph: Option<Arc<MemoryGraph>>,
    pub events: Arc<TaskEventLog>,
    pub scheduler: Arc<ModelScheduler>,
    /// Who the work is attributed to. The same holder `RuntimeDeps` uses, so a
    /// worker sees the session as it is now rather than as it was at start-up.
    pub session: Arc<std::sync::RwLock<Option<Session>>>,
    /// The child's own model loop, on the runtime the parent is using.
    ///
    /// `None` on a deployment with no runtime in reach. A role that needs a
    /// model then refuses by name; the four mechanical roles fall back to doing
    /// their tool work directly, and the record says which path ran — see
    /// `SpecialistWorker::run`.
    pub child_loop: Option<Arc<super::child_loop::ChildLoop>>,
    /// The run's own cancellation tokens.
    ///
    /// Propagated rather than re-derived: a person who pressed Stop stopped the
    /// task, and a child that carried on because it had its own timer would be
    /// a child doing work nobody is waiting for.
    pub cancellations: Arc<crate::agent_runtime::cancellation::RunCancellations>,
}

impl WorkerServices {
    fn session(&self) -> Result<Session, String> {
        self.session
            .read()
            .map_err(|_| "the session lock is poisoned".to_string())?
            .clone()
            .ok_or_else(|| {
                "nobody is signed in, so a worker has no clearance to read anything under"
                    .to_string()
            })
    }
}

/// One worker, serving one declared profile.
pub struct SpecialistWorker {
    profile: String,
    /// The profile's own Markdown body, which is the child's system prompt.
    ///
    /// Held rather than looked up per call: the profiles are compiled once at
    /// start-up and a worker that re-read the file would be a worker whose
    /// instructions could change under a run.
    instructions: String,
    services: Arc<WorkerServices>,
}

impl SpecialistWorker {
    /// Every profile this module can actually perform.
    ///
    /// Named here rather than inferred from the profiles directory, because the
    /// question "is there code for this role" is answered by this list and not
    /// by whether somebody wrote a Markdown file.
    pub const PERFORMABLE: &'static [&'static str] = &[
        "knowledge-retriever",
        "document-extractor",
        "calculation-checker",
        "artifact-reviewer",
        // Registerable now that a child can drive a model — a worker for this
        // role without one would have nothing to write. It is the one profile
        // that *requires* the loop: see `NEEDS_A_MODEL`.
        "code-worker",
    ];

    /// Roles that cannot be performed without a model.
    ///
    /// The other four are mechanical — a search, a calculation, a read, a
    /// re-open — and a deployment whose runtime has not started can still do
    /// them. Writing a program is not mechanical, so `code-worker` refuses
    /// rather than doing a fraction of it.
    pub const NEEDS_A_MODEL: &'static [&'static str] = &["code-worker"];

    pub fn new(
        profile: impl Into<String>,
        instructions: impl Into<String>,
        services: Arc<WorkerServices>,
    ) -> Self {
        Self {
            profile: profile.into(),
            instructions: instructions.into(),
            services,
        }
    }

    /// Registers one worker per performable profile this deployment ships.
    ///
    /// Takes the loaded profiles so a deployment that removed a profile file
    /// does not get a worker for it: the declaration is what makes a role exist,
    /// and this is what makes it performable.
    pub fn register_all(
        profiles: &[super::profile::AgentProfile],
        services: Arc<WorkerServices>,
    ) -> Vec<Arc<dyn ChildWorker>> {
        profiles
            .iter()
            .filter(|profile| Self::PERFORMABLE.contains(&profile.name.as_str()))
            .map(|profile| {
                Arc::new(Self::new(
                    profile.name.clone(),
                    profile.instructions.clone(),
                    Arc::clone(&services),
                )) as Arc<dyn ChildWorker>
            })
            .collect()
    }

    /// The task memory this child reads and writes through.
    fn memory(&self, packet: &ChildTaskPacket, policy: &EffectivePolicy) -> Option<TaskMemory> {
        let graph = self.services.graph.as_ref()?;
        Some(TaskMemory::new(
            Arc::clone(graph),
            &packet.task_id,
            // The agent, not the child. A child id belongs to one attempt; the
            // memory belongs to the worker across all of them.
            if packet.agent_id.trim().is_empty() {
                packet.profile.clone()
            } else {
                packet.agent_id.clone()
            },
            &packet.parent_run_id,
            policy.classification_ceiling,
            None,
        ))
    }

    /// What this child stops on.
    ///
    /// ## Why the run is consulted rather than re-registered
    ///
    /// `RunCancellations::register` *inserts* a fresh token for every id it is
    /// given, so a child registering under its parent's run id would replace
    /// the run's own token with an uncancelled one — and a person who had
    /// already pressed Stop would watch the child carry on. That is a real
    /// defect and it is the reason this is two things rather than one: the
    /// child registers only itself, and the run is asked separately at every
    /// checkpoint.
    fn cancellation(&self, packet: &ChildTaskPacket) -> Stopping {
        Stopping {
            own: self
                .services
                .cancellations
                .register(&[packet.child_id.as_str()]),
            cancellations: Arc::clone(&self.services.cancellations),
            run_id: packet.parent_run_id.clone(),
        }
    }
}

#[async_trait]
impl ChildWorker for SpecialistWorker {
    fn profile(&self) -> &str {
        &self.profile
    }

    async fn run(
        &self,
        packet: &ChildTaskPacket,
        policy: &EffectivePolicy,
    ) -> Result<ChildResult, String> {
        let cancel = self.cancellation(packet);
        cancel.check()?;

        let session = self.services.session()?;

        // -- The world this child was told to work from ------------------
        //
        // Before anything else, and before the card is reserved: a worker
        // waiting on a sibling that never finished should not be holding a
        // model while it waits.
        let memory = self.memory(packet, policy);
        let started_at_revision = match &memory {
            Some(memory) => {
                match memory
                    .await_requirement(packet.requirement, sibling_wait(packet))
                    .await
                {
                    Ok(reached) => Some(reached),
                    Err(unavailable) => {
                        // Not a failure of this worker: the work it depends on
                        // has not landed. Its own ending, so a parent can tell
                        // "the retrieval was wrong" from "the retrieval never
                        // ran".
                        return Ok(ChildResult::ended(
                            &packet.child_id,
                            &packet.profile,
                            ChildStatus::Failed,
                            packet.required_schema,
                            Vec::new(),
                            unavailable.explain(),
                            0,
                        ));
                    }
                }
            }
            None => None,
        };

        cancel.check()?;

        // -- The card ----------------------------------------------------
        //
        // Held for the length of the work and released when this returns. The
        // card is one heavy call for the whole machine (see
        // `subagents::scheduling`), so four logically parallel children take
        // turns — and the parent that dispatched this one gives way first.
        //
        // The parent's suspension is written down *before* this child asks,
        // because that is the moment the parent starts awaiting work that needs
        // the same card: a crash while the child runs must still show why the
        // parent was not holding its model. Only for a child that needs a model
        // — a mechanical routine needs no card and suspends nobody.
        let lease = match &packet.model_id {
            Some(model_id) => {
                let scheduler = &self.services.scheduler;
                let parent = packet.parent_run_id.as_str();
                if !parent.is_empty()
                    && (scheduler.binding(parent).is_some() || scheduler.holds(parent))
                {
                    scheduler.suspend(
                        parent,
                        Some(packet.child_id.as_str()),
                        &format!(
                            "{parent} delegated to {} ({}), which needs the GPU; the parent \
                             releases it until the child returns",
                            packet.child_id, packet.profile
                        ),
                    );
                }
                let request = crate::subagents::scheduling::LeaseRequest::new(
                    packet.child_id.as_str(),
                    crate::subagents::scheduling::LeaseClass::Child,
                    model_id.as_str(),
                )
                .child_of(parent)
                // The definition's allowed models, enforced here: the packet
                // only records whether routing stayed inside them.
                .eligible(
                    packet
                        .model_policy
                        .as_ref()
                        .map(|policy| policy.eligible_model_ids.clone())
                        .unwrap_or_default(),
                )
                .waiting(remaining(packet))
                .cancelled_by(cancel.own.clone());
                match scheduler.acquire(request).await {
                    Ok(lease) => Some(lease),
                    Err(refusal) => {
                        return Ok(ChildResult::ended(
                            &packet.child_id,
                            &packet.profile,
                            ChildStatus::Failed,
                            packet.required_schema,
                            Vec::new(),
                            refusal.explain(),
                            0,
                        ))
                    }
                }
            }
            None => None,
        };

        cancel.check()?;

        // -- The work ----------------------------------------------------
        //
        // The model loop first, whenever there is one. A child that drives a
        // model decides *which* of its tools to call and in what order, which
        // is the difference between a worker and a subroutine — and it does so
        // inside a plan narrowed to exactly its granted set, so the gateway
        // refuses anything wider. See `subagents::child_loop`.
        //
        // The mechanical routines below remain the fallback for the four roles
        // that are mechanical anyway. They are not a lesser version of the same
        // thing: they call the same tools and build findings from the same
        // receipts, with the ordering fixed rather than chosen.
        let runnable_loop = self
            .services
            .child_loop
            .as_ref()
            .filter(|child_loop| child_loop.available() && packet.model_id.is_some());

        let outcome = match runnable_loop {
            Some(child_loop) => self.via_model(child_loop, packet, policy, &cancel).await,
            // No runtime, or no model. A role that cannot be done without one
            // says so rather than doing a fraction of it.
            None if Self::NEEDS_A_MODEL.contains(&self.profile.as_str()) => Err(format!(
                "the {} role has to write and run a program, which needs a model, and no agent \
                 runtime is available on this machine right now. Nothing was done.",
                self.profile
            )),
            None => match self.profile.as_str() {
                "knowledge-retriever" => self.retrieve(packet, policy, &session, &cancel),
                "document-extractor" => self.extract(packet, policy, &cancel),
                "calculation-checker" => {
                    self.check_calculations(packet, policy, &session, memory.as_ref(), &cancel)
                }
                "artifact-reviewer" => self.review(packet, policy, &cancel),
                other => Err(format!(
                    "this build has no worker for the {other} role, so nothing was done"
                )),
            },
        };
        let mut work = match outcome {
            Ok(work) => work,
            Err(detail) => {
                return Ok(ChildResult::ended(
                    &packet.child_id,
                    &packet.profile,
                    ChildStatus::Failed,
                    packet.required_schema,
                    Vec::new(),
                    detail,
                    1,
                ))
            }
        };

        cancel.check()?;

        // -- Publishing --------------------------------------------------
        //
        // The findings go into the task's shared memory, and what comes back to
        // the parent is the id and the revision. A sibling reads them there;
        // nothing is passed between workers directly.
        if let Some(memory) = memory.as_ref() {
            for claim in std::mem::take(&mut work.claims) {
                match memory.publish(
                    claim,
                    // A receipt, because every claim these workers make is a
                    // tool's own output rather than a model's account of it.
                    // See the module header: that is the whole reason these
                    // roles are performed this way.
                    Provenance::ToolReceipt {
                        run_id: packet.parent_run_id.clone(),
                        tool: work.tool.as_str().to_string(),
                        event_seq: 0,
                    },
                    lease.as_ref().map(|lease| lease.model_id.clone()),
                    &session,
                ) {
                    Ok(published) => work.published.push(published),
                    Err(unavailable) => work.uncertainty.push(format!(
                        "a finding was not published: {}",
                        unavailable.explain()
                    )),
                }
            }
        } else {
            work.uncertainty.push(
                "this deployment has no runtime memory graph, so nothing this worker found was \
                 published to the task's shared memory. The findings below are in this result \
                 only."
                    .to_string(),
            );
        }

        // What the parent is handed: the findings, and — the part a sibling
        // needs — where they landed.
        let mut detail = String::new();
        if let Some(reached) = started_at_revision {
            detail.push_str(&format!(
                "Read the task's shared memory at revision {reached}.\n"
            ));
        }
        if let Some(lease) = &lease {
            detail.push_str(&format!("Model: {}.\n", lease.describe()));
        }
        for published in &work.published {
            detail.push_str(&format!("Published {}.\n", published.describe()));
        }

        let mut result = ChildResult::completed(
            &packet.child_id,
            &packet.profile,
            packet.required_schema,
            work.findings,
            work.confidence,
            work.uncertainty,
            work.turns,
        );
        if !detail.is_empty() {
            result.detail = Some(detail);
        }
        // Where the findings landed, as ids a sibling can read for itself. The
        // detail string above says the same thing in prose for a person; this
        // is the part the parent's completion check and the next worker use.
        result.published = work
            .published
            .iter()
            .map(|published| published.item_id.clone())
            .collect();
        Ok(result)
    }
}

/// What one routine produced, before it is published.
struct Work {
    findings: Vec<Finding>,
    claims: Vec<Claim>,
    published: Vec<Published>,
    uncertainty: Vec<String>,
    confidence: f32,
    turns: u32,
    tool: ToolName,
}

impl Work {
    fn of(tool: ToolName) -> Self {
        Self {
            findings: Vec::new(),
            claims: Vec::new(),
            published: Vec::new(),
            uncertainty: Vec::new(),
            confidence: 0.0,
            turns: 0,
            tool,
        }
    }
}

impl SpecialistWorker {
    /// Runs this child as a model loop, and builds its findings from receipts.
    ///
    /// ## Why the findings are not the model's answer
    ///
    /// The loop returns prose. Prose is what a model produces whether or not it
    /// did the work — the failure `certification`'s header describes, where a
    /// retrieval that returned three passages instead of five reports three
    /// passages and reads exactly like success.
    ///
    /// So the model's text is *not* folded in. What is kept is what the gateway
    /// settled: the tool calls this child actually made, and the passages the
    /// index actually returned to it. A worker that called nothing produces no
    /// findings, however confidently it wrote.
    async fn via_model(
        &self,
        child_loop: &super::child_loop::ChildLoop,
        packet: &ChildTaskPacket,
        policy: &EffectivePolicy,
        cancel: &Stopping,
    ) -> Result<Work, String> {
        cancel.check()?;
        // The definition this child was *sent under*, not the one this worker
        // was built with. The packet carries the instructions pinned at dispatch
        // — see `subagents::definitions` — so an administrator's edit reaches
        // the next child that runs, and a child already running keeps its own.
        // Empty only for a packet from before definitions were pinned, and then
        // the construction-time body is the only text there is.
        let instructions = if packet.instructions.trim().is_empty() {
            self.instructions.as_str()
        } else {
            packet.instructions.as_str()
        };
        let report = child_loop
            .run(packet, policy, instructions)
            .await
            .map_err(|detail| format!("this worker's model loop did not run: {detail}"))?;
        cancel.check()?;

        let mut work = Work::of(
            // The tool a claim is attributed to. The child chose which to call,
            // so the honest label is the one it used most — and where it called
            // several, the first, because that is the one the rest rest on.
            report
                .receipts()
                .next()
                .and_then(|call| ToolName::from_str(&call.tool))
                .unwrap_or(ToolName::SearchDocuments),
        );
        work.turns = report.tool_calls.len() as u32;

        // Evidence first, because a finding that cites one is worth more than a
        // finding that does not, and these are the passages the index returned
        // rather than anything the model said about them.
        for (position, hit) in report.passages.iter().enumerate() {
            work.findings.push(Finding {
                statement: format!("{} bears on this task.", hit.citation()),
                evidence: vec![EvidenceRef {
                    marker: Some(position + 1),
                    document_sha256: hit.document_sha256.clone(),
                    page: Some(hit.page),
                    citation: hit.citation(),
                }],
            });
            work.claims.push(Claim {
                kind: MemoryKind::Fact,
                content: format!("{}: {}", subject_of(&packet.objective), hit.citation()),
                sources: vec![SourceRef {
                    sha256: hit.document_sha256.clone(),
                    locator: format!("page {}", hit.page),
                    extraction_revision: None,
                }],
                artifacts: Vec::new(),
                confidence: Some(0.9),
                causal_parents: Vec::new(),
                idempotency_key: format!("{}:{}", packet.idempotency_key, hit.chunk_id),
            });
        }

        // Then everything else it did. A calculation, a file read, a produced
        // artifact — each is a receipt the gateway wrote, and each is reported
        // as what the tool returned rather than as what the model said about it.
        for call in report.receipts() {
            if call.tool == ToolName::SearchDocuments.as_str() {
                // Already accounted for, in the passages above, and with
                // evidence attached rather than as a line of text.
                continue;
            }
            work.findings.push(Finding {
                statement: format!("{}: {}", call.tool, call.detail),
                evidence: Vec::new(),
            });
            work.claims.push(Claim {
                kind: MemoryKind::ToolObservation,
                content: format!("{}: {}", call.tool, call.detail),
                sources: Vec::new(),
                artifacts: Vec::new(),
                confidence: Some(1.0),
                causal_parents: Vec::new(),
                idempotency_key: format!("{}:{}:{}", packet.idempotency_key, call.tool, call.at),
            });
        }

        let refused = report
            .tool_calls
            .iter()
            .filter(|call| {
                !matches!(
                    call.outcome,
                    crate::agent_runtime::tasks::CallOutcome::Succeeded
                )
            })
            .count();
        if refused > 0 {
            work.uncertainty.push(format!(
                "{refused} of this worker's {} tool call(s) did not succeed, so anything they \
                 would have established is missing.",
                report.tool_calls.len()
            ));
        }
        if !report.completed {
            // The loop's own ending, carried rather than smoothed over. The
            // manager decides the status; this is why.
            work.uncertainty.push(format!(
                "the worker's own loop did not finish cleanly: {}. What is above is what it had \
                 done by then.",
                report.outcome
            ));
        }
        work.uncertainty.push(format!(
            "ran as a model loop on {} at {} ({} token window)",
            report.model_id, report.base_url, report.served_window
        ));

        // A model that called nothing established nothing, whatever it wrote.
        work.confidence = if work.findings.is_empty() { 0.0 } else { 1.0 };
        Ok(work)
    }

    /// Finds the passages that bear on one question, and cites them.
    fn retrieve(
        &self,
        packet: &ChildTaskPacket,
        policy: &EffectivePolicy,
        session: &Session,
        cancel: &Stopping,
    ) -> Result<Work, String> {
        require(policy, ToolName::SearchDocuments)?;
        cancel.check()?;

        // The question is the objective, widened by any expression the parent
        // pointed at. Never the parent's transcript: a packet carries no
        // contents, and there is nothing here that could read one.
        let mut query = packet.objective.clone();
        for input in &packet.inputs {
            if let InputRef::Expression { expression } = input {
                query.push(' ');
                query.push_str(expression);
            }
        }

        let hits = self
            .services
            .index
            .search(session, &query, RETRIEVAL_LIMIT)
            .map_err(|error| format!("the document index could not be searched: {error}"))?;

        let mut work = Work::of(ToolName::SearchDocuments);
        work.turns = 1;
        if hits.is_empty() {
            work.uncertainty.push(format!(
                "nothing in the connected collections matched {query:?}. This is a finding: do \
                 not assert what no source says."
            ));
            work.confidence = 1.0;
            return Ok(work);
        }

        for (position, hit) in hits.iter().enumerate() {
            work.findings.push(Finding {
                // The citation, not the passage. A finding carrying the text
                // would be a second copy of it under a second set of clearance
                // assumptions — the rule `result.rs` documents.
                statement: format!("{} bears on this question.", hit.citation()),
                evidence: vec![EvidenceRef {
                    marker: Some(position + 1),
                    document_sha256: hit.document_sha256.clone(),
                    page: Some(hit.page),
                    citation: hit.citation(),
                }],
            });
            work.claims.push(Claim {
                kind: MemoryKind::Fact,
                // The `subject: statement` shape the conflict model matches on.
                content: format!("{}: {}", subject_of(&packet.objective), hit.citation()),
                sources: vec![SourceRef {
                    sha256: hit.document_sha256.clone(),
                    locator: format!("page {}", hit.page),
                    extraction_revision: None,
                }],
                artifacts: Vec::new(),
                confidence: Some(0.9),
                causal_parents: Vec::new(),
                idempotency_key: format!("{}:{}", packet.idempotency_key, hit.chunk_id),
            });
        }
        // Retrieval either found passages or it did not; nothing here was
        // inferred.
        work.confidence = 1.0;
        Ok(work)
    }

    /// Reads the files a task points at, and reports what was not read.
    fn extract(
        &self,
        packet: &ChildTaskPacket,
        policy: &EffectivePolicy,
        cancel: &Stopping,
    ) -> Result<Work, String> {
        require(policy, ToolName::ReadScopedFile)?;

        let root = policy.inherited.workspace_root.clone();
        let mut work = Work::of(ToolName::ReadScopedFile);
        let wanted: Vec<&String> = packet
            .inputs
            .iter()
            .filter_map(|input| match input {
                InputRef::WorkspaceFile { path } => Some(path),
                _ => None,
            })
            .collect();
        if wanted.is_empty() {
            return Err(
                "this worker reads files a task points at, and the task pointed at none. Name \
                 the workspace files to read."
                    .to_string(),
            );
        }

        for path in wanted {
            cancel.check()?;
            work.turns += 1;
            // Confined to the run's own directory. Checked rather than trusted:
            // a path that escapes is the one thing a read-only worker must not
            // be able to do.
            let resolved = match confine(&root, path) {
                Ok(resolved) => resolved,
                Err(detail) => {
                    work.uncertainty
                        .push(format!("{path} was not read: {detail}"));
                    continue;
                }
            };
            match std::fs::read_to_string(&resolved) {
                Ok(text) => {
                    let lines = text.lines().count();
                    let bytes = text.len();
                    work.findings.push(Finding {
                        statement: format!("{path} holds {lines} line(s), {bytes} byte(s)."),
                        evidence: Vec::new(),
                    });
                    work.claims.push(Claim {
                        kind: MemoryKind::ToolObservation,
                        content: format!("{path}: read, {lines} line(s), {bytes} byte(s)"),
                        sources: Vec::new(),
                        artifacts: Vec::new(),
                        confidence: Some(1.0),
                        causal_parents: Vec::new(),
                        idempotency_key: format!("{}:{path}", packet.idempotency_key),
                    });
                }
                Err(error) => work
                    .uncertainty
                    .push(format!("{path} could not be read: {error}")),
            }
        }
        work.confidence = if work.uncertainty.is_empty() { 1.0 } else { 0.6 };
        Ok(work)
    }

    /// Re-derives figures through the deterministic engine.
    ///
    /// This is the worker that reads what another one published: expressions
    /// come from the packet *and* from the task's shared memory, so a retriever
    /// that found a figure produces something this checks without either of
    /// them exchanging a transcript.
    fn check_calculations(
        &self,
        packet: &ChildTaskPacket,
        policy: &EffectivePolicy,
        session: &Session,
        memory: Option<&TaskMemory>,
        cancel: &Stopping,
    ) -> Result<Work, String> {
        require(policy, ToolName::RunCalculation)?;

        let mut expressions: Vec<String> = packet
            .inputs
            .iter()
            .filter_map(|input| match input {
                InputRef::Expression { expression } => Some(expression.clone()),
                _ => None,
            })
            .collect();

        // The graph read. Everything a sibling published to this task that
        // looks like something to check.
        let mut from_memory = 0usize;
        if let Some(memory) = memory {
            for kind in [MemoryKind::Fact, MemoryKind::ToolObservation] {
                let published = memory
                    .read_kind(session, kind)
                    .map_err(|unavailable| unavailable.explain())?;
                for item in published {
                    if let Some(expression) = expression_in(&item.content) {
                        if !expressions.iter().any(|held| held == &expression) {
                            expressions.push(expression);
                            from_memory += 1;
                        }
                    }
                }
            }
        }

        if expressions.is_empty() {
            return Err(
                "this worker re-derives figures, and neither the task nor the shared memory \
                 offered one to check. Nothing was calculated."
                    .to_string(),
            );
        }

        let mut work = Work::of(ToolName::RunCalculation);
        if from_memory > 0 {
            work.uncertainty.push(format!(
                "{from_memory} of these came from what another worker published to this task \
                 rather than from the request."
            ));
        }

        for expression in expressions {
            cancel.check()?;
            work.turns += 1;
            match crate::orchestrator::calculation::evaluate(&expression) {
                Ok(record) => {
                    work.findings.push(Finding {
                        statement: format!("{expression} = {}", record.formatted),
                        evidence: Vec::new(),
                    });
                    work.claims.push(Claim {
                        kind: MemoryKind::ToolObservation,
                        content: format!("{expression}: {}", record.formatted),
                        sources: Vec::new(),
                        artifacts: Vec::new(),
                        confidence: Some(1.0),
                        causal_parents: Vec::new(),
                        idempotency_key: format!("{}:{expression}", packet.idempotency_key),
                    });
                }
                Err(error) => work
                    .uncertainty
                    .push(format!("{expression} could not be evaluated: {}", error.message)),
            }
        }
        // The engine computed these; nothing was inferred.
        work.confidence = 1.0;
        Ok(work)
    }

    /// Re-opens what a task produced and reports what is actually in it.
    fn review(
        &self,
        packet: &ChildTaskPacket,
        policy: &EffectivePolicy,
        cancel: &Stopping,
    ) -> Result<Work, String> {
        require(policy, ToolName::ValidateArtifact)?;

        let root = policy.inherited.workspace_root.clone();
        let mut work = Work::of(ToolName::ValidateArtifact);
        let mut reviewed = 0usize;

        for input in &packet.inputs {
            let (artifact_id, revision, sha256) = match input {
                InputRef::Artifact {
                    artifact_id,
                    revision,
                    sha256,
                } => (artifact_id.clone(), *revision, sha256.clone()),
                _ => continue,
            };
            cancel.check()?;
            work.turns += 1;
            reviewed += 1;

            // The file, re-opened. "Ready" has to be a statement about the
            // bytes rather than about the call that wrote them, which is the
            // whole of this profile's description.
            let resolved = match confine(&root, &artifact_id) {
                Ok(resolved) => resolved,
                Err(detail) => {
                    work.uncertainty
                        .push(format!("{artifact_id} was not opened: {detail}"));
                    continue;
                }
            };
            match std::fs::metadata(&resolved) {
                Ok(metadata) if metadata.len() == 0 => work.uncertainty.push(format!(
                    "{artifact_id} exists and is empty, so it is not a usable file."
                )),
                Ok(metadata) => {
                    work.findings.push(Finding {
                        statement: format!(
                            "{artifact_id} revision {revision} exists and holds {} byte(s).",
                            metadata.len()
                        ),
                        evidence: Vec::new(),
                    });
                    work.claims.push(Claim {
                        kind: MemoryKind::ArtifactRef,
                        content: format!(
                            "{artifact_id}: revision {revision} re-opened, {} byte(s)",
                            metadata.len()
                        ),
                        sources: Vec::new(),
                        artifacts: vec![ArtifactRef {
                            artifact_id: artifact_id.clone(),
                            revision,
                            sha256: sha256.clone(),
                        }],
                        confidence: Some(1.0),
                        causal_parents: Vec::new(),
                        idempotency_key: format!("{}:{artifact_id}", packet.idempotency_key),
                    });
                }
                Err(error) => work
                    .uncertainty
                    .push(format!("{artifact_id} could not be opened: {error}")),
            }
        }

        if reviewed == 0 {
            return Err(
                "this worker re-opens files a task produced, and the task pointed at none. Name \
                 the artifacts to review."
                    .to_string(),
            );
        }
        work.confidence = 1.0;
        Ok(work)
    }
}

/// Everything that can stop this child, asked together.
///
/// Its own token, and the run it belongs to. A child that only watched its own
/// would carry on after the person stopped the task, and one that only watched
/// the run could not be stopped individually.
struct Stopping {
    own: CancelToken,
    cancellations: Arc<crate::agent_runtime::cancellation::RunCancellations>,
    run_id: String,
}

impl Stopping {
    /// Refuses as soon as either has been stopped.
    fn check(&self) -> Result<(), String> {
        self.own.check()?;
        if self.cancellations.is_cancelled(&self.run_id) {
            return Err(
                "the task this worker belongs to was stopped, so it did not run. Nothing was                  done."
                    .to_string(),
            );
        }
        Ok(())
    }
}

/// The longest a worker waits for a sibling's result before saying it has not
/// arrived.
///
/// ## Why this is not the child's whole deadline
///
/// Because a worker that spent its entire budget waiting would have none left
/// to do the work even if the result landed on the last second — and the
/// manager's own timeout would fire at the same moment, so the run would be told
/// "it ran out of time" rather than "the thing it was waiting for never
/// finished". Those send an operator to different places.
///
/// Fifteen seconds, and bounded again by a quarter of whatever the child has
/// left, so a short-deadline worker does not wait past its own ending. A
/// sibling that has not landed by then is not about to, and re-dispatching is
/// free: the idempotency ledger recognises the retry.
const MAX_SIBLING_WAIT: std::time::Duration = std::time::Duration::from_secs(15);

/// How long this child may wait for a sibling.
fn sibling_wait(packet: &ChildTaskPacket) -> std::time::Duration {
    remaining(packet)
        .checked_div(4)
        .unwrap_or(MAX_SIBLING_WAIT)
        .min(MAX_SIBLING_WAIT)
}

/// Refuses a tool this child was not granted.
///
/// By name, and before anything happens. A worker that quietly did less because
/// a tool was missing would look exactly like a worker that found less.
fn require(policy: &EffectivePolicy, tool: ToolName) -> Result<(), String> {
    if policy.tools.contains(&tool) {
        return Ok(());
    }
    Err(format!(
        "this worker needs {}, which it was not granted — either its profile does not request \
         it, or the run it belongs to does not hold it and a child is never given more than its \
         parent. Nothing was done.",
        tool.as_str()
    ))
}

/// Resolves a path inside the run's own directory, or refuses.
///
/// The confinement a read-only child must not be able to leave. Both sides are
/// canonicalised before they are compared, so `..` and a symlink are the same
/// question rather than two — a check on the textual path would pass a link that
/// points outside.
fn confine(root: &std::path::Path, relative: &str) -> Result<std::path::PathBuf, String> {
    let candidate = root.join(relative);
    let resolved = candidate
        .canonicalize()
        .map_err(|error| format!("it is not there ({error})"))?;
    let base = root
        .canonicalize()
        .map_err(|error| format!("the run's own directory could not be resolved ({error})"))?;
    if resolved.starts_with(&base) {
        Ok(resolved)
    } else {
        Err(format!(
            "it resolves to {}, which is outside the directory this run owns",
            resolved.display()
        ))
    }
}

/// How long this child has left.
fn remaining(packet: &ChildTaskPacket) -> std::time::Duration {
    (packet.deadline - chrono::Utc::now())
        .to_std()
        .unwrap_or(std::time::Duration::from_millis(0))
}

/// The leading words of an objective, as the subject a claim is filed under.
///
/// Bounded to a few words: the subject is a matching key for the conflict model,
/// and a key that is a whole sentence matches only itself.
fn subject_of(objective: &str) -> String {
    let words: Vec<&str> = objective.split_whitespace().take(6).collect();
    if words.is_empty() {
        "finding".to_string()
    } else {
        words.join(" ").to_ascii_lowercase()
    }
}

/// An arithmetic expression inside a published claim, where there is one.
///
/// Looks for the `subject: statement` shape these workers write, and takes the
/// statement when it parses as arithmetic. Deliberately conservative: a claim
/// that is prose comes back `None` rather than being fed to the engine as a
/// malformed expression, because a worker whose uncertainty list is full of
/// parse errors is a worker nobody reads.
fn expression_in(content: &str) -> Option<String> {
    let statement = content.split_once(':').map(|(_, rest)| rest.trim())?;
    if statement.is_empty() {
        return None;
    }
    // Arithmetic has an operator in it.
    if !statement.contains(['+', '-', '*', '/', '^']) {
        return None;
    }
    crate::orchestrator::calculation::evaluate(statement)
        .ok()
        .map(|_| statement.to_string())
}
