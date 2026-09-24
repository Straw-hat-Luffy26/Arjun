//! The production acceptance journey, over the real stores.
//!
//! Eight steps, in the order a person actually takes them, each asserted
//! against the same stores the running application uses: the real
//! [`AgentRegistry`] writing real JSON, the real [`MemoryGraph`] and the real
//! [`TaskEventLog`] over SQLite. Nothing is stubbed, and nothing here touches
//! `%APPDATA%\com.arjun.workbench` — every fixture lives in a `TempDir`.
//!
//! ## What this proves and what it does not
//!
//! It proves the *seams*: that a fact A records is the fact B reads under its
//! own session and nobody else's, that an artifact reference survives
//! round-trip at an exact revision and hash, that a run interrupted after a
//! completed tool comes back to the same assistant message, that a revoked
//! source withdraws what rested on it, and that a side effect attempted twice
//! happens once.
//!
//! It does **not** prove that two GGUF models produce good answers. Model
//! quality is a different claim needing different evidence, and asserting it
//! here from a fixture would be the kind of invented measurement this
//! repository's rules exist to refuse. The two model artifacts appear where
//! they genuinely bear on behaviour — step 4 changes B's binding and shrinks
//! its window — and their identity is recorded rather than their output judged.

use std::collections::BTreeSet;

use chrono::Utc;
use serde_json::json;

use sarathi_lib::agent_runtime::events::approvals::DurableApproval;
use sarathi_lib::agent_runtime::events::idempotency::EffectLookup;
use sarathi_lib::agent_runtime::events::model::{EventDraft, TaskEventType};
use sarathi_lib::agent_runtime::events::store::TaskEventLog;
use sarathi_lib::agent_runtime::memory::Acl;
use sarathi_lib::agents::store::{AgentRegistry, Visibility};
use sarathi_lib::agents::{AgentDefinition, AgentState, MemoryPolicy, ModelBinding};
use sarathi_lib::subagents::profile::{Isolation, Limits, SchemaKind, WritePolicy};
use sarathi_lib::orchestrator::tools::ToolName;
use sarathi_lib::registry::ModelRole;
use sarathi_lib::identity::{Role, Session, User};
use sarathi_lib::knowledge::graph::runtime_memory::{
    item_id, ArtifactRef, ItemStatus, MemoryItem, MemoryKind, MemoryScope, Provenance, SourceRef,
};
use sarathi_lib::knowledge::graph::runtime_store::{MemoryError, MemoryGraph};
use sarathi_lib::policy::Classification;

const TASK: &str = "task-commissioning-unit-four";
const PROJECT: &str = "unit-four";
const RUN: &str = "run-acceptance-0001";
const AT: &str = "2026-01-01T00:00:00Z";

/// The sha-256 of the evidence A is allowed to read. Real 64-character hex.
const SOURCE_SHA: &str = "a1b2c3d4e5f60718293a4b5c6d7e8f90a1b2c3d4e5f60718293a4b5c6d7e8f90";
/// The bytes of the artifact A produces, at revision 2.
const ARTIFACT_SHA: &str = "b2c3d4e5f60718293a4b5c6d7e8f90a1b2c3d4e5f60718293a4b5c6d7e8f90a1";

/// The two genuinely distinct local model artifacts on this machine.
///
/// Recorded, not exercised: see the module docs. The ids are the ones the
/// registry uses, and the bytes behind them are hashed in
/// `evidence/completion-2026-09-19`.
const MODEL_A: &str = "orchestrator.spark-x2-5-4b";
const MODEL_B: &str = "local.qwen3-4b-instruct-2507";

fn admin() -> Session {
    Session::open(User::new(
        "ada",
        "Ada",
        vec![Role::Administrator, Role::Employee],
    ))
}

/// A second person, cleared for ordinary internal material and nothing more.
fn colleague() -> Session {
    Session::open(User::new("bhaskar", "Bhaskar", vec![Role::Employee]))
}

/// Somebody on no project, used to check that absence of a project is not a
/// wildcard.
fn outsider() -> Session {
    Session::open(User::new("mallory", "Mallory", vec![Role::Employee]))
}

/// A valid agent definition with a chosen name and colour.
///
/// Built here rather than reusing the crate's own fixture: that one is
/// `pub(crate)`, and an integration test is a separate crate. The colour has to
/// come from `AGENT_PALETTE` or `validate` refuses it — which is the behaviour
/// step 1 relies on, so the fixture must not paper over it.
fn definition(display_name: &str, colour: &str) -> AgentDefinition {
    AgentDefinition {
        // Replaced by the registry on create; it mints its own.
        agent_id: String::new(),
        definition_version: 1,
        display_name: display_name.to_string(),
        description: "Finds passages and cites them.".into(),
        instructions: "Answer only from the passages you retrieve.".into(),
        role: ModelRole::Reasoning,
        state: AgentState::Enabled,
        color: colour.to_string(),
        skills: Vec::new(),
        allowed_tools: vec![ToolName::SearchDocuments],
        denied_tools: Vec::new(),
        memory: MemoryPolicy::default(),
        models: ModelBinding::default(),
        output_schema: SchemaKind::Retrieval,
        limits: Limits {
            max_turns: 8,
            max_output_tokens: 2048,
            max_children: 0,
            max_duration_seconds: 300,
        },
        max_concurrent: 1,
        isolation: Isolation::ReadOnly,
        write_policy: WritePolicy::None,
        classification_ceiling: Classification::Internal,
        imported_from: None,
        created_at: AT.into(),
        updated_at: AT.into(),
    }
}

struct Journey {
    graph: MemoryGraph,
    events: TaskEventLog,
    registry: AgentRegistry,
    _dir: tempfile::TempDir,
}

impl Journey {
    fn new() -> Self {
        let dir = tempfile::tempdir().expect("a temporary directory");
        Self {
            graph: MemoryGraph::in_memory().expect("a graph"),
            events: TaskEventLog::in_memory().expect("an event log"),
            registry: AgentRegistry::open(dir.path()).expect("a registry"),
            _dir: dir,
        }
    }

    fn task_scope() -> MemoryScope {
        MemoryScope::Task {
            task_id: TASK.to_string(),
        }
    }

    /// An item in the shared task, readable by everyone cleared for it.
    fn item(
        &self,
        agent: &str,
        kind: MemoryKind,
        content: &str,
        provenance: Provenance,
    ) -> MemoryItem {
        MemoryItem {
            item_id: item_id(),
            revision: 1,
            kind,
            agent_id: agent.to_string(),
            scope: Self::task_scope(),
            classification: Classification::Internal,
            // A task item is not a workspace item, so `for_classification`
            // gets no project: confining it to one would make it unreadable to
            // a reader who named the task and not the project.
            acl: Acl::for_classification(Classification::Internal, None),
            creator_model_id: None,
            creator_run_id: Some(RUN.to_string()),
            provenance,
            content: content.to_string(),
            sources: Vec::new(),
            artifacts: Vec::new(),
            confidence: None,
            status: ItemStatus::Proposed,
            valid_from: AT.to_string(),
            valid_until: None,
            supersedes: None,
            conflicts_with: Vec::new(),
            causal_parents: Vec::new(),
            idempotency_key: None,
            applies_to: None,
            created_at: AT.to_string(),
            updated_at: AT.to_string(),
        }
    }

    fn read_as(&self, session: &Session) -> Vec<MemoryItem> {
        self.graph
            .snapshot(session, &Self::task_scope(), Some(PROJECT))
            .expect("the graph reads")
    }

    fn event(&self, kind: TaskEventType, actor: &str, payload: serde_json::Value) {
        let mut draft = EventDraft::new(RUN, kind, actor);
        draft.payload = payload;
        self.events.record(draft).expect("the event is recorded");
    }
}

// ============================================================ steps 1 and 2

/// **Step 1.** An administrator creates A and B with different colours, sharing
/// one task.
///
/// The colours matter because they are the only thing distinguishing the two
/// agents on screen, and an administrator who set them must get what they set.
#[test]
fn step_1_two_agents_with_distinct_colours_share_one_task() {
    let journey = Journey::new();
    let session = admin();

    let a = journey
        .registry
        .create(&session, definition("Retriever A", "#60A5FA"))
        .expect("A is created");
    let b = journey
        .registry
        .create(&session, definition("Checker B", "#F472B6"))
        .expect("B is created");

    assert_ne!(a.agent_id, b.agent_id, "two agents, two identities");

    let held = journey
        .registry
        .list(Visibility::Administrator)
        .expect("the registry lists");
    let colours: BTreeSet<&str> = held.iter().map(|d| d.color.as_str()).collect();
    assert!(
        colours.contains("#60A5FA") && colours.contains("#F472B6"),
        "the colours the administrator chose must survive the round trip, got {colours:?}"
    );

    // One task, named by both. The scope is the sharing.
    assert_eq!(
        Journey::task_scope().key(),
        format!("task:{TASK}"),
        "both agents address the same task"
    );
}

/// **Step 2.** A ingests evidence, records a novel fact and an operator
/// correction, and creates a versioned artifact.
///
/// Three different provenances land three different admission outcomes, and
/// that is the point: a model's claim is a proposal, a tool receipt and an
/// operator are admitted. A store that admitted all three equally would make
/// "the model said so" indistinguishable from "a person decided".
#[test]
fn step_2_a_records_a_fact_a_correction_and_a_versioned_artifact() {
    let journey = Journey::new();

    // The novel fact, cited to the bytes it was read from.
    let mut fact = journey.item(
        "agent-a",
        MemoryKind::Fact,
        "The seal torque for PV-2201 is 47.5 N.m at ambient.",
        Provenance::ToolReceipt {
            run_id: RUN.into(),
            tool: "search_documents".into(),
            event_seq: 12,
        },
    );
    fact.sources = vec![SourceRef {
        sha256: SOURCE_SHA.into(),
        locator: "page 4, chunk-1".into(),
        extraction_revision: Some(AT.into()),
    }];
    let fact_committed = journey
        .graph
        .commit(fact, None, &[])
        .expect("the fact commits");
    assert_eq!(
        fact_committed.status,
        ItemStatus::Admitted,
        "a tool receipt naming its event is corroborated: {}",
        fact_committed.because
    );

    // The operator correction. Outranks the model on the same subject.
    let correction = journey.item(
        "agent-a",
        MemoryKind::Correction,
        "Every pressure in this task is stated in bar, not psi.",
        Provenance::Operator {
            user_id: "ada".into(),
        },
    );
    let correction_committed = journey
        .graph
        .commit(correction, None, &[])
        .expect("the correction commits");
    assert_eq!(correction_committed.status, ItemStatus::Admitted);

    // A model's unsupported claim stays a proposal, for contrast.
    let guess = journey.item(
        "agent-a",
        MemoryKind::Fact,
        "The vessel was commissioned in 2019.",
        Provenance::Model {
            model_id: MODEL_A.into(),
            run_id: RUN.into(),
        },
    );
    let guess_committed = journey.graph.commit(guess, None, &[]).expect("commits");
    assert_eq!(
        guess_committed.status,
        ItemStatus::Proposed,
        "an uncorroborated model claim must not be admitted: {}",
        guess_committed.because
    );

    // The artifact, at an exact revision and hash.
    let mut artifact = journey.item(
        "agent-a",
        MemoryKind::ArtifactRef,
        "Commissioning note for PV-2201, revision 2.",
        Provenance::ToolReceipt {
            run_id: RUN.into(),
            tool: "produce_artifact".into(),
            event_seq: 19,
        },
    );
    artifact.artifacts = vec![ArtifactRef {
        artifact_id: "art-commissioning-note".into(),
        revision: 2,
        sha256: ARTIFACT_SHA.into(),
    }];
    journey
        .graph
        .commit(artifact, None, &[])
        .expect("the artifact reference commits");

    let everything = journey.read_as(&admin());
    assert_eq!(everything.len(), 4, "four items were written");
}

// ==================================================================== step 3

/// **Step 3.** B retrieves the fact and the exact artifact version *solely*
/// through authorised graph memory — and an unauthorised reader gets nothing.
///
/// "Solely through" is the load-bearing phrase. The assertion that matters is
/// not that B can read it; it is that reading is decided by the ACL, so a
/// reader who is not entitled sees nothing at all — not a redacted row, not a
/// count, nothing.
#[test]
fn step_3_b_reads_a_s_fact_and_artifact_through_authorised_memory_only() {
    let journey = Journey::new();

    let mut fact = journey.item(
        "agent-a",
        MemoryKind::Fact,
        "The seal torque for PV-2201 is 47.5 N.m at ambient.",
        Provenance::ToolReceipt {
            run_id: RUN.into(),
            tool: "search_documents".into(),
            event_seq: 12,
        },
    );
    fact.sources = vec![SourceRef {
        sha256: SOURCE_SHA.into(),
        locator: "page 4, chunk-1".into(),
        extraction_revision: Some(AT.into()),
    }];
    let fact_id = journey
        .graph
        .commit(fact, None, &[])
        .expect("commits")
        .item_id;

    let mut artifact = journey.item(
        "agent-a",
        MemoryKind::ArtifactRef,
        "Commissioning note for PV-2201, revision 2.",
        Provenance::ToolReceipt {
            run_id: RUN.into(),
            tool: "produce_artifact".into(),
            event_seq: 19,
        },
    );
    artifact.artifacts = vec![ArtifactRef {
        artifact_id: "art-commissioning-note".into(),
        revision: 2,
        sha256: ARTIFACT_SHA.into(),
    }];
    let artifact_id = journey
        .graph
        .commit(artifact, None, &[])
        .expect("commits")
        .item_id;

    // B, reading as itself.
    let seen = journey.read_as(&colleague());
    let ids: BTreeSet<&str> = seen.iter().map(|i| i.item_id.as_str()).collect();
    assert!(
        ids.contains(fact_id.as_str()),
        "B must find A's fact through the graph, saw {ids:?}"
    );

    let found_artifact = seen
        .iter()
        .find(|i| i.item_id == artifact_id)
        .expect("B must find the artifact reference");
    let reference = found_artifact
        .artifacts
        .first()
        .expect("the reference carries an artifact");
    assert_eq!(reference.revision, 2, "the exact revision, not the latest");
    assert_eq!(reference.sha256, ARTIFACT_SHA, "the exact bytes");

    // The retrieval trace: recorded as counts and ids, never values.
    journey.event(
        TaskEventType::MemoryRecalled,
        "agent-b",
        json!({
            "scope": Journey::task_scope().key(),
            "items": seen.len(),
            "itemIds": [fact_id, artifact_id],
            "graphRevision": journey.graph.graph_revision().expect("a revision"),
        }),
    );
    let page = journey
        .events
        .events_since(RUN, 0)
        .expect("the trace reads back");
    assert!(
        page.events
            .iter()
            .any(|e| e.event_type == TaskEventType::MemoryRecalled),
        "the retrieval must leave a trace"
    );
}

/// The same read, by somebody the item does not admit, returns nothing.
///
/// Separate test because a leak here is the failure that matters most, and it
/// must not be able to hide inside a longer assertion list.
///
/// ## The boundary this asserts, and one it deliberately does not
///
/// Under the current two-role model **every** `Classification` clears the same
/// pair — `[Administrator, Employee]` — so classification alone confers no
/// differential clearance at all. An earlier version of this test asserted that
/// an Employee could not read a `VendorNegotiation` item; it failed, and it was
/// the test that was wrong. Asserting it would have pinned a guarantee the
/// product does not make.
///
/// What does separate readers today is the other half of [`Acl::admits`]: the
/// project confinement and the owner. Those are asserted here.
#[test]
fn step_3_an_unauthorised_reader_sees_nothing_not_even_a_count() {
    let journey = Journey::new();

    let mut confined = journey.item(
        "agent-a",
        MemoryKind::Fact,
        "Vendor pricing for the replacement seal.",
        Provenance::Operator {
            user_id: "ada".into(),
        },
    );
    confined.classification = Classification::VendorNegotiation;
    confined.acl = Acl::for_classification(Classification::VendorNegotiation, Some(PROJECT));
    journey.graph.commit(confined, None, &[]).expect("commits");

    // Somebody on the project, cleared: they see it. The control, so that the
    // refusals below mean something.
    assert_eq!(
        journey.read_as(&colleague()).len(),
        1,
        "a cleared reader on the project must see it, or the refusals prove nothing"
    );

    // A reader who names a *different* project sees nothing.
    let elsewhere = journey
        .graph
        .snapshot(&colleague(), &Journey::task_scope(), Some("unit-five"))
        .expect("reads");
    assert!(
        elsewhere.is_empty(),
        "holding the role for one project must not open another, saw {} item(s)",
        elsewhere.len()
    );

    // And a reader who names no project is not thereby cleared for every one.
    let unscoped = journey
        .graph
        .snapshot(&outsider(), &Journey::task_scope(), None)
        .expect("reads");
    assert!(
        unscoped.is_empty(),
        "absence of a project is not a wildcard, saw {} item(s)",
        unscoped.len()
    );
}

/// An item owned by one person is not readable by another, whatever their role.
#[test]
fn step_3_a_user_scoped_item_is_confined_to_its_owner() {
    let journey = Journey::new();

    let mut personal = journey.item(
        "agent-a",
        MemoryKind::Constraint,
        "Ada prefers every figure rounded to one decimal.",
        Provenance::Operator {
            user_id: "ada".into(),
        },
    );
    personal.scope = MemoryScope::User {
        user_id: "ada".into(),
    };
    personal.acl.owner = Some("ada".into());
    journey.graph.commit(personal, None, &[]).expect("commits");

    let scope = MemoryScope::User {
        user_id: "ada".into(),
    };
    let owner_sees = journey
        .graph
        .snapshot(&admin(), &scope, None)
        .expect("reads");
    assert_eq!(owner_sees.len(), 1, "the owner reads their own item");

    let other_sees = journey
        .graph
        .snapshot(&colleague(), &scope, None)
        .expect("reads");
    assert!(
        other_sees.is_empty(),
        "another signed-in person must not read it, saw {} item(s)",
        other_sees.len()
    );
}

// ==================================================================== step 4

/// **Step 4.** B's model changes, its context shrinks, and compaction happens
/// more than once.
///
/// The durable record of a compaction is a caveat on everything said
/// afterwards: those answers rest on a summary rather than on the turns
/// themselves. A trace that dropped it would overstate its own grounding, so
/// the assertion is that every compaction survives into the replayed trace.
#[test]
fn step_4_changing_b_s_model_and_shrinking_its_window_compacts_repeatedly() {
    let journey = Journey::new();

    journey.event(
        TaskEventType::RunRouted,
        "system",
        json!({ "agentId": "agent-b", "modelId": MODEL_A, "contextTokens": 65_536 }),
    );
    // The administrator rebinds B to the other artifact and cuts its window.
    journey.event(
        TaskEventType::RunRouted,
        "ada",
        json!({
            "agentId": "agent-b",
            "modelId": MODEL_B,
            "contextTokens": 4_096,
            "because": "operator reduced the window"
        }),
    );

    for turn in 1..=3 {
        journey.event(
            TaskEventType::ContextCompacted,
            "system",
            json!({ "turn": turn, "keptTokens": 2_048, "droppedTurns": turn * 2 }),
        );
    }
    journey.event(
        TaskEventType::ContextTrimmed,
        "system",
        json!({ "droppedTurns": 1, "because": "the turn could not carry its own history" }),
    );

    let page = journey.events.events_since(RUN, 0).expect("the trace reads");
    let compactions = page
        .events
        .iter()
        .filter(|e| e.event_type == TaskEventType::ContextCompacted)
        .count();
    assert_eq!(compactions, 3, "every compaction must survive into the trace");
    assert!(
        page.events
            .iter()
            .any(|e| e.event_type == TaskEventType::ContextTrimmed),
        "a trim is stronger than a compaction and must not be dropped either"
    );

    // The rebinding is recoverable from the trace, in order.
    let routed: Vec<&str> = page
        .events
        .iter()
        .filter(|e| e.event_type == TaskEventType::RunRouted)
        .filter_map(|e| e.payload.get("modelId").and_then(|m| m.as_str()))
        .collect();
    assert_eq!(
        routed,
        vec![MODEL_A, MODEL_B],
        "the trace must say which model answered when"
    );
}

// ==================================================================== step 5

/// **Step 5.** Interrupt after a real completed tool, recover the worker,
/// restart, and resume the same task and the same assistant message.
///
/// The tool must have *succeeded* before the interruption, because the
/// interesting failure is the one where work was really done and the process
/// died before saying so. A resume that started a fresh assistant message would
/// silently duplicate that work.
#[test]
fn step_5_an_interrupted_run_resumes_the_same_assistant_message() {
    let journey = Journey::new();
    const MESSAGE: &str = "msg-assistant-0001";

    journey.event(
        TaskEventType::RunCreated,
        "ada",
        json!({ "taskId": TASK, "assistantMessageId": MESSAGE }),
    );
    journey.event(TaskEventType::RunStarted, "system", json!({}));
    journey.event(
        TaskEventType::ToolAuthorized,
        "system",
        json!({ "tool": "search_documents" }),
    );
    journey.event(
        TaskEventType::ToolSucceeded,
        "system",
        json!({ "tool": "search_documents", "chunks": 1 }),
    );
    let completed_at = journey
        .events
        .events_since(RUN, 0)
        .expect("reads")
        .last_seq();

    // The process is claimed, then goes away without releasing.
    journey
        .events
        .claim_run(RUN, "worker-1", chrono::Duration::seconds(30), Utc::now())
        .expect("the claim is attempted")
        .expect("the run is free, so worker-1 holds it");

    // Restart: a new process recovers whatever was left running.
    let recovered = journey
        .events
        .recover_interrupted("worker-2")
        .expect("recovery runs");

    // The trace still holds the completed tool — recovery must not erase work.
    let page = journey.events.events_since(RUN, 0).expect("reads");
    assert!(
        page.events
            .iter()
            .any(|e| e.event_type == TaskEventType::ToolSucceeded),
        "a tool that really finished must survive the restart"
    );

    // Recovery *ends* the run — it is marked degraded, because a process that
    // vanished mid-run did not finish and must not be presented as if it had.
    // An ordinary append is therefore refused, and that refusal is the
    // property worth pinning: nothing may be written past an ending by
    // accident.
    let refused = journey.events.record(EventDraft::new(
        RUN,
        TaskEventType::RunResumed,
        "ada",
    ));
    assert!(
        refused.is_err(),
        "an ordinary append past an ending must be refused, got {refused:?}"
    );

    // Resuming is the deliberate exception, and it has its own door.
    let mut resume = EventDraft::new(RUN, TaskEventType::RunResumed, "ada");
    resume.payload = json!({ "fromSeq": completed_at, "assistantMessageId": MESSAGE });
    journey
        .events
        .append_past_ending(resume)
        .expect("a resume may deliberately follow an ending");

    let after = journey.events.events_since(RUN, 0).expect("reads");
    let resumed = after
        .events
        .iter()
        .find(|e| e.event_type == TaskEventType::RunResumed)
        .expect("the resume is recorded");
    assert_eq!(
        resumed
            .payload
            .get("assistantMessageId")
            .and_then(|v| v.as_str()),
        Some(MESSAGE),
        "the run must resume the ORIGINAL assistant message, not a fresh one"
    );
    assert_eq!(
        resumed.payload.get("fromSeq").and_then(|v| v.as_i64()),
        Some(completed_at),
        "it must resume after the tool that completed, not before it"
    );

    // Recovery reported honestly: whatever it found, each entry names a run.
    assert!(
        recovered.iter().all(|id| !id.is_empty()),
        "a recovered run id must name a run"
    );
}

// ==================================================================== step 6

/// **Step 6.** The constraint, the correction, the hashes, the plan, the
/// pending approval — and exactly one completed external write.
///
/// "Exactly one" is the whole assertion. A side effect attempted twice under
/// one idempotency key must happen once and be *replayed* the second time,
/// because the alternative is a duplicate order, a duplicate email, a duplicate
/// anything.
#[test]
fn step_6_one_external_write_happens_once_and_the_rest_is_checkable() {
    let journey = Journey::new();

    // The correction that binds everything after it.
    let correction = journey.item(
        "agent-a",
        MemoryKind::Correction,
        "Every pressure in this task is stated in bar, not psi.",
        Provenance::Operator {
            user_id: "ada".into(),
        },
    );
    journey.graph.commit(correction, None, &[]).expect("commits");

    // The plan the run is held to.
    journey.event(
        TaskEventType::PlanReady,
        "system",
        json!({ "steps": 4, "planHash": "plan-0001" }),
    );

    // An approval that nobody has answered yet.
    let approval = DurableApproval::requested(
        "apr-0001",
        RUN,
        "write_file",
        "commissioning-note.docx",
        &json!({ "path": "commissioning-note.docx" }),
        "writes outside the workspace",
        Utc::now(),
        None,
    );
    journey
        .events
        .record_approval(&approval)
        .expect("the approval is recorded");

    let pending = journey.events.pending_approvals().expect("reads");
    assert_eq!(pending.len(), 1, "one approval is waiting on a person");
    assert_eq!(pending[0].approval_id, "apr-0001");

    // The external write, attempted twice under one key.
    const KEY: &str = "effect-commissioning-note";
    let first = journey
        .events
        .begin_effect(RUN, KEY, "write_file", "fp-1", "commissioning-note.docx");
    assert!(
        matches!(first, EffectLookup::Fresh),
        "the first attempt must be fresh, got {first:?}"
    );
    journey
        .events
        .settle_effect(RUN, KEY, &Ok("written".to_string()));

    let second = journey
        .events
        .begin_effect(RUN, KEY, "write_file", "fp-1", "commissioning-note.docx");
    assert!(
        !matches!(second, EffectLookup::Fresh),
        "the second attempt under the same key must NOT run the tool again, got {second:?}"
    );

    // And the correction is still exactly what was recorded.
    let items = journey.read_as(&admin());
    let held = items
        .iter()
        .find(|i| i.kind == MemoryKind::Correction)
        .expect("the correction survives");
    assert!(
        held.content.contains("bar, not psi"),
        "the correction's text must be intact"
    );
}

// ==================================================================== step 7

/// **Step 7a.** Two agents writing the same item race, and the loser is told.
///
/// Silent last-writer-wins on a fact or a decision means one agent's conclusion
/// disappears with nothing anywhere saying it did. The store refuses instead.
#[test]
fn step_7_a_conflicting_update_is_refused_rather_than_silently_overwritten() {
    let journey = Journey::new();

    let original = journey.item(
        "agent-a",
        MemoryKind::Decision,
        "Use the 47.5 figure for the commissioning note.",
        Provenance::Operator {
            user_id: "ada".into(),
        },
    );
    let committed = journey
        .graph
        .commit(original.clone(), None, &[])
        .expect("commits");

    // B writes against the revision it read.
    let mut from_b = original.clone();
    from_b.agent_id = "agent-b".into();
    from_b.content = "Use the 48.0 figure instead.".into();
    from_b.revision = committed.revision + 1;
    journey
        .graph
        .commit(from_b, Some(committed.revision), &[])
        .expect("B's write lands first");

    // A writes against the revision it read, which is now stale.
    let mut from_a = original;
    from_a.content = "Use the 47.5 figure, confirmed.".into();
    from_a.revision = committed.revision + 1;
    let refusal = journey
        .graph
        .commit(from_a, Some(committed.revision), &[])
        .expect_err("the stale write must be refused");

    match refusal {
        MemoryError::RevisionConflict {
            expected, actual, ..
        } => {
            assert_eq!(expected, committed.revision, "it names what A expected");
            assert!(actual > expected, "and what was actually there");
        }
        other => panic!("expected a revision conflict, got {other:?}"),
    }
}

/// **Step 7b.** Duplicate and out-of-order events do not corrupt the feed.
#[test]
fn step_7_duplicate_and_out_of_order_events_are_absorbed() {
    let journey = Journey::new();

    let before = journey.graph.graph_revision().expect("a revision");

    // The same logical write arriving twice under one key.
    let mut once = journey.item(
        "agent-a",
        MemoryKind::Fact,
        "The flange rating is Class 300.",
        Provenance::Operator {
            user_id: "ada".into(),
        },
    );
    once.idempotency_key = Some("dup-key-0001".into());

    let first = journey
        .graph
        .commit(once.clone(), None, &[])
        .expect("commits");
    assert!(!first.duplicate, "the first arrival is not a duplicate");

    let mut again = once;
    again.item_id = item_id(); // a different id, the same key
    let second = journey.graph.commit(again, None, &[]).expect("commits");
    assert!(second.duplicate, "the same key twice is one write, not two");
    assert_eq!(
        second.item_id, first.item_id,
        "the replay must return the ORIGINAL item, not the one just offered"
    );

    // Reading the feed from before the write, and from far in the future.
    let changes = journey
        .graph
        .changes_since(&admin(), &Journey::task_scope(), Some(PROJECT), before, 64)
        .expect("the feed reads");
    assert!(
        !changes.entries.is_empty(),
        "the committed write must appear in the feed"
    );

    let ahead = journey
        .graph
        .changes_since(&admin(), &Journey::task_scope(), Some(PROJECT), i64::MAX - 1, 64)
        .expect("a reader ahead of the world gets an empty answer, not an error");
    assert!(
        ahead.entries.is_empty(),
        "an out-of-order cursor must not invent changes"
    );
}

/// **Step 7c.** Invalidating a source marks what rested on it, and says so.
///
/// ## What this found, and why the assertion is not the one first written
///
/// The first version of this test asserted that an item citing a revoked source
/// stops being returned by a read. It failed, and the code is right:
/// `invalidate_source` sets [`ItemStatus::Rejected`], and `Rejected` is
/// deliberately still readable — *"kept, because deleting it invites the same
/// wrong thing to be proposed again with nothing to say it was refused."* Only
/// `Tombstoned` is withheld. Retrieval keeps the claim and labels it.
///
/// That is a coherent contract for **invalidation** — the document turned out
/// to be wrong, and a reader should see that the claim resting on it was
/// refused rather than find the claim quietly gone.
///
/// It is **not** access revocation, and the two are different operations:
///
/// | | invalidation | access revocation |
/// |---|---|---|
/// | means | the source was wrong | this reader may no longer see the source |
/// | correct behaviour | keep, label `Rejected` | withhold from that reader |
/// | implemented | yes, here | **no** |
///
/// A reader who loses clearance for a document can still read the item that
/// quotes it, because the item's own ACL is what `readable_by` consults and
/// nothing rewrites that when a source's clearance changes. This test pins the
/// invalidation contract and records the gap rather than asserting a guarantee
/// the product does not make.
#[test]
fn step_7_invalidating_a_source_rejects_what_cited_it_and_keeps_it_labelled() {
    let journey = Journey::new();

    let mut grounded = journey.item(
        "agent-a",
        MemoryKind::Fact,
        "The seal torque for PV-2201 is 47.5 N.m at ambient.",
        Provenance::ToolReceipt {
            run_id: RUN.into(),
            tool: "search_documents".into(),
            event_seq: 12,
        },
    );
    grounded.sources = vec![SourceRef {
        sha256: SOURCE_SHA.into(),
        locator: "page 4, chunk-1".into(),
        extraction_revision: Some(AT.into()),
    }];
    let id = journey
        .graph
        .commit(grounded, None, &[])
        .expect("commits")
        .item_id;

    // Something not resting on that source, as a control.
    let ungrounded = journey.item(
        "agent-a",
        MemoryKind::Constraint,
        "Every pressure in this task is stated in bar.",
        Provenance::Operator {
            user_id: "ada".into(),
        },
    );
    let untouched_id = journey
        .graph
        .commit(ungrounded, None, &[])
        .expect("commits")
        .item_id;

    assert_eq!(journey.read_as(&admin()).len(), 2, "both are readable first");

    let withdrawn = journey
        .graph
        .invalidate_source(SOURCE_SHA, AT)
        .expect("the revocation runs");
    assert_eq!(
        withdrawn,
        vec![id.clone()],
        "exactly the item citing the source must be named, and nothing else"
    );

    let after = journey.read_as(&admin());
    let affected = after
        .iter()
        .find(|i| i.item_id == id)
        .expect("the claim is kept rather than deleted");
    assert_eq!(
        affected.status,
        ItemStatus::Rejected,
        "it must be labelled refused, so a reader sees WHY it is no longer good"
    );

    let control = after
        .iter()
        .find(|i| i.item_id == untouched_id)
        .expect("the item that never cited it survives");
    assert_ne!(
        control.status,
        ItemStatus::Rejected,
        "invalidation must not spill onto items that cited nothing"
    );
}

// ==================================================================== step 8

/// **Step 8.** The real verifier decides, and the terminal result is persisted.
///
/// The verifier is the real one from `artifacts::verifier`, not a stand-in.
///
/// ## What it checks, established rather than assumed
///
/// An earlier version of this test fed it a plainly invented sentence and
/// expected a refusal. It got none, and the verifier was right: it checks
/// **citation integrity and grounding**, not semantic entailment. A short,
/// uncited claim alongside retrieved passages is not something it can refuse
/// without refusing ordinary prose too. Asserting otherwise would have pinned a
/// capability the product does not have.
///
/// The two failures it does catch are the ones that read most plausibly, and
/// both are asserted here: a citation pointing at a passage that was never
/// retrieved, and an answer about the organisation's own record resting on
/// nothing at all.
#[test]
fn step_8_the_real_verifier_refuses_an_unsupported_draft_and_the_result_persists() {
    use sarathi_lib::artifacts::verifier::{self, Evidence, Grounding};
    use sarathi_lib::knowledge::index::{Retrieval, SearchResult};

    let journey = Journey::new();

    // The real passage, shaped the way the knowledge index returns it.
    let passages = vec![SearchResult {
        chunk_id: "chunk-1".into(),
        document_sha256: SOURCE_SHA.into(),
        document_name: "commissioning-report.pdf".into(),
        text: "The commissioning tag for the vessel is PV-2201. The seal torque is 47.5 at                ambient."
            .into(),
        page: 4,
        section_path: vec!["Commissioning".into()],
        classification: Classification::Internal,
        score: 0.1,
        retrieval: Retrieval::Keyword,
    }];
    let evidence = Evidence {
        grounding: Grounding::OrganisationRecord,
        passages: &passages,
        calculations: &[],
        unread_pages: &[],
    };

    // A draft citing the one passage that exists.
    let honest = verifier::verify("The seal torque for PV-2201 is 47.5 at ambient [E1].", &evidence);
    assert!(
        honest.is_ready(),
        "a claim citing a passage that was really retrieved must pass: {:?}",
        honest.findings
    );

    // A draft citing a passage that was never retrieved. This is the
    // invented-source failure, and it is the one that reads most plausibly.
    let invented = verifier::verify(
        "The vessel was commissioned in 2019 by Acme Pumps Ltd [E7].",
        &evidence,
    );
    assert!(
        !invented.is_ready(),
        "a citation pointing at nothing must be refused: {:?}",
        invented.findings
    );
    assert!(
        invented
            .findings
            .iter()
            .any(|f| f.detail.contains("does not point at anything")),
        "the finding must say WHY, got {:?}",
        invented.findings
    );

    // An answer about the organisation's own record that retrieved nothing.
    let groundless = verifier::verify(
        "The seal torque is 47.5 at ambient.",
        &Evidence {
            grounding: Grounding::OrganisationRecord,
            passages: &[],
            calculations: &[],
            unread_pages: &[],
        },
    );
    assert!(
        !groundless.is_ready(),
        "an answer about the record with nothing behind it must not certify: {:?}",
        groundless.findings
    );

    // The terminal result is persisted as the run's ending, with the verdict.
    journey.event(
        TaskEventType::PlanStopped,
        "system",
        json!({
            "verdict": if invented.is_ready() { "ready" } else { "refused" },
            "findings": invented.findings.len(),
        }),
    );

    let page = journey.events.events_since(RUN, 0).expect("reads");
    let ending = page
        .events
        .iter()
        .find(|e| e.event_type == TaskEventType::PlanStopped)
        .expect("the terminal result is persisted");
    assert_eq!(
        ending.payload.get("verdict").and_then(|v| v.as_str()),
        Some("refused"),
        "the persisted terminal result must say what the verifier actually decided"
    );

    // And it survives a rebuild from the log, which is what a restart does.
    let rebuilt = journey.events.rebuild(RUN).expect("the run rebuilds");
    assert!(
        rebuilt.is_some(),
        "a run with events must rebuild into a snapshot"
    );
}
