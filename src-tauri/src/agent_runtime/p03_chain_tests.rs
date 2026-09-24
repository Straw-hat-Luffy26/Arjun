//! The P03 contract harness: Spark → OCR → Qwen → reviewer → Spark, then a
//! process restart, with a file already written and a correction arriving
//! mid-task.
//!
//! ## What this is, and what it is not
//!
//! **A deterministic contract test, not the native chain.** Everything Rust
//! owns is real: the lease service, the `context.refresh` / `context.count` /
//! `context.settle` handlers, the tool-boundary release, the memory graph and
//! the task event log — on disk, so the restart below reopens the same files.
//! The model servers are a fake backend, because the properties asserted here
//! are Rust's: who holds the GPU, what each model's context carries, and
//! whether a settled effect can happen twice. No model generates anything.
//!
//! The native chain — real Spark, real Unlimited-OCR, real Qwen, a real
//! reviewer role — needs the roles P05, P06 and P10 build, and runs on the
//! target machine. Its runnable gate is `scripts/p03-native-gate.mjs`, which
//! reports itself blocked, not passed, where its prerequisites are absent.
//!
//! ## The two things that must survive every hop
//!
//! - **A file written before the chain began.** Its receipt must reach every
//!   model's context exactly once, naming the exact artifact revision and hash,
//!   and the effect ledger must answer a repeat of the write with the recorded
//!   outcome rather than running it again — before and after the restart.
//! - **A correction made while the children run.** Every model that runs after
//!   it must be given it, as a mandatory constraint, exactly once.

use std::sync::Arc;

use serde_json::{json, Value};

use super::rounds::{refresh, settle, RoundBook};
use super::RuntimeDeps;
use crate::agent_runtime::events::{EffectLookup, TaskEventLog, TaskEventType};
use crate::agent_runtime::memory::Acl;
use crate::knowledge::graph::runtime_memory::{
    item_id, ArtifactRef, ItemStatus, MemoryItem, MemoryKind, MemoryScope, Provenance,
};
use crate::knowledge::graph::runtime_store::MemoryGraph;
use crate::policy::Classification;
use crate::subagents::scheduling::tests::{FakeBackend, OCR, QWEN, REVIEWER, SPARK};
use crate::subagents::scheduling::{
    LeaseClass, LeaseRequest, ModelScheduler, RunBinding, ServingBackend, ONE_HEAVY_CALL,
};

const PARENT: &str = "spark-run";
const NOTE_SHA: &str = "4be1c6a95e9d6c3b27f0e2a1d8c7b6a594837261504f3e2d1c0b9a8f7e6d5c4b";
const WRITE_KEY: &str = "spark-run:artifact.create_approval_note:approval-note.docx";

struct Process {
    deps: Arc<RuntimeDeps>,
    leases: Arc<ModelScheduler>,
    backend: Arc<FakeBackend>,
    graph: Arc<MemoryGraph>,
    _scratch: tempfile::TempDir,
}

/// One process's worth of services over the stores in `dir`.
fn start_process(dir: &std::path::Path) -> Process {
    let (deps, scratch) = crate::agent_runtime::tests::deps_with(Arc::new(std::sync::RwLock::new(
        Some(crate::identity::Session::open(crate::identity::User::new(
            "priya",
            "Priya Sharma",
            vec![crate::identity::Role::Employee],
        ))),
    )));
    // The scratch directory the helper made is for its own stores; the ones
    // that must survive the restart are opened on `dir`.

    let events = Arc::new(TaskEventLog::open(dir).expect("the event log opens on disk"));
    let graph = Arc::new(MemoryGraph::open(dir).expect("the graph opens on disk"));
    let backend = FakeBackend::new();
    let registry = crate::subagents::scheduling::tests::registry();
    let leases = Arc::new(ModelScheduler::with_backend(
        Arc::clone(&registry),
        Arc::clone(&backend) as Arc<dyn ServingBackend>,
        ONE_HEAVY_CALL,
    ));
    {
        let events = Arc::clone(&events);
        leases.observe(Arc::new(move |event: &crate::subagents::LeaseEvent| {
            crate::subagents::persist_lease_event(&events, event);
        }));
    }

    let mut owned = Arc::try_unwrap(deps).ok().expect("the only handle");
    owned.events = events;
    owned.memory_graph = Some(Arc::clone(&graph));
    owned.registry = Some(registry);
    owned.leases = Some(Arc::clone(&leases));
    owned.cancellations = Some(Arc::new(crate::agent_runtime::cancellation::RunCancellations::new()));
    owned.rounds = Arc::new(RoundBook::with_counter(Arc::new(
        super::rounds::tests::PerChars(4),
    )));
    {
        let mut plans = owned.plans.lock().expect("plans");
        for run in [PARENT, "child-ocr", "child-qwen", "child-review"] {
            plans.insert(
                run.to_string(),
                crate::orchestrator::plan::PlanRun::new(
                    run,
                    vec!["the chain".to_string()],
                    crate::orchestrator::plan::Budget::standard(
                        crate::orchestrator::tools::ToolName::ALL.to_vec(),
                    ),
                ),
            );
        }
    }
    leases.bind_run(
        PARENT,
        RunBinding {
            model_id: SPARK.into(),
            class: LeaseClass::Parent,
            parent: None,
            eligible: Vec::new(),
        },
    );
    Process {
        deps: Arc::new(owned),
        leases,
        backend,
        graph,
        _scratch: scratch,
    }
}

fn item(kind: MemoryKind, content: &str, provenance: Provenance) -> MemoryItem {
    MemoryItem {
        item_id: item_id(),
        revision: 1,
        kind,
        agent_id: "ag-spark".into(),
        scope: MemoryScope::Task {
            task_id: PARENT.into(),
        },
        classification: Classification::Internal,
        acl: Acl::for_classification(Classification::Internal, None),
        creator_model_id: None,
        creator_run_id: Some(PARENT.into()),
        provenance,
        content: content.into(),
        sources: Vec::new(),
        artifacts: Vec::new(),
        confidence: None,
        status: ItemStatus::Proposed,
        valid_from: "2026-09-24T00:00:00Z".into(),
        valid_until: None,
        supersedes: None,
        conflicts_with: Vec::new(),
        causal_parents: Vec::new(),
        idempotency_key: None,
        applies_to: None,
        created_at: "2026-09-24T00:00:00Z".into(),
        updated_at: "2026-09-24T00:00:00Z".into(),
    }
}

fn round_request(run: &str, capability: &str) -> Value {
    json!({
        "runId": run,
        // One task: the children work on the parent's shared memory.
        "taskId": PARENT,
        "agentId": format!("ag-{run}"),
        "definitionVersion": 1,
        "modelId": "whatever the run was started with",
        "servedWindow": 16_384,
        "question": "approval note pressure",
        "alreadyCarried": [],
        "capability": capability,
        "reservedToolSchemas": 1_000,
        "reservedOutput": 1_024,
        "reservedFraming": 128,
        "reservedSafety": 256,
    })
}

/// How many blocks of a round carry `needle`.
fn carried(reply: &Value, needle: &str) -> usize {
    reply["blocks"]
        .as_array()
        .expect("blocks")
        .iter()
        .filter(|block| block["content"].as_str().unwrap_or_default().contains(needle))
        .count()
}

fn kind_of(reply: &Value, needle: &str) -> String {
    reply["blocks"]
        .as_array()
        .expect("blocks")
        .iter()
        .find(|block| block["content"].as_str().unwrap_or_default().contains(needle))
        .map(|block| block["kind"].as_str().unwrap_or_default().to_string())
        .unwrap_or_default()
}

fn at_most_one_holder(process: &Process, when: &str) {
    let holders = process.leases.snapshot().holders.len();
    assert!(holders <= 1, "{holders} heavy calls were in flight {when}");
}

/// A delegated child, run the way the worker and child loop run one: the
/// parent suspended on the record, the child's lease, its model served, its
/// own model round compiled from the shared task memory, and everything given
/// back.
async fn run_child(process: &Process, child: &str, model: &str, capability: &str) -> Value {
    process
        .leases
        .suspend(PARENT, Some(child), &format!("{PARENT} delegated to {child}"));
    let guard = process
        .leases
        .acquire(LeaseRequest::new(child, LeaseClass::Child, model).child_of(PARENT))
        .await
        .expect("the child is admitted without waiting on its parent");
    at_most_one_holder(process, &format!("while {child} ran"));
    process
        .leases
        .ensure_served(child, model)
        .await
        .expect("the child's model is served under its lease");
    process.leases.bind_run(
        child,
        RunBinding {
            model_id: model.into(),
            class: LeaseClass::Child,
            parent: Some(PARENT.into()),
            eligible: Vec::new(),
        },
    );
    let reply = refresh(round_request(child, capability), &process.deps)
        .await
        .expect("the child's round opens");
    assert_eq!(reply["lease"]["reentered"], true, "the child's round did not re-enter its lease");
    settle(json!({ "runId": child }), &process.deps).expect("settled");
    drop(guard);
    process.leases.forget(child);
    // On an 8 GB card the next model's admission stops this one; the fake
    // backend is told so, so every hop is a real restart of the next server.
    process.backend.evict(model);
    reply
}

#[tokio::test]
async fn a_prior_write_and_a_correction_survive_four_models_and_a_restart_exactly_once() {
    let dir = tempfile::tempdir().expect("a directory for the stores");

    // ── Before the chain: a file was written, and its receipt recorded ──────
    let first = start_process(dir.path());
    assert!(matches!(
        first.deps.events.begin_effect(
            PARENT,
            WRITE_KEY,
            "artifact.create_approval_note",
            "args-fp",
            "approval-note.docx"
        ),
        EffectLookup::Fresh
    ));
    first.deps.events.settle_effect(
        PARENT,
        WRITE_KEY,
        &Ok("approval-note.docx written as art-note@1".to_string()),
    );
    let mut receipt = item(
        MemoryKind::ToolObservation,
        "approval-note.docx was written",
        Provenance::ToolReceipt {
            run_id: PARENT.into(),
            tool: "artifact.create_approval_note".into(),
            event_seq: 1,
        },
    );
    receipt.artifacts = vec![ArtifactRef {
        artifact_id: "art-note".into(),
        revision: 1,
        sha256: NOTE_SHA.into(),
    }];
    first.graph.commit(receipt, None, &[]).expect("receipt");
    first
        .graph
        .commit(
            item(
                MemoryKind::Goal,
                "Produce the approval note from the scanned inspection report",
                Provenance::Operator {
                    user_id: "priya".into(),
                },
            ),
            None,
            &[],
        )
        .expect("goal");
    let exact_receipt = format!("art-note@1 sha256:{NOTE_SHA}");

    // ── Spark plans ──────────────────────────────────────────────────────
    let spark = refresh(round_request(PARENT, "orchestration"), &first.deps)
        .await
        .expect("Spark's round opens");
    assert_eq!(spark["endpoint"]["modelId"], SPARK);
    assert_eq!(carried(&spark, &exact_receipt), 1);
    let spark_url = spark["endpoint"]["baseUrl"].clone();
    settle(json!({ "runId": PARENT }), &first.deps).expect("settled");
    first.backend.evict(SPARK);

    // ── OCR reads the scan ───────────────────────────────────────────────
    let ocr = run_child(&first, "child-ocr", OCR, "extraction").await;
    assert_eq!(carried(&ocr, &exact_receipt), 1, "OCR's context lost or doubled the write");
    first
        .graph
        .commit(
            item(
                MemoryKind::Fact,
                "Page 2 records the design pressure as 150 psi",
                Provenance::Model {
                    model_id: OCR.into(),
                    run_id: "child-ocr".into(),
                },
            ),
            None,
            &[],
        )
        .expect("the OCR finding");

    // ── The operator corrects the figure while the chain is running ─────────
    first
        .graph
        .commit(
            item(
                MemoryKind::Correction,
                "The design pressure is 10 bar; the 150 psi reading is superseded",
                Provenance::Operator {
                    user_id: "priya".into(),
                },
            ),
            None,
            &[],
        )
        .expect("the correction");
    let correction = "design pressure is 10 bar";

    // ── Qwen drafts, the reviewer checks ─────────────────────────────────
    for (child, model, capability) in [
        ("child-qwen", QWEN, "document"),
        ("child-review", REVIEWER, "review"),
    ] {
        let reply = run_child(&first, child, model, capability).await;
        assert_eq!(carried(&reply, &exact_receipt), 1, "{child} lost or doubled the write");
        assert_eq!(carried(&reply, correction), 1, "{child} lost or doubled the correction");
        assert_eq!(kind_of(&reply, correction), "constraint");
    }

    // ── Spark resumes on a server that was stopped under it ─────────────────
    let resumed = refresh(round_request(PARENT, "orchestration"), &first.deps)
        .await
        .expect("Spark resumes");
    assert_eq!(resumed["endpoint"]["restarted"], true);
    assert_eq!(resumed["endpoint"]["cache"], "cold", "a cache was presented as carried across");
    assert_ne!(resumed["endpoint"]["baseUrl"], spark_url);
    assert!(resumed["lease"]["resumedAfterMs"].is_u64(), "the resumption was not recognised");
    assert_eq!(carried(&resumed, &exact_receipt), 1);
    assert_eq!(carried(&resumed, correction), 1);
    assert_eq!(kind_of(&resumed, correction), "constraint");
    settle(json!({ "runId": PARENT }), &first.deps).expect("settled");

    // The write is not repeated: the ledger answers with what happened.
    assert!(matches!(
        first.deps.events.begin_effect(
            PARENT,
            WRITE_KEY,
            "artifact.create_approval_note",
            "args-fp",
            "approval-note.docx"
        ),
        EffectLookup::Settled(_)
    ));

    // The parent gave way three times and came back once, on the record, in
    // that order.
    let kinds: Vec<TaskEventType> = first
        .deps
        .events
        .events_since(PARENT, 0)
        .expect("history")
        .events
        .into_iter()
        .map(|event| event.event_type)
        .collect();
    let suspended = kinds
        .iter()
        .filter(|kind| **kind == TaskEventType::ModelLeaseSuspended)
        .count();
    assert_eq!(suspended, 3);
    let last_suspended = kinds
        .iter()
        .rposition(|kind| *kind == TaskEventType::ModelLeaseSuspended)
        .expect("suspended");
    let resumed_at = kinds
        .iter()
        .rposition(|kind| *kind == TaskEventType::ModelLeaseResumed)
        .expect("resumed");
    assert!(last_suspended < resumed_at);
    assert!(first.leases.snapshot().is_idle());

    // ── The process goes away ────────────────────────────────────────────
    let cursor_before = first.graph.graph_revision().expect("cursor");
    drop(first);

    // ── And comes back over the same files ───────────────────────────────
    let second = start_process(dir.path());
    assert!(second.leases.snapshot().is_idle(), "a lease survived the restart");
    assert_eq!(second.graph.graph_revision().expect("cursor"), cursor_before);

    let after = refresh(round_request(PARENT, "orchestration"), &second.deps)
        .await
        .expect("the restarted run's round opens");
    assert_eq!(after["endpoint"]["cache"], "cold");
    assert_eq!(carried(&after, &exact_receipt), 1, "the restart lost or doubled the write");
    assert_eq!(carried(&after, correction), 1, "the restart lost or doubled the correction");
    assert_eq!(kind_of(&after, correction), "constraint");
    settle(json!({ "runId": PARENT }), &second.deps).expect("settled");

    assert!(matches!(
        second.deps.events.begin_effect(
            PARENT,
            WRITE_KEY,
            "artifact.create_approval_note",
            "args-fp",
            "approval-note.docx"
        ),
        EffectLookup::Settled(_)
    ));

    // Nothing in the chain wrote a second receipt or a second correction.
    let session = second.deps.session().expect("signed in");
    let (items, _) = second
        .graph
        .snapshot_scopes(
            &session,
            &[MemoryScope::Task {
                task_id: PARENT.into(),
            }],
            None,
        )
        .expect("read");
    let count = |kind: MemoryKind| items.iter().filter(|item| item.kind == kind).count();
    assert_eq!(count(MemoryKind::ToolObservation), 1);
    assert_eq!(count(MemoryKind::Correction), 1);
}
