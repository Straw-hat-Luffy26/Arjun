//! Two real workers on one task, reached the way production reaches them.
//!
//! ## What makes these different from the manager's own tests
//!
//! `subagents::tests` drives the lifecycle with stub workers, which is the right
//! way to test the lifecycle: the lanes, the deadlines and the narrowing have to
//! hold whatever a worker does, so the worker should not be real.
//!
//! These test the opposite property. Every worker here is the one `lib.rs`
//! registers — [`SpecialistWorker`], through
//! [`SpecialistWorker::register_all`] — and every delegation goes through
//! `LocalToolRunner::run` with `ToolName::AgentDelegateReadonly`, which is the
//! *model-facing* surface. Nothing calls `ChildWorker::run` directly. If the
//! registration were removed, or the tool stopped reaching the manager, or the
//! manager stopped reaching the registered worker, these fail — and a test that
//! called the worker directly would not.
//!
//! The stores are real too: a real `KnowledgeIndex` with a real document in it,
//! a real `MemoryGraph`, a real `TaskEventLog` with its migrations. What is not
//! real is the GPU, because none of these roles needs one — see the
//! `subagents::worker` header on why these four are performed without a model
//! turn, and `subagents::scheduling`, whose residency rule is tested against
//! figures rather than against whatever card a developer happens to have.

use std::sync::Arc;

use serde_json::json;

use super::graph_io::TaskMemory;
use super::worker::{SpecialistWorker, WorkerServices};
use super::{load_profiles, InheritedPolicy, SubagentManager};
use crate::agent_runtime::cancellation::RunCancellations;
use crate::agent_runtime::events::TaskEventLog;
use crate::identity::{Role, Session, User};
use crate::knowledge::chunking::{Chunk, ChunkKind};
use crate::knowledge::graph::runtime_memory::MemoryKind;
use crate::knowledge::graph::runtime_store::MemoryGraph;
use crate::knowledge::index::KnowledgeIndex;
use crate::orchestrator::executor::ToolRunner;
use crate::orchestrator::runner::LocalToolRunner;
use crate::orchestrator::tools::{ToolCall, ToolName};
use crate::policy::Classification;
use crate::registry::{ModelManifest, ModelRegistry, ModelRole};

const RUN: &str = "run-workers-1";

/// A tag no model and no document outside this test has ever seen, so a finding
/// that carries it came from the index rather than from anywhere else.
const TAG: &str = "VX-7741-QRT";

fn session() -> Session {
    Session::open(User::new(
        "priya",
        "Priya Sharma",
        vec![Role::Employee, Role::Administrator],
    ))
}

/// The tools a parent run holds. A child's set is filtered from this, so a tool
/// missing here is a tool no child can be given however its profile is written.
fn parent_tools() -> Vec<ToolName> {
    vec![
        ToolName::SearchDocuments,
        ToolName::RunCalculation,
        ToolName::ReadScopedFile,
        ToolName::ValidateArtifact,
        ToolName::AgentDelegateReadonly,
    ]
}

/// The profiles directory this checkout ships.
fn profiles_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(|root| root.join("agents"))
        .unwrap_or_default()
}

fn model_registry() -> ModelRegistry {
    let mut parent = crate::registry::tests::entry("model-parent", 9.0, vec![ModelRole::Reasoning]);
    parent.supports_structured_output = true;
    let mut small = crate::registry::tests::entry(
        "model-small",
        1.0,
        vec![ModelRole::Embedding, ModelRole::DocumentOcr],
    );
    small.supports_structured_output = true;
    ModelRegistry::from_manifest(
        ModelManifest {
            models: vec![parent, small],
        },
        std::path::PathBuf::from("models/registry.json"),
    )
    .expect("a well-formed manifest")
}

fn services(
    index: Arc<KnowledgeIndex>,
    graph: Arc<MemoryGraph>,
    events: Arc<TaskEventLog>,
    cancellations: Arc<RunCancellations>,
) -> Arc<WorkerServices> {
    Arc::new(WorkerServices {
        index,
        graph: Some(graph),
        events,
        scheduler: Arc::new(super::ModelScheduler::new(
            Arc::new(model_registry()),
            Arc::new(crate::serving::ModelServers::new()),
        )),
        session: Arc::new(std::sync::RwLock::new(Some(session()))),
        // No runtime in these tests, so the four mechanical roles take their
        // fallback path and `code-worker` refuses by name. The model loop
        // itself is exercised against a real runtime and two real models in
        // `tests/subagent_model_loop_live.rs`.
        child_loop: None,
        cancellations,
        analyst: None,
    })
}

/// A manager with the production workers registered — exactly the call `lib.rs`
/// makes.
fn manager_with_workers(
    events: Arc<TaskEventLog>,
    services: Arc<WorkerServices>,
) -> SubagentManager {
    let profiles = load_profiles(&profiles_dir());
    assert!(
        !profiles.profiles.is_empty(),
        "no subagent profiles were loaded from {}",
        profiles_dir().display()
    );
    let mut manager = SubagentManager::new(profiles.profiles.clone(), events);
    for worker in SpecialistWorker::register_all(&profiles.profiles, services) {
        manager = manager.with_worker(worker);
    }
    manager
}

/// Everything a delegation needs, assembled the way `lib.rs` assembles it.
struct World {
    manager: SubagentManager,
    index: Arc<KnowledgeIndex>,
    graph: Arc<MemoryGraph>,
    events: Arc<TaskEventLog>,
    cancellations: Arc<RunCancellations>,
    models: ModelRegistry,
    inherited: InheritedPolicy,
    session: Session,
    workspace: std::path::PathBuf,
    _dir: tempfile::TempDir,
}

impl World {
    fn new() -> Self {
        Self::with_tools(parent_tools())
    }

    fn with_tools(tools: Vec<ToolName>) -> Self {
        let dir = tempfile::tempdir().expect("a temporary directory");
        let workspace = dir.path().join(RUN);
        std::fs::create_dir_all(&workspace).expect("the run's own directory");

        let index = Arc::new(KnowledgeIndex::open(dir.path()).expect("an index"));
        index
            .index_document(
                "commissioning-report.pdf",
                Classification::Internal,
                &[Chunk {
                    id: "chunk-1".into(),
                    document_sha256: "c".repeat(64),
                    ordinal: 0,
                    text: format!(
                        "The commissioning tag for the vessel is {TAG}. The seal torque is \
                         47.5 at ambient."
                    ),
                    page: 4,
                    section_path: vec!["Commissioning".into()],
                    kind: ChunkKind::Prose,
                    char_count: 96,
                }],
            )
            .expect("the document is indexed");

        let graph = Arc::new(MemoryGraph::in_memory().expect("a graph"));
        let events = Arc::new(TaskEventLog::in_memory().expect("an event log"));
        // As `lib.rs` wires it: receipts resolve against the task event log.
        graph.set_receipts(
            Arc::clone(&events) as Arc<dyn crate::knowledge::graph::receipts::ReceiptLedger>
        );
        let cancellations = Arc::new(RunCancellations::new());
        let manager = manager_with_workers(
            Arc::clone(&events),
            services(
                Arc::clone(&index),
                Arc::clone(&graph),
                Arc::clone(&events),
                Arc::clone(&cancellations),
            ),
        );

        let session = session();
        let inherited =
            InheritedPolicy::of_run(&session, Classification::Internal, &workspace, &tools);

        Self {
            manager,
            index,
            graph,
            events,
            cancellations,
            models: model_registry(),
            inherited,
            session,
            workspace,
            _dir: dir,
        }
    }

    /// The production tool surface. Nothing in these tests calls a worker
    /// directly.
    async fn delegate(&self, arguments: serde_json::Value) -> Result<String, String> {
        self.delegate_through(&self.manager, arguments).await
    }

    async fn delegate_through(
        &self,
        manager: &SubagentManager,
        arguments: serde_json::Value,
    ) -> Result<String, String> {
        let mut runner = LocalToolRunner::new(self.index.as_ref(), &self.session);
        runner.subagents = Some(manager);
        runner.inherited = Some(&self.inherited);
        runner.run_workspace = Some(&self.workspace);
        runner.models = Some(&self.models);
        runner.parent_model = Some("model-parent");
        runner.task_id = Some(RUN);

        runner
            .run(
                ToolName::AgentDelegateReadonly,
                &ToolCall::new("agent.delegate_readonly", arguments),
                None,
            )
            .await
    }

    fn task_memory(&self, agent: &str) -> TaskMemory {
        TaskMemory::new(
            Arc::clone(&self.graph),
            RUN,
            agent,
            RUN,
            Classification::Internal,
            None,
        )
    }

    fn started_children(&self) -> Vec<crate::agent_runtime::events::TaskEvent> {
        self.events
            .events_since(RUN, 0)
            .expect("the events read")
            .events
            .into_iter()
            .filter(|event| {
                event.event_type == crate::agent_runtime::events::TaskEventType::SubagentStarted
            })
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// The registration is real
// ─────────────────────────────────────────────────────────────────────────────

/// The headline: every role this deployment declares has a worker.
///
/// All five, including `code-worker` — which is performable now that a child can
/// drive a model of its own. What it cannot do is run *without* one, which is
/// the next test.
#[test]
fn every_declared_role_has_a_worker() {
    let world = World::new();
    for profile in SpecialistWorker::PERFORMABLE {
        assert!(
            world.manager.has_worker(profile),
            "{profile} is declared and has no worker"
        );
    }
    assert!(
        SpecialistWorker::PERFORMABLE.contains(&"code-worker"),
        "code-worker should be performable once a child can drive a model"
    );
    assert!(
        world.manager.has_worker("code-worker"),
        "code-worker is declared and has no worker"
    );
}

/// The proof the task asks for: the delegation *tool* reaches a registered
/// worker, and that worker does real work.
#[tokio::test]
async fn the_production_delegation_tool_reaches_a_registered_worker() {
    let world = World::new();
    let out = world
        .delegate(json!({
            "profile": "knowledge-retriever",
            "task": "commissioning tag for the vessel",
        }))
        .await
        .expect("the tool runs");

    assert!(
        out.contains("status: completed"),
        "the worker did not finish: {out}"
    );
    // The citation could only have come from the index this test seeded.
    assert!(
        out.contains("commissioning-report.pdf"),
        "no real retrieval happened: {out}"
    );
    assert!(
        out.contains("Published"),
        "the worker did not publish to the task's shared memory: {out}"
    );
}

/// A read-only delegation cannot start a role that writes.
///
/// `agent.delegate_readonly` is automatic -- nobody is asked -- because the
/// child it starts cannot write. `code-worker` writes files and runs code. Until
/// plan P01 this test sent `code-worker` through the read-only tool and
/// expected it to *start*, which is the widening P01 names; the Rust side now
/// refuses it before anything is recorded or run.
#[tokio::test]
async fn a_read_only_delegation_cannot_start_a_role_that_writes() {
    let world = World::new();
    let refused = world
        .delegate(json!({
            "profile": "code-worker",
            "task": "write a script",
            "files": ["main.py"],
        }))
        .await
        .expect_err("a writer was started through the read-only tool");

    assert!(refused.contains("read-only"), "{refused}");
    assert!(refused.contains("Nothing was started"), "{refused}");
}

/// A role that needs a model, asked for on a machine with no runtime started,
/// refuses by name rather than doing a fraction of the work.
///
/// These tests build their services with `child_loop: None`, which is what a
/// deployment looks like before any run has started. The four mechanical roles
/// carry on; writing a program is not mechanical, so this one stops.
///
/// Dispatched in writer mode through the manager, because that is now the only
/// way `code-worker` starts at all -- see the test above.
#[tokio::test]
async fn a_role_that_needs_a_model_refuses_when_there_is_none() {
    let world = World::new();
    let spawned = world
        .manager
        .spawn(
            "code-worker",
            &world.inherited,
            "write a script",
            vec![crate::subagents::InputRef::WorkspaceFile {
                path: "main.py".to_string(),
            }],
            crate::subagents::certification::Decision {
                model_id: "model-parent".to_string(),
                role: crate::registry::ModelRole::Coding,
                cheaper_than_parent: false,
                reason: "the run's own model".to_string(),
                tier: None,
                score: None,
            },
            &crate::subagents::manager::Dispatch::for_task("code-worker", RUN).writing(),
        )
        .await
        .expect("a writer dispatch is accepted");
    let result = spawned.result();
    let out = format!(
        "status: {} {}",
        result.status.as_str(),
        result.detail.clone().unwrap_or_default()
    );

    assert!(out.contains("status: failed"), "{out}");
    assert!(
        out.contains("needs a model"),
        "the refusal does not say why: {out}"
    );
    assert!(
        out.contains("Nothing was done"),
        "the refusal does not say what was left alone: {out}"
    );
    // And it is the *only* role that stops for this reason.
    let mechanical = world
        .delegate(json!({
            "profile": "knowledge-retriever",
            "task": "commissioning tag",
        }))
        .await
        .expect("the tool answers");
    assert!(mechanical.contains("status: completed"), "{mechanical}");
}

// ─────────────────────────────────────────────────────────────────────────────
// Two workers on one task
// ─────────────────────────────────────────────────────────────────────────────

/// The acceptance case. A publishes a fact; B reads it out of the graph and
/// acts on it. Nothing passes between them but a revision number.
#[tokio::test]
async fn one_worker_publishes_and_another_reads_it_through_the_graph() {
    let world = World::new();

    // A: the retriever. It finds the passage and publishes what it found.
    let first = world
        .delegate(json!({
            "profile": "knowledge-retriever",
            "task": "seal torque at ambient",
        }))
        .await
        .expect("A runs");
    assert!(first.contains("status: completed"), "{first}");

    // What A left behind, read the way anybody would read it.
    let published = world
        .task_memory("knowledge-retriever")
        .read_kind(&world.session, MemoryKind::Fact)
        .expect("the graph reads");
    assert!(
        !published.is_empty(),
        "A published nothing for B to find: {first}"
    );
    let revision = world
        .task_memory("knowledge-retriever")
        .revision()
        .expect("a revision");

    // B: a different worker, on the same task, told to wait for A's revision.
    let second = world
        .delegate(json!({
            "profile": "calculation-checker",
            "task": "re-derive the figures this task will rely on",
            "expressions": ["47.5 * 2"],
            "after_revision": revision,
        }))
        .await
        .expect("B runs");

    assert!(second.contains("status: completed"), "{second}");
    assert!(
        second.contains("95"),
        "B did not actually calculate anything: {second}"
    );
    assert!(
        second.contains("shared memory at revision"),
        "B did not record which world it read: {second}"
    );

    // And the two are attributed to different agents in one task's memory.
    let everything = world
        .task_memory("knowledge-retriever")
        .read(&world.session)
        .expect("reads");
    let agents: std::collections::BTreeSet<&str> = everything
        .iter()
        .map(|item| item.agent_id.as_str())
        .collect();
    assert!(
        agents.contains("knowledge-retriever") && agents.contains("calculation-checker"),
        "both workers should have written to one task's memory, found {agents:?}"
    );
}

/// B waiting for a revision that never lands is told the work it depends on has
/// not finished — rather than starting on a world that does not hold it.
#[tokio::test]
async fn a_worker_waiting_for_a_result_that_never_lands_does_not_start() {
    let world = World::new();
    let out = world
        .delegate(json!({
            "profile": "calculation-checker",
            "task": "check a figure",
            "expressions": ["2 + 2"],
            "after_revision": 9_999,
        }))
        .await
        .expect("the tool answers");

    assert!(out.contains("status: failed"), "{out}");
    assert!(
        out.contains("has not finished"),
        "the refusal does not name what it was waiting for: {out}"
    );
    // It did not calculate anything, because it never began.
    assert!(!out.contains("= 4"), "{out}");
}

/// Nothing of the parent's reaches a child but the objective and the
/// references.
///
/// The packet type makes this structural — there is no variant that carries
/// text — and this is the behavioural half: a worker does its own retrieval, and
/// something the parent knows and did not pass as a reference does not appear.
#[tokio::test]
async fn a_child_is_given_references_and_never_a_transcript() {
    let world = World::new();
    let out = world
        .delegate(json!({
            "profile": "knowledge-retriever",
            "task": "commissioning tag",
            // The parent's own private reasoning, in an argument the delegation
            // does not understand. It must not reach the child.
            "notes": "the parent believes the tag is WRONG-0000",
        }))
        .await
        .expect("the tool runs");

    assert!(
        out.contains("commissioning-report.pdf"),
        "the child did not do its own retrieval: {out}"
    );
    assert!(
        !out.contains("WRONG-0000"),
        "the parent's own text reached the child: {out}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Scope, permissions and refusals
// ─────────────────────────────────────────────────────────────────────────────

/// A child is never given a tool its parent does not hold, so a run without
/// `run_calculation` cannot delegate a calculation check however the profile is
/// written.
#[tokio::test]
async fn a_child_cannot_hold_a_tool_its_parent_does_not() {
    let world = World::with_tools(vec![
        ToolName::SearchDocuments,
        ToolName::AgentDelegateReadonly,
    ]);
    let out = world
        .delegate(json!({
            "profile": "calculation-checker",
            "task": "check a figure",
            "expressions": ["2 + 2"],
        }))
        .await
        .expect("the tool answers");

    assert!(out.contains("status: failed"), "{out}");
    assert!(
        out.contains("not granted"),
        "the refusal does not name the missing tool: {out}"
    );
    assert!(
        out.contains("never given more than its parent"),
        "the refusal does not say why: {out}"
    );
}

/// A worker pointed outside the run's own directory is refused, and the file it
/// was pointed at is not read.
#[tokio::test]
async fn a_worker_cannot_read_outside_the_run_it_belongs_to() {
    let world = World::new();
    let out = world
        .delegate(json!({
            "profile": "document-extractor",
            "task": "read what this task produced",
            "files": ["../../secret.txt", "../secret.txt"],
        }))
        .await
        .expect("the tool answers");

    assert!(
        out.contains("was not read"),
        "the escape was not refused: {out}"
    );
}

/// A worker that is pointed at nothing is refused before it starts, rather than
/// being dispatched with an empty packet.
#[tokio::test]
async fn a_worker_that_needs_references_is_not_dispatched_without_any() {
    let world = World::new();
    let refusal = world
        .delegate(json!({
            "profile": "document-extractor",
            "task": "read the files",
        }))
        .await
        .expect_err("a packet with no inputs is refused");
    assert!(refusal.contains("named none"), "{refusal}");
    assert!(
        world.started_children().is_empty(),
        "a child was started with an empty packet"
    );
}

/// A worker inside the run's directory reads what is actually there.
#[tokio::test]
async fn a_worker_reads_a_file_the_run_owns() {
    let world = World::new();
    std::fs::write(world.workspace.join("draft.md"), "one\ntwo\nthree\n")
        .expect("the run writes a file");

    let out = world
        .delegate(json!({
            "profile": "document-extractor",
            "task": "read the draft",
            "files": ["draft.md"],
        }))
        .await
        .expect("the tool runs");

    assert!(out.contains("status: completed"), "{out}");
    assert!(out.contains("3 line(s)"), "it did not read the file: {out}");
}

// ─────────────────────────────────────────────────────────────────────────────
// Duplicates, cancellation and model choice
// ─────────────────────────────────────────────────────────────────────────────

/// The same piece of work asked for twice starts one child, and the second
/// caller is told it is reusing the first's answer.
#[tokio::test]
async fn the_same_work_dispatched_twice_starts_one_child() {
    let world = World::new();
    let arguments = json!({
        "profile": "knowledge-retriever",
        "task": "commissioning tag for the vessel",
    });

    let first = world.delegate(arguments.clone()).await.expect("first");
    let second = world.delegate(arguments).await.expect("second");

    assert!(first.contains("status: completed"), "{first}");
    assert!(
        second.contains("reused from an earlier call"),
        "a second child was started: {second}"
    );
    assert_eq!(
        world.started_children().len(),
        1,
        "two children were started for one piece of work"
    );
}

/// The duplicate check survives the process, which is the case an in-memory map
/// cannot cover.
///
/// The manager's slot map is rebuilt with the manager; the ledger is not. A
/// second manager over the *same* event log finds the first's record and does
/// not dispatch again — which is what a run interrupted mid-delegation and
/// picked back up would do.
#[tokio::test]
async fn a_duplicate_dispatch_after_a_crash_is_recognised() {
    let world = World::new();
    let arguments = json!({
        "profile": "knowledge-retriever",
        "task": "commissioning tag for the vessel",
    });
    world.delegate(arguments.clone()).await.expect("first");

    // The process dies and comes back: a fresh manager, with a fresh slot map,
    // over the same durable log.
    let restarted = manager_with_workers(
        Arc::clone(&world.events),
        services(
            Arc::clone(&world.index),
            Arc::clone(&world.graph),
            Arc::clone(&world.events),
            Arc::clone(&world.cancellations),
        ),
    );

    let after = world
        .delegate_through(&restarted, arguments)
        .await
        .expect("the tool answers");

    assert!(
        after.contains("already done under the same key")
            || after.contains("reused from an earlier call"),
        "the restarted process dispatched the work a second time: {after}"
    );
}

/// A child's model is chosen against the registry rather than asserted.
#[tokio::test]
async fn a_childs_model_is_chosen_and_recorded() {
    let world = World::new();
    world
        .delegate(json!({
            "profile": "knowledge-retriever",
            "task": "commissioning tag",
        }))
        .await
        .expect("the tool runs");

    let started = world
        .started_children()
        .into_iter()
        .next()
        .expect("a child started");
    let model = started.payload.get("model").expect("a model decision");
    let model_id = model
        .get("modelId")
        .and_then(|value| value.as_str())
        .unwrap_or_default();

    assert_ne!(
        model_id, "parent-inherited",
        "the routing placeholder is still in use"
    );
    assert!(
        world.models.find(model_id).is_some(),
        "a child was routed to {model_id:?}, which no registry resolves"
    );
    // And the reason is recorded, so an operator can see what it rested on.
    assert!(
        model
            .get("reason")
            .and_then(|value| value.as_str())
            .is_some_and(|reason| !reason.trim().is_empty()),
        "no reason was recorded for the routing decision"
    );
}

/// A child records the parent/child relationship durably, so a completion check
/// after a restart still knows the run delegated.
#[tokio::test]
async fn the_parent_child_relationship_is_durable() {
    let world = World::new();
    world
        .delegate(json!({
            "profile": "knowledge-retriever",
            "task": "commissioning tag",
            "deliverable": "a citation for the tag",
        }))
        .await
        .expect("the tool runs");

    let children = crate::agent_runtime::completion::children_of(&world.events, RUN);
    assert_eq!(children.len(), 1, "{children:?}");
    let child = &children[0];
    assert_eq!(child.profile, "knowledge-retriever");
    assert!(child.complete, "{child:?}");
    assert_eq!(child.deliverable, "a citation for the tag");
    // The ids a later step can go and read, carried on the durable record
    // rather than only in the result the parent happened to be holding.
    assert!(
        !child.published.is_empty(),
        "the child published nothing a sibling could read: {child:?}"
    );
    assert!(
        child.honoured(),
        "the worker's output did not honour its contract: {}",
        child.explain()
    );
}

/// Stopping the run stops its children.
#[tokio::test]
async fn cancelling_the_run_cancels_its_workers() {
    let world = World::new();

    // The person presses Stop. The child registers under the same run id, so it
    // finds the token already cancelled.
    world.cancellations.register(&[RUN]);
    assert!(world.cancellations.cancel(RUN), "the run was not cancellable");

    let out = world
        .delegate(json!({
            "profile": "knowledge-retriever",
            "task": "commissioning tag",
        }))
        .await;

    match out {
        // The worker refuses; the manager turns that into a typed failure.
        Ok(rendered) => assert!(
            !rendered.contains("status: completed"),
            "a cancelled run's worker finished anyway: {rendered}"
        ),
        // Or the refusal reaches the caller directly. Both are correct; what
        // must not happen is a completed child.
        Err(refusal) => assert!(!refusal.is_empty()),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// The parent's completion check
// ─────────────────────────────────────────────────────────────────────────────

/// A child that claims success with nothing behind it does not make the run
/// complete.
#[test]
fn completion_rejects_a_child_that_claims_success_with_no_evidence() {
    use crate::agent_runtime::completion::{verify, ChildOutcome, CompletionInputs};

    let fake = ChildOutcome {
        child_id: "c-1".into(),
        profile: "knowledge-retriever".into(),
        status: "completed".into(),
        // Everything a lying worker would set.
        complete: true,
        deliverable: "a citation for each figure".into(),
        findings: 6,
        // And the one thing it cannot fake: something a reader could check.
        evidenced: 0,
        published: Vec::new(),
        blocking: false,
    };

    let verification = verify(
        &CompletionInputs {
            has_answer: true,
            grounding_ready: Some(true),
            children: vec![fake],
            ..Default::default()
        },
        chrono::Utc::now(),
    );

    assert!(
        !verification.passed(),
        "a child's word was enough: {}",
        verification.explain()
    );
    assert!(
        verification
            .blocking()
            .iter()
            .any(|criterion| criterion.criterion_id == "children.contract_honoured"),
        "{}",
        verification.explain()
    );
}

/// A worker a later step was waiting for, which did not finish, blocks the run —
/// and reads differently from one nobody needed.
#[test]
fn a_dependent_step_is_blocked_by_a_worker_that_did_not_finish() {
    use crate::agent_runtime::completion::{verify, ChildOutcome, CompletionInputs};

    let child = |blocking: bool| ChildOutcome {
        child_id: "c-1".into(),
        profile: "knowledge-retriever".into(),
        status: "timed_out".into(),
        complete: false,
        deliverable: String::new(),
        findings: 0,
        evidenced: 0,
        published: Vec::new(),
        blocking,
    };

    let blocked = verify(
        &CompletionInputs {
            has_answer: true,
            grounding_ready: Some(true),
            children: vec![child(true)],
            ..Default::default()
        },
        chrono::Utc::now(),
    );
    assert!(blocked
        .blocking()
        .iter()
        .any(|criterion| criterion.criterion_id == "children.none_blocking"));

    let merely_degraded = verify(
        &CompletionInputs {
            has_answer: true,
            grounding_ready: Some(true),
            children: vec![child(false)],
            ..Default::default()
        },
        chrono::Utc::now(),
    );
    assert!(
        !merely_degraded
            .blocking()
            .iter()
            .any(|criterion| criterion.criterion_id == "children.none_blocking"),
        "a worker nobody was waiting for should not read as a blocked step"
    );
}

/// A worker that finished and found nothing is an answer, not a failure.
#[test]
fn a_worker_that_legitimately_found_nothing_does_not_block_the_run() {
    use crate::agent_runtime::completion::{verify, ChildOutcome, CompletionInputs};

    let verification = verify(
        &CompletionInputs {
            has_answer: true,
            grounding_ready: Some(true),
            children: vec![ChildOutcome {
                child_id: "c-1".into(),
                profile: "knowledge-retriever".into(),
                status: "completed".into(),
                complete: true,
                deliverable: String::new(),
                findings: 0,
                evidenced: 0,
                published: Vec::new(),
                blocking: false,
            }],
            ..Default::default()
        },
        chrono::Utc::now(),
    );
    assert!(verification.passed(), "{}", verification.explain());
}

// ─────────────────────────────────────────────────────────────────────────────
// Plan P02: receipts, versions and sharing on the production delegation path
// ─────────────────────────────────────────────────────────────────────────────

/// Every item a worker published in this task, whatever agent wrote it.
fn published_items(world: &World) -> Vec<crate::knowledge::graph::runtime_memory::MemoryItem> {
    world
        .task_memory("reader")
        .read(&world.session)
        .expect("the graph reads")
}

/// **A publishes a real observation; B reads its version.**
///
/// A is dispatched through the production tool path (`agent.delegate_readonly`
/// → manager → worker) and re-derives two figures. Each finding is published
/// with the receipt of *its own* calculation -- two different events on A's
/// own child run, not one event, not the parent's run -- and each receipt
/// resolves against the event log, so each is admitted. B, a different agent,
/// reads the exact version A published, at the revision it landed at.
#[tokio::test]
async fn a_publishes_real_observations_each_on_its_own_receipt_and_b_reads_that_version() {
    use crate::knowledge::graph::receipts::ReceiptLedger;
    use crate::knowledge::graph::runtime_memory::{Basis, ItemStatus, Provenance};

    let world = World::new();
    let out = world
        .delegate(json!({
            "profile": "calculation-checker",
            "task": "re-derive the wall thickness margin",
            "expressions": ["9.0 - 8.2", "47.5 * 2"],
        }))
        .await
        .expect("A runs");
    assert!(out.contains("status: completed"), "{out}");

    let observations: Vec<_> = published_items(&world)
        .into_iter()
        .filter(|item| item.kind == MemoryKind::ToolObservation)
        .collect();
    assert_eq!(observations.len(), 2, "two calculations, two observations: {out}");

    let mut seqs = std::collections::BTreeSet::new();
    for item in &observations {
        let Provenance::ToolReceipt {
            run_id,
            tool,
            event_seq,
            output_sha256,
        } = &item.provenance
        else {
            panic!("an observation was published without a receipt: {item:?}");
        };
        // The child's own run, never the parent's.
        assert_ne!(run_id, RUN, "attributed to the parent run");
        assert_eq!(tool, ToolName::RunCalculation.as_str());
        // Resolved in the event log, which is what admitted it.
        world
            .events
            .verify(run_id, *event_seq, tool, output_sha256.as_deref().expect("a hash"))
            .expect("the receipt is backed by a real event");
        assert_eq!(item.status, ItemStatus::Admitted, "{item:?}");
        assert_eq!(item.basis(), Basis::Measured);
        seqs.insert(*event_seq);
    }
    assert_eq!(seqs.len(), 2, "two findings share one receipt: {seqs:?}");

    // B reads the version A published, at the revision it landed at.
    let chosen = &observations[0];
    let versions = world
        .graph
        .versions_of(&world.session, &chosen.item_id, None)
        .expect("the history reads");
    assert_eq!(versions.len(), 1);
    let landed_at = versions[0].graph_revision;
    let b = world.task_memory("ag-other-agent");
    b.await_requirement(
        super::Requirement::AtLeast {
            graph_revision: landed_at,
        },
        std::time::Duration::from_secs(2),
    )
    .await
    .expect("B reaches A's revision");
    let as_of = world
        .graph
        .snapshot_as_of(
            &world.session,
            &crate::knowledge::graph::runtime_memory::MemoryScope::Task {
                task_id: RUN.to_string(),
            },
            None,
            landed_at,
        )
        .expect("B reads as of A's revision");
    let read = as_of
        .items
        .iter()
        .find(|item| item.item_id == chosen.item_id)
        .expect("B found A's item at A's revision");
    assert_eq!(read.revision, chosen.revision);
    assert_eq!(read.content, chosen.content);
    assert_eq!(read.agent_id, "calculation-checker");
}

/// A retrieval's passages are what a source says, each on the search that
/// returned it; nothing a model inferred is labelled as either.
#[tokio::test]
async fn a_retrieval_is_published_as_source_text_on_its_own_search() {
    use crate::knowledge::graph::runtime_memory::{Basis, ItemStatus};

    let world = World::new();
    world
        .delegate(json!({ "profile": "knowledge-retriever", "task": "seal torque at ambient" }))
        .await
        .expect("A runs");
    let facts: Vec<_> = published_items(&world)
        .into_iter()
        .filter(|item| item.kind == MemoryKind::Fact)
        .collect();
    assert!(!facts.is_empty());
    for fact in facts {
        assert_eq!(fact.basis(), Basis::SourceText, "{fact:?}");
        assert_eq!(fact.status, ItemStatus::Admitted, "{}", fact.content);
        assert!(!fact.sources.is_empty(), "a source-text claim with no source");
    }
}

/// **Invalid receipts remain proposals.** A claim naming an event that does
/// not exist, an event of another tool, or a failed call is kept -- labelled --
/// and never admitted.
#[tokio::test]
async fn a_receipt_the_event_log_does_not_back_stays_a_proposal() {
    use crate::agent_runtime::events::{EventDraft, TaskEventType};
    use crate::knowledge::graph::receipts::Receipt;
    use crate::knowledge::graph::runtime_memory::SourceRef;
    use super::graph_io::Claim;

    let world = World::new();
    let failed = world
        .events
        .append(
            EventDraft::new("child-x", TaskEventType::ToolFailed, "priya").with(json!({
                "toolCallId": "tc-1",
                "tool": "knowledge.search_authorized",
                "reason": "the index was unavailable",
            })),
        )
        .expect("appended");

    let forged = |key: &str, seq: i64, tool: &str| Claim {
        kind: MemoryKind::Fact,
        content: format!("forged {key}: the torque is 90"),
        sources: vec![SourceRef {
            sha256: "c".repeat(64),
            locator: "page 4".into(),
            extraction_revision: None,
        }],
        artifacts: Vec::new(),
        confidence: Some(1.0),
        causal_parents: Vec::new(),
        idempotency_key: key.to_string(),
        receipt: Some(Receipt {
            run_id: "child-x".into(),
            tool: tool.into(),
            event_seq: seq,
            output_sha256: "d".repeat(64),
        }),
        depends_on: Vec::new(),
    };
    let memory = world.task_memory("ag-forger");
    for claim in [
        forged("no-such-event", 9_999, "knowledge.search_authorized"),
        forged("failed-call", failed.seq, "knowledge.search_authorized"),
        forged("wrong-tool", failed.seq, "calculation.evaluate_with_units"),
    ] {
        let key = claim.idempotency_key.clone();
        let published = memory.publish(claim, None, &world.session).expect("kept");
        assert_eq!(published.status, "proposed", "{key} was admitted: {}", published.because);
        assert!(published.because.contains("does not back it"), "{key}: {}", published.because);
    }
}

/// **Private scratch versus task sharing**, from the definition's own policy.
///
/// A worker whose definition does not share with its task publishes into its
/// own scratch: it reads its own findings back, and a sibling does not see them.
#[tokio::test]
async fn a_definition_that_does_not_share_publishes_privately() {
    use crate::subagents::{certification::Decision, manager::Dispatch, InputRef};

    let world = World::new();
    // The manager resolves this bundled role, and the packet carries sharing.
    // A registry row with sharing off is the other way in; here the packet is
    // what the worker reads, so the dispatch goes straight to the manager with
    // a definition source that turns sharing off.
    struct Private(Vec<crate::subagents::AgentProfile>);
    impl crate::subagents::DefinitionSource for Private {
        fn resolve(
            &self,
            key: &str,
        ) -> Result<crate::subagents::ResolvedDefinition, crate::subagents::Unresolved> {
            let profile = self
                .0
                .iter()
                .find(|p| p.name == key)
                .ok_or(crate::subagents::Unresolved::NoDefinition { key: key.into() })?;
            let mut resolved = crate::subagents::ResolvedDefinition::from_bundled(profile);
            resolved.shared_with_task = false;
            Ok(resolved)
        }
    }
    let profiles = load_profiles(&profiles_dir()).profiles;
    let manager = manager_with_workers(
        Arc::clone(&world.events),
        services(
            Arc::clone(&world.index),
            Arc::clone(&world.graph),
            Arc::clone(&world.events),
            Arc::clone(&world.cancellations),
        ),
    )
    .with_definitions(Arc::new(Private(profiles)));

    manager
        .spawn(
            "calculation-checker",
            &world.inherited,
            "a private check",
            vec![InputRef::Expression {
                expression: "3 * 3".into(),
            }],
            Decision {
                model_id: "model-parent".into(),
                role: ModelRole::Reasoning,
                cheaper_than_parent: false,
                reason: "the run's own model".into(),
                tier: None,
                score: None,
            },
            &Dispatch::for_task("calculation-checker", RUN),
        )
        .await
        .expect("runs");

    let mine = world
        .task_memory("calculation-checker")
        .read(&world.session)
        .expect("reads");
    assert!(
        mine.iter().any(|item| item.content.contains("3 * 3")),
        "the agent could not read its own scratch: {mine:?}"
    );
    let sibling = world.task_memory("knowledge-retriever").read(&world.session).expect("reads");
    assert!(
        !sibling.iter().any(|item| item.content.contains("3 * 3")),
        "a sibling read another agent's private scratch"
    );
}

/// The worker's receipts are real events on the child's own run, one per
/// action, and a publication reaches the parent run's history through the
/// outbox after the graph committed.
#[tokio::test]
async fn a_publication_reaches_the_parent_run_through_the_outbox() {
    use crate::agent_runtime::events::TaskEventType;

    let world = World::new();
    world
        .delegate(json!({
            "profile": "calculation-checker",
            "task": "one figure",
            "expressions": ["2 + 2"],
        }))
        .await
        .expect("runs");
    // The deliverer `lib.rs` runs after each commit and at start.
    let report = world
        .graph
        .deliver_pending(&[world.events.as_ref()], "2026-09-24T00:00:00Z")
        .expect("delivers");
    assert_eq!(report.delivered, 1, "{report:?}");
    let published: Vec<_> = world
        .events
        .events_since(RUN, 0)
        .expect("reads")
        .events
        .into_iter()
        .filter(|event| event.event_type == TaskEventType::MemoryPublished)
        .collect();
    assert_eq!(published.len(), 1);
    assert_eq!(published[0].payload["agentId"], "calculation-checker");
}
