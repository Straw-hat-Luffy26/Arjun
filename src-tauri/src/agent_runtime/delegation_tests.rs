//! The orchestrator's tools, driven the way the main run drives them.
//!
//! Every call goes through the runtime's own `tool.authorize` then
//! `tool.execute` (`authorize`, `execute`), so the gateway, the plan budget, the
//! approval queue, the dispatcher and the durable receipts are all in the path.
//! The workers are the production `SpecialistWorker`s `lib.rs` registers
//! (retriever, calculation checker, code worker, …) over a real index, a real
//! memory graph and a real event log — except where a test needs a child that
//! is still running when something happens to it (a cancel, a correction, a
//! crash). Those use [`Stalling`], a stub that never finishes on its own and
//! says so; what is under test there is the stop path, not the work.
//!
//! Evidence label: deterministic test. No model is involved; see
//! `tests/agent_runtime.rs` for the same tools reached by a model through the
//! real Node runtime.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use serde_json::{json, Value};

use super::super::*;
use crate::agent_runtime::cancellation::RunCancellations;
use crate::agent_runtime::events::plans::JobStatus;
use crate::agent_runtime::events::{TaskEventLog, TaskEventType};
use crate::agent_runtime::task_plan::StepStatus;
use crate::identity::{Role, Session, User};
use crate::knowledge::chunking::{Chunk, ChunkKind};
use crate::knowledge::graph::runtime_memory::{ItemStatus, MemoryKind, MemoryScope};
use crate::knowledge::graph::runtime_store::MemoryGraph;
use crate::policy::Classification;
use crate::registry::{ModelManifest, ModelRegistry, ModelRole};
use crate::subagents::manager::ChildWorker;
use crate::subagents::{
    load_profiles, ChildResult, EffectivePolicy, SpecialistWorker, SubagentManager, WorkerServices,
};

const RUN: &str = "r";
const TAG: &str = "VX-7741-QRT";

fn session() -> Session {
    Session::open(User::new("priya", "Priya Sharma", vec![Role::Employee, Role::Administrator]))
}

fn reviewer() -> Session {
    Session::open(User::new("ravi", "Ravi Menon", vec![Role::Administrator]))
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
        ModelManifest { models: vec![parent, small] },
        std::path::PathBuf::from("models/registry.json"),
    )
    .expect("a well-formed manifest")
}

fn profiles_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(|root| root.join("agents"))
        .unwrap_or_default()
}

/// A child that is still working when the test acts on it. It never finishes
/// on its own; only a stop ends it.
struct Stalling(&'static str);

#[async_trait]
impl ChildWorker for Stalling {
    fn profile(&self) -> &str {
        self.0
    }
    async fn run(
        &self,
        _packet: &crate::subagents::ChildTaskPacket,
        _policy: &EffectivePolicy,
    ) -> Result<ChildResult, String> {
        tokio::time::sleep(Duration::from_secs(600)).await;
        Err("a stalling stub reached its own end, which no test waits for".into())
    }
}

struct World {
    deps: Arc<RuntimeDeps>,
    graph: Arc<MemoryGraph>,
    /// The stop table `agent_abort_run` writes into, as the manager watches it.
    cancellations: Arc<RunCancellations>,
    _dir: Option<tempfile::TempDir>,
}

#[derive(Default)]
struct Options {
    /// Roles performed by [`Stalling`] instead of their production worker.
    stalling: Vec<&'static str>,
    /// Roles left with no worker at all, as a build that has not got one yet.
    missing: Vec<&'static str>,
    /// A durable event log at this directory instead of an in-memory one.
    events_at: Option<std::path::PathBuf>,
}

impl World {
    fn new() -> Self {
        Self::with(Options::default())
    }

    fn with(options: Options) -> Self {
        let (base, dir) = super::super::tests::deps_with(Arc::new(std::sync::RwLock::new(Some(session()))));
        base.index
            .index_document(
                "commissioning-report.pdf",
                Classification::Internal,
                &[Chunk {
                    id: "chunk-1".into(),
                    document_sha256: "c".repeat(64),
                    ordinal: 0,
                    text: format!(
                        "The commissioning tag for the vessel is {TAG}. The seal torque is 47.5 \
                         newton metres at ambient."
                    ),
                    page: 4,
                    section_path: vec!["Commissioning".into()],
                    kind: ChunkKind::Prose,
                    char_count: 96,
                }],
            )
            .expect("the document is indexed");
        std::fs::write(dir.path().join("runs").join(RUN).join("a.txt"), "line one\nline two\n").expect("a workspace file");

        let events = match &options.events_at {
            Some(path) => Arc::new(TaskEventLog::open(path).expect("a durable event log")),
            None => base.events.clone(),
        };
        let graph = Arc::new(MemoryGraph::in_memory().expect("a graph").with_receipts(events.clone()));
        let registry = Arc::new(model_registry());
        let cancellations = Arc::new(RunCancellations::new());
        let services = Arc::new(WorkerServices {
            index: base.index.clone(),
            graph: Some(graph.clone()),
            events: events.clone(),
            scheduler: Arc::new(crate::subagents::ModelScheduler::new(
                registry.clone(),
                Arc::new(crate::serving::ModelServers::new()),
            )),
            session: base.session.clone(),
            child_loop: None,
            cancellations: cancellations.clone(),
            analyst: Some(crate::subagents::AnalystServices {
                extraction: base.extraction.clone(),
                documents: base.documents.clone(),
                conversations: base.run_to_conversation.clone(),
            }),
        });
        let profiles = load_profiles(&profiles_dir()).profiles;
        assert!(!profiles.is_empty(), "no profiles in {}", profiles_dir().display());
        let mut manager =
            SubagentManager::new(profiles.clone(), events.clone()).with_cancellations(cancellations.clone());
        for worker in SpecialistWorker::register_all(&profiles, services) {
            let role = worker.profile().to_string();
            if options.missing.contains(&role.as_str()) || options.stalling.contains(&role.as_str()) {
                continue;
            }
            manager = manager.with_worker(worker);
        }
        for role in &options.stalling {
            manager = manager.with_worker(Arc::new(Stalling(role)));
        }

        base.checkpoints.lock().unwrap().insert(
            RUN.to_string(),
            crate::agent_runtime::resume::CheckpointSeed {
                attempt_id: "attempt-1".into(),
                plan_hash: "plan".into(),
                policy_hash: "policy".into(),
                workspace_hash: "workspace".into(),
                model_id: "model-parent".into(),
                committed_notes: Default::default(),
                manifest: None,
            },
        );

        let deps = Arc::new(RuntimeDeps {
            memory_graph: Some(graph.clone()),
            conversation_artifacts: base.conversation_artifacts.clone(),
            registry: Some(registry),
            index: base.index.clone(),
            session: base.session.clone(),
            workspaces: base.workspaces.clone(),
            approvals: base.approvals.clone(),
            calculations: base.calculations.clone(),
            passages: base.passages.clone(),
            produced: base.produced.clone(),
            calls: base.calls.clone(),
            plans: base.plans.clone(),
            events,
            skills: base.skills.clone(),
            hooks: base.hooks.clone(),
            memory: base.memory.clone(),
            checkpoints: base.checkpoints.clone(),
            emit: base.emit.clone(),
            emit_durable: base.emit_durable.clone(),
            subagents: Arc::new(manager),
            multimodal: base.multimodal.clone(),
            audit_health: base.audit_health.clone(),
            documents: base.documents.clone(),
            run_to_conversation: base.run_to_conversation.clone(),
            notebooks: base.notebooks.clone(),
            jobs: Arc::default(),
            extraction: base.extraction.clone(),
        });
        World { deps, graph, cancellations, _dir: Some(dir) }
    }

    fn plan(&self) -> task_plan::TaskPlan {
        self.deps.events.latest_plan(RUN).expect("readable").expect("a plan")
    }

    fn step(&self, id: &str) -> task_plan::Step {
        self.plan().step(id).cloned().expect("the step exists")
    }

    fn jobs(&self) -> Vec<crate::agent_runtime::events::plans::JobRecord> {
        self.deps.events.jobs_for_run(RUN).expect("readable")
    }

    fn events_of(&self, kind: TaskEventType) -> Vec<crate::agent_runtime::events::TaskEvent> {
        self.deps
            .events
            .events_since(RUN, 0)
            .expect("readable")
            .events
            .into_iter()
            .filter(|event| event.event_type == kind)
            .collect()
    }

    /// Waits until a job has settled, however long the lanes take.
    async fn settled(&self, job_id: &str) -> crate::agent_runtime::events::plans::JobRecord {
        for _ in 0..400 {
            let job = self.deps.events.job(job_id).unwrap().expect("the job exists");
            if !job.status.is_live() && self.step(&job.step_id).job_id.as_deref() != Some(job_id) {
                return job;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        panic!("job {job_id} never settled");
    }
}

fn request(tool: &str, args: Value) -> Value {
    json!({
        "runId": RUN,
        "toolCallId": format!("tc-{tool}-{}", uuid::Uuid::new_v4()),
        "tool": tool,
        "args": args,
    })
}

/// Authorise, then spend the grant — the loop's own two calls.
async fn call(deps: &Arc<RuntimeDeps>, tool: &str, args: Value) -> Result<String, String> {
    let request = request(tool, args);
    let allow = authorize(request.clone(), deps).await.map_err(|e| format!("authorise: {}", e.message))?;
    spend(deps, request, allow).await
}

/// Authorise a call a person is asked about, answering as that person.
async fn decided(deps: &Arc<RuntimeDeps>, tool: &str, args: Value, approve: bool) -> Result<String, String> {
    let request = request(tool, args);
    let queue = deps.approvals.clone();
    let waiting = tokio::spawn({
        let deps = deps.clone();
        let request = request.clone();
        async move { authorize(request, &deps).await }
    });
    let item = loop {
        if let Some(item) = queue.pending().first().cloned() {
            break item;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    };
    queue
        .decide(&reviewer(), &item.request.id, approve, (!approve).then_some("not now"))
        .expect("decided");
    let allow = waiting.await.expect("finished").map_err(|e| format!("authorise: {}", e.message))?;
    spend(deps, request, allow).await
}

async fn spend(deps: &Arc<RuntimeDeps>, request: Value, allow: Value) -> Result<String, String> {
    let grant = allow
        .get("grant")
        .and_then(Value::as_str)
        .ok_or_else(|| format!("refused: {}", allow.get("reason").and_then(Value::as_str).unwrap_or("?")))?
        .to_string();
    let mut spent = request;
    spent["grant"] = json!(grant);
    let result = execute(spent, deps).await.map_err(|e| e.message)?;
    Ok(result["text"].as_str().unwrap_or_default().to_string())
}

fn job_id_in(text: &str) -> String {
    text.split_whitespace()
        .find(|word| word.starts_with("job-"))
        .map(|word| word.trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '-').to_string())
        .unwrap_or_else(|| panic!("no job id in: {text}"))
}

/// A plan with a retriever step and a calculation step that depends on it.
async fn two_specialist_plan(deps: &Arc<RuntimeDeps>) {
    let written = call(
        deps,
        "task.plan_update",
        json!({
            "base_version": 0,
            "operations": [
                {"op": "set_goal", "text": "Confirm the seal torque figure and its source"},
                {"op": "add_constraint", "text": "Cite the commissioning report for every figure"},
                {"op": "add_step", "id": "find", "title": "Find the torque in the commissioning report",
                 "kind": "delegate", "role": "knowledge-retriever",
                 "acceptance": ["childCompleted", "memoryPublished"]},
                {"op": "add_step", "id": "check", "title": "Re-derive the doubled torque",
                 "kind": "delegate", "role": "calculation-checker", "depends_on": ["find"]},
            ]
        }),
    )
    .await
    .expect("the plan is written");
    assert!(written.contains("Plan version 1 written"), "{written}");
}

#[tokio::test]
async fn a_dependent_two_specialist_plan_runs_in_order_on_receipts() {
    let world = World::new();
    let deps = &world.deps;
    two_specialist_plan(deps).await;

    // The dependent step cannot start first, and asking costs no attempt.
    let early = call(deps, "agent.delegate", json!({
        "step": "check", "role": "calculation-checker",
        "objective": "Re-derive twice the seal torque", "expressions": ["47.5 * 2"],
    }))
    .await
    .expect_err("the dependent step waits");
    assert!(early.contains("waits on find (pending)"), "{early}");
    assert_eq!(world.step("check").repair.attempts, 0);
    assert!(world.jobs().is_empty());

    // First specialist: the real retriever over the real index.
    let found = call(deps, "agent.delegate", json!({
        "step": "find", "role": "knowledge-retriever",
        "objective": "commissioning tag for the vessel",
        "wait_seconds": 30,
    }))
    .await
    .expect("the retriever job is dispatched");
    assert!(found.contains("knowledge-retriever"), "{found}");
    let first = world.settled(&job_id_in(&found)).await;
    assert_eq!(first.status, JobStatus::Completed, "{:?}", first.result);
    // The definition resolved at dispatch, and the lease decision, are recorded.
    assert_eq!(first.definition_origin, "bundled-profile");
    assert!(!first.lease.is_empty());
    let find = world.step("find");
    assert_eq!(find.status, StepStatus::Completed, "{:?}", find.note);
    assert!(find.receipts.iter().any(|receipt| receipt.reference == first.job_id && receipt.event_seq.is_some()));
    assert!(!find.outputs.is_empty(), "the retriever published nothing");

    // Second specialist, now ready, waits for the first one's graph revision.
    let checked = call(deps, "agent.delegate", json!({
        "step": "check", "role": "calculation-checker",
        "objective": "Re-derive twice the seal torque", "expressions": ["47.5 * 2"],
        "wait_seconds": 30,
    }))
    .await
    .expect("the checker job is dispatched");
    let second = world.settled(&job_id_in(&checked)).await;
    assert_eq!(second.status, JobStatus::Completed, "{:?}", second.result);
    assert_eq!(world.step("check").status, StepStatus::Completed, "{:?} / {:?}", world.step("check").note, second.result);
    let started = world.events_of(TaskEventType::SubagentStarted);
    assert_eq!(started.len(), 2);
    let after = &started[1].payload["requirement"];
    assert_eq!(after["mode"], "atLeast", "the checker did not wait for the retriever: {after}");

    // The backend's gate agrees, and so does the completion verifier.
    let gate = task_plan::gate(&world.plan());
    assert!(gate.unfinished.is_empty(), "{:?}", gate.unfinished);
    let verdict = completion::verify(
        &completion::CompletionInputs {
            has_answer: true,
            grounding_ready: Some(true),
            task_plan: Some(gate),
            ..Default::default()
        },
        chrono::Utc::now(),
    );
    assert!(verdict.criteria.iter().any(|c| c.criterion_id == "taskplan.steps_accepted"
        && c.status == completion::CriterionStatus::Passed));

    // Goal, constraint and plan are in the task's shared memory, admitted on
    // the receipts of the calls that returned them.
    let items = world
        .graph
        .snapshot(&session(), &MemoryScope::Task { task_id: RUN.into() }, None)
        .expect("readable");
    for (kind, id) in [
        (MemoryKind::Goal, format!("mi-goal-{RUN}")),
        (MemoryKind::Plan, format!("mi-plan-{RUN}")),
        (MemoryKind::Constraint, format!("mi-constraint-{RUN}-1")),
    ] {
        let item = items.iter().find(|item| item.item_id == id).unwrap_or_else(|| panic!("{id} missing"));
        assert_eq!(item.kind, kind);
        assert_eq!(item.status, ItemStatus::Admitted, "{id} was not admitted on its receipt");
    }
}

#[tokio::test]
async fn a_failed_child_fails_its_step_then_blocks_it_and_the_run_is_reported_unfinished() {
    let world = World::new();
    let deps = &world.deps;
    call(deps, "task.plan_update", json!({
        "base_version": 0,
        "operations": [
            {"op": "set_goal", "text": "Write and test a unit converter"},
            {"op": "add_step", "id": "code", "title": "Write the converter", "kind": "delegate",
             "role": "code-worker", "max_attempts": 3},
        ]
    }))
    .await
    .expect("plan");

    // The production code worker cannot run without a model runtime, and says
    // so: a real failure, not a stubbed one. It writes, so a person approves.
    let args = json!({
        "step": "code", "role": "code-worker", "objective": "Write a psi to bar converter",
        "files": ["a.txt"], "wait_seconds": 30,
    });
    let first = decided(deps, "agent.delegate", args.clone(), true).await.expect("dispatched");
    let job = world.settled(&job_id_in(&first)).await;
    assert_eq!(job.status, JobStatus::Failed, "{:?}", job.result);
    assert_eq!(job.mode, "writer");
    assert_eq!(world.step("code").status, StepStatus::Failed);

    // Retried once, it fails the same way: no progress, so the step blocks
    // with attempts still left, and the model cannot reopen it again.
    call(deps, "task.plan_update", json!({
        "base_version": world.plan().version,
        "operations": [{"op": "reopen_step", "id": "code", "reason": "try once more"}]
    }))
    .await
    .expect("reopened");
    let second = decided(deps, "agent.delegate", args, true).await.expect("dispatched");
    world.settled(&job_id_in(&second)).await;
    let step = world.step("code");
    assert_eq!(step.status, StepStatus::Blocked);
    assert!(step.note.as_deref().unwrap_or_default().contains("no progress"), "{:?}", step.note);
    assert_eq!(step.repair.attempts, 2);
    assert_eq!(step.receipts.len(), 2);
    let refused = call(deps, "task.plan_update", json!({
        "base_version": world.plan().version,
        "operations": [{"op": "reopen_step", "id": "code", "reason": "again"}]
    }))
    .await
    .expect_err("a third identical attempt is refused");
    assert!(refused.contains("same way"), "{refused}");

    // The honest unfinished state, from the backend.
    let verdict = completion::verify(
        &completion::CompletionInputs {
            has_answer: true,
            grounding_ready: Some(true),
            task_plan: Some(task_plan::gate(&world.plan())),
            ..Default::default()
        },
        chrono::Utc::now(),
    );
    assert!(!verdict.passed());
    let unfinished = verdict
        .criteria
        .iter()
        .find(|c| c.criterion_id == "taskplan.steps_accepted")
        .expect("the plan criterion");
    assert!(unfinished.evidence.contains("code (blocked"), "{}", unfinished.evidence);
}

#[tokio::test]
async fn writer_jobs_need_a_person_and_never_come_through_the_read_only_alias() {
    let world = World::new();
    let deps = &world.deps;
    call(deps, "task.plan_update", json!({
        "base_version": 0,
        "operations": [
            {"op": "set_goal", "text": "Fix the converter"},
            {"op": "add_step", "id": "code", "title": "Fix it", "kind": "delegate", "role": "code-worker"},
            {"op": "add_step", "id": "find", "title": "Find", "kind": "delegate", "role": "knowledge-retriever"},
        ]
    }))
    .await
    .expect("plan");

    // 1. The read-only alias refuses a role that writes, whatever it is asked.
    let alias = call(deps, "agent.delegate_readonly", json!({
        "profile": "code-worker", "task": "Fix the converter", "files": ["a.txt"],
    }))
    .await
    .expect_err("the alias does not start a writer");
    assert!(alias.contains("read-only delegation cannot start it"), "{alias}");

    // 2. agent.delegate for a writer is put to a person; a no starts nothing.
    let args = json!({
        "step": "code", "role": "code-worker", "objective": "Fix the converter", "files": ["a.txt"],
    });
    let refused = decided(deps, "agent.delegate", args.clone(), false).await.expect_err("rejected");
    assert!(refused.starts_with("refused"), "{refused}");
    assert!(world.jobs().is_empty(), "a rejected writer job was recorded");
    assert_eq!(world.step("code").status, StepStatus::Pending);

    // 3. The handler refuses a writer call that carries no recorded approval,
    //    even if something handed it straight to the dispatcher.
    let bypass = CallParams {
        run_id: RUN.into(),
        tool_call_id: "tc-bypass".into(),
        tool: "agent.delegate".into(),
        args: args.clone(),
        model: None,
    };
    let direct = super::delegate(deps, &bypass, &session(), &ToolCall::new("agent.delegate".to_string(), args))
        .await
        .expect_err("no approval, no writer");
    assert!(direct.contains("carries no person's approval"), "{direct}");
    assert!(world.jobs().is_empty());

    // 4. A read-only role asking for a tool it is not given is refused before
    //    anything runs; it cannot be widened through `allowed_tools`.
    let widened = call(deps, "agent.delegate", json!({
        "step": "find", "role": "knowledge-retriever", "objective": "Find the tag",
        "allowed_tools": ["workspace.write_text"],
    }))
    .await
    .expect_err("no widening");
    assert!(widened.contains("is not given workspace.write_text"), "{widened}");
    assert!(world.jobs().is_empty());
}

#[tokio::test]
async fn a_correction_mid_flight_stops_the_job_and_reopens_its_step_uncharged() {
    let world = World::with(Options { stalling: vec!["document-extractor"], ..Default::default() });
    let deps = &world.deps;
    call(deps, "task.plan_update", json!({
        "base_version": 0,
        "operations": [
            {"op": "set_goal", "text": "Summarise the inspection file"},
            {"op": "add_step", "id": "read", "title": "Read the file", "kind": "delegate",
             "role": "document-extractor"},
        ]
    }))
    .await
    .expect("plan");
    let started = call(deps, "agent.delegate", json!({
        "step": "read", "role": "document-extractor", "objective": "Read the inspection file",
        "files": ["a.txt"], "wait_seconds": 0,
    }))
    .await
    .expect("dispatched");
    let job_id = job_id_in(&started);
    assert_eq!(world.step("read").status, StepStatus::Running);

    // A person steers the run.
    let stopped = super::record_correction(deps, RUN, "Use the 2019 revision of the file", "priya");
    assert_eq!(stopped, vec![job_id.clone()]);
    let job = world.settled(&job_id).await;
    assert_eq!(job.status, JobStatus::Cancelled);

    let plan = world.plan();
    assert_eq!(plan.corrections.len(), 1);
    let step = plan.step("read").unwrap();
    assert_eq!(step.status, StepStatus::Pending, "{:?}", step.note);
    assert_eq!(step.repair.attempts, 0, "a superseded attempt was charged");
    assert!(task_plan::project(&plan).contains("Use the 2019 revision"));

    // The correction is in the task's shared memory, as a person's, admitted.
    let correction = world
        .graph
        .snapshot(&session(), &MemoryScope::Task { task_id: RUN.into() }, None)
        .expect("readable")
        .into_iter()
        .find(|item| item.kind == MemoryKind::Correction)
        .expect("the correction was published");
    assert_eq!(correction.status, ItemStatus::Admitted);
}

#[tokio::test]
async fn a_duplicate_delegation_starts_no_second_job() {
    let world = World::with(Options { stalling: vec!["document-extractor"], ..Default::default() });
    let deps = &world.deps;
    two_specialist_plan(deps).await;
    call(deps, "task.plan_update", json!({
        "base_version": 1,
        "operations": [{"op": "add_step", "id": "read", "title": "Read", "kind": "delegate",
                        "role": "document-extractor"}]
    }))
    .await
    .expect("step added");

    let args = json!({
        "step": "read", "role": "document-extractor", "objective": "Read the file",
        "files": ["a.txt"], "wait_seconds": 0,
    });
    let first = call(deps, "agent.delegate", args.clone()).await.expect("dispatched");
    let again = call(deps, "agent.delegate", args).await.expect("answered from the running job");
    assert!(again.contains("no second job was started"), "{again}");
    assert_eq!(job_id_in(&again), job_id_in(&first));
    assert_eq!(world.jobs().len(), 1);

    // And after completion, the same request is answered from the receipt.
    let found = call(deps, "agent.delegate", json!({
        "step": "find", "role": "knowledge-retriever", "objective": "seal torque at ambient",
        "wait_seconds": 30,
    }))
    .await
    .expect("dispatched");
    world.settled(&job_id_in(&found)).await;
    let repeat = call(deps, "agent.delegate", json!({
        "step": "find", "role": "knowledge-retriever", "objective": "seal torque at ambient",
    }))
    .await
    .expect("answered from the receipt");
    assert!(repeat.contains("already completed"), "{repeat}");
    assert_eq!(world.jobs().len(), 2);
    assert_eq!(world.events_of(TaskEventType::SubagentStarted).len(), 2);
    super::stop_run(deps, RUN);
}

#[tokio::test]
async fn agent_cancel_stops_a_running_job_and_the_step_can_be_reopened() {
    let world = World::with(Options { stalling: vec!["document-extractor"], ..Default::default() });
    let deps = &world.deps;
    call(deps, "task.plan_update", json!({
        "base_version": 0,
        "operations": [
            {"op": "set_goal", "text": "Read the file"},
            {"op": "add_step", "id": "read", "title": "Read", "kind": "delegate", "role": "document-extractor"},
        ]
    }))
    .await
    .expect("plan");
    let started = call(deps, "agent.delegate", json!({
        "step": "read", "role": "document-extractor", "objective": "Read the file",
        "files": ["a.txt"], "wait_seconds": 0,
    }))
    .await
    .expect("dispatched");
    let job_id = job_id_in(&started);

    let stopping = call(deps, "agent.cancel", json!({"job": job_id, "reason": "not needed"})).await.expect("cancel");
    assert!(stopping.contains("told to stop"), "{stopping}");
    let status = call(deps, "agent.status", json!({"job": job_id, "wait_seconds": 10})).await.expect("status");
    assert!(status.contains("cancelled"), "{status}");
    assert_eq!(world.settled(&job_id).await.status, JobStatus::Cancelled);
    assert_eq!(world.step("read").status, StepStatus::Cancelled);
    let stopped = world.events_of(TaskEventType::SubagentStopped);
    assert_eq!(stopped.last().unwrap().payload["status"], "cancelled");

    // A job from another task is not visible, and not stoppable, here.
    let foreign = call(deps, "agent.cancel", json!({"job": "job-not-this-run"})).await.expect_err("unknown");
    assert!(foreign.contains("no job"), "{foreign}");

    call(deps, "task.plan_update", json!({
        "base_version": world.plan().version,
        "operations": [{"op": "reopen_step", "id": "read", "reason": "needed after all"}]
    }))
    .await
    .expect("a cancelled step can be reopened");
    assert_eq!(world.step("read").status, StepStatus::Pending);
}

#[tokio::test]
async fn a_restart_marks_the_running_job_interrupted_and_the_step_is_retried_fresh() {
    let durable = tempfile::tempdir().expect("a directory for the durable log");

    // The first process: a retriever job is in flight when it goes away.
    let before = World::with(Options {
        stalling: vec!["knowledge-retriever"],
        events_at: Some(durable.path().to_path_buf()),
        ..Default::default()
    });
    two_specialist_plan(&before.deps).await;
    let started = call(&before.deps, "agent.delegate", json!({
        "step": "find", "role": "knowledge-retriever", "objective": "seal torque at ambient",
        "wait_seconds": 0,
    }))
    .await
    .expect("dispatched");
    let lost = job_id_in(&started);
    assert_eq!(before.step("find").status, StepStatus::Running);

    // The next process opens the same file and recovers before anything runs.
    let reopened = TaskEventLog::open(durable.path()).expect("the log reopens");
    let interrupted = super::recover_interrupted_jobs(&reopened);
    assert_eq!(interrupted, vec![lost.clone()]);
    let job = reopened.job(&lost).unwrap().unwrap();
    assert_eq!(job.status, JobStatus::Interrupted);
    let plan = reopened.latest_plan(RUN).unwrap().unwrap();
    let step = plan.step("find").unwrap();
    assert_eq!(step.status, StepStatus::Failed, "a reader's interrupted step is retryable");
    assert!(step.receipts.iter().any(|receipt| receipt.status == "interrupted"));
    // The old process's child is gone with it.
    super::stop_run(&before.deps, RUN);
    drop(before);

    // The resumed run: production workers, the same durable record.
    let after = World::with(Options { events_at: Some(durable.path().to_path_buf()), ..Default::default() });
    call(&after.deps, "task.plan_update", json!({
        "base_version": after.plan().version,
        "operations": [{"op": "reopen_step", "id": "find", "reason": "the process restarted"}]
    }))
    .await
    .expect("reopened");
    let retried = call(&after.deps, "agent.delegate", json!({
        "step": "find", "role": "knowledge-retriever", "objective": "seal torque at ambient",
        "wait_seconds": 30,
    }))
    .await
    .expect("dispatched again");
    let job = after.settled(&job_id_in(&retried)).await;
    assert_eq!(job.status, JobStatus::Completed, "{:?}", job.result);
    assert_eq!(job.attempt, 2, "the retry is a new attempt, keyed apart from the lost one");
    assert_eq!(after.step("find").status, StepStatus::Completed);
}

#[tokio::test]
async fn a_role_with_no_worker_is_reported_blocked_and_never_as_done() {
    let world = World::with(Options { missing: vec!["calculation-checker"], ..Default::default() });
    let deps = &world.deps;

    // Capability discovery says so before anything is tried.
    let capabilities = call(deps, "capability.search", json!({"query": "calculation"})).await.expect("search");
    assert!(capabilities.contains("calculation-checker"), "{capabilities}");
    assert!(capabilities.contains("no worker in this build"), "{capabilities}");

    call(deps, "task.plan_update", json!({
        "base_version": 0,
        "operations": [
            {"op": "set_goal", "text": "Check the figure"},
            {"op": "add_step", "id": "check", "title": "Check", "kind": "delegate",
             "role": "calculation-checker"},
        ]
    }))
    .await
    .expect("plan");
    let dispatched = call(deps, "agent.delegate", json!({
        "step": "check", "role": "calculation-checker", "objective": "Re-derive 47.5 * 2",
        "expressions": ["47.5 * 2"], "wait_seconds": 10,
    }))
    .await
    .expect("recorded");
    let job = world.settled(&job_id_in(&dispatched)).await;
    assert_eq!(job.status, JobStatus::Blocked);
    let step = world.step("check");
    assert_eq!(step.status, StepStatus::Blocked);
    assert!(step.note.as_deref().unwrap_or_default().contains("a worker for the calculation-checker role"));
    assert!(world.events_of(TaskEventType::SubagentStarted).is_empty(), "nothing ran");
    let gate = task_plan::gate(&world.plan());
    assert_eq!(gate.completed, 0);
}

#[tokio::test]
async fn a_review_of_published_findings_rechecks_them_before_the_step_completes() {
    let world = World::new();
    let deps = &world.deps;
    call(deps, "task.plan_update", json!({
        "base_version": 0,
        "operations": [
            {"op": "set_goal", "text": "Find the tag and have it checked"},
            {"op": "add_step", "id": "find", "title": "Find", "kind": "delegate",
             "role": "knowledge-retriever", "acceptance": ["childCompleted", "reviewPassed"]},
            {"op": "add_step", "id": "review", "title": "Review", "kind": "review", "depends_on": ["find"]},
        ]
    }))
    .await
    .expect("plan");

    // Not reviewable until the work exists.
    let early = call(deps, "task.request_review", json!({"step": "find"})).await.expect_err("nothing yet");
    assert!(early.contains("pending"), "{early}");

    let found = call(deps, "agent.delegate", json!({
        "step": "find", "role": "knowledge-retriever", "objective": "seal torque at ambient",
        "wait_seconds": 30,
    }))
    .await
    .expect("dispatched");
    world.settled(&job_id_in(&found)).await;
    // Done by its own account; held for review by the plan.
    assert_eq!(world.step("find").status, StepStatus::AwaitingReview);

    let review = call(deps, "task.request_review", json!({"step": "find"})).await.expect("reviewed");
    assert!(review.contains("backend:"), "{review}");
    assert!(review.contains("passed"), "{review}");
    assert_eq!(world.step("find").status, StepStatus::Completed);
    let gate = task_plan::gate(&world.plan());
    assert!(gate.needs_review);
}

#[tokio::test]
async fn a_simple_question_is_not_offered_the_orchestrator() {
    let plan = crate::agent_runtime::planning::derive("What is the design pressure of vessel V-101?");
    for tool in [ToolName::TaskPlanUpdate, ToolName::AgentDelegate, ToolName::TaskRequestReview] {
        assert!(!plan.budget.permits(tool), "{} offered to a one-search question", tool.as_str());
    }
    let orchestrated = crate::agent_runtime::planning::derive(
        "Write an approval note on the seal failure; delegate the retrieval to a specialist",
    );
    for tool in [
        ToolName::TaskPlanUpdate,
        ToolName::AgentDelegate,
        ToolName::AgentStatus,
        ToolName::AgentCancel,
        ToolName::TaskRequestReview,
    ] {
        assert!(orchestrated.budget.permits(tool), "{} missing", tool.as_str());
    }
}

#[tokio::test]
async fn a_writer_job_is_escalated_to_a_person_by_the_gateway_path() {
    let world = World::new();
    let deps = &world.deps;
    let writer = CallParams {
        run_id: RUN.into(),
        tool_call_id: "tc-w".into(),
        tool: "agent.delegate".into(),
        args: json!({"step": "code", "role": "code-worker", "objective": "Fix it"}),
        model: None,
    };
    let summary = super::approval_needed(deps, &writer).expect("a writer is escalated");
    assert!(summary.contains("can write"), "{summary}");
    let reader = CallParams {
        args: json!({"step": "find", "role": "knowledge-retriever", "objective": "Find it"}),
        ..writer
    };
    assert!(super::approval_needed(deps, &reader).is_none(), "a reader needs nobody");
}

#[tokio::test]
async fn a_coordinators_own_deliverable_is_not_complete_until_an_independent_review_passes() {
    let world = World::new();
    let deps = &world.deps;
    deps.run_to_conversation.bind(RUN, "c-p05");
    call(deps, "task.plan_update", json!({
        "base_version": 0,
        "operations": [
            {"op": "set_goal", "text": "An approval note on shell thickness at point C"},
            {"op": "add_step", "id": "note", "title": "Write the note", "kind": "direct",
             "acceptance": ["toolSucceeded:artifact.create_approval_note", "reviewPassed"]},
        ]
    }))
    .await
    .expect("plan");

    // The coordinator writes the note itself, through the real producer.
    call(deps, "artifact.create_approval_note", json!({
        "path": "note.docx",
        "title": "Shell thickness at point C",
        "sections": [
            {"heading": "Findings", "level": 1, "blocks": [
                {"kind": "paragraph", "text": "Point C measured 8.2 mm against the governing minimum."}
            ]},
            {"heading": "Recommendation", "level": 1, "blocks": [
                {"kind": "paragraph", "text": "Assess point C before the unit returns to service."}
            ]}
        ]
    }))
    .await
    .expect("the note is produced");
    let produced = world
        .events_of(TaskEventType::ToolSucceeded)
        .into_iter()
        .find(|event| event.payload["tool"] == "artifact.create_approval_note")
        .expect("the producer's receipt");
    let record = deps
        .conversation_artifacts
        .list("priya", "c-p05")
        .expect("lists")
        .into_iter()
        .next()
        .expect("the note was registered");

    // A made-up receipt does not settle the step; the real one does, and the
    // step then waits for review rather than counting as done.
    let forged = call(deps, "task.plan_update", json!({
        "base_version": 1,
        "operations": [{"op": "claim_receipt", "id": "note", "event_seq": produced.seq + 1000}]
    }))
    .await
    .expect_err("no such receipt");
    assert!(forged.contains("not a successful tool call"), "{forged}");
    call(deps, "task.plan_update", json!({
        "base_version": 1,
        "operations": [{"op": "claim_receipt", "id": "note", "event_seq": produced.seq}]
    }))
    .await
    .expect("claimed");
    assert_eq!(world.step("note").status, StepStatus::AwaitingReview);

    // Review: the P04 ladder runs on the exact version, then the independent
    // reviewer. In this build the reviewer role re-opens workspace files and
    // cannot read a conversation artifact (the Deliverable Reviewer is P10),
    // so the review does not pass -- and the step is not complete.
    let review = call(deps, "task.request_review", json!({
        "step": "note", "artifacts": [record.reference().to_string()],
    }))
    .await
    .expect("the review ran");
    assert!(review.contains(&format!("backend: {}", record.reference())), "{review}");
    assert!(review.contains("did not pass"), "{review}");
    let step = world.step("note");
    assert_eq!(step.status, StepStatus::Partial, "{:?}", step.note);
    assert!(step.receipts.iter().any(|r| r.kind == task_plan::ReceiptKind::Review && r.status == "failed"));

    let verdict = completion::verify(
        &completion::CompletionInputs {
            has_answer: true,
            grounding_ready: Some(true),
            task_plan: Some(task_plan::gate(&world.plan())),
            ..Default::default()
        },
        chrono::Utc::now(),
    );
    assert!(!verdict.passed(), "the note was called done without a passed review");
    assert!(verdict.criteria.iter().any(|c| c.criterion_id == "taskplan.independent_review"
        && c.status == completion::CriterionStatus::Failed));
}

#[tokio::test]
async fn stopping_the_run_stops_its_running_jobs() {
    let world = World::with(Options { stalling: vec!["document-extractor"], ..Default::default() });
    let deps = &world.deps;
    call(deps, "task.plan_update", json!({
        "base_version": 0,
        "operations": [
            {"op": "set_goal", "text": "Read the file"},
            {"op": "add_step", "id": "read", "title": "Read", "kind": "delegate", "role": "document-extractor"},
        ]
    }))
    .await
    .expect("plan");
    let started = call(deps, "agent.delegate", json!({
        "step": "read", "role": "document-extractor", "objective": "Read the file",
        "files": ["a.txt"], "wait_seconds": 0,
    }))
    .await
    .expect("dispatched");
    let job_id = job_id_in(&started);

    // What `agent_abort_run` does to the table when a person presses Stop.
    world.cancellations.register(&[RUN]);
    assert!(world.cancellations.cancel(RUN));

    let job = world.settled(&job_id).await;
    assert_eq!(job.status, JobStatus::Cancelled);
    assert!(
        job.result.as_ref().and_then(|r| r["summary"].as_str()).unwrap_or_default().contains("stopped"),
        "{:?}",
        job.result
    );
    assert_eq!(world.step("read").status, StepStatus::Cancelled);
}

#[tokio::test]
async fn a_review_step_settles_the_steps_it_reviewed() {
    let world = World::new();
    let deps = &world.deps;
    call(deps, "task.plan_update", json!({
        "base_version": 0,
        "operations": [
            {"op": "set_goal", "text": "Find the torque and have it reviewed"},
            {"op": "add_step", "id": "find", "title": "Find", "kind": "delegate",
             "role": "knowledge-retriever", "acceptance": ["childCompleted", "reviewPassed"]},
            {"op": "add_step", "id": "review", "title": "Review", "kind": "review", "depends_on": ["find"]},
        ]
    }))
    .await
    .expect("plan");
    let found = call(deps, "agent.delegate", json!({
        "step": "find", "role": "knowledge-retriever", "objective": "seal torque at ambient",
        "wait_seconds": 30,
    }))
    .await
    .expect("dispatched");
    world.settled(&job_id_in(&found)).await;
    assert_eq!(world.step("find").status, StepStatus::AwaitingReview);

    let before = world.plan().version;
    let review = call(deps, "task.request_review", json!({"step": "review"})).await.expect("reviewed");
    assert!(review.contains("passed"), "{review}");
    let plan = world.plan();
    assert_eq!(plan.version, before + 1, "one plan version for the whole verdict");
    assert_eq!(plan.step("review").unwrap().status, StepStatus::Completed);
    assert_eq!(plan.step("find").unwrap().status, StepStatus::Completed, "the reviewed step was left waiting");
    let gate = task_plan::gate(&plan);
    assert!(gate.reviewed && gate.unfinished.is_empty(), "{gate:?}");
}

#[tokio::test]
async fn an_artifact_from_another_conversation_is_not_reviewed_or_read() {
    let world = World::new();
    let deps = &world.deps;
    // The same person's note, produced in a different conversation.
    deps.run_to_conversation.bind(RUN, "c-other");
    call(deps, "artifact.create_approval_note", json!({
        "path": "note.docx",
        "title": "Elsewhere",
        "sections": [{"heading": "Findings", "level": 1, "blocks": [
            {"kind": "paragraph", "text": "Produced in another conversation."}
        ]}]
    }))
    .await
    .expect("produced");
    let elsewhere = deps.conversation_artifacts.list("priya", "c-other").unwrap().remove(0);

    // This run now belongs to a different conversation.
    deps.run_to_conversation.bind(RUN, "c-this");
    call(deps, "task.plan_update", json!({
        "base_version": 0,
        "operations": [
            {"op": "set_goal", "text": "Review a note"},
            {"op": "add_step", "id": "note", "title": "Note", "kind": "direct",
             "acceptance": ["toolSucceeded:artifact.create_approval_note", "reviewPassed"]},
            {"op": "add_step", "id": "check", "title": "Check it", "kind": "delegate",
             "role": "artifact-reviewer", "inputs": [elsewhere.reference().to_string()]},
        ]
    }))
    .await
    .expect("plan");
    let produced = world
        .events_of(TaskEventType::ToolSucceeded)
        .into_iter()
        .find(|event| event.payload["tool"] == "artifact.create_approval_note")
        .unwrap();
    call(deps, "task.plan_update", json!({
        "base_version": 1,
        "operations": [{"op": "claim_receipt", "id": "note", "event_seq": produced.seq}]
    }))
    .await
    .expect("claimed");

    let review = call(deps, "task.request_review", json!({
        "step": "note", "artifacts": [elsewhere.reference().to_string()],
    }))
    .await
    .expect_err("not this conversation's");
    assert!(review.contains("does not exist or is not readable here"), "{review}");
    let delegated = call(deps, "agent.delegate", json!({
        "step": "check", "role": "artifact-reviewer", "objective": "Review the note",
    }))
    .await
    .expect_err("not this conversation's");
    assert!(delegated.contains("does not exist in this conversation"), "{delegated}");
    assert!(world.jobs().iter().all(|job| job.step_id != "check" || job.status != JobStatus::Completed));
}
