//! Moving an agent between models, driven through the production handoff.
//!
//! ## What these drive, and what they deliberately do not
//!
//! They drive the real [`Handoff`]: a real [`AgentRegistry`] on a temporary
//! directory, a real [`TaskEventLog`] on real SQLite with its real migrations
//! and its partial unique index, a real [`ModelRegistry`], a real
//! [`MemoryGraph`], and the real `AgentRegistry::rebind_model` with its version
//! guard. Nothing here is a mock of the thing under test.
//!
//! They do **not** start a model. `VerifyDestination::LoadAndVerify` ends in
//! `serving::admission::admit` and `ModelServers::endpoint_for`, which need
//! `llama-server` and several gigabytes of weights; a test that claimed to prove
//! a successful handoff of live work without them would be claiming something it
//! had not done. That half is `tests/model_binding_handoff_live.rs`, which uses
//! two models actually installed on the machine and skips loudly when they are
//! not there.
//!
//! What *is* proved here is everything either side of the load, and it is where
//! the interesting mistakes live: the drain boundary, the checkpoint that has to
//! exist before anything is unloaded, the refusals, the version guards under
//! concurrent administration, the bounded rollback, and the three-valued reading
//! of a crash.
//!
//! ## Why the load failure is a real load failure
//!
//! The target in the active-run cases points at weights that are not on disk, so
//! `endpoint_for` refuses deterministically on every machine — with or without
//! `llama-server` installed. That makes "the destination could not be served"
//! reproducible rather than environment-dependent, and it exercises exactly the
//! path that matters: the checkpoint was taken first, so the undo has something
//! to protect.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Mutex;

use serde_json::json;

use super::events::{EventDraft, TaskEventLog, TaskEventType};
use super::memory::{CompletedEffect, RunMemory};
use super::model_handoff::{Handoff, HandoffRequest, VerifyDestination};
use super::model_transition::{
    ModelFingerprint, TransitionOutcome, TransitionPhase, TransitionRecord,
};
use super::resume::CheckpointSeed;
use crate::agents::store::AgentRegistry;
use crate::agents::{AgentDefinition, AgentState, MemoryPolicy, AGENT_PALETTE};
use crate::identity::{Role, Session, User};
use crate::knowledge::graph::runtime_store::MemoryGraph;
use crate::orchestrator::tools::ToolName;
use crate::policy::Classification;
use crate::registry::{ModelEntry, ModelManifest, ModelRegistry, ModelRole};
use crate::serving::ModelServers;
use crate::subagents::profile::{Isolation, Limits, SchemaKind, WritePolicy};

const RUN: &str = "run-1";
const ATTEMPT: &str = "attempt-1";

// ─────────────────────────────────────────────────────────────────────────────
// The world
// ─────────────────────────────────────────────────────────────────────────────

fn administrator() -> Session {
    Session::open(User::new(
        "priya",
        "Priya Sharma",
        vec![Role::Employee, Role::Administrator],
    ))
}

fn employee() -> Session {
    Session::open(User::new("ravi", "Ravi Kumar", vec![Role::Employee]))
}

/// A registered model. `weights_exist` is the difference between a target that
/// can be admitted and one whose load fails the same way on every machine.
fn entry(id: &str, weights_exist: bool) -> ModelEntry {
    let mut entry = crate::registry::tests::entry(id, 4.0, vec![ModelRole::Reasoning]);
    entry.supports_structured_output = true;
    entry.context_length = 8_192;
    entry.path = if weights_exist {
        PathBuf::from(format!("{id}.gguf"))
    } else {
        PathBuf::from(format!("{id}-not-on-disk.gguf"))
    };
    entry
}

fn model_registry(entries: Vec<ModelEntry>) -> ModelRegistry {
    ModelRegistry::from_manifest(
        ModelManifest { models: entries },
        PathBuf::from("models/registry.json"),
    )
    .expect("the manifest is well formed")
}

fn definition(model: &str) -> AgentDefinition {
    AgentDefinition {
        // Minted by the registry on create; whatever is here is discarded.
        agent_id: String::new(),
        definition_version: 0,
        display_name: "Knowledge retriever".into(),
        description: "Finds passages and cites them.".into(),
        instructions: "Answer only from the passages you retrieve.".into(),
        role: ModelRole::Reasoning,
        state: AgentState::Enabled,
        color: AGENT_PALETTE[0].into(),
        skills: Vec::new(),
        allowed_tools: vec![ToolName::SearchDocuments],
        denied_tools: Vec::new(),
        memory: MemoryPolicy::default(),
        models: crate::agents::ModelBinding {
            default_model_id: Some(model.into()),
            fallback_model_ids: Vec::new(),
            eligible_model_ids: Vec::new(),
        },
        output_schema: SchemaKind::Retrieval,
        limits: Limits {
            max_turns: 8,
            max_output_tokens: 2_048,
            max_children: 0,
            max_duration_seconds: 600,
        },
        max_concurrent: 1,
        isolation: Isolation::ReadOnly,
        write_policy: WritePolicy::None,
        classification_ceiling: Classification::Internal,
        imported_from: None,
        created_at: String::new(),
        updated_at: String::new(),
    }
}

/// Everything a handoff borrows, owned for the length of a test.
struct World {
    agents: AgentRegistry,
    events: TaskEventLog,
    models: ModelRegistry,
    servers: ModelServers,
    graph: MemoryGraph,
    checkpoints: Mutex<HashMap<String, CheckpointSeed>>,
    models_dir: PathBuf,
    agent_id: String,
    _dir: tempfile::TempDir,
}

impl World {
    /// An agent on `model-a`, with `model-b` also registered.
    fn new(target_weights_exist: bool) -> Self {
        Self::with_models(vec![
            entry("model-a", true),
            entry("model-b", target_weights_exist),
        ])
    }

    fn with_models(models: Vec<ModelEntry>) -> Self {
        let dir = tempfile::tempdir().expect("a temporary directory");
        let agents = AgentRegistry::open(dir.path()).expect("a fresh registry");
        let created = agents
            .create(&administrator(), definition("model-a"))
            .expect("an administrator may create an agent");
        let models_dir = dir.path().join("models");

        Self {
            agents,
            events: TaskEventLog::in_memory().expect("an in-memory event log"),
            models: model_registry(models),
            servers: ModelServers::new(),
            graph: MemoryGraph::in_memory().expect("an in-memory graph"),
            checkpoints: Mutex::new(HashMap::new()),
            models_dir,
            agent_id: created.agent_id,
            _dir: dir,
        }
    }

    fn handoff(&self) -> Handoff<'_> {
        Handoff {
            agents: &self.agents,
            events: &self.events,
            registry: &self.models,
            servers: &self.servers,
            graph: Some(&self.graph),
            checkpoints: &self.checkpoints,
            models_dir: &self.models_dir,
            leases: None,
        }
    }

    fn agent(&self) -> AgentDefinition {
        self.agents
            .resolve_for_provenance(&self.agent_id)
            .expect("the agent is resolvable")
    }

    fn version(&self) -> u64 {
        self.agent().definition_version
    }

    fn current_model(&self) -> Option<String> {
        self.agent().models.default_model_id
    }

    fn request(&self, target: &str, verify: VerifyDestination) -> HandoffRequest {
        HandoffRequest {
            agent_id: self.agent_id.clone(),
            target_model_id: target.into(),
            expected_definition_version: self.version(),
            reason: "moving to the model with the larger window".into(),
            run_id: None,
            task_id: None,
            project_id: None,
            verify,
            reserves: None,
        }
    }

    /// A run that has made a tool call and recorded the result — the boundary a
    /// handoff is meant to be able to proceed from.
    fn run_after_a_tool(&self) -> RunMemory {
        let notes = RunMemory {
            goal: "Compare the two tenders and write the note.".into(),
            completed: vec![CompletedEffect {
                tool: "artifact.create_docx".into(),
                target: "note.docx".into(),
                at: "2026-09-18T00:00:00Z".into(),
            }],
            ..RunMemory::default()
        };
        for kind in [
            TaskEventType::RunCreated,
            TaskEventType::RunStarted,
            TaskEventType::ToolAuthorized,
            TaskEventType::ToolSucceeded,
        ] {
            self.events
                .record(EventDraft::new(RUN, kind, "priya").with(json!({})))
                .expect("the event is recorded");
        }
        self.checkpoints
            .lock()
            .expect("a fresh lock")
            .insert(RUN.to_string(), seed(notes.clone()));
        notes
    }
}

fn seed(notes: RunMemory) -> CheckpointSeed {
    use super::context_manifest::{ContextManifest, HistoryBinding};
    CheckpointSeed {
        attempt_id: ATTEMPT.to_string(),
        committed_notes: notes,
        manifest: Some(ContextManifest::new(
            RUN,
            ATTEMPT,
            "conv-1",
            "a-run-1",
            "model-a",
            8_192,
            Vec::new(),
            None,
            HistoryBinding {
                carried: 2,
                dropped: 0,
                tokens: 400,
                pinned: Vec::new(),
                omitted_pins: Vec::new(),
            },
        )),
        plan_hash: "plan-hash".to_string(),
        policy_hash: "policy-hash".to_string(),
        workspace_hash: "workspace-hash".to_string(),
        model_id: "model-a".to_string(),
    }
}

/// Runs one future on a runtime built for this test.
fn block_on<F: std::future::Future>(future: F) -> F::Output {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("a test runtime")
        .block_on(future)
}

// ─────────────────────────────────────────────────────────────────────────────
// Reassigning an agent with no work in flight
// ─────────────────────────────────────────────────────────────────────────────

/// The acceptance case. An idle agent moves, and everything that identifies it
/// stays exactly where it was.
#[test]
fn an_idle_agent_is_reassigned_and_keeps_its_identity_and_memory() {
    let world = World::new(true);
    let before = world.agent();

    let record = block_on(world.handoff().execute(
        &administrator(),
        world.request("model-b", VerifyDestination::PlanOnly),
    ))
    .expect("the handoff opens");

    assert_eq!(record.outcome(), TransitionOutcome::Succeeded);
    assert_eq!(record.phase, TransitionPhase::Committed);
    assert_eq!(world.current_model().as_deref(), Some("model-b"));

    let after = world.agent();
    // The whole point of the module, asserted rather than assumed.
    assert_eq!(after.agent_id, before.agent_id, "the identity is preserved");
    assert_eq!(after.memory, before.memory, "memory ownership is unchanged");
    assert_eq!(after.skills, before.skills);
    assert_eq!(after.allowed_tools, before.allowed_tools);
    assert_eq!(after.created_at, before.created_at);
    assert_eq!(
        after.definition_version,
        before.definition_version + 1,
        "an accepted change moves the version exactly once"
    );
    assert_eq!(record.to_definition_version, after.definition_version);

    // Nothing was loaded, so nothing pretends to have measured a window.
    assert!(record.served.is_none());
    assert!(record.target_manifest_hash.is_none());
    assert!(record.freeze.is_some(), "the graph revision is still frozen");
}

/// Both digests are on the record, so somebody can join it to the exact bytes
/// even after a re-import moves an id.
#[test]
fn a_committed_handoff_records_both_model_identities() {
    let world = World::new(true);
    let record = block_on(world.handoff().execute(
        &administrator(),
        world.request("model-b", VerifyDestination::PlanOnly),
    ))
    .expect("the handoff opens");

    assert_eq!(record.from_model.model_id, "model-a");
    assert_eq!(record.to_model.model_id, "model-b");
    assert_ne!(record.from_model.digest, record.to_model.digest);
    assert_eq!(record.from_definition_version, 1);
    assert_eq!(record.to_definition_version, 2);
    assert!(record.freeze.as_ref().is_some_and(|f| f.graph_revision >= 0));
}

/// Moving there and back is two transitions, each with its own record, and the
/// agent ends exactly where it started.
#[test]
fn an_agent_can_be_moved_there_and_back() {
    let world = World::new(true);

    let out = block_on(world.handoff().execute(
        &administrator(),
        world.request("model-b", VerifyDestination::PlanOnly),
    ))
    .expect("out");
    assert_eq!(out.outcome(), TransitionOutcome::Succeeded);

    let back = block_on(world.handoff().execute(
        &administrator(),
        world.request("model-a", VerifyDestination::PlanOnly),
    ))
    .expect("back");
    assert_eq!(back.outcome(), TransitionOutcome::Succeeded);

    assert_eq!(world.current_model().as_deref(), Some("model-a"));
    assert_eq!(world.version(), 3, "two accepted changes, two versions");

    let history = world
        .events
        .transitions_for_agent(&world.agent_id, 10)
        .expect("the ledger reads");
    assert_eq!(history.len(), 2, "each move is its own record");
    assert!(history.iter().all(|record| !record.is_open()));
}

// ─────────────────────────────────────────────────────────────────────────────
// Refusals
// ─────────────────────────────────────────────────────────────────────────────

/// The headline rule: the edit form is not a way to change the model.
#[test]
fn an_edit_cannot_change_the_model_and_says_where_to_do_it() {
    let world = World::new(true);
    let mut edited = world.agent();
    edited.display_name = "Renamed".into();
    edited.models.default_model_id = Some("model-b".into());

    let refusal = world
        .agents
        .update(&administrator(), &world.agent_id, world.version(), edited)
        .expect_err("a form may not rebind a model");

    let sentence = refusal.explain();
    assert!(sentence.contains("model change action"), "{sentence}");
    assert_eq!(
        world.current_model().as_deref(),
        Some("model-a"),
        "nothing moved"
    );
    assert_eq!(world.version(), 1, "and no version was spent on it");
}

/// Every other field still saves. The refusal is about one field, not the form.
#[test]
fn an_edit_that_leaves_the_binding_alone_still_saves() {
    let world = World::new(true);
    let mut edited = world.agent();
    edited.display_name = "Renamed".into();

    world
        .agents
        .update(&administrator(), &world.agent_id, world.version(), edited)
        .expect("an ordinary edit is accepted");
    assert_eq!(world.agent().display_name, "Renamed");
    assert_eq!(world.current_model().as_deref(), Some("model-a"));
}

/// The acceptance case. An agent bound to a tool is not moved to a model that
/// cannot be asked to call one.
#[test]
fn a_target_that_cannot_call_this_agents_tools_is_refused_and_changes_nothing() {
    let mut toolless = entry("model-b", true);
    toolless.supports_structured_output = false;
    let world = World::with_models(vec![entry("model-a", true), toolless]);

    let record = block_on(world.handoff().execute(
        &administrator(),
        world.request("model-b", VerifyDestination::PlanOnly),
    ))
    .expect("the handoff opens, and then refuses");

    assert_eq!(record.outcome(), TransitionOutcome::Failed);
    assert_eq!(record.phase, TransitionPhase::Failed);
    assert!(record.refusal.is_some(), "the typed refusal is on the record");
    assert!(
        record
            .because
            .as_deref()
            .is_some_and(|line| line.contains("structured output")),
        "{:?}",
        record.because
    );
    assert_eq!(world.current_model().as_deref(), Some("model-a"));
    assert_eq!(world.version(), 1);
}

#[test]
fn a_model_that_is_not_registered_never_opens_a_handoff() {
    let world = World::new(true);
    let refusal = block_on(world.handoff().execute(
        &administrator(),
        world.request("model-z", VerifyDestination::PlanOnly),
    ))
    .expect_err("there is no such model");
    assert!(refusal.contains("model registry"), "{refusal}");
    assert!(
        world
            .events
            .transitions_for_agent(&world.agent_id, 10)
            .expect("the ledger reads")
            .is_empty(),
        "a handoff that never opened leaves no row"
    );
}

/// The refusal that matters is in Rust, on the mutation — and it is also asked
/// early, so an employee does not drain a run before being told no.
#[test]
fn somebody_who_is_not_an_administrator_cannot_move_an_agent() {
    let world = World::new(true);
    let refusal = block_on(
        world
            .handoff()
            .execute(&employee(), world.request("model-b", VerifyDestination::PlanOnly)),
    )
    .expect_err("an employee may not");
    assert!(refusal.contains("administrator"), "{refusal}");
    assert_eq!(world.current_model().as_deref(), Some("model-a"));
}

/// Handing over live work without loading the destination is not a combination
/// this accepts: the context has to be rebuilt for a window nobody has measured.
#[test]
fn live_work_cannot_be_handed_over_without_loading_the_destination() {
    let world = World::new(true);
    world.run_after_a_tool();

    let mut request = world.request("model-b", VerifyDestination::PlanOnly);
    request.run_id = Some(RUN.to_string());

    let refusal = block_on(world.handoff().execute(&administrator(), request))
        .expect_err("a run cannot be handed to an unmeasured window");
    assert!(refusal.contains("started and checked"), "{refusal}");
    assert_eq!(world.current_model().as_deref(), Some("model-a"));
}

// ─────────────────────────────────────────────────────────────────────────────
// Concurrent administration
// ─────────────────────────────────────────────────────────────────────────────

/// Two administrators with the same screen open. The second is refused rather
/// than silently undoing the first.
#[test]
fn a_concurrent_edit_refuses_the_handoff_before_it_opens() {
    let world = World::new(true);
    let stale = world.request("model-b", VerifyDestination::PlanOnly);

    // Somebody else renames the agent in the meantime.
    let mut edited = world.agent();
    edited.display_name = "Renamed by somebody else".into();
    world
        .agents
        .update(&administrator(), &world.agent_id, world.version(), edited)
        .expect("the other edit lands");

    let refusal = block_on(world.handoff().execute(&administrator(), stale))
        .expect_err("the version moved under it");
    assert!(refusal.contains("changed since"), "{refusal}");
    assert_eq!(world.current_model().as_deref(), Some("model-a"));
    assert!(
        world
            .events
            .transitions_for_agent(&world.agent_id, 10)
            .expect("the ledger reads")
            .is_empty(),
        "a refused handoff leaves no row to block the next one"
    );
}

/// Two handoffs of one agent would race over the same record, so the second is
/// refused while the first is open.
#[test]
fn one_agent_cannot_have_two_handoffs_under_way() {
    let world = World::new(true);

    // Park one by making the run need a person: an effect nobody settled.
    world.run_after_a_tool();
    world
        .events
        .begin_effect(RUN, "key-1", "artifact.create_xlsx", "fingerprint", "book.xlsx");

    let mut first = world.request("model-b", VerifyDestination::LoadAndVerify);
    first.run_id = Some(RUN.to_string());
    let parked = block_on(world.handoff().execute(&administrator(), first)).expect("opens");
    assert_eq!(parked.phase, TransitionPhase::AwaitingSettlement);
    assert_eq!(parked.outcome(), TransitionOutcome::Pending);

    // A second handoff, to a different model, is refused while that one is open.
    let second = world.request("model-a", VerifyDestination::PlanOnly);
    let refusal =
        block_on(world.handoff().execute(&administrator(), second)).expect_err("one at a time");
    assert!(refusal.contains("already being moved"), "{refusal}");
}

// ─────────────────────────────────────────────────────────────────────────────
// Live work: the drain boundary
// ─────────────────────────────────────────────────────────────────────────────

/// The acceptance case. An approval nobody has answered parks the handoff and
/// names it, rather than failing or proceeding.
#[test]
fn a_pending_approval_parks_the_handoff_and_says_what_to_settle() {
    let world = World::new(true);
    world.run_after_a_tool();
    world
        .events
        .record_approval(&crate::agent_runtime::events::DurableApproval::requested(
            "approval-1",
            RUN,
            "artifact.create_docx",
            "note.docx",
            &json!({"title": "Approval note"}),
            "it writes a document",
            chrono::Utc::now(),
            None,
        ))
        .expect("the approval is recorded");

    let mut request = world.request("model-b", VerifyDestination::LoadAndVerify);
    request.run_id = Some(RUN.to_string());
    let record = block_on(world.handoff().execute(&administrator(), request)).expect("opens");

    assert_eq!(record.phase, TransitionPhase::AwaitingSettlement);
    assert_eq!(record.outcome(), TransitionOutcome::Pending);
    assert_eq!(record.awaiting, vec!["approval-1".to_string()]);
    assert!(record.is_open(), "it is waiting, not finished");
    assert_eq!(world.current_model().as_deref(), Some("model-a"));
    // Nothing was written down, because nothing was about to be disturbed.
    assert!(record.freeze.is_none());
}

/// An effect nobody accounted for is the strongest reason to stop: a different
/// model cannot be told whether it happened.
#[test]
fn an_unaccounted_effect_parks_the_handoff() {
    let world = World::new(true);
    world.run_after_a_tool();
    world
        .events
        .begin_effect(RUN, "key-1", "artifact.create_xlsx", "fingerprint", "book.xlsx");

    let mut request = world.request("model-b", VerifyDestination::LoadAndVerify);
    request.run_id = Some(RUN.to_string());
    let record = block_on(world.handoff().execute(&administrator(), request)).expect("opens");

    assert_eq!(record.phase, TransitionPhase::AwaitingSettlement);
    assert_eq!(record.awaiting, vec!["key-1".to_string()]);
    assert_eq!(world.current_model().as_deref(), Some("model-a"));
}

/// A parked handoff is the same handoff when it is asked again, and once the
/// question is answered it proceeds past the drain.
#[test]
fn settling_the_question_lets_the_same_handoff_continue() {
    let world = World::new(false);
    world.run_after_a_tool();
    world
        .events
        .begin_effect(RUN, "key-1", "artifact.create_xlsx", "fingerprint", "book.xlsx");

    let mut request = world.request("model-b", VerifyDestination::LoadAndVerify);
    request.run_id = Some(RUN.to_string());
    let parked = block_on(world.handoff().execute(&administrator(), request.clone()))
        .expect("opens and parks");
    assert_eq!(parked.phase, TransitionPhase::AwaitingSettlement);

    // A person says what happened to it.
    world
        .events
        .settle_effect(RUN, "key-1", &Ok("wrote book.xlsx".to_string()));

    let resumed = block_on(world.handoff().execute(&administrator(), request)).expect("continues");
    assert_eq!(
        resumed.transition_id, parked.transition_id,
        "the same handoff, not a second one"
    );
    // It got past the drain: the state was written down before anything else.
    assert!(
        resumed.freeze.is_some(),
        "the run's state is saved before the model is touched"
    );
    assert!(
        resumed
            .history
            .iter()
            .any(|entry| entry.phase == TransitionPhase::Checkpointed),
        "the checkpoint phase is in the history"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Live work: the load fails, and the undo is bounded
// ─────────────────────────────────────────────────────────────────────────────

/// The acceptance case, and the one that proves the ordering. The run is drained
/// at a tool boundary, its state is written down, the destination fails to come
/// up, and the handoff rolls back with the resume point and the completed work
/// intact.
#[test]
fn a_destination_that_will_not_come_up_rolls_back_and_keeps_the_saved_state() {
    // `model-b`'s weights are not on disk, so the load refuses on every machine
    // — with or without llama-server installed.
    let world = World::new(false);
    let notes = world.run_after_a_tool();

    let mut request = world.request("model-b", VerifyDestination::LoadAndVerify);
    request.run_id = Some(RUN.to_string());
    let record = block_on(world.handoff().execute(&administrator(), request)).expect("opens");

    assert_eq!(record.outcome(), TransitionOutcome::RolledBack);
    assert_eq!(record.phase, TransitionPhase::RolledBack);
    assert_eq!(
        world.current_model().as_deref(),
        Some("model-a"),
        "the agent is on the model it was on"
    );
    assert_eq!(world.version(), 1, "no binding was ever written");

    // The checkpoint was taken before the load was attempted, and the rollback
    // did not touch it.
    let freeze = record.freeze.as_ref().expect("the state was written down");
    let saved = world
        .events
        .checkpoint(RUN)
        .expect("the checkpoint reads")
        .expect("there is one");
    assert_eq!(
        Some(&saved.checkpoint_hash),
        freeze.checkpoint_hash.as_ref(),
        "the resume point is the one this handoff established"
    );
    assert_eq!(saved.model_id, "model-a");
    assert!(saved.is_intact());

    // And the work already done is still recorded, so a resumption will not do
    // it again.
    assert!(
        saved.notes.has_done("artifact.create_docx", "note.docx"),
        "a completed effect must survive the undo"
    );
    assert_eq!(saved.notes.completed.len(), notes.completed.len());
}

/// The run's own history says where it stopped being on the old model, and that
/// the handoff was undone.
#[test]
fn a_rolled_back_handoff_is_in_the_runs_own_history() {
    let world = World::new(false);
    world.run_after_a_tool();

    let mut request = world.request("model-b", VerifyDestination::LoadAndVerify);
    request.run_id = Some(RUN.to_string());
    block_on(world.handoff().execute(&administrator(), request)).expect("opens");

    let page = world.events.events_since(RUN, 0).expect("the events read");
    let kinds: Vec<&str> = page
        .events
        .iter()
        .map(|event| event.event_type.as_str())
        .collect();
    assert!(
        kinds.contains(&"model_transition_started"),
        "the run records where it stopped: {kinds:?}"
    );
    assert!(
        kinds.contains(&"model_transition_rolled_back"),
        "and that the change was undone: {kinds:?}"
    );
    assert!(
        !kinds.contains(&"model_transition_committed"),
        "nothing was committed: {kinds:?}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Crashes
// ─────────────────────────────────────────────────────────────────────────────

/// A handoff left at `commit_pending` with the registry still on the source: the
/// write did not land, and that is read off the record rather than guessed.
#[test]
fn a_crash_before_the_binding_commit_is_settled_as_not_having_happened() {
    let world = World::new(true);
    let record = interrupted_at_commit(&world);
    world
        .events
        .save_transition(&record)
        .expect("the row is written");

    let settled = block_on(world.handoff().reconcile_open("system", None));
    assert_eq!(settled.len(), 1);
    assert_eq!(settled[0].outcome(), TransitionOutcome::RolledBack);
    assert_eq!(world.current_model().as_deref(), Some("model-a"));
    assert_eq!(world.version(), 1);
    assert!(
        settled[0]
            .because
            .as_deref()
            .is_some_and(|line| line.contains("before")),
        "{:?}",
        settled[0].because
    );
}

/// The same row, with the registry already holding the target: the write did
/// land, and the handoff is finished rather than undone.
#[test]
fn a_crash_after_the_binding_commit_is_settled_as_having_happened() {
    let world = World::new(true);
    let record = interrupted_at_commit(&world);
    world
        .events
        .save_transition(&record)
        .expect("the row is written");

    // The write that landed just before the process went away.
    world
        .agents
        .rebind_model(
            &administrator(),
            &world.agent_id,
            record.from_definition_version,
            record.to_binding.clone(),
            &record.transition_id,
        )
        .expect("the binding is written");

    let settled = block_on(world.handoff().reconcile_open("system", None));
    assert_eq!(settled.len(), 1);
    assert_eq!(settled[0].outcome(), TransitionOutcome::Succeeded);
    assert_eq!(settled[0].phase, TransitionPhase::Committed);
    assert_eq!(world.current_model().as_deref(), Some("model-b"));
}

/// The third value, which must not be collapsed into either of the others.
/// Somebody edited the agent between the crash and the sweep, so what happened
/// to the write cannot be read off the record.
#[test]
fn a_crash_with_a_concurrent_edit_asks_for_a_person() {
    let world = World::new(true);
    let record = interrupted_at_commit(&world);
    world
        .events
        .save_transition(&record)
        .expect("the row is written");

    // Somebody edits the agent while the process is down.
    let mut edited = world.agent();
    edited.display_name = "Renamed while it was down".into();
    world
        .agents
        .update(&administrator(), &world.agent_id, world.version(), edited)
        .expect("the edit lands");

    let settled = block_on(world.handoff().reconcile_open("system", None));
    assert_eq!(settled.len(), 1);
    assert_eq!(settled[0].outcome(), TransitionOutcome::RollbackFailed);
    assert!(
        settled[0].outcome().needs_human(),
        "an unreadable binding is a question for a person"
    );
    assert!(
        settled[0]
            .because
            .as_deref()
            .is_some_and(|line| line.contains("edited")),
        "{:?}",
        settled[0].because
    );
}

/// A row left open before anything was touched is closed, so the agent is not
/// blocked from ever being moved again.
#[test]
fn a_crash_before_anything_happened_releases_the_agent() {
    let world = World::new(true);
    let agent = world.agent();
    let opened = TransitionRecord::begin(
        "tr-open",
        &agent,
        ModelFingerprint::unregistered("model-a"),
        ModelFingerprint::unregistered("model-b"),
        agent.models.rebound_to("model-b"),
        "priya",
        "interrupted at the very start",
    );
    world.events.save_transition(&opened).expect("written");

    let settled = block_on(world.handoff().reconcile_open("system", None));
    assert_eq!(settled[0].outcome(), TransitionOutcome::Failed);
    assert!(
        world
            .events
            .open_transition_for_agent(&world.agent_id)
            .expect("the ledger reads")
            .is_none(),
        "the agent is free to be moved again"
    );

    // And it can be.
    let record = block_on(world.handoff().execute(
        &administrator(),
        world.request("model-b", VerifyDestination::PlanOnly),
    ))
    .expect("a fresh handoff opens");
    assert_eq!(record.outcome(), TransitionOutcome::Succeeded);
}

/// Sweeping twice changes nothing the second time.
#[test]
fn the_recovery_sweep_is_safe_to_run_again() {
    let world = World::new(true);
    world
        .events
        .save_transition(&interrupted_at_commit(&world))
        .expect("written");

    assert_eq!(
        block_on(world.handoff().reconcile_open("system", None)).len(),
        1
    );
    assert!(block_on(world.handoff().reconcile_open("system", None)).is_empty());
}

/// A handoff that reached the durable commit intent and went no further.
fn interrupted_at_commit(world: &World) -> TransitionRecord {
    let agent = world.agent();
    let mut record = TransitionRecord::begin(
        "tr-crashed",
        &agent,
        ModelFingerprint::of(world.models.find("model-a").expect("registered")),
        ModelFingerprint::of(world.models.find("model-b").expect("registered")),
        agent.models.rebound_to("model-b"),
        "priya",
        "interrupted while committing",
    );
    for phase in [
        TransitionPhase::Draining,
        TransitionPhase::Checkpointed,
        TransitionPhase::Validating,
        TransitionPhase::Loading,
        TransitionPhase::Recompiling,
        TransitionPhase::CommitPending,
    ] {
        record.advance(phase, None).expect("the documented order");
    }
    record
}
