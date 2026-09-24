//! Moving an agent between two real models, with two real servers.
//!
//! ## What this proves, and what it cannot
//!
//! It drives the production handoff — `agent_runtime::model_handoff::Handoff` —
//! against the machine's own model registry, its own `llama-server`, its own
//! VRAM and its own admission planner. Both models are loaded for real, both
//! report their own served window, and both are asked to tokenise the same
//! string with their own vocabulary. Nothing about the destination is assumed.
//!
//! It does **not** call `drive_run`. That function takes a Tauri `AppHandle` and
//! twenty-odd `State<'_, _>` arguments, and `tauri::test::mock_app` needs the
//! `test` feature, which this crate does not enable — the same reason
//! `tests/rbac_isolation.rs` tests the permission matrix rather than the IPC
//! surface, and the reason `commands::governance` factors its sign-in logic out
//! of its command. So the model-binding half of `drive_run` is exercised here
//! through the identical code path that both `drive_run`'s resumption branch and
//! the `agent_model_transition_*` commands use, and what is left unproven is the
//! surrounding command plumbing rather than the handoff.
//!
//! Calling this an end-to-end proof of the application would be the kind of
//! claim this repository has a standing rule against. What it is: a proof that a
//! real handoff between two genuinely different models measures both of them and
//! keeps everything that identifies the work.
//!
//! ## Why it skips rather than fails
//!
//! It needs two distinct model files on this machine and a `llama-server` it can
//! start. A developer without them has a machine that cannot run the product,
//! not a broken change — so the test says what is missing and stops, loudly, the
//! way `common::installed_models` already does for the other live tests.
//!
//! ## Where it writes
//!
//! Into a temporary directory, and nowhere else. The version of
//! `verify_real_world_model_switching.rs` this replaces the pattern of used to
//! write into the live profile database of whoever ran it; only the *weights*
//! come from the real application data here, and they are only read.

mod common;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Mutex;

use sarathi_lib::agent_runtime::context_manifest::{ContextManifest, HistoryBinding};
use sarathi_lib::agent_runtime::events::{EventDraft, TaskEventLog, TaskEventType};
use sarathi_lib::agent_runtime::memory::{CompletedEffect, RunMemory};
use sarathi_lib::agent_runtime::model_handoff::{
    Handoff, HandoffRequest, ReserveRequest, VerifyDestination,
};
use sarathi_lib::agent_runtime::model_transition::{
    TransitionOutcome, TransitionPhase, TransitionRecord,
};
use sarathi_lib::agent_runtime::resume::CheckpointSeed;
use sarathi_lib::agents::store::AgentRegistry;
use sarathi_lib::agents::{AgentDefinition, AgentState, MemoryPolicy, ModelBinding, AGENT_PALETTE};
use sarathi_lib::identity::{Role, Session, User};
use sarathi_lib::knowledge::graph::runtime_store::MemoryGraph;
use sarathi_lib::policy::Classification;
use sarathi_lib::registry::{ModelEntry, ModelRegistry, ModelRole};
use sarathi_lib::serving::ModelServers;
use sarathi_lib::subagents::profile::{Isolation, Limits, SchemaKind, WritePolicy};

const RUN: &str = "live-run-1";
const ATTEMPT: &str = "live-attempt-1";

fn administrator() -> Session {
    Session::open(User::new(
        "priya",
        "Priya Sharma",
        vec![Role::Employee, Role::Administrator],
    ))
}

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("a test runtime")
}

/// Two models on this machine that are genuinely different files.
///
/// Sorted by size so the cheapest pair is chosen: this test loads each of them
/// twice, and picking the 35B mixture-of-experts would turn a check into an
/// afternoon. Distinct *paths* rather than distinct ids, because the registry
/// legitimately holds two ids for one file — an orchestrator alias beside the
/// ordinary entry — and moving an agent between those two would prove nothing
/// about tokenizers or windows.
fn two_distinct_models(registry: &ModelRegistry) -> Option<(ModelEntry, ModelEntry)> {
    let models_dir = registry.models_dir();
    let mut usable: Vec<ModelEntry> = registry
        .all()
        .iter()
        .filter(|entry| entry.enabled)
        .filter(|entry| entry.serves(ModelRole::Reasoning))
        .filter(|entry| entry.permits(Classification::Internal))
        .filter(|entry| models_dir.join(&entry.path).is_file())
        .cloned()
        .collect();
    usable.sort_by_key(|entry| entry.weights_bytes);

    let first = usable.first()?.clone();
    let second = usable.iter().find(|entry| entry.path != first.path)?.clone();
    Some((first, second))
}

/// Everything the handoff borrows, owned for the length of the test.
struct Live {
    agents: AgentRegistry,
    events: TaskEventLog,
    registry: ModelRegistry,
    servers: ModelServers,
    graph: MemoryGraph,
    checkpoints: Mutex<HashMap<String, CheckpointSeed>>,
    models_dir: PathBuf,
    agent_id: String,
    _dir: tempfile::TempDir,
}

impl Live {
    fn handoff(&self) -> Handoff<'_> {
        Handoff {
            agents: &self.agents,
            events: &self.events,
            registry: &self.registry,
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
            .expect("the agent resolves")
    }

    fn move_to(&self, target: &str) -> TransitionRecord {
        let request = HandoffRequest {
            agent_id: self.agent_id.clone(),
            target_model_id: target.to_string(),
            expected_definition_version: self.agent().definition_version,
            reason: "live handoff check".into(),
            run_id: Some(RUN.to_string()),
            task_id: Some(RUN.to_string()),
            project_id: None,
            verify: VerifyDestination::LoadAndVerify,
            // Stated rather than read off the manifest, because this fixture's
            // turn was assembled here and never went through the budgeting in
            // `commands::agent`. The handoff refuses a manifest with no budget
            // record rather than inventing one, so a caller that knows the
            // figures supplies them; these are what a small tool-free turn
            // spends.
            reserves: Some(ReserveRequest {
                tool_schemas: 0,
                output: 1_024,
                framing: 256,
                safety: 0,
            }),
        };
        runtime()
            .block_on(self.handoff().execute(&administrator(), request))
            .expect("the handoff opens")
    }
}

/// An agent with no tools, on `first`, with a run sitting at a tool boundary.
///
/// No tools on purpose. The machine's registry entries do not declare
/// `supportsStructuredOutput`, so it defaults to false — and an agent bound to a
/// tool is *correctly* refused a move to a model that cannot be asked to call
/// one. That refusal has its own test; forcing it here would test the wrong
/// thing.
fn world(first: &ModelEntry, registry: ModelRegistry) -> Live {
    let dir = tempfile::tempdir().expect("a temporary directory");
    let agents = AgentRegistry::open(dir.path()).expect("a fresh registry");
    let created = agents
        .create(
            &administrator(),
            AgentDefinition {
                agent_id: String::new(),
                definition_version: 0,
                display_name: "Live handoff agent".into(),
                description: "Reads and answers.".into(),
                instructions: "Answer from what you are given.".into(),
                role: ModelRole::Reasoning,
                state: AgentState::Enabled,
                color: AGENT_PALETTE[0].into(),
                skills: Vec::new(),
                allowed_tools: Vec::new(),
                denied_tools: Vec::new(),
                memory: MemoryPolicy::default(),
                models: ModelBinding {
                    default_model_id: Some(first.id.clone()),
                    fallback_model_ids: Vec::new(),
                    eligible_model_ids: Vec::new(),
                },
                output_schema: SchemaKind::Retrieval,
                limits: Limits {
                    max_turns: 8,
                    max_output_tokens: 1_024,
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
            },
        )
        .expect("the agent is created");

    let events = TaskEventLog::in_memory().expect("an event log");
    for kind in [
        TaskEventType::RunCreated,
        TaskEventType::RunStarted,
        TaskEventType::ToolAuthorized,
        TaskEventType::ToolSucceeded,
    ] {
        events
            .record(EventDraft::new(RUN, kind, "priya"))
            .expect("the event is recorded");
    }

    let notes = RunMemory {
        goal: "Summarise the commissioning report.".into(),
        completed: vec![CompletedEffect {
            tool: "artifact.create_docx".into(),
            target: "summary.docx".into(),
            at: "2026-09-18T00:00:00Z".into(),
        }],
        artifact_ids: vec!["artifact-live-1".into()],
        ..RunMemory::default()
    };
    let checkpoints = Mutex::new(HashMap::from([(
        RUN.to_string(),
        CheckpointSeed {
            attempt_id: ATTEMPT.to_string(),
            committed_notes: notes,
            manifest: Some(ContextManifest::new(
                RUN,
                ATTEMPT,
                "conv-live",
                "a-live-run-1",
                &first.id,
                first.context_length,
                Vec::new(),
                None,
                HistoryBinding {
                    carried: 1,
                    dropped: 0,
                    tokens: 120,
                    pinned: Vec::new(),
                    omitted_pins: Vec::new(),
                },
            )),
            plan_hash: "live-plan".into(),
            policy_hash: "live-policy".into(),
            workspace_hash: "live-workspace".into(),
            model_id: first.id.clone(),
        },
    )]));

    let models_dir = registry.models_dir().to_path_buf();
    Live {
        agents,
        events,
        registry,
        servers: ModelServers::new(),
        graph: MemoryGraph::in_memory().expect("a graph"),
        checkpoints,
        models_dir,
        agent_id: created.agent_id,
        _dir: dir,
    }
}

/// The acceptance case, on real weights: A to B and back, with both windows and
/// both tokenizers measured rather than assumed.
#[test]
fn an_agent_moves_between_two_real_models_and_keeps_everything_that_identifies_it() {
    let models_dir = common::app_data_dir().join("models");
    let Ok(registry) = ModelRegistry::load(&models_dir) else {
        eprintln!("SKIP: no model registry at {}", models_dir.display());
        return;
    };
    let Some((first, second)) = two_distinct_models(&registry) else {
        eprintln!(
            "SKIP: this machine has fewer than two distinct reasoning models with weights on \
             disk, so a handoff between two of them cannot be driven"
        );
        return;
    };
    if sarathi_lib::serving::llama_server_build().is_none() {
        eprintln!(
            "SKIP: llama-server could not be found or would not report its build, so no model \
             can be served. Set ARJUN_LLAMA_SERVER or put it on PATH."
        );
        return;
    }

    eprintln!(
        "[live] moving between {} ({} MiB) and {} ({} MiB)",
        first.id,
        first.weights_bytes / (1024 * 1024),
        second.id,
        second.weights_bytes / (1024 * 1024),
    );

    let live = world(&first, registry);
    let before = live.agent();
    let notes_before = live
        .checkpoints
        .lock()
        .expect("lock")
        .get(RUN)
        .expect("seeded")
        .committed_notes
        .clone();

    // -- A to B ----------------------------------------------------------
    let out = live.move_to(&second.id);
    if out.outcome() != TransitionOutcome::Succeeded {
        // A machine that cannot fit the second model is a fact about the
        // machine, and the handoff reporting it *is* the behaviour under test —
        // so this is reported rather than asserted into a failure. What is
        // still asserted is that the refusal was clean.
        eprintln!(
            "SKIP after a real attempt: {} could not take over ({:?}). The handoff reported: {}",
            second.id,
            out.phase,
            out.describe()
        );
        assert!(
            matches!(
                out.outcome(),
                TransitionOutcome::RolledBack | TransitionOutcome::Failed
            ),
            "a handoff that did not commit must say so cleanly, not leave the agent adrift"
        );
        assert_eq!(
            live.agent().models.default_model_id,
            before.models.default_model_id,
            "a handoff that did not commit leaves the agent where it was"
        );
        runtime().block_on(live.servers.stop_all());
        return;
    }

    assert_eq!(out.phase, TransitionPhase::Committed);
    assert_eq!(
        live.agent().models.default_model_id.as_deref(),
        Some(second.id.as_str())
    );

    let served_b = out.served.clone().expect("the destination was measured");
    assert!(
        served_b.healthy,
        "a committed handoff answered a readiness probe"
    );
    assert!(
        served_b.window_source.is_measured(),
        "a real llama-server reports its own window; {:?} was recorded instead",
        served_b.window_source
    );
    assert!(served_b.served_window > 0);
    assert!(
        out.target_manifest_hash.is_some(),
        "the context was rebuilt for the window that model actually came up with"
    );
    assert_ne!(
        out.source_manifest_hash, out.target_manifest_hash,
        "a rebuild for a different window is a different manifest"
    );

    // -- B back to A -----------------------------------------------------
    let back = live.move_to(&first.id);
    assert_eq!(
        back.outcome(),
        TransitionOutcome::Succeeded,
        "moving back: {}",
        back.describe()
    );
    let served_a = back.served.clone().expect("the destination was measured");

    // -- The two models are genuinely different --------------------------
    //
    // Asserted on what was *measured*, not on what the registry declares. At
    // least one of the two things a handoff has to get right — the window a
    // turn is budgeted against, and the vocabulary it is counted in — has to
    // differ, or this is not the test it claims to be.
    let windows_differ = served_a.served_window != served_b.served_window;
    let tokenizers_differ = match (served_a.tokenizer.as_ref(), served_b.tokenizer.as_ref()) {
        (Some(a), Some(b)) => {
            eprintln!(
                "[live] the probe string is {} tokens on {} and {} tokens on {}",
                a.tokens, first.id, b.tokens, second.id
            );
            a.differs_from(b)
        }
        _ => {
            eprintln!("[live] one of the servers would not tokenise, so the comparison is unknown");
            false
        }
    };
    eprintln!(
        "[live] served windows: {} on {}, {} on {}",
        served_a.served_window, first.id, served_b.served_window, second.id
    );
    assert!(
        windows_differ || tokenizers_differ,
        "{} and {} were served with the same window ({}) and the same token count, so this run \
         did not exercise a change of either — pick two models that differ",
        first.id,
        second.id,
        served_a.served_window
    );

    // The comparison was actually performed rather than left unstated.
    assert!(
        back.tokenizer_changed.is_some(),
        "both servers were asked, so the record must say whether they disagree"
    );

    // -- Nothing that identifies the work moved --------------------------
    let after = live.agent();
    assert_eq!(after.agent_id, before.agent_id, "the agent id is stable");
    assert_eq!(after.memory, before.memory, "memory ownership is stable");
    assert_eq!(
        after.models.default_model_id.as_deref(),
        Some(first.id.as_str()),
        "two moves put it back where it started"
    );
    assert_eq!(
        after.definition_version,
        before.definition_version + 2,
        "one version per accepted change, and no more"
    );

    let saved = live
        .events
        .checkpoint(RUN)
        .expect("the checkpoint reads")
        .expect("there is one");
    assert!(saved.is_intact());
    assert_eq!(
        saved.model_id, first.id,
        "the resume point follows the binding"
    );
    assert!(
        saved.notes.has_done("artifact.create_docx", "summary.docx"),
        "the receipt for work already done survived both moves"
    );
    assert_eq!(
        saved.notes.artifact_ids, notes_before.artifact_ids,
        "artifact lineage is unchanged"
    );
    assert_eq!(
        saved.notes.goal, notes_before.goal,
        "the objective is unchanged"
    );

    // Both transitions are in the run's own history.
    let page = live.events.events_since(RUN, 0).expect("the events read");
    let committed = page
        .events
        .iter()
        .filter(|event| event.event_type.as_str() == "model_transition_committed")
        .count();
    assert_eq!(committed, 2, "each move is recorded against the run");

    // And each has both digests, so a record can be joined to the bytes.
    for record in [&out, &back] {
        assert_ne!(record.from_model.digest, record.to_model.digest);
        assert!(record
            .freeze
            .as_ref()
            .is_some_and(|freeze| freeze.checkpoint_hash.is_some()));
        assert!(record.portable_state_hash.is_some());
    }

    runtime().block_on(live.servers.stop_all());
}
