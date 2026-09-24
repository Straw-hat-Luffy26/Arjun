//! The agent registry, over IPC.
//!
//! ## Where the authorisation is, and where it is not
//!
//! It is in [`crate::agents::store`], on every mutation, and these commands add
//! nothing to it. That is deliberate: a check written here as well would be a
//! second opinion about authority, and the one that eventually disagrees is the
//! one nobody is looking at.
//!
//! What these add is the *audit line*. A refusal is recorded as well as an
//! acceptance, because "somebody who should not have been able to tried" is the
//! entry worth having, and a log that only holds successes cannot show it.
//!
//! ## Why the menu being hidden is not the control
//!
//! `AppMenu` hides the administration section from an employee. That is a
//! courtesy, not a boundary: the command is still registered and still
//! reachable by anything that can make an IPC call, which on a desktop
//! application is any script the person can be talked into running. The refusal
//! that matters is the one in Rust.

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Manager, State};

use crate::agents::store::{AgentRegistry, Mutation, RegistryError, Visibility};
use crate::agents::{AgentDefinition, AgentState, SkillBinding};
use crate::audit::{AuditKind, AuditService};
use crate::commands::agent::{Skills, Subagents};
use crate::commands::governance::{require_session, CurrentSession};

/// The registry, as the application holds it.
pub type Agents = Arc<AgentRegistry>;

/// An agent as a screen needs it: the definition, plus what will not work.
///
/// The two are separate because they answer different questions. The definition
/// is what somebody configured; `unresolved` is what this machine cannot
/// currently honour. Merging them would let an agent be presented as ready
/// because its record parsed, which is the specific thing the administration
/// screen must not do.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentView {
    pub definition: AgentDefinition,
    /// Models and skills this definition names that do not resolve right now,
    /// each as a sentence.
    pub unresolved: Vec<String>,
    /// True when this agent could be given work this moment.
    pub ready: bool,
    /// Whether a worker is registered for this agent role.
    ///
    /// Separate from `unresolved`, which is about what the *definition* names.
    /// A definition can name nothing missing and still have no worker, and that
    /// agent will accept a task and never perform it — so readiness has to
    /// depend on this too, and a screen has to be able to say which of the two
    /// is wrong.
    pub worker_available: bool,
}

/// Turns a refusal into the sentence the caller shows.
fn refused(error: RegistryError) -> String {
    error.explain()
}

/// Records what was attempted, whether or not it was allowed.
fn note(audit: &AuditService, actor: &str, summary: String, detail: serde_json::Value) {
    // Best effort. A registry change that happened and was not logged is worse
    // than one logged twice, and the alternative — refusing the change because
    // the log is unavailable — would take agent administration offline whenever
    // the audit database is busy.
    let _ = audit.record(actor, AuditKind::PolicyDecision, summary, Some(detail));
}

fn view(
    definition: AgentDefinition,
    known_models: &[String],
    known_skills: &[SkillBinding],
    worker_available: bool,
) -> AgentView {
    let problems = definition.unresolved_against(known_models, known_skills);
    AgentView {
        ready: definition.state.is_runnable() && problems.is_empty() && worker_available,
        unresolved: problems.iter().map(|problem| problem.explain()).collect(),
        worker_available,
        definition,
    }
}

/// What this machine currently has, for resolving a definition against reality.
///
/// ## Why this exists rather than the empty slices that were here
///
/// `view` was being called with `&[]` for both, which meant
/// [`AgentDefinition::unresolved_against`] reported *every* named model as
/// missing and `ready` was false for every agent that named one. The screen
/// this was written for would have shown a correctly configured deployment as
/// entirely broken.
///
/// The registries are read per call rather than cached. A snapshot is already
/// cached inside `SkillRegistry`, model entries are held in memory, and an
/// administration screen that told somebody an agent was fine because it read a
/// stale list would be the same bug in the other direction.
struct Installed {
    models: Vec<String>,
    skills: Vec<SkillBinding>,
}

/// The name a worker would be registered under for this agent.
///
/// ## Why this is not the role
///
/// Workers are registered by *profile* name — `artifact-reviewer`,
/// `code-worker` — not by model role. Asking `has_worker("reasoning")` returns
/// false for every agent that has one, which reports a working deployment as
/// entirely broken. That is the same lie as reporting a broken agent as ready,
/// just in the direction that is easier to miss.
///
/// ## Why this is not a name either
///
/// It used to be the imported profile's name, falling back to the display
/// name. Dispatch no longer works that way -- the manager finds a worker by the
/// capability the definition's output schema resolves to (plan P01; see
/// `subagents::definitions::capability_for`) -- so a name-keyed answer here
/// disagreed with what actually runs: a clone of the retriever, performed by
/// the retriever's worker, was shown as having no worker and never ready; and
/// an agent typed into the editor as "code-worker" was shown as ready for a
/// role its schema did not ask for.
///
/// `None` is a schema with a registered contract and no worker in this build
/// (document, deck, workbook). Such an agent is never reported ready.
pub(crate) fn worker_key(definition: &AgentDefinition) -> Option<&'static str> {
    crate::subagents::capability_for(definition.output_schema)
}

/// Whether a worker in this deployment performs `definition`.
pub(crate) fn has_worker_for(subagents: &Subagents, definition: &AgentDefinition) -> bool {
    worker_key(definition).is_some_and(|capability| subagents.has_worker(capability))
}

/// The names a child of this agent can carry in a `subagent_started` event.
///
/// Its registry id, which every child dispatched through the registry records
/// as `agentId` and `profile`; and, for an agent imported from a bundled
/// profile, that profile's name, which is what children dispatched before the
/// registry was consulted recorded. Never the display name.
pub(crate) fn trace_keys(definition: &AgentDefinition) -> Vec<String> {
    let mut keys = vec![definition.agent_id.clone()];
    if let Some(origin) = &definition.imported_from {
        keys.push(origin.profile_name.clone());
    }
    keys
}

fn installed(models: &Arc<crate::registry::ModelRegistry>, skills: &Skills) -> Installed {
    Installed {
        models: models.all().iter().map(|entry| entry.id.clone()).collect(),
        skills: skills
            .snapshot()
            .cards()
            .into_iter()
            .map(|card| SkillBinding {
                name: card.name,
                version: card.version,
                sha256: card.sha256,
            })
            .collect(),
    }
}

/// Every agent the caller may see.
#[tauri::command]
pub async fn agent_registry_list(
    session: State<'_, CurrentSession>,
    agents: State<'_, Agents>,
    skills: State<'_, Skills>,
    subagents: State<'_, Subagents>,
    models: State<'_, Arc<crate::registry::ModelRegistry>>,
) -> Result<Vec<AgentView>, String> {
    let signed_in = require_session(&session)?;
    let have = installed(&models, &skills);
    agents
        .list(Visibility::of(&signed_in))
        .map(|found| {
            found
                .into_iter()
                .map(|definition| {
                    let worker = has_worker_for(&subagents, &definition);
                    view(definition, &have.models, &have.skills, worker)
                })
                .collect()
        })
        .map_err(refused)
}

/// One agent, if the caller may see it.
#[tauri::command]
pub async fn agent_registry_get(
    agent_id: String,
    session: State<'_, CurrentSession>,
    agents: State<'_, Agents>,
    skills: State<'_, Skills>,
    subagents: State<'_, Subagents>,
    models: State<'_, Arc<crate::registry::ModelRegistry>>,
) -> Result<AgentView, String> {
    let signed_in = require_session(&session)?;
    let have = installed(&models, &skills);
    agents
        .get(&agent_id, Visibility::of(&signed_in))
        .map(|definition| {
            let worker = has_worker_for(&subagents, &definition);
            view(definition, &have.models, &have.skills, worker)
        })
        .map_err(refused)
}

/// Creates an agent. Administrator only.
#[tauri::command]
pub async fn agent_registry_create(
    definition: AgentDefinition,
    session: State<'_, CurrentSession>,
    agents: State<'_, Agents>,
    audit: State<'_, Arc<AuditService>>,
) -> Result<Mutation, String> {
    let signed_in = require_session(&session)?;
    let name = definition.display_name.clone();
    let outcome = agents.create(&signed_in, definition);
    note(
        &audit,
        &signed_in.user.id,
        match &outcome {
            Ok(mutation) => format!("created agent {} ({})", name, mutation.agent_id),
            Err(_) => format!("was refused the creation of agent {name}"),
        },
        serde_json::json!({
            "action": "create",
            "displayName": name,
            "allowed": outcome.is_ok(),
        }),
    );
    outcome.map_err(refused)
}

/// Replaces an agent's mutable fields, if nobody else changed it first.
#[tauri::command]
pub async fn agent_registry_update(
    agent_id: String,
    expected_version: u64,
    definition: AgentDefinition,
    session: State<'_, CurrentSession>,
    agents: State<'_, Agents>,
    audit: State<'_, Arc<AuditService>>,
) -> Result<Mutation, String> {
    let signed_in = require_session(&session)?;
    let outcome = agents.update(&signed_in, &agent_id, expected_version, definition);
    note(
        &audit,
        &signed_in.user.id,
        match &outcome {
            Ok(mutation) => format!(
                "changed agent {agent_id} to version {}",
                mutation.definition_version
            ),
            Err(error) => format!("did not change agent {agent_id}: {}", error.explain()),
        },
        serde_json::json!({
            "action": "update",
            "agentId": agent_id,
            "expectedVersion": expected_version,
            "allowed": outcome.is_ok(),
        }),
    );
    outcome.map_err(refused)
}

/// Copies an agent's configuration under a new identity and empty memory.
#[tauri::command]
pub async fn agent_registry_clone(
    agent_id: String,
    display_name: String,
    session: State<'_, CurrentSession>,
    agents: State<'_, Agents>,
    audit: State<'_, Arc<AuditService>>,
) -> Result<Mutation, String> {
    let signed_in = require_session(&session)?;
    let outcome = agents.clone_agent(&signed_in, &agent_id, &display_name);
    note(
        &audit,
        &signed_in.user.id,
        match &outcome {
            Ok(mutation) => format!(
                "cloned agent {agent_id} as {} ({})",
                display_name, mutation.agent_id
            ),
            Err(error) => format!("did not clone agent {agent_id}: {}", error.explain()),
        },
        serde_json::json!({
            "action": "clone",
            "agentId": agent_id,
            "displayName": display_name,
            "allowed": outcome.is_ok(),
        }),
    );
    outcome.map_err(refused)
}

/// Enables, disables or archives an agent.
#[tauri::command]
pub async fn agent_registry_set_state(
    agent_id: String,
    expected_version: u64,
    state: AgentState,
    session: State<'_, CurrentSession>,
    agents: State<'_, Agents>,
    audit: State<'_, Arc<AuditService>>,
) -> Result<Mutation, String> {
    let signed_in = require_session(&session)?;
    let outcome = agents.set_state(&signed_in, &agent_id, expected_version, state);
    note(
        &audit,
        &signed_in.user.id,
        match &outcome {
            Ok(mutation) if mutation.unchanged => {
                format!("left agent {agent_id} {}", state.as_str())
            }
            Ok(_) => format!("set agent {agent_id} to {}", state.as_str()),
            Err(error) => format!(
                "did not set agent {agent_id} to {}: {}",
                state.as_str(),
                error.explain()
            ),
        },
        serde_json::json!({
            "action": "setState",
            "agentId": agent_id,
            "state": state.as_str(),
            "expectedVersion": expected_version,
            "allowed": outcome.is_ok(),
        }),
    );
    outcome.map_err(refused)
}

// ─────────────────────────────────────────────────────────────────────────────
// Changing the model an agent runs on
//
// Not a field on `agent_registry_update`, and that is the point of this
// section. `AgentRegistry::update` refuses a submitted model binding outright —
// see `RegistryError::ModelBindingNeedsTransition` — because changing the model
// an agent runs on has to drain its work, settle what is in flight, save its
// state, prove the new model can hold the task, and be able to undo itself.
// None of that is something a saved form can do.
//
// So these commands are the surface for it, and every one of them is a thin
// shim over `agent_runtime::model_handoff::Handoff`, which is where the order of
// the phases lives and which a test can drive without a Tauri app.
// ─────────────────────────────────────────────────────────────────────────────

/// A transition as a screen needs it.
///
/// The record plus the three things a surface would otherwise re-derive: which
/// of the five outcomes it is, the sentence to show, and whether somebody has to
/// act. Derived here rather than in TypeScript so the phase-to-outcome mapping
/// exists once.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TransitionView {
    pub record: crate::agent_runtime::model_transition::TransitionRecord,
    /// `succeeded`, `pending`, `failed`, `rolledBack` or `rollbackFailed`.
    pub outcome: String,
    /// One sentence, in an operator's terms.
    pub summary: String,
    /// True when the agent may be bound to a model nothing is serving, so it
    /// should not be given work until a person has looked.
    pub needs_human: bool,
    /// What the new model was deliberately not given, when this committed.
    ///
    /// Surfaced rather than left implicit, because the first answer after a
    /// handoff can legitimately be shaped differently — the previous model's
    /// summary of its own transcript did not cross — and somebody reading that
    /// difference deserves to know why rather than reporting a regression.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub not_carried: Vec<DroppedNote>,
}

/// One thing a handoff left behind, and why.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DroppedNote {
    pub what: String,
    pub because: String,
}

/// Where an agent stands with respect to its model.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TransitionStatus {
    pub agent_id: String,
    /// `None` means routing chooses per turn for this agent's role.
    pub current_model_id: Option<String>,
    /// Sent back with a change request so a concurrent edit is refused rather
    /// than silently overwritten.
    pub definition_version: u64,
    /// The handoff under way, if there is one. While this is present the agent
    /// cannot be moved again — a second handoff would race over one record.
    pub open: Option<TransitionView>,
    /// Newest first.
    pub history: Vec<TransitionView>,
}

fn view_transition(
    record: crate::agent_runtime::model_transition::TransitionRecord,
) -> TransitionView {
    use crate::agent_runtime::model_transition::{DroppedState, TransitionPhase};

    let outcome = record.outcome();
    let summary = record.describe();
    TransitionView {
        not_carried: if record.phase == TransitionPhase::Committed {
            DroppedState::ALL
                .iter()
                .map(|dropped| DroppedNote {
                    what: dropped.as_str().to_string(),
                    because: dropped.explain().to_string(),
                })
                .collect()
        } else {
            Vec::new()
        },
        outcome: outcome.as_str().to_string(),
        needs_human: outcome.needs_human(),
        summary,
        record,
    }
}

/// The runtime memory graph, for a handoff that has to recompile context.
///
/// Opened per call rather than held in state. `MemoryGraph::open` creates its
/// tables, and a model change is an administrative act that happens rarely —
/// paying a schema check for it is cheaper than another piece of application
/// state to keep in step. `None` when it cannot be opened, and the handoff then
/// refuses to recompile rather than recompiling against an empty set.
fn open_graph(app: &AppHandle) -> Option<crate::knowledge::graph::runtime_store::MemoryGraph> {
    let dir = app.path().app_data_dir().ok()?;
    match crate::knowledge::graph::runtime_store::MemoryGraph::open(&dir) {
        Ok(graph) => Some(graph),
        Err(error) => {
            log::error!(
                "[agents] the runtime memory graph could not be opened, so a model change cannot \
                 rebuild a task's context: {error}"
            );
            None
        }
    }
}

/// Moves an agent to a different model, as a handoff with states.
///
/// ## What the two kinds of answer mean
///
/// `Err` means the handoff was never opened: not an administrator, no such
/// agent, a version conflict, or another handoff already under way. Nothing
/// happened and nothing was recorded beyond the audit line.
///
/// `Ok` means a record exists, and the record says where it got to — including
/// when that is "refused" or "rolled back". Read `outcome`: a handoff parked at
/// `pending` is waiting for a person to settle an approval or an unaccounted
/// effect, and asking again once they have continues it.
#[tauri::command]
#[allow(clippy::too_many_arguments)]
pub async fn agent_model_transition_begin(
    app: AppHandle,
    request: crate::agent_runtime::model_handoff::HandoffRequest,
    session: State<'_, CurrentSession>,
    agents: State<'_, Agents>,
    events: State<'_, crate::commands::agent::TaskEvents>,
    registry: State<'_, Arc<crate::registry::ModelRegistry>>,
    servers: State<'_, Arc<crate::serving::ModelServers>>,
    checkpoints: State<'_, crate::commands::agent::RunCheckpoints>,
    audit: State<'_, Arc<AuditService>>,
) -> Result<TransitionView, String> {
    let signed_in = require_session(&session)?;
    let graph = open_graph(&app);

    let handoff = crate::agent_runtime::model_handoff::Handoff {
        agents: agents.inner(),
        events: events.inner(),
        registry: registry.inner(),
        servers: servers.inner(),
        graph: graph.as_ref(),
        checkpoints: checkpoints.inner(),
        models_dir: registry.models_dir(),
        leases: tauri::Manager::try_state::<Arc<crate::subagents::ModelScheduler>>(&app)
            .map(|held| held.inner().as_ref()),
    };

    let asked = request.clone();
    let outcome = handoff.execute(&signed_in, request).await;

    note(
        &audit,
        &signed_in.user.id,
        match &outcome {
            Ok(record) => format!(
                "moved agent {} from {} to {} ({}): {}",
                asked.agent_id,
                record.from_model.model_id,
                record.to_model.model_id,
                record.phase.as_str(),
                record.outcome().as_str()
            ),
            Err(refusal) => format!(
                "was not able to move agent {} to {}: {refusal}",
                asked.agent_id, asked.target_model_id
            ),
        },
        serde_json::json!({
            "action": "modelTransition",
            "agentId": asked.agent_id,
            "targetModelId": asked.target_model_id,
            "expectedVersion": asked.expected_definition_version,
            "runId": asked.run_id,
            "verify": asked.verify,
            "transitionId": outcome.as_ref().ok().map(|r| r.transition_id.clone()),
            "phase": outcome.as_ref().ok().map(|r| r.phase.as_str()),
            "outcome": outcome.as_ref().ok().map(|r| r.outcome().as_str()),
            // Both digests, so the audit line can be joined to the exact bytes
            // this agent was moved between even after a re-import moves an id.
            "fromModelDigest": outcome.as_ref().ok().map(|r| r.from_model.digest.clone()),
            "toModelDigest": outcome.as_ref().ok().map(|r| r.to_model.digest.clone()),
            "opened": outcome.is_ok(),
        }),
    );

    outcome.map(view_transition)
}

/// Where an agent stands with respect to its model, and what has been tried.
#[tauri::command]
pub async fn agent_model_transition_status(
    agent_id: String,
    session: State<'_, CurrentSession>,
    agents: State<'_, Agents>,
    events: State<'_, crate::commands::agent::TaskEvents>,
) -> Result<TransitionStatus, String> {
    let signed_in = require_session(&session)?;
    // Visibility-checked like any other read: an employee cannot use this to
    // discover which agents a deployment has disabled.
    let definition = agents
        .get(&agent_id, Visibility::of(&signed_in))
        .map_err(refused)?;

    Ok(TransitionStatus {
        current_model_id: definition.models.default_model_id.clone(),
        definition_version: definition.definition_version,
        open: events
            .open_transition_for_agent(&agent_id)?
            .map(view_transition),
        history: events
            .transitions_for_agent(&agent_id, 25)?
            .into_iter()
            .map(view_transition)
            .collect(),
        agent_id,
    })
}

/// Undoes a handoff that has not settled.
///
/// Bounded: it puts the binding back and, if something was released to make
/// room, serves the source model again. It does not unwind the work — a document
/// written between the checkpoint and the failure stays written, and its receipt
/// stays in the run's notes, which is what stops a resumption writing it twice.
#[tauri::command]
#[allow(clippy::too_many_arguments)]
pub async fn agent_model_transition_rollback(
    app: AppHandle,
    transition_id: String,
    session: State<'_, CurrentSession>,
    agents: State<'_, Agents>,
    events: State<'_, crate::commands::agent::TaskEvents>,
    registry: State<'_, Arc<crate::registry::ModelRegistry>>,
    servers: State<'_, Arc<crate::serving::ModelServers>>,
    checkpoints: State<'_, crate::commands::agent::RunCheckpoints>,
    audit: State<'_, Arc<AuditService>>,
) -> Result<TransitionView, String> {
    let signed_in = require_session(&session)?;
    let graph = open_graph(&app);

    let handoff = crate::agent_runtime::model_handoff::Handoff {
        agents: agents.inner(),
        events: events.inner(),
        registry: registry.inner(),
        servers: servers.inner(),
        graph: graph.as_ref(),
        checkpoints: checkpoints.inner(),
        models_dir: registry.models_dir(),
        leases: tauri::Manager::try_state::<Arc<crate::subagents::ModelScheduler>>(&app)
            .map(|held| held.inner().as_ref()),
    };

    let outcome = handoff.rollback(&signed_in, &transition_id).await;
    note(
        &audit,
        &signed_in.user.id,
        match &outcome {
            Ok(record) => format!(
                "undid the move of agent {} to {} ({})",
                record.agent_id,
                record.to_model.model_id,
                record.phase.as_str()
            ),
            Err(refusal) => format!("did not undo model change {transition_id}: {refusal}"),
        },
        serde_json::json!({
            "action": "modelTransitionRollback",
            "transitionId": transition_id,
            "phase": outcome.as_ref().ok().map(|r| r.phase.as_str()),
            "outcome": outcome.as_ref().ok().map(|r| r.outcome().as_str()),
            "allowed": outcome.is_ok(),
        }),
    );
    outcome.map(view_transition)
}

/// Settles every handoff the process died in the middle of.
///
/// Run at start-up by [`reconcile_interrupted_transitions`], and reachable here
/// so an operator can ask for it after a crash without restarting the
/// application. Safe to run twice: a row that has settled is not returned by
/// `open_transitions`, so a second sweep finds nothing to do.
#[tauri::command]
#[allow(clippy::too_many_arguments)]
pub async fn agent_model_transition_reconcile(
    app: AppHandle,
    session: State<'_, CurrentSession>,
    agents: State<'_, Agents>,
    events: State<'_, crate::commands::agent::TaskEvents>,
    registry: State<'_, Arc<crate::registry::ModelRegistry>>,
    servers: State<'_, Arc<crate::serving::ModelServers>>,
    checkpoints: State<'_, crate::commands::agent::RunCheckpoints>,
    audit: State<'_, Arc<AuditService>>,
) -> Result<Vec<TransitionView>, String> {
    let signed_in = require_session(&session)?;
    if !crate::agents::store::AgentRegistry::may_administer(&signed_in) {
        return Err(RegistryError::NotAdministrator {
            action: "settle an interrupted model change".into(),
        }
        .explain());
    }
    let graph = open_graph(&app);

    let handoff = crate::agent_runtime::model_handoff::Handoff {
        agents: agents.inner(),
        events: events.inner(),
        registry: registry.inner(),
        servers: servers.inner(),
        graph: graph.as_ref(),
        checkpoints: checkpoints.inner(),
        models_dir: registry.models_dir(),
        leases: tauri::Manager::try_state::<Arc<crate::subagents::ModelScheduler>>(&app)
            .map(|held| held.inner().as_ref()),
    };

    let settled = handoff
        .reconcile_open(&signed_in.user.id, Some(&signed_in))
        .await;
    if !settled.is_empty() {
        note(
            &audit,
            &signed_in.user.id,
            format!("settled {} interrupted model change(s)", settled.len()),
            serde_json::json!({
                "action": "modelTransitionReconcile",
                "settled": settled
                    .iter()
                    .map(|record| serde_json::json!({
                        "transitionId": record.transition_id,
                        "agentId": record.agent_id,
                        "phase": record.phase.as_str(),
                        "outcome": record.outcome().as_str(),
                    }))
                    .collect::<Vec<_>>(),
            }),
        );
    }
    Ok(settled.into_iter().map(view_transition).collect())
}

/// The colours an agent may be given, so a picker does not restate them.
#[tauri::command]
pub async fn agent_registry_palette() -> Result<Vec<String>, String> {
    Ok(crate::agents::AGENT_PALETTE
        .iter()
        .map(|colour| (*colour).to_string())
        .collect())
}

/// Imports every bundled profile that is not already mapped to an agent.
///
/// Runs at start-up and is safe to run again: see
/// [`AgentRegistry::import_bundled`] for the three outcomes and why a re-import
/// of an unchanged profile does not move a version number.
pub fn import_bundled_profiles(
    agents: &AgentRegistry,
    profiles: &[crate::subagents::profile::AgentProfile],
) {
    let mut imported = 0;
    let mut updated = 0;
    for profile in profiles {
        match agents.import_bundled(profile) {
            Ok(mutation) if mutation.unchanged => {}
            Ok(mutation) if mutation.definition_version == 1 => imported += 1,
            Ok(_) => updated += 1,
            Err(error) => log::warn!(
                "[agents] the bundled profile {} was not imported: {}",
                profile.name,
                error.explain()
            ),
        }
    }
    if imported > 0 || updated > 0 {
        log::info!("[agents] {imported} profile(s) imported, {updated} updated from the bundle");
    }
}

/// Settles every model change the previous run of this process died inside.
///
/// Called once at start-up. Nothing here needs anybody to be signed in — see
/// [`crate::agent_runtime::model_handoff::Handoff::reconcile_open`] for why a
/// recovery sweep is given no authority and does not need any — and a state this
/// application has not managed yet simply means there is nothing to sweep.
pub async fn reconcile_interrupted_transitions(app: &AppHandle) {
    use crate::agent_runtime::events::SYSTEM_ACTOR;

    // Every piece has to be there. A sweep that ran with half of them would be
    // a sweep that could not reload a source model or read a registry, and a
    // partial answer about which model an agent is bound to is the one kind of
    // answer this whole module exists to avoid.
    let (Some(agents), Some(events), Some(registry), Some(servers), Some(checkpoints)) = (
        app.try_state::<Agents>(),
        app.try_state::<crate::commands::agent::TaskEvents>(),
        app.try_state::<Arc<crate::registry::ModelRegistry>>(),
        app.try_state::<Arc<crate::serving::ModelServers>>(),
        app.try_state::<crate::commands::agent::RunCheckpoints>(),
    ) else {
        log::debug!(
            "[agents] the model-change ledger was not swept: this session is running without one \
             of the registries it needs"
        );
        return;
    };

    let graph = open_graph(app);
    let handoff = crate::agent_runtime::model_handoff::Handoff {
        agents: agents.inner(),
        events: events.inner(),
        registry: registry.inner(),
        servers: servers.inner(),
        graph: graph.as_ref(),
        checkpoints: checkpoints.inner(),
        models_dir: registry.models_dir(),
        leases: tauri::Manager::try_state::<Arc<crate::subagents::ModelScheduler>>(app)
            .map(|held| held.inner().as_ref()),
    };

    let settled = handoff.reconcile_open(SYSTEM_ACTOR, None).await;
    for record in &settled {
        log::info!(
            "[agents] handoff {} for {} was settled as {}: {}",
            record.transition_id,
            record.agent_id,
            record.outcome().as_str(),
            record.describe()
        );
    }
}

/// Opens the registry, or explains why the application is starting without one.
pub fn open(app: &AppHandle) -> Option<Agents> {
    let app_data = app.path().app_data_dir().ok()?;
    match AgentRegistry::open(&app_data) {
        Ok(registry) => Some(Arc::new(registry)),
        Err(error) => {
            // Said out loud at start rather than as a refusal on turn three.
            log::error!(
                "[agents] the agent registry could not be opened: {}",
                error.explain()
            );
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agents::ImportOrigin;

    /// The lookup key is the worker's capability, not the role.
    ///
    /// This is pinned because the first version asked
    /// `has_worker("reasoning")`, which is never a registered key — workers
    /// register under profile names — so every agent in a healthy deployment
    /// was reported as having no worker. A screen that says a working
    /// deployment is broken is as wrong as one that says a broken deployment
    /// is fine; this test is the one that notices.
    #[test]
    fn the_worker_key_is_the_profile_name_not_the_role() {
        let mut imported = crate::agents::tests::definition("ag-1");
        imported.display_name = "Renamed by an administrator".into();
        imported.imported_from = Some(ImportOrigin {
            source: "bundle".into(),
            profile_name: "artifact-reviewer".into(),
            profile_sha256: "a".repeat(64),
        });

        imported.output_schema = crate::subagents::SchemaKind::Review;

        // Not "reasoning", and not the display name either: a rename must not
        // change which worker performs the work.
        assert_eq!(worker_key(&imported), Some("artifact-reviewer"));
        assert_ne!(worker_key(&imported), Some(imported.role.label()));
    }

    /// Plan P01: which worker performs an agent is decided by what it
    /// produces, never by what it is called -- the same rule dispatch uses.
    ///
    /// This replaced a test that pinned the opposite: an agent with no import
    /// origin fell back to its display name, so one *named* "code-worker" was
    /// reported as having the code worker, and a clone of the retriever was
    /// reported as having none.
    #[test]
    fn the_worker_is_the_capability_of_the_output_and_not_a_name() {
        // A clone: no origin, a new name, the retriever's schema.
        let mut clone = crate::agents::tests::definition("ag-clone");
        clone.imported_from = None;
        clone.display_name = "Seal-wear retriever".into();
        clone.output_schema = crate::subagents::SchemaKind::Retrieval;
        assert_eq!(worker_key(&clone), Some("knowledge-retriever"));

        // A name that happens to be a worker's is not that worker.
        let mut named = crate::agents::tests::definition("ag-named");
        named.imported_from = None;
        named.display_name = "code-worker".into();
        named.output_schema = crate::subagents::SchemaKind::Retrieval;
        assert_eq!(worker_key(&named), Some("knowledge-retriever"));

        // A registered contract with no worker yet has no worker key at all,
        // so it can never be reported ready.
        for schema in [
            crate::subagents::SchemaKind::Document,
            crate::subagents::SchemaKind::Deck,
            crate::subagents::SchemaKind::Workbook,
        ] {
            let mut writer = crate::agents::tests::definition("ag-writer");
            writer.output_schema = schema;
            assert_eq!(worker_key(&writer), None, "{schema:?} claims a worker");
        }
    }

    /// A child of this agent is found in the trace by its id, and -- for an
    /// imported agent -- by the bundled role name older events recorded.
    #[test]
    fn a_childs_trace_names_its_agent_by_id_and_never_by_display_name() {
        let mut imported = crate::agents::tests::definition("ag-3");
        imported.display_name = "Passage Finder".into();
        imported.imported_from = Some(ImportOrigin {
            source: "bundle".into(),
            profile_name: "knowledge-retriever".into(),
            profile_sha256: "b".repeat(64),
        });
        let keys = trace_keys(&imported);
        assert_eq!(keys, vec!["ag-3".to_string(), "knowledge-retriever".to_string()]);
        assert!(!keys.contains(&"Passage Finder".to_string()));
    }
}
