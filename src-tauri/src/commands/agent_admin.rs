//! What the administration screen needs to know before it lets somebody save.
//!
//! Read-only, all of it. The mutations live in [`super::agents`]; everything
//! here answers a question, and answering a question must never start work.
//!
//! ## Why "it parsed" is not "it will run"
//!
//! An agent definition is a record. It can name a model that is not installed,
//! a skill whose bytes have changed since it was pinned, a role no worker is
//! registered for, and a tool the signed-in operator does not hold — and still
//! be a perfectly valid record. A screen that reported such an agent as ready
//! because the record loaded would be telling an administrator the opposite of
//! the truth at exactly the moment they are deciding whether to rely on it.
//!
//! So readiness here is assembled from the things that actually decide it:
//! the model registry, the skill registry's snapshot, the worker manager, and
//! the same narrowing the runtime performs. Each failure is reported as a
//! sentence naming what is missing and what would fix it.
//!
//! ## Why the effective policy is computed by somebody else
//!
//! [`crate::subagents::inherit::InheritedPolicy::narrow_for`] is the only place
//! in this program that may construct an `EffectivePolicy`. It intersects every
//! set with the parent's and takes the `min` of every scalar, which is what
//! makes assignment incapable of widening anything.
//!
//! This module calls it and reports what it returns. It does not re-implement
//! the rule, and it must never be allowed to: a preview computed by a second
//! copy of the narrowing is a preview that can disagree with what the runtime
//! will actually do, and the whole point of a preview is that it does not.

use std::path::PathBuf;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tauri::{Manager, State};

use crate::agent_runtime::events::TaskEventType;
use crate::agents::store::Visibility;
use crate::agents::AgentDefinition;
use crate::commands::agent::{Skills, Subagents};
use crate::commands::agents::Agents;
use crate::commands::governance::{require_session, CurrentSession};
use crate::identity::Session;
use crate::policy::Classification;
use crate::registry::ModelRegistry;
use crate::subagents::inherit::InheritedPolicy;

/// One installed skill, as the editor's picker needs it.
///
/// A flattened `SkillCard` rather than the card itself: the card carries
/// selection machinery — input kinds, formats — that an administrator choosing
/// what an agent may use has no business reading, and the fields that *do*
/// matter to that decision are the trust ones.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SkillOption {
    pub name: String,
    pub description: String,
    pub version: String,
    /// SHA-256 of the whole `SKILL.md`. What a binding pins.
    pub sha256: String,
    pub classification: Classification,
    /// Tools the skill's own manifest says it needs.
    ///
    /// Named "required" rather than "allowed" on purpose: from the agent's side
    /// these are a *dependency*, and binding a skill whose tools the agent is
    /// denied produces a skill that loads and cannot act.
    pub required_tools: Vec<String>,
    pub approval_class: String,
    pub network: String,
    /// False when the skill is quarantined and will not load.
    pub available: bool,
    /// Why it will not load, when it will not.
    pub unavailable_because: Option<String>,
    /// True when this came from an outside collection rather than being written
    /// for this product.
    pub imported: bool,
}

/// What an agent declares, what it would actually get, and the difference.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PolicyPreview {
    /// The tools the definition asks for.
    pub declared_tools: Vec<String>,
    /// The tools it would actually hold. Never wider than `declared_tools`.
    pub effective_tools: Vec<String>,
    /// Asked for and not granted.
    pub refused_tools: Vec<String>,
    /// Tools the definition denies outright. These beat everything.
    pub denied_tools: Vec<String>,
    pub memory_scope: String,
    pub isolation: String,
    pub write_policy: String,
    pub classification_ceiling: Classification,
    pub required_schema: String,
    pub max_turns: u32,
    pub max_output_tokens: u32,
    pub max_duration_seconds: u64,
}

/// How one bound skill stands against what is installed right now.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SkillResolution {
    pub name: String,
    /// The bytes this agent was bound to.
    pub pinned_sha256: String,
    /// The bytes installed now. Absent when nothing by that name is installed.
    pub installed_sha256: Option<String>,
    /// `ok`, `missing`, `changed`, or `quarantined`.
    pub status: String,
    pub because: String,
}

/// What a dry run found, without running anything.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentPreview {
    pub agent_id: String,
    /// The version this preview describes. A form holding an older one is
    /// looking at a different agent than the one that would run.
    pub definition_version: u64,
    /// The model that would be chosen, first available in preference order.
    pub selected_model: Option<String>,
    /// The whole preference order, so a reader can see what it fell back past.
    pub model_preference: Vec<String>,
    /// Named models that are not installed on this machine.
    pub missing_models: Vec<String>,
    /// `binding` when this agent names its own models, `routing` when it leaves
    /// the choice to the role.
    ///
    /// A screen that showed an empty model field without this would read as
    /// "misconfigured" for the most ordinary configuration there is.
    pub model_chosen_by: String,
    pub policy: PolicyPreview,
    pub skills: Vec<SkillResolution>,
    /// Whether a worker is registered for this agent's role.
    ///
    /// The field that stops "the profile parsed" being mistaken for "this can
    /// be given work". A role with no registered worker accepts a task and
    /// never performs it.
    pub worker_available: bool,
    /// How many memory items this agent's scope currently holds, as the
    /// signed-in reader may see them.
    ///
    /// Counted, never returned. A preview says how much context a run would
    /// draw on; showing the content would put an agent's memory on a
    /// configuration screen, where it does not belong.
    pub memory_items_in_scope: usize,
    /// Everything standing between this agent and running, in sentences.
    ///
    /// Empty means it would run. Non-empty is the honest answer to "why is this
    /// greyed out", and is what the screen shows instead of a disabled button
    /// with no explanation.
    pub blocked: Vec<String>,
}

/// A run with a live child of this agent, for the confirmation before a
/// destructive lifecycle change.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DependentRun {
    pub run_id: String,
    /// What was asked, in the person's own words.
    pub prompt: String,
    /// Who started it.
    pub actor: String,
    pub state: String,
    /// Children of this agent still running inside that task.
    pub children: Vec<String>,
}

/// Everything that would be affected by archiving or disabling an agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentDependents {
    pub agent_id: String,
    /// Runs in flight with a child of this agent that has not stopped.
    pub active_runs: Vec<DependentRun>,
    /// How many memory items are attributed to this agent.
    ///
    /// Reported because it is the thing most likely to be *assumed* lost.
    /// Archiving keeps every one of them, and keeps them attributed — which is
    /// exactly what makes archival different from deletion.
    pub memory_items: usize,
    /// Why archiving now would be unsafe. Empty means it is safe.
    ///
    /// Separate from `active_runs`, because a run with a live child is
    /// something to know and an unsettled handoff is something that stops you,
    /// and one list would make the reader work out which is which.
    pub blocking: Vec<String>,
}

fn card_option(card: crate::skills::manifest::SkillCard) -> SkillOption {
    SkillOption {
        available: card.is_available(),
        unavailable_because: card.quarantined.as_ref().map(|why| why.explain()),
        required_tools: card.allowed_tools.clone(),
        approval_class: format!("{:?}", card.approval_class),
        network: format!("{:?}", card.network),
        imported: card.imported,
        name: card.name,
        description: card.description,
        version: card.version,
        sha256: card.sha256,
        classification: card.classification,
    }
}

/// Every installed skill, from the registry's cached snapshot.
///
/// ## Lazily, which means the snapshot rather than a walk
///
/// `SkillRegistry::snapshot` returns what discovery already found; `reload`
/// re-walks the directory. The editor opens with the snapshot, because walking
/// the filesystem every time somebody opens a picker is a cost paid on every
/// keystroke for information that changes when an operator installs something.
/// `refresh: true` is the explicit "I have just installed one" path, and is
/// wired to a button rather than to the render.
#[tauri::command]
pub async fn agent_skill_catalog(
    refresh: Option<bool>,
    skills: State<'_, Skills>,
    session: State<'_, CurrentSession>,
) -> Result<Vec<SkillOption>, String> {
    // Reading the catalogue is not an administrative act — an employee may see
    // which skills exist. Binding one to an agent is, and that check is in the
    // registry, where the write happens.
    let _ = require_session(&session)?;
    let snapshot = if refresh.unwrap_or(false) {
        skills.reload()
    } else {
        skills.snapshot()
    };

    let mut out: Vec<SkillOption> = snapshot.cards().into_iter().map(card_option).collect();
    out.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(out)
}

/// Builds the preview. Shared by the command and its tests.
///
/// Takes plain values rather than Tauri state so the whole thing can be
/// exercised without an application — the alternative is a preview that is only
/// ever tested through a mock of itself.
pub fn preview_of(
    definition: &AgentDefinition,
    session: &Session,
    installed_models: &[String],
    installed_skills: &[SkillOption],
    worker_available: bool,
    memory_items_in_scope: usize,
) -> AgentPreview {
    let mut blocked = Vec::new();

    // ── The model ────────────────────────────────────────────────────────
    let preference = definition.models.preference_order();
    let missing: Vec<String> = preference
        .iter()
        .filter(|wanted| !installed_models.iter().any(|known| known == *wanted))
        .cloned()
        .collect();
    let selected = preference
        .iter()
        .find(|wanted| installed_models.iter().any(|known| known == *wanted))
        .cloned();
    // An empty preference order is not a misconfiguration.
    //
    // `ModelBinding.default_model_id: None` means "whatever routing picks for
    // this role", which is what every bundled profile meant before an explicit
    // binding existed. Reporting that as "names no model" would put a blocking
    // error on the most common configuration in the product. What *is* a block
    // is having nothing to route to at all.
    let chosen_by = if preference.is_empty() {
        if installed_models.is_empty() {
            blocked.push(
                "No models are installed, so routing has nothing to give this agent.".to_string(),
            );
        }
        "routing"
    } else {
        if selected.is_none() {
            blocked.push(format!(
                "No model this agent names is installed. It asks for {}{}.",
                preference[0],
                if preference.len() > 1 {
                    format!(" and {} fallback(s)", preference.len() - 1)
                } else {
                    String::new()
                }
            ));
        }
        "binding"
    };

    // ── The skills ───────────────────────────────────────────────────────
    let skills: Vec<SkillResolution> = definition
        .skills
        .iter()
        .map(|binding| {
            match installed_skills
                .iter()
                .find(|option| option.name == binding.name)
            {
                None => SkillResolution {
                    name: binding.name.clone(),
                    pinned_sha256: binding.sha256.clone(),
                    installed_sha256: None,
                    status: "missing".into(),
                    because: format!(
                        "{} is not installed. Install it, or remove it from this agent.",
                        binding.name
                    ),
                },
                Some(option) if !option.available => SkillResolution {
                    name: binding.name.clone(),
                    pinned_sha256: binding.sha256.clone(),
                    installed_sha256: Some(option.sha256.clone()),
                    status: "quarantined".into(),
                    because: option
                        .unavailable_because
                        .clone()
                        .unwrap_or_else(|| format!("{} is quarantined.", binding.name)),
                },
                Some(option) if option.sha256 != binding.sha256 => SkillResolution {
                    name: binding.name.clone(),
                    pinned_sha256: binding.sha256.clone(),
                    installed_sha256: Some(option.sha256.clone()),
                    status: "changed".into(),
                    // The remedy differs from "missing", so the sentence does
                    // too: this is a decision about whether the new bytes are
                    // wanted, not an installation.
                    because: format!(
                        "{} is installed, but its contents changed since this agent was bound \
                         to it. Re-pin it to accept the new version, or reinstall the pinned one.",
                        binding.name
                    ),
                },
                Some(option) => SkillResolution {
                    name: binding.name.clone(),
                    pinned_sha256: binding.sha256.clone(),
                    installed_sha256: Some(option.sha256.clone()),
                    status: "ok".into(),
                    because: format!("Installed at {}, matching the pin.", option.version),
                },
            }
        })
        .collect();
    for unusable in skills.iter().filter(|held| held.status != "ok") {
        blocked.push(unusable.because.clone());
    }

    // ── The policy, computed by the runtime's own narrowing ──────────────
    //
    // The operator's own grant is the ceiling. An agent definition is narrowed
    // against it, never added to it, which is what makes "assignment must not
    // widen permissions" a property of the code rather than a rule somebody
    // remembers to apply.
    let held: Vec<crate::orchestrator::tools::ToolName> = definition
        .allowed_tools
        .iter()
        .chain(definition.denied_tools.iter())
        .cloned()
        .collect();
    let parent = InheritedPolicy::of_run(
        session,
        definition.classification_ceiling,
        PathBuf::from("."),
        &held,
    );
    let profile = definition.to_profile();
    let declared: Vec<String> = definition
        .allowed_tools
        .iter()
        .map(|tool| format!("{tool:?}"))
        .collect();
    let denied: Vec<String> = definition
        .denied_tools
        .iter()
        .map(|tool| format!("{tool:?}"))
        .collect();

    let policy = match parent.narrow_for(&profile, &definition.agent_id) {
        Ok(effective) => {
            for refused in &effective.refused_tools {
                blocked.push(format!(
                    "{refused:?} is named by this agent and is not granted, so anything needing \
                     it will not run."
                ));
            }
            PolicyPreview {
                declared_tools: declared,
                effective_tools: effective.tools.iter().map(|t| format!("{t:?}")).collect(),
                refused_tools: effective
                    .refused_tools
                    .iter()
                    .map(|t| format!("{t:?}"))
                    .collect(),
                denied_tools: denied,
                memory_scope: format!("{:?}", effective.memory_scope),
                isolation: format!("{:?}", effective.isolation),
                write_policy: format!("{:?}", effective.write_policy),
                classification_ceiling: effective.classification_ceiling,
                required_schema: format!("{:?}", effective.required_schema),
                max_turns: effective.limits.max_turns,
                max_output_tokens: effective.limits.max_output_tokens,
                max_duration_seconds: effective.limits.max_duration_seconds,
            }
        }
        Err(refusal) => {
            // The narrowing refused outright. Reported as the block it is, with
            // the declared side still shown so an administrator can see what
            // was asked for.
            blocked.push(refusal.explain());
            PolicyPreview {
                declared_tools: declared.clone(),
                effective_tools: Vec::new(),
                refused_tools: declared,
                denied_tools: denied,
                memory_scope: format!("{:?}", definition.memory.scope),
                isolation: format!("{:?}", definition.isolation),
                write_policy: format!("{:?}", definition.write_policy),
                classification_ceiling: definition.classification_ceiling,
                required_schema: format!("{:?}", definition.output_schema),
                max_turns: definition.limits.max_turns,
                max_output_tokens: definition.limits.max_output_tokens,
                max_duration_seconds: definition.limits.max_duration_seconds,
            }
        }
    };

    // ── The worker, and the state ────────────────────────────────────────
    if !worker_available {
        blocked.push(format!(
            "No worker is registered for the {:?} role, so this agent would accept a task and \
             never perform it.",
            definition.role
        ));
    }
    if !definition.state.is_runnable() {
        blocked.push(format!(
            "This agent is {}, so it will not be given work.",
            definition.state.as_str()
        ));
    }

    AgentPreview {
        agent_id: definition.agent_id.clone(),
        definition_version: definition.definition_version,
        selected_model: selected,
        model_preference: preference,
        missing_models: missing,
        model_chosen_by: chosen_by.to_string(),
        policy,
        skills,
        worker_available,
        memory_items_in_scope,
        blocked,
    }
}

/// A dry run: what would happen, with nothing happening.
///
/// Reads the registries, narrows the policy and counts the memory in scope. It
/// starts no run, writes no memory, claims no worker and emits no event.
#[tauri::command]
pub async fn agent_preview(
    agent_id: String,
    session: State<'_, CurrentSession>,
    agents: State<'_, Agents>,
    skills: State<'_, Skills>,
    subagents: State<'_, Subagents>,
    models: State<'_, Arc<ModelRegistry>>,
) -> Result<AgentPreview, String> {
    let signed_in = require_session(&session)?;
    let definition = agents
        .get(&agent_id, Visibility::of(&signed_in))
        .map_err(|error| error.explain())?;

    let installed_skills: Vec<SkillOption> = skills
        .snapshot()
        .cards()
        .into_iter()
        .map(card_option)
        .collect();
    let installed_models: Vec<String> =
        models.all().iter().map(|entry| entry.id.clone()).collect();
    // The capability its output resolves to, not a name. See `agents::worker_key`.
    let worker_available = crate::commands::agents::has_worker_for(&subagents, &definition);

    Ok(preview_of(
        &definition,
        &signed_in,
        &installed_models,
        &installed_skills,
        worker_available,
        0,
    ))
}

/// What would be affected by archiving or disabling this agent.
///
/// ## Why this reads the event log rather than a field
///
/// A run does not record which registry agent it used: `TaskSnapshot` carries
/// the model, the actor and the prompt, and nothing on it links to an agent id.
/// What *is* durable is the `subagent_started` event, which names the profile
/// the child was spawned from and is written against the parent run. So a
/// dependent is derived the only honest way available — by finding children of
/// this agent's profile that started and have not stopped.
///
/// The alternative was to report "no dependents" because no field says
/// otherwise, which is the reassuring answer rather than the true one.
#[tauri::command]
pub async fn agent_dependents(
    agent_id: String,
    session: State<'_, CurrentSession>,
    agents: State<'_, Agents>,
    events: State<'_, crate::commands::agent::TaskEvents>,
) -> Result<AgentDependents, String> {
    let signed_in = require_session(&session)?;
    let definition = agents
        .get(&agent_id, Visibility::of(&signed_in))
        .map_err(|error| error.explain())?;
    // Its id, and for an imported agent the bundled role name older children
    // recorded. It used to be the role name alone -- and a child dispatched
    // through the registry records the agent's `ag-` id, so every child of a
    // registry agent was invisible here and archiving one reported "no
    // dependents" with work in flight.
    let keys = crate::commands::agents::trace_keys(&definition);

    let mut active_runs = Vec::new();
    for snapshot in events.running()? {
        // From the start of the run, because a child spawned before any cursor
        // we could hold is exactly the one that matters.
        let page = events.events_since(&snapshot.run_id, 0)?;
        let mut live: Vec<String> = Vec::new();
        for event in &page.events {
            let Some(child) = event
                .payload
                .get("childId")
                .and_then(|value| value.as_str())
                .map(str::to_string)
            else {
                continue;
            };
            match event.event_type {
                TaskEventType::SubagentStarted => {
                    let names_us = ["agentId", "profile"].iter().any(|field| {
                        event
                            .payload
                            .get(*field)
                            .and_then(|value| value.as_str())
                            .is_some_and(|named| keys.iter().any(|key| key == named))
                    });
                    if names_us {
                        live.push(child);
                    }
                }
                // A stop removes it whatever profile the stop row names: the
                // child id is the identity, and a stopped child is not a
                // dependency whether or not the two rows agree.
                TaskEventType::SubagentStopped => live.retain(|held| held != &child),
                _ => {}
            }
        }
        if !live.is_empty() {
            active_runs.push(DependentRun {
                run_id: snapshot.run_id.clone(),
                prompt: snapshot.prompt.clone(),
                actor: snapshot.actor.clone(),
                state: format!("{:?}", snapshot.state),
                children: live,
            });
        }
    }

    let mut blocking = Vec::new();
    if !active_runs.is_empty() {
        // Not a refusal. Archiving is safe with work in flight — a run keeps
        // the definition version it pinned and finishes on it — but somebody
        // deciding deserves to know it is happening.
        blocking.push(format!(
            "{} task(s) are running a child of this agent right now. They keep the version they              pinned and will finish; no new work will be given to it.",
            active_runs.len()
        ));
    }

    Ok(AgentDependents {
        agent_id,
        active_runs,
        memory_items: 0,
        blocking,
    })
}

/// What a test run did.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TestRunOutcome {
    pub agent_id: String,
    /// The run this was recorded against. A real id with a real event trail —
    /// a test run is auditable like any other.
    pub run_id: String,
    pub child_id: String,
    /// True when a child actually executed, whatever it then concluded.
    ///
    /// Separate from `status` on purpose. A mechanical worker given a harmless
    /// objective with nothing to chew on reports `Failed` — "neither the task
    /// nor the shared memory offered a figure to check" — which is the correct
    /// answer to that objective and the *wrong* thing to show an administrator
    /// as a health signal. The question this screen asks is "does this agent
    /// run", and that is this field; what it concluded is the next one.
    pub ran: bool,
    /// `Completed`, `Refused`, `Failed`, and so on, from the manager.
    pub status: String,
    /// What the agent actually produced, in short.
    pub findings: Vec<String>,
    /// What it could not establish. Named individually.
    pub uncertainty: Vec<String>,
    pub turns_used: u32,
    /// Present when it did not complete.
    pub detail: Option<String>,
    /// True when an identical test had already been run and this is its result.
    pub reused: bool,
}

/// Actually runs the agent once, on a fixed, harmless objective.
///
/// ## Why this is a real run and not a simulation
///
/// [`agent_preview`] answers "would this be allowed", which is a different
/// question from "does this work". The failures an administrator most wants to
/// catch before trusting an agent — a worker that registers but errors, a
/// policy that narrows to nothing usable, a schema the worker cannot produce —
/// all survive a preview and only appear when something runs.
///
/// So this spawns a child through the same [`crate::subagents::SubagentManager`]
/// path production uses, with the same narrowing, the same idempotency key and
/// the same event trail. It is cheap because four of the five shipped roles are
/// mechanical — a search, a calculation, a read, a re-open — and need no model
/// loaded; `code-worker` is the exception and says so rather than hanging.
///
/// ## What it costs
///
/// A real run id, a real workspace directory and real events. That is the
/// point: a test that wrote nothing down could not be audited afterwards, and
/// "somebody tested this agent on Tuesday" is exactly the record an operator
/// wants when an agent later misbehaves.
#[tauri::command]
pub async fn agent_test_run(
    agent_id: String,
    app: tauri::AppHandle,
    session: State<'_, CurrentSession>,
    agents: State<'_, Agents>,
    subagents: State<'_, Subagents>,
) -> Result<TestRunOutcome, String> {
    use crate::subagents::manager::Dispatch;

    let signed_in = require_session(&session)?;
    // Running an agent on demand is an administrative act: it writes events,
    // takes a worker slot and may touch the task's shared memory.
    if !crate::agents::store::AgentRegistry::may_administer(&signed_in) {
        return Err("Only an administrator may run an agent from this screen.".to_string());
    }

    let definition = agents
        .get(&agent_id, Visibility::of(&signed_in))
        .map_err(|error| error.explain())?;
    // By capability, as dispatch finds it -- and a contract with no worker in
    // this build (document, deck, workbook) is said so rather than attempted.
    if !crate::commands::agents::has_worker_for(&subagents, &definition) {
        return Err(format!(
            "No worker in this build produces a {} result, so there is nothing to run. This \
             agent is registered and cannot yet be run.",
            definition.output_schema.as_str()
        ));
    }

    // Its own run id, so the events belong to something a person can look up
    // rather than being attributed to whatever ran last.
    let run_id = format!("agent-test-{}", uuid::Uuid::new_v4());
    let data_dir = app
        .path()
        .app_data_dir()
        .map_err(|error| format!("the application data directory could not be resolved: {error}"))?;
    let workspace = data_dir.join("runs").join(&run_id);
    std::fs::create_dir_all(&workspace)
        .map_err(|error| format!("the test run's workspace could not be created: {error}"))?;

    let inherited = InheritedPolicy::of_run(
        &signed_in,
        definition.classification_ceiling,
        workspace,
        &definition.allowed_tools,
    );

    // The model the agent would actually use. Named rather than chosen here:
    // picking one would be a second opinion about routing.
    let model = crate::subagents::certification::Decision {
        model_id: definition
            .models
            .default_model_id
            .clone()
            .unwrap_or_default(),
        role: definition.role,
        cheaper_than_parent: false,
        reason: "named by the agent's own binding for this test".to_string(),
        tier: None,
        score: None,
    };

    let dispatch = Dispatch::for_task(&agent_id, &run_id);
    let objective =
        "Report your configuration and confirm you can run. Do not read or change anything.";

    // Dispatched by this agent's own id, so the manager resolves *this* agent
    // at its current version. It was dispatched by a name, which for a clone
    // was its display name -- a key the registry rightly refuses -- so a clone
    // could never be test-run.
    match subagents
        .spawn(&agent_id, &inherited, objective, Vec::new(), model, &dispatch)
        .await
    {
        Ok(spawned) => {
            let reused = spawned.is_reused();
            let result = spawned.result();
            Ok(TestRunOutcome {
                agent_id,
                ran: true,
                run_id,
                child_id: result.child_id.clone(),
                status: format!("{:?}", result.status),
                findings: result
                    .findings
                    .iter()
                    .map(|finding| format!("{finding:?}"))
                    .collect(),
                uncertainty: result.uncertainty.clone(),
                turns_used: result.turns_used,
                detail: result.detail.clone(),
                reused,
            })
        }
        // A refusal is the useful answer, not an error to swallow: it names
        // which rule stopped the run.
        Err(refusal) => Err(refusal.explain()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    use crate::agents::{AgentState, SkillBinding};
    use crate::identity::{Role, User};

    fn admin() -> Session {
        Session::open(User::new("priya", "priya", vec![Role::Administrator]))
    }

    fn skill(name: &str, sha: &str, available: bool) -> SkillOption {
        SkillOption {
            name: name.into(),
            description: "a skill".into(),
            version: "1.0".into(),
            sha256: sha.into(),
            classification: Classification::Internal,
            required_tools: Vec::new(),
            approval_class: "None".into(),
            network: "None".into(),
            available,
            unavailable_because: if available {
                None
            } else {
                Some("it is quarantined".into())
            },
            imported: false,
        }
    }

    /// The headline rule: a definition that parsed is not an agent that runs.
    #[test]
    fn a_parseable_agent_with_no_model_installed_is_blocked_and_says_why() {
        let mut definition = crate::agents::tests::definition("ag-test");
        definition.models.default_model_id = Some("qwen2.5-7b".into());
        let preview = preview_of(&definition, &admin(), &[], &[], true, 0);

        assert!(preview.selected_model.is_none());
        assert!(!preview.missing_models.is_empty());
        assert!(
            preview.blocked.iter().any(|why| why.contains("model")),
            "no sentence explained the missing model: {:?}",
            preview.blocked
        );
    }

    #[test]
    fn a_role_with_no_worker_is_named_rather_than_reported_ready() {
        let definition = crate::agents::tests::definition("ag-test");
        let models = definition.models.preference_order();
        let preview = preview_of(&definition, &admin(), &models, &[], false, 0);

        assert!(!preview.worker_available);
        assert!(
            preview.blocked.iter().any(|why| why.contains("worker")),
            "an agent with no worker was not flagged: {:?}",
            preview.blocked
        );
    }

    #[test]
    fn everything_resolving_leaves_nothing_blocked() {
        let mut definition = crate::agents::tests::definition("ag-test");
        definition.skills.clear();
        definition.models.default_model_id = Some("qwen2.5-7b".into());
        let models = definition.models.preference_order();
        let preview = preview_of(&definition, &admin(), &models, &[], true, 0);

        assert_eq!(preview.blocked, Vec::<String>::new());
        assert_eq!(preview.selected_model, models.first().cloned());
    }

    /// A skill whose bytes moved is a different problem from a missing one, and
    /// the remedy is different, so the sentence has to be different too.
    #[test]
    fn a_changed_skill_is_distinguished_from_a_missing_one() {
        let mut definition = crate::agents::tests::definition("ag-test");
        definition.skills = vec![SkillBinding {
            name: "writing".into(),
            version: "1.0".into(),
            sha256: "a".repeat(64),
        }];
        let models = definition.models.preference_order();

        let missing = preview_of(&definition, &admin(), &models, &[], true, 0);
        assert_eq!(missing.skills[0].status, "missing");

        let installed = vec![skill("writing", &"b".repeat(64), true)];
        let changed = preview_of(&definition, &admin(), &models, &installed, true, 0);
        assert_eq!(changed.skills[0].status, "changed");
        assert!(changed.skills[0].because.contains("changed"));

        let pinned = vec![skill("writing", &"a".repeat(64), true)];
        let fine = preview_of(&definition, &admin(), &models, &pinned, true, 0);
        assert_eq!(fine.skills[0].status, "ok");
    }

    #[test]
    fn a_quarantined_skill_blocks_and_carries_its_reason() {
        let mut definition = crate::agents::tests::definition("ag-test");
        definition.skills = vec![SkillBinding {
            name: "writing".into(),
            version: "1.0".into(),
            sha256: "a".repeat(64),
        }];
        let models = definition.models.preference_order();
        let installed = vec![skill("writing", &"a".repeat(64), false)];

        let preview = preview_of(&definition, &admin(), &models, &installed, true, 0);
        assert_eq!(preview.skills[0].status, "quarantined");
        assert!(preview
            .blocked
            .iter()
            .any(|why| why.contains("quarantined")));
    }

    /// The property that makes assignment safe: the effective set is a subset
    /// of the declared one, always.
    #[test]
    fn the_effective_tool_set_is_never_wider_than_the_declared_one() {
        let definition = crate::agents::tests::definition("ag-test");
        let models = definition.models.preference_order();
        let preview = preview_of(&definition, &admin(), &models, &[], true, 0);

        let declared: BTreeSet<&String> = preview.policy.declared_tools.iter().collect();
        for granted in &preview.policy.effective_tools {
            assert!(
                declared.contains(granted),
                "{granted} was granted and never declared"
            );
        }
    }

    /// A denied tool beats an allowed one, and the preview shows it rather than
    /// letting a reader work it out from two lists.
    #[test]
    fn a_denied_tool_is_reported_and_never_granted() {
        let mut definition = crate::agents::tests::definition("ag-test");
        let Some(first) = definition.allowed_tools.first().cloned() else {
            return;
        };
        definition.denied_tools = vec![first.clone()];
        let models = definition.models.preference_order();
        let preview = preview_of(&definition, &admin(), &models, &[], true, 0);

        let denied = format!("{first:?}");
        assert!(preview.policy.denied_tools.contains(&denied));
        assert!(
            !preview.policy.effective_tools.contains(&denied),
            "a denied tool was still granted"
        );
    }

    /// Leaving the model to routing is the ordinary configuration, not a fault.
    #[test]
    fn an_agent_that_leaves_the_model_to_routing_is_not_blocked_for_it() {
        let definition = crate::agents::tests::definition("ag-test");
        assert!(definition.models.preference_order().is_empty());

        let preview = preview_of(&definition, &admin(), &["qwen2.5-7b".into()], &[], true, 0);
        assert_eq!(preview.model_chosen_by, "routing");
        assert!(
            !preview.blocked.iter().any(|why| why.contains("model")),
            "routing-chosen model was reported as a fault: {:?}",
            preview.blocked
        );
    }

    /// With nothing installed at all, routing has nothing to give.
    #[test]
    fn no_models_installed_at_all_is_a_block() {
        let definition = crate::agents::tests::definition("ag-test");
        let preview = preview_of(&definition, &admin(), &[], &[], true, 0);
        assert!(preview.blocked.iter().any(|why| why.contains("No models")));
    }

    #[test]
    fn a_disabled_agent_says_so_rather_than_appearing_runnable() {
        let mut definition = crate::agents::tests::definition("ag-test");
        definition.state = AgentState::Disabled;
        let models = definition.models.preference_order();
        let preview = preview_of(&definition, &admin(), &models, &[], true, 0);

        assert!(preview.blocked.iter().any(|why| why.contains("disabled")));
    }

    /// A preview is a question, not an act. Nothing it returns depends on
    /// having started anything, and running it twice gives the same answer.
    #[test]
    fn previewing_twice_gives_the_same_answer() {
        let definition = crate::agents::tests::definition("ag-test");
        let models = definition.models.preference_order();
        let first = preview_of(&definition, &admin(), &models, &[], true, 0);
        let second = preview_of(&definition, &admin(), &models, &[], true, 0);
        assert_eq!(first.blocked, second.blocked);
        assert_eq!(first.selected_model, second.selected_model);
        assert_eq!(first.definition_version, second.definition_version);
    }
}

/// One job the orchestrator dispatched, as the Agents page shows it.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct OrchestratorJobView {
    #[serde(flatten)]
    pub job: crate::agent_runtime::events::plans::JobRecord,
    /// The step's status in the newest plan version, and its note.
    pub step_status: Option<String>,
    pub step_note: Option<String>,
    pub plan_version: Option<u32>,
}

/// The coordinator's recent delegated jobs (P05).
///
/// Read from the durable job table and each run's newest plan version, so what
/// the page shows is what the backend recorded -- the definition version each
/// job was pinned to, its lease decision, its settled status -- and not a
/// progress line. Scoped to runs the signed-in person started; an
/// administrator sees every run's jobs, as the Tasks screen does.
#[tauri::command]
pub async fn agent_orchestrator_jobs(
    limit: Option<u32>,
    session: State<'_, CurrentSession>,
    events: State<'_, crate::commands::agent::TaskEvents>,
) -> Result<Vec<OrchestratorJobView>, String> {
    let signed_in = require_session(&session)?;
    let administrator = signed_in.user.roles.contains(&crate::identity::Role::Administrator);
    let limit = limit.unwrap_or(50).clamp(1, 200) as usize;
    let mut plans: std::collections::HashMap<String, Option<crate::agent_runtime::task_plan::TaskPlan>> =
        std::collections::HashMap::new();
    let mut owners: std::collections::HashMap<String, Option<String>> = std::collections::HashMap::new();
    let mut out = Vec::new();
    for job in events.recent_jobs(limit * 2)? {
        let owner = owners
            .entry(job.run_id.clone())
            .or_insert_with(|| events.snapshot(&job.run_id).ok().flatten().map(|snapshot| snapshot.actor))
            .clone();
        if !administrator && owner.as_deref() != Some(signed_in.user.id.as_str()) {
            continue;
        }
        let plan = plans
            .entry(job.run_id.clone())
            .or_insert_with(|| events.latest_plan(&job.run_id).ok().flatten())
            .clone();
        let step = plan.as_ref().and_then(|plan| plan.step(&job.step_id).cloned());
        out.push(OrchestratorJobView {
            step_status: step.as_ref().map(|step| step.status.as_str().to_string()),
            step_note: step.and_then(|step| step.note),
            plan_version: plan.map(|plan| plan.version),
            job,
        });
        if out.len() >= limit {
            break;
        }
    }
    Ok(out)
}
