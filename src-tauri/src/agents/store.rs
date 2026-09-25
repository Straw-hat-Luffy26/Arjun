//! Where agent definitions live, and the rules for changing one.
//!
//! ## Why this is in the data directory and not beside the profiles
//!
//! The bundled profiles are in the installer's resource directory. On Windows
//! that is under `Program Files`, which an ordinary user cannot write and which
//! the next upgrade replaces wholesale. Writing an agent's colour there would
//! mean configuration that needs administrator rights to change and is silently
//! discarded when the application updates.
//!
//! So the profiles stay read-only and shipped, and everything a deployment
//! decides lives in `<appdata>/agents/registry.json`, which belongs to the
//! deployment. The link between the two is [`super::ImportOrigin`].
//!
//! ## The import is idempotent because it is keyed on the profile, not the file
//!
//! Start-up imports every bundled profile. Running that a second time must not
//! produce a second copy of each agent, and — more importantly — must not
//! discard the colour, the model binding or the enabled state somebody chose.
//!
//! So an import is keyed on `imported_from.profile_name`. A profile already
//! mapped to an agent updates only the fields the *profile* owns (its
//! description, its tool requests, its limits) and leaves the fields the
//! *deployment* owns alone. A profile whose hash has not changed is not even
//! that: it is a no-op, and the version number does not move.
//!
//! ## Why every mutation takes an expected version
//!
//! Two administrators with the same screen open is not a rare case; it is the
//! ordinary case for a screen that lists things. Last-writer-wins on a record
//! that decides what an agent may do means one person's tool restriction
//! disappears without either of them seeing anything. An expected version turns
//! that into a refusal somebody can answer.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use serde::{Deserialize, Serialize};

use super::{
    AgentDefinition, AgentState, ImportOrigin, InvalidDefinition, MemoryPolicy, ModelBinding,
    PinnedDefinition, AGENT_PALETTE, AGENT_SCHEMA_VERSION,
};
use crate::identity::{Role, Session};
use crate::subagents::profile::AgentProfile;

/// The file as it is written.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct StoredRegistry {
    schema_version: u32,
    agents: Vec<AgentDefinition>,
}

/// Why an operation on the registry did not happen.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "error", rename_all = "camelCase")]
pub enum RegistryError {
    /// The caller is not an administrator.
    ///
    /// Checked in Rust, on every mutation, rather than by hiding a menu. A
    /// hidden menu is a hidden menu; the command is still there and still
    /// reachable by anything that can make an IPC call.
    NotAdministrator { action: String },
    /// No agent with that id.
    ///
    /// The same answer for "does not exist" and "you may not see it", so a
    /// refusal cannot be used to discover which ids are real.
    NoSuchAgent { agent_id: String },
    /// Somebody else changed it first.
    VersionConflict {
        agent_id: String,
        expected: u64,
        actual: u64,
    },
    /// The definition itself is not acceptable.
    Invalid { detail: InvalidDefinition },
    /// The stored file was written by a build this one does not understand.
    UnknownSchema { found: u32, understood: u32 },
    /// The file could not be read or written.
    Storage { detail: String },
    /// An archived or disabled agent cannot be given work.
    NotRunnable { agent_id: String, state: String },
    /// An edit tried to change which models this agent uses.
    ///
    /// ## Why this is refused rather than applied
    ///
    /// Because it was applied, and that was the defect. `models` sat in this
    /// form beside the agent's colour and description, so reassigning a model
    /// was one write: the next turn routed elsewhere, the conversation carried
    /// on, and nothing had checked that the new model could serve the window
    /// the previous turn was budgeted against, that it could call the tools the
    /// agent is bound to, or that whatever was in flight had settled first.
    ///
    /// A model change is a handoff with states, a checkpoint and a rollback —
    /// see [`crate::agent_runtime::model_transition`]. This form keeps every
    /// other editable field and sends that one through the handoff.
    ModelBindingNeedsTransition { agent_id: String },
}

impl RegistryError {
    pub fn explain(&self) -> String {
        match self {
            Self::NotAdministrator { action } => format!(
                "Only an administrator may {action}. Agents decide what a model is allowed to do \
                 on this machine, so changing one is an administrative act."
            ),
            Self::NoSuchAgent { agent_id } => {
                format!("There is no agent {agent_id:?} available to you.")
            }
            Self::VersionConflict {
                agent_id,
                expected,
                actual,
            } => format!(
                "{agent_id} has been changed since this form was opened (you have version \
                 {expected}; it is now {actual}). Reload it and make the change again — saving \
                 this would silently undo whatever the other edit did."
            ),
            Self::Invalid { detail } => detail.explain(),
            Self::UnknownSchema { found, understood } => format!(
                "The agent registry was written in format {found}; this build reads {understood}. \
                 It has not been changed, because acting on a record this build only partly \
                 understands is how an agent ends up with permissions nobody granted."
            ),
            Self::Storage { detail } => {
                format!("The agent registry could not be read or written: {detail}")
            }
            Self::NotRunnable { agent_id, state } => format!(
                "{agent_id} is {state}. Its history still names it, and it cannot be given work \
                 until it is enabled again."
            ),
            Self::ModelBindingNeedsTransition { agent_id } => format!(
                "This form does not change which model {agent_id} runs on. Moving an agent to a \
                 different model has to stop its work at a safe point, settle anything in \
                 flight, save its state, check the new model can actually hold the task, and be \
                 able to undo itself — none of which a saved form can do. Every other change you \
                 made can be saved; use the model change action for that one."
            ),
        }
    }
}

/// Who is asking, and therefore what they may see.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Visibility {
    /// Everything, including disabled and archived agents.
    Administrator,
    /// Only agents that can actually be given work.
    Operator,
}

impl Visibility {
    pub fn of(session: &Session) -> Self {
        if session.user.roles.contains(&Role::Administrator) {
            Visibility::Administrator
        } else {
            Visibility::Operator
        }
    }
}

/// What an accepted mutation did, for the audit line and the caller.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Mutation {
    pub agent_id: String,
    /// The version the record is at now.
    pub definition_version: u64,
    /// One of `created`, `updated`, `cloned`, `enabled`, `disabled`,
    /// `archived`, `imported`.
    pub action: String,
    /// True when nothing changed — a re-import of an unchanged profile, or a
    /// state change to the state it already holds. Reported rather than
    /// silently treated as a write, so an audit log does not fill with no-op
    /// edits.
    pub unchanged: bool,
}

/// The registry, held in memory and mirrored to one file.
#[derive(Debug)]
pub struct AgentRegistry {
    path: PathBuf,
    agents: Mutex<Vec<AgentDefinition>>,
}

impl AgentRegistry {
    /// Opens the registry beside the rest of the application data.
    ///
    /// A missing file is an empty registry, not a failure: it is what a fresh
    /// installation has, and the import that follows start-up fills it.
    pub fn open(app_data_dir: &Path) -> Result<Self, RegistryError> {
        let dir = app_data_dir.join("agents");
        std::fs::create_dir_all(&dir).map_err(|error| RegistryError::Storage {
            detail: format!("could not create {}: {error}", dir.display()),
        })?;
        let path = dir.join("registry.json");

        let agents = match std::fs::read_to_string(&path) {
            Ok(text) => {
                let stored: StoredRegistry =
                    serde_json::from_str(&text).map_err(|error| RegistryError::Storage {
                        detail: format!("{} could not be parsed: {error}", path.display()),
                    })?;
                if stored.schema_version != AGENT_SCHEMA_VERSION {
                    return Err(RegistryError::UnknownSchema {
                        found: stored.schema_version,
                        understood: AGENT_SCHEMA_VERSION,
                    });
                }
                stored.agents
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Vec::new(),
            Err(error) => {
                return Err(RegistryError::Storage {
                    detail: error.to_string(),
                })
            }
        };

        Ok(Self {
            path,
            agents: Mutex::new(agents),
        })
    }

    fn held(&self) -> Result<std::sync::MutexGuard<'_, Vec<AgentDefinition>>, RegistryError> {
        self.agents.lock().map_err(|_| RegistryError::Storage {
            detail: "the agent registry was left locked by a failed write".into(),
        })
    }

    /// Writes the whole file, atomically.
    ///
    /// Through a temporary file and a rename, because a half-written registry is
    /// a deployment whose agents have no rules.
    fn persist(&self, agents: &[AgentDefinition]) -> Result<(), RegistryError> {
        if self.path.as_os_str().is_empty() {
            return Ok(());
        }
        let stored = StoredRegistry {
            schema_version: AGENT_SCHEMA_VERSION,
            agents: agents.to_vec(),
        };
        let text =
            serde_json::to_string_pretty(&stored).map_err(|error| RegistryError::Storage {
                detail: error.to_string(),
            })?;
        let temporary = self.path.with_extension("json.tmp");
        std::fs::write(&temporary, text).map_err(|error| RegistryError::Storage {
            detail: format!("could not write {}: {error}", temporary.display()),
        })?;
        std::fs::rename(&temporary, &self.path).map_err(|error| RegistryError::Storage {
            detail: format!("could not replace {}: {error}", self.path.display()),
        })
    }

    // ── Reading ──────────────────────────────────────────────────────────

    /// Every agent this caller may see, by display name.
    pub fn list(&self, visibility: Visibility) -> Result<Vec<AgentDefinition>, RegistryError> {
        let held = self.held()?;
        let mut out: Vec<AgentDefinition> = held
            .iter()
            .filter(|agent| match visibility {
                Visibility::Administrator => true,
                Visibility::Operator => agent.state.is_visible_to_operators(),
            })
            .cloned()
            .collect();
        out.sort_by(|a, b| a.display_name.cmp(&b.display_name));
        Ok(out)
    }

    /// One agent, if this caller may see it.
    pub fn get(
        &self,
        agent_id: &str,
        visibility: Visibility,
    ) -> Result<AgentDefinition, RegistryError> {
        let held = self.held()?;
        held.iter()
            .find(|agent| agent.agent_id == agent_id)
            .filter(|agent| match visibility {
                Visibility::Administrator => true,
                Visibility::Operator => agent.state.is_visible_to_operators(),
            })
            .cloned()
            .ok_or_else(|| RegistryError::NoSuchAgent {
                agent_id: agent_id.to_string(),
            })
    }

    /// One agent whatever its state, for attributing history.
    ///
    /// ## Why archived agents stay resolvable
    ///
    /// Because the graph, the task record and the audit log all name the agent
    /// that did the work, and an archived agent that could not be resolved
    /// would turn every one of those records into an id nobody can read. An
    /// agent is retired from *being given work*; it is not retired from having
    /// done it.
    pub fn resolve_for_provenance(&self, agent_id: &str) -> Result<AgentDefinition, RegistryError> {
        let held = self.held()?;
        held.iter()
            .find(|agent| agent.agent_id == agent_id)
            .cloned()
            .ok_or_else(|| RegistryError::NoSuchAgent {
                agent_id: agent_id.to_string(),
            })
    }

    /// The definition a starting run should pin, or why it cannot.
    pub fn pin_for_run(&self, agent_id: &str) -> Result<PinnedDefinition, RegistryError> {
        let agent = self.resolve_for_provenance(agent_id)?;
        if !agent.state.is_runnable() {
            return Err(RegistryError::NotRunnable {
                agent_id: agent.agent_id,
                state: agent.state.as_str().to_string(),
            });
        }
        Ok(agent.pin())
    }

    // ── Writing ──────────────────────────────────────────────────────────

    /// The same gate, asked in advance.
    ///
    /// Exists for one caller: [`crate::agent_runtime::model_handoff`], which
    /// would otherwise drain a run, take a checkpoint and load several
    /// gigabytes of weights before [`Self::rebind_model`] refused the write.
    /// The refusal that matters is still the one on the mutation — this is the
    /// same predicate asked early so the expensive work is not done for
    /// somebody who may not have it done.
    ///
    /// Deliberately the same function rather than a second reading of the
    /// roles: a second opinion about authority is the one that eventually
    /// disagrees.
    pub fn may_administer(session: &Session) -> bool {
        Self::require_administrator(session, "").is_ok()
    }

    /// The gate every mutation passes.
    fn require_administrator(session: &Session, action: &str) -> Result<(), RegistryError> {
        if session.user.roles.contains(&Role::Administrator) {
            Ok(())
        } else {
            Err(RegistryError::NotAdministrator {
                action: action.to_string(),
            })
        }
    }

    /// Adds an agent. The id is minted here, never taken from the caller.
    pub fn create(
        &self,
        session: &Session,
        mut definition: AgentDefinition,
    ) -> Result<Mutation, RegistryError> {
        Self::require_administrator(session, "create an agent")?;

        // Minted here so a caller cannot choose an id that collides with an
        // existing agent's history, or re-use one that was archived.
        definition.agent_id = format!("ag-{}", uuid::Uuid::new_v4());
        definition.definition_version = 1;
        let now = chrono::Utc::now().to_rfc3339();
        definition.created_at = now.clone();
        definition.updated_at = now;
        definition
            .validate()
            .map_err(|detail| RegistryError::Invalid { detail })?;

        let mut held = self.held()?;
        held.push(definition.clone());
        self.persist(&held)?;

        Ok(Mutation {
            agent_id: definition.agent_id,
            definition_version: 1,
            action: "created".into(),
            unchanged: false,
        })
    }

    /// Replaces an agent's mutable fields, if nobody else has changed it first.
    ///
    /// The id, the creation time and the import origin are taken from the
    /// stored record rather than from the caller: they are not editable, and
    /// accepting them from a form would make them editable by anything that can
    /// post one.
    pub fn update(
        &self,
        session: &Session,
        agent_id: &str,
        expected_version: u64,
        edited: AgentDefinition,
    ) -> Result<Mutation, RegistryError> {
        Self::require_administrator(session, "change an agent")?;

        let mut held = self.held()?;
        let Some(position) = held.iter().position(|agent| agent.agent_id == agent_id) else {
            return Err(RegistryError::NoSuchAgent {
                agent_id: agent_id.to_string(),
            });
        };
        if held[position].definition_version != expected_version {
            return Err(RegistryError::VersionConflict {
                agent_id: agent_id.to_string(),
                expected: expected_version,
                actual: held[position].definition_version,
            });
        }

        let current = &held[position];

        // The model binding is not editable through this form.
        //
        // It used to be, and that was the whole of what "changing an agent's
        // model" meant: one field among the colour and the description, saved
        // in one write, with nothing drained, nothing checked and nothing
        // recorded. See `RegistryError::ModelBindingNeedsTransition`, and
        // `rebind_model` below for the only path that may write this field.
        //
        // Refused rather than ignored. Silently dropping the submitted binding
        // would let an administrator save a form, see no error, and believe the
        // agent had moved.
        if !current.models.names_same_models(&edited.models) {
            return Err(RegistryError::ModelBindingNeedsTransition {
                agent_id: agent_id.to_string(),
            });
        }

        let next = AgentDefinition {
            // Not editable. A rename or a model reassignment keeps the identity
            // and therefore keeps the memory and the history attached to it.
            agent_id: current.agent_id.clone(),
            definition_version: current.definition_version + 1,
            created_at: current.created_at.clone(),
            imported_from: current.imported_from.clone(),
            updated_at: chrono::Utc::now().to_rfc3339(),
            ..edited
        };
        next.validate()
            .map_err(|detail| RegistryError::Invalid { detail })?;

        let version = next.definition_version;
        held[position] = next;
        self.persist(&held)?;

        Ok(Mutation {
            agent_id: agent_id.to_string(),
            definition_version: version,
            action: "updated".into(),
            unchanged: false,
        })
    }

    /// Writes the one field [`Self::update`] refuses: which model this agent
    /// runs on.
    ///
    /// ## Why this is a separate method, and why it takes a transition id
    ///
    /// Because this is the commit point of a handoff, not an edit. Everything
    /// that makes a model change safe — draining the run, settling its effects,
    /// checkpointing, validating the target, loading it, recompiling the
    /// context for the window it actually came up with — happens in
    /// [`crate::agent_runtime::model_handoff`] before this is called. By the
    /// time it is called, the only thing left is the write.
    ///
    /// The transition id is carried so the write is attributable to the handoff
    /// that earned it. A `rebind_model` reachable without one would be the old
    /// defect with a longer name.
    ///
    /// ## What it deliberately does not touch
    ///
    /// The agent id, the memory policy, the skills, the tools, the created time
    /// and the import origin. A model is a thing an agent currently uses; the
    /// agent is what remembers and what history is attributed to. Those are
    /// separate identities and this changes exactly one of them.
    ///
    /// The version moves, like any accepted change, so a run that pinned the
    /// previous definition can tell that it is not the current one — and so the
    /// handoff's own record can say which version the registry will be at when
    /// the write lands. See
    /// [`crate::agent_runtime::model_transition::PendingCommit::reconcile`],
    /// which is how a crash between the two stores is read.
    pub fn rebind_model(
        &self,
        session: &Session,
        agent_id: &str,
        expected_version: u64,
        binding: ModelBinding,
        transition_id: &str,
    ) -> Result<Mutation, RegistryError> {
        Self::require_administrator(session, "change the model an agent runs on")?;

        let mut held = self.held()?;
        let Some(position) = held.iter().position(|agent| agent.agent_id == agent_id) else {
            return Err(RegistryError::NoSuchAgent {
                agent_id: agent_id.to_string(),
            });
        };
        // The same guard an edit passes, and here it is the one that makes
        // concurrent administration safe: a handoff that validated a target
        // against version 4 must not commit onto version 5, because the edit
        // that produced 5 may have narrowed the tools or the classification
        // ceiling the target was checked against.
        if held[position].definition_version != expected_version {
            return Err(RegistryError::VersionConflict {
                agent_id: agent_id.to_string(),
                expected: expected_version,
                actual: held[position].definition_version,
            });
        }

        // Already there. Reported as a no-op rather than written, so a retried
        // commit — the recovery path re-running a handoff whose write landed
        // before the crash — does not move the version a second time.
        if held[position].models.names_same_models(&binding) {
            return Ok(Mutation {
                agent_id: agent_id.to_string(),
                definition_version: held[position].definition_version,
                action: "rebound".into(),
                unchanged: true,
            });
        }

        let mut next = held[position].clone();
        next.models = binding;
        next.definition_version += 1;
        next.updated_at = chrono::Utc::now().to_rfc3339();
        // Checked even though only one field moved: a binding whose default is
        // not in a non-empty eligible set would be an agent that cannot route,
        // and the validation in `model_transition` is the caller's rather than
        // this one's.
        next.validate()
            .map_err(|detail| RegistryError::Invalid { detail })?;

        let version = next.definition_version;
        held[position] = next;
        self.persist(&held)?;

        log::info!(
            "[agents] {agent_id} is now on {} at version {version}, committed by handoff \
             {transition_id}",
            held[position]
                .models
                .default_model_id
                .as_deref()
                .unwrap_or("whatever routing picks for its role")
        );

        Ok(Mutation {
            agent_id: agent_id.to_string(),
            definition_version: version,
            action: "rebound".into(),
            unchanged: false,
        })
    }

    /// Copies an agent's configuration under a new identity.
    ///
    /// ## What a clone deliberately does not carry
    ///
    /// Its private memory, and its history. Memory is keyed by `agent_id` and
    /// the clone has a new one, so the copy starts empty by construction rather
    /// than by a deletion somebody has to remember to do. That is the safe
    /// direction: a clone made to try a different model would otherwise inherit
    /// conclusions the original reached under the old one.
    ///
    /// The import origin is not carried either. A clone is a new agent this
    /// deployment made, not a second mapping for the same bundled profile —
    /// carrying it would make the next import ambiguous about which of the two
    /// it owns.
    pub fn clone_agent(
        &self,
        session: &Session,
        agent_id: &str,
        display_name: &str,
    ) -> Result<Mutation, RegistryError> {
        Self::require_administrator(session, "clone an agent")?;

        let source = self.resolve_for_provenance(agent_id)?;
        let now = chrono::Utc::now().to_rfc3339();
        let copy = AgentDefinition {
            agent_id: format!("ag-{}", uuid::Uuid::new_v4()),
            definition_version: 1,
            display_name: display_name.trim().to_string(),
            imported_from: None,
            created_at: now.clone(),
            updated_at: now,
            ..source
        };
        copy.validate()
            .map_err(|detail| RegistryError::Invalid { detail })?;

        let mut held = self.held()?;
        held.push(copy.clone());
        self.persist(&held)?;

        Ok(Mutation {
            agent_id: copy.agent_id,
            definition_version: 1,
            action: "cloned".into(),
            unchanged: false,
        })
    }

    /// Moves an agent between enabled, disabled and archived.
    ///
    /// A version bump, because it changes what the agent may do and a run that
    /// pinned the previous version should be able to tell.
    pub fn set_state(
        &self,
        session: &Session,
        agent_id: &str,
        expected_version: u64,
        state: AgentState,
    ) -> Result<Mutation, RegistryError> {
        Self::require_administrator(session, "enable, disable or archive an agent")?;

        let mut held = self.held()?;
        let Some(position) = held.iter().position(|agent| agent.agent_id == agent_id) else {
            return Err(RegistryError::NoSuchAgent {
                agent_id: agent_id.to_string(),
            });
        };
        if held[position].definition_version != expected_version {
            return Err(RegistryError::VersionConflict {
                agent_id: agent_id.to_string(),
                expected: expected_version,
                actual: held[position].definition_version,
            });
        }
        // Already there. Reported as a no-op rather than written, so an audit
        // log does not fill with edits that changed nothing.
        if held[position].state == state {
            return Ok(Mutation {
                agent_id: agent_id.to_string(),
                definition_version: held[position].definition_version,
                action: state.as_str().to_string(),
                unchanged: true,
            });
        }

        held[position].state = state;
        held[position].definition_version += 1;
        held[position].updated_at = chrono::Utc::now().to_rfc3339();
        let version = held[position].definition_version;
        self.persist(&held)?;

        Ok(Mutation {
            agent_id: agent_id.to_string(),
            definition_version: version,
            action: state.as_str().to_string(),
            unchanged: false,
        })
    }

    // ── Import ───────────────────────────────────────────────────────────

    /// Brings a bundled profile into the registry, or updates the one it maps
    /// to, or does nothing.
    ///
    /// The three outcomes, and the rule for each:
    ///
    /// - **No mapping yet.** A new agent, with a fresh id and the next unused
    ///   palette colour.
    /// - **Mapped, and the profile's hash has changed.** The fields the
    ///   *profile* owns are updated. The fields the *deployment* owns — colour,
    ///   state, display name, model binding — are left exactly as they are,
    ///   because an upgrade that reset somebody's configuration would be an
    ///   upgrade that loses their work.
    /// - **Mapped, and the hash is the same.** Nothing, including the version
    ///   number. Start-up runs this every launch, and a version that moved
    ///   every time would make `definition_version` meaningless.
    pub fn import_bundled(&self, profile: &AgentProfile) -> Result<Mutation, RegistryError> {
        let mut held = self.held()?;

        let existing = held.iter().position(|agent| {
            agent
                .imported_from
                .as_ref()
                .is_some_and(|origin| origin.profile_name == profile.name)
        });

        if let Some(position) = existing {
            let unchanged = held[position]
                .imported_from
                .as_ref()
                .is_some_and(|origin| origin.profile_sha256 == profile.sha256);

            // A row this import itself wrote wrongly, repaired once.
            //
            // Every import before this fix stored the profile's one-line
            // description as the agent's instructions, so an imported agent was
            // told what it is *called* rather than what it is *for* (plan §13).
            // Those rows have an unchanged hash and would otherwise never be
            // touched again. The signature of the defect is exact -- the stored
            // instructions equal the description while the profile carries a
            // different body -- and the instructions are a field the profile
            // owns (the changed-hash branch below overwrites them), so repairing
            // them takes nothing an administrator set. A version bump, because a
            // run that pinned the old text should be able to tell.
            if unchanged
                && held[position].instructions == profile.description
                && !bundled_instructions(profile).eq(&profile.description)
            {
                let agent = &mut held[position];
                agent.instructions = bundled_instructions(profile);
                agent.definition_version += 1;
                agent.updated_at = chrono::Utc::now().to_rfc3339();
                let outcome = Mutation {
                    agent_id: agent.agent_id.clone(),
                    definition_version: agent.definition_version,
                    action: "imported".into(),
                    unchanged: false,
                };
                self.persist(&held)?;
                return Ok(outcome);
            }

            if unchanged {
                return Ok(Mutation {
                    agent_id: held[position].agent_id.clone(),
                    definition_version: held[position].definition_version,
                    action: "imported".into(),
                    unchanged: true,
                });
            }

            {
                let agent = &mut held[position];
                // The profile's half only.
                agent.description = profile.description.clone();
                agent.instructions = bundled_instructions(profile);
                agent.role = profile.model_role;
                agent.allowed_tools = profile.allowed_tools.clone();
                agent.denied_tools = profile.disallowed_tools.clone();
                agent.limits = profile.limits.clone();
                agent.isolation = profile.isolation;
                agent.write_policy = profile.write_policy;
                agent.classification_ceiling = profile.classification_ceiling;
                agent.output_schema = profile.required_schema;
                agent.memory.scope = profile.memory_scope;
                agent.imported_from = Some(ImportOrigin {
                    source: "bundled".into(),
                    profile_name: profile.name.clone(),
                    profile_sha256: profile.sha256.clone(),
                });
                agent.definition_version += 1;
                agent.updated_at = chrono::Utc::now().to_rfc3339();
            }

            let outcome = Mutation {
                agent_id: held[position].agent_id.clone(),
                definition_version: held[position].definition_version,
                action: "imported".into(),
                unchanged: false,
            };
            self.persist(&held)?;
            return Ok(outcome);
        }

        let colour = AGENT_PALETTE[held.len() % AGENT_PALETTE.len()].to_string();
        let now = chrono::Utc::now().to_rfc3339();
        let definition = AgentDefinition {
            agent_id: format!("ag-{}", uuid::Uuid::new_v4()),
            definition_version: 1,
            display_name: profile.name.clone(),
            description: profile.description.clone(),
            instructions: bundled_instructions(profile),
            role: profile.model_role,
            state: AgentState::Enabled,
            color: colour,
            skills: Vec::new(),
            allowed_tools: profile.allowed_tools.clone(),
            denied_tools: profile.disallowed_tools.clone(),
            memory: MemoryPolicy {
                scope: profile.memory_scope,
                // What every bundled worker has always done: publish its
                // findings to the task, where a sibling reads them. This was
                // hard-coded `false` while nothing read it (plan §3 finding 8);
                // P02 enforces it, so an import now records the role's real
                // behaviour. Rows imported earlier keep what they stored -- a
                // stored `false` may be a person's choice, and nothing here can
                // tell it from the old default -- and the Agents screen is where
                // it is changed.
                shared_with_task: true,
            },
            models: ModelBinding {
                default_model_id: None,
                fallback_model_ids: Vec::new(),
                eligible_model_ids: profile.eligible_models.clone(),
            },
            output_schema: profile.required_schema,
            limits: profile.limits.clone(),
            max_concurrent: 1,
            isolation: profile.isolation,
            write_policy: profile.write_policy,
            classification_ceiling: profile.classification_ceiling,
            imported_from: Some(ImportOrigin {
                source: "bundled".into(),
                profile_name: profile.name.clone(),
                profile_sha256: profile.sha256.clone(),
            }),
            created_at: now.clone(),
            updated_at: now,
        };
        definition
            .validate()
            .map_err(|detail| RegistryError::Invalid { detail })?;

        let outcome = Mutation {
            agent_id: definition.agent_id.clone(),
            definition_version: 1,
            action: "imported".into(),
            unchanged: false,
        };
        held.push(definition);
        self.persist(&held)?;
        Ok(outcome)
    }
}

/// What an imported agent is told it is for.
///
/// The profile's Markdown body -- the role, its method, what it must refuse --
/// and not its one-line description, which is what every import stored before
/// plan §13 found it. The description stands in only when a profile genuinely
/// has no body, because an agent told nothing at all would be worse than one
/// told its summary.
fn bundled_instructions(profile: &AgentProfile) -> String {
    if profile.instructions.trim().is_empty() {
        profile.description.clone()
    } else {
        profile.instructions.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agents::tests::definition;
    use crate::identity::User;
    use crate::subagents::profile::{Isolation, Limits, MemoryScope, SchemaKind, WritePolicy};

    fn admin() -> Session {
        Session::open(User::new("ada", "Ada", vec![Role::Administrator]))
    }

    fn employee() -> Session {
        Session::open(User::new("priya", "Priya", vec![Role::Employee]))
    }

    fn registry() -> (AgentRegistry, tempfile::TempDir) {
        let dir = tempfile::tempdir().expect("temp dir");
        let registry = AgentRegistry::open(dir.path()).expect("opens");
        (registry, dir)
    }

    fn profile(name: &str, sha: &str) -> AgentProfile {
        AgentProfile {
            name: name.into(),
            instructions: "Find the passages that bear on the question.".into(),
            description: "Finds passages.".into(),
            version: "1".into(),
            model_role: crate::registry::ModelRole::Reasoning,
            eligible_models: Vec::new(),
            allowed_tools: vec![crate::orchestrator::tools::ToolName::SearchDocuments],
            disallowed_tools: Vec::new(),
            limits: Limits {
                max_turns: 8,
                max_output_tokens: 2048,
                max_children: 0,
                max_duration_seconds: 300,
            },
            isolation: Isolation::ReadOnly,
            memory_scope: MemoryScope::None,
            network_permitted: false,
            write_policy: WritePolicy::None,
            classification_ceiling: crate::policy::Classification::Internal,
            required_schema: SchemaKind::Retrieval,
            sha256: sha.into(),
        }
    }

    /// Start-up runs the import every launch. Running it twice must not make
    /// two agents, and must not move a version number.
    #[test]
    fn importing_the_same_profile_twice_changes_nothing_the_second_time() {
        let (registry, _dir) = registry();

        let first = registry
            .import_bundled(&profile("retriever", "aaaa"))
            .expect("imports");
        assert!(!first.unchanged);
        assert_eq!(first.definition_version, 1);

        let second = registry
            .import_bundled(&profile("retriever", "aaaa"))
            .expect("imports");
        assert!(second.unchanged, "a second import made a change");
        assert_eq!(second.agent_id, first.agent_id, "a second agent was created");
        assert_eq!(second.definition_version, 1, "the version moved for nothing");
        assert_eq!(
            registry.list(Visibility::Administrator).expect("lists").len(),
            1
        );
    }

    /// An upgrade ships a changed profile. The profile's half updates; the
    /// deployment's half is left alone.
    #[test]
    fn a_changed_profile_updates_its_own_fields_and_keeps_the_deployments() {
        let (registry, _dir) = registry();
        let imported = registry
            .import_bundled(&profile("retriever", "aaaa"))
            .expect("imports");

        // The deployment configures it.
        let mut configured = registry
            .get(&imported.agent_id, Visibility::Administrator)
            .expect("reads");
        configured.display_name = "Evidence finder".into();
        configured.color = AGENT_PALETTE[5].into();
        registry
            .update(&admin(), &imported.agent_id, 1, configured)
            .expect("updates");

        // The model goes through the handoff's commit point rather than the
        // form: `update` refuses a submitted binding outright. See
        // `RegistryError::ModelBindingNeedsTransition`.
        registry
            .rebind_model(
                &admin(),
                &imported.agent_id,
                2,
                ModelBinding {
                    default_model_id: Some("orchestrator.spark-x2-5-4b".into()),
                    ..ModelBinding::default()
                },
                "tr-test",
            )
            .expect("rebinds");

        // The upgrade ships a new profile body.
        let mut changed = profile("retriever", "bbbb");
        changed.description = "Finds passages and cites them.".into();
        let reimported = registry.import_bundled(&changed).expect("re-imports");
        assert!(!reimported.unchanged);

        let after = registry
            .get(&imported.agent_id, Visibility::Administrator)
            .expect("reads");
        assert_eq!(after.description, "Finds passages and cites them.");
        assert_eq!(after.display_name, "Evidence finder", "a rename was reset");
        assert_eq!(after.color, AGENT_PALETTE[5], "a chosen colour was reset");
        assert_eq!(
            after.models.default_model_id.as_deref(),
            Some("orchestrator.spark-x2-5-4b"),
            "a model binding was reset"
        );
    }

    /// P06 extends the shipped document extractor into the Document & Vision
    /// Analyst. The upgrade is the real one: the 1.0.0 profile as it shipped
    /// (kept verbatim under `testdata/`), configured by a deployment, then the
    /// 1.1.0 profile in `agents/`. Same agent, next version, the deployment's
    /// settings kept, the new tools granted.
    #[test]
    fn the_document_extractor_upgrade_keeps_its_identity_and_its_settings() {
        use crate::orchestrator::tools::ToolName;
        let sha = |text: &str| hex::encode(<sha2::Sha256 as sha2::Digest>::digest(text.as_bytes()));
        let old_text = include_str!("testdata/document-extractor-1.0.0.md");
        let new_text = include_str!("../../../agents/document-extractor.md");
        let old = crate::subagents::profile::compile(old_text, "document-extractor", &sha(old_text))
            .expect("1.0.0 compiles");
        let new = crate::subagents::profile::compile(new_text, "document-extractor", &sha(new_text))
            .expect("1.1.0 compiles");
        assert_eq!(old.name, new.name, "the upgrade renamed the agent");

        let (registry, _dir) = registry();
        let imported = registry.import_bundled(&old).expect("imports 1.0.0");
        let mut configured = registry
            .get(&imported.agent_id, Visibility::Administrator)
            .expect("reads");
        configured.display_name = "Drawing reader".into();
        configured.color = AGENT_PALETTE[3].into();
        configured.memory.shared_with_task = false;
        registry
            .update(&admin(), &imported.agent_id, imported.definition_version, configured)
            .expect("configured");
        let version = registry
            .get(&imported.agent_id, Visibility::Administrator)
            .expect("reads")
            .definition_version;
        registry
            .rebind_model(
                &admin(),
                &imported.agent_id,
                version,
                ModelBinding {
                    default_model_id: Some("unlimited-ocr-q6-k".into()),
                    fallback_model_ids: vec!["unlimited-ocr-q4-k-m".into()],
                    ..ModelBinding::default()
                },
                "tr-p06",
            )
            .expect("bound");
        let before = registry
            .get(&imported.agent_id, Visibility::Administrator)
            .expect("reads");

        let upgraded = registry.import_bundled(&new).expect("imports 1.1.0");
        assert_eq!(upgraded.agent_id, imported.agent_id, "a second agent was created");
        assert_eq!(upgraded.definition_version, before.definition_version + 1);
        let after = registry
            .get(&upgraded.agent_id, Visibility::Administrator)
            .expect("reads");
        assert_eq!(after.display_name, "Drawing reader");
        assert_eq!(after.color, AGENT_PALETTE[3]);
        assert!(!after.memory.shared_with_task, "a deployment's sharing choice was reset");
        assert_eq!(after.models.default_model_id.as_deref(), Some("unlimited-ocr-q6-k"));
        assert_eq!(after.models.fallback_model_ids, vec!["unlimited-ocr-q4-k-m".to_string()]);
        for tool in [
            ToolName::MediaExtractFindings,
            ToolName::DocumentLayoutMap,
            ToolName::DocumentOcrRegions,
            ToolName::DocumentExtractTables,
            ToolName::DocumentRenderRegions,
            ToolName::ReadAttachedPages,
        ] {
            assert!(after.allowed_tools.contains(&tool), "{} was not granted", tool.as_str());
        }
        assert!(after.denied_tools.contains(&ToolName::WriteScopedFile), "the writer denial was lost");
        assert!(after.instructions.contains("Document & Vision Analyst"));
    }

    /// P07 extends the shipped knowledge retriever. The same real upgrade as
    /// the extractor's: 1.0.0 as it shipped, configured, then 1.1.0 — one
    /// agent, the next version, the settings kept, the retrieval tools granted.
    #[test]
    fn the_knowledge_retriever_upgrade_keeps_its_identity_and_its_settings() {
        use crate::orchestrator::tools::ToolName;
        let sha = |text: &str| hex::encode(<sha2::Sha256 as sha2::Digest>::digest(text.as_bytes()));
        let old_text = include_str!("testdata/knowledge-retriever-1.0.0.md");
        let new_text = include_str!("../../../agents/knowledge-retriever.md");
        let old = crate::subagents::profile::compile(old_text, "knowledge-retriever", &sha(old_text))
            .expect("1.0.0 compiles");
        let new = crate::subagents::profile::compile(new_text, "knowledge-retriever", &sha(new_text))
            .expect("1.1.0 compiles");
        assert_eq!(old.name, new.name, "the upgrade renamed the agent");

        let (registry, _dir) = registry();
        let imported = registry.import_bundled(&old).expect("imports 1.0.0");
        let mut configured = registry
            .get(&imported.agent_id, Visibility::Administrator)
            .expect("reads");
        configured.display_name = "SOP finder".into();
        configured.color = AGENT_PALETTE[2].into();
        registry
            .update(&admin(), &imported.agent_id, imported.definition_version, configured)
            .expect("configured");
        let before = registry
            .get(&imported.agent_id, Visibility::Administrator)
            .expect("reads");

        let upgraded = registry.import_bundled(&new).expect("imports 1.1.0");
        assert_eq!(upgraded.agent_id, imported.agent_id, "a second agent was created");
        assert_eq!(upgraded.definition_version, before.definition_version + 1);
        let after = registry
            .get(&upgraded.agent_id, Visibility::Administrator)
            .expect("reads");
        assert_eq!(after.display_name, "SOP finder");
        assert_eq!(after.color, AGENT_PALETTE[2]);
        for tool in [
            ToolName::SearchDocuments,
            ToolName::KnowledgeHybridSearch,
            ToolName::LoadMoreEvidence,
            ToolName::KnowledgeSourceVersion,
            ToolName::KnowledgeRerank,
            ToolName::MemoryNeighbours,
        ] {
            assert!(after.allowed_tools.contains(&tool), "{} was not granted", tool.as_str());
        }
        assert!(after.denied_tools.contains(&ToolName::WriteScopedFile), "the writer denial was lost");
        assert!(after.instructions.contains("Deterministically"));
    }

    #[test]
    fn a_registry_survives_a_restart() {
        let dir = tempfile::tempdir().expect("temp dir");
        let agent_id = {
            let registry = AgentRegistry::open(dir.path()).expect("opens");
            registry
                .create(&admin(), definition("ignored"))
                .expect("creates")
                .agent_id
        };

        let reopened = AgentRegistry::open(dir.path()).expect("reopens");
        let agent = reopened
            .get(&agent_id, Visibility::Administrator)
            .expect("the agent survived the restart");
        assert_eq!(agent.display_name, "Knowledge retriever");
    }

    /// The id is minted by the registry, so a caller cannot choose one that
    /// collides with an existing agent's history.
    #[test]
    fn a_caller_cannot_choose_an_agents_id() {
        let (registry, _dir) = registry();
        let created = registry
            .create(&admin(), definition("i-chose-this"))
            .expect("creates");
        assert_ne!(created.agent_id, "i-chose-this");
        assert!(created.agent_id.starts_with("ag-"));
    }

    #[test]
    fn an_employee_cannot_change_anything() {
        let (registry, _dir) = registry();
        let created = registry.create(&admin(), definition("x")).expect("creates");

        for refusal in [
            registry.create(&employee(), definition("x")).unwrap_err(),
            registry
                .update(&employee(), &created.agent_id, 1, definition("x"))
                .unwrap_err(),
            registry
                .clone_agent(&employee(), &created.agent_id, "copy")
                .unwrap_err(),
            registry
                .set_state(&employee(), &created.agent_id, 1, AgentState::Disabled)
                .unwrap_err(),
        ] {
            assert!(
                matches!(refusal, RegistryError::NotAdministrator { .. }),
                "{refusal:?}"
            );
        }
    }

    /// Two administrators with the same form open. The second save is refused
    /// rather than silently undoing the first.
    #[test]
    fn a_concurrent_edit_is_refused_with_both_versions_named() {
        let (registry, _dir) = registry();
        let created = registry.create(&admin(), definition("x")).expect("creates");

        let first = registry
            .get(&created.agent_id, Visibility::Administrator)
            .expect("reads");
        let mut second = first.clone();

        let mut first_edit = first;
        first_edit.display_name = "Named by the first editor".into();
        registry
            .update(&admin(), &created.agent_id, 1, first_edit)
            .expect("the first save lands");

        second.display_name = "Named by the second editor".into();
        let refusal = registry
            .update(&admin(), &created.agent_id, 1, second)
            .expect_err("the second save must be refused");
        assert_eq!(
            refusal,
            RegistryError::VersionConflict {
                agent_id: created.agent_id.clone(),
                expected: 1,
                actual: 2,
            }
        );
        assert!(refusal.explain().contains("Reload"));

        let after = registry
            .get(&created.agent_id, Visibility::Administrator)
            .expect("reads");
        assert_eq!(after.display_name, "Named by the first editor");
    }

    /// A rename keeps the identity, which is what keeps the memory and the
    /// history attached to it.
    ///
    /// A model reassignment keeps it too — but it does not happen here. This
    /// form refuses a submitted binding, because changing the model an agent
    /// runs on has to drain its work, save its state and prove the new model
    /// can hold the task. Both halves are asserted below: the refusal, and the
    /// identity surviving the reassignment when it goes the right way.
    #[test]
    fn a_rename_and_a_model_reassignment_preserve_id_and_colour() {
        let (registry, _dir) = registry();
        let created = registry.create(&admin(), definition("x")).expect("creates");
        let before = registry
            .get(&created.agent_id, Visibility::Administrator)
            .expect("reads");

        // A form that tries to do both is refused, and nothing is written —
        // including the rename, so an administrator is never left believing
        // half a save went through.
        let mut both = before.clone();
        both.display_name = "Something else entirely".into();
        both.models.default_model_id = Some("another-model".into());
        let refusal = registry
            .update(&admin(), &created.agent_id, before.definition_version, both)
            .expect_err("a form may not rebind a model");
        assert!(matches!(
            refusal,
            RegistryError::ModelBindingNeedsTransition { .. }
        ));
        assert_eq!(
            registry
                .get(&created.agent_id, Visibility::Administrator)
                .expect("reads")
                .display_name,
            before.display_name,
            "a refused save changes nothing at all"
        );

        // The rename on its own is accepted.
        let mut renamed = before.clone();
        renamed.display_name = "Something else entirely".into();
        registry
            .update(&admin(), &created.agent_id, before.definition_version, renamed)
            .expect("updates");

        // And the reassignment through the handoff's commit point.
        registry
            .rebind_model(
                &admin(),
                &created.agent_id,
                before.definition_version + 1,
                ModelBinding {
                    default_model_id: Some("another-model".into()),
                    ..ModelBinding::default()
                },
                "tr-test",
            )
            .expect("rebinds");

        let after = registry
            .get(&created.agent_id, Visibility::Administrator)
            .expect("reads");
        assert_eq!(after.agent_id, before.agent_id, "the identity moved");
        assert_eq!(after.color, before.color, "the colour changed on a rename");
        assert_eq!(after.created_at, before.created_at);
        assert_eq!(after.display_name, "Something else entirely");
        assert_eq!(
            after.models.default_model_id.as_deref(),
            Some("another-model")
        );
        assert_eq!(
            after.definition_version,
            before.definition_version + 2,
            "one version for the rename and one for the reassignment"
        );
    }

    /// Rebinding to the binding already held is a no-op, so a recovery that
    /// re-runs a commit whose write landed does not move the version again.
    #[test]
    fn rebinding_to_the_same_models_is_reported_as_no_change() {
        let (registry, _dir) = registry();
        let created = registry.create(&admin(), definition("x")).expect("creates");
        let before = registry
            .get(&created.agent_id, Visibility::Administrator)
            .expect("reads");

        let outcome = registry
            .rebind_model(
                &admin(),
                &created.agent_id,
                before.definition_version,
                before.models.clone(),
                "tr-test",
            )
            .expect("accepted");
        assert!(outcome.unchanged);
        assert_eq!(outcome.definition_version, before.definition_version);
    }

    /// Only an administrator may move an agent between models.
    #[test]
    fn an_employee_cannot_rebind_a_model() {
        let (registry, _dir) = registry();
        let created = registry.create(&admin(), definition("x")).expect("creates");
        let refusal = registry
            .rebind_model(
                &employee(),
                &created.agent_id,
                1,
                ModelBinding {
                    default_model_id: Some("another-model".into()),
                    ..ModelBinding::default()
                },
                "tr-test",
            )
            .expect_err("an employee may not");
        assert!(matches!(refusal, RegistryError::NotAdministrator { .. }));
    }

    /// A clone is a different agent, so its memory is empty by construction:
    /// memory is keyed by `agent_id` and the clone has a new one.
    #[test]
    fn a_clone_gets_a_new_identity_and_no_inherited_origin() {
        let (registry, _dir) = registry();
        let created = registry.create(&admin(), definition("x")).expect("creates");
        registry
            .import_bundled(&profile("retriever", "aaaa"))
            .expect("imports");
        let bundled = registry
            .list(Visibility::Administrator)
            .expect("lists")
            .into_iter()
            .find(|agent| agent.imported_from.is_some())
            .expect("the imported one");

        let copy = registry
            .clone_agent(&admin(), &bundled.agent_id, "A second retriever")
            .expect("clones");

        assert_ne!(copy.agent_id, bundled.agent_id);
        assert_ne!(copy.agent_id, created.agent_id);
        let cloned = registry
            .get(&copy.agent_id, Visibility::Administrator)
            .expect("reads");
        assert_eq!(cloned.display_name, "A second retriever");
        assert_eq!(cloned.definition_version, 1);
        assert!(
            cloned.imported_from.is_none(),
            "a clone must not claim to be the bundled profile's mapping, or the next import \
             would not know which of the two it owns"
        );
        // The configuration is copied, which is the point of a clone.
        assert_eq!(cloned.allowed_tools, bundled.allowed_tools);
    }

    /// An archived agent is retired from being given work, not from having done
    /// it.
    #[test]
    fn an_archived_agent_stays_resolvable_for_history_and_refuses_new_work() {
        let (registry, _dir) = registry();
        let created = registry.create(&admin(), definition("x")).expect("creates");
        registry
            .set_state(&admin(), &created.agent_id, 1, AgentState::Archived)
            .expect("archives");

        // Still there for anything that has to name who did the work.
        let resolved = registry
            .resolve_for_provenance(&created.agent_id)
            .expect("an archived agent is still resolvable");
        assert_eq!(resolved.state, AgentState::Archived);

        // And cannot be given any.
        let refusal = registry
            .pin_for_run(&created.agent_id)
            .expect_err("an archived agent cannot start a run");
        assert!(matches!(refusal, RegistryError::NotRunnable { .. }));
    }

    /// An operator sees what can run, and nothing else.
    #[test]
    fn an_operator_is_shown_only_agents_that_can_actually_run() {
        let (registry, _dir) = registry();
        let enabled = registry.create(&admin(), definition("a")).expect("creates");
        let disabled = registry.create(&admin(), definition("b")).expect("creates");
        registry
            .set_state(&admin(), &disabled.agent_id, 1, AgentState::Disabled)
            .expect("disables");

        let operator_sees = registry.list(Visibility::Operator).expect("lists");
        assert_eq!(operator_sees.len(), 1);
        assert_eq!(operator_sees[0].agent_id, enabled.agent_id);
        assert_eq!(
            registry.list(Visibility::Administrator).expect("lists").len(),
            2
        );

        // And the refusal for a hidden agent is the same as for one that does
        // not exist, so it cannot be used to discover which ids are real.
        let hidden = registry
            .get(&disabled.agent_id, Visibility::Operator)
            .unwrap_err();
        let absent = registry.get("ag-nothing", Visibility::Operator).unwrap_err();
        assert!(matches!(hidden, RegistryError::NoSuchAgent { .. }));
        assert!(matches!(absent, RegistryError::NoSuchAgent { .. }));
    }

    /// Setting a state to the one it already holds is a no-op, so an audit log
    /// does not fill with edits that changed nothing.
    #[test]
    fn a_state_change_to_the_current_state_is_reported_as_unchanged() {
        let (registry, _dir) = registry();
        let created = registry.create(&admin(), definition("x")).expect("creates");
        let outcome = registry
            .set_state(&admin(), &created.agent_id, 1, AgentState::Enabled)
            .expect("accepts");
        assert!(outcome.unchanged);
        assert_eq!(outcome.definition_version, 1, "the version moved for nothing");
    }

    /// A definition that would exceed a hard ceiling is refused at the store,
    /// not only in the type.
    #[test]
    fn an_escalating_definition_is_refused_on_the_way_in() {
        let (registry, _dir) = registry();
        let mut escalating = definition("x");
        escalating.limits.max_turns = 10_000;
        let refusal = registry.create(&admin(), escalating).unwrap_err();
        assert!(matches!(refusal, RegistryError::Invalid { .. }));
        assert!(registry
            .list(Visibility::Administrator)
            .expect("lists")
            .is_empty());
    }

    /// A registry from a newer build is refused rather than half-read.
    #[test]
    fn a_registry_written_by_a_newer_build_is_refused() {
        let dir = tempfile::tempdir().expect("temp dir");
        std::fs::create_dir_all(dir.path().join("agents")).expect("dir");
        std::fs::write(
            dir.path().join("agents").join("registry.json"),
            r#"{"schemaVersion": 99, "agents": []}"#,
        )
        .expect("writes");

        let refusal = AgentRegistry::open(dir.path()).unwrap_err();
        assert_eq!(
            refusal,
            RegistryError::UnknownSchema {
                found: 99,
                understood: AGENT_SCHEMA_VERSION
            }
        );
    }

    /// The registry writes to its own directory and never to the installer's.
    #[test]
    fn the_registry_writes_only_into_the_application_data_directory() {
        let dir = tempfile::tempdir().expect("temp dir");
        let registry = AgentRegistry::open(dir.path()).expect("opens");
        registry.create(&admin(), definition("x")).expect("creates");

        let written = dir.path().join("agents").join("registry.json");
        assert!(
            written.is_file(),
            "the registry was not written where it says"
        );
        assert!(std::fs::read_to_string(&written)
            .expect("readable")
            .contains("Knowledge retriever"));
    }

    /// A run pins the definition it started under, so an edit made while it is
    /// working does not change the rules under it.
    #[test]
    fn an_edit_during_a_run_does_not_move_what_that_run_pinned() {
        let (registry, _dir) = registry();
        let created = registry.create(&admin(), definition("x")).expect("creates");

        let pinned = registry.pin_for_run(&created.agent_id).expect("pins");
        assert_eq!(pinned.definition_version, 1);

        let mut edited = registry
            .get(&created.agent_id, Visibility::Administrator)
            .expect("reads");
        edited.denied_tools = vec![crate::orchestrator::tools::ToolName::ExecuteCode];
        registry
            .update(&admin(), &created.agent_id, 1, edited)
            .expect("updates");

        let current = registry
            .get(&created.agent_id, Visibility::Administrator)
            .expect("reads");
        assert!(
            !pinned.matches(&current),
            "the pin must be able to tell that the definition moved"
        );
        assert_eq!(
            pinned.definition_version, 1,
            "the pinned copy itself must not change under the run"
        );
    }

    /// A colour outside the palette is refused, and the refusal names the
    /// palette rather than saying "invalid".
    ///
    /// The palette is not decoration: the memory graph colours a node and its
    /// authored edges by agent, and a colour from outside the eight would not
    /// hold its weight against the others on that black surface — or, worse,
    /// would collide with the neutral reserved for things no agent owns.
    #[test]
    fn a_colour_outside_the_palette_is_refused_on_the_way_in() {
        let (registry, _dir) = registry();
        let mut wrong = definition("ignored");
        wrong.color = "#123456".into();

        let refusal = registry
            .create(&admin(), wrong)
            .expect_err("a colour outside the palette must be refused");
        let sentence = refusal.explain();
        assert!(
            sentence.contains(crate::agents::AGENT_PALETTE[0]),
            "the refusal did not name the palette: {sentence}"
        );

        // And the neutral reserved for shared, unowned items is not an agent
        // colour either.
        let mut neutral = definition("ignored");
        neutral.color = crate::agents::SHARED_COLOR.into();
        assert!(
            registry.create(&admin(), neutral).is_err(),
            "the shared neutral was accepted as an agent colour"
        );
    }

    /// Every colour the palette offers is actually accepted.
    ///
    /// The other half of the test above: a validator that refused everything
    /// would pass that one and make the screen unusable.
    #[test]
    fn every_palette_colour_is_accepted() {
        let (registry, _dir) = registry();
        for color in crate::agents::AGENT_PALETTE {
            let mut candidate = definition("ignored");
            candidate.color = color.to_string();
            registry
                .create(&admin(), candidate)
                .unwrap_or_else(|error| panic!("{color} was refused: {}", error.explain()));
        }
    }
}
