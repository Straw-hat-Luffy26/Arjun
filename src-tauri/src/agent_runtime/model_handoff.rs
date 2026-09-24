//! Running a model-binding transition, in the order [`super::model_transition`]
//! says it has to happen.
//!
//! ## Why this is a plain struct and not a Tauri command
//!
//! Because `drive_run` is not callable from a test. It takes an `AppHandle` and
//! twenty-odd `State<'_, _>` arguments, and `tauri::test::mock_app` needs a
//! feature this crate does not enable — the reason
//! [`crate::commands::governance`] factors `attempt_sign_in` out of its command,
//! and the reason `tests/rbac_isolation.rs` tests the permission matrix rather
//! than the IPC surface.
//!
//! So the handoff lives here, taking references a test can construct: a real
//! `AgentRegistry` on a temporary directory, a real `TaskEventLog` on a real
//! SQLite file, a real `ModelRegistry`, and a real `ModelServers` that starts
//! real `llama-server` children. The IPC commands in
//! [`crate::commands::agents`] and the resumption path in
//! [`crate::commands::agent`] are thin shims over this, so the code a test
//! drives is the code production runs.
//!
//! ## The three entry points, and what each promises
//!
//! - [`Handoff::execute`] moves an agent, or says where it got to. It returns
//!   `Err` only when the transition could not be *opened* — not an
//!   administrator, no such agent, a version conflict, another handoff already
//!   under way. Once a record exists every outcome comes back as `Ok(record)`
//!   with the phase and the reason on it, because the record is the product.
//! - [`Handoff::rollback`] undoes one that is still open.
//! - [`Handoff::reconcile_open`] is the start-up sweep that reads the registry
//!   to decide what a crash meant.
//!
//! ## What happens on a card that holds one model
//!
//! The checkpoint is taken before anything is unloaded, which is the whole
//! reason [`TransitionPhase::Checkpointed`] comes before
//! [`TransitionPhase::Loading`]. `serving::admission::admit` will evict the
//! source to make room for the target — that is its job, and on an 8 GB card
//! with a 9B model it is the ordinary case rather than the exception. If the
//! target then fails to come up, the source is *gone*, and the rollback has to
//! reload it; if that also fails the outcome is
//! [`TransitionPhase::RollbackFailed`] and a person is told, rather than the
//! agent being left pointing at a model nothing is serving.

use std::collections::{BTreeSet, HashMap};
use std::path::Path;
use std::sync::Mutex;

use serde::{Deserialize, Serialize};
use serde_json::json;

use super::context_compiler::{ContextCompiler, FrozenScope, Reserves};
use super::context_manifest::ContextManifest;
use super::events::{ApprovalStatus, EventDraft, TaskEventLog, TaskEventType};
use super::memory::CompletedEffect;
use super::model_transition::{
    drain_for, fits_mandatory, validate_target, BindingRequirements, DrainVerdict, DroppedState,
    ModelFingerprint, PortableState, Reconciliation, RollbackPlan, ServedBinding, SourceFreeze,
    TransitionPhase, TransitionRecord, ValidationRefusal,
};
use super::resume::CheckpointSeed;
use crate::agents::store::{AgentRegistry, RegistryError};
use crate::agents::AgentDefinition;
use crate::identity::Session;
use crate::knowledge::graph::runtime_store::MemoryGraph;
use crate::registry::ModelRegistry;
use crate::serving::ModelServers;

/// How far a handoff goes in proving the destination works.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum VerifyDestination {
    /// Start the target, health-check it, probe its tokenizer, and recompile
    /// the task's context against the window it actually came up with.
    ///
    /// The only honest setting when there is work to hand over: the run
    /// continues on this model immediately, so "it can probably be served" is
    /// not good enough.
    LoadAndVerify,
    /// Plan the admission and stop. Nothing is loaded and nothing is evicted.
    ///
    /// For reassigning an idle agent. Loading a 12B model to change a setting on
    /// an agent nobody is using would evict whatever a *different* person's
    /// conversation is mid-turn on — a real harm in exchange for confirming
    /// something the next run confirms anyway. The record then says the window
    /// was never measured, rather than filling in the registry's declared
    /// figure and passing it off as one.
    PlanOnly,
}

/// What an administrator asked for.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HandoffRequest {
    pub agent_id: String,
    pub target_model_id: String,
    /// The definition version the screen was opened at. A concurrent edit moves
    /// it and this is refused, because that edit may have narrowed the tools or
    /// the classification ceiling the target would be validated against.
    pub expected_definition_version: u64,
    /// What the administrator said they were doing. Bounded when stored.
    #[serde(default)]
    pub reason: String,
    /// The run being handed over. `None` reassigns an idle agent.
    #[serde(default)]
    pub run_id: Option<String>,
    /// The memory scope the context is compiled against. Defaults to the run
    /// id, which is this product's task identity — see
    /// [`super::events::checkpoint::RunCheckpoint`], whose `run_id` is "the
    /// logical task, stable across every attempt".
    #[serde(default)]
    pub task_id: Option<String>,
    #[serde(default)]
    pub project_id: Option<String>,
    pub verify: VerifyDestination,
    /// What the turn spends before any context is added, when the caller knows
    /// better than the source manifest does.
    ///
    /// Normally `None`, and then the reserves are read off the source
    /// manifest's own budget record — the figures the interrupted turn was
    /// actually budgeted with. A source manifest with no budget record is
    /// refused rather than defaulted: a reserve nobody measured, used to decide
    /// whether a context fits, is the shape of fabricated evidence this
    /// repository has a standing rule against.
    #[serde(default)]
    pub reserves: Option<ReserveRequest>,
}

/// Reserves as a caller may state them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReserveRequest {
    pub tool_schemas: u32,
    pub output: u32,
    pub framing: u32,
    /// Held back for what counting may miss. Zero on a request or a source
    /// manifest from before the reserve existed, which is what those turns
    /// were actually budgeted with.
    #[serde(default)]
    pub safety: u32,
}

impl ReserveRequest {
    fn total(self) -> u32 {
        self.tool_schemas
            .saturating_add(self.output)
            .saturating_add(self.framing)
            .saturating_add(self.safety)
    }

    fn as_reserves(self) -> Reserves {
        Reserves {
            tool_schemas: self.tool_schemas,
            output: self.output,
            framing: self.framing,
            safety: self.safety,
            // Nothing here consults a model's own tokenizer to *measure* the
            // blocks, and saying `tokenizer` when an estimate was used is how a
            // turn overruns a window it was told it fitted. The tokenizer probe
            // in the record is a comparison between two models, not a
            // measurement of this context.
            counted_by: "estimate".to_string(),
            window_source: None,
        }
    }
}

/// Why a handoff stopped.
///
/// Four kinds rather than one string, because they leave the machine in
/// different states and therefore need different endings. A refusal that never
/// touched anything is [`TransitionPhase::Failed`]; one that stopped after the
/// target was admitted has to put the source back.
#[derive(Debug, Clone)]
enum Failure {
    /// The destination was not acceptable. Carries the typed reason.
    Refused(ValidationRefusal),
    /// Somebody changed the agent between the validation and the write.
    Conflict(String),
    /// Something could not be read or written.
    Storage(String),
    /// The state machine was asked for an edge that does not exist. A defect in
    /// this module, reported as one rather than as an operator's problem.
    Defect(String),
}

impl Failure {
    fn refused(refusal: ValidationRefusal) -> Self {
        Failure::Refused(refusal)
    }

    fn conflict(detail: impl Into<String>) -> Self {
        Failure::Conflict(detail.into())
    }

    fn storage(detail: impl Into<String>) -> Self {
        Failure::Storage(detail.into())
    }

    fn defect(detail: impl Into<String>) -> Self {
        Failure::Defect(detail.into())
    }

    fn explain(&self) -> String {
        match self {
            Failure::Refused(refusal) => refusal.explain(),
            Failure::Conflict(detail) | Failure::Storage(detail) => detail.clone(),
            Failure::Defect(detail) => format!(
                "The model change stopped because of a fault in ARJUN rather than anything you \
                 did: {detail}"
            ),
        }
    }

    /// Whether the machine has been changed and needs putting back.
    ///
    /// The *phase* decides, because the phase is what says whether anything was
    /// touched: only the load and what follows it can have taken a model's
    /// memory. The refusal is consulted as well, and the two agree by
    /// construction — every refusal that is not pre-flight is one that can only
    /// be raised at or after the load. Both are checked so that adding a
    /// refusal on the wrong side of that line fails loudly rather than
    /// silently skipping an undo.
    fn needs_rollback(&self, phase: TransitionPhase) -> bool {
        let disturbed = phase.may_have_released_memory();
        match self {
            Failure::Refused(refusal) => disturbed || !refusal.is_pre_flight(),
            // A conflict is only ever raised at the commit, by which point the
            // target is loaded and the source may be gone.
            Failure::Conflict(_) => true,
            Failure::Storage(_) | Failure::Defect(_) => disturbed,
        }
    }
}

/// Everything a handoff needs, and nothing that starts one.
pub struct Handoff<'a> {
    pub agents: &'a AgentRegistry,
    pub events: &'a TaskEventLog,
    pub registry: &'a ModelRegistry,
    pub servers: &'a ModelServers,
    /// `None` on a deployment whose runtime memory graph could not be opened.
    /// The handoff then refuses to recompile rather than recompiling against an
    /// empty set, which would read as "this task remembers nothing".
    pub graph: Option<&'a MemoryGraph>,
    /// The live runs' checkpoint seeds, by run id. See
    /// [`crate::commands::agent::RunCheckpoints`].
    pub checkpoints: &'a Mutex<HashMap<String, CheckpointSeed>>,
    pub models_dir: &'a Path,
    /// The one lease service. A handoff's load — and a rollback's reload —
    /// hold the card, because admission may stop servers to make room and must
    /// never stop one another generation is using. `None` on a path with no
    /// scheduler (the handoff's own unit tests), which then loads as it did
    /// before the service existed.
    pub leases: Option<&'a crate::subagents::ModelScheduler>,
}

impl Handoff<'_> {
    // -- Entry points -----------------------------------------------------

    /// Moves an agent to a different model, or says where the attempt got to.
    pub async fn execute(
        &self,
        session: &Session,
        request: HandoffRequest,
    ) -> Result<TransitionRecord, String> {
        // Asked before anything expensive. The refusal that *matters* is on
        // `rebind_model` at the very end; this is the same predicate asked
        // early so a run is not drained and gigabytes of weights are not read
        // for somebody who may not have the write done.
        if !AgentRegistry::may_administer(session) {
            return Err(RegistryError::NotAdministrator {
                action: "change the model an agent runs on".into(),
            }
            .explain());
        }

        // Archived and disabled agents resolve here on purpose. Reassigning a
        // disabled agent's model is preparing it to be enabled, which is a
        // reasonable thing to do and a different question from whether it may
        // be given work.
        let agent = self
            .agents
            .resolve_for_provenance(&request.agent_id)
            .map_err(|error| error.explain())?;

        if agent.definition_version != request.expected_definition_version {
            return Err(RegistryError::VersionConflict {
                agent_id: request.agent_id.clone(),
                expected: request.expected_definition_version,
                actual: agent.definition_version,
            }
            .explain());
        }

        // Handing over live work without loading the destination is not a
        // combination this will accept.
        //
        // The run continues on the new model the moment the binding commits,
        // and its context has to be rebuilt to fit that model's *served*
        // window — which is not knowable until a server reports it, and which
        // on a constrained card is routinely a quarter of what the registry
        // declares. Committing without it would be binding a task to a window
        // nobody measured and finding out on the next turn.
        if request.verify == VerifyDestination::PlanOnly && request.run_id.is_some() {
            return Err(
                "This run is being handed over, so the new model has to be started and checked \
                 first: its context has to be rebuilt to fit the window that model actually \
                 comes up with, and that figure is not known until it is serving. Ask again \
                 without asking to skip the load."
                    .to_string(),
            );
        }

        // A handoff already under way is either this one being retried or a
        // second one that must not start. They are told apart by what they are
        // moving to: the same target from the same version is a retry of the
        // same act.
        let mut record = match self.events.open_transition_for_agent(&request.agent_id)? {
            Some(open)
                if open.to_model.model_id == request.target_model_id
                    && open.from_definition_version == request.expected_definition_version =>
            {
                // Only the phases before anything was touched may be picked up.
                // A transition abandoned at `validating` or later is a process
                // that died mid-flight, and `reconcile_open` is what reads the
                // registry to decide what that meant — re-running the middle of
                // it blind would be the second guess this module exists to
                // avoid.
                if !matches!(
                    open.phase,
                    TransitionPhase::Requested
                        | TransitionPhase::Draining
                        | TransitionPhase::AwaitingSettlement
                ) {
                    return Err(format!(
                        "This handoff stopped at {}, which means the process went away in the \
                         middle of it. It is settled by the recovery sweep rather than by being \
                         started again, because what happened to the half-done work has to be \
                         read off the record rather than guessed.",
                        open.phase.as_str().replace('_', " ")
                    ));
                }
                open
            }
            Some(open) => {
                return Err(format!(
                    "{} is already being moved to {} (handoff {}, currently {}). Two handoffs of \
                     one agent would race over the same record, so this one was not started.",
                    request.agent_id,
                    open.to_model.model_id,
                    open.transition_id,
                    open.phase.as_str().replace('_', " ")
                ))
            }
            None => self.open(session, &request, &agent)?,
        };

        // From here on the record exists, so every outcome is `Ok(record)` and
        // the phase carries the answer. Returning `Err` past this point would
        // throw away the one durable account of what was attempted.
        Ok(self.drive(session, &request, &agent, &mut record).await)
    }

    /// Undoes a handoff that has not settled.
    ///
    /// Bounded: it restores the binding and, if something was released to make
    /// room, the source model. It does not unwind the work — a document written
    /// between the checkpoint and the failure stays written, and its receipt
    /// stays in the notes, which is what stops a resumption writing it twice.
    pub async fn rollback(
        &self,
        session: &Session,
        transition_id: &str,
    ) -> Result<TransitionRecord, String> {
        if !AgentRegistry::may_administer(session) {
            return Err(RegistryError::NotAdministrator {
                action: "undo a model change".into(),
            }
            .explain());
        }
        let mut record = self
            .events
            .transition(transition_id)?
            .ok_or_else(|| format!("There is no model change recorded as {transition_id}."))?;

        if record.phase.is_terminal() {
            return Err(format!(
                "That model change has already settled ({}). Moving the agent back is a change of \
                 its own, with its own checks — start one rather than undoing a finished record.",
                record.phase.as_str().replace('_', " ")
            ));
        }
        if matches!(
            record.phase,
            TransitionPhase::Requested
                | TransitionPhase::Draining
                | TransitionPhase::AwaitingSettlement
        ) {
            // Nothing was touched, so there is nothing to put back. Closed as
            // abandoned rather than rolled back, because saying "rolled back"
            // would imply something had been undone.
            record.because = Some(
                "Abandoned before anything was changed: the agent is on the model it was on, and \
                 its run was never interrupted."
                    .into(),
            );
            let _ = record.advance(TransitionPhase::Failed, None);
            self.note_abandoned(
                &session.user.id,
                &record,
                TaskEventType::ModelTransitionFailed,
            );
            self.events.save_transition(&record)?;
            return Ok(record);
        }

        self.undo(Some(session), &mut record).await;
        self.note_abandoned(
            &session.user.id,
            &record,
            TaskEventType::ModelTransitionRolledBack,
        );
        self.events.save_transition(&record)?;
        Ok(record)
    }

    /// Settles every handoff the process died in the middle of.
    ///
    /// Called once at start-up, before anything is given work. A row still open
    /// then is not a handoff in progress — nothing is running it — so leaving it
    /// would block the agent from ever being moved again, because the ledger's
    /// partial unique index treats an unsettled row as one under way.
    ///
    /// The interesting case is [`TransitionPhase::CommitPending`], which is the
    /// one window where the two stores can disagree. See
    /// [`super::model_transition::PendingCommit::reconcile`]: the answer is read
    /// off the registry and is three-valued, and the third value is reported
    /// rather than guessed.
    ///
    /// ## Why `authority` is separate from `actor`, and usually `None`
    ///
    /// Because nobody is signed in at start-up, and a recovery sweep that
    /// fabricated an administrator session to get its work done would be a
    /// path by which anything that runs at start-up holds administrative
    /// authority. There is no such path here.
    ///
    /// It is not needed, either, and that is worth stating rather than relying
    /// on. Closing a ledger row is ARJUN's own bookkeeping. Reading the registry
    /// to find out whether a write landed is a read. And the only case that
    /// would need a *write* is putting a committed binding back — which cannot
    /// arise from an unsettled row, because a binding that was written means the
    /// phase reconciles as [`Reconciliation::Landed`] and the handoff is
    /// finished rather than undone. If it ever did arise, the restore is refused
    /// for want of authority and the outcome becomes
    /// [`TransitionPhase::RollbackFailed`], which asks a person — the same
    /// answer as any other undo that could not complete.
    pub async fn reconcile_open(
        &self,
        actor: &str,
        authority: Option<&Session>,
    ) -> Vec<TransitionRecord> {
        let open = match self.events.open_transitions() {
            Ok(open) => open,
            Err(error) => {
                log::error!(
                    "[agents] the unsettled model changes could not be read, so none were \
                     reconciled: {error}"
                );
                return Vec::new();
            }
        };
        if open.is_empty() {
            return Vec::new();
        }
        log::info!(
            "[agents] {} model change(s) were interrupted and are being settled",
            open.len()
        );

        let mut settled = Vec::new();
        for mut record in open {
            self.reconcile_one(actor, authority, &mut record).await;
            if let Err(error) = self.events.save_transition(&record) {
                log::error!(
                    "[agents] handoff {} was reconciled as {} and the row could not be updated: \
                     {error}",
                    record.transition_id,
                    record.phase.as_str()
                );
            }
            settled.push(record);
        }
        settled
    }

    // -- Opening ----------------------------------------------------------

    /// Writes the row, before anything is touched.
    fn open(
        &self,
        session: &Session,
        request: &HandoffRequest,
        agent: &AgentDefinition,
    ) -> Result<TransitionRecord, String> {
        let target = self.registry.find(&request.target_model_id).ok_or_else(|| {
            ValidationRefusal::NotRegistered {
                model_id: request.target_model_id.clone(),
            }
            .explain()
        })?;

        // The model the agent is on now. `unregistered` when its entry has
        // since been removed, and also when it has no default at all — in which
        // case routing was choosing per turn and the digest is over a sentinel
        // rather than over any bytes. `basis` says so either way, so nothing
        // reads the digest as a claim about a file.
        let from_model = match agent.models.default_model_id.as_deref() {
            Some(id) => self
                .registry
                .find(id)
                .map(ModelFingerprint::of)
                .unwrap_or_else(|| ModelFingerprint::unregistered(id)),
            None => ModelFingerprint::unregistered("(no default; routed by role)"),
        };

        let mut opened = TransitionRecord::begin(
            format!("tr-{}", uuid::Uuid::new_v4()),
            agent,
            from_model,
            ModelFingerprint::of(target),
            agent.models.rebound_to(&request.target_model_id),
            &session.user.id,
            &request.reason,
        );
        opened.run_id = request.run_id.clone();
        opened.task_id = request.task_id.clone().or_else(|| request.run_id.clone());

        // Written before anything is touched. The row is what makes the partial
        // unique index refuse a genuine race, and what the recovery sweep finds
        // if the process dies on the next line.
        self.events.save_transition(&opened)?;
        Ok(opened)
    }

    // -- Driving ----------------------------------------------------------

    /// Everything after the record exists. Never returns an error: it records
    /// one.
    async fn drive(
        &self,
        session: &Session,
        request: &HandoffRequest,
        agent: &AgentDefinition,
        record: &mut TransitionRecord,
    ) -> TransitionRecord {
        if let Err(failure) = self.attempt(session, request, agent, record).await {
            self.settle_failure(&session.user.id, Some(session), record, failure)
                .await;
        }
        // Best effort, and logged rather than propagated: the record in memory
        // is what the caller is about to be shown, and a write that failed here
        // leaves a stale row rather than a wrong answer. The recovery sweep
        // finds an unsettled row and settles it.
        if let Err(error) = self.events.save_transition(record) {
            log::error!(
                "[agents] handoff {} reached {} and the row could not be updated: {error}",
                record.transition_id,
                record.phase.as_str()
            );
        }
        record.clone()
    }

    /// The phases, in order.
    async fn attempt(
        &self,
        session: &Session,
        request: &HandoffRequest,
        agent: &AgentDefinition,
        record: &mut TransitionRecord,
    ) -> Result<(), Failure> {
        // -- Drain -------------------------------------------------------
        if record.phase == TransitionPhase::AwaitingSettlement {
            // A retry, after somebody settled what was in flight.
            record.awaiting.clear();
            record.because = None;
            self.step(record, TransitionPhase::Draining, None)?;
        } else if record.phase == TransitionPhase::Requested {
            self.step(record, TransitionPhase::Draining, None)?;
        }

        match self.drain_verdict(request)? {
            DrainVerdict::Safe => {}
            DrainVerdict::Drain { because } => {
                // Left open at `draining`. The run is finishing a round or a
                // tool; the same request repeated once it reaches a boundary
                // continues from here. Deliberately not a wait loop: an IPC
                // command that blocked for the length of a model round would
                // hold the connection for minutes and still time out.
                record.because = Some(format!(
                    "{because}. Ask again once it reaches a boundary — nothing has been changed."
                ));
                return Ok(());
            }
            DrainVerdict::NeedsSettlement { because, keys } => {
                record.awaiting = keys;
                record.because = Some(because);
                self.step(record, TransitionPhase::AwaitingSettlement, None)?;
                return Ok(());
            }
        }

        // -- Checkpoint and freeze ---------------------------------------
        let frozen = self.freeze(session, request, record)?;

        // -- Validate ----------------------------------------------------
        self.step(record, TransitionPhase::Validating, None)?;
        let target = self
            .registry
            .find(&request.target_model_id)
            .ok_or_else(|| {
                Failure::refused(ValidationRefusal::NotRegistered {
                    model_id: request.target_model_id.clone(),
                })
            })?
            .clone();

        validate_target(
            &target,
            agent.models.default_model_id.as_deref(),
            &BindingRequirements::of(agent),
        )
        .map_err(Failure::refused)?;

        // What the source model makes of the probe string, asked while it is
        // still warm. Admission a few lines below may evict it to make room,
        // and after that there is nothing left to ask.
        let source_probe = self.probe_source(agent).await;

        if request.verify == VerifyDestination::PlanOnly {
            record.because = Some(
                "The destination was checked against this agent's policy, role, classification \
                 ceiling and tools, and was not loaded. Its served window is therefore not known \
                 yet and no context was recompiled — there was no work in flight that needed it."
                    .into(),
            );
            self.step(record, TransitionPhase::CommitPending, None)?;
            return self.commit(session, request, record, None).await;
        }

        // -- Load and health-check ---------------------------------------
        self.step(record, TransitionPhase::Loading, None)?;
        // Held until this attempt returns: the load, the health check and the
        // recompile all happen with nothing else generating.
        let _load_lease = self
            .lease_for(&format!("handoff:{}", record.transition_id), &target.id)
            .await?;
        let capabilities =
            crate::ai_engine::gguf_meta::capabilities(&self.models_dir.join(&target.path));

        let admitted = crate::serving::admission::admit(self.servers, &target, self.models_dir)
            .await
            .map_err(|error| {
                Failure::refused(ValidationRefusal::AdmissionRefused {
                    model_id: target.id.clone(),
                    detail: error.to_string(),
                })
            })?;
        // Written onto the record before anything else can fail, because it is
        // what a rollback reads to decide whether the source has to be served
        // again.
        record.released = admitted.released.clone();
        record.released_in_process = admitted.released_in_process;
        if !admitted.released.is_empty() || admitted.released_in_process {
            // Said out loud, because this is the case where a rollback has to
            // put something back rather than merely stop something.
            log::info!(
                "[agents] handoff {}: {} was admitted by releasing {}{}",
                record.transition_id,
                target.id,
                admitted.released.join(", "),
                if admitted.released_in_process {
                    " and the in-process model"
                } else {
                    ""
                }
            );
        }

        let endpoint = self
            .servers
            .endpoint_for(&target, self.models_dir, &admitted.plan)
            .await
            .map_err(|error| {
                Failure::refused(ValidationRefusal::NeverHealthy {
                    model_id: target.id.clone(),
                    detail: error.to_string(),
                })
            })?;

        let served = self
            .servers
            .health_check(
                &endpoint,
                target.context_length,
                capabilities.supports_toggled_reasoning,
            )
            .await;
        if !served.healthy {
            return Err(Failure::refused(ValidationRefusal::NeverHealthy {
                model_id: target.id.clone(),
                detail: format!("{} did not answer a readiness probe", endpoint.base_url),
            }));
        }

        record.tokenizer_changed = match (source_probe.as_ref(), served.tokenizer.as_ref()) {
            (Some(before), Some(after)) => Some(before.differs_from(after)),
            // One of them would not answer. Left unknown rather than reported
            // as unchanged: an unasked question and a negative answer are not
            // the same fact.
            _ => None,
        };
        record.served = Some(served.clone());

        // -- Recompile for the window it actually came up with -----------
        self.step(record, TransitionPhase::Recompiling, None)?;
        let recompiled = self.recompile(session, request, record, &frozen, &served)?;

        // -- Commit ------------------------------------------------------
        self.step(record, TransitionPhase::CommitPending, None)?;
        self.commit(session, request, record, recompiled).await
    }

    // -- The phases, one method each --------------------------------------

    /// The card, for a load this handoff is about to perform.
    async fn lease_for(
        &self,
        owner: &str,
        model_id: &str,
    ) -> Result<Option<crate::subagents::LeaseGuard>, Failure> {
        let Some(leases) = self.leases else {
            return Ok(None);
        };
        leases
            .acquire(crate::subagents::LeaseRequest::new(
                owner,
                crate::subagents::LeaseClass::Parent,
                model_id,
            ))
            .await
            .map(Some)
            .map_err(|refusal| {
                Failure::refused(ValidationRefusal::AdmissionRefused {
                    model_id: model_id.to_string(),
                    detail: refusal.explain(),
                })
            })
    }

    fn step(
        &self,
        record: &mut TransitionRecord,
        next: TransitionPhase,
        detail: Option<String>,
    ) -> Result<(), Failure> {
        record.advance(next, detail).map_err(Failure::defect)
    }

    /// Whether the run may be handed over from where it is.
    fn drain_verdict(&self, request: &HandoffRequest) -> Result<DrainVerdict, Failure> {
        let Some(run_id) = request.run_id.as_deref() else {
            // No run. Nothing to drain, nothing in flight, nothing to settle.
            return Ok(DrainVerdict::Safe);
        };

        let snapshot = self
            .events
            .snapshot(run_id)
            .map_err(Failure::storage)?
            .ok_or_else(|| {
                Failure::storage(format!(
                    "{run_id} has no recorded state, so there is nothing to hand over"
                ))
            })?;

        let unsettled: Vec<String> = self
            .events
            .unsettled_effects_for_run(run_id)
            .map_err(Failure::storage)?
            .into_iter()
            .map(|effect| effect.idempotency_key)
            .collect();

        let pending: Vec<String> = self
            .events
            .approvals_for_run(run_id)
            .map_err(Failure::storage)?
            .into_iter()
            .filter(|approval| approval.status == ApprovalStatus::Pending)
            .map(|approval| approval.approval_id)
            .collect();

        Ok(drain_for(snapshot.state, &unsettled, &pending))
    }

    /// Writes the task state down and fixes the revision the handoff is against.
    ///
    /// Before anything is loaded and before anything is evicted. On a card that
    /// holds one model the next phase *will* unload the source, and a checkpoint
    /// taken after that is a checkpoint that may never be taken.
    fn freeze(
        &self,
        session: &Session,
        request: &HandoffRequest,
        record: &mut TransitionRecord,
    ) -> Result<SourceFreeze, Failure> {
        let graph_revision = match self.graph {
            Some(graph) => graph
                .graph_revision()
                .map_err(|error| Failure::storage(error.explain()))?,
            // No graph on this deployment. Recorded as -1 rather than 0, which
            // is a real revision: a cursor saying "there was no changefeed" must
            // not read back as "the changefeed was at the beginning".
            None => -1,
        };

        let Some(run_id) = request.run_id.clone() else {
            let freeze = SourceFreeze {
                graph_revision,
                last_event_seq: -1,
                source_manifest_hash: None,
                checkpoint_hash: None,
                at: chrono::Utc::now().to_rfc3339(),
            };
            record.freeze = Some(freeze.clone());
            self.step(
                record,
                TransitionPhase::Checkpointed,
                Some("There is no run to hand over, so there was no task state to write down.".into()),
            )?;
            return Ok(freeze);
        };

        let seed = self.seed_for(&run_id)?.ok_or_else(|| {
            // A run with no seed started before seeds existed, or its start did
            // not complete. Either way no checkpoint can be assembled, and one
            // built from defaults would claim a world nobody observed — the
            // rule `RuntimeDeps::checkpoints` already documents.
            Failure::storage(format!(
                "{run_id} has no checkpoint seed on this process, so its state cannot be written \
                 down before the model is changed. Nothing was touched. Let the run finish, or \
                 stop it, and reassign the model then."
            ))
        })?;

        let snapshot = self
            .events
            .snapshot(&run_id)
            .map_err(Failure::storage)?
            .ok_or_else(|| Failure::storage(format!("{run_id} has no recorded state")))?;

        let checkpoint = seed.checkpoint(
            &run_id,
            snapshot.state,
            snapshot.seq,
            seed.committed_notes.clone(),
            None,
            seed.manifest.clone(),
            // Empty by construction: `drain_verdict` refused anything else a
            // few lines ago, so there is nothing unsettled left to carry.
            Vec::new(),
        );
        let landed =
            super::resume::checkpoint_now(self.events, &checkpoint).map_err(Failure::storage)?;
        if !landed {
            // The store refused it — a newer checkpoint already stands, or this
            // one would have emptied a fuller record. Treated as a failure of
            // the handoff rather than shrugged off: proceeding would unload the
            // source model believing in a resume point this handoff did not
            // establish.
            return Err(Failure::storage(format!(
                "{run_id}'s state could not be saved before the model change (a newer checkpoint \
                 already stands). Nothing was touched."
            )));
        }

        let freeze = SourceFreeze {
            graph_revision,
            last_event_seq: snapshot.seq,
            source_manifest_hash: seed
                .manifest
                .as_ref()
                .map(|manifest| manifest.manifest_hash.clone()),
            checkpoint_hash: Some(checkpoint.checkpoint_hash.clone()),
            at: chrono::Utc::now().to_rfc3339(),
        };
        record.source_manifest_hash = freeze.source_manifest_hash.clone();
        record.attempt_id = Some(seed.attempt_id.clone());
        record.freeze = Some(freeze.clone());
        self.step(record, TransitionPhase::Checkpointed, None)?;

        // The point in the run's own history where it stopped being on the
        // model that produced the events before it. Recorded here rather than
        // at the commit, because this is the moment the run actually stopped.
        let _ = self.events.record(
            EventDraft::new(
                &run_id,
                TaskEventType::ModelTransitionStarted,
                &session.user.id,
            )
            .with(json!({
                "transitionId": record.transition_id,
                "agentId": record.agent_id,
                "fromModelId": record.from_model.model_id,
                "toModelId": record.to_model.model_id,
                "fromModelDigest": record.from_model.digest,
                "toModelDigest": record.to_model.digest,
                "fromDefinitionVersion": record.from_definition_version,
                "graphRevision": freeze.graph_revision,
                "checkpointHash": freeze.checkpoint_hash,
                "state": snapshot.state.as_str(),
            })),
        );

        Ok(freeze)
    }

    /// Rebuilds the task's context for the target's real window.
    ///
    /// Returns the manifest to put on the run's seed once the binding lands, or
    /// `None` when there was no run to rebuild for.
    fn recompile(
        &self,
        session: &Session,
        request: &HandoffRequest,
        record: &mut TransitionRecord,
        frozen: &SourceFreeze,
        served: &ServedBinding,
    ) -> Result<Option<ContextManifest>, Failure> {
        let Some(run_id) = request.run_id.clone() else {
            return Ok(None);
        };
        let Some(graph) = self.graph else {
            return Err(Failure::storage(
                "this deployment has no runtime memory graph, so the task's context cannot be \
                 rebuilt for a different window. Nothing was rebound."
                    .to_string(),
            ));
        };

        let seed = self
            .seed_for(&run_id)?
            .ok_or_else(|| Failure::storage(format!("{run_id} has no checkpoint seed")))?;

        let Some(source_manifest) = seed.manifest.clone() else {
            return Err(Failure::storage(format!(
                "{run_id} has no record of what its turn was built from, so there is nothing to \
                 rebuild for a different window. Nothing was rebound."
            )));
        };

        // The reserves the interrupted turn was actually budgeted with, or the
        // ones the caller stated. Never a default: a reserve nobody measured,
        // used to decide whether a context fits, is a plausible number standing
        // in for a measured one.
        let reserves = match request.reserves {
            Some(stated) => stated,
            None => match source_manifest.budget.as_ref() {
                Some(budget) => ReserveRequest {
                    tool_schemas: budget.reserved_tool_schemas,
                    output: budget.reserved_output,
                    framing: budget.reserved_framing,
                    safety: budget.reserved_safety,
                },
                None => {
                    return Err(Failure::storage(format!(
                        "{run_id}'s turn was recorded without what it reserved for tool schemas, \
                         its reply and the chat template, so there is no measured figure to \
                         budget the new model's window against. Nothing was rebound — a guessed \
                         reserve here would decide whether an operator's correction fits."
                    )))
                }
            },
        };

        let frozen_scope = FrozenScope {
            task_id: record.task_id.clone().unwrap_or_else(|| run_id.clone()),
            agent_id: record.agent_id.clone(),
            definition_version: record.from_definition_version,
            // The frozen cursor, not a fresh one. See
            // `ContextCompiler::recompile_for_binding`.
            graph_revision: frozen.graph_revision,
            model_id: record.from_model.model_id.clone(),
            template_id: None,
            served_window: source_manifest.served_window,
            project_id: request.project_id.clone(),
            // What the agent does, from its definition, so the procedures that
            // applied before the handoff are the ones that apply after it.
            capability: self
                .agents
                .resolve_for_provenance(&record.agent_id)
                .ok()
                .and_then(|agent| crate::subagents::capability_for(agent.output_schema))
                .map(str::to_string),
            now: chrono::Utc::now().to_rfc3339(),
        };

        // The source manifest with the model and the window replaced, and
        // everything it read left alone. The documents, the notebook selection
        // and the history it carried are what make this the same turn.
        let mut base = source_manifest.clone();
        base.model_id = record.to_model.model_id.clone();
        base.served_window = served.served_window;

        let compiled = ContextCompiler::new(graph)
            .recompile_for_binding(
                session,
                &frozen_scope,
                &record.to_model.model_id,
                served.template_id.clone(),
                served.served_window,
                base,
                // The objective, which is what lexical ranking is against. Not
                // the original prompt: the prompt is on the snapshot and the
                // goal is what Rust accepted as the run's objective.
                &seed.committed_notes.goal,
                &BTreeSet::new(),
                Reserves {
                    window_source: Some(served.window_source.as_str().to_string()),
                    ..reserves.as_reserves()
                },
            )
            .map_err(|error| Failure::storage(error.explain()))?;

        // The refusal this whole phase exists for, checked against the measured
        // window — which is why it cannot be a pre-flight check. See
        // `ValidationRefusal::is_pre_flight`.
        fits_mandatory(
            &record.to_model.model_id,
            served.served_window,
            compiled.mandatory_tokens(),
            reserves.total(),
        )
        .map_err(Failure::refused)?;
        if compiled.mandatory_overflowed {
            return Err(Failure::refused(ValidationRefusal::ContextTooSmall {
                model_id: record.to_model.model_id.clone(),
                affords: served.served_window.saturating_sub(reserves.total()),
                mandatory: compiled.mandatory_tokens(),
            }));
        }

        // What the new model is given, rebuilt rather than transferred.
        //
        // The artifact list is the run's own `artifact_ids`, which
        // `state_commit` admits only for artifacts this run's produced-file
        // table holds. Carried so the record can be checked for having left
        // them alone: a handoff that changed one would have rewritten the
        // lineage of a deliverable.
        let portable = PortableState::new(
            seed.committed_notes.clone(),
            compiled
                .manifest
                .graph
                .as_ref()
                .map(|binding| binding.selected.clone())
                .unwrap_or_default(),
            source_references(&source_manifest),
            seed.committed_notes.completed.clone(),
            seed.committed_notes.artifact_ids.clone(),
        );

        record.target_manifest_hash = Some(compiled.manifest.manifest_hash.clone());
        record.portable_state_hash = Some(portable.state_hash.clone());
        record.artifact_hashes = portable.artifact_hashes.clone();

        Ok(Some(compiled.manifest))
    }

    /// Writes the binding, and only then moves the run's resume point onto it.
    async fn commit(
        &self,
        session: &Session,
        request: &HandoffRequest,
        record: &mut TransitionRecord,
        recompiled: Option<ContextManifest>,
    ) -> Result<(), Failure> {
        let mutation = self
            .agents
            .rebind_model(
                session,
                &record.agent_id,
                record.from_definition_version,
                record.to_binding.clone(),
                &record.transition_id,
            )
            .map_err(|error| match error {
                // A concurrent edit between the validation and the write. The
                // target was checked against a definition that is no longer
                // current, so committing would bind a model nobody checked
                // against the rules now in force.
                RegistryError::VersionConflict { .. } => Failure::conflict(error.explain()),
                other => Failure::storage(other.explain()),
            })?;

        // The registry's own answer about which version it is at now, rather
        // than the arithmetic this record did when it opened. They agree in the
        // ordinary case; when they do not, the registry is right.
        record.to_definition_version = mutation.definition_version;
        self.step(record, TransitionPhase::Committed, None)?;

        // The resume point moves onto the target, after the binding.
        //
        // A crash between these two writes leaves the registry on the target
        // and the run's checkpoint on the source. That is safe, and it is the
        // documented pinning rule rather than a lost update: `PinnedDefinition`
        // says an edit takes effect at the next boundary, and this run's next
        // boundary is its resumption. It would continue on the model it was on,
        // and the new binding would apply to the run after it.
        //
        // The event goes first so the checkpoint below is taken at a sequence
        // past the one the freeze wrote — `save_checkpoint` is sequence-guarded
        // and would otherwise drop it as a write that could move the resume
        // point backwards.
        if let Some(run_id) = request.run_id.as_deref() {
            let _ = self.events.record(
                EventDraft::new(
                    run_id,
                    TaskEventType::ModelTransitionCommitted,
                    &session.user.id,
                )
                .with(json!({
                    "transitionId": record.transition_id,
                    "agentId": record.agent_id,
                    "taskId": record.task_id,
                    "attemptId": record.attempt_id,
                    "fromModelId": record.from_model.model_id,
                    "toModelId": record.to_model.model_id,
                    "fromModelDigest": record.from_model.digest,
                    "toModelDigest": record.to_model.digest,
                    "modelDigestBasis": record.to_model.basis,
                    "fromDefinitionVersion": record.from_definition_version,
                    "toDefinitionVersion": record.to_definition_version,
                    "graphRevision": record.freeze.as_ref().map(|freeze| freeze.graph_revision),
                    "sourceManifestHash": record.source_manifest_hash,
                    "targetManifestHash": record.target_manifest_hash,
                    "portableStateHash": record.portable_state_hash,
                    "artifactHashes": record.artifact_hashes,
                    "servedWindow": record.served.as_ref().map(|served| served.served_window),
                    "windowSource": record
                        .served
                        .as_ref()
                        .map(|served| served.window_source.as_str()),
                    "tokenizerChanged": record.tokenizer_changed,
                    "droppedState": DroppedState::ALL
                        .iter()
                        .map(|dropped| dropped.as_str())
                        .collect::<Vec<_>>(),
                    "outcome": record.outcome().as_str(),
                })),
            );

            self.move_resume_point(run_id, record, recompiled);
        }
        Ok(())
    }

    /// Puts the recompiled manifest and the target model onto the run's resume
    /// point, in memory and on disk.
    ///
    /// Both, and the disk half is the one that matters. The seed lives in this
    /// process and dies with it; the checkpoint is what a resumption after a
    /// restart reads, and a resumption reads the *model* off it — see
    /// `commands::agent`'s `Continuation`. Without this write the binding would
    /// have moved while every resumption went on choosing the old model, which
    /// is the same silent divergence between configuration and behaviour this
    /// module exists to remove.
    fn move_resume_point(
        &self,
        run_id: &str,
        record: &TransitionRecord,
        recompiled: Option<ContextManifest>,
    ) {
        let updated = {
            let Ok(mut seeds) = self.checkpoints.lock() else {
                log::error!(
                    "[agents] handoff {}: the checkpoint seeds were locked, so {run_id}'s resume \
                     point still names {}. The binding moved; the run will continue on the model \
                     it was on and the new binding applies to the next run.",
                    record.transition_id,
                    record.from_model.model_id
                );
                return;
            };
            let Some(seed) = seeds.get_mut(run_id) else {
                // No seed: the run is not live in this process. Nothing to
                // move, and the binding applies to the next run either way.
                return;
            };
            seed.model_id = record.to_model.model_id.clone();
            if let Some(manifest) = recompiled {
                seed.manifest = Some(manifest);
            }
            seed.clone()
        };

        // Taken at the sequence the commit event just reached, so the
        // sequence guard in `save_checkpoint` accepts it over the one the
        // freeze wrote.
        let Ok(Some(snapshot)) = self.events.snapshot(run_id) else {
            log::error!(
                "[agents] handoff {}: {run_id}'s state could not be read back, so its saved \
                 resume point still names {}. The run would continue on that model; the binding \
                 applies from the next run.",
                record.transition_id,
                record.from_model.model_id
            );
            return;
        };
        let checkpoint = updated.checkpoint(
            run_id,
            snapshot.state,
            snapshot.seq,
            updated.committed_notes.clone(),
            None,
            updated.manifest.clone(),
            Vec::new(),
        );
        match super::resume::checkpoint_now(self.events, &checkpoint) {
            Ok(true) => {}
            // Not an error for the handoff: the binding is committed and the
            // work is safe either way. Said out loud because it decides which
            // model a resumption picks, and somebody reading a continuation
            // that ran on the old model deserves to find this line.
            Ok(false) => log::warn!(
                "[agents] handoff {}: a newer checkpoint already stands for {run_id}, so its \
                 saved resume point still names {}. The binding is committed; a resumption would \
                 continue on the older model until the run checkpoints again.",
                record.transition_id,
                record.from_model.model_id
            ),
            Err(error) => log::error!(
                "[agents] handoff {}: {run_id}'s resume point could not be moved onto {}: \
                 {error}",
                record.transition_id,
                record.to_model.model_id
            ),
        }
    }

    // -- Endings ----------------------------------------------------------

    /// Records why a handoff stopped, and undoes what needs undoing.
    async fn settle_failure(
        &self,
        actor: &str,
        authority: Option<&Session>,
        record: &mut TransitionRecord,
        failure: Failure,
    ) {
        let because = failure.explain();
        record.because = Some(because.clone());
        if let Failure::Refused(refusal) = &failure {
            record.refusal = Some(refusal.clone());
        }

        if !failure.needs_rollback(record.phase) {
            if let Err(detail) = record.advance(TransitionPhase::Failed, Some(because)) {
                log::error!("[agents] handoff {}: {detail}", record.transition_id);
            }
            self.note_abandoned(actor, record, TaskEventType::ModelTransitionFailed);
            return;
        }

        self.undo(authority, record).await;
        self.note_abandoned(actor, record, TaskEventType::ModelTransitionRolledBack);
    }

    /// The bounded undo: the binding and the source model, and nothing else.
    async fn undo(&self, authority: Option<&Session>, record: &mut TransitionRecord) {
        // The effects the run already had. Carried into the plan so the undo can
        // be *checked* for having kept them rather than trusted to.
        let preserve = self.completed_effects(record);
        let plan = record.rollback_plan(preserve);

        if let Err(detail) = record.advance(TransitionPhase::RollingBack, None) {
            log::error!("[agents] handoff {}: {detail}", record.transition_id);
            return;
        }

        match self.apply_rollback(authority, record, &plan).await {
            Ok(()) => {
                // The assertion, not an assumption. A rollback that lost a
                // receipt would let the resumed run write the same document a
                // second time.
                let after = self
                    .completed_notes(record)
                    .unwrap_or_else(|| super::memory::RunMemory {
                        completed: plan.preserve_effects.clone(),
                        ..Default::default()
                    });
                match plan.preserves(&after) {
                    Ok(()) => {
                        let _ = record.advance(TransitionPhase::RolledBack, None);
                    }
                    Err(lost) => {
                        let detail = format!(
                            "{} completed action(s) are no longer in this run's notes after the \
                             undo, starting with {}. A resumption could repeat them, so this \
                             needs a person rather than being reported as a clean rollback.",
                            lost.len(),
                            lost.first()
                                .map(|effect| effect.target.clone())
                                .unwrap_or_default()
                        );
                        record.because = Some(detail.clone());
                        let _ = record.advance(TransitionPhase::RollbackFailed, Some(detail));
                    }
                }
            }
            Err(detail) => {
                record.because = Some(format!(
                    "{} The agent could not be put back on {}: {detail}",
                    record.because.clone().unwrap_or_default(),
                    record.from_model.model_id
                ));
                let _ = record.advance(TransitionPhase::RollbackFailed, Some(detail));
            }
        }
    }

    /// Puts the binding and the source model back.
    async fn apply_rollback(
        &self,
        authority: Option<&Session>,
        record: &TransitionRecord,
        plan: &RollbackPlan,
    ) -> Result<(), String> {
        // The binding, when it was written. A transition that failed before the
        // commit never wrote one, and `rebind_model` reports that as a no-op
        // rather than moving the version again.
        let current = self
            .agents
            .resolve_for_provenance(&record.agent_id)
            .map_err(|error| error.explain())?;
        if !current.models.names_same_models(&plan.restore_binding) {
            // The one step in an undo that writes, and therefore the one that
            // needs somebody's authority. The start-up sweep has none — see
            // `reconcile_open` — and cannot reach this branch, because a
            // binding that was written reconciles as `Landed` and is finished
            // rather than undone. Refused rather than performed anyway, so that
            // if it ever is reached the answer is a person rather than a write
            // nobody authorised.
            let Some(authority) = authority else {
                return Err(format!(
                    "{} is bound to {} and putting it back needs an administrator, which a                      recovery sweep is not. Sign in and undo it.",
                    record.agent_id, record.to_model.model_id
                ));
            };
            self.agents
                .rebind_model(
                    authority,
                    &record.agent_id,
                    current.definition_version,
                    plan.restore_binding.clone(),
                    &record.transition_id,
                )
                .map_err(|error| error.explain())?;
        }

        if !plan.reload_source {
            return Ok(());
        }
        let Some(source_id) = plan.restore_binding.default_model_id.as_deref() else {
            // Nothing to put back: routing was choosing per turn.
            return Ok(());
        };
        if self.servers.is_warm(source_id) {
            return Ok(());
        }
        let Some(entry) = self.registry.find(source_id).cloned() else {
            return Err(format!(
                "{source_id} is no longer in the model registry, so it cannot be served again"
            ));
        };

        // The explicit reload-failure path. On a card that holds one model the
        // source was evicted to make room for the target, so putting the
        // binding back is not enough — something has to be serving it.
        // Reloading the source is a load like any other: under the card.
        let _reload_lease = match self.leases {
            Some(leases) => leases
                .acquire(crate::subagents::LeaseRequest::new(
                    format!("handoff-rollback:{}", record.transition_id),
                    crate::subagents::LeaseClass::Parent,
                    entry.id.as_str(),
                ))
                .await
                .ok(),
            None => None,
        };
        let admitted = crate::serving::admission::admit(self.servers, &entry, self.models_dir)
            .await
            .map_err(|error| error.to_string())?;
        self.servers
            .endpoint_for(&entry, self.models_dir, &admitted.plan)
            .await
            .map(|_| ())
            .map_err(|error| error.to_string())
    }

    /// Settles one interrupted handoff.
    async fn reconcile_one(
        &self,
        actor: &str,
        authority: Option<&Session>,
        record: &mut TransitionRecord,
    ) {
        match record.phase {
            // Nothing had been touched. Closed as abandoned so the agent is not
            // blocked from ever being moved again.
            TransitionPhase::Requested
            | TransitionPhase::Draining
            | TransitionPhase::AwaitingSettlement => {
                record.because = Some(
                    "ARJUN restarted before this model change did anything. The agent is on the \
                     model it was on, and its run was never interrupted. Start the change again \
                     when you want it."
                        .into(),
                );
                let _ = record.advance(TransitionPhase::Failed, None);
                self.note_abandoned(actor, record, TaskEventType::ModelTransitionFailed);
            }

            // The one ambiguous window. Read off the registry, three-valued.
            TransitionPhase::CommitPending => {
                let pending = record.pending_commit();
                let current = self.agents.resolve_for_provenance(&record.agent_id);
                let verdict = match &current {
                    Ok(agent) => pending.reconcile(agent),
                    // The agent is gone. Nothing can be read off a record that
                    // is not there, and nothing should be guessed.
                    Err(_) => Reconciliation::Indeterminate,
                };
                match verdict {
                    Reconciliation::Landed => {
                        record.because = Some(format!(
                            "ARJUN restarted just after this change was written. The agent is on \
                             {} at version {}, which is what the change intended.",
                            record.to_model.model_id, pending.next_version
                        ));
                        let _ = record.advance(TransitionPhase::Committed, None);
                        self.note_committed_after_recovery(actor, record);
                    }
                    Reconciliation::DidNotLand => {
                        record.because = Some(format!(
                            "ARJUN restarted before this change was written. The agent is still \
                             on {}, and its last saved state is intact.",
                            record.from_model.model_id
                        ));
                        self.undo(authority, record).await;
                        self.note_abandoned(
                            actor,
                            record,
                            TaskEventType::ModelTransitionRolledBack,
                        );
                    }
                    Reconciliation::Indeterminate => {
                        record.because = Some(format!(
                            "ARJUN restarted while this change was being written, and the agent \
                             has been edited since — it is at version {} and this change expected \
                             {} or {}. Which model it is bound to cannot be read off the record, \
                             so somebody has to look before it is given work.",
                            current
                                .as_ref()
                                .map(|agent| agent.definition_version.to_string())
                                .unwrap_or_else(|_| "unknown".into()),
                            pending.expected_version,
                            pending.next_version
                        ));
                        let _ = record.advance(TransitionPhase::RollingBack, None);
                        let _ = record.advance(TransitionPhase::RollbackFailed, None);
                        self.note_abandoned(
                            actor,
                            record,
                            TaskEventType::ModelTransitionRolledBack,
                        );
                    }
                }
            }

            // Past the checkpoint and before the write. The binding was never
            // changed; what may need putting back is the source model.
            TransitionPhase::Checkpointed
            | TransitionPhase::Validating
            | TransitionPhase::Loading
            | TransitionPhase::Recompiling
            | TransitionPhase::RollingBack => {
                record.because = Some(format!(
                    "ARJUN restarted while this model change was under way ({}). The agent is \
                     still on {}, and the state saved before the change is the point its run \
                     continues from.",
                    record.phase.as_str().replace('_', " "),
                    record.from_model.model_id
                ));
                if record.phase != TransitionPhase::RollingBack {
                    self.undo(authority, record).await;
                } else {
                    // It was already undoing when the process went away. Finish
                    // what it started rather than starting a second undo.
                    let plan = record.rollback_plan(self.completed_effects(record));
                    match self.apply_rollback(authority, record, &plan).await {
                        Ok(()) => {
                            let _ = record.advance(TransitionPhase::RolledBack, None);
                        }
                        Err(detail) => {
                            let _ =
                                record.advance(TransitionPhase::RollbackFailed, Some(detail));
                        }
                    }
                }
                self.note_abandoned(actor, record, TaskEventType::ModelTransitionRolledBack);
            }

            // `open_transitions` only returns unsettled rows, so a terminal
            // phase here would mean a row whose `settled_at` was never written.
            // Left alone and said out loud rather than moved: the phase is the
            // answer, and the column is a copy of it.
            TransitionPhase::Committed
            | TransitionPhase::RolledBack
            | TransitionPhase::Failed
            | TransitionPhase::RollbackFailed => {
                log::warn!(
                    "[agents] handoff {} is recorded as unsettled and its phase is {}; the phase \
                     is authoritative and the row is being closed to match",
                    record.transition_id,
                    record.phase.as_str()
                );
                record.settled_at = Some(chrono::Utc::now().to_rfc3339());
            }
        }
    }

    // -- Small readers ----------------------------------------------------

    fn seed_for(&self, run_id: &str) -> Result<Option<CheckpointSeed>, Failure> {
        let held = self
            .checkpoints
            .lock()
            .map_err(|_| Failure::storage("the checkpoint seeds were left locked".to_string()))?;
        Ok(held.get(run_id).cloned())
    }

    /// The notes this run's seed currently holds.
    fn completed_notes(&self, record: &TransitionRecord) -> Option<super::memory::RunMemory> {
        let run_id = record.run_id.as_deref()?;
        self.checkpoints
            .lock()
            .ok()?
            .get(run_id)
            .map(|seed| seed.committed_notes.clone())
    }

    /// The side effects this run is already known to have had.
    fn completed_effects(&self, record: &TransitionRecord) -> Vec<CompletedEffect> {
        self.completed_notes(record)
            .map(|notes| notes.completed)
            .unwrap_or_default()
    }

    /// What the source model makes of the probe string, while it is still warm.
    async fn probe_source(
        &self,
        agent: &AgentDefinition,
    ) -> Option<super::model_transition::TokenizerProbe> {
        let id = agent.models.default_model_id.as_deref()?;
        let endpoint = self.servers.warm_endpoint(id)?;
        let declared = self
            .registry
            .find(id)
            .map(|entry| entry.context_length)
            .unwrap_or(0);
        self.servers
            .health_check(&endpoint, declared, false)
            .await
            .tokenizer
    }

    fn note_committed_after_recovery(&self, actor: &str, record: &TransitionRecord) {
        let Some(run_id) = record.run_id.as_deref() else {
            return;
        };
        let _ = self.events.record(
            EventDraft::new(run_id, TaskEventType::ModelTransitionCommitted, actor).with(json!({
                "transitionId": record.transition_id,
                "agentId": record.agent_id,
                "toModelId": record.to_model.model_id,
                "toDefinitionVersion": record.to_definition_version,
                "reconciled": Reconciliation::Landed.as_str(),
                "because": record.because,
            })),
        );
    }

    fn note_abandoned(&self, actor: &str, record: &TransitionRecord, kind: TaskEventType) {
        let Some(run_id) = record.run_id.as_deref() else {
            return;
        };
        let _ = self.events.record(
            EventDraft::new(run_id, kind, actor).with(json!({
                "transitionId": record.transition_id,
                "agentId": record.agent_id,
                "fromModelId": record.from_model.model_id,
                "toModelId": record.to_model.model_id,
                "phase": record.phase.as_str(),
                "outcome": record.outcome().as_str(),
                "because": record.because,
                "refusal": record.refusal,
                // Whether the run's resume point is still the one it had. A
                // rollback never rewrites it, which is what makes "the last
                // committed task state is preserved" a checkable claim rather
                // than a promise.
                "keptCheckpointHash": record
                    .freeze
                    .as_ref()
                    .and_then(|freeze| freeze.checkpoint_hash.clone()),
            })),
        );
    }
}

/// Every content address the source turn read.
///
/// Documents by hash, and the notebook sources the person explicitly chose. A
/// `SourceSelection::All` was already resolved to a concrete list when the turn
/// was frozen, so what comes back pins the sources the original turn actually
/// read rather than whatever is in the notebook now.
fn source_references(manifest: &ContextManifest) -> Vec<String> {
    let mut out: Vec<String> = manifest
        .documents
        .iter()
        .map(|document| document.sha256.clone())
        .collect();
    if let Some(research) = manifest.research.as_ref() {
        if let Some(chosen) = research.selection.explicit_list() {
            out.extend(chosen.iter().cloned());
        }
        out.extend(research.node_ids.iter().cloned());
        out.extend(research.assertion_ids.iter().cloned());
    }
    out.extend(manifest.content_hashes.iter().cloned());
    out.sort();
    out.dedup();
    out
}
