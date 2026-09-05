//! Starting a child, holding it to its limits, and recording what happened.
//!
//! ## What the manager owns and what it does not
//!
//! It owns the **lifecycle**: deciding whether a child may exist at all,
//! enforcing the concurrency lanes, applying the deadline, recording the start
//! and the stop, and keeping the idempotency ledger. It does not own the work.
//! That is a [`ChildWorker`], which the manager calls and does not trust.
//!
//! The split matters for one reason: everything security-relevant is on this
//! side of it. A worker cannot widen its policy, extend its deadline, escape
//! its lane or report a status the manager did not set, because it is handed a
//! finished [`EffectivePolicy`] and its return value passes back through here.
//! A buggy or hostile worker gets a wrong *answer* into the run — which is what
//! the parent's verification is for — and not a wrong *permission*.
//!
//! ## Two lanes
//!
//! Requirement 5. Read-only workers share a semaphore and run several at once:
//! they cannot affect each other's results, and the operator waits for the
//! slowest rather than the sum. Writers and approval-sensitive workers take an
//! exclusive lock, because two writers to one workspace have an order and it
//! should not be whichever finished last, and because an approver shown three
//! requests at once cannot tell which belongs to what.
//!
//! ## Idempotency
//!
//! A key gets a slot, and the slot is what a second caller waits on. So two
//! attempts at one piece of work never both run: the second blocks until the
//! first has an answer, then returns that answer. A retry after an ambiguous
//! failure finds the child rather than starting a second one.

use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;
use chrono::Utc;
use serde_json::json;
use tokio::sync::{Mutex, Semaphore};

use crate::agent_runtime::events::{EventDraft, TaskEventLog, TaskEventType};

use super::certification::Decision;
use super::inherit::{EffectivePolicy, InheritRefusal, InheritedPolicy};
use super::packet::{ChildTaskPacket, InputRef};
use super::profile::AgentProfile;
use super::result::{ChildResult, ChildStatus};

/// How many read-only workers may run at once.
///
/// Four rather than unbounded: each one holds a model turn and a share of the
/// machine, and a parent that fanned out to twenty would make the run slower
/// than doing them in sequence.
pub const MAX_CONCURRENT_READERS: usize = 4;

/// Does a child's actual work.
///
/// Implemented outside the manager so that everything the manager enforces
/// stays enforceable regardless of what a worker does. A worker is handed the
/// packet and the policy it must respect; it cannot obtain a wider one.
#[async_trait]
pub trait ChildWorker: Send + Sync {
    /// The profile this worker serves.
    fn profile(&self) -> &str;

    /// Rust selects references from the parent's actual resources, under the
    /// narrowed policy. Workers revalidate these immediately before reading.
    fn handoff_inputs(&self, _policy: &EffectivePolicy) -> Result<Vec<InputRef>, String> { Ok(Vec::new()) }
    fn validate_inputs(&self, _policy: &EffectivePolicy, _inputs: &[InputRef]) -> Result<(), String> { Ok(()) }
    fn result_evidence(&self, _policy: &EffectivePolicy, _result: &ChildResult) -> Result<Vec<crate::knowledge::SearchResult>, String> { Ok(Vec::new()) }
    fn restore_evidence(&self, _policy: &EffectivePolicy, _record: &crate::agent_runtime::events::children::ChildRecord) -> Result<(), String> { Ok(()) }

    /// Runs the work.
    ///
    /// Returning `Err` is a worker saying it failed; the manager turns that
    /// into a [`ChildStatus::Failed`] result rather than letting the error
    /// escape, so a parent always gets a typed answer.
    async fn run(
        &self,
        packet: &ChildTaskPacket,
        policy: &EffectivePolicy,
    ) -> Result<ChildResult, String>;
}

/// Why a child was not started.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpawnRefusal {
    /// No profile of that name.
    UnknownProfile { name: String },
    /// The inherited policy would not permit it.
    Policy { refusal: InheritRefusal },
    /// No worker is registered for the profile.
    ///
    /// A separate refusal from an unknown profile on purpose: the role exists
    /// and is correctly declared, and this build simply cannot perform it.
    NoWorker { profile: String },
    Unavailable { detail: String },
}

impl SpawnRefusal {
    pub fn explain(&self) -> String {
        match self {
            SpawnRefusal::Unavailable { detail } => detail.clone(),
            SpawnRefusal::UnknownProfile { name } => {
                format!("There is no subagent profile called {name:?}.")
            }
            SpawnRefusal::Policy { refusal } => refusal.explain(),
            SpawnRefusal::NoWorker { profile } => format!(
                "The {profile} role is declared but this build has no worker for it, so nothing \
                 was started. Nothing was done and no result exists."
            ),
        }
    }

    /// The refusal as a result, so a parent always has something typed.
    pub fn as_result(&self, child_id: &str, profile: &str, schema: super::profile::SchemaKind) -> ChildResult {
        ChildResult::ended(
            child_id,
            profile,
            ChildStatus::Refused,
            schema,
            Vec::new(),
            self.explain(),
            0,
        )
    }
}

/// What a spawn came to.
#[derive(Debug, Clone, PartialEq)]
pub enum Spawned {
    /// A child ran, and this is its result.
    Fresh(ChildResult),
    /// This exact work had already been done under the same idempotency key.
    /// The existing child's result, unchanged.
    Existing(ChildResult),
}

impl Spawned {
    pub fn result(&self) -> &ChildResult {
        match self {
            Spawned::Fresh(result) | Spawned::Existing(result) => result,
        }
    }

    pub fn is_reused(&self) -> bool {
        matches!(self, Spawned::Existing(_))
    }
}

/// One idempotency slot: the lock a second caller waits on, and the answer.
type Slot = Arc<Mutex<()>>;

/// The Rust side of subagents.
pub struct SubagentManager {
    profiles: BTreeMap<String, AgentProfile>,
    workers: BTreeMap<String, Arc<dyn ChildWorker>>,
    events: Arc<TaskEventLog>,
    /// The read-only lane.
    readers: Arc<Semaphore>,
    /// The lane writers and approval-sensitive workers take one at a time.
    exclusive: Arc<Mutex<()>>,
    /// Idempotency key to slot.
    slots: Mutex<BTreeMap<String, Slot>>,
}

impl SubagentManager {
    pub fn new(profiles: Vec<AgentProfile>, events: Arc<TaskEventLog>) -> Self {
        Self {
            profiles: profiles
                .into_iter()
                .map(|profile| (profile.name.clone(), profile))
                .collect(),
            workers: BTreeMap::new(),
            events,
            readers: Arc::new(Semaphore::new(MAX_CONCURRENT_READERS)),
            exclusive: Arc::new(Mutex::new(())),
            slots: Mutex::new(BTreeMap::new()),
        }
    }

    /// Registers the thing that performs one profile's work.
    pub fn with_worker(mut self, worker: Arc<dyn ChildWorker>) -> Self {
        self.workers.insert(worker.profile().to_string(), worker);
        self
    }

    pub fn profile(&self, name: &str) -> Option<&AgentProfile> {
        self.profiles.get(name)
    }

    pub fn profiles(&self) -> impl Iterator<Item = &AgentProfile> {
        self.profiles.values()
    }

    /// Whether this build can actually perform a role.
    pub fn has_worker(&self, profile: &str) -> bool {
        self.workers.contains_key(profile)
    }

    pub fn handoff_inputs(&self, profile: &str, inherited: &InheritedPolicy) -> Result<Vec<InputRef>, String> {
        let (_, policy) = self.plan(profile, inherited, "handoff").map_err(|e| e.explain())?;
        self.workers.get(profile).ok_or("No worker is registered for this profile.")?.handoff_inputs(&policy)
    }

    /// Works out what a child would be permitted, without starting it.
    ///
    /// Separate from [`Self::spawn`] so a parent can decide whether a worker is
    /// worth starting, and so the narrowing is testable without a worker.
    pub fn plan(
        &self,
        profile_name: &str,
        inherited: &InheritedPolicy,
        child_id: &str,
    ) -> Result<(AgentProfile, EffectivePolicy), SpawnRefusal> {
        let profile = self
            .profiles
            .get(profile_name)
            .ok_or_else(|| SpawnRefusal::UnknownProfile {
                name: profile_name.to_string(),
            })?;
        let policy = inherited
            .narrow_for(profile, child_id)
            .map_err(|refusal| SpawnRefusal::Policy { refusal })?;
        Ok((profile.clone(), policy))
    }

    /// Starts a child and waits for it.
    ///
    /// Every path through this returns a typed [`ChildResult`]: a refusal, a
    /// failure, a timeout and a success all come back the same shape, so a
    /// parent cannot accidentally handle only the happy one.
    pub async fn spawn(
        &self,
        profile_name: &str,
        inherited: &InheritedPolicy,
        objective: &str,
        inputs: Vec<InputRef>,
        model: Decision,
    ) -> Result<Spawned, SpawnRefusal> {
        let key = super::packet::derive_idempotency_key(
            &inherited_run_id(inherited),
            profile_name,
            objective,
            &inputs,
        );

        // The slot is taken before anything else, so a second attempt at this
        // work waits here rather than starting a second child.
        let slot = {
            let mut slots = self.slots.lock().await;
            Arc::clone(
                slots
                    .entry(key.clone())
                    .or_insert_with(|| Arc::new(Mutex::new(()))),
            )
        };
        let child_id = uuid::Uuid::new_v4().to_string();
        let (profile, policy) = self.plan(profile_name, inherited, &child_id)?;
        let policy_hash = crate::agent_runtime::events::digest(&serde_json::to_string(&(&profile, inherited)).unwrap());
        if let Some(worker) = self.workers.get(profile_name) {
            worker.validate_inputs(&policy, &inputs).map_err(|detail| SpawnRefusal::Unavailable { detail })?;
        }
        let _held = slot.lock().await;
        if let Some(saved) = self.events.child_result(&inherited_run_id(inherited), &key, &policy_hash)
            .map_err(|detail| SpawnRefusal::Unavailable { detail })? {
            if let Some(worker) = self.workers.get(profile_name) {
                worker.restore_evidence(&policy, &saved).map_err(|detail| SpawnRefusal::Unavailable { detail })?;
            }
            return Ok(Spawned::Existing(saved.result));
        }
        let packet = ChildTaskPacket::new(&child_id, inherited_run_id(inherited), &key, objective, inputs, &policy, Utc::now());

        let Some(worker) = self.workers.get(&profile.name).cloned() else {
            let refusal = SpawnRefusal::NoWorker {
                profile: profile.name.clone(),
            };
            let result = refusal.as_result(&child_id, &profile.name, profile.required_schema);
            // Recorded even though nothing ran. A parent that asked for a
            // worker this build does not have should see that in the trace
            // rather than only in a returned error.
            self.events.save_child_result(&crate::agent_runtime::events::children::ChildRecord { packet, result: result.clone(), evidence: Vec::new() }, &policy_hash)
                .map_err(|detail| SpawnRefusal::Unavailable { detail })?;
            self.record_stop(inherited, &child_id, &result, &model);
            return Ok(Spawned::Fresh(result));
        };

        self.record_start(inherited, &packet, &policy, &model);

        // Queueing consumes the same absolute lifetime as execution.
        let budget = (packet.deadline - Utc::now()).to_std().unwrap_or_default();
        let outcome = tokio::time::timeout(budget, async {
            let _reader;
            let _writer;
            if policy.is_concurrent() {
                _reader = self.readers.clone().acquire_owned().await.ok();
            } else {
                _writer = Some(self.exclusive.clone().lock_owned().await);
            }
            worker.run(&packet, &policy).await
        }).await;

        let result = match outcome {
            Ok(Ok(mut produced)) => {
                // The worker's own status is not taken on trust for the fields
                // that decide whether the parent may rely on it. A worker that
                // returned a result answering a different packet is a worker
                // that answered a different question.
                if !produced.answers(&packet) || packet.is_expired(Utc::now()) {
                    produced = ChildResult::ended(
                        &child_id,
                        &profile.name,
                        ChildStatus::Failed,
                        profile.required_schema,
                        Vec::new(),
                        "the worker returned a result for a different task or shape".to_string(),
                        produced.turns_used.min(policy.limits.max_turns),
                    );
                }
                produced
            }
            Ok(Err(detail)) => ChildResult::ended(
                &child_id,
                &profile.name,
                ChildStatus::Failed,
                profile.required_schema,
                Vec::new(),
                detail,
                0,
            ),
            Err(_) => ChildResult::ended(
                &child_id,
                &profile.name,
                ChildStatus::TimedOut,
                profile.required_schema,
                Vec::new(),
                format!(
                    "it reached its {} second limit and was stopped. Anything it had found was \
                     not returned, and the work was not completed.",
                    policy.limits.max_duration_seconds
                ),
                0,
            ),
        };

        let evidence = worker.result_evidence(&policy, &result).map_err(|detail| SpawnRefusal::Unavailable { detail })?;
        self.events.save_child_result(&crate::agent_runtime::events::children::ChildRecord { packet, result: result.clone(), evidence }, &policy_hash)
            .map_err(|detail| SpawnRefusal::Unavailable { detail })?;
        self.record_stop(inherited, &child_id, &result, &model);
        Ok(Spawned::Fresh(result))
    }

    /// Records that a child began, with everything requirement 7 asks for.
    fn record_start(
        &self,
        inherited: &InheritedPolicy,
        packet: &ChildTaskPacket,
        policy: &EffectivePolicy,
        model: &Decision,
    ) {
        let draft = EventDraft::idempotent(
            inherited_run_id(inherited),
            TaskEventType::SubagentStarted,
            &inherited.user_id,
            &packet.child_id,
        )
        .with(json!({
            "childId": packet.child_id,
            "profile": packet.profile,
            "idempotencyKey": packet.idempotency_key,
            // The manifest: what this child was permitted, not what it asked for.
            "manifest": {
                "allowedTools": policy.tools.iter().map(|t| t.as_str()).collect::<Vec<_>>(),
                "refusedTools": policy.refused_tools.iter().map(|t| t.as_str()).collect::<Vec<_>>(),
                "maxTurns": policy.limits.max_turns,
                "maxOutputTokens": policy.limits.max_output_tokens,
                "maxChildren": policy.limits.max_children,
                "isolation": policy.isolation.as_str(),
                "memoryScope": policy.memory_scope.as_str(),
                "writePolicy": policy.write_policy.as_str(),
                "networkPermitted": policy.inherited.network_permitted,
                "classificationCeiling": policy.classification_ceiling.label(),
                "requiredSchema": policy.required_schema.as_str(),
                "depth": policy.inherited.depth,
            },
            "policyHash": packet.policy_hash,
            // The model decision, with the reason it rested on.
            "model": {
                "modelId": model.model_id,
                "role": model.role.label(),
                "cheaperThanParent": model.cheaper_than_parent,
                "reason": model.reason,
            },
            // References only. A packet carries no contents, and neither does
            // its record.
            "inputs": packet.inputs.iter().map(InputRef::describe).collect::<Vec<_>>(),
            "deadline": packet.deadline.to_rfc3339(),
        }));
        self.remember(draft);
    }

    /// Records how a child ended.
    fn record_stop(
        &self,
        inherited: &InheritedPolicy,
        child_id: &str,
        result: &ChildResult,
        model: &Decision,
    ) {
        let draft = EventDraft::idempotent(
            inherited_run_id(inherited),
            TaskEventType::SubagentStopped,
            &inherited.user_id,
            child_id,
        )
        .with(json!({
            "childId": child_id,
            "profile": result.profile,
            // The status is the manager's, and it is what the parent reads.
            // A failure, a timeout and a cancellation are each named rather
            // than folded into a generic ending.
            "status": result.status.as_str(),
            "complete": result.status.is_complete(),
            "findings": result.findings.len(),
            "confidence": result.confidence,
            "uncertainty": result.uncertainty.len(),
            "turnsUsed": result.turns_used,
            "resultHash": result.result_hash,
            "modelId": model.model_id,
        }));
        self.remember(draft);
    }

    fn remember(&self, draft: EventDraft) {
        // Best-effort, like every other durable write: a history that could not
        // be written is a degradation the log reports, not a reason to fail a
        // child that has already done its work.
        if let Err(error) = self.events.record(draft) {
            log::warn!("[subagents] an event was not recorded: {error}");
        }
    }
}

/// The run a policy belongs to.
///
/// A child's workspace root is the run's directory, and its last segment is the
/// run id — so the policy already carries it and there is no second field to
/// fall out of step with the first.
fn inherited_run_id(inherited: &InheritedPolicy) -> String {
    inherited
        .workspace_root
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("unknown-run")
        .to_string()
}
