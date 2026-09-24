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
use crate::orchestrator::tools::ToolName;

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
    /// The registry had no runnable definition for this key.
    ///
    /// Distinct from `UnknownProfile` because the registry was consulted and
    /// answered: the agent may exist and be disabled, or ask for a capability no
    /// worker implements. `detail` says which.
    Unresolved { key: String, detail: String },
    /// A read-only delegation named a role that writes.
    ///
    /// `agent.delegate_readonly` promises the model -- in its name, its
    /// description and its approval class -- that the child it starts cannot
    /// write, produce a document or run code. Until this refusal existed, only
    /// the TypeScript schema kept that promise; the Rust side, which is where
    /// authority is decided, would start `code-worker` through it and hand the
    /// child `workspace.write_text` and `sandbox.run_code` whenever the parent
    /// held them. The effects were still approved call by call, but the
    /// delegation was not, and a tool called read-only was not.
    NeedsWriterDelegation { key: String, isolation: String },
}

impl SpawnRefusal {
    pub fn explain(&self) -> String {
        match self {
            SpawnRefusal::UnknownProfile { name } => {
                format!("There is no subagent profile called {name:?}.")
            }
            SpawnRefusal::Policy { refusal } => refusal.explain(),
            SpawnRefusal::NoWorker { profile } => format!(
                "The {profile} role is declared but this build has no worker for it, so nothing \
                 was started. Nothing was done and no result exists."
            ),
            SpawnRefusal::Unresolved { key, detail } => {
                format!("Nothing was started for {key:?}. {detail}")
            }
            SpawnRefusal::NeedsWriterDelegation { key, isolation } => format!(
                "{key} is declared {isolation}: it writes files or runs code, so a read-only \
                 delegation cannot start it. A job for a role that writes goes through \
                 agent.delegate, which asks a person first. Nothing was started."
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
type Slot = Arc<Mutex<Option<ChildResult>>>;

/// Applies a dispatch's mode to a policy the narrowing already produced.
///
/// Read-only does two things, in this order. A role *declared* as writing is
/// refused, because starting it without its tools would produce a worker that
/// fails for a reason nobody asked it about. Then any tool that is not read-only
/// is removed from what remains and recorded as refused -- so a read-only role
/// an administrator edited to hold a write tool still reaches its worker, still
/// cannot write, and the trace says which tool was withheld and why.
fn enforce_mode(
    resolved: &super::definitions::ResolvedDefinition,
    mut policy: EffectivePolicy,
    mode: DelegationMode,
) -> Result<EffectivePolicy, SpawnRefusal> {
    if mode == DelegationMode::Writer {
        return Ok(policy);
    }
    if resolved.profile.isolation != super::profile::Isolation::ReadOnly {
        return Err(SpawnRefusal::NeedsWriterDelegation {
            key: resolved.agent_id.clone(),
            isolation: resolved.profile.isolation.as_str().to_string(),
        });
    }
    let (kept, withheld): (Vec<ToolName>, Vec<ToolName>) =
        policy.tools.into_iter().partition(|tool| tool.is_read_only());
    policy.tools = kept;
    policy.refused_tools.extend(withheld);
    if policy.tools.is_empty() {
        return Err(SpawnRefusal::NeedsWriterDelegation {
            key: resolved.agent_id.clone(),
            isolation: "with no read-only tool".to_string(),
        });
    }
    Ok(policy)
}

/// Who a child is and what it owes, beyond the policy it runs under.
///
/// ## Why this is separate from the policy
///
/// The policy decides what a child *may do*; this decides what it *is for*.
/// Getting the first wrong is a permissions bug and getting the second wrong is
/// a wasted worker, and keeping them apart is what stops a dispatch site
/// reaching for a field that widens authority when it meant to name a task.
///
/// Every field has a safe default, so a caller with no task identity to carry
/// gets a child scoped to the run rather than a child with an empty scope key
/// shared with every other task on the machine.
#[derive(Debug, Clone, Default)]
pub struct Dispatch {
    /// The agent from the deployment's registry. Memory is keyed by it.
    pub agent_id: String,
    /// The task the child joins. Empty means the parent's run.
    pub task_id: String,
    /// What counts as done, in the parent's words. Read by the parent's
    /// completion check — see [`crate::agent_runtime::completion`].
    pub deliverable: String,
    /// The graph position this child's inputs were authorised at, and therefore
    /// how it waits for a sibling's result.
    pub requirement: super::graph_io::Requirement,
    /// Whether the child may write. Read-only unless a caller says otherwise.
    pub mode: DelegationMode,
    /// Which attempt at the parent run is sending this child. Empty when the
    /// parent has no checkpoint yet.
    pub attempt_id: String,
    /// The child's id, when the caller must be able to name it -- and stop it
    /// -- before it finishes. Generated here when absent.
    pub child_id: Option<String>,
    /// What else the work is keyed on, beyond the run, role, objective and
    /// inputs. A plan step and its attempt number, for a job: the same
    /// objective under two steps is two pieces of work, and a retry of a
    /// failed attempt must not be answered with the failure it is retrying.
    /// Empty keeps the key every existing caller computes.
    pub key_scope: String,
    /// Stops the child when cancelled. Watched alongside the run's own stop, so
    /// either ends it.
    pub cancel: Option<crate::agent_runtime::cancellation::CancelToken>,
    /// A deadline tighter than the definition's own limit. Never looser: the
    /// shorter of the two applies.
    pub deadline: Option<std::time::Duration>,
}

/// What a delegation lets a child do to the world.
///
/// ## Why this is on the dispatch and not only on the definition
///
/// A definition says what a role *can* do; the mode says what *this* dispatch
/// permits. `agent.delegate_readonly` sends `ReadOnly`, and that is enforced
/// here in Rust whatever the definition declares, because the tool's name and
/// its approval class (automatic -- nobody is asked) are a promise to the model
/// and to the person reading the audit line. A definition an administrator
/// edited to grant a write tool does not get to break that promise through a
/// read-only call.
///
/// `Writer` exists as a contract and is not offered to any model in this
/// build. Plan P01 asks for writer semantics without widening the read-only
/// tool; the model-facing writer delegation belongs to the phases that build
/// the writer roles (P05, P11), and offering it before then would expose a
/// capability nothing has qualified.
///
/// ## What `Writer` permits, and what it never does
///
/// 1. **It never widens.** The narrowing against the parent's grant runs first
///    and is the same call in both modes: a writer child holds at most the
///    write tools its parent holds, and a definition asking for more has them
///    refused and recorded.
/// 2. **Every effect is still approved.** The child's calls go through the
///    same gateway as the parent's; a tool that asks a person asks a person.
/// 3. **Writers do not overlap.** A child left holding any tool that is not
///    read-only runs in the exclusive lane, whatever its declared isolation --
///    so a read-only role an administrator edited to hold a write tool cannot
///    write beside another writer through the reader lane.
/// 4. **Only one constructor.** [`Dispatch::writing`] is the only way to ask
///    for it, and `agent.delegate_readonly` never calls it: the tool's runner
///    builds `ReadOnly` unconditionally, so no argument a model writes can turn
///    a read-only delegation into a writer.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum DelegationMode {
    /// The child keeps only read-only tools, and a role declared as writing is
    /// refused outright rather than started without its tools.
    #[default]
    ReadOnly,
    /// The child keeps what the narrowing left it. Every effectful call it
    /// makes still goes through the gateway's approval, as the parent's do.
    Writer,
}

impl DelegationMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            DelegationMode::ReadOnly => "readOnly",
            DelegationMode::Writer => "writer",
        }
    }
}

impl Dispatch {
    /// The ordinary case: a child on the parent's own task, with nothing to
    /// wait for.
    pub fn for_task(agent_id: impl Into<String>, task_id: impl Into<String>) -> Self {
        Self {
            agent_id: agent_id.into(),
            task_id: task_id.into(),
            ..Self::default()
        }
    }

    /// The same, told to wait until a sibling's result has landed.
    pub fn after(mut self, graph_revision: i64) -> Self {
        self.requirement = super::graph_io::Requirement::AtLeast { graph_revision };
        self
    }

    pub fn delivering(mut self, deliverable: impl Into<String>) -> Self {
        self.deliverable = deliverable.into();
        self
    }

    /// Permits the child to write. See [`DelegationMode::Writer`].
    pub fn writing(mut self) -> Self {
        self.mode = DelegationMode::Writer;
        self
    }

    /// Names the attempt at the parent run that is sending this child.
    pub fn in_attempt(mut self, attempt_id: impl Into<String>) -> Self {
        self.attempt_id = attempt_id.into();
        self
    }

    /// Gives the child an id chosen by the caller.
    pub fn as_child(mut self, child_id: impl Into<String>) -> Self {
        self.child_id = Some(child_id.into());
        self
    }

    /// Keys the work on `scope` as well. See [`Dispatch::key_scope`].
    pub fn scoped_to(mut self, scope: impl Into<String>) -> Self {
        self.key_scope = scope.into();
        self
    }

    /// Lets the caller stop the child.
    pub fn stoppable(mut self, token: crate::agent_runtime::cancellation::CancelToken) -> Self {
        self.cancel = Some(token);
        self
    }

    /// Bounds the child more tightly than its definition does.
    pub fn within(mut self, deadline: std::time::Duration) -> Self {
        self.deadline = Some(deadline);
        self
    }
}

/// How long a stopped child is given to finish on its own before it is
/// abandoned. A worker checks its stop between steps, so this is the length of
/// one step, not a second deadline.
const STOP_GRACE: std::time::Duration = std::time::Duration::from_secs(2);

/// How often a running child's stop signals are looked at. The run's stop is a
/// table lookup rather than a notification, so it is polled.
const STOP_POLL: std::time::Duration = std::time::Duration::from_millis(50);

/// The Rust side of subagents.
pub struct SubagentManager {
    profiles: BTreeMap<String, AgentProfile>,
    /// Where a dispatch reads the definition it runs under.
    ///
    /// `None` is a deployment with no registry, and then the bundled profiles
    /// above are the definitions -- exactly the behaviour before this existed.
    /// With a registry, it is consulted on every dispatch and not cached here,
    /// which is what makes an administrator's edit reach the next child. See
    /// [`super::definitions`].
    definitions: Option<Arc<dyn super::definitions::DefinitionSource>>,
    workers: BTreeMap<String, Arc<dyn ChildWorker>>,
    events: Arc<TaskEventLog>,
    /// The read-only lane.
    readers: Arc<Semaphore>,
    /// The lane writers and approval-sensitive workers take one at a time.
    exclusive: Arc<Mutex<()>>,
    /// Idempotency key to slot.
    slots: Mutex<BTreeMap<String, Slot>>,
    /// The table a person's Stop lands in. Watched while a child runs, so a
    /// stopped task stops its children even when a worker does not look.
    cancellations: Option<Arc<crate::agent_runtime::cancellation::RunCancellations>>,
}

impl SubagentManager {
    pub fn new(profiles: Vec<AgentProfile>, events: Arc<TaskEventLog>) -> Self {
        Self {
            profiles: profiles
                .into_iter()
                .map(|profile| (profile.name.clone(), profile))
                .collect(),
            definitions: None,
            workers: BTreeMap::new(),
            events,
            readers: Arc::new(Semaphore::new(MAX_CONCURRENT_READERS)),
            exclusive: Arc::new(Mutex::new(())),
            slots: Mutex::new(BTreeMap::new()),
            cancellations: None,
        }
    }

    /// Watches `cancellations` for the parent run and for each child.
    pub fn with_cancellations(
        mut self,
        cancellations: Arc<crate::agent_runtime::cancellation::RunCancellations>,
    ) -> Self {
        self.cancellations = Some(cancellations);
        self
    }

    /// The stop table this manager watches, when it was given one.
    pub fn cancellations(
        &self,
    ) -> Option<Arc<crate::agent_runtime::cancellation::RunCancellations>> {
        self.cancellations.clone()
    }

    /// Reads each dispatch's definition from `source` rather than from the
    /// profiles this manager was built with.
    pub fn with_definitions(
        mut self,
        source: Arc<dyn super::definitions::DefinitionSource>,
    ) -> Self {
        self.definitions = Some(source);
        self
    }

    /// The definition a new dispatch for `key` runs under, read now.
    ///
    /// The registry decides when there is one. A key it does not know falls
    /// back to the bundled profile of that name, marked as such, so a role whose
    /// import failed is still performable and the trace says where its
    /// definition came from. Any other refusal -- an agent an administrator
    /// disabled, a capability nothing implements -- stands: falling back in
    /// those cases would be the setting having no effect.
    pub fn resolve(
        &self,
        key: &str,
    ) -> Result<super::definitions::ResolvedDefinition, SpawnRefusal> {
        use super::definitions::{ResolvedDefinition, Unresolved};

        let bundled = || {
            self.profiles
                .get(key)
                .map(ResolvedDefinition::from_bundled)
                .ok_or_else(|| SpawnRefusal::UnknownProfile {
                    name: key.to_string(),
                })
        };

        match &self.definitions {
            None => bundled(),
            Some(source) => match source.resolve(key) {
                Ok(resolved) => Ok(resolved),
                Err(Unresolved::NoDefinition { .. }) if self.profiles.contains_key(key) => {
                    bundled()
                }
                Err(unresolved) => Err(SpawnRefusal::Unresolved {
                    key: key.to_string(),
                    detail: unresolved.explain(),
                }),
            },
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
        let resolved = self.resolve(profile_name)?;
        let policy = self.narrow(&resolved, inherited, child_id)?;
        Ok((resolved.profile, policy))
    }

    /// The narrowing, over a definition already resolved.
    ///
    /// The one call that decides authority, and it is the same call whether the
    /// definition came from the registry or a bundled file:
    /// [`InheritedPolicy::narrow_for`] intersects with the parent's grant, so a
    /// definition -- however an administrator edited it -- can only ever be
    /// narrower than the run that dispatched it.
    fn narrow(
        &self,
        resolved: &super::definitions::ResolvedDefinition,
        inherited: &InheritedPolicy,
        child_id: &str,
    ) -> Result<EffectivePolicy, SpawnRefusal> {
        inherited
            .narrow_for(&resolved.profile, child_id)
            .map_err(|refusal| SpawnRefusal::Policy { refusal })
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
        dispatch: &Dispatch,
    ) -> Result<Spawned, SpawnRefusal> {
        let run_id = inherited_run_id(inherited);
        let key = if dispatch.key_scope.is_empty() {
            super::packet::derive_idempotency_key(&run_id, profile_name, objective, &inputs)
        } else {
            super::packet::derive_idempotency_key(
                &run_id,
                profile_name,
                &format!("{objective}\u{1e}{}", dispatch.key_scope),
                &inputs,
            )
        };

        // The slot is taken before anything else, so a second attempt at this
        // work waits here rather than starting a second child.
        let slot = {
            let mut slots = self.slots.lock().await;
            Arc::clone(
                slots
                    .entry(key.clone())
                    .or_insert_with(|| Arc::new(Mutex::new(None))),
            )
        };
        let mut held = slot.lock().await;
        if let Some(existing) = held.as_ref() {
            return Ok(Spawned::Existing(existing.clone()));
        }

        // The durable half of the same question, and the half that survives the
        // process.
        //
        // The slot above is an in-memory map: it stops two callers in *this*
        // process both starting a child, and it dies with the process. A run
        // that was interrupted mid-delegation and picked back up would find an
        // empty map and dispatch the work a second time — which for a retrieval
        // is wasteful and for anything with an effect is the duplicate this
        // ledger exists to prevent. So the intent goes on disk first, keyed the
        // same way, through the same table every side-effecting tool uses.
        let child_id = dispatch
            .child_id
            .clone()
            .filter(|id| !id.trim().is_empty())
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        // Resolved exactly once for this child, and what is resolved is copied
        // into the packet below. Nothing after this line reads the registry, so
        // an edit saved while this child runs cannot change what it runs under.
        //
        // And resolved *before* the durable intent below is written. A refusal
        // here -- an agent an administrator disabled, a key nothing answers to,
        // a narrowing the parent's grant will not permit -- means nothing was
        // attempted, and recording an intent for it would leave a pending row
        // that a later retry, after the agent was re-enabled, would trip over
        // as "already under way". The ledger records work, not refusals of it.
        let resolved = self.resolve(profile_name)?;
        let policy = self.narrow(&resolved, inherited, &child_id)?;
        let policy = enforce_mode(&resolved, policy, dispatch.mode)?;
        let profile = resolved.profile.clone();

        if let Some(recalled) = self.recall(&run_id, &key, objective, dispatch.mode) {
            *held = Some(recalled.clone());
            return Ok(Spawned::Existing(recalled));
        }

        // The worker is found by capability -- derived from the definition's
        // output schema -- and not by its name. A registry agent's name is its
        // `ag-` id, which no worker is registered under; a bundled profile's
        // name is also its capability, which is why the second lookup keeps
        // every deployment without a registry working exactly as before.
        let worker = self
            .workers
            .get(&resolved.capability)
            .or_else(|| self.workers.get(&profile.name))
            .cloned();
        let Some(worker) = worker else {
            let refusal = SpawnRefusal::NoWorker {
                profile: profile.name.clone(),
            };
            let result = refusal.as_result(&child_id, &profile.name, profile.required_schema);
            // Recorded even though nothing ran. A parent that asked for a
            // worker this build does not have should see that in the trace
            // rather than only in a returned error.
            self.record_stop(inherited, &child_id, &result, &model);
            *held = Some(result.clone());
            return Ok(Spawned::Fresh(result));
        };

        let packet = ChildTaskPacket::new(
            &child_id,
            &run_id,
            &key,
            objective,
            inputs,
            &policy,
            Utc::now(),
        )
        .assigned_to(
            // An agent id the caller did not supply falls back to the profile
            // name, which is stable for the life of the deployment and is what
            // the memory would otherwise be keyed by nothing at all.
            if dispatch.agent_id.trim().is_empty() {
                profile.name.clone()
            } else {
                dispatch.agent_id.clone()
            },
            dispatch.task_id.clone(),
            dispatch.deliverable.clone(),
            dispatch.requirement,
        )
        .pinned_to(&resolved)
        .attempted_in(dispatch.attempt_id.clone())
        .governed_by(resolved.model_policy(&model))
        .routed_to(Some(model.model_id.clone()).filter(|id| !id.trim().is_empty()));

        self.record_start(inherited, &packet, &policy, &model, dispatch.mode);

        // The lane. Held for exactly as long as the work, and released before
        // the stop is recorded so a slow event write does not hold the lane.
        //
        // Concurrent only if the isolation says so *and* nothing the child was
        // left holding writes. The second half is what keeps writers from
        // overlapping when a read-only-declared role was edited to hold a write
        // tool and dispatched as a writer -- see `DelegationMode`.
        let _reader;
        let _writer;
        if policy.is_concurrent() && policy.tools.iter().all(|tool| tool.is_read_only()) {
            _reader = self.readers.clone().acquire_owned().await.ok();
        } else {
            _writer = Some(self.exclusive.clone().lock_owned().await);
        }

        let defined = std::time::Duration::from_secs(policy.limits.max_duration_seconds.max(1));
        let budget = dispatch.deadline.map_or(defined, |deadline| deadline.min(defined));
        let mut work = Box::pin(tokio::time::timeout(budget, worker.run(&packet, &policy)));

        // Raced against every way this child can be stopped: its own token, its
        // id in the stop table, and the run it belongs to. A worker checks its
        // stop between steps; this is what makes a stop hold when one step is
        // long, or when the worker does not look at all.
        let stopped = tokio::select! {
            outcome = &mut work => Err(outcome),
            reason = self.stop_signal(dispatch, &run_id, &child_id) => Ok(reason),
        };
        let outcome = match stopped {
            Err(outcome) => outcome,
            Ok(reason) => {
                if let Some(table) = &self.cancellations {
                    table.cancel(&child_id);
                }
                // One step's grace. A worker that finishes the work it was on
                // is reported as it finished; one that does not is cancelled.
                match tokio::time::timeout(STOP_GRACE, &mut work).await {
                    Ok(Ok(Ok(finished))) if finished.status.is_complete() => Ok(Ok(finished)),
                    _ => Ok(Ok(ChildResult::ended(
                        &child_id,
                        &profile.name,
                        ChildStatus::Cancelled,
                        profile.required_schema,
                        Vec::new(),
                        reason,
                        0,
                    ))),
                }
            }
        };

        let result = match outcome {
            Ok(Ok(mut produced)) => {
                // The worker's own status is not taken on trust for the fields
                // that decide whether the parent may rely on it. A worker that
                // returned a result answering a different packet is a worker
                // that answered a different question.
                if !produced.answers(&packet) {
                    produced = ChildResult::ended(
                        &child_id,
                        &profile.name,
                        ChildStatus::Failed,
                        profile.required_schema,
                        Vec::new(),
                        "the worker returned a result for a different task or shape".to_string(),
                        produced.turns_used,
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

        self.record_stop(inherited, &child_id, &result, &model);
        // Settled on disk as well as in the slot, so the next process to pick
        // this run up finds the answer rather than the intent. An outcome that
        // is never settled stays `pending` and is promoted to `unknown` at the
        // next start — which is the correct answer when nobody can say what
        // happened, and is how a worker that died mid-flight is reported.
        self.settle(&run_id, &key, &result);
        *held = Some(result.clone());
        Ok(Spawned::Fresh(result))
    }

    /// Resolves when this child has been told to stop, with the reason.
    async fn stop_signal(&self, dispatch: &Dispatch, run_id: &str, child_id: &str) -> String {
        let mut polls: u32 = 0;
        loop {
            // The run's durable ending, looked at about once a second: a job
            // the coordinator left running when its run finished has nobody
            // left to report to, and is stopped rather than orphaned.
            if polls % 20 == 0 && self.events.ending(run_id).is_some() {
                return "the run it belonged to ended before it finished".to_string();
            }
            polls = polls.wrapping_add(1);
            if dispatch.cancel.as_ref().is_some_and(|token| token.is_cancelled()) {
                return "it was stopped by the run that dispatched it before it finished"
                    .to_string();
            }
            if let Some(table) = &self.cancellations {
                if table.is_cancelled(run_id) {
                    return "the task it belonged to was stopped before it finished".to_string();
                }
                if table.is_cancelled(child_id) {
                    return "it was stopped before it finished".to_string();
                }
            }
            match &dispatch.cancel {
                Some(token) => {
                    tokio::select! {
                        _ = token.cancelled() => {}
                        _ = tokio::time::sleep(STOP_POLL) => {}
                    }
                }
                None => tokio::time::sleep(STOP_POLL).await,
            }
        }
    }

    /// What the durable ledger already knows about this piece of work.
    ///
    /// `None` means "go ahead", and the intent has been recorded. `Some` is an
    /// answer from a previous attempt — a settled result to hand back, or a
    /// refusal for the two cases where carrying on would be wrong.
    fn recall(
        &self,
        run_id: &str,
        key: &str,
        objective: &str,
        mode: DelegationMode,
    ) -> Option<ChildResult> {
        use crate::agent_runtime::events::EffectLookup;

        let fingerprint = crate::agent_runtime::events::args_fingerprint(&json!({
            "objective": objective,
        }));
        // Recorded under the tool that can do what this child may do, so the
        // effect ledger says a writer was started when one was.
        let tool = match mode {
            DelegationMode::ReadOnly => ToolName::AgentDelegateReadonly,
            DelegationMode::Writer => ToolName::AgentDelegate,
        };
        match self.events.begin_effect(run_id, key, tool.as_str(), &fingerprint, key) {
            // Never seen. The intent is now on disk and the child may start.
            EffectLookup::Fresh => None,
            // Done before. The recorded ending, rebuilt as a typed result so
            // the parent handles it exactly as it would a fresh one.
            EffectLookup::Settled(recorded) => Some(if recorded.succeeded() {
                // `ended` is for a child that did not finish, and asserts as
                // much. A settled success is a *completed* child whose answer
                // is being reused rather than recomputed.
                let mut replay = ChildResult::completed(
                    key,
                    "",
                    super::profile::SchemaKind::Retrieval,
                    Vec::new(),
                    1.0,
                    Vec::new(),
                    0,
                );
                replay.detail = Some(format!(
                    "This work was already done under the same key, and its recorded outcome is \
                     reused rather than a second child being started: {}",
                    recorded.result
                ));
                replay
            } else {
                ChildResult::ended(
                    key,
                    "",
                    ChildStatus::Failed,
                    super::profile::SchemaKind::Retrieval,
                    Vec::new(),
                    format!(
                        "This work was already attempted under the same key and did not \
                         succeed, so it was not started again: {}",
                        recorded.result
                    ),
                    0,
                )
            }),
            // Another attempt is running right now. Refused rather than queued:
            // the in-memory slot above is what serialises two callers in one
            // process, so reaching here means two *processes*, and the second
            // should not add a third.
            EffectLookup::InFlight(recorded) => Some(ChildResult::ended(
                key,
                "",
                ChildStatus::Refused,
                super::profile::SchemaKind::Retrieval,
                Vec::new(),
                format!(
                    "This exact piece of work is already being done by another attempt at this \
                     run, begun at {}. Nothing was started.",
                    recorded.at
                ),
                0,
            )),
            // Interrupted, and nobody can say whether it finished. The one case
            // that needs a person; a child restarted here could repeat whatever
            // the first one had already done.
            EffectLookup::Unknown(recorded) => Some(ChildResult::ended(
                key,
                "",
                ChildStatus::Failed,
                super::profile::SchemaKind::Retrieval,
                Vec::new(),
                recorded.unknown_refusal(),
                0,
            )),
            EffectLookup::Conflict(conflict) => Some(ChildResult::ended(
                key,
                "",
                ChildStatus::Refused,
                super::profile::SchemaKind::Retrieval,
                Vec::new(),
                conflict.to_string(),
                0,
            )),
        }
    }

    /// Records how this piece of work ended, durably.
    fn settle(&self, run_id: &str, key: &str, result: &ChildResult) {
        let outcome = if result.status.is_complete() {
            Ok(format!(
                "{} finding(s) from {}",
                result.findings.len(),
                result.profile
            ))
        } else {
            Err(format!(
                "{} {}",
                result.profile,
                result.status.describe()
            ))
        };
        self.events.settle_effect(run_id, key, &outcome);
    }

    /// Records that a child began, with everything requirement 7 asks for.
    fn record_start(
        &self,
        inherited: &InheritedPolicy,
        packet: &ChildTaskPacket,
        policy: &EffectivePolicy,
        model: &Decision,
        mode: DelegationMode,
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
            // The parent/child relationship, durably. This row *is* the record
            // that this task had this worker on it: the run id is the envelope's,
            // the task and the agent are here, and a reader joining them back
            // together needs nothing that lives in memory.
            "agentId": packet.agent_id,
            // Which definition, exactly. The version and the instructions' hash
            // rather than the text: enough to tell two children apart when an
            // administrator edited the agent between them, and nothing a reader
            // of the trace should not see.
            "definitionVersion": packet.definition_version,
            "definitionOrigin": packet.definition_origin,
            "capability": packet.capability,
            "instructionsSha256": packet.instructions_sha256,
            "sharedWithTask": packet.shared_with_task,
            // The skills the definition was bound to, by hash; and which
            // attempt at which job this child is.
            "skills": packet.skills,
            "attemptId": packet.attempt_id,
            "jobId": packet.job_id(),
            "delegationMode": mode.as_str(),
            "modelPolicy": packet.model_policy,
            "taskId": packet.task_id,
            "deliverable": packet.deliverable,
            "requirement": packet.requirement,
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
            // How many of those a reader could actually check. The number the
            // parent's completion criterion turns on: six findings citing
            // nothing is six sentences, and folding them into an answer would
            // be citing the worker rather than a source.
            "evidenced": result
                .findings
                .iter()
                .filter(|finding| !finding.evidence.is_empty())
                .count(),
            "confidence": result.confidence,
            "uncertainty": result.uncertainty.len(),
            "turnsUsed": result.turns_used,
            // The ids a sibling or a later step can go and read. Recorded
            // rather than left in the result alone, because the completion
            // check runs after a restart and reads this log rather than
            // anything held in memory.
            "published": result.published,
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
