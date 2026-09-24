//! The one lease/admission service for heavy model work on this machine.
//!
//! ## What was here, and why it was not enough
//!
//! This module used to decide two things for subagent workers only: whether a
//! worker's model could be *resident* beside what was running, and a per-model
//! semaphore of [`REQUESTS_PER_MODEL_BEFORE_P03`] request slots. Neither said
//! anything about the parent run, the OCR pages, a vision call, or a background
//! job — each of which reached `serving::admission::admit` on its own. So on the
//! 8 GB card this product is built for:
//!
//! - Two generations could run at once on the same warm model (the per-model
//!   semaphore permitted two), sharing a KV cache the window was budgeted
//!   without.
//! - A child's admission could evict the parent's server while the parent was
//!   awaiting that child. The parent's next round then went to a base URL that
//!   no longer answered.
//! - A child waiting for the card behind a parent that was waiting for the
//!   child was a deadlock with nothing to break it.
//!
//! ## The policy now
//!
//! **One heavy call at a time, across the process.** [`ONE_HEAVY_CALL`] is the
//! capacity of a single book that every heavy consumer asks: the parent's model
//! rounds, a child's model rounds, OCR pages, vision calls and background jobs.
//! A holder has the card; nobody else generates or loads until it gives it
//! back. Because nothing else is in flight while a lease is held, admission may
//! evict any *idle* server to make room — it can never pull a model out from
//! under a generation, which is the property the old per-component limits could
//! not state.
//!
//! **The parent never holds the card while it awaits a child.** A child's
//! request names its parent. If any ancestor holds a lease, the ancestor's lease
//! is suspended — recorded durably through the observer first, then released —
//! and the child is admitted. The parent re-acquires on its next round and its
//! endpoint is re-bound ([`ModelScheduler::ensure_served`]), because the server
//! it was talking to may have been stopped to make room.
//!
//! **Bounded, timed, cancellable, fair.** At most [`MAX_WAITING`] requests
//! queue; beyond that a request is refused immediately rather than joining a
//! queue nobody will reach. Every wait has a deadline and a cancellation token.
//! Interactive work is served first-come-first-served; background work waits
//! behind it but ages into the same tier after [`BACKGROUND_AGES_AFTER`], so a
//! steady stream of chat turns cannot starve it forever. A holder past its own
//! hold limit is expired when somebody is waiting, so a crashed runtime cannot
//! strand the card.
//!
//! ## What residency still decides
//!
//! Whether a model is warm, fits alongside, or must evict — [`Residency`] and
//! [`plan_residency`] — is still measured and reported on the child's start
//! record. It no longer decides *concurrency*: that is the book's job, and the
//! answer is always one.

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use tokio::sync::oneshot;

use crate::agent_runtime::cancellation::CancelToken;
use crate::registry::{ModelEntry, ModelRegistry};
use crate::serving::admission::measure_budget;
use crate::serving::fallback::LoadFailure;
use crate::serving::{Endpoint, ModelServers, ServingError};

/// How many heavy model calls may be in flight on this machine at once.
///
/// One. The target is an RTX 5060 with 8 GB, where Spark Q8 at a 16k window
/// leaves about 2.6 GB free (P00-OBS-3) — not a second model, and not a second
/// generation sharing the first one's KV cache. Co-residency is a later,
/// *measured* optimisation (plan §4), not a starting assumption.
pub const ONE_HEAVY_CALL: usize = 1;

/// What the per-model semaphore permitted before this module became the global
/// book. Kept as a named constant so the regression test that proves it no
/// longer holds can say what it is disproving.
pub const REQUESTS_PER_MODEL_BEFORE_P03: usize = 2;

/// How many requests may wait for the card before new ones are refused.
///
/// Sixteen: more than the nine roles plus the parent and an OCR batch, and few
/// enough that a runaway fan-out is refused while it is still small.
pub const MAX_WAITING: usize = 16;

/// How long background work waits before it is treated as interactive.
///
/// The starvation guard. Without it a busy operator's stream of turns would
/// keep a background index job queued for the whole session.
pub const BACKGROUND_AGES_AFTER: Duration = Duration::from_secs(30);

/// How long one holder may keep the card before a waiter may expire it.
///
/// Thirty minutes: longer than any single generation or OCR batch this product
/// runs, and short enough that a runtime that died between taking a lease and
/// giving it back does not hold the machine for the rest of the day.
pub const DEFAULT_HOLD_LIMIT: Duration = Duration::from_secs(30 * 60);

/// How often a waiter looks for an overdue holder to expire.
const WAIT_TICK: Duration = Duration::from_millis(250);

/// How long a request waits for the card unless it says otherwise.
pub const DEFAULT_WAIT: Duration = Duration::from_secs(10 * 60);

/// What a served model costs beyond its weights, for the co-residency report.
///
/// 1 GiB. Not a measurement, and not presented as one: it is a floor chosen so
/// that a model judged to fit alongside another has room for a modest KV cache
/// and the runtime's buffers. The precise figure for a model about to be
/// started is [`crate::ai_engine::vram_planner`]'s, which reads the GGUF
/// header; this is the conservative guard in front of it.
pub const CO_RESIDENT_HEADROOM: u64 = 1024 * 1024 * 1024;

// ─────────────────────────────────────────────────────────────────────────────
// Who is asking
// ─────────────────────────────────────────────────────────────────────────────

/// What kind of heavy work a lease is for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum LeaseClass {
    /// A top-level run's model rounds — Spark, as the orchestrator.
    Parent,
    /// A delegated worker's model rounds.
    Child,
    /// A page through the OCR model.
    Ocr,
    /// An image through a vision model.
    Vision,
    /// Anything nobody is waiting on right now: indexing, learning jobs.
    Background,
}

impl LeaseClass {
    pub fn as_str(self) -> &'static str {
        match self {
            LeaseClass::Parent => "parent",
            LeaseClass::Child => "child",
            LeaseClass::Ocr => "ocr",
            LeaseClass::Vision => "vision",
            LeaseClass::Background => "background",
        }
    }

    /// Whether somebody is waiting on this at the screen.
    pub fn interactive(self) -> bool {
        !matches!(self, LeaseClass::Background)
    }
}

/// One request for the card.
#[derive(Debug, Clone)]
pub struct LeaseRequest {
    /// Who holds it: a run id, a child id, `ocr:<run>`. Re-entrant per owner.
    pub owner: String,
    pub class: LeaseClass,
    /// The registry id of the model the holder will use.
    pub model_id: String,
    /// The owner that dispatched this one, when there is one. What makes the
    /// parent-before-child suspension possible.
    pub parent: Option<String>,
    /// Models this owner's definition allows. Empty means any registered model.
    ///
    /// Enforced here because the job packet records `within_eligible` and says
    /// holding routing to the binding is the scheduler's (see
    /// `subagents::packet::ModelPolicy`).
    pub eligible: Vec<String>,
    /// How long to wait for the card.
    pub wait_for: Duration,
    /// How long the card may be held before a waiter may expire it.
    pub hold_limit: Duration,
    pub cancel: CancelToken,
}

impl LeaseRequest {
    pub fn new(owner: impl Into<String>, class: LeaseClass, model_id: impl Into<String>) -> Self {
        Self {
            owner: owner.into(),
            class,
            model_id: model_id.into(),
            parent: None,
            eligible: Vec::new(),
            wait_for: DEFAULT_WAIT,
            hold_limit: DEFAULT_HOLD_LIMIT,
            cancel: CancelToken::never(),
        }
    }

    pub fn child_of(mut self, parent: impl Into<String>) -> Self {
        let parent = parent.into();
        self.parent = (!parent.is_empty()).then_some(parent);
        self
    }

    pub fn eligible(mut self, eligible: Vec<String>) -> Self {
        self.eligible = eligible;
        self
    }

    pub fn waiting(mut self, wait_for: Duration) -> Self {
        self.wait_for = wait_for;
        self
    }

    pub fn holding_at_most(mut self, hold_limit: Duration) -> Self {
        self.hold_limit = hold_limit;
        self
    }

    pub fn cancelled_by(mut self, cancel: CancelToken) -> Self {
        self.cancel = cancel;
        self
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// What can go wrong
// ─────────────────────────────────────────────────────────────────────────────

/// Why a request was not given the card, or its model was not served.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SchedulingRefusal {
    /// Nothing in the registry has that id.
    NotRegistered { model_id: String },
    /// The machine cannot hold it.
    WontFit { model_id: String, detail: String },
    /// The deadline passed while waiting for the card.
    WaitedTooLong { model_id: String, seconds: u64 },
    /// The queue is full. Refused immediately rather than queued behind work
    /// that will not finish inside anybody's deadline.
    QueueFull { model_id: String, waiting: usize },
    /// Somebody stopped the run while it waited.
    Cancelled { model_id: String },
    /// The model is outside what this owner's definition allows.
    OutsideEligible {
        model_id: String,
        eligible: Vec<String>,
    },
    /// A model was to be served for an owner that does not hold the card.
    NotHeld { owner: String },
    /// The model would not come up. Carries the classified failure; the
    /// fallback decision is the caller's, because only the caller knows where
    /// the run stood.
    LoadFailed { failure: LoadFailure },
}

impl SchedulingRefusal {
    pub fn kind(&self) -> &'static str {
        match self {
            Self::NotRegistered { .. } => "notRegistered",
            Self::WontFit { .. } => "wontFit",
            Self::WaitedTooLong { .. } => "waitedTooLong",
            Self::QueueFull { .. } => "queueFull",
            Self::Cancelled { .. } => "cancelled",
            Self::OutsideEligible { .. } => "outsideEligible",
            Self::NotHeld { .. } => "notHeld",
            Self::LoadFailed { .. } => "loadFailed",
        }
    }

    pub fn explain(&self) -> String {
        match self {
            Self::NotRegistered { model_id } => format!(
                "{model_id} is not in the model registry on this machine, so no work can be \
                 given it."
            ),
            Self::WontFit { model_id, detail } => {
                format!("{model_id} cannot be used here: {detail}")
            }
            Self::WaitedTooLong { model_id, seconds } => format!(
                "this work waited {seconds} second(s) for the GPU to run {model_id} and ran out \
                 of time. Nothing was done. The machine is running more model work than it has \
                 memory for; try again when it is quieter."
            ),
            Self::QueueFull { model_id, waiting } => format!(
                "{waiting} piece(s) of model work are already waiting for the GPU, so this \
                 request for {model_id} was refused rather than queued. Nothing was done."
            ),
            Self::Cancelled { model_id } => format!(
                "stopped while waiting for the GPU to run {model_id}. Nothing was done."
            ),
            Self::OutsideEligible { model_id, eligible } => format!(
                "{model_id} is not one of the models this agent's definition allows ({}), so it \
                 was not given the GPU. Nothing was done.",
                eligible.join(", ")
            ),
            Self::NotHeld { owner } => format!(
                "{owner} asked for its model to be served without holding the GPU lease. \
                 Loading a model without the lease could evict one mid-generation, so it was \
                 refused."
            ),
            Self::LoadFailed { failure } => format!(
                "{} could not be loaded ({}): {}",
                failure.model_id,
                failure.kind.as_str(),
                failure.detail
            ),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// What the book records
// ─────────────────────────────────────────────────────────────────────────────

/// A holder gave up the card so something it is waiting on could have it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Suspension {
    pub owner: String,
    pub model_id: String,
    pub class: LeaseClass,
    /// The descendant it gave way to, when that is why.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub for_owner: Option<String>,
    /// Whether it was holding the card at that moment. A parent that had
    /// already released at its tool boundary is still recorded as suspended,
    /// so its resumption is recognised as one.
    pub released_capacity: bool,
    pub because: String,
    /// RFC 3339, UTC.
    pub at: String,
}

/// What the book tells whoever records it. See [`ModelScheduler::observe`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", tag = "kind")]
pub enum LeaseEvent {
    Suspended(Suspension),
    Resumed {
        owner: String,
        model_id: String,
        class: LeaseClass,
        suspended_for_ms: u64,
    },
    /// A holder past its hold limit was expired so a waiter could proceed.
    Expired {
        owner: String,
        model_id: String,
        held_ms: u64,
    },
}

impl LeaseEvent {
    /// The owner whose record this belongs on.
    pub fn owner(&self) -> &str {
        match self {
            LeaseEvent::Suspended(suspension) => &suspension.owner,
            LeaseEvent::Resumed { owner, .. } | LeaseEvent::Expired { owner, .. } => owner,
        }
    }
}

/// One holder, as a snapshot reports it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Holding {
    pub owner: String,
    pub class: LeaseClass,
    pub model_id: String,
    pub grant: u64,
    /// Held for a model round driven from the runtime.
    pub round: bool,
    /// Held by Rust-side guards.
    pub guards: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent: Option<String>,
    pub held_ms: u64,
}

/// One waiter, as a snapshot reports it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Waiting {
    pub owner: String,
    pub class: LeaseClass,
    pub model_id: String,
    pub waited_ms: u64,
}

/// The book, read once. What a status panel and a leak check both want.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LeaseSnapshot {
    pub capacity: usize,
    pub holders: Vec<Holding>,
    pub waiting: Vec<Waiting>,
    pub suspended: Vec<Suspension>,
}

impl LeaseSnapshot {
    /// Whether nothing is held and nothing waits. The leak check.
    pub fn is_idle(&self) -> bool {
        self.holders.is_empty() && self.waiting.is_empty()
    }
}

/// What admitting a request settled.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Admitted {
    pub grant: u64,
    pub class: LeaseClass,
    pub waited: Duration,
    /// True when this owner already held the card and the request re-entered.
    pub reentered: bool,
    /// Set when this owner had been suspended and is now back.
    pub resumed_after: Option<Duration>,
    /// Ancestors suspended to let this request in. Already recorded.
    pub suspended: Vec<Suspension>,
}

// ─────────────────────────────────────────────────────────────────────────────
// The book itself: synchronous, and the only place a decision is taken
// ─────────────────────────────────────────────────────────────────────────────

struct Holder {
    grant: u64,
    class: LeaseClass,
    model_id: String,
    round: bool,
    guards: u32,
    parent: Option<String>,
    since: Instant,
    hold_limit: Duration,
}

struct Waiter {
    ticket: u64,
    owner: String,
    class: LeaseClass,
    model_id: String,
    parent: Option<String>,
    round: bool,
    hold_limit: Duration,
    enqueued: Instant,
    wake: oneshot::Sender<u64>,
}

struct Suspended {
    record: Suspension,
    since: Instant,
}

struct Book {
    capacity: usize,
    /// See [`BACKGROUND_AGES_AFTER`]. A field so a test can age work in
    /// milliseconds rather than waiting half a minute.
    ages_after: Duration,
    holders: BTreeMap<String, Holder>,
    waiting: Vec<Waiter>,
    suspended: BTreeMap<String, Suspended>,
    /// Who dispatched whom, for the ancestor walk. Kept past a holder's release
    /// because a grandparent may still be holding when a grandchild asks.
    lineage: HashMap<String, String>,
    next_grant: u64,
    next_ticket: u64,
}

enum Admit {
    Granted(Admitted),
    Queued {
        ticket: u64,
        wake: oneshot::Receiver<u64>,
    },
    Refused(SchedulingRefusal),
}

impl Book {
    fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            ages_after: BACKGROUND_AGES_AFTER,
            holders: BTreeMap::new(),
            waiting: Vec::new(),
            suspended: BTreeMap::new(),
            lineage: HashMap::new(),
            next_grant: 1,
            next_ticket: 1,
        }
    }

    /// Ancestors of `owner` that currently hold the card, nearest first.
    fn holding_ancestors(&self, owner: &str, parent: Option<&str>) -> Vec<String> {
        let mut found = Vec::new();
        let mut cursor = parent.map(str::to_string);
        let mut guard = 0;
        while let Some(ancestor) = cursor {
            // A cycle in the lineage would be a defect elsewhere; it must not
            // become an infinite loop here.
            guard += 1;
            if guard > 32 || ancestor == owner {
                break;
            }
            if self.holders.contains_key(&ancestor) {
                found.push(ancestor.clone());
            }
            cursor = self.lineage.get(&ancestor).cloned();
        }
        found
    }

    fn suspend(
        &mut self,
        owner: &str,
        for_owner: Option<&str>,
        because: &str,
        fallback_class: LeaseClass,
        fallback_model: &str,
    ) -> Suspension {
        let released = self.holders.remove(owner);
        let record = Suspension {
            owner: owner.to_string(),
            model_id: released
                .as_ref()
                .map(|held| held.model_id.clone())
                .unwrap_or_else(|| fallback_model.to_string()),
            class: released
                .as_ref()
                .map(|held| held.class)
                .unwrap_or(fallback_class),
            for_owner: for_owner.map(str::to_string),
            released_capacity: released.is_some(),
            because: because.to_string(),
            at: chrono::Utc::now().to_rfc3339(),
        };
        self.suspended.insert(
            owner.to_string(),
            Suspended {
                record: record.clone(),
                since: Instant::now(),
            },
        );
        record
    }

    fn take_resumption(&mut self, owner: &str) -> Option<Duration> {
        self.suspended
            .remove(owner)
            .map(|suspended| suspended.since.elapsed())
    }

    fn admit(&mut self, request: &LeaseRequest, round: bool) -> Admit {
        if let Some(parent) = &request.parent {
            self.lineage.insert(request.owner.clone(), parent.clone());
        }

        // Re-entry: the same owner already holds the card. A worker's guard
        // and its own model rounds are one holder, not two.
        if let Some(held) = self.holders.get_mut(&request.owner) {
            if round {
                held.round = true;
            } else {
                held.guards = held.guards.saturating_add(1);
            }
            return Admit::Granted(Admitted {
                grant: held.grant,
                class: held.class,
                waited: Duration::ZERO,
                reentered: true,
                resumed_after: None,
                suspended: Vec::new(),
            });
        }

        // Room now, and nobody queued ahead: granted on the spot. Somebody
        // queued ahead is served first even when there is room, which only
        // happens in the instant between a release and its dispatch.
        if self.holders.len() < self.capacity && self.waiting.is_empty() {
            // Taken only on a grant. A request that queues, times out or is
            // refused has not resumed anything.
            let resumed_after = self.take_resumption(&request.owner);
            let grant = self.grant_to(
                &request.owner,
                request.class,
                &request.model_id,
                request.parent.clone(),
                round,
                request.hold_limit,
            );
            return Admit::Granted(Admitted {
                grant,
                class: request.class,
                waited: Duration::ZERO,
                reentered: false,
                resumed_after,
                suspended: Vec::new(),
            });
        }

        if self.waiting.len() >= MAX_WAITING {
            return Admit::Refused(SchedulingRefusal::QueueFull {
                model_id: request.model_id.clone(),
                waiting: self.waiting.len(),
            });
        }

        let (wake, receiver) = oneshot::channel();
        let ticket = self.next_ticket;
        self.next_ticket += 1;
        self.waiting.push(Waiter {
            ticket,
            owner: request.owner.clone(),
            class: request.class,
            model_id: request.model_id.clone(),
            parent: request.parent.clone(),
            round,
            hold_limit: request.hold_limit,
            enqueued: Instant::now(),
            wake,
        });
        Admit::Queued {
            ticket,
            wake: receiver,
        }
    }

    fn grant_to(
        &mut self,
        owner: &str,
        class: LeaseClass,
        model_id: &str,
        parent: Option<String>,
        round: bool,
        hold_limit: Duration,
    ) -> u64 {
        let grant = self.next_grant;
        self.next_grant += 1;
        self.holders.insert(
            owner.to_string(),
            Holder {
                grant,
                class,
                model_id: model_id.to_string(),
                round,
                guards: u32::from(!round),
                parent,
                since: Instant::now(),
                hold_limit,
            },
        );
        grant
    }

    /// Hands the card to whoever should have it next. Returns holders expired
    /// to make that possible.
    fn dispatch(&mut self) -> Vec<LeaseEvent> {
        let mut expired = Vec::new();
        // A holder past its limit is expired only when somebody is waiting:
        // expiring an idle machine's only holder would help nobody.
        if !self.waiting.is_empty() && self.holders.len() >= self.capacity {
            let overdue: Vec<String> = self
                .holders
                .iter()
                .filter(|(_, held)| held.since.elapsed() > held.hold_limit)
                .map(|(owner, _)| owner.clone())
                .collect();
            for owner in overdue {
                if let Some(held) = self.holders.remove(&owner) {
                    log::warn!(
                        "[scheduling] {owner} held the GPU for {:?}, past its limit of {:?}, \
                         while work was waiting; its lease was expired",
                        held.since.elapsed(),
                        held.hold_limit
                    );
                    expired.push(LeaseEvent::Expired {
                        owner,
                        model_id: held.model_id,
                        held_ms: held.since.elapsed().as_millis() as u64,
                    });
                }
            }
        }

        while self.holders.len() < self.capacity && !self.waiting.is_empty() {
            // Interactive first, and background once it has aged; then oldest.
            let next = self
                .waiting
                .iter()
                .enumerate()
                .min_by_key(|(_, waiter)| {
                    let tier = if waiter.class.interactive()
                        || waiter.enqueued.elapsed() >= self.ages_after
                    {
                        0u8
                    } else {
                        1u8
                    };
                    (tier, waiter.ticket)
                })
                .map(|(index, _)| index);
            let Some(index) = next else { break };
            let waiter = self.waiting.remove(index);
            let grant = self.grant_to(
                &waiter.owner,
                waiter.class,
                &waiter.model_id,
                waiter.parent.clone(),
                waiter.round,
                waiter.hold_limit,
            );
            // A waiter that gave up between being chosen and being told has
            // dropped its receiver. The grant is taken back at once, so a
            // departed waiter can never hold the card.
            if waiter.wake.send(grant).is_err() {
                self.holders.remove(&waiter.owner);
            }
        }
        expired
    }

    /// Removes a waiter that stopped waiting. `None` when it had already been
    /// granted in the meantime — the caller then owns that grant.
    fn abandon(&mut self, ticket: u64) -> bool {
        let before = self.waiting.len();
        self.waiting.retain(|waiter| waiter.ticket != ticket);
        self.waiting.len() != before
    }

    fn release_owner(&mut self, owner: &str) -> bool {
        self.holders.remove(owner).is_some()
    }

    fn snapshot(&self) -> LeaseSnapshot {
        LeaseSnapshot {
            capacity: self.capacity,
            holders: self
                .holders
                .iter()
                .map(|(owner, held)| Holding {
                    owner: owner.clone(),
                    class: held.class,
                    model_id: held.model_id.clone(),
                    grant: held.grant,
                    round: held.round,
                    guards: held.guards,
                    parent: held.parent.clone(),
                    held_ms: held.since.elapsed().as_millis() as u64,
                })
                .collect(),
            waiting: self
                .waiting
                .iter()
                .map(|waiter| Waiting {
                    owner: waiter.owner.clone(),
                    class: waiter.class,
                    model_id: waiter.model_id.clone(),
                    waited_ms: waiter.enqueued.elapsed().as_millis() as u64,
                })
                .collect(),
            suspended: self
                .suspended
                .values()
                .map(|suspended| suspended.record.clone())
                .collect(),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Serving, abstracted so every branch is testable without a GPU
// ─────────────────────────────────────────────────────────────────────────────

/// Where the service gets model servers from.
///
/// The production implementation is [`ServersBackend`]: `ModelServers` for the
/// processes, `serving::admission` for the plan. A test supplies its own, which
/// is how a load failure, an OOM and a restart onto a smaller window are all
/// exercised on a machine with no GPU and no `llama-server`.
#[async_trait::async_trait]
pub trait ServingBackend: Send + Sync {
    /// The endpoint of a server already up and ready, if there is one.
    fn warm_endpoint(&self, model_id: &str) -> Option<Endpoint>;
    /// What would have to happen for this model to run now. Report only.
    fn residency(&self, entry: &ModelEntry) -> Residency;
    /// Admits and starts the model. Called only while the caller holds the
    /// card, so anything admission evicts is idle.
    async fn start(&self, entry: &ModelEntry) -> Result<Endpoint, ServingError>;
}

/// `ModelServers` plus `serving::admission`: the production backend.
pub struct ServersBackend {
    pub servers: Arc<ModelServers>,
    pub models_dir: std::path::PathBuf,
}

#[async_trait::async_trait]
impl ServingBackend for ServersBackend {
    fn warm_endpoint(&self, model_id: &str) -> Option<Endpoint> {
        self.servers.warm_endpoint(model_id)
    }

    fn residency(&self, entry: &ModelEntry) -> Residency {
        if self.servers.is_warm(&entry.id) {
            return Residency::AlreadyWarm;
        }
        let installed = crate::system_analyzer::gpu_collector::installed_gpus()
            .iter()
            .map(|gpu| gpu.dedicated_video_memory_bytes)
            .max()
            .unwrap_or(0);
        let budget = measure_budget(installed);
        let ram = crate::system_analyzer::memory_collector::detect_memory();
        plan_residency(
            entry.weights_bytes,
            budget.bytes(),
            budget.measured(),
            installed,
            ram.available_bytes,
            !self.servers.running_model_ids().is_empty(),
        )
    }

    async fn start(&self, entry: &ModelEntry) -> Result<Endpoint, ServingError> {
        let admitted =
            crate::serving::admission::admit(&self.servers, entry, &self.models_dir).await?;
        if !admitted.released.is_empty() {
            log::info!(
                "[scheduling] released {} to make room for {} (idle: the lease holder is the \
                 only heavy call in flight)",
                admitted.released.join(", "),
                entry.id
            );
        }
        self.servers
            .endpoint_for(entry, &self.models_dir, &admitted.plan)
            .await
    }
}

/// The endpoint a holder may use, and what it cost to get it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Rebind {
    pub model_id: String,
    pub base_url: String,
    pub served_model_id: String,
    /// The window the server was started with, when it said; else the
    /// registry's declared window, and `window_source` says which.
    pub served_window: u32,
    /// `server` or `registryDeclared`.
    pub window_source: String,
    /// The server was already up and ready.
    pub warm: bool,
    /// This owner was bound to a different server process before — the old one
    /// was stopped to make room. Its prompt cache is gone and nothing was
    /// carried across: see `cache`.
    pub restarted: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub previous_base_url: Option<String>,
    /// `sameServer` or `cold`. There is no third value. KV and prompt caches
    /// belong to one server process and one model's attention geometry; they
    /// are never transferred, so a new process always starts cold.
    pub cache: String,
}

/// A run's standing binding: which model its rounds lease, as whom.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RunBinding {
    pub model_id: String,
    pub class: LeaseClass,
    #[serde(default)]
    pub parent: Option<String>,
    #[serde(default)]
    pub eligible: Vec<String>,
}

type Observer = Arc<dyn Fn(&LeaseEvent) + Send + Sync>;

// ─────────────────────────────────────────────────────────────────────────────
// The service
// ─────────────────────────────────────────────────────────────────────────────

/// Hands out the card, measured against the machine, one heavy call at a time.
pub struct ModelScheduler {
    registry: Arc<ModelRegistry>,
    backend: Arc<dyn ServingBackend>,
    book: Arc<Mutex<Book>>,
    /// The last endpoint each owner was bound to, so a changed server process
    /// is reported as a restart rather than passed off as the same one.
    bound: Mutex<HashMap<String, String>>,
    /// Run bindings, so a runtime-driven round knows which model it leases.
    runs: Mutex<HashMap<String, RunBinding>>,
    observer: Mutex<Option<Observer>>,
}

impl ModelScheduler {
    /// The production service: one heavy call, `ModelServers` behind it.
    pub fn new(registry: Arc<ModelRegistry>, servers: Arc<ModelServers>) -> Self {
        let models_dir = registry.models_dir().to_path_buf();
        Self::with_backend(
            registry,
            Arc::new(ServersBackend {
                servers,
                models_dir,
            }),
            ONE_HEAVY_CALL,
        )
    }

    /// Any backend, any capacity. Capacity above one is for a measured
    /// co-residency configuration and for tests; production passes
    /// [`ONE_HEAVY_CALL`].
    pub fn with_backend(
        registry: Arc<ModelRegistry>,
        backend: Arc<dyn ServingBackend>,
        capacity: usize,
    ) -> Self {
        Self {
            registry,
            backend,
            book: Arc::new(Mutex::new(Book::new(capacity))),
            bound: Mutex::new(HashMap::new()),
            runs: Mutex::new(HashMap::new()),
            observer: Mutex::new(None),
        }
    }

    /// Sets how long background work waits before it ages into the
    /// interactive tier. Production keeps [`BACKGROUND_AGES_AFTER`].
    pub fn with_aging(self, ages_after: Duration) -> Self {
        self.book().ages_after = ages_after;
        self
    }

    /// Installs what records suspensions, resumptions and expiries.
    ///
    /// Called with the event *before* the capacity it describes is handed on,
    /// so a durable record of "the parent gave way to its child" exists before
    /// the child can do anything.
    pub fn observe(&self, observer: Observer) {
        if let Ok(mut held) = self.observer.lock() {
            *held = Some(observer);
        }
    }

    fn notify(&self, event: &LeaseEvent) {
        let observer = self
            .observer
            .lock()
            .ok()
            .and_then(|held| held.as_ref().map(Arc::clone));
        if let Some(observer) = observer {
            observer(event);
        }
    }

    fn book(&self) -> std::sync::MutexGuard<'_, Book> {
        // A poisoned book is a panic elsewhere while it was held. The state
        // inside is still the last consistent one — every mutation is a single
        // insert or remove — so it is recovered rather than taking every
        // future model call down with it.
        self.book.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// What would have to happen for this model to be usable right now.
    pub fn residency_of(&self, entry: &ModelEntry) -> Residency {
        self.backend.residency(entry)
    }

    /// The book, read once.
    pub fn snapshot(&self) -> LeaseSnapshot {
        self.book().snapshot()
    }

    /// Whether this owner holds the card.
    pub fn holds(&self, owner: &str) -> bool {
        self.book().holders.contains_key(owner)
    }

    // -- Run bindings -----------------------------------------------------

    /// Records which model a run's rounds lease, and as whom.
    pub fn bind_run(&self, run_id: &str, binding: RunBinding) {
        if let Some(parent) = &binding.parent {
            self.book().lineage.insert(run_id.to_string(), parent.clone());
        }
        if let Ok(mut runs) = self.runs.lock() {
            runs.insert(run_id.to_string(), binding);
        }
    }

    /// Binds a run and returns a guard that forgets it — lease, suspension,
    /// binding — when dropped, however the run ends. The leak guard for a
    /// driver with many early returns.
    pub fn bind_run_until_dropped(
        self: &Arc<Self>,
        run_id: &str,
        binding: RunBinding,
    ) -> ForgetOnDrop {
        self.bind_run(run_id, binding);
        ForgetOnDrop {
            scheduler: Arc::clone(self),
            owner: run_id.to_string(),
        }
    }

    /// Removes a run's binding without touching any lease it holds. For a
    /// child loop that has finished while its worker still holds the card.
    pub fn unbind_run(&self, run_id: &str) {
        if let Ok(mut runs) = self.runs.lock() {
            runs.remove(run_id);
        }
        if let Ok(mut bound) = self.bound.lock() {
            bound.remove(run_id);
        }
    }

    pub fn binding(&self, run_id: &str) -> Option<RunBinding> {
        self.runs.lock().ok()?.get(run_id).cloned()
    }

    /// Forgets everything about an owner: its lease, its suspension, its
    /// binding and its lineage. Called when a run ends, however it ends — the
    /// leak guard.
    pub fn forget(&self, owner: &str) {
        let expired = {
            let mut book = self.book();
            book.release_owner(owner);
            book.suspended.remove(owner);
            book.lineage.remove(owner);
            book.dispatch()
        };
        for event in &expired {
            self.notify(event);
        }
        if let Ok(mut runs) = self.runs.lock() {
            runs.remove(owner);
        }
        if let Ok(mut bound) = self.bound.lock() {
            bound.remove(owner);
        }
    }

    // -- Taking and giving back -------------------------------------------

    fn validate(&self, request: &LeaseRequest) -> Result<ModelEntry, SchedulingRefusal> {
        let entry = self
            .registry
            .find(&request.model_id)
            .cloned()
            .ok_or_else(|| SchedulingRefusal::NotRegistered {
                model_id: request.model_id.clone(),
            })?;
        if !request.eligible.is_empty() && !request.eligible.contains(&request.model_id) {
            return Err(SchedulingRefusal::OutsideEligible {
                model_id: request.model_id.clone(),
                eligible: request.eligible.clone(),
            });
        }
        let residency = self.backend.residency(&entry);
        if let Residency::WontFit { .. } = &residency {
            return Err(SchedulingRefusal::WontFit {
                model_id: request.model_id.clone(),
                detail: residency.explain(),
            });
        }
        Ok(entry)
    }

    /// Suspends every ancestor of `request` that holds the card: recorded
    /// first, then released.
    fn suspend_holding_ancestors(&self, request: &LeaseRequest) -> Vec<Suspension> {
        let ancestors = self
            .book()
            .holding_ancestors(&request.owner, request.parent.as_deref());
        let mut suspended = Vec::new();
        for ancestor in ancestors {
            // Recorded before release: build the record from the book, tell the
            // observer, and only then take the holder out.
            let preview = {
                let book = self.book();
                book.holders.get(&ancestor).map(|held| Suspension {
                    owner: ancestor.clone(),
                    model_id: held.model_id.clone(),
                    class: held.class,
                    for_owner: Some(request.owner.clone()),
                    released_capacity: true,
                    because: format!(
                        "{} is awaiting {}, which needs the GPU; its lease is suspended so the two \
                         cannot wait on each other",
                        ancestor, request.owner
                    ),
                    at: chrono::Utc::now().to_rfc3339(),
                })
            };
            let Some(preview) = preview else { continue };
            self.notify(&LeaseEvent::Suspended(preview.clone()));
            let record = {
                let mut book = self.book();
                let record = book.suspend(
                    &ancestor,
                    Some(&request.owner),
                    &preview.because,
                    preview.class,
                    &preview.model_id,
                );
                let expired = book.dispatch();
                drop(book);
                for event in &expired {
                    self.notify(event);
                }
                record
            };
            suspended.push(record);
        }
        suspended
    }

    async fn admit(
        &self,
        request: &LeaseRequest,
        round: bool,
    ) -> Result<Admitted, SchedulingRefusal> {
        self.validate(request)?;
        if request.cancel.is_cancelled() {
            return Err(SchedulingRefusal::Cancelled {
                model_id: request.model_id.clone(),
            });
        }

        let suspended = self.suspend_holding_ancestors(request);
        let started = Instant::now();

        let admitted = {
            let mut book = self.book();
            book.admit(request, round)
        };

        let result = match admitted {
            Admit::Granted(mut admitted) => {
                admitted.suspended = suspended;
                Ok(admitted)
            }
            Admit::Refused(refusal) => Err(refusal),
            Admit::Queued { ticket, mut wake } => {
                // Waited in ticks rather than in one sleep, so that a holder
                // past its hold limit is expired while somebody is waiting for
                // it — not only when some other holder happens to release.
                let deadline = started + request.wait_for;
                let grant = loop {
                    let now = Instant::now();
                    if now >= deadline || request.cancel.is_cancelled() {
                        break None;
                    }
                    let tick = (deadline - now).min(WAIT_TICK);
                    tokio::select! {
                        granted = &mut wake => break granted.ok(),
                        _ = tokio::time::sleep(tick) => {
                            let expired = self.book().dispatch();
                            for event in &expired {
                                self.notify(event);
                            }
                        }
                        _ = request.cancel.cancelled() => break None,
                    }
                };
                let grant = match grant {
                    Some(grant) => Some(grant),
                    None => {
                        // Stopped waiting. Either it is still queued — take it
                        // out — or it was granted in the same instant, in which
                        // case the grant is accepted rather than leaked.
                        let mut book = self.book();
                        if book.abandon(ticket) {
                            None
                        } else {
                            book.holders.get(&request.owner).map(|held| held.grant)
                        }
                    }
                };
                match grant {
                    Some(grant) => {
                        let resumed_after = self.book().take_resumption(&request.owner);
                        Ok(Admitted {
                            grant,
                            class: request.class,
                            waited: started.elapsed(),
                            reentered: false,
                            resumed_after,
                            suspended,
                        })
                    }
                    None => Err(if request.cancel.is_cancelled() {
                        SchedulingRefusal::Cancelled {
                            model_id: request.model_id.clone(),
                        }
                    } else {
                        SchedulingRefusal::WaitedTooLong {
                            model_id: request.model_id.clone(),
                            seconds: request.wait_for.as_secs(),
                        }
                    }),
                }
            }
        };

        if let Ok(admitted) = &result {
            if let Some(after) = admitted.resumed_after {
                self.notify(&LeaseEvent::Resumed {
                    owner: request.owner.clone(),
                    model_id: request.model_id.clone(),
                    class: request.class,
                    suspended_for_ms: after.as_millis() as u64,
                });
            }
        }
        result
    }

    /// Takes the card for Rust-side work, released when the guard drops.
    ///
    /// Re-entrant per owner: a worker holding a guard whose own model rounds
    /// then ask for the card get the same holder, not a deadlock with itself.
    pub async fn acquire(&self, request: LeaseRequest) -> Result<LeaseGuard, SchedulingRefusal> {
        let admitted = self.admit(&request, false).await?;
        Ok(LeaseGuard {
            owner: request.owner,
            model_id: request.model_id,
            grant: admitted.grant,
            admitted,
            book: Arc::clone(&self.book),
        })
    }

    /// Takes the card for one model round driven from the runtime.
    ///
    /// Not a guard, because the round outlives the RPC that started it. Given
    /// back by [`Self::release_round`] — when the round settles, when the run
    /// reaches a tool boundary, or when the run ends ([`Self::forget`]).
    pub async fn acquire_round(&self, request: LeaseRequest) -> Result<Admitted, SchedulingRefusal> {
        self.admit(&request, true).await
    }

    /// Gives back a round's hold. Idempotent. Returns whether the card was
    /// released as a result (it is kept while a Rust-side guard still holds).
    pub fn release_round(&self, owner: &str) -> bool {
        let (released, expired) = {
            let mut book = self.book();
            let released = match book.holders.get_mut(owner) {
                Some(held) => {
                    held.round = false;
                    if held.guards == 0 {
                        book.holders.remove(owner);
                        true
                    } else {
                        false
                    }
                }
                None => false,
            };
            let expired = if released { book.dispatch() } else { Vec::new() };
            (released, expired)
        };
        for event in &expired {
            self.notify(event);
        }
        released
    }

    /// Suspends an owner explicitly: recorded, then released.
    ///
    /// For the path that knows it is about to await capacity-dependent work —
    /// delegation — and wants the record written whether or not the owner was
    /// holding at that instant.
    pub fn suspend(&self, owner: &str, for_owner: Option<&str>, because: &str) -> Suspension {
        let (class, model) = {
            let book = self.book();
            match book.holders.get(owner) {
                Some(held) => (held.class, held.model_id.clone()),
                None => {
                    let binding = self.binding(owner);
                    (
                        binding.as_ref().map(|b| b.class).unwrap_or(LeaseClass::Parent),
                        binding.map(|b| b.model_id).unwrap_or_default(),
                    )
                }
            }
        };
        let preview = Suspension {
            owner: owner.to_string(),
            model_id: model.clone(),
            class,
            for_owner: for_owner.map(str::to_string),
            released_capacity: self.holds(owner),
            because: because.to_string(),
            at: chrono::Utc::now().to_rfc3339(),
        };
        self.notify(&LeaseEvent::Suspended(preview));
        let (record, expired) = {
            let mut book = self.book();
            let record = book.suspend(owner, for_owner, because, class, &model);
            (record, book.dispatch())
        };
        for event in &expired {
            self.notify(event);
        }
        record
    }

    /// Takes the card and serves the model in one step, for Rust-side work that
    /// runs start to finish in one place — an OCR page, a handoff's load, an
    /// administrator's swap. The returned guard releases the card and forgets
    /// the owner when dropped; a failure to serve releases it at once.
    pub async fn serve(
        self: &Arc<Self>,
        request: LeaseRequest,
    ) -> Result<(Rebind, ServedLease), SchedulingRefusal> {
        let owner = request.owner.clone();
        let model_id = request.model_id.clone();
        let guard = self.acquire(request).await?;
        let forget = ForgetOnDrop {
            scheduler: Arc::clone(self),
            owner: owner.clone(),
        };
        let rebind = self.ensure_served(&owner, &model_id).await?;
        Ok((
            rebind,
            ServedLease {
                _guard: guard,
                _forget: forget,
            },
        ))
    }

    /// Makes the holder's model reachable, starting it if it has to.
    ///
    /// Only for a holder: loading without the card could evict a model in the
    /// middle of somebody else's generation. Every call compares the endpoint
    /// with the one this owner had before, so a server that was stopped to make
    /// room and started again is reported as `restarted` with a cold cache.
    pub async fn ensure_served(&self, owner: &str, model_id: &str) -> Result<Rebind, SchedulingRefusal> {
        if !self.holds(owner) {
            return Err(SchedulingRefusal::NotHeld {
                owner: owner.to_string(),
            });
        }
        let entry = self
            .registry
            .find(model_id)
            .cloned()
            .ok_or_else(|| SchedulingRefusal::NotRegistered {
                model_id: model_id.to_string(),
            })?;

        let (endpoint, warm) = match self.backend.warm_endpoint(&entry.id) {
            Some(endpoint) => (endpoint, true),
            None => match self.backend.start(&entry).await {
                Ok(endpoint) => (endpoint, false),
                Err(error) => {
                    return Err(SchedulingRefusal::LoadFailed {
                        failure: LoadFailure::from_serving(&entry.id, &error),
                    })
                }
            },
        };

        let previous = self
            .bound
            .lock()
            .ok()
            .and_then(|mut bound| bound.insert(owner.to_string(), endpoint.base_url.clone()));
        let restarted = previous
            .as_deref()
            .is_some_and(|before| before != endpoint.base_url);
        let (served_window, window_source) = match endpoint.context_tokens {
            Some(tokens) => (tokens, "server"),
            None => (entry.context_length, "registryDeclared"),
        };

        Ok(Rebind {
            model_id: entry.id.clone(),
            base_url: endpoint.base_url.clone(),
            served_model_id: endpoint.served_model_id.clone(),
            served_window,
            window_source: window_source.to_string(),
            warm,
            restarted,
            previous_base_url: previous.filter(|_| restarted),
            cache: if warm && !restarted {
                "sameServer".to_string()
            } else {
                "cold".to_string()
            },
        })
    }
}

/// The card and a served model, held together. See [`ModelScheduler::serve`].
///
/// Fields drop in order: the guard releases the card, then the owner is
/// forgotten.
pub struct ServedLease {
    _guard: LeaseGuard,
    _forget: ForgetOnDrop,
}

/// Forgets an owner when dropped. See [`ModelScheduler::bind_run_until_dropped`].
pub struct ForgetOnDrop {
    scheduler: Arc<ModelScheduler>,
    owner: String,
}

impl Drop for ForgetOnDrop {
    fn drop(&mut self) {
        self.scheduler.forget(&self.owner);
    }
}

/// A Rust-side hold on the card. Dropping it releases the card, unless the
/// owner is also holding for a runtime round.
///
/// Deliberately not `Clone`: two holders of one guard would be two claims on
/// one reservation, and the second release would free something the first
/// still needs.
pub struct LeaseGuard {
    pub owner: String,
    pub model_id: String,
    grant: u64,
    pub admitted: Admitted,
    book: Arc<Mutex<Book>>,
}

/// The name the worker has always used for its hold. Kept so the child's start
/// record reads the same.
pub type ModelLease = LeaseGuard;

impl LeaseGuard {
    pub fn describe(&self) -> String {
        format!(
            "{} (the GPU lease, class {}{})",
            self.model_id,
            self.admitted.class.as_str(),
            if self.admitted.waited > Duration::from_millis(50) {
                format!(", after waiting {:?}", self.admitted.waited)
            } else {
                String::new()
            }
        )
    }

    pub fn grant(&self) -> u64 {
        self.grant
    }
}

impl std::fmt::Debug for LeaseGuard {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("LeaseGuard")
            .field("owner", &self.owner)
            .field("model_id", &self.model_id)
            .field("grant", &self.grant)
            .finish()
    }
}

impl Drop for LeaseGuard {
    fn drop(&mut self) {
        let mut book = self.book.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let release = match book.holders.get_mut(&self.owner) {
            // Only the grant this guard was issued under. A holder that was
            // suspended and re-admitted since has a new grant, and an old guard
            // must not release it.
            Some(held) if held.grant == self.grant => {
                held.guards = held.guards.saturating_sub(1);
                held.guards == 0 && !held.round
            }
            _ => false,
        };
        if release {
            book.holders.remove(&self.owner);
            // Expiries raised here are logged by `dispatch`; there is no
            // observer in reach from a destructor, and none is needed — an
            // expiry is only possible while somebody waits, and that somebody's
            // own admission records it.
            let _ = book.dispatch();
        }
    }
}

/// Writes a lease event onto the owner's own run record.
///
/// What the production observer does. An owner that is not a run the log knows
/// — a child's own id, an OCR batch — has nowhere to write, and that is logged
/// at debug rather than treated as a fault: the parent's record is the one that
/// has to show it gave way, and the parent is a run.
pub fn persist_lease_event(
    events: &crate::agent_runtime::events::TaskEventLog,
    event: &LeaseEvent,
) {
    use crate::agent_runtime::events::{EventDraft, TaskEventType};
    let (kind, payload) = match event {
        LeaseEvent::Suspended(suspension) => (
            TaskEventType::ModelLeaseSuspended,
            serde_json::to_value(suspension).unwrap_or_default(),
        ),
        LeaseEvent::Resumed {
            model_id,
            class,
            suspended_for_ms,
            ..
        } => (
            TaskEventType::ModelLeaseResumed,
            serde_json::json!({
                "modelId": model_id,
                "class": class,
                "suspendedForMs": suspended_for_ms,
            }),
        ),
        LeaseEvent::Expired {
            model_id, held_ms, ..
        } => (
            TaskEventType::ModelLeaseSuspended,
            serde_json::json!({
                "modelId": model_id,
                "expired": true,
                "heldMs": held_ms,
                "because": "the lease was held past its limit while other work waited, and was \
                            expired",
            }),
        ),
    };
    let draft = EventDraft::new(event.owner(), kind, LEASE_ACTOR).with(payload);
    if let Err(error) = events.record(draft) {
        log::debug!(
            "[scheduling] the lease event for {} was not recorded on a run: {error}",
            event.owner()
        );
    }
}

/// Who a lease event is attributed to. Not a person: the scheduler decided.
pub const LEASE_ACTOR: &str = "arjun-scheduler";

// ─────────────────────────────────────────────────────────────────────────────
// Residency: reported, measured, never the concurrency decision
// ─────────────────────────────────────────────────────────────────────────────

/// What has to happen before a model may be used.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Residency {
    /// It is already serving. Nothing has to be loaded or evicted.
    AlreadyWarm,
    /// There is measured room for it alongside what is running.
    FitsAlongside { free_bytes: u64, needs_bytes: u64 },
    /// Something must be evicted first, or the driver would not say what is
    /// free and guessing is not an answer.
    Serialise { because: String },
    /// It cannot run on this machine at all, whatever is evicted.
    WontFit {
        needs_bytes: u64,
        vram_bytes: u64,
        ram_bytes: u64,
    },
}

impl Residency {
    pub fn as_str(&self) -> &'static str {
        match self {
            Residency::AlreadyWarm => "alreadyWarm",
            Residency::FitsAlongside { .. } => "fitsAlongside",
            Residency::Serialise { .. } => "serialise",
            Residency::WontFit { .. } => "wontFit",
        }
    }

    /// Whether loading this model means evicting another one first.
    pub fn needs_exclusive_card(&self) -> bool {
        matches!(self, Residency::Serialise { .. })
    }

    pub fn explain(&self) -> String {
        match self {
            Residency::AlreadyWarm => "the model is already serving".to_string(),
            Residency::FitsAlongside {
                free_bytes,
                needs_bytes,
            } => format!(
                "{} of video memory is free and this model needs about {}, so it loads without \
                 evicting what is already loaded",
                crate::serving::human_bytes(*free_bytes),
                crate::serving::human_bytes(*needs_bytes)
            ),
            Residency::Serialise { because } => because.clone(),
            Residency::WontFit {
                needs_bytes,
                vram_bytes,
                ram_bytes,
            } => format!(
                "this model needs about {} and the machine has {} of video memory and {} of free \
                 system memory, so no worker can run it here",
                crate::serving::human_bytes(*needs_bytes),
                crate::serving::human_bytes(*vram_bytes),
                crate::serving::human_bytes(*ram_bytes)
            ),
        }
    }
}

/// The residency rule, over figures rather than over a machine.
///
/// Pure, so every branch can be tested — including the ones a developer's own
/// card would never produce. The caller measures; this decides.
///
/// ## Why the headroom is not a fraction
///
/// `weights_bytes` is the file, and a served model costs more than its file: the
/// KV cache, the compute buffers and the runtime's own allocation. The planner
/// in [`crate::ai_engine::vram_planner`] models that properly for the model it
/// is about to start. Here the question is coarser — *may a second model join
/// the first* — and the answer has to be conservative, because being wrong in
/// the optimistic direction is the spill this module exists to prevent. So a
/// model only fits alongside when the measured free memory holds its weights
/// **and** [`CO_RESIDENT_HEADROOM`] on top.
pub fn plan_residency(
    weights_bytes: u64,
    free_vram_bytes: u64,
    vram_measured: bool,
    installed_vram_bytes: u64,
    available_ram_bytes: u64,
    something_already_running: bool,
) -> Residency {
    // Nothing on this machine can hold it, whatever is evicted first.
    if weights_bytes > installed_vram_bytes.saturating_add(available_ram_bytes) {
        return Residency::WontFit {
            needs_bytes: weights_bytes,
            vram_bytes: installed_vram_bytes,
            ram_bytes: available_ram_bytes,
        };
    }

    // Nothing else is loaded, so there is nothing to be co-resident *with* and
    // the ordinary admission path will place it.
    if !something_already_running {
        return Residency::FitsAlongside {
            free_bytes: free_vram_bytes,
            needs_bytes: weights_bytes,
        };
    }

    // The driver would not say. "Nobody can say how much is free" does not read
    // as "enough" — see `VramBudget::InstalledOnly`, which exists because
    // routing once planned against a number that was not a measurement.
    if !vram_measured {
        return Residency::Serialise {
            because: "this machine's driver does not report free video memory, so whether a \
                      second model fits alongside the first cannot be measured. The model that \
                      is loaded is released first rather than both being spilled into system \
                      memory."
                .to_string(),
        };
    }

    let needs = weights_bytes.saturating_add(CO_RESIDENT_HEADROOM);
    if free_vram_bytes >= needs {
        Residency::FitsAlongside {
            free_bytes: free_vram_bytes,
            needs_bytes: needs,
        }
    } else {
        Residency::Serialise {
            because: format!(
                "{} of video memory is free and a second model needs about {}, so the loaded \
                 model is released first rather than both being spilled into system memory",
                crate::serving::human_bytes(free_vram_bytes),
                crate::serving::human_bytes(needs)
            ),
        }
    }
}

#[cfg(test)]
#[path = "scheduling_tests.rs"]
pub(crate) mod tests;
