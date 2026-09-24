//! The lease service, driven through its public surface against a fake backend.
//!
//! Every test here asserts on the book itself — who holds the card, who waits,
//! who is suspended — rather than on a log line, because the book is what every
//! heavy caller in the product now asks.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU16, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use super::*;
use crate::registry::tests::entry;
use crate::registry::{ModelManifest, ModelRole, Runtime};

/// A backend with no processes: endpoints are invented ports on loopback, and
/// a test decides which models are warm, which fail and what window each start
/// comes up with.
#[derive(Default)]
pub(crate) struct FakeBackend {
    warm: Mutex<HashMap<String, Endpoint>>,
    windows: Mutex<HashMap<String, Vec<u32>>>,
    failures: Mutex<HashMap<String, &'static str>>,
    pub(crate) starts: AtomicUsize,
    port: AtomicU16,
}

impl FakeBackend {
    pub(crate) fn new() -> Arc<Self> {
        let backend = Self::default();
        backend.port.store(41_000, Ordering::SeqCst);
        Arc::new(backend)
    }

    /// The window the next start of `model` comes up with; queued.
    pub(crate) fn next_window(&self, model: &str, window: u32) {
        self.windows
            .lock()
            .unwrap()
            .entry(model.to_string())
            .or_default()
            .push(window);
    }

    pub(crate) fn fail_with(&self, model: &str, how: &'static str) {
        self.failures.lock().unwrap().insert(model.to_string(), how);
    }

    /// What admission does to an idle server when something else needs room.
    pub(crate) fn evict(&self, model: &str) {
        self.warm.lock().unwrap().remove(model);
    }
}

#[async_trait::async_trait]
impl ServingBackend for FakeBackend {
    fn warm_endpoint(&self, model_id: &str) -> Option<Endpoint> {
        self.warm.lock().unwrap().get(model_id).cloned()
    }

    fn residency(&self, entry: &ModelEntry) -> Residency {
        if self.warm.lock().unwrap().contains_key(&entry.id) {
            Residency::AlreadyWarm
        } else if entry.weights_bytes > 64 * 1024 * 1024 * 1024 {
            Residency::WontFit {
                needs_bytes: entry.weights_bytes,
                vram_bytes: 8 * 1024 * 1024 * 1024,
                ram_bytes: 16 * 1024 * 1024 * 1024,
            }
        } else {
            Residency::Serialise {
                because: "an 8 GB card with another model loaded".into(),
            }
        }
    }

    async fn start(&self, entry: &ModelEntry) -> Result<Endpoint, ServingError> {
        self.starts.fetch_add(1, Ordering::SeqCst);
        if let Some(how) = self.failures.lock().unwrap().get(&entry.id).copied() {
            return Err(match how {
                "oom" => ServingError::NeverReady {
                    model: entry.id.clone(),
                    base_url: "http://127.0.0.1:1/v1".into(),
                    detail: "cudaMalloc failed: out of memory".into(),
                },
                _ => ServingError::WeightsMissing {
                    model: entry.id.clone(),
                    path: "/nowhere.gguf".into(),
                },
            });
        }
        let port = self.port.fetch_add(1, Ordering::SeqCst);
        let window = {
            let mut windows = self.windows.lock().unwrap();
            windows
                .get_mut(&entry.id)
                .and_then(|queued| (!queued.is_empty()).then(|| queued.remove(0)))
        };
        let endpoint = Endpoint {
            base_url: format!("http://127.0.0.1:{port}/v1"),
            served_model_id: entry.id.clone(),
            managed: true,
            runtime: Runtime::LlamaCpp,
            context_tokens: window,
        };
        self.warm
            .lock()
            .unwrap()
            .insert(entry.id.clone(), endpoint.clone());
        Ok(endpoint)
    }
}

pub(crate) const SPARK: &str = "spark-x2.5-4b-q8_0";
pub(crate) const QWEN: &str = "qwen3.5-9b-q4_k_s";
pub(crate) const OCR: &str = "unlimited-ocr-q6-k";
pub(crate) const REVIEWER: &str = "nemotron3-nano-4b-q8_0";

pub(crate) fn registry() -> Arc<ModelRegistry> {
    let models = vec![
        entry(SPARK, 4.0, vec![ModelRole::Reasoning]),
        entry(QWEN, 9.0, vec![ModelRole::Coding, ModelRole::Vision]),
        entry(OCR, 3.0, vec![ModelRole::DocumentOcr]),
        entry(REVIEWER, 4.0, vec![ModelRole::Reasoning]),
        entry("enormous", 400.0, vec![ModelRole::Reasoning]),
    ];
    Arc::new(
        ModelRegistry::from_manifest(
            ModelManifest { models },
            std::path::PathBuf::from("models/registry.json"),
        )
        .expect("a well-formed manifest"),
    )
}

fn scheduler(backend: Arc<FakeBackend>) -> ModelScheduler {
    ModelScheduler::with_backend(registry(), backend, ONE_HEAVY_CALL)
}

/// Records every lease event, in order, with the book's holders at the moment
/// the observer was called — which is what proves "recorded before released".
fn recorder(scheduler: &ModelScheduler) -> Arc<Mutex<Vec<(LeaseEvent, Vec<String>)>>> {
    let seen: Arc<Mutex<Vec<(LeaseEvent, Vec<String>)>>> = Arc::default();
    let sink = Arc::clone(&seen);
    let book = Arc::clone(&scheduler.book);
    scheduler.observe(Arc::new(move |event: &LeaseEvent| {
        let holders: Vec<String> = book
            .lock()
            .map(|book| book.holders.keys().cloned().collect())
            .unwrap_or_default();
        sink.lock().unwrap().push((event.clone(), holders));
    }));
    seen
}

fn quick(owner: &str, class: LeaseClass, model: &str) -> LeaseRequest {
    LeaseRequest::new(owner, class, model).waiting(Duration::from_secs(5))
}

/// Whether a future completes within a short window. Used to show that a
/// request is *waiting*, not merely slow.
async fn completes_soon<F: std::future::Future>(future: F) -> Option<F::Output> {
    tokio::time::timeout(Duration::from_millis(150), future).await.ok()
}

// ─────────────────────────────────────────────────────────────────────────────
// One heavy call
// ─────────────────────────────────────────────────────────────────────────────

/// The regression this phase exists for. The per-model semaphore permitted
/// [`REQUESTS_PER_MODEL_BEFORE_P03`] requests on one warm model at once; the
/// global book permits one.
#[tokio::test]
async fn two_rounds_on_the_same_warm_model_no_longer_run_at_once() {
    assert_eq!(REQUESTS_PER_MODEL_BEFORE_P03, 2, "the old policy, for the record");
    let backend = FakeBackend::new();
    let scheduler = Arc::new(scheduler(Arc::clone(&backend)));

    scheduler
        .acquire_round(quick("run-a", LeaseClass::Parent, SPARK))
        .await
        .expect("the first round is admitted");

    let second = {
        let scheduler = Arc::clone(&scheduler);
        tokio::spawn(async move {
            scheduler
                .acquire_round(quick("run-b", LeaseClass::Parent, SPARK))
                .await
        })
    };
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert!(!second.is_finished(), "a second round on the same model ran alongside the first");
    assert_eq!(scheduler.snapshot().waiting.len(), 1);

    assert!(scheduler.release_round("run-a"));
    let admitted = second.await.expect("joins").expect("admitted after release");
    assert!(admitted.waited >= Duration::from_millis(90));
    assert_eq!(scheduler.snapshot().holders.len(), 1);
    assert_eq!(scheduler.snapshot().holders[0].owner, "run-b");
}

/// Parent, children, OCR, vision and background all ask one book. Many at
/// once, measured: never more than one in flight.
#[tokio::test]
async fn every_class_of_heavy_work_shares_one_slot() {
    let backend = FakeBackend::new();
    let scheduler = Arc::new(scheduler(backend));
    let in_flight = Arc::new(AtomicUsize::new(0));
    let peak = Arc::new(AtomicUsize::new(0));

    let classes = [
        (LeaseClass::Parent, SPARK),
        (LeaseClass::Child, QWEN),
        (LeaseClass::Ocr, OCR),
        (LeaseClass::Vision, QWEN),
        (LeaseClass::Background, SPARK),
    ];
    let mut tasks = Vec::new();
    for round in 0..3 {
        for (index, (class, model)) in classes.iter().enumerate() {
            let scheduler = Arc::clone(&scheduler);
            let in_flight = Arc::clone(&in_flight);
            let peak = Arc::clone(&peak);
            let owner = format!("{}-{round}-{index}", class.as_str());
            let (class, model) = (*class, *model);
            tasks.push(tokio::spawn(async move {
                let guard = scheduler
                    .acquire(quick(&owner, class, model).waiting(Duration::from_secs(20)))
                    .await
                    .expect("admitted eventually");
                let now = in_flight.fetch_add(1, Ordering::SeqCst) + 1;
                peak.fetch_max(now, Ordering::SeqCst);
                tokio::time::sleep(Duration::from_millis(5)).await;
                in_flight.fetch_sub(1, Ordering::SeqCst);
                drop(guard);
            }));
        }
    }
    for task in tasks {
        task.await.expect("joins");
    }
    assert_eq!(peak.load(Ordering::SeqCst), 1, "two heavy calls were in flight at once");
    assert!(scheduler.snapshot().is_idle(), "a lease leaked: {:?}", scheduler.snapshot());
}

/// The measurement above is only worth something if it would see a second
/// holder. With a capacity of two it does.
#[tokio::test]
async fn the_in_flight_measurement_would_see_a_second_holder() {
    let backend = FakeBackend::new();
    let scheduler = Arc::new(ModelScheduler::with_backend(registry(), backend, 2));
    let a = scheduler.acquire(quick("a", LeaseClass::Child, QWEN)).await.expect("a");
    let b = completes_soon(scheduler.acquire(quick("b", LeaseClass::Child, SPARK)))
        .await
        .expect("capacity two admits a second holder at once")
        .expect("b");
    assert_eq!(scheduler.snapshot().holders.len(), 2);
    drop((a, b));
    assert!(scheduler.snapshot().is_idle());
}

// ─────────────────────────────────────────────────────────────────────────────
// Parent and child
// ─────────────────────────────────────────────────────────────────────────────

/// The deadlock the plan names: a parent holding the card while it awaits a
/// child that needs the card. The child's request suspends the parent —
/// recorded while the parent still held it — and is admitted at once.
#[tokio::test]
async fn a_child_suspends_its_parent_before_taking_the_card_and_never_waits_on_it() {
    let backend = FakeBackend::new();
    let scheduler = Arc::new(scheduler(backend));
    let seen = recorder(&scheduler);

    scheduler
        .acquire_round(quick("parent", LeaseClass::Parent, SPARK))
        .await
        .expect("parent round");

    let child = completes_soon(
        scheduler.acquire(quick("child", LeaseClass::Child, OCR).child_of("parent")),
    )
    .await
    .expect("the child waited on the parent that is waiting on it")
    .expect("child admitted");
    assert_eq!(child.admitted.suspended.len(), 1);
    assert_eq!(child.admitted.suspended[0].owner, "parent");
    assert!(child.admitted.suspended[0].released_capacity);

    // Recorded *before* release: at the moment the observer ran, the parent
    // was still the holder.
    let events = seen.lock().unwrap().clone();
    let (first, holders_then) = &events[0];
    assert!(matches!(first, LeaseEvent::Suspended(s) if s.owner == "parent" && s.for_owner.as_deref() == Some("child")));
    assert_eq!(holders_then, &vec!["parent".to_string()], "the suspension was recorded after the release");

    let snapshot = scheduler.snapshot();
    assert_eq!(snapshot.holders.len(), 1);
    assert_eq!(snapshot.holders[0].owner, "child");
    assert_eq!(snapshot.suspended.len(), 1);

    // The parent's next round waits for the child, then resumes.
    let parent_again = {
        let scheduler = Arc::clone(&scheduler);
        tokio::spawn(async move {
            scheduler
                .acquire_round(quick("parent", LeaseClass::Parent, SPARK))
                .await
        })
    };
    tokio::time::sleep(Duration::from_millis(60)).await;
    assert!(!parent_again.is_finished(), "the parent resumed while its child held the card");
    drop(child);
    let resumed = parent_again.await.expect("joins").expect("parent resumes");
    assert!(resumed.resumed_after.is_some());
    assert!(scheduler.snapshot().suspended.is_empty());
    assert!(seen
        .lock()
        .unwrap()
        .iter()
        .any(|(event, _)| matches!(event, LeaseEvent::Resumed { owner, .. } if owner == "parent")));
}

/// Spark → OCR → Qwen → reviewer → Spark: every hop is one holder, the parent
/// is suspended while its children run, and nothing is left behind.
#[tokio::test]
async fn a_parent_and_three_children_in_turn_never_overlap_and_leave_the_book_empty() {
    let backend = FakeBackend::new();
    let scheduler = Arc::new(scheduler(Arc::clone(&backend)));
    let seen = recorder(&scheduler);

    scheduler.bind_run(
        "spark-run",
        RunBinding {
            model_id: SPARK.into(),
            class: LeaseClass::Parent,
            parent: None,
            eligible: Vec::new(),
        },
    );
    scheduler
        .acquire_round(quick("spark-run", LeaseClass::Parent, SPARK))
        .await
        .expect("parent");
    let first = scheduler.ensure_served("spark-run", SPARK).await.expect("served");

    for (child, model, class) in [
        ("ocr-child", OCR, LeaseClass::Ocr),
        ("qwen-child", QWEN, LeaseClass::Child),
        ("review-child", REVIEWER, LeaseClass::Child),
    ] {
        // Admission for the child evicts the parent's idle server, as it does
        // on an 8 GB card.
        backend.evict(SPARK);
        let guard = scheduler
            .acquire(quick(child, class, model).child_of("spark-run"))
            .await
            .expect("child admitted");
        let served = scheduler.ensure_served(child, model).await.expect("child served");
        assert_eq!(served.model_id, model);
        assert_eq!(scheduler.snapshot().holders.len(), 1);
        drop(guard);
        scheduler.forget(child);
    }

    // The parent comes back to a server that was stopped under it.
    scheduler
        .acquire_round(quick("spark-run", LeaseClass::Parent, SPARK))
        .await
        .expect("parent resumes");
    let rebound = scheduler.ensure_served("spark-run", SPARK).await.expect("rebound");
    assert!(rebound.restarted, "a new server process was passed off as the old one");
    assert_eq!(rebound.cache, "cold");
    assert_ne!(rebound.base_url, first.base_url);

    scheduler.forget("spark-run");
    assert!(scheduler.snapshot().is_idle());
    assert!(scheduler.snapshot().suspended.is_empty());
    let suspensions = seen
        .lock()
        .unwrap()
        .iter()
        .filter(|(event, _)| matches!(event, LeaseEvent::Suspended(_)))
        .count();
    assert_eq!(suspensions, 1, "only the first child found the parent holding");
}

/// A grandchild suspends every ancestor that holds, not only its parent.
#[tokio::test]
async fn a_grandchild_suspends_a_holding_grandparent() {
    let backend = FakeBackend::new();
    let scheduler = scheduler(backend);
    scheduler
        .acquire_round(quick("grandparent", LeaseClass::Parent, SPARK))
        .await
        .expect("grandparent");
    // The middle generation is known to the book but not holding.
    scheduler.bind_run(
        "parent",
        RunBinding {
            model_id: QWEN.into(),
            class: LeaseClass::Child,
            parent: Some("grandparent".into()),
            eligible: Vec::new(),
        },
    );
    let grandchild = completes_soon(
        scheduler.acquire(quick("grandchild", LeaseClass::Child, OCR).child_of("parent")),
    )
    .await
    .expect("the grandchild waited on its grandparent")
    .expect("admitted");
    assert_eq!(grandchild.admitted.suspended[0].owner, "grandparent");
}

/// Delegation suspends explicitly, and the record exists even when the parent
/// had already released at its tool boundary.
#[tokio::test]
async fn an_explicit_suspension_is_recorded_even_when_nothing_was_held() {
    let backend = FakeBackend::new();
    let scheduler = scheduler(backend);
    let seen = recorder(&scheduler);
    scheduler.bind_run(
        "parent",
        RunBinding {
            model_id: SPARK.into(),
            class: LeaseClass::Parent,
            parent: None,
            eligible: Vec::new(),
        },
    );
    let suspension = scheduler.suspend("parent", Some("child"), "awaiting child");
    assert!(!suspension.released_capacity);
    assert_eq!(suspension.model_id, SPARK);
    assert_eq!(seen.lock().unwrap().len(), 1);

    let admitted = scheduler
        .acquire_round(quick("parent", LeaseClass::Parent, SPARK))
        .await
        .expect("resumes");
    assert!(admitted.resumed_after.is_some(), "the resumption was not recognised");
}

// ─────────────────────────────────────────────────────────────────────────────
// Bounded, timed, cancellable, fair
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn the_queue_is_bounded_and_refuses_rather_than_growing() {
    let backend = FakeBackend::new();
    let scheduler = Arc::new(scheduler(backend));
    let holder = scheduler.acquire(quick("holder", LeaseClass::Parent, SPARK)).await.expect("h");

    let mut waiters = Vec::new();
    for index in 0..MAX_WAITING {
        let scheduler = Arc::clone(&scheduler);
        waiters.push(tokio::spawn(async move {
            scheduler
                .acquire(quick(&format!("w{index}"), LeaseClass::Child, QWEN))
                .await
                .map(drop)
        }));
    }
    tokio::time::sleep(Duration::from_millis(80)).await;
    assert_eq!(scheduler.snapshot().waiting.len(), MAX_WAITING);

    let refused = completes_soon(scheduler.acquire(quick("one-too-many", LeaseClass::Child, QWEN)))
        .await
        .expect("a full queue answered at once");
    assert!(matches!(refused, Err(SchedulingRefusal::QueueFull { .. })), "{refused:?}");

    drop(holder);
    for waiter in waiters {
        waiter.await.expect("joins").expect("every queued waiter is served in turn");
    }
    assert!(scheduler.snapshot().is_idle());
}

#[tokio::test]
async fn a_wait_that_times_out_leaves_nothing_behind() {
    let backend = FakeBackend::new();
    let scheduler = scheduler(backend);
    let _holder = scheduler.acquire(quick("holder", LeaseClass::Parent, SPARK)).await.expect("h");

    let refused = scheduler
        .acquire(quick("late", LeaseClass::Child, QWEN).waiting(Duration::from_millis(60)))
        .await
        .expect_err("timed out");
    assert!(matches!(refused, SchedulingRefusal::WaitedTooLong { .. }));
    let snapshot = scheduler.snapshot();
    assert!(snapshot.waiting.is_empty(), "a timed-out waiter stayed queued");
    assert_eq!(snapshot.holders.len(), 1);
}

#[tokio::test]
async fn a_cancelled_wait_returns_promptly_and_leaves_nothing_behind() {
    let backend = FakeBackend::new();
    let scheduler = Arc::new(scheduler(backend));
    let _holder = scheduler.acquire(quick("holder", LeaseClass::Parent, SPARK)).await.expect("h");

    let stop = CancelToken::never();
    let waiting = {
        let scheduler = Arc::clone(&scheduler);
        let stop = stop.clone();
        tokio::spawn(async move {
            scheduler
                .acquire(
                    quick("stopped", LeaseClass::Child, QWEN)
                        .waiting(Duration::from_secs(60))
                        .cancelled_by(stop),
                )
                .await
                .map(drop)
        })
    };
    tokio::time::sleep(Duration::from_millis(40)).await;
    stop.cancel();
    let refused = completes_soon(waiting)
        .await
        .expect("cancellation was not prompt")
        .expect("joins")
        .expect_err("cancelled");
    assert!(matches!(refused, SchedulingRefusal::Cancelled { .. }));
    assert!(scheduler.snapshot().waiting.is_empty());

    // Already cancelled: refused before queueing at all.
    let refused = scheduler
        .acquire(quick("x", LeaseClass::Child, QWEN).cancelled_by(CancelToken::cancelled_now()))
        .await
        .expect_err("cancelled");
    assert!(matches!(refused, SchedulingRefusal::Cancelled { .. }));
}

/// Interactive work goes first while background work is fresh…
#[tokio::test]
async fn fresh_background_work_waits_behind_interactive_work() {
    let backend = FakeBackend::new();
    let scheduler = Arc::new(scheduler(backend));
    let holder = scheduler.acquire(quick("holder", LeaseClass::Parent, SPARK)).await.expect("h");
    let order = Arc::new(Mutex::new(Vec::new()));

    let spawn = |owner: &'static str, class: LeaseClass| {
        let scheduler = Arc::clone(&scheduler);
        let order = Arc::clone(&order);
        tokio::spawn(async move {
            let guard = scheduler.acquire(quick(owner, class, SPARK)).await.expect("admitted");
            order.lock().unwrap().push(owner);
            tokio::time::sleep(Duration::from_millis(5)).await;
            drop(guard);
        })
    };
    let background = spawn("background", LeaseClass::Background);
    tokio::time::sleep(Duration::from_millis(20)).await;
    let interactive = spawn("interactive", LeaseClass::Child);
    tokio::time::sleep(Duration::from_millis(20)).await;
    drop(holder);
    background.await.unwrap();
    interactive.await.unwrap();
    assert_eq!(*order.lock().unwrap(), vec!["interactive", "background"]);
}

/// …and once it has aged, it is served in arrival order, so a stream of
/// interactive work cannot starve it.
#[tokio::test]
async fn aged_background_work_is_not_starved_by_newer_interactive_work() {
    let backend = FakeBackend::new();
    let scheduler =
        Arc::new(scheduler(backend).with_aging(Duration::from_millis(10)));
    let holder = scheduler.acquire(quick("holder", LeaseClass::Parent, SPARK)).await.expect("h");
    let order = Arc::new(Mutex::new(Vec::new()));

    let spawn = |owner: &'static str, class: LeaseClass| {
        let scheduler = Arc::clone(&scheduler);
        let order = Arc::clone(&order);
        tokio::spawn(async move {
            let guard = scheduler.acquire(quick(owner, class, SPARK)).await.expect("admitted");
            order.lock().unwrap().push(owner);
            drop(guard);
        })
    };
    let background = spawn("background", LeaseClass::Background);
    tokio::time::sleep(Duration::from_millis(30)).await;
    let interactive = spawn("interactive", LeaseClass::Child);
    tokio::time::sleep(Duration::from_millis(10)).await;
    drop(holder);
    background.await.unwrap();
    interactive.await.unwrap();
    assert_eq!(*order.lock().unwrap(), vec!["background", "interactive"]);
}

/// A holder that never gives the card back — a runtime that died between
/// taking a round and settling it — is expired once somebody is waiting.
#[tokio::test]
async fn a_holder_past_its_limit_is_expired_for_a_waiter_and_the_expiry_is_recorded() {
    let backend = FakeBackend::new();
    let scheduler = scheduler(backend);
    let seen = recorder(&scheduler);
    scheduler
        .acquire_round(
            quick("stranded", LeaseClass::Parent, SPARK).holding_at_most(Duration::from_millis(20)),
        )
        .await
        .expect("stranded round");

    let admitted = scheduler
        .acquire_round(quick("next", LeaseClass::Parent, SPARK).waiting(Duration::from_secs(3)))
        .await
        .expect("admitted after the stranded holder was expired");
    assert!(admitted.waited < Duration::from_secs(2));
    assert!(seen
        .lock()
        .unwrap()
        .iter()
        .any(|(event, _)| matches!(event, LeaseEvent::Expired { owner, .. } if owner == "stranded")));
    assert!(!scheduler.holds("stranded"));
}

// ─────────────────────────────────────────────────────────────────────────────
// What may be leased
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn a_model_outside_the_definitions_eligible_set_is_refused() {
    let backend = FakeBackend::new();
    let scheduler = scheduler(backend);
    let refused = scheduler
        .acquire(quick("child", LeaseClass::Child, QWEN).eligible(vec![SPARK.into()]))
        .await
        .expect_err("outside the set");
    assert!(matches!(refused, SchedulingRefusal::OutsideEligible { .. }));
    assert!(refused.explain().contains(SPARK));
    assert!(scheduler.snapshot().is_idle());

    // Inside the set, and an empty set, both pass.
    drop(
        scheduler
            .acquire(quick("child", LeaseClass::Child, SPARK).eligible(vec![SPARK.into()]))
            .await
            .expect("inside"),
    );
    drop(scheduler.acquire(quick("child", LeaseClass::Child, QWEN)).await.expect("any"));
}

#[tokio::test]
async fn unregistered_and_impossible_models_are_refused_without_queueing() {
    let backend = FakeBackend::new();
    let scheduler = scheduler(backend);
    assert!(matches!(
        scheduler.acquire(quick("x", LeaseClass::Child, "nobody")).await,
        Err(SchedulingRefusal::NotRegistered { .. })
    ));
    assert!(matches!(
        scheduler.acquire(quick("x", LeaseClass::Child, "enormous")).await,
        Err(SchedulingRefusal::WontFit { .. })
    ));
    assert!(scheduler.snapshot().is_idle());
}

// ─────────────────────────────────────────────────────────────────────────────
// Re-entry, stale guards and leaks
// ─────────────────────────────────────────────────────────────────────────────

/// A worker's guard and its own model rounds are one holder.
#[tokio::test]
async fn an_owner_re_entering_is_one_holder_not_a_deadlock_with_itself() {
    let backend = FakeBackend::new();
    let scheduler = scheduler(backend);
    let guard = scheduler.acquire(quick("child", LeaseClass::Child, QWEN)).await.expect("guard");
    let round = completes_soon(scheduler.acquire_round(quick("child", LeaseClass::Child, QWEN)))
        .await
        .expect("the child deadlocked with itself")
        .expect("round");
    assert!(round.reentered);
    assert_eq!(round.grant, guard.grant());

    // The round settles; the guard still holds.
    assert!(!scheduler.release_round("child"));
    assert!(scheduler.holds("child"));
    drop(guard);
    assert!(scheduler.snapshot().is_idle());
}

/// A guard issued before a suspension cannot release the grant that followed.
#[tokio::test]
async fn an_old_guard_cannot_release_a_newer_grant() {
    let backend = FakeBackend::new();
    let scheduler = scheduler(backend);
    let old = scheduler.acquire(quick("owner", LeaseClass::Parent, SPARK)).await.expect("old");
    scheduler.suspend("owner", None, "test");
    let new = scheduler.acquire(quick("owner", LeaseClass::Parent, SPARK)).await.expect("new");
    assert_ne!(old.grant(), new.grant());
    drop(old);
    assert!(scheduler.holds("owner"), "a stale guard released a newer grant");
    drop(new);
    assert!(scheduler.snapshot().is_idle());
}

/// However a driver leaves — an early `?`, a panic unwinding — the run's
/// lease and binding go with it.
#[tokio::test]
async fn a_bound_run_is_forgotten_when_its_driver_leaves_by_any_path() {
    let backend = FakeBackend::new();
    let scheduler = Arc::new(scheduler(backend));
    async fn driver(scheduler: &Arc<ModelScheduler>) -> Result<(), String> {
        let _bound = scheduler.bind_run_until_dropped(
            "run",
            RunBinding {
                model_id: SPARK.into(),
                class: LeaseClass::Parent,
                parent: None,
                eligible: Vec::new(),
            },
        );
        scheduler
            .acquire_round(quick("run", LeaseClass::Parent, SPARK))
            .await
            .map_err(|refusal| refusal.explain())?;
        Err("the run failed half way".into())
    }
    assert!(driver(&scheduler).await.is_err());
    assert!(scheduler.snapshot().is_idle(), "{:?}", scheduler.snapshot());
    assert!(scheduler.binding("run").is_none());
}

/// One OCR page: lease, serve, read, done — and nothing left behind, whether
/// the page was read or the model would not load.
#[tokio::test]
async fn serving_one_piece_of_work_leaves_nothing_behind_either_way() {
    let backend = FakeBackend::new();
    let scheduler = Arc::new(scheduler(Arc::clone(&backend)));
    {
        let (rebind, _lease) = scheduler
            .serve(quick("ocr:page-1", LeaseClass::Ocr, OCR))
            .await
            .expect("served");
        assert_eq!(rebind.model_id, OCR);
        assert!(scheduler.holds("ocr:page-1"));
    }
    assert!(scheduler.snapshot().is_idle());

    backend.evict(OCR);
    backend.fail_with(OCR, "missing");
    let refused = scheduler
        .serve(quick("ocr:page-2", LeaseClass::Ocr, OCR))
        .await
        .err()
        .expect("the model would not load");
    assert!(matches!(refused, SchedulingRefusal::LoadFailed { .. }));
    assert!(scheduler.snapshot().is_idle(), "a failed load kept the card");
}

#[tokio::test]
async fn forgetting_an_owner_releases_everything_it_had() {
    let backend = FakeBackend::new();
    let scheduler = scheduler(backend);
    scheduler.bind_run(
        "run",
        RunBinding {
            model_id: SPARK.into(),
            class: LeaseClass::Parent,
            parent: None,
            eligible: Vec::new(),
        },
    );
    scheduler
        .acquire_round(quick("run", LeaseClass::Parent, SPARK))
        .await
        .expect("round");
    scheduler.forget("run");
    let snapshot = scheduler.snapshot();
    assert!(snapshot.is_idle());
    assert!(snapshot.suspended.is_empty());
    assert!(scheduler.binding("run").is_none());
}

// ─────────────────────────────────────────────────────────────────────────────
// Serving, rebinding and failure
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn a_model_is_never_served_to_an_owner_without_the_lease() {
    let backend = FakeBackend::new();
    let scheduler = scheduler(Arc::clone(&backend));
    let refused = scheduler.ensure_served("nobody", SPARK).await.expect_err("not held");
    assert!(matches!(refused, SchedulingRefusal::NotHeld { .. }));
    assert_eq!(backend.starts.load(Ordering::SeqCst), 0, "a model was loaded without a lease");
}

#[tokio::test]
async fn a_warm_rebind_is_the_same_server_and_says_so() {
    let backend = FakeBackend::new();
    let scheduler = scheduler(Arc::clone(&backend));
    scheduler.acquire_round(quick("run", LeaseClass::Parent, SPARK)).await.expect("round");
    let first = scheduler.ensure_served("run", SPARK).await.expect("started");
    assert!(!first.warm);
    let second = scheduler.ensure_served("run", SPARK).await.expect("warm");
    assert!(second.warm && !second.restarted);
    assert_eq!(second.cache, "sameServer");
    assert_eq!(backend.starts.load(Ordering::SeqCst), 1);
}

/// The small-window transition at the serving layer: the parent's server was
/// stopped and came back with a smaller window. The rebind reports the new
/// window and that nothing was carried across.
#[tokio::test]
async fn a_rebind_onto_a_smaller_window_reports_the_window_it_actually_got() {
    let backend = FakeBackend::new();
    backend.next_window(SPARK, 16_384);
    backend.next_window(SPARK, 8_192);
    let scheduler = scheduler(Arc::clone(&backend));

    scheduler.acquire_round(quick("run", LeaseClass::Parent, SPARK)).await.expect("round");
    let before = scheduler.ensure_served("run", SPARK).await.expect("served");
    assert_eq!(before.served_window, 16_384);
    assert_eq!(before.window_source, "server");

    backend.evict(SPARK);
    let after = scheduler.ensure_served("run", SPARK).await.expect("rebound");
    assert_eq!(after.served_window, 8_192);
    assert!(after.restarted);
    assert_eq!(after.cache, "cold");
    assert_eq!(after.previous_base_url.as_deref(), Some(before.base_url.as_str()));
}

#[tokio::test]
async fn an_out_of_memory_load_is_classified_and_the_quantisation_is_untouched() {
    let backend = FakeBackend::new();
    backend.fail_with(QWEN, "oom");
    let scheduler = scheduler(Arc::clone(&backend));
    scheduler.acquire_round(quick("run", LeaseClass::Child, QWEN)).await.expect("round");
    let refused = scheduler.ensure_served("run", QWEN).await.expect_err("oom");
    match &refused {
        SchedulingRefusal::LoadFailed { failure } => {
            assert_eq!(failure.kind, crate::serving::fallback::LoadFailureKind::OutOfMemory);
            assert_eq!(failure.model_id, QWEN, "a different model was loaded in its place");
        }
        other => panic!("expected a load failure, got {other:?}"),
    }
    // The caller decides what the failure means; the book gives the card back
    // when told, and nothing is left holding it.
    scheduler.forget("run");
    assert!(scheduler.snapshot().is_idle());
    assert_eq!(
        scheduler.registry.find(QWEN).and_then(|entry| entry.quantization.clone()),
        Some("Q4_K_M".to_string()),
        "the registry entry's quantisation changed"
    );
}

/// A restart: a new process has a new book. Nothing is held across it, and a
/// run coming back is served cold — there is no cache to carry and none is
/// pretended.
#[tokio::test]
async fn after_a_restart_nothing_is_held_and_the_resumed_run_starts_cold() {
    let backend = FakeBackend::new();
    let before = scheduler(Arc::clone(&backend));
    before.acquire_round(quick("run", LeaseClass::Parent, SPARK)).await.expect("round");
    before.ensure_served("run", SPARK).await.expect("served");
    drop(before);

    // The process went away; so did its servers.
    let backend = FakeBackend::new();
    let after = scheduler(Arc::clone(&backend));
    assert!(after.snapshot().is_idle(), "a lease survived a restart");
    after.acquire_round(quick("run", LeaseClass::Parent, SPARK)).await.expect("round");
    let rebound = after.ensure_served("run", SPARK).await.expect("served");
    assert!(!rebound.warm);
    assert_eq!(rebound.cache, "cold");
}

#[test]
fn residency_is_still_reported_and_is_not_the_concurrency_rule() {
    const GIB: u64 = 1024 * 1024 * 1024;
    // The machine this product is built for: an 8 GB card with a model already
    // on it. A second 4 GB model does not join it.
    assert!(plan_residency(4 * GIB, 3 * GIB, true, 8 * GIB, 16 * GIB, true).needs_exclusive_card());
    // A small model with room still fits alongside…
    assert!(matches!(
        plan_residency(2 * GIB, 6 * GIB, true, 24 * GIB, 32 * GIB, true),
        Residency::FitsAlongside { .. }
    ));
    // …weights that only just fit do not…
    assert!(plan_residency(2 * GIB + GIB / 2, 3 * GIB, true, 8 * GIB, 16 * GIB, true)
        .needs_exclusive_card());
    // …an unmeasurable card is not an optimistic answer…
    assert!(plan_residency(2 * GIB, 24 * GIB, false, 24 * GIB, 32 * GIB, true)
        .explain()
        .contains("cannot be measured"));
    // …and a model bigger than the machine is refused rather than queued.
    assert!(matches!(
        plan_residency(64 * GIB, GIB, true, 8 * GIB, 16 * GIB, true),
        Residency::WontFit { .. }
    ));
}
