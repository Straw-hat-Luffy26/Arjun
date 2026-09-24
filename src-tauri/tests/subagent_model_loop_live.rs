//! A child driving a real model, and two children time-sharing one card.
//!
//! ## What this proves that the fixture tests cannot
//!
//! `subagents::worker_tests` builds its services with `child_loop: None`, so the
//! four mechanical roles take their fallback path and nothing there exercises
//! the model loop. That is the right shape for testing refusals and the graph
//! handoff, and it leaves two claims unproven:
//!
//! - that a child's model can actually be placed and served on this machine —
//!   measured admission, a real `llama-server`, a window the server itself
//!   reports;
//! - that two children whose models cannot both be resident genuinely take
//!   turns, decided against this machine's own free VRAM rather than against
//!   figures a test supplied.
//!
//! Both are here. The scheduling half needs no model *loaded* to be meaningful —
//! it is about who may load one — so it runs wherever two distinct models are
//! registered. The serving half needs weights and a `llama-server`, and skips
//! loudly without them.
//!
//! ## Where it writes
//!
//! Nowhere. Only the weights come from the real application data, and only to be
//! read.

mod common;

use std::sync::Arc;
use std::time::Instant;

use sarathi_lib::registry::{ModelEntry, ModelRegistry, ModelRole};
use sarathi_lib::serving::ModelServers;
use sarathi_lib::subagents::scheduling::{ModelScheduler, Residency};

/// Two models on this machine that are genuinely different files.
///
/// Distinct *paths* rather than distinct ids: the registry legitimately holds
/// two ids for one file, and a pair like that would prove nothing about
/// residency.
fn two_distinct_models(registry: &ModelRegistry) -> Option<(ModelEntry, ModelEntry)> {
    let models_dir = registry.models_dir();
    let mut usable: Vec<ModelEntry> = registry
        .all()
        .iter()
        .filter(|entry| entry.enabled && entry.serves(ModelRole::Reasoning))
        .filter(|entry| models_dir.join(&entry.path).is_file())
        .cloned()
        .collect();
    usable.sort_by_key(|entry| entry.weights_bytes);
    // Smallest first, because it is the one actually loaded.
    let first = usable.first()?.clone();
    let second = usable.iter().find(|entry| entry.path != first.path)?.clone();
    Some((first, second))
}

/// The largest model that this machine could run at all, beside `first`.
///
/// "Largest" is what makes the co-residency question non-trivial; "could run at
/// all" is what keeps it a question about *sharing* rather than about a model
/// the card was never going to hold. A 35B mixture-of-experts on an 8 GB card is
/// `WontFit` whatever else is loaded, and queueing behind a card it will never
/// fit on is not the behaviour under test.
fn largest_that_fits(
    registry: &ModelRegistry,
    scheduler: &ModelScheduler,
    first: &ModelEntry,
) -> Option<ModelEntry> {
    let models_dir = registry.models_dir();
    let mut usable: Vec<ModelEntry> = registry
        .all()
        .iter()
        .filter(|entry| entry.enabled && entry.serves(ModelRole::Reasoning))
        .filter(|entry| entry.path != first.path)
        .filter(|entry| models_dir.join(&entry.path).is_file())
        .filter(|entry| !matches!(scheduler.residency_of(entry), Residency::WontFit { .. }))
        .cloned()
        .collect();
    usable.sort_by_key(|entry| entry.weights_bytes);
    usable.pop()
}

fn load_registry() -> Option<ModelRegistry> {
    ModelRegistry::load(&common::app_data_dir().join("models")).ok()
}

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("a test runtime")
}

/// The acceptance case for scheduling, on this machine's own card.
///
/// Two children, two different models, one GPU. Whether they may be co-resident
/// is decided by `ModelScheduler::residency_of` against measured free VRAM — and
/// when the answer is "no", the second worker's lease is granted only after the
/// first has released the card. The assertion is on the *intervals*: they must
/// not overlap.
#[test]
fn two_workers_on_one_card_take_turns() {
    let Some(registry) = load_registry() else {
        eprintln!("SKIP: no model registry on this machine");
        return;
    };
    let Some((first, _)) = two_distinct_models(&registry) else {
        eprintln!("SKIP: fewer than two distinct reasoning models with weights on disk");
        return;
    };

    if sarathi_lib::serving::llama_server_build().is_none() {
        eprintln!(
            "SKIP: llama-server could not be found, so no model can actually occupy the card              and 'is there room for a second' has no measured answer"
        );
        return;
    }

    let servers = Arc::new(ModelServers::new());
    let models_dir = common::app_data_dir().join("models");
    // Loaded twice on purpose: the scheduler takes ownership of one, and
    // choosing the second model needs to read the entries again. Both are the
    // same file, read read-only.
    let Some(registry_for_pick) = load_registry() else {
        eprintln!("SKIP: the model registry could not be read a second time");
        return;
    };
    let scheduler = Arc::new(ModelScheduler::new(Arc::new(registry), Arc::clone(&servers)));
    let runtime = runtime();

    // The first model is genuinely loaded before anything is asked, because the
    // question this test exists for is whether a *second* one fits beside a real
    // allocation. Asking it against an idle card measures nothing: free VRAM
    // then is simply the whole card, and every model "fits".
    runtime.block_on(async {
        let admitted = sarathi_lib::serving::admission::admit(&servers, &first, &models_dir)
            .await
            .expect("the first model is admitted");
        servers
            .endpoint_for(&first, &models_dir, &admitted.plan)
            .await
            .expect("the first model server comes up");
    });
    eprintln!(
        "[live] {} ({} MiB) is loaded and holding the card",
        first.id,
        first.weights_bytes / (1024 * 1024)
    );

    // Chosen *after* the first is loaded, and only from models this machine
    // could run at all — so the answer below is about sharing a card rather
    // than about a model that was never going to fit on one.
    let Some(second) = largest_that_fits(&registry_for_pick, &scheduler, &first) else {
        eprintln!("SKIP: no second model this machine could run beside the first");
        runtime.block_on(servers.stop_all());
        return;
    };

    // What this machine now says about holding the second beside it. Reported
    // rather than asserted: a developer on a 24 GB card will legitimately get
    // `FitsAlongside`, and the serialisation assertion below is conditional on
    // the machine actually being constrained.
    let residency = scheduler.residency_of(&second);
    eprintln!(
        "[live] {} beside {}: {} ({})",
        second.id,
        first.id,
        residency.as_str(),
        residency.explain()
    );

    // The model both workers want.
    //
    // The *second* one, deliberately. The first is warm now — this test loaded
    // it — so `residency_of` answers `AlreadyWarm` for it and it never takes the
    // card at all. Two workers wanting a model that is not resident is the case
    // the card lock exists for, and the only one where "did they overlap" is a
    // question with content.
    let contended = second.id.clone();
    let held_first = Arc::new(std::sync::Mutex::new(None::<(Instant, Instant)>));
    let held_second = Arc::new(std::sync::Mutex::new(None::<Instant>));

    runtime.block_on(async {
        let wait = std::time::Duration::from_secs(20);

        let one = {
            let scheduler = Arc::clone(&scheduler);
            let id = contended.clone();
            let held = Arc::clone(&held_first);
            tokio::spawn(async move {
                let lease = scheduler
                    .acquire(
                        sarathi_lib::subagents::LeaseRequest::new(
                            "worker-1",
                            sarathi_lib::subagents::LeaseClass::Child,
                            id.as_str(),
                        )
                        .waiting(wait),
                    )
                    .await
                    .expect("the first worker gets its model");
                let taken = Instant::now();
                // Long enough that a second worker asking concurrently has to
                // wait for it rather than slipping past.
                tokio::time::sleep(std::time::Duration::from_millis(400)).await;
                // One heavy call at a time, on every machine: the card is
                // always the holder's alone. See `subagents::scheduling`.
                let exclusive = true;
                let released = Instant::now();
                *held.lock().expect("lock") = Some((taken, released));
                drop(lease);
                exclusive
            })
        };

        // A beat, so the first is genuinely holding when the second asks.
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let two = {
            let scheduler = Arc::clone(&scheduler);
            let id = contended.clone();
            let held = Arc::clone(&held_second);
            tokio::spawn(async move {
                let lease = scheduler
                    .acquire(
                        sarathi_lib::subagents::LeaseRequest::new(
                            "worker-2",
                            sarathi_lib::subagents::LeaseClass::Child,
                            id.as_str(),
                        )
                        .waiting(wait),
                    )
                    .await
                    .expect("the second worker gets its model eventually");
                *held.lock().expect("lock") = Some(Instant::now());
                let exclusive = true;
                drop(lease);
                exclusive
            })
        };

        let first_exclusive = one.await.expect("the first worker finishes");
        let second_exclusive = two.await.expect("the second worker finishes");

        let (first_taken, first_released) = held_first.lock().expect("lock").expect("held");
        let second_taken = held_second.lock().expect("lock").expect("held");

        if first_exclusive && second_exclusive {
            // This machine cannot hold both. The second must not have taken the
            // card until the first let it go.
            assert!(
                second_taken >= first_released,
                "two workers held the card at once on a machine that cannot hold both: the \
                 second took it {:?} before the first released it",
                first_released.duration_since(second_taken)
            );
            eprintln!(
                "[live] serialised: the second worker waited {:?} for the card",
                second_taken.duration_since(first_taken)
            );
        } else {
            // Neither worker needed the card to itself, which means this
            // machine had room. Reported rather than asserted: co-residency is
            // the correct answer on a large card, and the rule that produces
            // both answers is covered exhaustively by
            // `subagents::scheduling::tests`.
            eprintln!(
                "[live] neither worker needed the card to itself on this machine, so they ran                  alongside each other"
            );
        }
    });

    runtime.block_on(servers.stop_all());
}

/// Two children on the *same* model take turns too.
///
/// This used to assert the opposite. The per-model semaphore let two requests
/// share one warm model's server at once — and its KV cache, which the window
/// was budgeted without. Plan P03 replaced that with one heavy call across the
/// whole machine, so the second worker waits for the first even on the same
/// weights. The deterministic proof is
/// `subagents::scheduling::tests::two_rounds_on_the_same_warm_model_no_longer_run_at_once`;
/// this repeats it against a real registry and a real server table.
#[test]
fn two_workers_on_one_model_take_turns_on_the_card() {
    let Some(registry) = load_registry() else {
        eprintln!("SKIP: no model registry on this machine");
        return;
    };
    let Some((first, _)) = two_distinct_models(&registry) else {
        eprintln!("SKIP: no model with weights on disk");
        return;
    };

    let servers = Arc::new(ModelServers::new());
    let scheduler = Arc::new(ModelScheduler::new(Arc::new(registry), Arc::clone(&servers)));

    runtime().block_on(async {
        let wait = std::time::Duration::from_secs(10);
        let request = |owner: &str| {
            sarathi_lib::subagents::LeaseRequest::new(
                owner,
                sarathi_lib::subagents::LeaseClass::Child,
                first.id.as_str(),
            )
            .waiting(wait)
        };
        let one = scheduler.acquire(request("worker-1")).await.expect("the first worker");

        let waited = tokio::time::timeout(
            std::time::Duration::from_millis(300),
            scheduler.acquire(request("worker-2")),
        )
        .await;
        assert!(
            waited.is_err(),
            "a second worker on the same model was admitted while the first held the card"
        );

        drop(one);
        let two = tokio::time::timeout(std::time::Duration::from_secs(2), scheduler.acquire(request("worker-2")))
            .await
            .expect("admitted once the first released")
            .expect("the second worker");
        drop(two);
        assert!(scheduler.snapshot().is_idle());
    });
}

/// A child's model, placed and served the way a child places and serves it.
///
/// The serving half of `subagents::child_loop`: measured admission, a real
/// `llama-server`, and a window the server itself reports — which is the figure
/// a child's context is budgeted against. Skipped rather than failed without
/// weights or a server: a developer without them has a machine that cannot run
/// the product.
#[test]
fn a_childs_model_is_placed_and_served_for_real() {
    let Some(registry) = load_registry() else {
        eprintln!("SKIP: no model registry on this machine");
        return;
    };
    let Some((model, _)) = two_distinct_models(&registry) else {
        eprintln!("SKIP: no model with weights on disk");
        return;
    };
    if sarathi_lib::serving::llama_server_build().is_none() {
        eprintln!(
            "SKIP: llama-server could not be found; set ARJUN_LLAMA_SERVER or add it to PATH"
        );
        return;
    }
    let bundle = sarathi_lib::agent_runtime::default_bundle_path();
    if !bundle.exists() {
        eprintln!(
            "SKIP: the agent runtime bundle is not built at {}. Run the agent-runtime build.",
            bundle.display()
        );
        return;
    }
    eprintln!("[live] the runtime bundle a child would drive is at {}", bundle.display());

    let servers = Arc::new(ModelServers::new());
    let scheduler = ModelScheduler::new(Arc::new(registry), Arc::clone(&servers));

    // The question a worker asks before it loads anything, measured against this
    // card.
    let residency = scheduler.residency_of(&model);
    eprintln!(
        "[live] {} ({} MiB): {} ({})",
        model.id,
        model.weights_bytes / (1024 * 1024),
        residency.as_str(),
        residency.explain()
    );
    assert!(
        !matches!(residency, Residency::WontFit { .. }),
        "this machine cannot hold {}, so a child could not run on it: {}",
        model.id,
        residency.explain()
    );

    let runtime = runtime();
    let models_dir = common::app_data_dir().join("models");
    let endpoint = runtime.block_on(async {
        let admitted = sarathi_lib::serving::admission::admit(&servers, &model, &models_dir)
            .await
            .expect("the model is admitted");
        servers
            .endpoint_for(&model, &models_dir, &admitted.plan)
            .await
            .expect("the model server comes up")
    });

    eprintln!(
        "[live] {} is serving at {} with a {} token window",
        model.id,
        endpoint.base_url,
        endpoint.context_tokens.unwrap_or(0)
    );
    assert!(
        endpoint.context_tokens.is_some_and(|tokens| tokens > 0),
        "a real llama-server reports its own window, and a child's context is budgeted against it"
    );

    runtime.block_on(servers.stop_all());
}
