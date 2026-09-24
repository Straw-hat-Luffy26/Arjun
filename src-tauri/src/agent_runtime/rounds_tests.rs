//! The round boundary, driven through the production RPC handlers.
//!
//! Every test here calls `context.refresh`, `context.count`, `context.settle`
//! and `tool.authorize` exactly as the runtime does over the wire, against a
//! real memory graph, a real event log and the real lease service. Only the
//! model server is a fake: it is the one thing a test machine without a GPU
//! cannot provide, and every property asserted here is Rust's, not the model's.

use std::sync::atomic::Ordering;
use std::sync::Arc;

use serde_json::{json, Value};

use super::*;
use crate::agent_runtime::events::TaskEventType;
use crate::agent_runtime::memory::Acl;
use crate::knowledge::graph::runtime_memory::{
    item_id, ItemStatus, MemoryItem, MemoryKind, MemoryScope, Provenance,
};
use crate::knowledge::graph::runtime_store::MemoryGraph;
use crate::policy::Classification;
use crate::subagents::scheduling::tests::{FakeBackend, OCR, QWEN, SPARK};
use crate::subagents::scheduling::{LeaseClass, ModelScheduler, RunBinding, ONE_HEAVY_CALL};

const RUN: &str = "r";

/// Counts one token per `chars` characters of every message and of the tools.
///
/// Two of these with different densities are two tokenizers: the same request
/// costs a different number of tokens under each, which is the whole reason a
/// request must be counted by the model it is going to.
pub(crate) struct PerChars(pub(crate) usize);

#[async_trait::async_trait]
impl RenderedCounter for PerChars {
    async fn count(&self, _base_url: &str, payload: &Value) -> Result<u32, String> {
        let text = text_of(payload);
        let tools = payload.get("tools").map(|t| t.to_string()).unwrap_or_default();
        Ok(((text.chars().count() + tools.chars().count()) / self.0).max(1) as u32)
    }
}

/// A server that cannot render for counting (vLLM, a proxy).
struct CannotRender;

#[async_trait::async_trait]
impl RenderedCounter for CannotRender {
    async fn count(&self, _base_url: &str, _payload: &Value) -> Result<u32, String> {
        Err("apply-template answered 404".into())
    }
}

struct World {
    deps: Arc<RuntimeDeps>,
    leases: Arc<ModelScheduler>,
    backend: Arc<FakeBackend>,
    graph: Arc<MemoryGraph>,
    _dir: tempfile::TempDir,
}

fn world_with(counter: Arc<dyn RenderedCounter>) -> World {
    let (deps, dir) = crate::agent_runtime::tests::deps_with(Arc::new(std::sync::RwLock::new(Some(
        crate::identity::Session::open(crate::identity::User::new(
            "priya",
            "Priya Sharma",
            vec![crate::identity::Role::Employee],
        )),
    ))));
    let backend = FakeBackend::new();
    let registry = crate::subagents::scheduling::tests::registry();
    let leases = Arc::new(ModelScheduler::with_backend(
        Arc::clone(&registry),
        Arc::clone(&backend) as Arc<dyn crate::subagents::scheduling::ServingBackend>,
        ONE_HEAVY_CALL,
    ));
    let events = Arc::clone(&deps.events);
    leases.observe(Arc::new(move |event: &crate::subagents::LeaseEvent| {
        crate::subagents::persist_lease_event(&events, event);
    }));
    let graph = Arc::new(MemoryGraph::in_memory().expect("a graph"));
    let mut owned = Arc::try_unwrap(deps).ok().expect("the only handle");
    owned.memory_graph = Some(Arc::clone(&graph));
    owned.registry = Some(registry);
    owned.leases = Some(Arc::clone(&leases));
    owned.cancellations = Some(Arc::new(crate::agent_runtime::cancellation::RunCancellations::new()));
    owned.rounds = Arc::new(RoundBook::with_counter(counter));
    leases.bind_run(
        RUN,
        RunBinding {
            model_id: SPARK.into(),
            class: LeaseClass::Parent,
            parent: None,
            eligible: Vec::new(),
        },
    );
    World {
        deps: Arc::new(owned),
        leases,
        backend,
        graph,
        _dir: dir,
    }
}

fn world() -> World {
    world_with(Arc::new(PerChars(4)))
}

fn remember(graph: &MemoryGraph, kind: MemoryKind, content: &str) -> MemoryItem {
    let mut item = MemoryItem {
        item_id: item_id(),
        revision: 1,
        kind,
        agent_id: "ag-spark".into(),
        scope: MemoryScope::Task {
            task_id: RUN.into(),
        },
        classification: Classification::Internal,
        acl: Acl::for_classification(Classification::Internal, None),
        creator_model_id: None,
        creator_run_id: Some(RUN.into()),
        provenance: Provenance::Operator {
            user_id: "priya".into(),
        },
        content: content.into(),
        sources: Vec::new(),
        artifacts: Vec::new(),
        confidence: None,
        status: ItemStatus::Proposed,
        valid_from: "2026-01-01T00:00:00Z".into(),
        valid_until: None,
        supersedes: None,
        conflicts_with: Vec::new(),
        causal_parents: Vec::new(),
        idempotency_key: None,
        applies_to: None,
        created_at: "2026-01-01T00:00:00Z".into(),
        updated_at: "2026-01-01T00:00:00Z".into(),
    };
    let committed = graph.commit(item.clone(), None, &[]).expect("commits");
    item.status = committed.status;
    item.revision = committed.revision;
    item
}

fn refresh_request(served_window_claimed: u32) -> Value {
    json!({
        "runId": RUN,
        "taskId": RUN,
        "agentId": "ag-spark",
        "definitionVersion": 3,
        "modelId": SPARK,
        // What the runtime was told at run start. The round must budget
        // against what the server actually has, not this.
        "servedWindow": served_window_claimed,
        "question": "approval note",
        "alreadyCarried": [],
        "capability": "orchestration",
        "reservedToolSchemas": 1_000,
        "reservedOutput": 1_024,
        "reservedFraming": 128,
        "reservedSafety": 256,
    })
}

async fn refresh_at(world: &World, claimed: u32) -> Result<Value, WireError> {
    refresh(refresh_request(claimed), &world.deps).await
}

fn payload(messages: Value) -> Value {
    json!({ "model": SPARK, "messages": messages, "max_tokens": 1_024, "stream": true })
}

fn system_with(blocks: &Value) -> Value {
    let joined = blocks
        .as_array()
        .map(|blocks| {
            blocks
                .iter()
                .filter_map(|b| b.get("content").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join("\n")
        })
        .unwrap_or_default();
    json!({ "role": "system", "content": format!("--- TASK MEMORY ---\n{joined}") })
}

fn kinds_recorded(world: &World) -> Vec<String> {
    world
        .deps
        .events
        .events_since(RUN, 0)
        .expect("history")
        .events
        .into_iter()
        .map(|event| event.event_type.as_str().to_string())
        .collect()
}

fn refused_code(result: &Result<Value, WireError>) -> Option<&str> {
    result.as_ref().err().map(|error| error.code.as_str())
}

// ─────────────────────────────────────────────────────────────────────────────

/// The round takes the card, re-binds the endpoint, and budgets against the
/// window the server actually came up with — not the one the run was told.
#[tokio::test]
async fn a_round_takes_the_lease_and_budgets_against_the_served_window() {
    let world = world();
    world.backend.next_window(SPARK, 8_192);
    remember(&world.graph, MemoryKind::Goal, "Produce the approval note");

    let reply = refresh_at(&world, 32_768).await.expect("the round opens");
    assert_eq!(reply["lease"]["status"], "held");
    assert!(world.leases.holds(RUN));
    assert_eq!(reply["endpoint"]["servedWindow"], 8_192);
    assert_eq!(reply["endpoint"]["windowSource"], "server");
    assert_eq!(reply["budget"]["window"], 8_192, "budgeted against the claimed window");
    assert_eq!(reply["budget"]["reservedSafety"], 256);
    assert_eq!(reply["budget"]["windowSource"], "server");
    assert_eq!(reply["graph"]["available"], true);

    // The manifest is on the durable record: selected versions, the true
    // cursor, the budget.
    let events = world.deps.events.events_since(RUN, 0).expect("history").events;
    let compiled = events
        .iter()
        .find(|event| event.event_type == TaskEventType::ContextCompiled)
        .expect("the manifest was recorded");
    let manifest = &compiled.payload["manifest"];
    assert_eq!(
        manifest["graph"]["graphRevision"].as_i64(),
        Some(world.graph.graph_revision().expect("cursor"))
    );
    assert_eq!(manifest["graph"]["selected"][0]["scope"], "task");
    assert_eq!(manifest["budget"]["window"], 8_192);
    assert_eq!(manifest["manifestVersion"], 3);
}

/// A tool call is the end of the round that issued it. The parent gives the
/// card back before any tool — a delegation in particular — runs.
#[tokio::test]
async fn a_tool_boundary_gives_the_card_back_before_the_tool_runs() {
    let world = world();
    refresh_at(&world, 16_384).await.expect("round");
    assert!(world.leases.holds(RUN));
    let _ = crate::agent_runtime::authorize(
        json!({
            "runId": RUN,
            "toolCallId": "tc-1",
            "tool": "read_scoped_file",
            "args": { "path": "notes.txt" }
        }),
        &world.deps,
    )
    .await;
    assert!(!world.leases.holds(RUN), "the parent held the card into its tool call");
}

/// Settling gives the card back and meets the count with the server's report.
#[tokio::test]
async fn settling_releases_the_card_and_reconciles_the_count() {
    let world = world();
    let opened = refresh_at(&world, 16_384).await.expect("round");
    let request = payload(json!([
        system_with(&opened["blocks"]),
        { "role": "user", "content": "Draft the approval note." }
    ]));
    let counted = count(json!({ "runId": RUN, "payload": request, "callIndex": 0 }), &world.deps)
        .await
        .expect("counted");
    assert_eq!(counted["countedBy"], "tokenizer");
    let tokens = counted["inputTokens"].as_u64().expect("a count") as u32;

    let settled = settle(
        json!({ "runId": RUN, "inputTokens": tokens + 3, "outputTokens": 40, "stopReason": "stop" }),
        &world.deps,
    )
    .expect("settled");
    assert_eq!(settled["released"], true);
    assert!(!world.leases.holds(RUN));
    let responded = world
        .deps
        .events
        .events_since(RUN, 0)
        .expect("history")
        .events
        .into_iter()
        .find(|event| event.event_type == TaskEventType::ModelResponded)
        .expect("recorded");
    assert_eq!(responded.payload["deltaFromCount"], 3);
    assert!(kinds_recorded(&world).contains(&"model_requested".to_string()));
}

/// The mandatory state does not fit: the round is refused with the numbers,
/// nothing is trimmed, the model is not called and the card is given back.
#[tokio::test]
async fn a_mandatory_overflow_refuses_the_round_and_gives_the_card_back() {
    let world = world();
    world.backend.next_window(SPARK, 4_096);
    remember(&world.graph, MemoryKind::Correction, &"every pressure in bar ".repeat(400));

    let refused = refresh_at(&world, 32_768).await;
    assert_eq!(refused_code(&refused), Some(code::ROUND_REFUSED));
    let message = refused.unwrap_err().message;
    assert!(message.contains("4096-token window"), "{message}");
    assert!(message.contains("Nothing was dropped"), "{message}");
    assert!(!world.leases.holds(RUN), "a refused round kept the card");
    // Recorded, overflow and all.
    let compiled = world
        .deps
        .events
        .events_since(RUN, 0)
        .expect("history")
        .events
        .into_iter()
        .find(|event| event.event_type == TaskEventType::ContextCompiled)
        .expect("recorded");
    assert_eq!(compiled.payload["mandatoryOverflowed"], true);
}

/// A round whose context cannot be compiled after it was given the card —
/// here, because nobody is signed in any more — is refused, not degraded: the
/// runtime would otherwise call the model without the card and without the
/// round's corrections. The card goes back at once rather than at the run's
/// next tool boundary or its end.
#[tokio::test]
async fn a_round_whose_context_cannot_be_compiled_is_refused_and_gives_the_card_back() {
    let world = world();
    world.backend.next_window(SPARK, 32_768);
    *world.deps.session.write().expect("session lock") = None;

    let failed = refresh_at(&world, 32_768).await;
    assert_eq!(refused_code(&failed), Some(code::ROUND_REFUSED));
    let message = failed.unwrap_err().message;
    assert!(message.contains("could not be compiled"), "{message}");
    assert!(message.contains("No one is signed in"), "{message}");
    assert!(!world.leases.holds(RUN), "a refused round kept the card");
    assert!(world.leases.snapshot().is_idle());
}

/// The small-window transition: the model's server is stopped to make room
/// for a child and comes back with half the window. The next round is
/// re-bound, told the cache is cold, and compiled for the window it now has —
/// and when the mandatory state no longer fits, refused rather than trimmed.
#[tokio::test]
async fn a_small_window_transition_is_rebound_recompiled_and_refused_when_it_must_be() {
    let world = world();
    world.backend.next_window(SPARK, 16_384);
    world.backend.next_window(SPARK, 4_096);
    // About 3 000 estimated tokens: fits 16 384 less reserves, not 4 096.
    remember(&world.graph, MemoryKind::Plan, &"step ".repeat(2_400));

    let first = refresh_at(&world, 16_384).await.expect("fits the large window");
    // A first start is a new process: cold, and not a restart of anything.
    assert_eq!(first["endpoint"]["cache"], "cold");
    assert_eq!(first["endpoint"]["restarted"], false);
    assert_eq!(first["budget"]["window"], 16_384);
    let first_url = first["endpoint"]["baseUrl"].clone();
    settle(json!({ "runId": RUN }), &world.deps).expect("settled");

    world.backend.evict(SPARK);
    let second = refresh_at(&world, 16_384).await;
    assert_eq!(refused_code(&second), Some(code::ROUND_REFUSED));
    assert!(second.unwrap_err().message.contains("4096-token window"));
    assert!(!world.leases.holds(RUN));
    // It did rebind — to a new process — before it refused.
    let _ = first_url;
    assert_eq!(world.backend.starts.load(Ordering::SeqCst), 2);
}

/// The same request under two tokenizers: one fits, one does not. Only the
/// served model's own count can tell which case a round is in.
#[tokio::test]
async fn overflow_depends_on_the_tokenizer_and_the_dense_one_is_refused() {
    let messages = json!([
        { "role": "user", "content": "PV-2201 gasket torque 47.5 N·m; ".repeat(300) }
    ]);
    // ~9 600 characters: 2 400 tokens at four characters a token, 9 600 at one.
    let sparse = world_with(Arc::new(PerChars(4)));
    sparse.backend.next_window(SPARK, 8_192);
    refresh_at(&sparse, 8_192).await.expect("round");
    let fits = count(json!({ "runId": RUN, "payload": payload(messages.clone()) }), &sparse.deps)
        .await
        .expect("fits under the sparse tokenizer");
    assert_eq!(fits["fit"]["fit"], "fits");

    let dense = world_with(Arc::new(PerChars(1)));
    dense.backend.next_window(SPARK, 8_192);
    refresh_at(&dense, 8_192).await.expect("round");
    let refused = count(json!({ "runId": RUN, "payload": payload(messages) }), &dense.deps).await;
    assert_eq!(refused_code(&refused), Some(code::ROUND_REFUSED));
    let message = refused.unwrap_err().message;
    assert!(message.contains("own template and tokenizer"), "{message}");
    assert!(message.contains("6912"), "the limit was not the window less the reserves: {message}");
    assert!(!dense.leases.holds(RUN), "a refused request kept the card");
}

/// A tool result far larger than the window is not sent: the count refuses it,
/// with the numbers, and the projection shows what the compiler authorised was
/// still present rather than displaced.
#[tokio::test]
async fn a_huge_tool_output_is_refused_before_it_reaches_the_model() {
    let world = world();
    world.backend.next_window(SPARK, 8_192);
    remember(&world.graph, MemoryKind::Goal, "Summarise the scan");
    let opened = refresh_at(&world, 8_192).await.expect("round");

    let huge = "row,value\n".repeat(20_000);
    let request = payload(json!([
        system_with(&opened["blocks"]),
        { "role": "user", "content": "Summarise the scan." },
        { "role": "assistant", "content": null, "tool_calls": [
            { "id": "call-1", "type": "function", "function": { "name": "workspace.read_text", "arguments": "{}" } }
        ]},
        { "role": "tool", "tool_call_id": "call-1", "content": huge }
    ]));
    let refused = count(json!({ "runId": RUN, "payload": request }), &world.deps).await;
    assert_eq!(refused_code(&refused), Some(code::ROUND_REFUSED));
    let requested = world
        .deps
        .events
        .events_since(RUN, 0)
        .expect("history")
        .events
        .into_iter()
        .find(|event| event.event_type == TaskEventType::ModelRequested)
        .expect("recorded");
    assert_eq!(requested.payload["fit"]["fit"], "overflows");
    assert_eq!(requested.payload["projection"]["unaccounted"], json!([]));
}

/// A request whose authorised block was dropped between compilation and the
/// wire is recorded as such.
#[tokio::test]
async fn a_block_dropped_before_the_wire_is_recorded_on_the_request() {
    let world = world();
    remember(&world.graph, MemoryKind::Correction, "Pressures in bar");
    refresh_at(&world, 16_384).await.expect("round");
    let request = payload(json!([{ "role": "user", "content": "Go." }]));
    let counted = count(json!({ "runId": RUN, "payload": request }), &world.deps)
        .await
        .expect("counted");
    let unaccounted = counted["projection"]["unaccounted"].as_array().expect("a list");
    assert!(unaccounted.iter().any(|entry| entry.as_str().unwrap().starts_with("missing:")));
}

/// Images are counted separately, never guessed: unmeasured until the server's
/// own usage report says what they cost, then charged at that cost.
#[tokio::test]
async fn image_cost_is_learned_from_the_server_and_then_charged() {
    let world = world();
    world.backend.next_window(SPARK, 8_192);
    let with_images = || {
        payload(json!([{ "role": "user", "content": [
            { "type": "text", "text": "Read these two pages." },
            { "type": "image_url", "image_url": { "url": "data:image/png;base64,AAAA" } },
            { "type": "image_url", "image_url": { "url": "data:image/png;base64,BBBB" } }
        ]}]))
    };

    refresh_at(&world, 8_192).await.expect("round");
    let first = count(json!({ "runId": RUN, "payload": with_images() }), &world.deps)
        .await
        .expect("allowed, recorded as not fully counted");
    assert_eq!(first["fit"]["fit"], "imagesUnmeasured");
    assert!(first["countedBy"].as_str().unwrap().contains("2 image(s) not yet measured"));
    let text = first["inputTokens"].as_u64().unwrap() as u32;

    let settled = settle(json!({ "runId": RUN, "inputTokens": text + 2 * 3_500 }), &world.deps)
        .expect("settled");
    assert_eq!(settled["imageCostLearned"], 3_500);

    refresh_at(&world, 8_192).await.expect("round");
    let second = count(json!({ "runId": RUN, "payload": with_images() }), &world.deps).await;
    // 2 × 3 500 plus the text is over 8 192 − 1 024 − 256 = 6 912.
    assert_eq!(refused_code(&second), Some(code::ROUND_REFUSED));
}

/// A server that cannot render the request says so; the count is recorded as
/// not made rather than replaced by an estimate.
#[tokio::test]
async fn a_server_that_cannot_render_is_recorded_as_uncounted_not_estimated() {
    let world = world_with(Arc::new(CannotRender));
    refresh_at(&world, 8_192).await.expect("round");
    let counted = count(
        json!({ "runId": RUN, "payload": payload(json!([{ "role": "user", "content": "hi" }])) }),
        &world.deps,
    )
    .await
    .expect("not a refusal");
    assert_eq!(counted["fit"]["fit"], "uncounted");
    assert_eq!(counted["countedBy"], "uncounted");
    assert_eq!(counted["inputTokens"], 0);
}

/// The model will not load. The round is refused as recoverable, the decision
/// is recorded, the card is given back, and no other quantisation is loaded.
#[tokio::test]
async fn a_load_failure_is_refused_as_recoverable_and_changes_nothing_else() {
    let world = world();
    world.backend.fail_with(SPARK, "oom");
    let refused = refresh_at(&world, 16_384).await;
    assert_eq!(refused_code(&refused), Some(code::ROUND_REFUSED));
    let message = refused.unwrap_err().message;
    assert!(message.contains("outOfMemory"), "{message}");
    assert!(message.contains("unchanged"), "{message}");
    assert!(!world.leases.holds(RUN));

    let failed = world
        .deps
        .events
        .events_since(RUN, 0)
        .expect("history")
        .events
        .into_iter()
        .find(|event| event.event_type == TaskEventType::ModelLoadFailed)
        .expect("recorded");
    assert_eq!(failed.payload["kind"], "outOfMemory");
    assert_eq!(failed.payload["decision"], "retryLater");
    assert_eq!(failed.payload["recoverable"], true);
    // Only the failed model was ever asked for.
    assert_eq!(world.backend.starts.load(Ordering::SeqCst), 1);
}

/// A run stopped while it waits for the card stops waiting.
#[tokio::test]
async fn a_stopped_run_stops_waiting_for_the_card() {
    let world = world();
    let _held = world
        .leases
        .acquire(crate::subagents::LeaseRequest::new("someone-else", LeaseClass::Ocr, OCR))
        .await
        .expect("another holder");
    let token = world
        .deps
        .cancellations
        .as_ref()
        .unwrap()
        .register(&[RUN]);
    let deps = Arc::clone(&world.deps);
    let waiting = tokio::spawn(async move { refresh(refresh_request(16_384), &deps).await });
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    token.cancel();
    let refused = tokio::time::timeout(std::time::Duration::from_secs(2), waiting)
        .await
        .expect("cancellation was prompt")
        .expect("joins");
    assert_eq!(refused_code(&refused), Some(code::ROUND_REFUSED));
    assert!(refused.unwrap_err().message.contains("stopped"));
    assert!(world.leases.snapshot().waiting.is_empty());
}

/// An operator's correction recorded between two rounds of one run reaches the
/// next round, as a mandatory constraint, at a later cursor.
#[tokio::test]
async fn a_mid_task_correction_reaches_the_very_next_round() {
    let world = world();
    remember(&world.graph, MemoryKind::Goal, "Produce the approval note");
    let before = refresh_at(&world, 16_384).await.expect("round one");
    assert!(!before["blocks"].to_string().contains("in bar"));
    settle(json!({ "runId": RUN }), &world.deps).expect("settled");

    remember(&world.graph, MemoryKind::Correction, "Use bar, not psi, for every pressure");
    let after = refresh_at(&world, 16_384).await.expect("round two");
    let correction = after["blocks"]
        .as_array()
        .unwrap()
        .iter()
        .find(|block| block["content"].as_str().unwrap().contains("Use bar"))
        .expect("the correction reached the next round");
    assert_eq!(correction["kind"], "constraint");
    assert!(after["graphRevision"].as_i64() > before["graphRevision"].as_i64());
}

/// A parent that delegates to a child needing the card is suspended on the
/// record before the child runs, and resumed on the record at its next round.
#[tokio::test]
async fn delegation_suspends_the_parent_on_the_record_and_its_next_round_resumes_it() {
    let world = world();
    refresh_at(&world, 16_384).await.expect("parent round");
    // What the worker does for a model-driven child.
    world.leases.suspend(RUN, Some("child-1"), "delegated to child-1");
    let child = world
        .leases
        .acquire(crate::subagents::LeaseRequest::new("child-1", LeaseClass::Child, QWEN).child_of(RUN))
        .await
        .expect("the child is admitted at once");
    world.backend.evict(SPARK);
    drop(child);
    world.leases.forget("child-1");

    let resumed = refresh_at(&world, 16_384).await.expect("parent resumes");
    assert_eq!(resumed["endpoint"]["restarted"], true);
    assert_eq!(resumed["endpoint"]["cache"], "cold");
    assert!(resumed["lease"]["resumedAfterMs"].is_u64());

    let kinds = kinds_recorded(&world);
    let suspended = kinds.iter().position(|k| k == "model_lease_suspended").expect("suspended");
    let resumed_at = kinds.iter().position(|k| k == "model_lease_resumed").expect("resumed");
    assert!(suspended < resumed_at);
}

/// A run nobody bound says so instead of pretending to hold the card.
#[tokio::test]
async fn an_unbound_run_says_it_is_unbound() {
    let world = world();
    world.leases.forget(RUN);
    let reply = refresh_at(&world, 16_384).await.expect("round");
    assert_eq!(reply["lease"]["status"], "unbound");
    assert!(reply["endpoint"].is_null());
    assert_eq!(reply["budget"]["window"], 16_384);
}

#[test]
fn the_limit_excludes_the_reply_and_the_safety_reserve_but_not_the_schemas() {
    // Framing and tool schemas are inside the rendered count, so they are not
    // subtracted again.
    assert_eq!(
        judge(Ok(6_912), 0, None, 8_192, 1_024, 256),
        Fit::Fits { total: 6_912, limit: 6_912 }
    );
    assert_eq!(
        judge(Ok(6_913), 0, None, 8_192, 1_024, 256),
        Fit::Overflows { total: 6_913, limit: 6_912, over: 1 }
    );
    assert!(matches!(
        judge(Ok(100), 2, None, 8_192, 1_024, 256),
        Fit::ImagesUnmeasured { images: 2, .. }
    ));
    assert!(matches!(
        judge(Err("no".into()), 0, None, 8_192, 1_024, 256),
        Fit::Uncounted { .. }
    ));
}

/// Round state does not outlive the runs it belongs to.
#[tokio::test]
async fn round_state_for_finished_runs_is_pruned() {
    let world = world();
    refresh_at(&world, 16_384).await.expect("round");
    // A run whose plan has gone — it ended — but whose state was left behind.
    world.deps.rounds.put("finished-run", RoundState::default());
    refresh_at(&world, 16_384).await.expect("next round");
    assert!(world.deps.rounds.state("finished-run").is_none());
    assert!(world.deps.rounds.state(RUN).is_some());
}

/// The compaction's source range reaches the durable record — positions and
/// tool-call ids only, never text.
#[test]
fn a_compactions_source_range_is_recorded_without_any_text() {
    let world = world();
    crate::agent_runtime::recording::remember_loop_event(
        &world.deps,
        &json!({
            "runId": RUN,
            "event": {
                "type": "context_compacted",
                "ordinal": 1,
                "tokensBefore": 9_000,
                "tokensAfter": 4_000,
                "messagesSummarised": 12,
                "refinedExistingSummary": false,
                "toolResultsCleared": 0,
                "sourceRange": {
                    "from": 0,
                    "to": 12,
                    "toolCalls": ["call_0", "call_1"],
                    "text": "this must not be copied"
                },
                "ledger": {}
            }
        }),
    );
    let compacted = world
        .deps
        .events
        .events_since(RUN, 0)
        .expect("history")
        .events
        .into_iter()
        .find(|event| event.event_type == TaskEventType::ContextCompacted)
        .expect("recorded");
    assert_eq!(compacted.payload["sourceRange"]["to"], 12);
    assert_eq!(compacted.payload["sourceRange"]["toolCalls"], json!(["call_0", "call_1"]));
    assert!(!compacted.payload.to_string().contains("must not be copied"));
}
