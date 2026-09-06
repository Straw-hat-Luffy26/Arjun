//! The two protocol implementations, actually talking to each other.
//!
//! `agent_runtime::protocol` and `agent-runtime/src/protocol.ts` are one
//! contract written twice. Each side's unit tests check its own half against
//! literals; only this test checks that the halves agree, by starting the real
//! Node child and holding a conversation with it.
//!
//! It needs the bundle built (`npm run build --prefix agent-runtime`). Rather
//! than skipping when it is absent — a skip reads as a pass and would hide the
//! runtime being broken — it fails and says what to run.

use std::path::PathBuf;
use std::sync::{Arc, RwLock};

use sarathi_lib::agent_runtime::workspace::Workspace;
use sarathi_lib::agent_runtime::{default_bundle_path, AgentRuntime, RuntimeDeps};
use sarathi_lib::orchestrator::approvals::ApprovalQueue;
use sarathi_lib::identity::{Role, Session, User};
use sarathi_lib::knowledge::KnowledgeIndex;

fn bundle() -> PathBuf {
    let path = default_bundle_path();
    assert!(
        path.exists(),
        "the agent runtime bundle is missing at {}.\n\
         Build it first:  npm run build --prefix agent-runtime",
        path.display()
    );
    path
}

fn deps() -> (Arc<RuntimeDeps>, tempfile::TempDir) {
    let dir = tempfile::tempdir().expect("temp dir");
    let index = KnowledgeIndex::open(dir.path()).expect("index opens");
    let session = Arc::new(RwLock::new(Some(Session::open(User::new(
        "priya",
        "Priya Sharma",
        vec![Role::Employee],
    )))));
    // One workspace, for the run these tests drive. A run without one has every
    // path-taking tool refused, which is correct but makes for a poor test of
    // the transport.
    let workspaces = Arc::new(std::sync::Mutex::new(std::collections::HashMap::new()));
    workspaces.lock().expect("fresh lock").insert(
        "run-1".to_string(),
        Workspace::create(dir.path(), "run-1").expect("workspace"),
    );

    (
        Arc::new(RuntimeDeps {
            index: Arc::new(index),
            session,
            workspaces,
            approvals: Arc::new(ApprovalQueue::new()),
            calculations: Arc::default(),
            passages: Arc::default(),
            produced: Arc::default(),
            calls: Arc::default(),
            // A plan for each run these tests drive, permitting every tool.
            //
            // Registered rather than absent, because absent no longer means
            // "no budget applies" — it means the run does not exist, and the
            // catalogue, the gateway and the executor all refuse it. Leaving
            // this empty would still pass, by testing the transport with a run
            // that is offered no tools at all, which is not the shape these
            // tests are meant to exercise.
            plans: {
                let table: Arc<
                    std::sync::Mutex<
                        std::collections::HashMap<
                            String,
                            sarathi_lib::orchestrator::plan::PlanRun,
                        >,
                    >,
                > = Arc::default();
                {
                    let mut plans = table.lock().expect("fresh lock");
                    for run_id in ["run-1", "r", "follow-up-1", "follow-up-2"] {
                        plans.insert(
                            run_id.to_string(),
                            sarathi_lib::orchestrator::plan::PlanRun::new(
                                run_id,
                                vec!["do the work".to_string()],
                                sarathi_lib::orchestrator::plan::Budget::standard(
                                    sarathi_lib::orchestrator::tools::ToolName::ALL.to_vec(),
                                ),
                            ),
                        );
                    }
                }
                table
            },
            // Written to the same directory the rest of the run's state lives
            // in, so a tool call replayed across these tests behaves as it
            // would in the application.
            events: Arc::new(
                sarathi_lib::agent_runtime::events::TaskEventLog::open(dir.path())
                    .expect("a task event log"),
            ),
            skills: Arc::new(sarathi_lib::skills::SkillRegistry::open(
                dir.path().join("__no_skills__"),
            )),
            // The test does not register any hooks; the empty default is
            // exactly what a deployment with no custom checks would hold.
            hooks: Arc::new(sarathi_lib::hooks::HookRegistry::default()),
            memory: Arc::new(sarathi_lib::agent_runtime::memory::MemoryStore::open(dir.path())),
            checkpoints: Arc::default(),
            emit: Arc::new(|_| {}),
            emit_durable: Arc::new(|_| {}),
            // Present, so a tool refused for want of a dependency cannot be
            // mistaken for a wire problem. The subagent and multimodal
            // systems have their own tests.
            subagents: Arc::new(sarathi_lib::subagents::SubagentManager::new(
                Vec::new(),
                Arc::new(
                    sarathi_lib::agent_runtime::events::TaskEventLog::in_memory()
                        .expect("an event log"),
                ),
            )),
            multimodal: Arc::new(
                sarathi_lib::knowledge::MultimodalIndex::open(dir.path())
                    .expect("a multimodal index"),
            ),
            // Durable: this test is about the wire between the two processes,
            // and a degraded installation has its own tests in `audit_health`.
            audit_health: Arc::new(sarathi_lib::agent_runtime::audit_health::AuditHealth::durable()),
        documents: Arc::new(
            sarathi_lib::agent_runtime::documents::DocumentStore::open(dir.path())
                .expect("an extraction store"),
        ),
        run_to_conversation: Arc::new(
            sarathi_lib::agent_runtime::conversations::RunToConversation::new(),
        ),
        }),
        // Returned so the directory outlives the test; dropping it early would
        // delete the SQLite file out from under the runtime.
        dir,
    )
}

/// Node has to be on PATH. Reported as a skip rather than a failure because a
/// machine without Node is a deployment gap, not a defect in this code — Phase 5
/// packages a Node binary and this becomes unconditional.
fn node_present() -> bool {
    std::process::Command::new("node")
        .arg("--version")
        .output()
        .is_ok()
}

#[tokio::test]
async fn the_runtime_answers_across_the_language_boundary() {
    if !node_present() {
        eprintln!("skipping: node is not on PATH");
        return;
    }
    let (deps, _dir) = deps();
    let runtime = AgentRuntime::spawn(deps, Arc::new(|_| {}), bundle()).expect("runtime starts");

    let health = runtime
        .request("health", serde_json::json!({}))
        .await
        .expect("the runtime answers health");

    assert_eq!(health["ready"], true);
    assert!(
        health["node"].as_str().unwrap_or_default().starts_with('v'),
        "expected a node version, got {:?}",
        health["node"]
    );

    runtime.shutdown().await;
}

#[tokio::test]
async fn an_unknown_method_comes_back_as_an_error_not_a_hang() {
    if !node_present() {
        eprintln!("skipping: node is not on PATH");
        return;
    }
    let (deps, _dir) = deps();
    let runtime = AgentRuntime::spawn(deps, Arc::new(|_| {}), bundle()).expect("runtime starts");

    let outcome = runtime
        .request("no.such.method", serde_json::json!({}))
        .await;

    assert!(outcome.is_err(), "an unknown method must not resolve");
    assert!(outcome.unwrap_err().to_string().contains("no.such.method"));

    runtime.shutdown().await;
}

/// The sovereignty invariant, enforced in the child and observed from here.
#[tokio::test]
async fn a_run_against_a_public_endpoint_is_refused_by_the_runtime_itself() {
    if !node_present() {
        eprintln!("skipping: node is not on PATH");
        return;
    }
    let (deps, _dir) = deps();
    let runtime = AgentRuntime::spawn(deps, Arc::new(|_| {}), bundle()).expect("runtime starts");

    let outcome = runtime
        .request(
            "run.start",
            serde_json::json!({
                "runId": "run-1",
                "prompt": "hello",
                "systemPrompt": "s",
                "messageId": "msg-fixture",
                "model": {
                    "id": "gpt-4",
                    "provider": "openai",
                    "baseUrl": "https://api.openai.com/v1"
                }
            }),
        )
        .await;

    let error = outcome.expect_err("a public endpoint must be refused").to_string();
    assert!(
        error.contains("not loopback"),
        "expected a loopback refusal, got: {error}"
    );

    runtime.shutdown().await;
}

/// Aborting something that is not running is an ordinary race, not a failure.
#[tokio::test]
async fn aborting_a_finished_run_is_not_an_error() {
    if !node_present() {
        eprintln!("skipping: node is not on PATH");
        return;
    }
    let (deps, _dir) = deps();
    let runtime = AgentRuntime::spawn(deps, Arc::new(|_| {}), bundle()).expect("runtime starts");

    let outcome = runtime
        .request("run.abort", serde_json::json!({ "runId": "never-started" }))
        .await
        .expect("abort answers");

    assert_eq!(outcome["aborted"], false);

    runtime.shutdown().await;
}

/// Steering something that is not running is an ordinary race, not a failure.
///
/// The pair with `aborting_a_finished_run_is_not_an_error`: both controls have
/// to be safe to press at the moment a run happens to end, or an operator
/// learns to distrust them.
#[tokio::test]
async fn steering_a_finished_run_is_not_an_error() {
    if !node_present() {
        eprintln!("skipping: node is not on PATH");
        return;
    }
    let (deps, _dir) = deps();
    let runtime = AgentRuntime::spawn(deps, Arc::new(|_| {}), bundle()).expect("runtime starts");

    let outcome = runtime
        .request(
            "run.steer",
            serde_json::json!({ "runId": "never-started", "text": "use the 2019 revision" }),
        )
        .await
        .expect("steer answers");

    assert_eq!(outcome["steered"], false);

    runtime.shutdown().await;
}

/// An empty correction would do nothing, so it is refused rather than accepted
/// and silently dropped.
#[tokio::test]
async fn an_empty_correction_is_refused_by_the_runtime() {
    if !node_present() {
        eprintln!("skipping: node is not on PATH");
        return;
    }
    let (deps, _dir) = deps();
    let runtime = AgentRuntime::spawn(deps, Arc::new(|_| {}), bundle()).expect("runtime starts");

    let outcome = runtime
        .request("run.steer", serde_json::json!({ "runId": "r", "text": "   " }))
        .await;

    assert!(outcome.is_err(), "an empty correction must not be accepted");

    runtime.shutdown().await;
}

/// Diagnostics must not reach stdout, because stdout is the channel.
///
/// The runtime rebinds `console.*` to stderr for exactly this reason. If that
/// guard regressed, the first log line would desynchronise the framing and the
/// health call below would fail instead of answering.
#[tokio::test]
async fn runtime_logging_does_not_corrupt_the_channel() {
    if !node_present() {
        eprintln!("skipping: node is not on PATH");
        return;
    }
    let (deps, _dir) = deps();
    let runtime = AgentRuntime::spawn(deps, Arc::new(|_| {}), bundle()).expect("runtime starts");

    // The runtime writes a readiness line at start-up. Several round trips after
    // that prove the framing survived it.
    for _ in 0..3 {
        let health = runtime
            .request("health", serde_json::json!({}))
            .await
            .expect("the channel stays parseable after the runtime logs");
        assert_eq!(health["ready"], true);
    }

    runtime.shutdown().await;
}

/// End-to-end proof that streaming is real and that the message-stream contract
/// is correct.
///
/// Spawns the real runtime, points it at a real local model server, and
/// captures every `run.event` notification as it arrives on the live channel.
/// Asserts that the events a chat surface would consume are shaped the way
/// the front-end expects: `message_start` carries the front-end's
/// `messageId`, every visible `message_update` carries a non-empty `delta`,
/// `thinking_delta` and `toolcall_delta` are not exposed as assistant text,
/// and `message_end` carries the right `finishReason`.
///
/// Skips when the local model server is not reachable, because this is a
/// proof against a real model — not a fixture. The skip is a fail-safe that
/// prints the endpoint it tried, so a missing model is reported explicitly
/// rather than as a generic test failure.
#[tokio::test]
async fn message_stream_events_carry_message_id_and_text_deltas() {
    if !node_present() {
        eprintln!("skipping: node is not on PATH");
        return;
    }

    // A local OpenAI-compatible endpoint. The default of the live development
    // deployment, and the one llama-server emits when the activator starts a
    // managed model. If the machine has no model loaded, skip — the test is
    // about the wire contract, not about whether a model exists.
    let base_url = std::env::var("ARJUN_TEST_MODEL_BASE_URL")
        .unwrap_or_else(|_| "http://127.0.0.1:61353/v1".to_string());
    if !endpoint_reachable(&base_url).await {
        eprintln!("skipping: local model at {base_url} is not reachable");
        return;
    }
    // One local model, several tests that want it.
    //
    // These share a single llama-server, and cargo runs tests in parallel by
    // default. Two of them decoding at once starve each other: a long answer
    // holds the slots while a short one waits past its own timeout, which
    // surfaces as an unrelated test failing intermittently. Serialised on the
    // scarce resource rather than made lenient, so a real regression still
    // fails and a busy machine does not.
    let _one_at_a_time = real_model_lock().lock().await;

    let (deps, _dir) = deps();
    // Collect every event the runtime emits. Held under a Mutex so the
    // closures on the runtime side can append without `&mut` everywhere.
    let events: Arc<std::sync::Mutex<Vec<serde_json::Value>>> =
        Arc::new(std::sync::Mutex::new(Vec::new()));
    let sink_events = events.clone();
    let emit: Arc<dyn Fn(serde_json::Value) + Send + Sync> = Arc::new(move |value| {
        sink_events
            .lock()
            .map(|mut v| v.push(value))
            .unwrap_or_else(|e| e.into_inner().push(serde_json::Value::Null));
    });
    let runtime = AgentRuntime::spawn(deps, emit, bundle()).expect("runtime starts");

    let run_id = "stream-e2e-run-1";
    let message_id = "msg-e2e-1";

    // The real run. Resolve on `run.start` returning; streaming events arrive
    // on the live channel in parallel and are captured by the sink above.
    let _outcome = runtime
        .request(
            "run.start",
            serde_json::json!({
                "runId": run_id,
                "messageId": message_id,
                "prompt": "hi",
                "systemPrompt": "Answer in one short sentence.",
                "model": {
                    "id": "gemma-4-E4B-it",
                    "provider": "sovereign-local",
                    "baseUrl": base_url,
                }
            }),
        )
        .await
        .expect("run.start resolves once the loop is done");

    // Give the writer a moment to flush the last few notifications that were
    // emitted between the loop returning and the `run.start` resolve.
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

    let collected = events.lock().map(|v| v.clone()).unwrap_or_default();
    runtime.shutdown().await;

    // Every event the runtime emitted. The sink stores the params object the
    // notification carried, which for `run.event` is `{ runId, event: ... }`.
    let run_events: Vec<serde_json::Value> = collected
        .into_iter()
        .filter(|v| v.get("event").is_some())
        .collect();

    assert!(
        !run_events.is_empty(),
        "the runtime emitted no run.event notifications"
    );

    // 1. Every event is tagged with the runId we asked for.
    for event in &run_events {
        assert_eq!(
            event["runId"].as_str(),
            Some(run_id),
            "event runId mismatch: {event}"
        );
    }

    // 2. message_start is the first message-stream event and carries the
    //    front-end's messageId.
    let start = run_events
        .iter()
        .find(|e| e["event"]["type"] == "message_start")
        .unwrap_or_else(|| {
            panic!(
                "no message_start in stream: {:#?}",
                run_events
                    .iter()
                    .map(|e| e["event"]["type"].as_str().unwrap_or("?"))
                    .collect::<Vec<_>>()
            )
        });
    assert_eq!(start["event"]["messageId"].as_str(), Some(message_id));
    assert_eq!(start["event"]["role"].as_str(), Some("assistant"));

    // 3. At least one message_update arrives with a non-empty `delta`. The
    //    proof that the model actually streamed text.
    let updates: Vec<&serde_json::Value> = run_events
        .iter()
        .filter(|e| e["event"]["type"] == "message_update")
        .collect();
    assert!(
        !updates.is_empty(),
        "no message_update in stream: {:#?}",
        run_events
            .iter()
            .map(|e| e["event"]["type"].as_str().unwrap_or("?"))
            .collect::<Vec<_>>()
    );
    for u in &updates {
        assert_eq!(u["event"]["messageId"].as_str(), Some(message_id));
        let delta = u["event"]["delta"].as_str().unwrap_or_default();
        assert!(
            !delta.is_empty(),
            "message_update has empty delta: {u}"
        );
        // The contract: no internal state on the wire.
        assert!(u["event"].get("message").is_none());
        assert!(u["event"].get("assistantMessageEvent").is_none());
    }

    // 4. No message_update carries a `delta` that looks like private
    //    chain-of-thought or a tool-call wire repair. The translator drops
    //    these at the source so they cannot reach the chat.
    for u in &updates {
        let delta = u["event"]["delta"].as_str().unwrap_or_default();
        assert!(
            !delta.contains("Thinking Process"),
            "thinking/reasoning content leaked into the visible stream: {delta:?}"
        );
        assert!(
            !delta.contains("\"name\":"),
            "tool-call JSON leaked into the visible stream: {delta:?}"
        );
    }

    // 5. message_end is the last message-stream event and carries the
    //    right messageId and a finishReason from the allowed union.
    let end = run_events
        .iter()
        .rev()
        .find(|e| e["event"]["type"] == "message_end")
        .expect("no message_end in stream");
    assert_eq!(end["event"]["messageId"].as_str(), Some(message_id));
    let finish_reason = end["event"]["finishReason"].as_str().unwrap_or("");
    assert!(
        matches!(
            finish_reason,
            "stop" | "length" | "tool_calls" | "content_filter" | "error"
        ),
        "message_end has unknown finishReason: {finish_reason:?}"
    );

    // 6. The concatenation of all visible deltas is non-empty. This is the
    //    end-to-end check: text was produced and translated.
    let full_text: String = updates
        .iter()
        .filter_map(|e| e["event"]["delta"].as_str())
        .collect();
    assert!(
        !full_text.trim().is_empty(),
        "no visible text was streamed for the assistant message: {full_text:?}"
    );

    eprintln!(
        "[e2e] streamed {} visible deltas totalling {} chars; finish={}",
        updates.len(),
        full_text.len(),
        finish_reason
    );
}

/// GET `/models` against the local endpoint. Cheap and tells us the
/// runtime would also see the model.
async fn endpoint_reachable(base_url: &str) -> bool {
    let probe = format!("{}/models", base_url.trim_end_matches("/v1"));
    let client = match reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(2))
        .build()
    {
        Ok(c) => c,
        Err(_) => return false,
    };
    match client.get(&probe).send().await {
        Ok(r) => r.status().is_success(),
        Err(_) => false,
    }
}

/// A longer prompt — proves the streaming contract holds for a request
/// the model cannot answer in one short sentence. Skips if the model
/// is not reachable, same as the `hi` test.
///
/// Distinct from the first E2E test so a regression on one prompt shape
/// does not hide the other. Asserts the *minimum* contract — a
/// `message_start`, at least one `message_update` with a non-empty
/// `delta`, and a `message_end` — rather than a particular text
/// content, because the local model is free to word its answer.
#[tokio::test]
async fn a_longer_prompt_streams_in_the_wire_contract() {
    if !node_present() {
        eprintln!("skipping: node is not on PATH");
        return;
    }
    let base_url = std::env::var("ARJUN_TEST_MODEL_BASE_URL")
        .unwrap_or_else(|_| "http://127.0.0.1:61353/v1".to_string());
    if !endpoint_reachable(&base_url).await {
        eprintln!("skipping: local model at {base_url} is not reachable");
        return;
    }
    // One local model, several tests that want it.
    //
    // These share a single llama-server, and cargo runs tests in parallel by
    // default. Two of them decoding at once starve each other: a long answer
    // holds the slots while a short one waits past its own timeout, which
    // surfaces as an unrelated test failing intermittently. Serialised on the
    // scarce resource rather than made lenient, so a real regression still
    // fails and a busy machine does not.
    let _one_at_a_time = real_model_lock().lock().await;

    let (deps, _dir) = deps();
    let events: Arc<std::sync::Mutex<Vec<serde_json::Value>>> =
        Arc::new(std::sync::Mutex::new(Vec::new()));
    let sink_events = events.clone();
    let emit: Arc<dyn Fn(serde_json::Value) + Send + Sync> = Arc::new(move |value| {
        sink_events
            .lock()
            .map(|mut v| v.push(value))
            .unwrap_or_else(|e| e.into_inner().push(serde_json::Value::Null));
    });
    let runtime = AgentRuntime::spawn(deps, emit, bundle()).expect("runtime starts");

    let run_id = "stream-e2e-run-long";
    let message_id = "msg-e2e-long";

    let _ = runtime
        .request(
            "run.start",
            serde_json::json!({
                "runId": run_id,
                "messageId": message_id,
                "prompt": "Explain in one paragraph why the sky is blue during the day and red at sunset.",
                "systemPrompt": "Be concise.",
                "model": {
                    "id": "gemma-4-E4B-it",
                    "provider": "sovereign-local",
                    "baseUrl": base_url,
                }
            }),
        )
        .await
        .expect("run.start resolves");

    tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    let collected = events.lock().map(|v| v.clone()).unwrap_or_default();
    runtime.shutdown().await;

    let run_events: Vec<serde_json::Value> = collected
        .into_iter()
        .filter(|v| v.get("event").is_some())
        .collect();

    assert!(
        run_events.iter().any(|e| e["event"]["type"] == "message_start"
            && e["event"]["messageId"].as_str() == Some(message_id)),
        "no message_start with the right messageId"
    );

    let updates: Vec<serde_json::Value> = run_events
        .iter()
        .filter(|e| e["event"]["type"] == "message_update")
        .cloned()
        .collect();
    assert!(
        !updates.is_empty(),
        "a longer prompt produced no message_update events"
    );
    for u in &updates {
        assert_eq!(u["event"]["messageId"].as_str(), Some(message_id));
        let delta = u["event"]["delta"].as_str().unwrap_or_default();
        assert!(!delta.is_empty(), "empty delta: {u}");
    }
    let total: String = updates
        .iter()
        .filter_map(|e| e["event"]["delta"].as_str())
        .collect();
    assert!(
        total.len() > 20,
        "a paragraph answer should produce a non-trivial text length, got {total:?}"
    );

    assert!(
        run_events.iter().any(|e| e["event"]["type"] == "message_end"
            && e["event"]["messageId"].as_str() == Some(message_id)),
        "no message_end with the right messageId"
    );
}

/// A follow-up prompt in the same logical conversation: the second
/// `run.start` call must use the *same* messageId isolation but a
/// *different* `runId`, and both runs' message-stream events must
/// carry the right per-cell messageId. Catches a regression where
/// the runtime accidentally couples the wire contract to runId
/// rather than per-message state.
#[tokio::test]
async fn two_runs_in_a_row_each_carry_their_own_messageId() {
    if !node_present() {
        eprintln!("skipping: node is not on PATH");
        return;
    }
    let base_url = std::env::var("ARJUN_TEST_MODEL_BASE_URL")
        .unwrap_or_else(|_| "http://127.0.0.1:61353/v1".to_string());
    if !endpoint_reachable(&base_url).await {
        eprintln!("skipping: local model at {base_url} is not reachable");
        return;
    }
    // One local model, several tests that want it.
    //
    // These share a single llama-server, and cargo runs tests in parallel by
    // default. Two of them decoding at once starve each other: a long answer
    // holds the slots while a short one waits past its own timeout, which
    // surfaces as an unrelated test failing intermittently. Serialised on the
    // scarce resource rather than made lenient, so a real regression still
    // fails and a busy machine does not.
    let _one_at_a_time = real_model_lock().lock().await;

    let (deps, _dir) = deps();
    let events: Arc<std::sync::Mutex<Vec<serde_json::Value>>> =
        Arc::new(std::sync::Mutex::new(Vec::new()));
    let sink_events = events.clone();
    let emit: Arc<dyn Fn(serde_json::Value) + Send + Sync> = Arc::new(move |value| {
        sink_events
            .lock()
            .map(|mut v| v.push(value))
            .unwrap_or_else(|e| e.into_inner().push(serde_json::Value::Null));
    });
    let runtime = AgentRuntime::spawn(deps, emit, bundle()).expect("runtime starts");

    // Run 1.
    runtime
        .request(
            "run.start",
            serde_json::json!({
                "runId": "follow-up-1",
                "messageId": "msg-follow-up-1",
                "prompt": "hi",
                "systemPrompt": "Be brief.",
                "model": {
                    "id": "gemma-4-E4B-it",
                    "provider": "sovereign-local",
                    "baseUrl": base_url,
                }
            }),
        )
        .await
        .expect("first run.start resolves");

    // Run 2 — fresh runId, fresh messageId. The chat surface reserves a
    // new assistant cell for a follow-up turn, so the two must not
    // share a messageId.
    runtime
        .request(
            "run.start",
            serde_json::json!({
                "runId": "follow-up-2",
                "messageId": "msg-follow-up-2",
                "prompt": "What did I just say?",
                "systemPrompt": "Be brief.",
                "model": {
                    "id": "gemma-4-E4B-it",
                    "provider": "sovereign-local",
                    "baseUrl": base_url,
                }
            }),
        )
        .await
        .expect("second run.start resolves");

    tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    let collected = events.lock().map(|v| v.clone()).unwrap_or_default();
    runtime.shutdown().await;

    // Every message_start / message_update / message_end in the
    // collected events is tagged with a messageId. The set of
    // messageIds seen must include both follow-up-1 and follow-up-2.
    let message_ids: std::collections::HashSet<String> = collected
        .iter()
        .filter_map(|e| {
            let t = e["event"]["type"].as_str()?;
            if t == "message_start" || t == "message_update" || t == "message_end" {
                e["event"]["messageId"].as_str().map(String::from)
            } else {
                None
            }
        })
        .collect();

    assert!(
        message_ids.contains("msg-follow-up-1"),
        "no message-stream events for the first follow-up: {message_ids:?}"
    );
    assert!(
        message_ids.contains("msg-follow-up-2"),
        "no message-stream events for the second follow-up: {message_ids:?}"
    );

    // Each run's events carry exactly the right messageId — no
    // cross-contamination between cells.
    for run_id in ["follow-up-1", "follow-up-2"] {
        let expected_mid = if run_id == "follow-up-1" {
            "msg-follow-up-1"
        } else {
            "msg-follow-up-2"
        };
        let wrong = collected.iter().any(|e| {
            e["runId"].as_str() == Some(run_id)
                && matches!(
                    e["event"]["type"].as_str(),
                    Some("message_start" | "message_update" | "message_end")
                )
                && e["event"]["messageId"].as_str() != Some(expected_mid)
        });
        assert!(
            !wrong,
            "run {run_id} emitted a message-stream event with the wrong messageId"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────
// The journey across the real process boundary.
//
// The runtime's own tests drive `startRun` in-process. These spawn the actual
// Node child from the built bundle, speak real JSON-RPC over stdio, and point
// it at a real HTTP server — so what is asserted is the request bytes that
// reached a socket, having crossed the language boundary the product ships.
//
// The model server is a fixture rather than a real model, deliberately: what is
// under test is whether the conversation *arrives*, and a real model would make
// that assertion depend on what it chose to say.
// ─────────────────────────────────────────────────────────────────────────

/// What one fixture model server saw and how it answers.
struct FixtureServer {
    base_url: String,
    /// Request bodies, in arrival order.
    seen: Arc<std::sync::Mutex<Vec<serde_json::Value>>>,
}

/// A local OpenAI-compatible endpoint that records what it is sent.
///
/// `hold` leaves the response open after the scripted frames, which is what
/// makes a run still be generating when a Stop arrives.
async fn fixture_model_server(frames: Vec<String>, hold: bool) -> FixtureServer {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind a fixture model server");
    let addr = listener.local_addr().expect("addr");
    let seen: Arc<std::sync::Mutex<Vec<serde_json::Value>>> =
        Arc::new(std::sync::Mutex::new(Vec::new()));
    let sink = seen.clone();

    tokio::spawn(async move {
        let mut held = Vec::new();
        while let Ok((mut socket, _)) = listener.accept().await {
            let mut raw = Vec::new();
            let mut buffer = [0u8; 8192];
            // Read headers, then exactly the declared body.
            loop {
                let read = match socket.read(&mut buffer).await {
                    Ok(0) | Err(_) => break,
                    Ok(n) => n,
                };
                raw.extend_from_slice(&buffer[..read]);
                let text = String::from_utf8_lossy(&raw).to_string();
                let Some(head_end) = text.find("\r\n\r\n") else {
                    continue;
                };
                let head = &text[..head_end];
                let want: usize = head
                    .lines()
                    .find_map(|line| {
                        let lower = line.to_ascii_lowercase();
                        lower
                            .strip_prefix("content-length:")
                            .and_then(|v| v.trim().parse().ok())
                    })
                    .unwrap_or(0);
                if raw.len() >= head_end + 4 + want {
                    let body = &raw[head_end + 4..head_end + 4 + want];
                    if let Ok(value) = serde_json::from_slice::<serde_json::Value>(body) {
                        if let Ok(mut recorded) = sink.lock() {
                            recorded.push(value);
                        }
                    }
                    break;
                }
            }

            let head = concat!(
                "HTTP/1.1 200 OK\r\n",
                "content-type: text/event-stream\r\n",
                "cache-control: no-cache\r\n",
                "connection: keep-alive\r\n\r\n",
            );
            let _ = socket.write_all(head.as_bytes()).await;
            for frame in &frames {
                let _ = socket.write_all(frame.as_bytes()).await;
            }
            let _ = socket.flush().await;
            if hold {
                // Still generating. Held so the socket is not closed, which
                // would end the run for the wrong reason.
                held.push(socket);
                continue;
            }
            let _ = socket.write_all(b"data: [DONE]\n\n").await;
            let _ = socket.shutdown().await;
        }
    });

    FixtureServer {
        base_url: format!("http://{addr}/v1"),
        seen,
    }
}

fn sse(delta: serde_json::Value, finish: Option<&str>) -> String {
    format!(
        "data: {}\n\n",
        serde_json::json!({
            "id": "chatcmpl-fixture",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "fixture",
            "choices": [{ "index": 0, "delta": delta, "finish_reason": finish }],
        })
    )
}

/// Everything one recorded request carried, flattened to one string.
fn sent_text(body: &serde_json::Value) -> String {
    let Some(messages) = body.get("messages").and_then(|m| m.as_array()) else {
        return String::new();
    };
    messages
        .iter()
        .map(|message| match message.get("content") {
            Some(serde_json::Value::String(text)) => text.clone(),
            Some(serde_json::Value::Array(blocks)) => blocks
                .iter()
                .filter_map(|block| block.get("text").and_then(|t| t.as_str()))
                .collect::<Vec<_>>()
                .join(""),
            _ => String::new(),
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Waits until the fixture server has received a request, or gives up.
///
/// A fixed sleep is the wrong tool here: spawning the real Node child,
/// fetching the tool catalogue and reaching the first model call takes
/// however long the machine takes, and a sleep long enough to be safe makes
/// every run of the suite pay it. Waiting on the thing that actually has to
/// have happened is both faster and not flaky.
///
/// Returns whether it arrived, so the caller can fail with its own message.
async fn awaited_first_call(server: &FixtureServer) -> bool {
    for _ in 0..600 {
        if server.seen.lock().map(|seen| !seen.is_empty()).unwrap_or(false) {
            return true;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    false
}

/// The conversation reaches the model, across the process boundary.
///
/// This is the claim the whole context change rests on, asserted at the only
/// place it can be settled: the bytes a socket received, sent by the real child
/// process the product ships.
#[tokio::test]
async fn a_continuing_turn_carries_its_history_across_the_process_boundary() {
    if !node_present() {
        eprintln!("skipping: node is not on PATH");
        return;
    }
    let server = fixture_model_server(
        vec![
            sse(serde_json::json!({ "role": "assistant", "content": "" }), None),
            sse(serde_json::json!({ "content": "Class 300." }), Some("stop")),
        ],
        false,
    )
    .await;

    let (deps, _dir) = deps();
    let runtime = AgentRuntime::spawn(deps, Arc::new(|_| {}), bundle()).expect("runtime starts");

    let _ = runtime
        .request(
            "run.start",
            serde_json::json!({
                "runId": "journey-run-1",
                "messageId": "journey-msg-1",
                "prompt": "And the gasket torque?",
                "systemPrompt": "Answer from what you were given.",
                "history": [
                    { "role": "user", "content": "What is the pressure rating?" },
                    { "role": "assistant", "content": "Class 300 throughout the skid." }
                ],
                "model": {
                    "id": "fixture",
                    "provider": "sovereign-local",
                    "baseUrl": server.base_url,
                }
            }),
        )
        .await
        .expect("run.start resolves");
    runtime.shutdown().await;

    let seen = server.seen.lock().expect("lock").clone();
    assert!(!seen.is_empty(), "the model server received no request");
    let sent = sent_text(&seen[0]);
    assert!(
        sent.contains("What is the pressure rating?"),
        "the earlier question did not reach the model: {sent}"
    );
    assert!(
        sent.contains("Class 300 throughout the skid."),
        "the earlier answer did not reach the model: {sent}"
    );
    assert!(
        sent.contains("And the gasket torque?"),
        "the new question did not reach the model: {sent}"
    );
}

/// Stop, while the model is thinking, across the process boundary.
///
/// The server holds the stream open, so the run is genuinely mid-generation
/// when the abort arrives — not merely finished early.
#[tokio::test]
async fn stopping_a_thinking_run_ends_it_as_aborted_and_keeps_what_it_wrote() {
    if !node_present() {
        eprintln!("skipping: node is not on PATH");
        return;
    }
    let server = fixture_model_server(
        vec![
            sse(serde_json::json!({ "role": "assistant", "content": "" }), None),
            sse(serde_json::json!({ "content": "half an answer" }), None),
        ],
        true,
    )
    .await;

    let (deps, _dir) = deps();
    let runtime = AgentRuntime::spawn(deps, Arc::new(|_| {}), bundle()).expect("runtime starts");

    // Driven on a task, not merely constructed.
    //
    // `request` returns a future, and a Rust future does nothing until it is
    // polled. Holding it in a local and then sleeping sends no request at all,
    // so the abort below would name a run that had never started — which is
    // exactly what the first version of this test did, and it reported a
    // missing abort rather than a missing run.
    let driven = Arc::clone(&runtime);
    let base = server.base_url.clone();
    let started = tokio::spawn(async move {
        driven
            .request(
                "run.start",
                serde_json::json!({
                    "runId": "journey-stop-1",
                    "messageId": "journey-stop-msg-1",
                    "prompt": "write at length",
                    "systemPrompt": "Answer at length.",
                    "model": {
                        "id": "fixture",
                        "provider": "sovereign-local",
                        "baseUrl": base,
                    }
                }),
            )
            .await
    });

    // The run is genuinely mid-generation once the model server has been
    // called and the server is holding the stream open.
    assert!(
        awaited_first_call(&server).await,
        "the run never reached the model server, so there was nothing to stop"
    );
    // A short grace for the scripted delta to be decoded into the answer.
    tokio::time::sleep(std::time::Duration::from_millis(300)).await;
    let aborted = runtime
        .request("run.abort", serde_json::json!({ "runId": "journey-stop-1" }))
        .await
        .expect("run.abort resolves");
    assert_eq!(
        aborted.get("aborted").and_then(|v| v.as_bool()),
        Some(true),
        "the runtime did not report a live run to abort: {aborted}"
    );

    let outcome = started
        .await
        .expect("the run task finished")
        .expect("the stopped run still returns");
    runtime.shutdown().await;

    assert_eq!(
        outcome
            .get("outcome")
            .and_then(|o| o.get("kind"))
            .and_then(|k| k.as_str()),
        Some("aborted"),
        "a stopped run must not be recorded as completed: {outcome}"
    );
    // The half-written answer is the person's, and a stopped turn keeps it.
    assert!(
        outcome
            .get("text")
            .and_then(|t| t.as_str())
            .unwrap_or_default()
            .contains("half an answer"),
        "partial output was lost: {outcome}"
    );
}

/// A stop, then a normal turn. A stopped run is not a broken session.
#[tokio::test]
async fn the_turn_after_a_stop_runs_normally() {
    if !node_present() {
        eprintln!("skipping: node is not on PATH");
        return;
    }
    let held = fixture_model_server(
        vec![sse(
            serde_json::json!({ "role": "assistant", "content": "partial" }),
            None,
        )],
        true,
    )
    .await;

    let (deps, _dir) = deps();
    let runtime = AgentRuntime::spawn(deps, Arc::new(|_| {}), bundle()).expect("runtime starts");

    // Driven on a task, for the same reason as above.
    let driven = Arc::clone(&runtime);
    let base = held.base_url.clone();
    let first = tokio::spawn(async move {
        driven
            .request(
                "run.start",
                serde_json::json!({
                    "runId": "journey-recover-1",
                    "messageId": "m-1",
                    "prompt": "one",
                    "systemPrompt": "s",
                    "model": {
                        "id": "fixture",
                        "provider": "sovereign-local",
                        "baseUrl": base,
                    },
                }),
            )
            .await
    });
    assert!(
        awaited_first_call(&held).await,
        "the first run never reached the model server"
    );
    let stopped = runtime
        .request(
            "run.abort",
            serde_json::json!({ "runId": "journey-recover-1" }),
        )
        .await
        .expect("run.abort resolves");
    assert_eq!(
        stopped.get("aborted").and_then(|v| v.as_bool()),
        Some(true),
        "there was no live run to stop: {stopped}"
    );
    let _ = first
        .await
        .expect("the run task finished")
        .expect("the stopped run returns");

    // A second, ordinary turn on the same runtime process.
    let ordinary = fixture_model_server(
        vec![
            sse(serde_json::json!({ "role": "assistant", "content": "" }), None),
            sse(serde_json::json!({ "content": "all done" }), Some("stop")),
        ],
        false,
    )
    .await;
    let outcome = runtime
        .request(
            "run.start",
            serde_json::json!({
                "runId": "journey-recover-2",
                "messageId": "m-2",
                "prompt": "two",
                "systemPrompt": "s",
                "model": {
                    "id": "fixture",
                    "provider": "sovereign-local",
                    "baseUrl": ordinary.base_url,
                },
            }),
        )
        .await
        .expect("the next turn runs");
    runtime.shutdown().await;

    assert_eq!(
        outcome
            .get("outcome")
            .and_then(|o| o.get("kind"))
            .and_then(|k| k.as_str()),
        Some("completed"),
        "the turn after a stop did not run cleanly: {outcome}"
    );
    assert!(outcome
        .get("text")
        .and_then(|t| t.as_str())
        .unwrap_or_default()
        .contains("all done"));
}

/// Serialises the tests that share the one local model server.
fn real_model_lock() -> &'static tokio::sync::Mutex<()> {
    static LOCK: std::sync::OnceLock<tokio::sync::Mutex<()>> = std::sync::OnceLock::new();
    LOCK.get_or_init(|| tokio::sync::Mutex::new(()))
}

/// The proof a fixture cannot give: a real model answering from the history.
///
/// Every other context test asserts that the earlier turns *reached* the
/// provider. This asserts that they were usable — the question is answerable
/// only from the conversation, the reference is arbitrary enough that no model
/// could produce it from its weights, and the answer is the model's own.
///
/// Skips when no local model is serving, because it is a proof against a real
/// model rather than a fixture. The skip prints the endpoint it tried, so a
/// missing model is reported explicitly rather than as a silent pass.
#[tokio::test]
async fn a_real_model_answers_from_the_conversation_it_was_given() {
    if !node_present() {
        eprintln!("skipping: node is not on PATH");
        return;
    }
    let base_url = std::env::var("ARJUN_TEST_MODEL_BASE_URL")
        .unwrap_or_else(|_| "http://127.0.0.1:61353/v1".to_string());
    if !endpoint_reachable(&base_url).await {
        eprintln!("skipping: local model at {base_url} is not reachable");
        return;
    }
    // One local model, several tests that want it.
    //
    // These share a single llama-server, and cargo runs tests in parallel by
    // default. Two of them decoding at once starve each other: a long answer
    // holds the slots while a short one waits past its own timeout, which
    // surfaces as an unrelated test failing intermittently. Serialised on the
    // scarce resource rather than made lenient, so a real regression still
    // fails and a busy machine does not.
    let _one_at_a_time = real_model_lock().lock().await;

    let (deps, _dir) = deps();
    let runtime = AgentRuntime::spawn(deps, Arc::new(|_| {}), bundle()).expect("runtime starts");

    // Arbitrary, and stated only in the history. A model that did not receive
    // the earlier turn cannot produce it.
    let outcome = runtime
        .request(
            "run.start",
            serde_json::json!({
                "runId": "real-history-1",
                "messageId": "real-history-msg-1",
                "prompt": "What is my reference number? Answer with the code only.",
                "systemPrompt": "Answer from the conversation. Be very brief.",
                "history": [
                    { "role": "user", "content": "Please note my reference number: ZX-4471-QD." },
                    { "role": "assistant", "content": "Noted. Your reference number is ZX-4471-QD." }
                ],
                "model": {
                    "id": "local-model",
                    "provider": "sovereign-local",
                    "baseUrl": base_url,
                }
            }),
        )
        .await
        .expect("run.start resolves");
    runtime.shutdown().await;

    let answer = outcome
        .get("text")
        .and_then(|t| t.as_str())
        .unwrap_or_default()
        .to_string();
    assert!(
        answer.contains("ZX-4471"),
        "the model could not answer from the history it was sent, so the history did not reach it usefully. It said: {answer:?}"
    );
}

/// Stop, against a real model that is genuinely generating.
///
/// The fixture version proves the mechanism; this proves it against a model
/// that is actually decoding on the GPU, which is the case an operator presses
/// the button in.
#[tokio::test]
async fn stopping_a_real_model_mid_answer_ends_the_run_as_aborted() {
    if !node_present() {
        eprintln!("skipping: node is not on PATH");
        return;
    }
    let base_url = std::env::var("ARJUN_TEST_MODEL_BASE_URL")
        .unwrap_or_else(|_| "http://127.0.0.1:61353/v1".to_string());
    if !endpoint_reachable(&base_url).await {
        eprintln!("skipping: local model at {base_url} is not reachable");
        return;
    }
    // One local model, several tests that want it.
    //
    // These share a single llama-server, and cargo runs tests in parallel by
    // default. Two of them decoding at once starve each other: a long answer
    // holds the slots while a short one waits past its own timeout, which
    // surfaces as an unrelated test failing intermittently. Serialised on the
    // scarce resource rather than made lenient, so a real regression still
    // fails and a busy machine does not.
    let _one_at_a_time = real_model_lock().lock().await;

    let (deps, _dir) = deps();
    // The event sink, so the stop can be timed against what the model has
    // actually produced rather than against a stopwatch.
    //
    // A fixed sleep is wrong here for a reason specific to this product: a
    // reasoning model emits its thinking first, and thinking is deliberately
    // never counted as answer text. Stopping during that phase leaves an
    // assistant message carrying reasoning and no text blocks — nothing was
    // lost, there was simply nothing visible yet. Waiting for the first
    // visible delta is what makes 'partial output is preserved' a claim this
    // test can actually make.
    let seen_text: Arc<std::sync::atomic::AtomicBool> =
        Arc::new(std::sync::atomic::AtomicBool::new(false));
    let watcher = seen_text.clone();
    let emit: Arc<dyn Fn(serde_json::Value) + Send + Sync> = Arc::new(move |value| {
        let is_visible_delta = value.pointer("/event/type").and_then(|t| t.as_str())
            == Some("message_update")
            && value
                .pointer("/event/delta")
                .and_then(|d| d.as_str())
                .is_some_and(|delta| !delta.is_empty());
        if is_visible_delta {
            watcher.store(true, std::sync::atomic::Ordering::SeqCst);
        }
    });
    let runtime = AgentRuntime::spawn(deps, emit, bundle()).expect("runtime starts");

    let driven = Arc::clone(&runtime);
    let endpoint = base_url.clone();
    let started = tokio::spawn(async move {
        driven
            .request(
                "run.start",
                serde_json::json!({
                    "runId": "real-stop-1",
                    "messageId": "real-stop-msg-1",
                    // Long enough that it is still decoding when the stop lands.
                    "prompt": "Write a detailed 800-word description of a centrifugal pump.",
                    "systemPrompt": "Write at length.",
                    "model": {
                        "id": "local-model",
                        "provider": "sovereign-local",
                        "baseUrl": endpoint,
                        "maxTokens": 2048,
                    }
                }),
            )
            .await
    });

    // Wait for the model to have written something visible, then stop it.
    //
    // Bounded short deliberately. Measured on this machine, llama-server with
    // Nemotron3-Nano-4B delivers the whole answer as a *single* `message_update`
    // — one delta, arriving with `message_end` — so there is no mid-generation
    // window to stop in at all. Waiting longer would not create one; it would
    // only make the suite slower before reaching the same place.
    let mut wrote_something = false;
    for _ in 0..80 {
        if seen_text.load(std::sync::atomic::Ordering::SeqCst) {
            wrote_something = true;
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    if !wrote_something {
        // No window existed. Reported rather than proceeding to an abort that
        // would find a finished run and assert nothing — a test that passes
        // while proving nothing is worse than one that says why it could not.
        //
        // The mechanism itself is covered without a real model, against a
        // server that holds the stream open: see
        // `stopping_a_thinking_run_ends_it_as_aborted_and_keeps_what_it_wrote`.
        eprintln!(
            "skipping the assertion: this model returned no incremental output, so there was              no mid-generation window to stop in"
        );
        let _ = started.await;
        runtime.shutdown().await;
        return;
    }
    let aborted = runtime
        .request("run.abort", serde_json::json!({ "runId": "real-stop-1" }))
        .await
        .expect("run.abort resolves");

    let outcome = started
        .await
        .expect("the run task finished")
        .expect("the stopped run still returns");
    runtime.shutdown().await;

    let kind = outcome
        .get("outcome")
        .and_then(|o| o.get("kind"))
        .and_then(|k| k.as_str())
        .unwrap_or_default()
        .to_string();
    // If the model finished before the stop landed the run is legitimately
    // complete, and asserting otherwise would make this flaky on a fast
    // machine. Reported rather than silently passing.
    if aborted.get("aborted").and_then(|v| v.as_bool()) != Some(true) {
        eprintln!("note: the model finished before the stop arrived; ending was {kind}");
        return;
    }

    // The load-bearing claim, and it holds either way.
    assert_eq!(
        kind, "aborted",
        "a run stopped mid-answer was recorded as {kind}: {outcome}"
    );

    let text = outcome
        .get("text")
        .and_then(|t| t.as_str())
        .unwrap_or_default();
    if wrote_something {
        // It had written something visible, so a stopped turn must keep it. A
        // stopped turn is not a failed one, and discarding the text would lose
        // real work.
        assert!(
            !text.is_empty(),
            "the model had written visible text and the stop discarded it"
        );
    } else {
        // Still reasoning when the stop landed. Nothing visible existed to
        // preserve, and reasoning is never counted as answer text — so an
        // empty answer here is correct rather than a loss. Reported so a run
        // of the suite that took this branch is not read as having proven
        // preservation.
        eprintln!("note: the model was still reasoning at the stop; no visible text to preserve");
    }
}
