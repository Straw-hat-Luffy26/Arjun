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

#[path = "agent_runtime_durable_journey.rs"]
mod durable_journey;

fn execution(deps: &Arc<RuntimeDeps>, run_id: &str, message_id: &str, prompt: &str) -> serde_json::Value {
    use sarathi_lib::agent_runtime::{events::{EventDraft,TaskEventType},resume::{CheckpointSeed,policy_hash}};
    let signed_in=deps.session.read().unwrap().clone().unwrap();
    let class=sarathi_lib::policy::Classification::Internal;
    deps.events.record(EventDraft::new(run_id,TaskEventType::RunCreated,&signed_in.user.id).with(serde_json::json!({"promptShown":prompt}))).unwrap();
    deps.events.record(EventDraft::new(run_id,TaskEventType::RunClassified,&signed_in.user.id).with(serde_json::json!({"classification":class.label()}))).unwrap();
    deps.plans.lock().unwrap().entry(run_id.into()).or_insert_with(|| sarathi_lib::orchestrator::plan::PlanRun::new(run_id,vec!["do the work".into()],sarathi_lib::orchestrator::plan::Budget::standard(sarathi_lib::orchestrator::tools::ToolName::ALL.to_vec())));
    let claim=deps.events.claim_run(run_id,"fixture-worker",chrono::Duration::minutes(5),chrono::Utc::now()).unwrap().unwrap();
    let seed=CheckpointSeed {attempt_id:"attempt-1".into(),lease:claim,objective:prompt.into(),conversation_id:format!("conversation-{run_id}"),message_id:message_id.into(),deadline_ms:(chrono::Utc::now()+chrono::Duration::minutes(5)).timestamp_millis(),plan_hash:"fixture-plan".into(),workspace_hash:"fixture-workspace".into(),model_context: None, model_id:"fixture-model".into(),policy_hash:policy_hash(&signed_in,Some(class),&format!("{:?}",sarathi_lib::sovereignty::global_broker().mode()))};
    let identity=serde_json::json!({"protocolVersion":1,"attemptId":seed.attempt_id,"fenceToken":seed.lease.fence_token});
    deps.checkpoints.lock().unwrap().insert(run_id.into(),seed);
    identity
}

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
    (deps_in(&dir), dir)
}

fn deps_in(dir: &tempfile::TempDir) -> Arc<RuntimeDeps> {
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
        })
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
    assert_eq!(health["contextProtocolVersion"], 1);
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
                "execution": {"protocolVersion":1,"attemptId":"attempt-1","fenceToken":1},
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
    let runtime = AgentRuntime::spawn(deps.clone(), emit, bundle()).expect("runtime starts");

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
                "execution": execution(&deps, run_id, message_id, "hi"),
                "systemPrompt": "Answer in one short sentence.",
                "model": {
                    "id": "gemma-4-E4B-it",
                    "provider": "sovereign-local",
                    "baseUrl": base_url,
                    "contextWindow": 8192,
                    "maxTokens": 256,
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
    let runtime = AgentRuntime::spawn(deps.clone(), emit, bundle()).expect("runtime starts");

    let run_id = "stream-e2e-run-long";
    let message_id = "msg-e2e-long";

    let _ = runtime
        .request(
            "run.start",
            serde_json::json!({
                "runId": run_id,
                "messageId": message_id,
                "prompt": "Explain in one paragraph why the sky is blue during the day and red at sunset.",
                "execution": execution(&deps, run_id, message_id, "Explain in one paragraph why the sky is blue during the day and red at sunset."),
                "systemPrompt": "Be concise.",
                "model": {
                    "id": "gemma-4-E4B-it",
                    "provider": "sovereign-local",
                    "baseUrl": base_url,
                    "contextWindow": 8192,
                    "maxTokens": 256,
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
    let runtime = AgentRuntime::spawn(deps.clone(), emit, bundle()).expect("runtime starts");

    // Run 1.
    runtime
        .request(
            "run.start",
            serde_json::json!({
                "runId": "follow-up-1",
                "messageId": "msg-follow-up-1",
                "prompt": "hi",
                "systemPrompt": "Be brief.",
                "execution": execution(&deps, "follow-up-1", "msg-follow-up-1", "hi"),
                "model": {
                    "id": "gemma-4-E4B-it",
                    "provider": "sovereign-local",
                    "baseUrl": base_url,
                    "contextWindow": 8192,
                    "maxTokens": 256,
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
                "execution": execution(&deps, "follow-up-2", "msg-follow-up-2", "What did I just say?"),
                "model": {
                    "id": "gemma-4-E4B-it",
                    "provider": "sovereign-local",
                    "baseUrl": base_url,
                    "contextWindow": 8192,
                    "maxTokens": 256,
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
