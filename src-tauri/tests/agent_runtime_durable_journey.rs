//! Real Node worker + SQLite + gateway + executor. Only model responses are scripted.
//! This does not exercise the Tauri UI, model router, or a native inference model.
use super::*;
use axum::{routing::post, Json, Router};
use chrono::{Duration, Utc};
use sarathi_lib::agent_runtime::{
    completion::{self, CompletionInputs}, context_api::CoreCheckpoint,
    events::{ApprovalStatus, EventDraft, RunState, TaskEventType},
    outcome::RunOutcome, resume::CheckpointSeed,
};
use sarathi_lib::orchestrator::plan::{Budget, PlanRun};
use sarathi_lib::orchestrator::tools::ToolName;
use serde_json::{json, Value};
use std::sync::{atomic::{AtomicUsize, Ordering}, Mutex};

const RUN: &str = "run-1";
const PROMPT: &str = "Read source-0.txt through source-7.txt. Preserve exact ID PUMP-A17. Write final.txt containing PUMP-A17 verified, after approval.";

struct ServerGuard(tokio::task::JoinHandle<()>);
impl Drop for ServerGuard { fn drop(&mut self) { self.0.abort(); } }

fn response(delta: Value, reason: &str) -> String {
    let chunk = |delta: Value, finish: Value| format!("data: {}\n\n", json!({
        "id":"fixture", "object":"chat.completion.chunk", "created":0, "model":"fixture-model",
        "choices":[{"index":0,"delta":delta,"finish_reason":finish}]
    }));
    format!("{}{}{}data: [DONE]\n\n", chunk(json!({"role":"assistant"}), Value::Null),
        chunk(delta, Value::Null), chunk(json!({}), json!(reason)))
}

fn plan() -> PlanRun {
    PlanRun::new(RUN, vec!["read sources and write verified artifact".into()],
        Budget::standard(vec![ToolName::ReadScopedFile, ToolName::WriteScopedFile]))
}

fn input(base_url: &str, identity: Value) -> Value {
    json!({"runId":RUN,"messageId":"journey-message","prompt":PROMPT,
        "systemPrompt":"Use the authorized workspace tools. Preserve PUMP-A17 and obtain approval before writing.",
        "execution":identity,"model":{"id":"fixture-model","provider":"sovereign-local",
            "baseUrl":base_url,"contextWindow":4096,"maxTokens":256}})
}

fn restore(dir: &tempfile::TempDir, previous: &CheckpointSeed, attempt: &str) -> (Arc<RuntimeDeps>, CheckpointSeed) {
    let deps = deps_in(dir);
    let saved = deps.events.load_context(RUN).unwrap().unwrap();
    let core = CoreCheckpoint::from_stored(&saved).unwrap();
    assert_eq!(core.objective, PROMPT);
    assert_eq!(core.message_id, "journey-message");
    let mut restored_plan = plan();
    restored_plan.restore_progress(&core.plan).unwrap();
    deps.plans.lock().unwrap().insert(RUN.into(), restored_plan);
    deps.passages.lock().unwrap().insert(RUN.into(), core.passages);
    deps.calculations.lock().unwrap().insert(RUN.into(), core.calculations);
    deps.produced.lock().unwrap().insert(RUN.into(), core.produced);
    deps.calls.lock().unwrap().insert(RUN.into(), core.calls);
    let mut seed = previous.clone();
    seed.attempt_id = attempt.into();
    seed.lease = deps.events.claim_run(RUN, attempt, Duration::minutes(2), Utc::now()).unwrap().unwrap();
    assert!(seed.lease.fence_token > previous.lease.fence_token);
    deps.checkpoints.lock().unwrap().insert(RUN.into(), seed.clone());
    deps.events.record_fenced(EventDraft::new(RUN, TaskEventType::RunResumed, "priya"), &seed.lease).unwrap();
    (deps, seed)
}

fn identity(seed: &CheckpointSeed) -> Value {
    json!({"protocolVersion":1,"attemptId":seed.attempt_id,"fenceToken":seed.lease.fence_token})
}

async fn wait_for_approval(deps: &RuntimeDeps) -> String {
    tokio::time::timeout(std::time::Duration::from_secs(15), async {
        loop {
            if let Some(item) = deps.approvals.pending().first() { return item.request.id.clone(); }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
    }).await.expect("approval became visible")
}

#[tokio::test]
async fn compaction_worker_restart_and_approval_restart_preserve_the_task() {
    assert!(node_present(), "this integration gate requires Node, not a skipped test");
    let turns = Arc::new(AtomicUsize::new(0));
    let summaries = Arc::new(AtomicUsize::new(0));
    let requests: Arc<Mutex<Vec<Value>>> = Arc::default();
    let checkpoint_reached = Arc::new(tokio::sync::Notify::new());
    let app = Router::new().route("/v1/chat/completions", post({
        let turns = turns.clone(); let summaries = summaries.clone(); let requests = requests.clone();
        let checkpoint_reached = checkpoint_reached.clone();
        move |Json(body): Json<Value>| {
            let turns = turns.clone(); let summaries = summaries.clone(); let requests = requests.clone();
            let checkpoint_reached = checkpoint_reached.clone();
            async move {
                requests.lock().unwrap().push(body.clone());
                let is_summary = body.get("tools").and_then(Value::as_array).is_none_or(Vec::is_empty);
                let reply = if is_summary {
                    summaries.fetch_add(1, Ordering::SeqCst);
                    response(json!({"content":"Objective: read all eight source files; ID PUMP-A17. Completed source reads remain in canonical history. Next: finish remaining reads, request approval, write final.txt, verify the artifact."}), "stop")
                } else {
                    let turn = turns.fetch_add(1, Ordering::SeqCst);
                    if turn == 8 {
                        checkpoint_reached.notify_one();
                        // The first worker is killed while waiting for this provider response.
                        std::future::pending::<()>().await;
                    }
                    if turn < 8 {
                        response(json!({"tool_calls":[{"index":0,"id":format!("read-{turn}"),"type":"function",
                            "function":{"name":"workspace.read_text","arguments":json!({"path":format!("source-{turn}.txt")}).to_string()}}]}), "tool_calls")
                    } else if turn == 9 {
                        response(json!({"tool_calls":[{"index":0,"id":"write-final","type":"function",
                            "function":{"name":"workspace.write_text","arguments":json!({"path":"final.txt","content":"PUMP-A17 verified"}).to_string()}}]}), "tool_calls")
                    } else {
                        response(json!({"content":"PUMP-A17 verified. The approved artifact is final.txt."}), "stop")
                    }
                };
                ([("content-type", "text/event-stream")], reply)
            }
        }
    }));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base_url = format!("http://{}/v1", listener.local_addr().unwrap());
    let _server = ServerGuard(tokio::spawn(async move { axum::serve(listener, app).await.unwrap(); }));

    let (deps, dir) = deps();
    deps.plans.lock().unwrap().insert(RUN.into(), plan());
    let first_identity = execution(&deps, RUN, "journey-message", PROMPT);
    let seed = deps.checkpoints.lock().unwrap()[RUN].clone();
    for source in 0..8 {
        std::fs::write(dir.path().join(format!("runs/{RUN}/source-{source}.txt")),
            format!("Source {source} for PUMP-A17. {} END-OF-SOURCE-{source}", "measured source evidence; ".repeat(450))).unwrap();
    }
    let worker = AgentRuntime::spawn(deps.clone(), Arc::new(|_| {}), bundle()).unwrap();
    let running = tokio::spawn({ let worker = worker.clone(); let input = input(&base_url, first_identity);
        async move { worker.request("run.start", input).await } });
    tokio::time::timeout(std::time::Duration::from_secs(45), checkpoint_reached.notified()).await
        .unwrap_or_else(|_| panic!("did not reach restart boundary; events: {:?}", deps.events.snapshot(RUN)));
    assert!(summaries.load(Ordering::SeqCst) > 0, "real bounded summarizer requests must occur");
    let before = deps.events.load_context(RUN).unwrap().unwrap();
    assert_eq!(CoreCheckpoint::from_stored(&before).unwrap().plan.steps_taken, 8);
    let raw = deps.events.context_history(RUN, 0, 512).unwrap();
    assert!(raw.iter().any(|entry| entry.message.to_string().contains("END-OF-SOURCE-0")));
    assert!(raw.iter().any(|entry| entry.message.to_string().len() > 8000));
    worker.shutdown().await;
    assert!(tokio::time::timeout(std::time::Duration::from_secs(5), running).await.unwrap().unwrap().is_err());
    deps.events.release_claim(RUN, &seed.lease.owner, seed.lease.fence_token).unwrap();
    drop(worker); drop(deps);

    let (deps, seed2) = restore(&dir, &seed, "attempt-2");
    let worker = AgentRuntime::spawn(deps.clone(), Arc::new(|_| {}), bundle()).unwrap();
    let running = tokio::spawn({ let worker = worker.clone(); let input = input(&base_url, identity(&seed2));
        async move { worker.request("run.start", input).await } });
    let approval_id = wait_for_approval(&deps).await;
    assert!(!dir.path().join(format!("runs/{RUN}/final.txt")).exists());
    worker.shutdown().await;
    assert!(tokio::time::timeout(std::time::Duration::from_secs(5), running).await
        .expect("a pending approval must not stop dead-worker detection").unwrap().is_err());
    deps.events.release_claim(RUN, &seed2.lease.owner, seed2.lease.fence_token).unwrap();
    drop(worker); drop(deps);

    let (deps, seed3) = restore(&dir, &seed2, "attempt-3");
    let worker = AgentRuntime::spawn(deps.clone(), Arc::new(|_| {}), bundle()).unwrap();
    let running = tokio::spawn({ let worker = worker.clone(); let input = input(&base_url, identity(&seed3));
        async move { worker.request("run.start", input).await } });
    assert_eq!(wait_for_approval(&deps).await, approval_id, "restart must restore the same question");
    let reviewer = Session::open(User::new("reviewer", "Reviewer", vec![Role::Employee]));
    deps.approvals.decide_durable(&reviewer, &approval_id, true, None, |_| {
        deps.events.resolve_approval(&approval_id, ApprovalStatus::Approved, "reviewer", None, Utc::now())
            .and_then(|changed| if changed { Ok(()) } else { Err("decision was not persisted".into()) })
    }).unwrap();
    let result = tokio::time::timeout(std::time::Duration::from_secs(15), running).await.unwrap().unwrap().unwrap();
    worker.shutdown().await;
    assert_eq!(result["outcome"]["kind"], "completed", "{result}");
    assert_eq!(std::fs::read_to_string(dir.path().join(format!("runs/{RUN}/final.txt"))).unwrap(), "PUMP-A17 verified");
    let final_context = deps.events.load_context(RUN).unwrap().unwrap();
    let core = CoreCheckpoint::from_stored(&final_context).unwrap();
    // The last acknowledged core checkpoint predates the approval reservation;
    // restoring it and finishing the single write spends exactly one more step.
    assert_eq!(core.plan.steps_taken, 9);
    assert_eq!(core.calls.len(), 9, "eight reads and exactly one write");
    assert_eq!(core.produced.len(), 1);
    assert_eq!(deps.events.approvals_for_run(RUN).unwrap().len(), 1);
    let report = completion::verify(&CompletionInputs {
        unfinished_steps: core.plan.steps.iter().filter(|step| !step.done).count(),
        unknown_effects: deps.events.effect_obligations(RUN).unwrap().0,
        pending_approvals: deps.events.pending_approvals().unwrap().len(),
        artifacts: vec![("final.txt".into(), std::fs::read_to_string(dir.path().join(format!("runs/{RUN}/final.txt"))).unwrap() == "PUMP-A17 verified")],
        has_answer: !result["text"].as_str().unwrap().is_empty(), ..Default::default()
    }, Utc::now());
    assert!(report.passed(), "{}", report.explain());
    assert_eq!(report.enforce_outcome(RunOutcome::Completed), RunOutcome::Completed);
    deps.events.record_fenced(EventDraft::new(RUN, TaskEventType::CompletionVerified, "priya")
        .with(json!({"passed":report.passed(),"verification":report})), &seed3.lease).unwrap();
    deps.events.record_fenced(EventDraft::new(RUN, TaskEventType::RunCompleted, "priya"), &seed3.lease).unwrap();
    assert_eq!(deps.events.snapshot(RUN).unwrap().unwrap().state, RunState::Completed);
    assert_eq!(turns.load(Ordering::SeqCst), 11, "the interrupted approval must not re-request the model action");
    let raw_after = deps.events.context_history(RUN, 0, 512).unwrap();
    for before in &raw {
        let after = raw_after.iter().find(|entry| entry.seq == before.seq).unwrap();
        assert_eq!(after.message, before.message, "compaction/restart must not alter raw history");
    }
    let write = raw_after.iter().find(|entry| entry.message["content"].as_array()
        .is_some_and(|blocks| blocks.iter().any(|block| block["id"] == "write-final"))).unwrap();
    let operation_id = sarathi_lib::agent_runtime::events::operations::operation_id(RUN, write.seq, "write-final");
    let operation = deps.events.operation(RUN, &operation_id).unwrap().unwrap();
    assert_eq!(operation.attempts, 1);
    assert_eq!(operation.status, "succeeded");
    for request in requests.lock().unwrap().iter() {
        if request.get("tools").is_none() { continue; }
        assert!(request["messages"].to_string().contains("PUMP-A17"));
        let mut pending = std::collections::HashSet::new();
        for message in request["messages"].as_array().unwrap() {
            if message["role"] == "tool" {
                assert!(pending.remove(message["tool_call_id"].as_str().unwrap()), "orphan tool result");
            } else {
                assert!(pending.is_empty(), "incomplete tool batch before next message");
                if let Some(calls) = message["tool_calls"].as_array() {
                    for call in calls { assert!(pending.insert(call["id"].as_str().unwrap())); }
                }
            }
        }
        assert!(pending.is_empty());
    }
}
