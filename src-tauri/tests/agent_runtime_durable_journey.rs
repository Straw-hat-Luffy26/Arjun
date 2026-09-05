//! Production task driver + real Node worker, SQLite, gateway and executor.
//! Only model responses are scripted; the Tauri shell/router are not exercised.
//! This does not exercise the Tauri UI, model router, or a native inference model.
use super::*;
#[path = "agent_runtime_native_eval.rs"]
mod native_eval;
use axum::{routing::post, Json, Router};
use chrono::{Duration, Utc};
use sarathi_lib::agent_runtime::{
    context_api::CoreCheckpoint,
    events::{ApprovalStatus, EventDraft, RunState, TaskEventType},
    planning,
    resume::CheckpointSeed,
    task_driver::{self, DrivenTask, TaskDriver},
    tasks::{self, TaskRecord},
};
use sarathi_lib::orchestrator::plan::PlanRun;
use sarathi_lib::orchestrator::tools::ToolName;
use serde_json::{json, Value};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Mutex,
};

const RUN: &str = "run-1";
const PROMPT: &str = "Search the connected collection for PUMP-A17. Read source-0.txt through source-7.txt. Preserve exact ID PUMP-A17. Write final.txt containing PUMP-A17 verified, after approval, then read it back and cite the evidence.";
const ANSWER: &str = "PUMP-A17 verified [E1].";

#[tokio::test]
async fn requested_pause_saves_a_boundary_and_resume_does_not_repeat_the_read() {
    let reached = Arc::new(tokio::sync::Notify::new());
    let release = Arc::new(tokio::sync::Notify::new());
    let count = Arc::new(AtomicUsize::new(0));
    let app = Router::new().route("/v1/chat/completions", post({
        let reached = reached.clone(); let release = release.clone(); let count = count.clone();
        move || { let reached = reached.clone(); let release = release.clone(); let count = count.clone(); async move {
            let reply = if count.fetch_add(1, Ordering::SeqCst) == 0 {
                reached.notify_one(); release.notified().await;
                tool_response("read-before-pause", "workspace.read_text", json!({"path":"source-0.txt"}))
            } else { response(json!({"content": ANSWER}), "stop") };
            ([("content-type", "text/event-stream")], reply)
        }}
    }));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}/v1", listener.local_addr().unwrap());
    let _server = ServerGuard(tokio::spawn(async move { axum::serve(listener, app).await.unwrap(); }));
    let (deps, dir) = deps();
    deps.plans.lock().unwrap().insert(RUN.into(), plan());
    std::fs::write(dir.path().join("runs/run-1/source-0.txt"), "PUMP-A17 revision 2026").unwrap();
    let execution = execution(&deps, RUN, "journey-message", PROMPT);
    let seed = deps.checkpoints.lock().unwrap()[RUN].clone();
    let worker = AgentRuntime::spawn(deps.clone(), Arc::new(|_| {}), bundle()).unwrap();
    let running = tokio::spawn({ let deps=deps.clone(); let worker=worker.clone(); let seed=seed.clone(); let input=input(&url,execution);
        async move { drive(&deps,&worker,&seed,input).await }
    });
    tokio::time::timeout(std::time::Duration::from_secs(15), reached.notified()).await.unwrap();
    assert_eq!(worker.request("run.pause", json!({"runId":RUN})).await.unwrap()["requested"], true);
    release.notify_one();
    let result = tokio::time::timeout(std::time::Duration::from_secs(20), running).await.unwrap().unwrap();
    assert_eq!(result.outcome.kind(), "paused");
    assert_eq!(deps.events.load_context(RUN).unwrap().unwrap().view.phase,
        sarathi_lib::agent_runtime::events::context::ContextPhase::ModelReady);
    let saved = record(&result, &url, &deps);
    task_driver::publish(dir.path(), &saved, &deps.events, &seed.lease, json!({"outcome":"paused"}), |_| {}).unwrap();
    assert_eq!(deps.events.snapshot(RUN).unwrap().unwrap().state, RunState::Paused);
    assert!(tasks::load(dir.path(), RUN, Some("priya")).unwrap().completion_verification.is_none());
    assert!(deps.events.ending(RUN).is_none());
    worker.shutdown().await;
    deps.events.release_claim(RUN,&seed.lease.owner,seed.lease.fence_token).unwrap();
    let (restored, next) = restore(&dir,&seed,"after-pause");
    let worker = AgentRuntime::spawn(restored.clone(),Arc::new(|_| {}),bundle()).unwrap();
    let result = drive(&restored,&worker,&next,input(&url,identity(&next))).await;
    worker.shutdown().await;
    assert_eq!(result.response.unwrap()["outcome"]["kind"],"completed");
    assert_eq!(restored.calls.lock().unwrap()[RUN].iter().filter(|call| call.tool == "workspace.read_text").count(),1);
    assert_eq!(count.load(Ordering::SeqCst),2);
}

struct ServerGuard(tokio::task::JoinHandle<()>);
impl Drop for ServerGuard {
    fn drop(&mut self) {
        self.0.abort();
    }
}

fn response(delta: Value, reason: &str) -> String {
    let chunk = |delta: Value, finish: Value| {
        format!(
            "data: {}\n\n",
            json!({
                "id":"fixture", "object":"chat.completion.chunk", "created":0, "model":"fixture-model",
                "choices":[{"index":0,"delta":delta,"finish_reason":finish}]
            })
        )
    };
    format!(
        "{}{}{}data: [DONE]\n\n",
        chunk(json!({"role":"assistant"}), Value::Null),
        chunk(delta, Value::Null),
        chunk(json!({}), json!(reason))
    )
}

fn plan() -> PlanRun {
    let mut plan = planning::plan_for(RUN, PROMPT);
    // Keep the fixture catalogue small enough to force a 4k context journey.
    // This narrows capabilities; the production planner's steps are unchanged.
    plan.budget.permitted_tools = vec![
        ToolName::SearchDocuments,
        ToolName::ReadScopedFile,
        ToolName::WriteScopedFile,
    ];
    plan
}

fn input(base_url: &str, identity: Value) -> Value {
    json!({"runId":RUN,"messageId":"journey-message","prompt":PROMPT,
        "systemPrompt":"Use the authorized workspace tools. Preserve PUMP-A17 and obtain approval before writing.",
        "execution":identity,"model":{"id":"fixture-model","provider":"sovereign-local",
            "baseUrl":base_url,"contextWindow":4096,"maxTokens":256}})
}

fn restore(
    dir: &tempfile::TempDir,
    previous: &CheckpointSeed,
    attempt: &str,
) -> (Arc<RuntimeDeps>, CheckpointSeed) {
    let deps = deps_in(dir);
    let saved = deps.events.load_context(RUN).unwrap().unwrap();
    let core = CoreCheckpoint::from_stored(&saved).unwrap();
    assert_eq!(core.objective, PROMPT);
    assert_eq!(core.message_id, "journey-message");
    let mut restored_plan = plan();
    restored_plan.restore_progress(&core.plan).unwrap();
    deps.plans.lock().unwrap().insert(RUN.into(), restored_plan);
    deps.passages
        .lock()
        .unwrap()
        .insert(RUN.into(), core.passages);
    deps.calculations
        .lock()
        .unwrap()
        .insert(RUN.into(), core.calculations);
    deps.produced
        .lock()
        .unwrap()
        .insert(RUN.into(), core.produced);
    deps.calls.lock().unwrap().insert(RUN.into(), core.calls);
    let mut seed = previous.clone();
    seed.attempt_id = attempt.into();
    seed.lease = deps
        .events
        .claim_run(RUN, attempt, Duration::minutes(2), Utc::now())
        .unwrap()
        .unwrap();
    assert!(seed.lease.fence_token > previous.lease.fence_token);
    deps.checkpoints
        .lock()
        .unwrap()
        .insert(RUN.into(), seed.clone());
    deps.events
        .record_fenced(
            EventDraft::new(RUN, TaskEventType::RunResumed, "priya"),
            &seed.lease,
        )
        .unwrap();
    (deps, seed)
}

fn identity(seed: &CheckpointSeed) -> Value {
    json!({"protocolVersion":1,"attemptId":seed.attempt_id,"fenceToken":seed.lease.fence_token})
}

async fn wait_for_approval(deps: &RuntimeDeps) -> String {
    tokio::time::timeout(std::time::Duration::from_secs(15), async {
        loop {
            if let Some(item) = deps.approvals.pending().first() {
                return item.request.id.clone();
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("approval became visible")
}

#[tokio::test]
async fn compaction_worker_restart_and_approval_restart_preserve_the_task() {
    recovery_journey(ANSWER, true, None).await;
}

#[tokio::test]
async fn production_task_driver_rejects_an_invented_citation_after_recovery() {
    recovery_journey("PUMP-A17 verified [E99].", false, None).await;
}

async fn recovery_journey(answer: &'static str, expected_success: bool, destination: Option<u32>) {
    assert!(
        node_present(),
        "this integration gate requires Node, not a skipped test"
    );
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
                    if turn == 9 {
                        checkpoint_reached.notify_one();
                        // The first worker is killed while waiting for this provider response.
                        std::future::pending::<()>().await;
                    }
                    if turn == 0 {
                        tool_response("search-evidence", "knowledge.search_authorized", json!({"query":"PUMP-A17"}))
                    } else if turn < 9 {
                        let turn = turn - 1;
                        response(json!({"tool_calls":[{"index":0,"id":format!("read-{turn}"),"type":"function",
                            "function":{"name":"workspace.read_text","arguments":json!({"path":format!("source-{turn}.txt")}).to_string()}}]}), "tool_calls")
                    } else if turn == 10 {
                        response(json!({"tool_calls":[{"index":0,"id":"write-final","type":"function",
                            "function":{"name":"workspace.write_text","arguments":json!({"path":"final.txt","content":"PUMP-A17 verified"}).to_string()}}]}), "tool_calls")
                    } else if turn == 11 {
                        tool_response("read-final", "workspace.read_text", json!({"path":"final.txt"}))
                    } else {
                        response(json!({"content":answer}), "stop")
                    }
                };
                ([("content-type", "text/event-stream")], reply)
            }
        }
    }));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base_url = format!("http://{}/v1", listener.local_addr().unwrap());
    let _server = ServerGuard(tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    }));

    let (deps, dir) = deps();
    deps.plans.lock().unwrap().insert(RUN.into(), plan());
    index_evidence(&deps);
    let first_identity = execution(&deps, RUN, "journey-message", PROMPT);
    if destination.is_some() {
        deps.checkpoints
            .lock()
            .unwrap()
            .get_mut(RUN)
            .unwrap()
            .model_context = Some(model_context(4096, "fixture-model"));
    }
    let seed = deps.checkpoints.lock().unwrap()[RUN].clone();
    for source in 0..8 {
        std::fs::write(
            dir.path().join(format!("runs/{RUN}/source-{source}.txt")),
            format!(
                "Source {source} for PUMP-A17. {} END-OF-SOURCE-{source}",
                "measured source evidence; ".repeat(450)
            ),
        )
        .unwrap();
    }
    let worker = AgentRuntime::spawn(deps.clone(), Arc::new(|_| {}), bundle()).unwrap();
    let running = tokio::spawn({
        let worker = worker.clone();
        let input = input(&base_url, first_identity);
        async move { worker.request("run.start", input).await }
    });
    tokio::time::timeout(
        std::time::Duration::from_secs(45),
        checkpoint_reached.notified(),
    )
    .await
    .unwrap_or_else(|_| {
        panic!(
            "did not reach restart boundary; events: {:?}",
            deps.events.snapshot(RUN)
        )
    });
    assert!(
        summaries.load(Ordering::SeqCst) > 0,
        "real bounded summarizer requests must occur"
    );
    let before = deps.events.load_context(RUN).unwrap().unwrap();
    assert_eq!(
        CoreCheckpoint::from_stored(&before)
            .unwrap()
            .plan
            .steps_taken,
        9
    );
    assert_eq!(
        CoreCheckpoint::from_stored(&before).unwrap().passages.len(),
        1
    );
    let raw = deps.events.context_history(RUN, 0, 512).unwrap();
    assert!(raw
        .iter()
        .any(|entry| entry.message.to_string().contains("END-OF-SOURCE-0")));
    assert!(raw
        .iter()
        .any(|entry| entry.message.to_string().len() > 8000));
    worker.shutdown().await;
    assert!(
        tokio::time::timeout(std::time::Duration::from_secs(5), running)
            .await
            .unwrap()
            .unwrap()
            .is_err()
    );
    deps.events
        .release_claim(RUN, &seed.lease.owner, seed.lease.fence_token)
        .unwrap();
    drop(worker);
    drop(deps);

    let (deps, seed2) = restore(&dir, &seed, "attempt-2");
    let worker = AgentRuntime::spawn(deps.clone(), Arc::new(|_| {}), bundle()).unwrap();
    let running = tokio::spawn({
        let worker = worker.clone();
        let input = input(&base_url, identity(&seed2));
        async move { worker.request("run.start", input).await }
    });
    let approval_id = wait_for_approval(&deps).await;
    assert!(!dir.path().join(format!("runs/{RUN}/final.txt")).exists());
    worker.shutdown().await;
    assert!(
        tokio::time::timeout(std::time::Duration::from_secs(5), running)
            .await
            .expect("a pending approval must not stop dead-worker detection")
            .unwrap()
            .is_err()
    );
    deps.events
        .release_claim(RUN, &seed2.lease.owner, seed2.lease.fence_token)
        .unwrap();
    drop(worker);
    drop(deps);

    let (deps, mut seed3) = restore(&dir, &seed2, "attempt-3");
    if let Some(window) = destination {
        seed3.model_id = "destination-registry-00017".into();
        seed3.model_context = Some(model_context(window, &seed3.model_id));
        seed3.model_context.as_mut().unwrap().served_model_id = "destination-served-00017".into();
        deps.checkpoints
            .lock()
            .unwrap()
            .insert(RUN.into(), seed3.clone());
    }
    let worker = AgentRuntime::spawn(deps.clone(), Arc::new(|_| {}), bundle()).unwrap();
    let running = tokio::spawn({
        let worker = worker.clone();
        let deps = deps.clone();
        let mut input = input(&base_url, identity(&seed3));
        if let Some(model) = &seed3.model_context {
            input["model"]["id"] = json!(model.served_model_id);
            input["model"]["contextWindow"] = json!(model.context_window);
        }
        let seed = seed3.clone();
        async move { drive(&deps, &worker, &seed, input).await }
    });
    assert_eq!(
        wait_for_approval(&deps).await,
        approval_id,
        "restart must restore the same question"
    );
    let reviewer = Session::open(User::new("reviewer", "Reviewer", vec![Role::Employee]));
    deps.approvals
        .decide_durable(&reviewer, &approval_id, true, None, |_| {
            deps.events
                .resolve_approval(
                    &approval_id,
                    ApprovalStatus::Approved,
                    "reviewer",
                    None,
                    Utc::now(),
                )
                .and_then(|changed| {
                    if changed {
                        Ok(())
                    } else {
                        Err("decision was not persisted".into())
                    }
                })
        })
        .unwrap();
    let result = tokio::time::timeout(std::time::Duration::from_secs(15), running)
        .await
        .unwrap()
        .unwrap();
    worker.shutdown().await;
    assert_eq!(
        result.response.as_ref().unwrap()["outcome"]["kind"],
        "completed",
        "the model proposes success in both cases: {:?}",
        result.response
    );
    assert_eq!(
        result.outcome.is_success(),
        expected_success,
        "{}",
        result.completion.explain()
    );
    assert!(
        result.plan.unfinished().is_empty(),
        "steps must settle from real receipts and the answer check"
    );
    let verification = result.verification.as_ref().expect("answer was checked");
    assert_eq!(
        verification.is_ready(),
        expected_success,
        "{verification:?}"
    );
    assert_eq!(
        verification.citations_resolved,
        usize::from(expected_success)
    );
    assert_eq!(verification.coverage.passages_available, 1);
    assert!(result.record_failure.is_none());
    assert_eq!(
        std::fs::read_to_string(dir.path().join(format!("runs/{RUN}/final.txt"))).unwrap(),
        "PUMP-A17 verified"
    );
    let final_context = deps.events.load_context(RUN).unwrap().unwrap();
    if let Some(window) = destination {
        let transition = final_context.view.model_transition.as_ref().unwrap();
        assert_eq!(
            transition.status,
            sarathi_lib::agent_runtime::model_transition::TransitionStatus::Ready
        );
        assert_eq!(transition.from.context_window, 4096);
        assert_eq!(transition.to.context_window, window);
        assert_eq!(transition.to.model_id, "destination-registry-00017");
        assert_eq!(transition.to.served_model_id, "destination-served-00017");
        assert_eq!(final_context.checkpoint.model_id, transition.to.model_id);
        assert!(requests
            .lock()
            .unwrap()
            .iter()
            .any(|r| r["model"] == "destination-served-00017"));
    }
    let core = CoreCheckpoint::from_stored(&final_context).unwrap();
    assert_eq!(core.plan.steps_taken, 11);
    assert_eq!(
        core.calls.len(),
        11,
        "one search, nine reads and exactly one write"
    );
    assert!(core
        .calls
        .iter()
        .all(|call| call.outcome == tasks::CallOutcome::Succeeded));
    assert_eq!(core.produced.len(), 1);
    assert_eq!(
        core.passages.len(),
        1,
        "citation evidence survived both restarts"
    );
    assert_eq!(deps.events.approvals_for_run(RUN).unwrap().len(), 1);
    assert_eq!(result.completion.passed(), expected_success);
    let record = record(&result, &base_url, &deps);
    let published = task_driver::publish(
        dir.path(),
        &record,
        &deps.events,
        &seed3.lease,
        json!({"outcome":result.outcome.kind(),"failure":result.outcome.detail()}),
        |_| {},
    )
    .unwrap();
    assert_eq!(published, result.outcome);
    let saved_record = tasks::load(dir.path(), RUN, Some("priya")).unwrap();
    assert_eq!(saved_record.outcome, Some(result.outcome.clone()));
    assert_eq!(saved_record.approvals.len(), 1);
    assert_eq!(saved_record.approvals[0].id, approval_id);
    assert_eq!(saved_record.approvals[0].state, "approved");
    assert!(!saved_record.compactions.is_empty());
    assert_eq!(
        saved_record.completion_verification.unwrap().passed(),
        expected_success
    );
    assert_eq!(
        saved_record.verification.unwrap().is_ready(),
        expected_success
    );
    assert!(saved_record.plan.unfinished().is_empty());
    assert_eq!(
        deps.events.snapshot(RUN).unwrap().unwrap().state,
        if expected_success {
            RunState::Completed
        } else {
            RunState::Failed
        }
    );
    let history = deps.events.events_since(RUN, 0).unwrap().events;
    let checks = history
        .iter()
        .filter(|e| e.event_type == TaskEventType::CompletionVerified)
        .collect::<Vec<_>>();
    assert_eq!(
        checks.len(),
        1,
        "only the surviving driver may record completion verification"
    );
    assert_eq!(checks[0].payload["passed"], expected_success);
    let ending = history
        .iter()
        .find(|e| e.event_type == result.outcome.event_type())
        .unwrap();
    assert!(
        checks[0].seq < ending.seq,
        "verification must be durable before the terminal event"
    );
    assert_eq!(
        history
            .iter()
            .filter(|e| e.event_type == TaskEventType::RunCompleted)
            .count(),
        usize::from(expected_success)
    );
    assert_eq!(
        turns.load(Ordering::SeqCst),
        13,
        "the interrupted approval must not re-request the model action"
    );
    let raw_after = deps.events.context_history(RUN, 0, 512).unwrap();
    for before in &raw {
        let after = raw_after
            .iter()
            .find(|entry| entry.seq == before.seq)
            .unwrap();
        assert_eq!(
            after.message, before.message,
            "compaction/restart must not alter raw history"
        );
    }
    let write = raw_after
        .iter()
        .find(|entry| {
            entry.message["content"]
                .as_array()
                .is_some_and(|blocks| blocks.iter().any(|block| block["id"] == "write-final"))
        })
        .unwrap();
    let operation_id =
        sarathi_lib::agent_runtime::events::operations::operation_id(RUN, write.seq, "write-final");
    let operation = deps.events.operation(RUN, &operation_id).unwrap().unwrap();
    assert_eq!(operation.attempts, 1);
    assert_eq!(operation.status, "succeeded");
    for request in requests.lock().unwrap().iter() {
        if request.get("tools").is_none() {
            continue;
        }
        assert!(request["messages"].to_string().contains("PUMP-A17"));
        let mut pending = std::collections::HashSet::new();
        for message in request["messages"].as_array().unwrap() {
            if message["role"] == "tool" {
                assert!(
                    pending.remove(message["tool_call_id"].as_str().unwrap()),
                    "orphan tool result"
                );
            } else {
                assert!(
                    pending.is_empty(),
                    "incomplete tool batch before next message"
                );
                if let Some(calls) = message["tool_calls"].as_array() {
                    for call in calls {
                        assert!(pending.insert(call["id"].as_str().unwrap()));
                    }
                }
            }
        }
        assert!(pending.is_empty());
    }
}

#[tokio::test]
async fn production_task_driver_refuses_skipped_plan_steps() {
    assert!(node_present(), "this integration gate requires Node");
    let app = Router::new().route(
        "/v1/chat/completions",
        post(|| async {
            (
                [("content-type", "text/event-stream")],
                response(json!({"content":ANSWER}), "stop"),
            )
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base_url = format!("http://{}/v1", listener.local_addr().unwrap());
    let _server = ServerGuard(tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    }));
    let (deps, dir) = deps();
    deps.plans.lock().unwrap().insert(RUN.into(), plan());
    index_evidence(&deps);
    let execution = execution(&deps, RUN, "journey-message", PROMPT);
    let seed = deps.checkpoints.lock().unwrap()[RUN].clone();
    let worker = AgentRuntime::spawn(deps.clone(), Arc::new(|_| {}), bundle()).unwrap();
    let result = drive(&deps, &worker, &seed, input(&base_url, execution)).await;
    worker.shutdown().await;
    assert_eq!(
        result.response.as_ref().unwrap()["outcome"]["kind"],
        "completed"
    );
    assert!(!result.outcome.is_success());
    assert_eq!(
        result.plan.unfinished().len(),
        1,
        "a claim of completion cannot substitute for search receipts"
    );
    assert_eq!(
        result
            .verification
            .as_ref()
            .unwrap()
            .coverage
            .passages_available,
        0,
        "an indexed source is not evidence until this run retrieves it"
    );
    assert!(result
        .completion
        .blocking()
        .iter()
        .any(|c| c.criterion_id == "plan.every_step_reached"));
    assert!(result
        .completion
        .blocking()
        .iter()
        .any(|c| c.criterion_id == "answer.grounded"));
    task_driver::publish(
        dir.path(),
        &record(&result, &base_url, &deps),
        &deps.events,
        &seed.lease,
        json!({"outcome":result.outcome.kind(),"failure":result.outcome.detail()}),
        |_| {},
    )
    .unwrap();
    assert_eq!(deps.events.ending(RUN), Some(TaskEventType::RunFailed));
    assert!(!tasks::load(dir.path(), RUN, Some("priya"))
        .unwrap()
        .is_ready());
}

fn tool_response(id: &str, name: &str, args: Value) -> String {
    response(
        json!({"tool_calls":[{"index":0,"id":id,"type":"function",
        "function":{"name":name,"arguments":args.to_string()}}]}),
        "tool_calls",
    )
}

fn index_evidence(deps: &RuntimeDeps) {
    use sarathi_lib::{
        knowledge::{Chunk, ChunkKind},
        policy::Classification,
    };
    let text = "PUMP-A17 verified";
    deps.index
        .index_document(
            "Pump register",
            Classification::Internal,
            &[Chunk {
                id: "pump-register-chunk".into(),
                document_sha256: "pump-register-fixture".into(),
                ordinal: 0,
                char_count: text.len() as u32,
                text: text.into(),
                page: 1,
                section_path: Vec::new(),
                kind: ChunkKind::Prose,
            }],
        )
        .unwrap();
}

async fn drive(
    deps: &RuntimeDeps,
    worker: &AgentRuntime,
    seed: &CheckpointSeed,
    input: Value,
) -> DrivenTask {
    let lost = std::sync::atomic::AtomicBool::new(false);
    TaskDriver {
        run_id: RUN,
        prompt: PROMPT,
        actor: "priya",
        lease: &seed.lease,
        lease_lost: &lost,
        events: &deps.events,
        health: &deps.audit_health,
        plans: &deps.plans,
        passages: &deps.passages,
        calculations: &deps.calculations,
        produced: &deps.produced,
        calls: &deps.calls,
    }
    .run(
        worker,
        input,
        std::time::Duration::from_secs(30),
        |answer_chars| {
            deps.events
                .record_fenced(
                    EventDraft::new(RUN, TaskEventType::VerificationStarted, "priya")
                        .with(json!({"answerChars":answer_chars})),
                    &seed.lease,
                )
                .unwrap();
        },
        |_| {},
    )
    .await
}

fn record(result: &DrivenTask, base_url: &str, deps: &RuntimeDeps) -> TaskRecord {
    use sarathi_lib::{
        registry::{router::RoutingDecision, ModelRole, Runtime},
        serving::Endpoint,
    };
    TaskRecord {
        children: Vec::new(),
        run_id: RUN.into(),
        prompt: PROMPT.into(),
        user_id: "priya".into(),
        started_at: result.finished_at.to_rfc3339(),
        finished_at: result.finished_at.to_rfc3339(),
        duration_seconds: 0,
        routing: RoutingDecision {
            model_id: "fixture-model".into(),
            model_name: "Scripted fixture".into(),
            role: ModelRole::Reasoning,
            intent: "fixture".into(),
            confidence: 1.0,
            used_fallback: false,
            reasons: vec!["Deterministic test; no router or inference".into()],
            gpu_plan_summary: "No GPU used".into(),
            fully_on_gpu: false,
        },
        endpoint: Endpoint {
            base_url: base_url.into(),
            served_model_id: "fixture-model".into(),
            managed: false,
            runtime: Runtime::LlamaCpp,
        },
        plan: result.plan.clone(),
        answer: result.answer.clone(),
        turns: result.turns,
        verification: result.verification.clone(),
        completion_verification: Some(result.completion.clone()),
        artifacts: result.artifacts.clone(),
        evidence: TaskRecord::evidence_from(&result.passages),
        calculations: result.calculations.clone(),
        tool_calls: result.calls.clone(),
        approvals: deps
            .events
            .approvals_for_run(RUN)
            .unwrap()
            .into_iter()
            .map(|item| tasks::ApprovalRecord {
                arguments: item.display_arguments(),
                id: item.approval_id,
                tool: item.tool,
                target: item.target,
                consequences: item.reason,
                requested_at: item.created_at,
                state: item.status.as_str().into(),
                decided_by: item.resolved_by,
                decided_at: item.resolved_at,
                because: item.resolution,
            })
            .collect(),
        failure: result.outcome.detail().map(str::to_string),
        outcome: Some(result.outcome.clone()),
        compactions: deps
            .events
            .snapshot(RUN)
            .unwrap()
            .unwrap()
            .compaction_events,
        working_notes: None,
        context_ledger: None,
    }
}

fn model_context(
    window: u32,
    id: &str,
) -> sarathi_lib::agent_runtime::model_transition::ModelContext {
    sarathi_lib::agent_runtime::model_transition::ModelContext {
        model_id: id.into(),
        served_model_id: id.into(),
        provider: "sovereign-local".into(),
        context_window: window,
        max_tokens: 256,
        input: vec!["text".into()],
    }
}

#[tokio::test]
async fn production_driver_runs_registered_specialists_and_saves_their_results_in_the_parent() {
    specialist_journey(true).await;
}

#[tokio::test]
async fn production_driver_refuses_completion_when_a_specialist_failed() {
    specialist_journey(false).await;
}

async fn specialist_journey(complete: bool) {
    assert!(node_present());
    let turns = Arc::new(AtomicUsize::new(0));
    let app = Router::new().route("/v1/chat/completions", post({
        let turns = turns.clone();
        move |Json(_body): Json<Value>| {
            let turn = turns.fetch_add(1, Ordering::SeqCst);
            async move {
                let reply = match turn {
                    0 => tool_response("search", "knowledge.search_authorized", json!({"query":"PUMP-A17"})),
                    1..=8 => tool_response(&format!("read-{turn}"), "workspace.read_text", json!({"path":format!("source-{}.txt",turn-1)})),
                    9 => tool_response("calculate", "calculation.evaluate_with_units", json!({"expression": if complete { "3 bar * 2" } else { "3 bar + 2 kg" }})),
                    10 => tool_response("write", "workspace.write_text", json!({"path":"final.txt","content":"PUMP-A17 verified"})),
                    11..=14 => tool_response(&format!("delegate-{turn}"), "agent.delegate_readonly", json!({"profile":sarathi_lib::subagents::workers::PROFILES[turn-11],"task":"PUMP-A17"})),
                    15 => tool_response("read-final", "workspace.read_text", json!({"path":"final.txt"})),
                    _ => response(json!({"content":ANSWER}), "stop"),
                };
                ([("content-type", "text/event-stream")], reply)
            }
        }
    }));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base_url = format!("http://{}/v1", listener.local_addr().unwrap());
    let _server = ServerGuard(tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    }));
    let (mut deps, dir) = deps();
    let resources = sarathi_lib::subagents::workers::Resources {
        events: deps.events.clone(),
        index: deps.index.clone(),
        session: deps.session.clone(),
        workspaces: deps.workspaces.clone(),
        passages: deps.passages.clone(),
        calculations: deps.calculations.clone(),
        produced: deps.produced.clone(),
    };
    let profiles = sarathi_lib::subagents::load_profiles(
        &PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("agents"),
    );
    let manager = sarathi_lib::subagents::workers::register(
        sarathi_lib::subagents::SubagentManager::new(profiles.profiles, deps.events.clone()),
        resources,
    );
    Arc::get_mut(&mut deps).unwrap().subagents = Arc::new(manager);
    let mut planned = plan();
    planned.budget.max_steps = 24;
    planned.budget.permitted_tools.extend([
        ToolName::RunCalculation,
        ToolName::ValidateArtifact,
        ToolName::AgentDelegateReadonly,
    ]);
    deps.plans.lock().unwrap().insert(RUN.into(), planned);
    index_evidence(&deps);
    for source in 0..8 {
        std::fs::write(
            dir.path().join(format!("runs/{RUN}/source-{source}.txt")),
            "PUMP-A17 verified",
        )
        .unwrap();
    }
    let execution = execution(&deps, RUN, "journey-message", PROMPT);
    let seed = deps.checkpoints.lock().unwrap()[RUN].clone();
    let worker = AgentRuntime::spawn(deps.clone(), Arc::new(|_| {}), bundle()).unwrap();
    let running = tokio::spawn({
        let deps = deps.clone();
        let worker = worker.clone();
        let seed = seed.clone();
        let mut input = input(&base_url, execution);
        input["model"]["contextWindow"] = json!(32768);
        async move { drive(&deps, &worker, &seed, input).await }
    });
    let approval = wait_for_approval(&deps).await;
    let reviewer = Session::open(User::new("reviewer", "Reviewer", vec![Role::Employee]));
    deps.approvals
        .decide_durable(&reviewer, &approval, true, None, |_| {
            deps.events
                .resolve_approval(
                    &approval,
                    ApprovalStatus::Approved,
                    "reviewer",
                    None,
                    Utc::now(),
                )
                .map(|_| ())
        })
        .unwrap();
    let result = running.await.unwrap();
    worker.shutdown().await;
    assert_eq!(
        result.outcome.is_success(),
        complete,
        "{:?}: {}",
        result.outcome,
        result.completion.explain()
    );
    assert_eq!(turns.load(Ordering::SeqCst), 17);
    let children = deps.events.children_for_run(RUN).unwrap();
    assert_eq!(
        children.len(),
        4,
        "{:?}",
        deps.calls.lock().unwrap().get(RUN)
    );
    assert_eq!(
        children
            .iter()
            .all(|c| c.result.is_complete() && !c.packet.inputs.is_empty()),
        complete,
        "{children:?}"
    );
    if complete {
        assert!(children.iter().any(|c| c
            .result
            .findings
            .iter()
            .any(|f| f.statement.contains("6 bar"))));
    } else {
        assert!(children
            .iter()
            .any(|c| c.packet.profile == "calculation-checker" && !c.result.is_complete()));
        assert!(!result.completion.passed());
    }
    let record = record(&result, &base_url, &deps);
    task_driver::publish(
        dir.path(),
        &record,
        &deps.events,
        &seed.lease,
        json!({"outcome":result.outcome.kind()}),
        |_| {},
    )
    .unwrap();
    let saved = tasks::load(dir.path(), RUN, Some("priya")).unwrap();
    assert_eq!(saved.children.len(), 4);
    assert_eq!(
        saved.children[0].result.result_hash,
        children[0].result.result_hash
    );
    assert!(saved.evidence.iter().any(|e| e.marker == 1));
}

#[tokio::test]
async fn production_driver_transitions_to_smaller_window_with_pending_approval() {
    recovery_journey(ANSWER, true, Some(3584)).await;
}

#[tokio::test]
async fn production_driver_transitions_to_larger_window_with_pending_approval() {
    recovery_journey(ANSWER, true, Some(16384)).await;
}
