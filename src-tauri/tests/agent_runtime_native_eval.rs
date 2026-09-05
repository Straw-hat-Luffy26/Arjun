//! Opt-in native evaluation. No scripted provider, generated scores or downloads.
use super::*;
use sarathi_lib::agent_runtime::model_transition::ModelContext;
use serde::Deserialize;

#[derive(Clone, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct NativeModel { id: String, base_url: String, model_path: PathBuf, context_window: u32, max_tokens: u32 }

fn output(report: &Value) {
    let path = std::env::var_os("ARJUN_NATIVE_EVAL_OUTPUT").map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../evidence/native-reliability.json"));
    if let Some(parent) = path.parent() { std::fs::create_dir_all(parent).unwrap(); }
    std::fs::write(&path, serde_json::to_vec_pretty(report).unwrap()).unwrap();
    eprintln!("Native evaluation report: {}", path.display());
}

#[tokio::test]
#[ignore = "Requires ARJUN_NATIVE_EVAL_MODELS with two running local models; never downloads or substitutes scripted responses"]
async fn real_local_models_long_task_reliability() {
    let config = std::env::var("ARJUN_NATIVE_EVAL_MODELS").ok()
        .and_then(|raw| serde_json::from_str::<Vec<NativeModel>>(&raw).ok());
    let Some(models) = config.filter(|models| models.len() >= 2 && models[0].id != models[1].id && models[0].model_path != models[1].model_path) else {
        output(&json!({"schemaVersion":1,"status":"blocked","at":Utc::now(),
            "reason":"Configure at least two distinct real local models and loopback inference endpoints in ARJUN_NATIVE_EVAL_MODELS.",
            "modelTasksAttempted":0,"successfulCompletion":null,"retainedFacts":null,"retainedCorrections":null,
            "overflowRefusals":null,"duplicateEffects":null,"recoveryFailures":null,"reliabilityClaim":false}));
        panic!("Native evaluation prerequisites are missing; no reliability measurement was made.");
    };
    let client = reqwest::Client::builder().no_proxy().redirect(reqwest::redirect::Policy::none())
        .timeout(std::time::Duration::from_secs(5)).build().unwrap();
    for model in &models {
        let url = reqwest::Url::parse(&model.base_url).unwrap();
        assert!(url.scheme() == "http" && matches!(url.host_str(), Some("127.0.0.1" | "localhost" | "[::1]")), "Only loopback inference is permitted");
        assert!(model.model_path.is_file(), "The declared native model file must exist");
        let list: Value = client.get(format!("{}/models", model.base_url.trim_end_matches('/'))).send().await
            .expect("Native endpoint is unavailable").error_for_status().expect("Native model listing failed").json().await.unwrap();
        assert!(list["data"].as_array().is_some_and(|items| items.iter().any(|item| item["id"] == model.id)), "The endpoint does not serve the declared model");
    }

    let (mut deps, dir) = deps();
    index_evidence(&deps);
    for source in 0..8 {
        std::fs::write(dir.path().join(format!("runs/{RUN}/source-{source}.txt")), format!(
            "PUMP-A17 inspection source {source}. Recorded revision 2019. {} END-OF-SOURCE-{source}",
            "Inspection entry: compare the recorded revision against subsequent operator corrections. ".repeat(140))).unwrap();
    }
    deps.plans.lock().unwrap().insert(RUN.into(), plan());
    execution(&deps, RUN, "journey-message", PROMPT);
    let mut seed = deps.checkpoints.lock().unwrap()[RUN].clone();
    // One fixed lifetime allowance for native inference; restoration never resets it.
    seed.deadline_ms = (Utc::now() + Duration::minutes(15)).timestamp_millis();
    let correction = "Operator correction: the verified revision is 2026, not 2019. Include revision 2026 in final.txt and in the final answer.";
    let mut transitions = 0;
    let mut attempts = Vec::new();
    let mut completion = false;
    let mut recovery_failures = 0;
    let mut overflow_refusals = 0;
    let mut final_answer = String::new();
    for (phase, model_index) in [0, 1, 0].into_iter().enumerate() {
        let model = &models[model_index];
        seed.model_id = model.id.clone();
        seed.model_context = Some(ModelContext { model_id:model.id.clone(), served_model_id:model.id.clone(),
            provider:"sovereign-local".into(), context_window:model.context_window, max_tokens:model.max_tokens, input:vec!["text".into()] });
        seed.model_context.as_ref().unwrap().validate().unwrap();
        deps.checkpoints.lock().unwrap().insert(RUN.into(),seed.clone());
        let compacted = Arc::new(tokio::sync::Notify::new());
        let worker = AgentRuntime::spawn(deps.clone(), Arc::new({ let compacted=compacted.clone(); move |event| {
            if event["event"]["type"] == "context_compacted" { compacted.notify_one(); }
        }}), bundle()).unwrap();
        let mut params = input(&model.base_url,identity(&seed));
        params["model"] = json!({"id":model.id,"provider":"sovereign-local","baseUrl":model.base_url,
            "contextWindow":model.context_window,"maxTokens":model.max_tokens});
        params["deadlineMs"] = json!(seed.deadline_ms);
        let running = tokio::spawn({ let deps=deps.clone(); let worker=worker.clone(); let seed=seed.clone(); async move {
            let lost = std::sync::atomic::AtomicBool::new(false);
            TaskDriver { run_id:RUN,prompt:PROMPT,actor:"priya",lease:&seed.lease,lease_lost:&lost,
                events:&deps.events,health:&deps.audit_health,plans:&deps.plans,passages:&deps.passages,
                calculations:&deps.calculations,produced:&deps.produced,calls:&deps.calls }
                .run(&worker,params,std::time::Duration::from_secs(900), |_| {}, |_| {}).await
        }});
        tokio::pin!(running);
        let mut pause_sent = false;
        let result = loop {
            tokio::select! {
                result = &mut running => break result.unwrap(),
                _ = compacted.notified(), if phase < 2 && !pause_sent => {
                    if phase == 0 { worker.request("run.steer",json!({"runId":RUN,"text":correction})).await.unwrap(); }
                    worker.request("run.pause",json!({"runId":RUN})).await.unwrap();
                    pause_sent = true;
                },
                _ = tokio::time::sleep(std::time::Duration::from_millis(100)) => {
                    // Only the synthetic final.txt write is approved, via the real queue.
                    for pending in deps.approvals.pending() {
                        let request = &pending.request;
                        let allow = request.tool == "workspace.write_text" && request.target == "final.txt";
                        let reviewer = Session::open(User::new("reviewer","Evaluation reviewer",vec![Role::Employee]));
                        deps.approvals.decide_durable(&reviewer,&request.id,allow,Some("Isolated evaluation workspace only"), |_| {
                            deps.events.resolve_approval(&request.id,if allow {ApprovalStatus::Approved} else {ApprovalStatus::Rejected},"reviewer",None,Utc::now()).map(|_| ())
                        }).unwrap();
                    }
                    deps.events.renew_claim(RUN,&seed.lease.owner,seed.lease.fence_token,Duration::minutes(2),Utc::now()).unwrap();
                }
            }
        };
        final_answer = result.answer.clone();
        overflow_refusals += usize::from(result.outcome.detail().is_some_and(|text| text.contains("context") && (text.contains("fit") || text.contains("budget"))));
        completion = result.outcome.is_success();
        let paused = result.outcome.kind() == "paused";
        if phase > 0 && result.response.is_err() { recovery_failures += 1; }
        attempts.push(json!({"modelId":model.id,"window":model.context_window,"outcome":result.outcome,"turns":result.turns}));
        if paused || completion {
            let mut record = record(&result,&model.base_url,&deps);
            record.endpoint.model_id = model.id.clone();
            task_driver::publish(dir.path(),&record,&deps.events,&seed.lease,json!({"outcome":result.outcome.kind()}), |_| {}).unwrap();
        }
        worker.shutdown().await;
        deps.events.release_claim(RUN,&seed.lease.owner,seed.lease.fence_token).unwrap();
        if !paused || phase == 2 { break; }
        let saved = deps.events.load_context(RUN).unwrap().unwrap();
        let core = CoreCheckpoint::from_stored(&saved).unwrap();
        deps = deps_in(&dir);
        core.validate_evidence(&deps.index,&deps.session.read().unwrap().clone().unwrap()).unwrap();
        let mut next_plan = plan(); next_plan.restore_progress(&core.plan).unwrap();
        deps.plans.lock().unwrap().insert(RUN.into(),next_plan);
        deps.passages.lock().unwrap().insert(RUN.into(),core.passages);
        deps.calculations.lock().unwrap().insert(RUN.into(),core.calculations);
        deps.produced.lock().unwrap().insert(RUN.into(),core.produced);
        deps.calls.lock().unwrap().insert(RUN.into(),core.calls);
        seed.lease = deps.events.claim_run(RUN,&format!("native-attempt-{}",phase+1),Duration::minutes(2),Utc::now()).unwrap().unwrap();
        seed.attempt_id = format!("native-attempt-{}",phase+1);
        deps.events.record_fenced(EventDraft::new(RUN,TaskEventType::RunResumed,"priya"),&seed.lease).unwrap();
        transitions += 1;
    }
    let snapshot = deps.events.snapshot(RUN).unwrap().unwrap();
    let calls = deps.calls.lock().unwrap().get(RUN).cloned().unwrap_or_default();
    let writes = calls.iter().filter(|call| call.tool == "workspace.write_text" && call.outcome == tasks::CallOutcome::Succeeded).count();
    let artifact = std::fs::read_to_string(dir.path().join("runs/run-1/final.txt")).unwrap_or_default();
    let retained_fact = final_answer.contains("PUMP-A17") && artifact.contains("PUMP-A17");
    let retained_correction = final_answer.contains("2026") && artifact.contains("2026") && !artifact.contains("2019");
    let covered = transitions == 2 && snapshot.compactions >= 2;
    output(&json!({"schemaVersion":1,"status":if covered {"measured"} else {"incomplete_coverage"},"at":Utc::now(),
        "scope":"Production task driver, real Node workers, real local inference, isolated tools; excludes desktop IPC/UI",
        "modelTasksAttempted":1,"successfulCompletion":completion,"retainedFacts":{"correct":usize::from(retained_fact),"total":1},
        "retainedCorrections":{"correct":usize::from(retained_correction),"total":1},"overflowRefusals":overflow_refusals,
        "duplicateEffects":writes.saturating_sub(1),"recoveryFailures":recovery_failures,"modelTransitions":transitions,
        "compactions":snapshot.compactions,"attempts":attempts,"reliabilityClaim":false}));
    assert!(covered && completion && retained_fact && retained_correction && writes == 1 && recovery_failures == 0,
        "Native reliability acceptance failed or lacked required coverage; inspect the recorded measurements");
}
