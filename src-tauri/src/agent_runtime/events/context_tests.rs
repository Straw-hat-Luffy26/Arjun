use super::{context::ContextCommit, EventDraft, TaskEventLog, TaskEventType};
use crate::agent_runtime::resume::CheckpointSeed;
use chrono::{Duration, Utc};
use serde_json::json;

fn request() -> ContextCommit {
    serde_json::from_str(include_str!("../../../../contracts/runtime-context-v1.json")).unwrap()
}

#[test]
fn startup_does_not_spend_retry_budget_on_waiting_tasks() {
    let log = TaskEventLog::in_memory().unwrap();
    let seed = seed(&log);
    let run = &seed.lease.run_id;
    log.commit_context(&request(), &seed, "operator", json!({}), Utc::now()).unwrap();
    log.record_fenced(EventDraft::new(run, TaskEventType::RunPaused, "operator"), &seed.lease).unwrap();
    log.release_claim(run, &seed.lease.owner, seed.lease.fence_token).unwrap();
    for _ in 0..4 { assert!(log.recover_interrupted(super::SYSTEM_ACTOR).unwrap().is_empty()); }
    let snapshot = log.snapshot(run).unwrap().unwrap();
    assert_eq!(snapshot.state, super::RunState::Paused);
    assert_eq!(snapshot.recovery_attempts, 0);
}

#[test]
fn model_transitions_are_atomic_authority_bound_and_restartable() {
    use super::context::ContextPhase;
    use crate::agent_runtime::model_transition::{ModelContext, TransitionStatus};
    for window in [4096, 32768] {
        let dir = tempfile::tempdir().unwrap();
        let log = TaskEventLog::open(dir.path()).unwrap();
        let mut source = seed(&log);
        source.model_context = Some(ModelContext { model_id: "local".into(), served_model_id: "served-00017".into(), provider: "local".into(), context_window: 8192, max_tokens: 256, input: vec!["text".into()] });
        log.commit_context(&request(), &source, "operator", json!({}), Utc::now()).unwrap();
        let raw = log.context_history(&request().run_id, 0, 512).unwrap();
        let mut target = source.clone();
        target.model_id = "destination-00042".into();
        target.model_context.as_mut().unwrap().model_id = target.model_id.clone();
        target.model_context.as_mut().unwrap().context_window = window;
        let mut boundary = request();
        boundary.expected_revision = 1; boundary.commit_id = "transition-00001".into();
        boundary.entries.clear(); boundary.projection = None;
        assert!(log.commit_context(&boundary, &target, "operator", json!({}), Utc::now()).is_err());
        boundary.phase = ContextPhase::ModelTransitionStarted;
        let mut unauthorized = target.clone(); unauthorized.policy_hash = "changed".into();
        assert!(log.commit_context(&boundary, &unauthorized, "operator", json!({}), Utc::now()).is_err());
        let preparing = log.commit_context(&boundary, &target, "operator", json!({}), Utc::now()).unwrap();
        assert_eq!(preparing.model_context, source.model_context);
        drop(log);
        let log = TaskEventLog::open(dir.path()).unwrap();
        boundary.expected_revision = 2; boundary.commit_id = "retry-transition".into();
        let retry = log.commit_context(&boundary, &target, "operator", json!({}), Utc::now()).unwrap();
        assert_eq!(retry.model_transition.as_ref().unwrap().transition_id, "transition-00001");
        boundary.expected_revision = 3; boundary.commit_id = "ready-transition".into();
        boundary.phase = ContextPhase::ModelTransitionReady;
        assert!(log.commit_context(&boundary, &target, "operator", json!({}), Utc::now()).is_err());
        boundary.projection = request().projection;
        boundary.ledger.as_mut().unwrap().window = window;
        boundary.ledger.as_mut().unwrap().occupied = window;
        assert!(log.commit_context(&boundary, &target, "operator", json!({}), Utc::now()).is_err());
        boundary.ledger.as_mut().unwrap().occupied = 144;
        assert!(log.commit_context(&boundary, &source, "operator", json!({}), Utc::now()).is_err());
        log.conn.lock().unwrap().execute("INSERT INTO run_approvals (approval_id,run_id,tool,args_fingerprint,arguments,status,created_at) VALUES ('approval-00042',?1,'write_scoped_file','exact','{}','pending','2026-09-05T00:00:00Z')", [&boundary.run_id]).unwrap();
        assert!(log.commit_context(&boundary, &target, "operator", json!({}), Utc::now()).is_err(), "pending approvals must settle before destination inference");
        log.conn.lock().unwrap().execute("UPDATE run_approvals SET status='approved' WHERE approval_id='approval-00042'", []).unwrap();
        let ready = log.commit_context(&boundary, &target, "operator", json!({}), Utc::now()).unwrap();
        assert_eq!(ready.model_context, target.model_context);
        assert_eq!(ready.model_transition.unwrap().status, TransitionStatus::Ready);
        assert_eq!(log.checkpoint(&boundary.run_id).unwrap().unwrap().model_id, target.model_id);
        assert_eq!(serde_json::to_value(raw).unwrap(), serde_json::to_value(log.context_history(&boundary.run_id, 0, 512).unwrap()).unwrap());
    }
}

fn seed(log: &TaskEventLog) -> CheckpointSeed {
    let run = request().run_id;
    log.record(EventDraft::new(&run, TaskEventType::RunCreated, "operator")).unwrap();
    CheckpointSeed {
        attempt_id: "attempt-1".into(),
        objective: "Create report.txt".into(), conversation_id: "conversation".into(), message_id: "message".into(), deadline_ms: (Utc::now() + Duration::minutes(10)).timestamp_millis(),
        lease: log.claim_run(&run, "worker", Duration::seconds(60), Utc::now()).unwrap().unwrap(),
        plan_hash: "plan".into(), policy_hash: "policy".into(), workspace_hash: "workspace".into(), model_context: None, model_id: "local".into(),
    }
}

#[test]
fn durable_context_and_raw_history_survive_reopening_and_retry() {
    let dir = tempfile::tempdir().unwrap();
    let log = TaskEventLog::open(dir.path()).unwrap();
    let seed = seed(&log);
    let first = log.commit_context(&request(), &seed, "operator", json!({"stepsTaken": 3}), Utc::now()).unwrap();
    assert_eq!(first.revision, 1);
    let again = log.commit_context(&request(), &seed, "operator", json!({"stepsTaken": 3}), Utc::now()).unwrap();
    assert_eq!(again.revision, 1);
    drop(log);
    let reopened = TaskEventLog::open(dir.path()).unwrap();
    let saved = reopened.load_context(&request().run_id).unwrap().unwrap();
    assert_eq!(saved.view.notes.next_action, "Read source A-17");
    assert_eq!(saved.core_state["stepsTaken"], 3);
    assert_eq!(reopened.context_history(&request().run_id, 0, 20).unwrap().len(), 1);
    assert!(reopened.checkpoint(&request().run_id).unwrap().unwrap().is_intact());
}

#[test]
fn stale_context_writers_cannot_advance_after_lease_takeover() {
    let log = TaskEventLog::in_memory().unwrap();
    let seed = seed(&log);
    log.commit_context(&request(), &seed, "operator", json!({}), Utc::now()).unwrap();
    let later = Utc::now() + Duration::seconds(61);
    log.claim_run(&request().run_id, "new-worker", Duration::seconds(60), later).unwrap().unwrap();
    let mut next = request(); next.expected_revision = 1; next.commit_id = "commit-2".into();
    assert!(log.commit_context(&next, &seed, "operator", json!({}), later).is_err());
    assert_eq!(log.load_context(&next.run_id).unwrap().unwrap().view.revision, 1);
}

#[test]
fn raw_entries_are_append_only_and_identifier_conflicts_roll_back_the_batch() {
    let log = TaskEventLog::in_memory().unwrap();
    let seed = seed(&log);
    log.commit_context(&request(), &seed, "operator", json!({}), Utc::now()).unwrap();
    let mut next = request(); next.expected_revision = 1; next.commit_id = "commit-2".into();
    next.entries[0].message = json!({"role": "user", "content": "replacement"});
    assert!(log.commit_context(&next, &seed, "operator", json!({}), Utc::now()).is_err());
    assert_eq!(log.load_context(&next.run_id).unwrap().unwrap().view.revision, 1);
    let conn = log.conn.lock().unwrap();
    assert!(conn.execute("DELETE FROM run_context_messages", []).is_err());
    assert!(conn.execute("UPDATE run_context_commits SET body = '{}'", []).is_err());
}

#[test]
fn orphan_tool_results_cannot_be_committed_as_a_model_projection() {
    let log = TaskEventLog::in_memory().unwrap();
    let seed = seed(&log);
    let mut next = request();
    next.projection = Some(vec![json!({"role": "toolResult", "toolCallId": "missing", "content": []})]);
    assert!(log.commit_context(&next, &seed, "operator", json!({}), Utc::now()).is_err());
    assert!(log.load_context(&next.run_id).unwrap().is_none());
}

fn action(log: &TaskEventLog, seed: &CheckpointSeed, seq: i64, revision: i64) -> super::operations::Operation {
    use super::context::{ContextEntry, ContextPhase};
    let mut request = request();
    request.expected_revision = revision;
    request.commit_id = format!("action-{seq}");
    request.phase = ContextPhase::Observed;
    request.projection = None;
    request.entries = vec![ContextEntry { entry_id: format!("assistant-{seq}"), message: json!({
        "role":"assistant", "content":[{"type":"toolCall","id":"call-1","name":"workspace.write_text","arguments":{"path":"result.txt","content":"data"}}]
    }) }];
    let view = log.commit_context(&request,seed,"operator",json!({"stepsTaken":0}),Utc::now()).unwrap();
    assert_eq!(view.raw_seq,seq);
    log.propose_operation(&seed.lease,"operator",seq,"call-1",crate::orchestrator::tools::ToolName::WriteScopedFile,&json!({"path":"result.txt","content":"data"})).unwrap()
}

#[test]
fn a_full_tool_receipt_and_core_state_survive_a_crash_before_the_next_context_checkpoint() {
    use super::operations::ToolReceipt;
    let dir = tempfile::tempdir().unwrap();
    let log = TaskEventLog::open(dir.path()).unwrap();
    let mut seed = seed(&log);
    let op = action(&log,&seed,1,0);
    assert!(log.start_operation(&seed.lease,&op.id).unwrap().is_none());
    let text = "source-123 https://example.invalid/A-17\n".repeat(1000);
    let receipt = ToolReceipt::Result(json!({"text":text,"details":{"artifactId":"A-17"}}));
    log.finish_operation(&seed.lease,"operator",&op.id,&receipt,&json!({"stepsTaken":1,"produced":["result.txt"]})).unwrap();
    log.release_claim(&seed.lease.run_id,&seed.lease.owner,seed.lease.fence_token).unwrap();
    drop(log);
    let reopened = TaskEventLog::open(dir.path()).unwrap();
    seed.lease = reopened.claim_run(&seed.lease.run_id,"new-worker",Duration::minutes(2),Utc::now()).unwrap().unwrap();
    let replay = reopened.start_operation(&seed.lease,&op.id).unwrap().unwrap().into_result().unwrap();
    assert_eq!(replay["text"],text);
    assert_eq!(replay["details"]["artifactId"],"A-17");
    assert_eq!(reopened.operation(&seed.lease.run_id,&op.id).unwrap().unwrap().attempts,1);
    let saved = reopened.load_context(&seed.lease.run_id).unwrap().unwrap();
    assert_eq!(saved.core_state["stepsTaken"],1);
    assert_eq!(saved.view.raw_seq,1, "the worker never acknowledged the tool result");
}

#[test]
fn intentionally_repeating_identical_arguments_creates_a_new_operation() {
    let log = TaskEventLog::in_memory().unwrap();
    let seed = seed(&log);
    let first = action(&log,&seed,1,0);
    let second = action(&log,&seed,2,1);
    assert_ne!(first.id,second.id);
    assert_eq!(first.arguments,second.arguments);
    assert!(log.propose_operation(&seed.lease,"operator",2,"call-1",crate::orchestrator::tools::ToolName::WriteScopedFile,&json!({"path":"result.txt","content":"modified"})).is_err());
}

#[test]
fn a_failed_receipt_transaction_preserves_uncertainty_and_rejects_a_stale_writer() {
    let log = TaskEventLog::in_memory().unwrap();
    let seed = seed(&log);
    let op = action(&log,&seed,1,0);
    log.start_operation(&seed.lease,&op.id).unwrap();
    log.conn.lock().unwrap().execute_batch("CREATE TRIGGER fail_result BEFORE UPDATE OF result ON run_tool_operations BEGIN SELECT RAISE(ABORT,'injected disk failure'); END;").unwrap();
    let receipt = super::operations::ToolReceipt::Result(json!({"text":"written"}));
    assert!(log.finish_operation(&seed.lease,"operator",&op.id,&receipt,&json!({"stepsTaken":1})).is_err());
    assert_eq!(log.operation(&seed.lease.run_id,&op.id).unwrap().unwrap().status,"running");
    assert_eq!(log.load_context(&seed.lease.run_id).unwrap().unwrap().core_state["stepsTaken"],0);
    log.release_claim(&seed.lease.run_id,&seed.lease.owner,seed.lease.fence_token).unwrap();
    let new = log.claim_run(&seed.lease.run_id,"replacement",Duration::minutes(1),Utc::now()).unwrap().unwrap();
    assert!(log.finish_operation(&seed.lease,"operator",&op.id,&receipt,&json!({})).is_err());
    assert!(log.start_operation(&new,&op.id).is_err(),"an uncertain write cannot be retried");
}
