//! Real gateway/executor/store tests; no fake successful tool implementation.
use super::*;
use crate::agent_runtime::events::{context::ContextCommit, operations::operation_id, EventDraft, TaskEventLog, TaskEventType};
use crate::agent_runtime::resume::{policy_hash, CheckpointSeed};
use crate::policy::Classification;
use chrono::{Duration, Utc};

fn seeded() -> (Arc<RuntimeDeps>, tempfile::TempDir, CheckpointSeed, Value) {
    let (mut deps, dir) = deps_with(signed_in_user());
    Arc::get_mut(&mut deps).unwrap().events = Arc::new(TaskEventLog::open(dir.path()).unwrap());
    deps.events.record(EventDraft::new("r",TaskEventType::RunCreated,"priya").with(json!({"promptShown":"Write note.txt"}))).unwrap();
    deps.events.record(EventDraft::new("r",TaskEventType::RunClassified,"priya").with(json!({"classification":Classification::Internal.label()}))).unwrap();
    let seed = CheckpointSeed {
        attempt_id:"attempt-1".into(), objective:"Write note.txt".into(),conversation_id:"conversation-1".into(),message_id:"message-1".into(),deadline_ms:(Utc::now()+Duration::minutes(2)).timestamp_millis(),
        lease:deps.events.claim_run("r","worker-1",Duration::minutes(2),Utc::now()).unwrap().unwrap(),
        policy_hash:policy_hash(&deps.session().unwrap(),Some(Classification::Internal),&format!("{:?}",crate::sovereignty::global_broker().mode())),
        plan_hash:"plan".into(),workspace_hash:"workspace".into(),model_context: None, model_id:"local".into(),
    };
    deps.checkpoints.lock().unwrap().insert("r".into(),seed.clone());
    let call = json!({"runId":"r","attemptId":"attempt-1","fenceToken":seed.lease.fence_token,"operationSeq":1,"toolCallId":"call-1","tool":"workspace.write_text","args":{"path":"note.txt","content":"persisted result"}});
    let mut request: ContextCommit = serde_json::from_str(include_str!("../../../contracts/runtime-context-v1.json")).unwrap();
    request.run_id = "r".into(); request.fence_token=seed.lease.fence_token;
    request.phase=events::context::ContextPhase::Observed; request.projection=None;
    request.entries[0].message=json!({"role":"assistant","content":[{"type":"toolCall","id":"call-1","name":"workspace.write_text","arguments":call["args"]}]});
    context_api::commit(serde_json::to_value(request).unwrap(),&deps).unwrap();
    (deps,dir,seed,call)
}

#[tokio::test]
async fn a_completed_write_replays_after_reopening_without_another_write_or_step() {
    let (mut deps,dir,mut seed,mut call) = seeded();
    let waiter=tokio::spawn({ let deps=deps.clone();let call=call.clone();async move { authorize(call,&deps).await } });
    let pending = tokio::time::timeout(std::time::Duration::from_secs(5),async {
        loop {
            if let Some(item)=deps.approvals.pending().first().cloned() { break item; }
            tokio::task::yield_now().await;
        }
    }).await.unwrap();
    let reviewer=Session::open(User::new("ravi","Reviewer",vec![Role::Employee]));
    deps.approvals.decide_durable(&reviewer,&pending.request.id,true,None,|_| {
        deps.events.resolve_approval(&pending.request.id,events::ApprovalStatus::Approved,"ravi",None,Utc::now()).map(|saved| assert!(saved))
    }).unwrap();
    call["grant"]=waiter.await.unwrap().unwrap()["grant"].clone();
    let first=execute(call.clone(),&deps).await.unwrap();
    let path=dir.path().join("runs/r/note.txt");
    assert_eq!(std::fs::read_to_string(&path).unwrap(),"persisted result");
    let saved=deps.events.load_context("r").unwrap().unwrap();
    let core=context_api::CoreCheckpoint::from_stored(&saved).unwrap();
    assert_eq!(core.plan.steps_taken,1);
    assert_eq!(core.produced.len(),1);
    assert_eq!(saved.view.raw_seq,1,"no Node tool-result checkpoint was received");
    // This sentinel makes a duplicate write observable even if it uses the
    // same path and content; re-execution would overwrite it.
    std::fs::write(&path,"changed after the acknowledged write").unwrap();
    deps.events.release_claim("r",&seed.lease.owner,seed.lease.fence_token).unwrap();
    Arc::get_mut(&mut deps).unwrap().events=Arc::new(TaskEventLog::open(dir.path()).unwrap());
    seed.lease=deps.events.claim_run("r","worker-2",Duration::minutes(2),Utc::now()).unwrap().unwrap();
    seed.attempt_id="attempt-2".into();
    deps.checkpoints.lock().unwrap().insert("r".into(),seed.clone());
    call["attemptId"]=json!(seed.attempt_id); call["fenceToken"]=json!(seed.lease.fence_token);
    let grant=authorize(call.clone(),&deps).await.unwrap();
    call["grant"]=grant["grant"].clone();
    assert_eq!(execute(call,&deps).await.unwrap(),first);
    assert_eq!(std::fs::read_to_string(&path).unwrap(),"changed after the acknowledged write");
    assert_eq!(deps.plans.lock().unwrap()["r"].checkpoint_progress().steps_taken,1);
    assert_eq!(deps.events.operation("r",&operation_id("r",1,"call-1")).unwrap().unwrap().attempts,1);
}

#[tokio::test]
async fn a_forged_tool_source_cannot_obtain_an_approval_or_grant() {
    let (deps,_dir,_seed,mut call)=seeded();
    call["args"]["content"]=json!("different content");
    assert!(authorize(call,&deps).await.is_err());
    assert!(deps.approvals.pending().is_empty());
}
