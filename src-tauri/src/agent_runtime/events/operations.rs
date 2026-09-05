//! Private, complete tool receipts keyed by a canonical assistant action.
//! Transport retries keep the same key; a later identical action gets a new
//! message sequence and therefore a different key. The context store retains
//! the original arguments and this store retains the exact execution response.
use chrono::Utc;
use rusqlite::{params, Connection, OptionalExtension, Transaction, TransactionBehavior};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use super::{digest, lease, payload_hash, EventDraft, Lease, TaskEventLog, TaskEventType, SCHEMA_VERSION};
use crate::agent_runtime::{protocol::WireError, tool_policy::{class_of, retry_policy_of}};
use crate::orchestrator::tools::ToolName;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", tag = "kind", content = "value")]
pub enum ToolReceipt { Result(Value), Error(WireError) }

impl ToolReceipt {
    pub fn from_result(result: &Result<Value, WireError>) -> Self {
        match result { Ok(value) => Self::Result(value.clone()), Err(error) => Self::Error(error.clone()) }
    }
    pub fn into_result(self) -> Result<Value, WireError> {
        match self { Self::Result(value) => Ok(value), Self::Error(error) => Err(error) }
    }
}

#[derive(Debug, Clone)]
pub struct Operation {
    pub id: String,
    pub message_seq: i64,
    pub call_id: String,
    pub tool: String,
    pub arguments: Value,
    pub status: String,
    pub fence_token: i64,
    pub attempts: u32,
    pub receipt: Option<ToolReceipt>,
}

pub fn operation_id(run: &str, message_seq: i64, call: &str) -> String {
    digest(&json!(["operation.v1", run, message_seq, call]).to_string())
}

fn error(_: rusqlite::Error) -> String { "The durable tool operation store is unavailable.".into() }
fn encode(value: &impl Serialize) -> Result<String, String> { serde_json::to_string(value).map_err(|_| "The tool receipt could not be serialized.".into()) }
fn assert_lease(conn: &Connection, claim: &Lease) -> Result<(), String> {
    if !lease::holder(conn, &claim.run_id, Utc::now()).map_err(error)?
        .is_some_and(|held| held.owner == claim.owner && held.fence_token == claim.fence_token) {
        return Err("The tool operation belongs to an expired execution attempt.".into());
    }
    if super::store::ending_of(conn, &claim.run_id).map_err(error)?.is_some() {
        return Err("This run has already ended.".into());
    }
    Ok(())
}

fn get(conn: &Connection, run: &str, id: &str) -> Result<Option<Operation>, String> {
    let row = conn.query_row(
        "SELECT message_seq,tool_call_id,tool,arguments,args_hash,status,fence_token,attempts,result,result_hash FROM run_tool_operations WHERE run_id=?1 AND operation_id=?2",
        params![run,id], |row| Ok((row.get::<_,i64>(0)?,row.get::<_,String>(1)?,row.get::<_,String>(2)?,row.get::<_,String>(3)?,row.get::<_,String>(4)?,row.get::<_,String>(5)?,row.get::<_,i64>(6)?,row.get::<_,u32>(7)?,row.get::<_,Option<String>>(8)?,row.get::<_,Option<String>>(9)?)),
    ).optional().map_err(error)?;
    row.map(|(message_seq,call_id,tool,args,hash,status,fence_token,attempts,result,result_hash)| {
        let arguments: Value = serde_json::from_str(&args).map_err(|_| "The operation arguments are unreadable.")?;
        if payload_hash(&arguments) != hash { return Err("The operation arguments failed their integrity check.".into()); }
        let receipt = match (result, result_hash) {
            (Some(body),Some(hash)) if digest(&body) == hash => Some(serde_json::from_str(&body).map_err(|_| "The tool receipt is unreadable.")?),
            (None,None) if status == "proposed" || status == "running" || status == "unknown" => None,
            _ => return Err("The tool receipt failed its integrity check.".into()),
        };
        Ok(Operation { id:id.into(),message_seq,call_id,tool,arguments,status,fence_token,attempts,receipt })
    }).transpose()
}

fn record(conn: &Connection, run: &str, actor: &str, kind: TaskEventType, payload: Value) -> Result<(), String> {
    let draft = EventDraft::new(run,kind,actor).with(payload);
    conn.execute("INSERT INTO task_events (event_id,run_id,seq,event_type,at,actor,schema_version,payload,payload_hash) VALUES (?1,?2,(SELECT COALESCE(MAX(seq),0)+1 FROM task_events WHERE run_id=?2),?3,?4,?5,?6,?7,?8)",
        params![draft.event_id,run,kind.as_str(),Utc::now().to_rfc3339(),actor,SCHEMA_VERSION,encode(&draft.payload)?,payload_hash(&draft.payload)]).map_err(error)?;
    Ok(())
}

impl TaskEventLog {
    /// Called before authorization. Arguments must match an acknowledged raw
    /// assistant request; arbitrary caller-supplied idempotency keys are ignored.
    pub fn propose_operation(&self, claim: &Lease, actor: &str, message_seq: i64, call_id: &str, tool: ToolName, args: &Value) -> Result<Operation, String> {
        let conn = self.conn.lock().map_err(|_| "The operation store is unavailable.")?;
        let tx = Transaction::new_unchecked(&conn,TransactionBehavior::Immediate).map_err(error)?;
        assert_lease(&tx,claim)?;
        let (body,hash): (String,String) = tx.query_row("SELECT body,body_hash FROM run_context_messages WHERE run_id=?1 AND seq=?2",params![claim.run_id,message_seq],|row| Ok((row.get(0)?,row.get(1)?))).map_err(error)?;
        if digest(&body) != hash { return Err("The tool's source message failed its integrity check.".into()); }
        let message: Value = serde_json::from_str(&body).map_err(|_| "The tool's source message is unreadable.")?;
        let matches = message["role"] == "assistant" && message["content"].as_array().is_some_and(|blocks| blocks.iter().any(|block| {
            block["type"] == "toolCall" && block["id"].as_str() == Some(call_id)
                && block["name"].as_str().and_then(ToolName::from_str) == Some(tool)
                && block.get("arguments") == Some(args)
        }));
        if !matches { return Err("The tool call does not match its durable assistant request.".into()); }
        let id = operation_id(&claim.run_id,message_seq,call_id);
        if let Some(existing) = get(&tx,&claim.run_id,&id)? {
            if existing.tool != tool.as_str() || existing.arguments != *args { return Err("An operation id was reused for another action.".into()); }
            return Ok(existing);
        }
        let revision: i64 = tx.query_row("SELECT MAX(revision) FROM run_context_commits WHERE run_id=?1",[&claim.run_id],|row| row.get(0)).map_err(error)?;
        tx.execute("INSERT INTO run_tool_operations (run_id,operation_id,message_seq,tool_call_id,tool,arguments,args_hash,class,status,fence_token,base_revision,created_at,updated_at) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,'proposed',?9,?10,?11,?11)",
            params![claim.run_id,id,message_seq,call_id,tool.as_str(),encode(args)?,payload_hash(args),class_of(tool).as_str(),claim.fence_token,revision,Utc::now().to_rfc3339()]).map_err(error)?;
        record(&tx,&claim.run_id,actor,TaskEventType::ToolEffectPending,json!({"operationId":id,"toolCallId":call_id,"tool":tool.as_str(),"phase":"intent_created","class":class_of(tool).as_str()}))?;
        let operation = get(&tx,&claim.run_id,&id)?.ok_or("The operation was not recorded.")?;
        tx.commit().map_err(error)?;
        Ok(operation)
    }

    /// Serializes durable tools and rejects a stale worker inside the same
    /// transaction as dispatch intent. Only reads with a declared retry policy
    /// may be retried after an expired worker, with persisted attempt counts.
    pub fn start_operation(&self, claim: &Lease, id: &str) -> Result<Option<ToolReceipt>, String> {
        let conn = self.conn.lock().map_err(|_| "The operation store is unavailable.")?;
        let tx = Transaction::new_unchecked(&conn,TransactionBehavior::Immediate).map_err(error)?;
        assert_lease(&tx,claim)?;
        let held = get(&tx,&claim.run_id,id)?.ok_or("No durable operation intent exists.")?;
        if let Some(receipt) = held.receipt { return Ok(Some(receipt)); }
        let tool = ToolName::from_str(&held.tool).ok_or("The saved operation uses an unknown tool.")?;
        let policy = retry_policy_of(tool);
        if held.status != "proposed" && !(held.status == "running" && held.fence_token != claim.fence_token && policy.safe_to_retry && held.attempts <= u32::from(policy.max_retries)) {
            return Err("This operation is in flight or uncertain and requires reconciliation.".into());
        }
        tx.execute("UPDATE run_tool_operations SET status='running',fence_token=?3,attempts=attempts+1,updated_at=?4 WHERE run_id=?1 AND operation_id=?2",params![claim.run_id,id,claim.fence_token,Utc::now().to_rfc3339()]).map_err(error)?;
        tx.commit().map_err(error)?;
        Ok(None)
    }

    /// Full response, response digest and Rust resources land together. Losing
    /// the worker's subsequent context acknowledgment cannot lose this result.
    pub fn finish_operation(&self, claim: &Lease, actor: &str, id: &str, receipt: &ToolReceipt, core: &Value) -> Result<(), String> {
        self.settle_operation(claim,actor,id,receipt,core,"running")
    }

    pub fn decline_operation(&self, claim: &Lease, actor: &str, id: &str, refusal: WireError, core: &Value) -> Result<(), String> {
        self.settle_operation(claim,actor,id,&ToolReceipt::Error(refusal),core,"proposed")
    }

    fn settle_operation(&self, claim: &Lease, actor: &str, id: &str, receipt: &ToolReceipt, core: &Value, expected: &str) -> Result<(), String> {
        let result = encode(receipt)?;
        let core = encode(core)?;
        if result.len() > 32*1024*1024 || core.len() > 32*1024*1024 { return Err("The durable tool receipt exceeds its storage bound.".into()); }
        let conn = self.conn.lock().map_err(|_| "The operation store is unavailable.")?;
        let tx = Transaction::new_unchecked(&conn,TransactionBehavior::Immediate).map_err(error)?;
        assert_lease(&tx,claim)?;
        let status = if matches!(receipt,ToolReceipt::Result(_)) { "succeeded" } else { "failed" };
        if tx.execute("UPDATE run_tool_operations SET status=?4,result=?5,result_hash=?6,core_state=?7,core_hash=?8,updated_at=?9 WHERE run_id=?1 AND operation_id=?2 AND (fence_token=?3 OR ?10='proposed') AND status=?10",
            params![claim.run_id,id,claim.fence_token,status,result,digest(&result),core,digest(&core),Utc::now().to_rfc3339(),expected]).map_err(error)? != 1 { return Err("The tool result no longer belongs to this execution attempt.".into()); }
        record(&tx,&claim.run_id,actor,TaskEventType::CheckpointTaken,json!({"operationId":id,"phase":"tool_result_committed","status":status,"responseDigest":digest(&result)}))?;
        tx.commit().map_err(error)?;
        Ok(())
    }

    pub fn operation(&self, run: &str, id: &str) -> Result<Option<Operation>, String> {
        let conn = self.conn.lock().map_err(|_| "The operation store is unavailable.")?;
        get(&conn,run,id)
    }
}

/// Overlay only newer authoritative tool resources. Completed tools execute
/// sequentially, so the latest snapshot includes all preceding receipts.
pub(super) fn latest_core(conn: &Connection, run: &str, revision: i64) -> Result<Option<Value>, String> {
    let row: Option<(String,String)> = conn.query_row("SELECT core_state,core_hash FROM run_tool_operations WHERE run_id=?1 AND base_revision>=?2 AND status IN ('succeeded','failed') ORDER BY rowid DESC LIMIT 1",params![run,revision],|row| Ok((row.get(0)?,row.get(1)?))).optional().map_err(error)?;
    row.map(|(body,hash)| {
        if digest(&body) != hash { return Err("The saved tool resources failed their integrity check.".into()); }
        serde_json::from_str(&body).map_err(|_| "The saved tool resources are unreadable.".into())
    }).transpose()
}
