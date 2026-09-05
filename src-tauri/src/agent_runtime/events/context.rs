//! Private transcript bodies and bounded, versioned context projections.
//!
//! Unlike the redacted task event feed, this data has the same sensitivity as
//! the conversation. Only the owning signed-in run may read it through the
//! runtime boundary. Never publish a body or core_state on the UI event bus.
use std::collections::{BTreeMap, HashSet};

use chrono::{DateTime, Utc};
use rusqlite::{params, Connection, OptionalExtension, Transaction, TransactionBehavior};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use super::{digest, lease, payload_hash, EventDraft, RunCheckpoint, RunState, TaskEventLog, TaskEventType, SCHEMA_VERSION};
use crate::agent_runtime::{memory::RunMemory, resume::CheckpointSeed, tasks::ContextLedgerRecord};

pub const CONTEXT_PROTOCOL_VERSION: u32 = 1;
const MAX_BATCH_BYTES: usize = 32 * 1024 * 1024;
const MAX_CORE_BYTES: usize = 32 * 1024 * 1024;
const MAX_PROJECTION_BYTES: usize = 512 * 1024;
const MAX_NOTES_BYTES: usize = 64 * 1024;
const MAX_COMMITS: i64 = 4096;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum ContextPhase { Observed, ModelReady, BeforeTool, AfterTool, CompactionStarted, CompactionCompleted, Finished, Paused }

impl ContextPhase {
    fn event(self) -> TaskEventType {
        match self {
            Self::ModelReady => TaskEventType::ModelRequested,
            Self::CompactionStarted => TaskEventType::CompactionStarted,
            Self::CompactionCompleted => TaskEventType::ContextCompacted,
            _ => TaskEventType::CheckpointTaken,
        }
    }
    fn state(self) -> RunState {
        match self {
            Self::CompactionStarted => RunState::Compacting,
            Self::AfterTool => RunState::ToolResultRecorded,
            Self::Paused => RunState::Paused,
            _ => RunState::Running,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ContextEntry { pub entry_id: String, pub message: Value }

/// The nested runtime ledger is distinct from the flat historical task record.
/// Keep this wire shape shared with ContextLedgerSnapshot and the JSON fixture.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ContextLedgerWire {
    pub sections: BTreeMap<String, u32>,
    pub occupied: u32,
    pub committed: u32,
    pub window: u32,
    pub headroom: i64,
    pub compactions: u32,
    pub entities: Vec<Value>,
    pub reconciliations: Vec<Value>,
    pub itemisation_errors: Vec<Value>,
}

impl ContextLedgerWire {
    pub fn record(&self) -> ContextLedgerRecord {
        let count = |section: &str| self.sections.get(section).copied().unwrap_or(0);
        ContextLedgerRecord { system: count("system"), skill: count("skill"), tool_schema: count("toolSchema"),
            evidence: count("evidence"), notes: count("notes"), transcript: count("transcript"), compaction: count("compaction"),
            reserve: count("reserve"), occupied: self.occupied, committed: self.committed, window: self.window, headroom: self.headroom }
    }
}

/// Canonical fixture: contracts/runtime-context-v1.json. Identity and compare-
/// and-swap are mandatory, including on a request that only appends messages.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ContextCommit {
    pub protocol_version: u32,
    pub run_id: String,
    pub attempt_id: String,
    pub fence_token: i64,
    pub expected_revision: i64,
    pub commit_id: String,
    pub phase: ContextPhase,
    pub entries: Vec<ContextEntry>,
    pub projection: Option<Vec<Value>>,
    pub notes: RunMemory,
    pub ledger: Option<ContextLedgerWire>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ContextView {
    pub protocol_version: u32,
    pub run_id: String,
    pub revision: i64,
    pub checkpoint_id: String,
    pub raw_seq: i64,
    /// Raw messages through this cursor are already represented in `messages`.
    pub projection_seq: i64,
    pub phase: ContextPhase,
    pub messages: Vec<Value>,
    pub notes: RunMemory,
    pub ledger: Option<ContextLedgerWire>,
    pub pending_approvals: Vec<String>,
    pub unsettled_effects: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StoredContext {
    pub view: ContextView,
    pub checkpoint: RunCheckpoint,
    /// Rust-owned resource/plan snapshot. Never accepted from the model worker.
    pub core_state: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TranscriptEntry { pub seq: i64, pub entry_id: String, pub message: Value }

fn text_id(value: &str) -> bool { !value.is_empty() && value.len() <= 256 && !value.chars().any(char::is_control) }
fn storage_error(_: rusqlite::Error) -> String { "The durable context store could not commit or read this boundary.".into() }
fn encoded<T: Serialize>(value: &T) -> Result<String, String> { serde_json::to_string(value).map_err(|_| "The context record is not serializable.".into()) }

fn valid_message(message: &Value) -> bool {
    matches!(message.get("role").and_then(Value::as_str), Some("user" | "assistant" | "toolResult" | "custom" | "compactionSummary" | "branchSummary"))
}

/// A projection is provider-valid only if each tool result has one matching
/// preceding call, with no unfinished batch across a user/assistant boundary.
fn validate_projection(messages: &[Value]) -> Result<(), String> {
    if encoded(&messages)?.len() > MAX_PROJECTION_BYTES { return Err("The context projection exceeds its storage bound.".into()); }
    let mut pending = HashSet::new();
    for message in messages {
        if !valid_message(message) { return Err("The context projection contains an unknown message role.".into()); }
        match message["role"].as_str().unwrap_or_default() {
            "toolResult" => {
                let id = message["toolCallId"].as_str().unwrap_or_default();
                if !pending.remove(id) { return Err("The context projection contains an orphan or duplicate tool result.".into()); }
            }
            "assistant" => {
                if !pending.is_empty() { return Err("The context projection splits a tool batch.".into()); }
                if let Some(blocks) = message["content"].as_array() {
                    for block in blocks.iter().filter(|block| block["type"] == "toolCall") {
                        let id = block["id"].as_str().unwrap_or_default();
                        if !text_id(id) || !pending.insert(id.to_string()) { return Err("The context projection contains an invalid tool call id.".into()); }
                    }
                }
            }
            _ if !pending.is_empty() => return Err("The context projection interrupts a tool batch.".into()),
            _ => {}
        }
    }
    if !pending.is_empty() { return Err("The context projection has tool calls without results.".into()); }
    Ok(())
}

fn decode(body: &str, hash: &str) -> Result<StoredContext, String> {
    if digest(body) != hash { return Err("The durable context checkpoint failed its integrity check.".into()); }
    let record: StoredContext = serde_json::from_str(body).map_err(|_| "The durable context checkpoint is unreadable.")?;
    if record.view.protocol_version != CONTEXT_PROTOCOL_VERSION || !record.checkpoint.is_intact() {
        return Err("The durable context checkpoint has an unsupported version or invalid seal.".into());
    }
    validate_projection(&record.view.messages)?;
    Ok(record)
}

fn load(conn: &Connection, run: &str) -> Result<Option<StoredContext>, String> {
    let row: Option<(String, String)> = conn.query_row(
        "SELECT body, body_hash FROM run_context_commits WHERE run_id = ?1 ORDER BY revision DESC LIMIT 1",
        [run], |row| Ok((row.get(0)?, row.get(1)?)),
    ).optional().map_err(storage_error)?;
    row.map(|(body, hash)| decode(&body, &hash)).transpose()
}

impl TaskEventLog {
    /// Commit raw entries, the successor projection, structured state and a
    /// checkpoint in ONE transaction. A failed commit authorizes no next step.
    pub fn commit_context(&self, request: &ContextCommit, seed: &CheckpointSeed, actor: &str, core_state: Value, now: DateTime<Utc>) -> Result<ContextView, String> {
        if request.protocol_version != CONTEXT_PROTOCOL_VERSION || request.run_id != seed.lease.run_id
            || request.attempt_id != seed.attempt_id || request.fence_token != seed.lease.fence_token
            || !text_id(&request.commit_id) || request.expected_revision < 0 || request.entries.len() > 512 {
            return Err("The context boundary has an invalid version, identity, revision or size.".into());
        }
        if encoded(request)?.len() > MAX_BATCH_BYTES || encoded(&core_state)?.len() > MAX_CORE_BYTES || encoded(&request.notes)?.len() > MAX_NOTES_BYTES {
            return Err("The context boundary exceeds its durable storage limit.".into());
        }
        if request.phase == ContextPhase::ModelReady && request.projection.is_none() { return Err("A model request requires a durable projection.".into()); }
        if let Some(projection) = &request.projection { validate_projection(projection)?; }
        let request_hash = payload_hash(&serde_json::to_value(request).map_err(|_| "Invalid context request.")?);
        let conn = self.conn.lock().map_err(|_| "The durable context store is unavailable.")?;
        let tx = Transaction::new_unchecked(&conn, TransactionBehavior::Immediate).map_err(storage_error)?;
        let held = lease::holder(&tx, &request.run_id, now).map_err(storage_error)?;
        if !held.is_some_and(|held| held.owner == seed.lease.owner && held.fence_token == seed.lease.fence_token) {
            return Err("This attempt lost its execution lease and may not advance the task.".into());
        }
        let owner: Option<String> = tx.query_row("SELECT actor FROM task_events WHERE run_id = ?1 AND event_type = 'run_created' ORDER BY seq LIMIT 1", [&request.run_id], |row| row.get(0)).optional().map_err(storage_error)?;
        if owner.as_deref() != Some(actor) { return Err("This context does not belong to the active operator.".into()); }
        if super::store::ending_of(&tx, &request.run_id).map_err(storage_error)?.is_some() { return Err("This run already ended; its context cannot advance.".into()); }
        let duplicate: Option<(String, String, String)> = tx.query_row(
            "SELECT request_hash, body, body_hash FROM run_context_commits WHERE run_id = ?1 AND commit_id = ?2",
            params![request.run_id, request.commit_id], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        ).optional().map_err(storage_error)?;
        if let Some((hash, body, body_hash)) = duplicate {
            if hash != request_hash { return Err("A context commit id was reused for different content.".into()); }
            return Ok(decode(&body, &body_hash)?.view);
        }
        let previous = load(&tx, &request.run_id)?;
        // Human milestone signatures are Rust-owned. A worker cannot erase or
        // manufacture them by submitting a new copy of its working notes.
        if !request.notes.milestones.is_empty() { return Err("The worker cannot submit approval milestones.".into()); }
        let held_checkpoint: Option<String> = tx.query_row("SELECT body FROM run_checkpoints WHERE run_id = ?1", [&request.run_id], |row| row.get(0)).optional().map_err(storage_error)?;
        let mut notes = request.notes.clone();
        if let Some(body) = held_checkpoint {
            let checkpoint: RunCheckpoint = serde_json::from_str(&body).map_err(|_| "The prior checkpoint is unreadable.")?;
            if !checkpoint.is_intact() { return Err("The prior checkpoint failed its integrity check.".into()); }
            notes.milestones = checkpoint.notes.milestones;
        }
        let revision = previous.as_ref().map(|saved| saved.view.revision).unwrap_or(0);
        if revision != request.expected_revision || revision >= MAX_COMMITS { return Err("The context revision changed or the task exhausted its checkpoint budget.".into()); }
        let mut raw_seq = previous.as_ref().map(|saved| saved.view.raw_seq).unwrap_or(0);
        for entry in &request.entries {
            if !text_id(&entry.entry_id) || !valid_message(&entry.message) { return Err("The transcript entry is malformed.".into()); }
            let body = encoded(&entry.message)?;
            let hash = digest(&body);
            let existing: Option<String> = tx.query_row("SELECT body_hash FROM run_context_messages WHERE run_id = ?1 AND entry_id = ?2", params![request.run_id, entry.entry_id], |row| row.get(0)).optional().map_err(storage_error)?;
            if let Some(existing) = existing {
                if existing != hash { return Err("A transcript entry id was reused with different content.".into()); }
                continue;
            }
            raw_seq += 1;
            tx.execute("INSERT INTO run_context_messages (run_id,seq,entry_id,body,body_hash,at) VALUES (?1,?2,?3,?4,?5,?6)", params![request.run_id, raw_seq, entry.entry_id, body, hash, now.to_rfc3339()]).map_err(storage_error)?;
        }
        let keys = |sql: &str| -> Result<Vec<String>, String> {
            let mut statement = tx.prepare(sql).map_err(storage_error)?;
            let rows = statement.query_map([&request.run_id], |row| row.get(0)).map_err(storage_error)?;
            rows.collect::<rusqlite::Result<Vec<String>>>().map_err(storage_error)
        };
        let unsettled = keys("SELECT idempotency_key FROM task_tool_effects WHERE run_id = ?1 AND status IN ('pending','unknown') ORDER BY idempotency_key")?;
        let approvals = keys("SELECT approval_id FROM run_approvals WHERE run_id = ?1 AND status = 'pending' ORDER BY approval_id")?;
        let (messages, projection_seq) = match &request.projection {
            Some(projection) => (projection.clone(), raw_seq),
            None => previous.as_ref().map(|saved| (saved.view.messages.clone(), saved.view.projection_seq)).unwrap_or_default(),
        };
        let view = ContextView { protocol_version: CONTEXT_PROTOCOL_VERSION, run_id: request.run_id.clone(), revision: revision + 1, checkpoint_id: request.commit_id.clone(), raw_seq, projection_seq, phase: request.phase, messages, notes: notes.clone(), ledger: request.ledger.clone(), pending_approvals: approvals, unsettled_effects: unsettled.clone() };
        let event_seq: i64 = tx.query_row("SELECT COALESCE(MAX(seq),0)+1 FROM task_events WHERE run_id = ?1", [&request.run_id], |row| row.get(0)).map_err(storage_error)?;
        let checkpoint = seed.checkpoint(&request.run_id, request.phase.state(), event_seq, notes, request.ledger.as_ref().map(ContextLedgerWire::record), unsettled);
        let stored = StoredContext { view: view.clone(), checkpoint: checkpoint.clone(), core_state };
        let body = encoded(&stored)?;
        tx.execute("INSERT INTO run_context_commits (run_id,revision,commit_id,request_hash,fence_token,body,body_hash,at) VALUES (?1,?2,?3,?4,?5,?6,?7,?8)", params![request.run_id, view.revision, request.commit_id, request_hash, request.fence_token, body, digest(&body), now.to_rfc3339()]).map_err(storage_error)?;
        let draft = EventDraft::new(&request.run_id, request.phase.event(), actor).with(json!({ "contextRevision": view.revision, "checkpointId": view.checkpoint_id, "rawSeq": raw_seq, "phase": request.phase, "messages": request.entries.len() }));
        tx.execute("INSERT INTO task_events (event_id,run_id,seq,event_type,at,actor,schema_version,payload,payload_hash) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9)", params![draft.event_id, request.run_id, event_seq, draft.event_type.as_str(), now.to_rfc3339(), actor, SCHEMA_VERSION, encoded(&draft.payload)?, payload_hash(&draft.payload)]).map_err(storage_error)?;
        tx.execute("INSERT INTO run_checkpoints (run_id,attempt_id,last_event_seq,state,at,schema_version,checkpoint_hash,body) VALUES (?1,?2,?3,?4,?5,?6,?7,?8) ON CONFLICT(run_id) DO UPDATE SET attempt_id=excluded.attempt_id,last_event_seq=excluded.last_event_seq,state=excluded.state,at=excluded.at,schema_version=excluded.schema_version,checkpoint_hash=excluded.checkpoint_hash,body=excluded.body", params![request.run_id, request.attempt_id, event_seq, checkpoint.state.as_str(), checkpoint.at, checkpoint.schema_version, checkpoint.checkpoint_hash, encoded(&checkpoint)?]).map_err(storage_error)?;
        tx.commit().map_err(storage_error)?;
        Ok(view)
    }

    pub fn load_context(&self, run_id: &str) -> Result<Option<StoredContext>, String> {
        let conn = self.conn.lock().map_err(|_| "The durable context store is unavailable.")?;
        let mut saved = load(&conn, run_id)?;
        if let Some(saved) = &mut saved {
            if let Some(core) = super::operations::latest_core(&conn, run_id, saved.view.revision)? {
                saved.core_state = core;
            }
        }
        Ok(saved)
    }

    /// Paged, exact retrieval. Callers enforce run ownership before exposing it.
    pub fn context_history(&self, run_id: &str, after_seq: i64, limit: u32) -> Result<Vec<TranscriptEntry>, String> {
        let conn = self.conn.lock().map_err(|_| "The durable context store is unavailable.")?;
        let mut statement = conn.prepare("SELECT seq,entry_id,body,body_hash FROM run_context_messages WHERE run_id = ?1 AND seq > ?2 ORDER BY seq LIMIT ?3").map_err(storage_error)?;
        let rows = statement.query_map(params![run_id, after_seq.max(0), limit.clamp(1,512)], |row| Ok((row.get::<_,i64>(0)?,row.get::<_,String>(1)?,row.get::<_,String>(2)?,row.get::<_,String>(3)?))).map_err(storage_error)?;
        let mut result = Vec::new();
        let mut bytes = 0;
        for row in rows {
            let (seq, entry_id, body, hash) = row.map_err(storage_error)?;
            bytes += body.len();
            if bytes > MAX_BATCH_BYTES { return Err("The transcript page exceeds its read budget; request fewer entries.".into()); }
            if digest(&body) != hash { return Err("A transcript entry failed its integrity check.".into()); }
            result.push(TranscriptEntry { seq, entry_id, message: serde_json::from_str(&body).map_err(|_| "A transcript entry is unreadable.")? });
        }
        Ok(result)
    }
}
