//! Private, immutable handoffs and results attached to the parent run.
use super::{digest, TaskEventLog};
use crate::subagents::{ChildResult, ChildTaskPacket};
use rusqlite::{params, OptionalExtension};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ChildRecord {
    pub packet: ChildTaskPacket,
    pub result: ChildResult,
    #[serde(default)]
    pub evidence: Vec<crate::knowledge::SearchResult>,
}

fn decode(body: &str, hash: &str) -> Result<ChildRecord, String> {
    if digest(body) != hash {
        return Err("The child record failed its integrity check.".into());
    }
    let record: ChildRecord = serde_json::from_str(body).map_err(|e| e.to_string())?;
    if !record.result.answers(&record.packet) {
        return Err("The child result does not answer its packet.".into());
    }
    Ok(record)
}

impl TaskEventLog {
    pub fn child_result(
        &self,
        run: &str,
        key: &str,
        policy_hash: &str,
    ) -> Result<Option<ChildRecord>, String> {
        let conn = self
            .conn
            .lock()
            .map_err(|_| "The child store is unavailable.")?;
        let row: Option<(String,String,String)> = conn.query_row(
            "SELECT policy_hash,body,body_hash FROM run_child_results WHERE run_id=?1 AND idempotency_key=?2",
            params![run,key], |r| Ok((r.get(0)?,r.get(1)?,r.get(2)?))).optional().map_err(|e| e.to_string())?;
        row.map(|(policy, body, hash)| {
            if policy != policy_hash {
                return Err(
                    "The child's authorization changed; its result cannot be reused.".into(),
                );
            }
            decode(&body, &hash)
        })
        .transpose()
    }

    pub fn save_child_result(&self, record: &ChildRecord, policy_hash: &str) -> Result<(), String> {
        if !record.result.answers(&record.packet) {
            return Err("The child result does not answer its packet.".into());
        }
        let body = serde_json::to_string(record).map_err(|e| e.to_string())?;
        if body.len() > 256 * 1024 {
            return Err("The child record exceeds its storage budget.".into());
        }
        let conn = self
            .conn
            .lock()
            .map_err(|_| "The child store is unavailable.")?;
        conn.execute("INSERT INTO run_child_results (run_id,idempotency_key,policy_hash,body,body_hash) VALUES (?1,?2,?3,?4,?5)",
            params![record.packet.parent_run_id,record.packet.idempotency_key,policy_hash,body,digest(&body)])
            .map_err(|e| e.to_string())?;
        Ok(())
    }

    pub fn children_for_run(&self, run: &str) -> Result<Vec<ChildRecord>, String> {
        let conn = self
            .conn
            .lock()
            .map_err(|_| "The child store is unavailable.")?;
        let mut stmt = conn
            .prepare("SELECT body,body_hash FROM run_child_results WHERE run_id=?1 ORDER BY rowid")
            .map_err(|e| e.to_string())?;
        let rows = stmt
            .query_map([run], |r| {
                Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
            })
            .map_err(|e| e.to_string())?;
        rows.map(|r| {
            let (body, hash) = r.map_err(|e| e.to_string())?;
            decode(&body, &hash)
        })
        .collect()
    }
}
