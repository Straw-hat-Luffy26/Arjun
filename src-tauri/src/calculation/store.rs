//! Where calculation records are kept: append-only, one row per content
//! address, checked against its own hash on every read.
//!
//! Nothing here updates or deletes a record. A corrected input makes a new
//! record with a new id; the old one stays, and its standing in the shared
//! memory graph is what says it has gone stale. Checks are appended beside
//! the records, never folded into them.

use std::path::Path;
use std::sync::Mutex;

use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};

use super::record::CalcRecord;

pub struct CalculationStore {
    conn: Mutex<Connection>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stored {
    /// Written now.
    New,
    /// Already held, byte for byte.
    Existing,
    /// Already held under this id with different content; the first stays.
    Differs,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CheckRow {
    pub check_id: i64,
    pub calc_id: String,
    pub checker: String,
    pub verdict: String,
    pub detail: serde_json::Value,
    pub graph_item_id: Option<String>,
    pub created_at: String,
}

impl CalculationStore {
    pub fn open(dir: &Path) -> Result<Self, String> {
        std::fs::create_dir_all(dir).map_err(|e| format!("the calculation store's folder could not be created: {e}"))?;
        let conn = Connection::open(dir.join("calculations.db")).map_err(|e| format!("the calculation store could not be opened: {e}"))?;
        Self::init(conn)
    }

    pub fn in_memory() -> Result<Self, String> {
        Self::init(Connection::open_in_memory().map_err(|e| e.to_string())?)
    }

    fn init(conn: Connection) -> Result<Self, String> {
        conn.execute_batch(
            "PRAGMA journal_mode=WAL;
             CREATE TABLE IF NOT EXISTS calculation_records (
                 calc_id TEXT PRIMARY KEY,
                 operation TEXT NOT NULL,
                 status TEXT NOT NULL,
                 record TEXT NOT NULL,
                 record_sha256 TEXT NOT NULL,
                 created_at TEXT NOT NULL
             );
             CREATE TABLE IF NOT EXISTS calculation_holders (
                 calc_id TEXT NOT NULL,
                 owner TEXT NOT NULL,
                 run_id TEXT NOT NULL,
                 created_at TEXT NOT NULL,
                 PRIMARY KEY (calc_id, owner, run_id)
             );
             CREATE INDEX IF NOT EXISTS calculation_holders_run ON calculation_holders(run_id);
             CREATE TABLE IF NOT EXISTS calculation_links (
                 calc_id TEXT NOT NULL,
                 owner TEXT NOT NULL,
                 graph_item_id TEXT NOT NULL,
                 PRIMARY KEY (calc_id, owner)
             );
             CREATE TABLE IF NOT EXISTS calculation_checks (
                 check_id INTEGER PRIMARY KEY AUTOINCREMENT,
                 calc_id TEXT NOT NULL,
                 checker TEXT NOT NULL,
                 verdict TEXT NOT NULL,
                 detail TEXT NOT NULL,
                 graph_item_id TEXT,
                 created_at TEXT NOT NULL
             );",
        )
        .map_err(|e| format!("the calculation store could not be prepared: {e}"))?;
        Ok(Self { conn: Mutex::new(conn) })
    }

    fn lock(&self) -> Result<std::sync::MutexGuard<'_, Connection>, String> {
        self.conn.lock().map_err(|_| "the calculation store was left locked by a failed write".to_string())
    }

    /// Keeps `record` for `owner`, from `run_id`. Idempotent.
    pub fn put(&self, record: &CalcRecord, owner: &str, run_id: &str) -> Result<Stored, String> {
        let body = serde_json::to_string(record).map_err(|e| e.to_string())?;
        let sha = record.sha256();
        let now = chrono::Utc::now().to_rfc3339();
        let conn = self.lock()?;
        let held: Option<String> = conn
            .query_row("SELECT record_sha256 FROM calculation_records WHERE calc_id = ?1", params![record.id], |row| row.get(0))
            .optional()
            .map_err(|e| e.to_string())?;
        let outcome = match held {
            None => {
                conn.execute(
                    "INSERT INTO calculation_records (calc_id, operation, status, record, record_sha256, created_at)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                    params![record.id, record.operation.as_str(), record.status.as_str(), body, sha, now],
                )
                .map_err(|e| e.to_string())?;
                Stored::New
            }
            Some(existing) if existing == sha => Stored::Existing,
            Some(_) => Stored::Differs,
        };
        conn.execute(
            "INSERT OR IGNORE INTO calculation_holders (calc_id, owner, run_id, created_at) VALUES (?1, ?2, ?3, ?4)",
            params![record.id, owner, run_id, now],
        )
        .map_err(|e| e.to_string())?;
        Ok(outcome)
    }

    /// A record `owner` holds, re-hashed on the way out. A record whose bytes
    /// no longer match their hash is not returned.
    pub fn get(&self, calc_id: &str, owner: &str) -> Option<CalcRecord> {
        let conn = self.lock().ok()?;
        let (body, sha): (String, String) = conn
            .query_row(
                "SELECT r.record, r.record_sha256 FROM calculation_records r
                  JOIN calculation_holders h ON h.calc_id = r.calc_id
                 WHERE r.calc_id = ?1 AND h.owner = ?2 LIMIT 1",
                params![calc_id, owner],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()
            .ok()
            .flatten()?;
        let record: CalcRecord = serde_json::from_str(&body).ok()?;
        if record.sha256() != sha {
            log::warn!("[calculation] {calc_id} does not match its stored hash; not returned");
            return None;
        }
        Some(record)
    }

    /// What one run computed, oldest first.
    pub fn for_run(&self, run_id: &str, owner: &str) -> Vec<CalcRecord> {
        let Ok(conn) = self.lock() else { return Vec::new() };
        let Ok(mut statement) = conn.prepare(
            "SELECT r.record FROM calculation_records r
               JOIN calculation_holders h ON h.calc_id = r.calc_id
              WHERE h.run_id = ?1 AND h.owner = ?2
              ORDER BY h.created_at, r.calc_id",
        ) else {
            return Vec::new();
        };
        statement
            .query_map(params![run_id, owner], |row| row.get::<_, String>(0))
            .map(|rows| rows.filter_map(Result::ok).filter_map(|body| serde_json::from_str(&body).ok()).collect())
            .unwrap_or_default()
    }

    /// The shared-memory item a record was published as, for one owner.
    /// Set once.
    pub fn link(&self, calc_id: &str, owner: &str, graph_item_id: &str) -> Result<(), String> {
        self.lock()?
            .execute(
                "INSERT OR IGNORE INTO calculation_links (calc_id, owner, graph_item_id) VALUES (?1, ?2, ?3)",
                params![calc_id, owner, graph_item_id],
            )
            .map(|_| ())
            .map_err(|e| e.to_string())
    }

    pub fn graph_item(&self, calc_id: &str, owner: &str) -> Option<String> {
        self.lock()
            .ok()?
            .query_row(
                "SELECT graph_item_id FROM calculation_links WHERE calc_id = ?1 AND owner = ?2",
                params![calc_id, owner],
                |row| row.get(0),
            )
            .optional()
            .ok()
            .flatten()
    }

    pub fn record_check(&self, calc_id: &str, checker: &str, verdict: &str, detail: &serde_json::Value, graph_item_id: Option<&str>) -> Result<i64, String> {
        let conn = self.lock()?;
        conn.execute(
            "INSERT INTO calculation_checks (calc_id, checker, verdict, detail, graph_item_id, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![calc_id, checker, verdict, detail.to_string(), graph_item_id, chrono::Utc::now().to_rfc3339()],
        )
        .map_err(|e| e.to_string())?;
        Ok(conn.last_insert_rowid())
    }

    pub fn checks(&self, calc_id: &str) -> Vec<CheckRow> {
        let Ok(conn) = self.lock() else { return Vec::new() };
        let Ok(mut statement) = conn.prepare(
            "SELECT check_id, calc_id, checker, verdict, detail, graph_item_id, created_at
               FROM calculation_checks WHERE calc_id = ?1 ORDER BY check_id",
        ) else {
            return Vec::new();
        };
        statement
            .query_map(params![calc_id], |row| {
                Ok(CheckRow {
                    check_id: row.get(0)?,
                    calc_id: row.get(1)?,
                    checker: row.get(2)?,
                    verdict: row.get(3)?,
                    detail: serde_json::from_str(&row.get::<_, String>(4)?).unwrap_or_default(),
                    graph_item_id: row.get(5)?,
                    created_at: row.get(6)?,
                })
            })
            .map(|rows| rows.filter_map(Result::ok).collect())
            .unwrap_or_default()
    }

    #[cfg(test)]
    pub(crate) fn tamper(&self, calc_id: &str, body: &str) {
        self.lock().unwrap().execute("UPDATE calculation_records SET record = ?2 WHERE calc_id = ?1", params![calc_id, body]).unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calculation::inputs::CalcInput;
    use crate::calculation::ops::{evaluate, Options};

    #[test]
    fn a_record_is_kept_once_read_only_by_its_holders_and_refused_if_tampered() {
        let store = CalculationStore::in_memory().unwrap();
        let inputs = vec![CalcInput::parse_line("a = 2 m [user]").unwrap()];
        let record = evaluate("a * 3", inputs, &Options::default(), None);
        assert_eq!(store.put(&record, "priya", "run-1").unwrap(), Stored::New);
        assert_eq!(store.put(&record, "priya", "run-2").unwrap(), Stored::Existing);
        assert_eq!(store.get(&record.id, "priya").unwrap(), record);
        assert!(store.get(&record.id, "someone-else").is_none());
        assert_eq!(store.for_run("run-2", "priya").len(), 1);

        let mut altered = record.clone();
        altered.results[0].display = "7 m".into();
        store.tamper(&record.id, &serde_json::to_string(&altered).unwrap());
        assert!(store.get(&record.id, "priya").is_none(), "a record that no longer matches its hash is not served");
    }
}
