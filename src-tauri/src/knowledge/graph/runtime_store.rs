//! The authoritative store for runtime memory.
//!
//! ## Why there is an outbox, established rather than assumed
//!
//! `sarathi.db` is one file opened by several subsystems, each through its
//! **own** `Connection`: the task event log, the audit log, the notebook store,
//! the artifact store and the document store all call
//! `Connection::open(app_data_dir.join("sarathi.db"))` independently. A SQLite
//! transaction is scoped to a connection, so "they are in the same file" does
//! **not** make a write across two of them atomic. Two other authorities — the
//! conversation store and the agent registry — are JSON files and are not in
//! that database at all.
//!
//! So a commit that must also be seen elsewhere cannot be one transaction. What
//! it can be is one transaction *here* that writes the item, its revision and
//! an **outbox row** together, followed by a separate deliverer that drains the
//! outbox and marks each row delivered. That is recoverable: a crash between
//! the commit and the delivery leaves a row that is retried, and delivery is
//! idempotent because each row carries the key the receiver de-duplicates on.
//!
//! The alternative — writing both and hoping — is the failure this design
//! exists to refuse, and it is invisible when it happens.
//!
//! ## Publish only what is committed
//!
//! Nothing reaches the changefeed until its transaction has committed. The
//! revision in `agent_memory_log` is assigned *inside* the transaction, so a
//! reader that has seen revision N has seen a row that is durably there.
//!
//! ## Authorisation happens before traversal, not after it
//!
//! Every read resolves the authorised item set first and traverses only inside
//! it. Filtering afterwards is the mistake that leaks through counts, labels
//! and neighbour lists even when the final row list is clean — "this node has
//! 14 neighbours" is a disclosure about 14 rows the reader may not see.

use std::collections::{BTreeSet, HashMap};
use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};

use super::runtime_feed::{ChangeBatch, FeedChange, FeedEntry, MemorySnapshot, Subject, MAX_BATCH};
use super::runtime_memory::{
    admit, edge_id, may_supersede, EdgeKind, ItemStatus, MemoryEdge, MemoryItem, MemoryScope,
    MEMORY_SCHEMA_VERSION,
};
use crate::identity::Session;

/// Why a write did not happen.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "error", rename_all = "camelCase")]
pub enum MemoryError {
    /// Somebody else wrote a newer revision of this item first.
    ///
    /// Explicit, and never resolved by taking the later write. Silent
    /// last-writer-wins on a fact or a task decision means one agent's
    /// conclusion disappears with nothing anywhere saying it did.
    RevisionConflict {
        item_id: String,
        expected: u64,
        actual: u64,
    },
    /// The reader may not see what they named.
    ///
    /// The same answer for "does not exist" and "not yours", so a refusal
    /// cannot be used to discover which ids are real.
    NotFound { item_id: String },
    /// The supersession is not one this provenance may make.
    NotPermitted { because: String },
    Storage { detail: String },
}

impl MemoryError {
    pub fn explain(&self) -> String {
        match self {
            Self::RevisionConflict {
                item_id,
                expected,
                actual,
            } => format!(
                "{item_id} moved from revision {expected} to {actual} while this was being \
                 written. Re-read it and decide again — taking the later write silently would \
                 lose whatever the other one established."
            ),
            Self::NotFound { item_id } => format!("There is no memory item {item_id:?} for you."),
            Self::NotPermitted { because } => because.clone(),
            Self::Storage { detail } => {
                format!("The memory graph could not be read or written: {detail}")
            }
        }
    }
}

fn storage<E: std::fmt::Display>(error: E) -> MemoryError {
    MemoryError::Storage {
        detail: error.to_string(),
    }
}

/// What a commit did.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Committed {
    pub item_id: String,
    pub revision: u64,
    /// The changefeed position this landed at. Monotonic across the store.
    pub graph_revision: i64,
    /// True when an identical commit had already been applied and this call
    /// changed nothing. See the idempotency note on [`MemoryGraph::commit`].
    pub duplicate: bool,
    pub status: ItemStatus,
    /// Why the item is at that status, from the admission rules.
    pub because: String,
}

/// One pending cross-store effect.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OutboxRow {
    pub outbox_id: i64,
    /// Which store this is for: `conversations`, `agentRegistry`, `events`.
    pub target: String,
    /// The key the receiver de-duplicates on, so redelivery is harmless.
    pub idempotency_key: String,
    pub payload: String,
    pub created_at: String,
}

/// The runtime memory graph.
pub struct MemoryGraph {
    conn: Arc<Mutex<Connection>>,
    /// Told the new head revision after a change has committed.
    ///
    /// ## Why a callback rather than the store emitting an event itself
    ///
    /// This module must not know that a window exists. It is used by the
    /// subagent workers and by tests with no Tauri app at all, and a store that
    /// reached for an `AppHandle` would be unusable in both. The application
    /// installs one of these at start-up; everything else leaves it unset and
    /// the store stays silent.
    ///
    /// ## Why it carries a number and not the change
    ///
    /// Because the watcher fires for everybody, and what any particular reader
    /// may see is not decidable here. A revision is not a secret: it says that
    /// *something* moved, which is precisely the amount a reader needs to know
    /// in order to go and ask, as itself, what it is now allowed to see.
    watcher: Mutex<Option<Arc<dyn Fn(i64) + Send + Sync>>>,
}

impl MemoryGraph {
    /// Opens the graph in the application's database.
    ///
    /// Its own tables, prefixed `agent_memory_`. The prefix is not decoration:
    /// `sarathi.db` already contains a `documents` table created twice with
    /// incompatible schemas by two different subsystems, and `projects` and
    /// `memory_nodes` are taken by the legacy memory engine. See the note at
    /// the top of `knowledge/graph/mod.rs`.
    pub fn open(app_data_dir: &Path) -> Result<Self> {
        std::fs::create_dir_all(app_data_dir)?;
        let conn = Connection::open(app_data_dir.join("sarathi.db"))
            .context("could not open the runtime memory graph")?;
        Self::prepare(&conn)?;
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
            watcher: Mutex::new(None),
        })
    }

    /// An in-memory graph, for tests.
    pub fn in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()?;
        Self::prepare(&conn)?;
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
            watcher: Mutex::new(None),
        })
    }

    fn prepare(conn: &Connection) -> Result<()> {
        // A five-second wait, for the reason `events::store` sets one: this
        // file is shared, and a write that fails outright the first time a
        // document is indexed concurrently is a write that loses a fact.
        conn.busy_timeout(std::time::Duration::from_secs(5))?;
        conn.execute_batch(&format!(
            "
            CREATE TABLE IF NOT EXISTS agent_memory_items (
                item_id          TEXT PRIMARY KEY,
                revision         INTEGER NOT NULL,
                scope_key        TEXT NOT NULL,
                agent_id         TEXT NOT NULL,
                kind             TEXT NOT NULL,
                status           TEXT NOT NULL,
                idempotency_key  TEXT,
                schema_version   INTEGER NOT NULL DEFAULT {MEMORY_SCHEMA_VERSION},
                body             TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS agent_memory_items_scope
                ON agent_memory_items (scope_key);
            CREATE UNIQUE INDEX IF NOT EXISTS agent_memory_items_idem
                ON agent_memory_items (idempotency_key)
                WHERE idempotency_key IS NOT NULL;

            CREATE TABLE IF NOT EXISTS agent_memory_edges (
                edge_id    TEXT PRIMARY KEY,
                from_item  TEXT NOT NULL,
                to_item    TEXT NOT NULL,
                kind       TEXT NOT NULL,
                agent_id   TEXT NOT NULL,
                scope_key  TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS agent_memory_edges_from
                ON agent_memory_edges (from_item);
            CREATE INDEX IF NOT EXISTS agent_memory_edges_to
                ON agent_memory_edges (to_item);

            CREATE TABLE IF NOT EXISTS agent_memory_log (
                revision  INTEGER PRIMARY KEY AUTOINCREMENT,
                item_id   TEXT NOT NULL,
                scope_key TEXT NOT NULL,
                change    TEXT NOT NULL,
                at        TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS agent_memory_log_scope
                ON agent_memory_log (scope_key, revision);

            -- How far back the feed can still be replayed from.
            --
            -- A reader whose cursor is below this cannot be caught up by
            -- replay, and is told to take a fresh snapshot instead of being
            -- handed the rows that happen to survive. One row, enforced by the
            -- CHECK: a floor with two values is a floor nobody can trust.
            CREATE TABLE IF NOT EXISTS agent_memory_feed_floor (
                id    INTEGER PRIMARY KEY CHECK (id = 1),
                floor INTEGER NOT NULL
            );
            INSERT OR IGNORE INTO agent_memory_feed_floor (id, floor) VALUES (1, 0);

            CREATE TABLE IF NOT EXISTS agent_memory_outbox (
                outbox_id       INTEGER PRIMARY KEY AUTOINCREMENT,
                target          TEXT NOT NULL,
                idempotency_key TEXT NOT NULL,
                payload         TEXT NOT NULL,
                created_at      TEXT NOT NULL,
                delivered_at    TEXT
            );
            CREATE INDEX IF NOT EXISTS agent_memory_outbox_pending
                ON agent_memory_outbox (delivered_at);
            "
        ))?;

        // `subject_kind` arrived after the log did, so it is added rather than
        // declared. A database written by the previous version has rows without
        // it, and every one of those rows is about an item — which is exactly
        // what the default says, and why the column can be added without a
        // backfill pass.
        //
        // The error is swallowed only for the one case that means "already
        // there". SQLite has no `ADD COLUMN IF NOT EXISTS`, and a `PRAGMA
        // table_info` round trip to ask first would be the same check written
        // twice.
        if let Err(error) = conn.execute(
            "ALTER TABLE agent_memory_log ADD COLUMN subject_kind TEXT NOT NULL DEFAULT 'item'",
            [],
        ) {
            let message = error.to_string();
            if !message.contains("duplicate column name") {
                return Err(error).context("the changefeed's subject column could not be added");
            }
        }
        Ok(())
    }

    // ── Writing ──────────────────────────────────────────────────────────

    /// Writes an item, or reports why it was not written.
    ///
    /// ## Idempotency
    ///
    /// When the item carries an `idempotency_key`, a second commit with the
    /// same key returns the first one's result with `duplicate: true` and
    /// changes nothing. That is what makes a retry after a lost acknowledgement
    /// safe: the writer cannot tell whether the first attempt landed, and this
    /// makes it not need to.
    ///
    /// ## Optimistic version check
    ///
    /// `expected_revision` is the revision the writer read. A mismatch is a
    /// [`MemoryError::RevisionConflict`], never a silent overwrite — two agents
    /// writing the same claim must both find out.
    ///
    /// ## What reaches the outbox
    ///
    /// Anything the caller passes in `effects` is written in the *same*
    /// transaction as the item. A crash after this returns leaves rows a
    /// deliverer retries; a crash before it leaves neither the item nor the
    /// effect, which is the pair that must not come apart.
    pub fn commit(
        &self,
        mut item: MemoryItem,
        expected_revision: Option<u64>,
        effects: &[(String, String, String)],
    ) -> Result<Committed, MemoryError> {
        let mut conn = self.conn.lock().map_err(|_| MemoryError::Storage {
            detail: "the memory graph was left locked by a failed write".into(),
        })?;

        let outcome = admit(&item);
        item.status = outcome.status;

        let transaction = conn.transaction().map_err(storage)?;

        // A key already applied is the same write arriving twice.
        if let Some(key) = item.idempotency_key.as_deref() {
            let held: Option<(String, i64)> = transaction
                .query_row(
                    "SELECT item_id, revision FROM agent_memory_items WHERE idempotency_key = ?1",
                    params![key],
                    |row| Ok((row.get(0)?, row.get(1)?)),
                )
                .optional()
                .map_err(storage)?;
            if let Some((item_id, revision)) = held {
                let graph_revision = Self::latest_revision(&transaction).unwrap_or(0);
                return Ok(Committed {
                    item_id,
                    revision: revision as u64,
                    graph_revision,
                    duplicate: true,
                    status: outcome.status,
                    because: outcome.because,
                });
            }
        }

        let existing: Option<i64> = transaction
            .query_row(
                "SELECT revision FROM agent_memory_items WHERE item_id = ?1",
                params![item.item_id],
                |row| row.get(0),
            )
            .optional()
            .map_err(storage)?;

        match (existing, expected_revision) {
            (Some(actual), Some(expected)) if actual as u64 != expected => {
                return Err(MemoryError::RevisionConflict {
                    item_id: item.item_id,
                    expected,
                    actual: actual as u64,
                })
            }
            // A writer that expected nothing and found something is the same
            // conflict: it believed it was creating this item.
            (Some(actual), None) => {
                return Err(MemoryError::RevisionConflict {
                    item_id: item.item_id,
                    expected: 0,
                    actual: actual as u64,
                })
            }
            (None, Some(expected)) if expected > 0 => {
                return Err(MemoryError::RevisionConflict {
                    item_id: item.item_id,
                    expected,
                    actual: 0,
                })
            }
            _ => {}
        }

        item.revision = existing.map(|held| held as u64 + 1).unwrap_or(1);
        let body = serde_json::to_string(&item).map_err(storage)?;
        let scope_key = item.scope.key();

        transaction
            .execute(
                "INSERT INTO agent_memory_items
                     (item_id, revision, scope_key, agent_id, kind, status,
                      idempotency_key, schema_version, body)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
                 ON CONFLICT(item_id) DO UPDATE SET
                     revision = excluded.revision,
                     status = excluded.status,
                     body = excluded.body",
                params![
                    item.item_id,
                    item.revision as i64,
                    scope_key,
                    item.agent_id,
                    item.kind.as_str(),
                    item.status.as_str(),
                    item.idempotency_key,
                    MEMORY_SCHEMA_VERSION,
                    body,
                ],
            )
            .map_err(storage)?;

        transaction
            .execute(
                "INSERT INTO agent_memory_log (item_id, scope_key, change, at)
                 VALUES (?1, ?2, ?3, ?4)",
                params![
                    item.item_id,
                    scope_key,
                    if existing.is_some() {
                        "updated"
                    } else {
                        "created"
                    },
                    item.updated_at,
                ],
            )
            .map_err(storage)?;

        for (target, key, payload) in effects {
            transaction
                .execute(
                    "INSERT INTO agent_memory_outbox
                         (target, idempotency_key, payload, created_at)
                     VALUES (?1, ?2, ?3, ?4)",
                    params![target, key, payload, item.updated_at],
                )
                .map_err(storage)?;
        }

        let graph_revision = Self::latest_revision(&transaction).unwrap_or(0);
        transaction.commit().map_err(storage)?;
        // The connection lock goes before the watcher runs. A watcher that
        // reached back into the graph would otherwise deadlock on a mutex this
        // frame is still holding, and "notifying somebody wedges the writer" is
        // a failure that only shows up under load.
        drop(conn);
        self.announce(graph_revision);

        Ok(Committed {
            item_id: item.item_id,
            revision: item.revision,
            graph_revision,
            duplicate: false,
            status: outcome.status,
            because: outcome.because,
        })
    }

    fn latest_revision(conn: &Connection) -> Option<i64> {
        conn.query_row("SELECT MAX(revision) FROM agent_memory_log", [], |row| {
            row.get::<_, Option<i64>>(0)
        })
        .ok()
        .flatten()
    }

    /// Records a correction: the new item is written and the old one is marked
    /// superseded, with an edge saying what replaced it.
    ///
    /// Nothing is deleted. "The model said 150 PSI and a person changed it to
    /// 10 bar" is the record somebody will need, and erasing the first half
    /// turns a correction into an assertion with no history.
    pub fn correct(
        &self,
        correction: MemoryItem,
        supersedes_item: &str,
    ) -> Result<Committed, MemoryError> {
        let existing = self.raw(supersedes_item)?.ok_or(MemoryError::NotFound {
            item_id: supersedes_item.to_string(),
        })?;
        may_supersede(&correction, &existing)
            .map_err(|because| MemoryError::NotPermitted { because })?;

        let mut correction = correction;
        correction.supersedes = Some(supersedes_item.to_string());
        let committed = self.commit(correction.clone(), None, &[])?;

        let mut conn = self.conn.lock().map_err(|_| MemoryError::Storage {
            detail: "the memory graph was left locked by a failed write".into(),
        })?;
        let transaction = conn.transaction().map_err(storage)?;

        let mut superseded = existing;
        superseded.status = ItemStatus::Superseded;
        let body = serde_json::to_string(&superseded).map_err(storage)?;
        transaction
            .execute(
                "UPDATE agent_memory_items SET status = ?1, body = ?2 WHERE item_id = ?3",
                params![ItemStatus::Superseded.as_str(), body, supersedes_item],
            )
            .map_err(storage)?;

        let id = edge_id(&correction.item_id, EdgeKind::Supersedes, supersedes_item);
        transaction
            .execute(
                "INSERT INTO agent_memory_edges
                     (edge_id, from_item, to_item, kind, agent_id, scope_key, created_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
                 ON CONFLICT(edge_id) DO NOTHING",
                params![
                    id,
                    correction.item_id,
                    supersedes_item,
                    EdgeKind::Supersedes.as_str(),
                    correction.agent_id,
                    correction.scope.key(),
                    correction.updated_at,
                ],
            )
            .map_err(storage)?;

        transaction
            .execute(
                "INSERT INTO agent_memory_log (item_id, scope_key, change, at)
                 VALUES (?1, ?2, 'superseded', ?3)",
                params![
                    supersedes_item,
                    superseded.scope.key(),
                    correction.updated_at
                ],
            )
            .map_err(storage)?;

        let revision = Self::latest_revision(&transaction).unwrap_or(0);
        transaction.commit().map_err(storage)?;
        drop(conn);
        self.announce(revision);
        Ok(committed)
    }

    /// Links two items. Idempotent on the triple.
    /// ## Why this writes a log row and why it is conditional
    ///
    /// A link is half of what the graph means. An edge that reached the tables
    /// but not [`agent_memory_log`](Self::changes_since) would be invisible to
    /// every reader already subscribed: they would see the two items appear and
    /// never learn that one contradicts the other, until something unrelated
    /// made them re-snapshot. A contradiction that only shows up after a reload
    /// is worse than no contradiction edge at all, because the picture in front
    /// of somebody looks complete.
    ///
    /// The row is written only when the insert actually inserted. `ON CONFLICT
    /// DO NOTHING` makes a repeat harmless in the edge table, and announcing a
    /// change that did not happen would wake every window to re-fetch an
    /// identical graph — which is how an idempotent writer retrying in a loop
    /// turns into a redraw loop.
    pub fn link(&self, edge: MemoryEdge) -> Result<(), MemoryError> {
        let mut conn = self.conn.lock().map_err(|_| MemoryError::Storage {
            detail: "the memory graph was left locked by a failed write".into(),
        })?;
        let transaction = conn.transaction().map_err(storage)?;
        let scope_key = edge.scope.key();
        let inserted = transaction
            .execute(
                "INSERT INTO agent_memory_edges
                     (edge_id, from_item, to_item, kind, agent_id, scope_key, created_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
                 ON CONFLICT(edge_id) DO NOTHING",
                params![
                    edge.edge_id,
                    edge.from_item,
                    edge.to_item,
                    edge.kind.as_str(),
                    edge.agent_id,
                    scope_key,
                    edge.created_at,
                ],
            )
            .map_err(storage)?;

        if inserted > 0 {
            transaction
                .execute(
                    "INSERT INTO agent_memory_log
                         (item_id, scope_key, change, at, subject_kind)
                     VALUES (?1, ?2, 'linked', ?3, 'edge')",
                    params![edge.edge_id, scope_key, edge.created_at],
                )
                .map_err(storage)?;
        }

        let revision = Self::latest_revision(&transaction).unwrap_or(0);
        transaction.commit().map_err(storage)?;
        drop(conn);
        if inserted > 0 {
            self.announce(revision);
        }
        Ok(())
    }

    /// Marks an item deleted without removing its row.
    pub fn tombstone(&self, item_id: &str, at: &str) -> Result<(), MemoryError> {
        let Some(mut item) = self.raw(item_id)? else {
            return Err(MemoryError::NotFound {
                item_id: item_id.to_string(),
            });
        };
        item.status = ItemStatus::Tombstoned;
        item.updated_at = at.to_string();
        let body = serde_json::to_string(&item).map_err(storage)?;

        let mut conn = self.conn.lock().map_err(|_| MemoryError::Storage {
            detail: "the memory graph was left locked by a failed write".into(),
        })?;
        let transaction = conn.transaction().map_err(storage)?;
        transaction
            .execute(
                "UPDATE agent_memory_items SET status = ?1, body = ?2 WHERE item_id = ?3",
                params![ItemStatus::Tombstoned.as_str(), body, item_id],
            )
            .map_err(storage)?;
        transaction
            .execute(
                "INSERT INTO agent_memory_log (item_id, scope_key, change, at)
                 VALUES (?1, ?2, 'tombstoned', ?3)",
                params![item_id, item.scope.key(), at],
            )
            .map_err(storage)?;
        let revision = Self::latest_revision(&transaction).unwrap_or(0);
        transaction.commit().map_err(storage)?;
        drop(conn);
        self.announce(revision);
        Ok(())
    }

    /// Invalidates every item whose evidence came from a source that has been
    /// removed or whose permission changed.
    ///
    /// Returns the items affected. They are marked `Rejected` rather than
    /// deleted: "this was believed on the strength of a document nobody may
    /// read any more" is a fact worth keeping, and a projection or retrieval
    /// cache that still holds them has to be told which ones to drop.
    pub fn invalidate_source(&self, sha256: &str, at: &str) -> Result<Vec<String>, MemoryError> {
        let affected: Vec<MemoryItem> = self
            .all_raw()?
            .into_iter()
            .filter(|item| item.sources.iter().any(|source| source.sha256 == sha256))
            .filter(|item| item.is_readable())
            .collect();

        let mut conn = self.conn.lock().map_err(|_| MemoryError::Storage {
            detail: "the memory graph was left locked by a failed write".into(),
        })?;
        let transaction = conn.transaction().map_err(storage)?;
        let mut ids = Vec::new();
        for mut item in affected {
            item.status = ItemStatus::Rejected;
            item.updated_at = at.to_string();
            let body = serde_json::to_string(&item).map_err(storage)?;
            transaction
                .execute(
                    "UPDATE agent_memory_items SET status = ?1, body = ?2 WHERE item_id = ?3",
                    params![ItemStatus::Rejected.as_str(), body, item.item_id],
                )
                .map_err(storage)?;
            transaction
                .execute(
                    "INSERT INTO agent_memory_log (item_id, scope_key, change, at)
                     VALUES (?1, ?2, 'sourceInvalidated', ?3)",
                    params![item.item_id, item.scope.key(), at],
                )
                .map_err(storage)?;
            ids.push(item.item_id);
        }
        let revision = Self::latest_revision(&transaction).unwrap_or(0);
        transaction.commit().map_err(storage)?;
        drop(conn);
        // Announced even when nothing matched: the watcher is a hint to go and
        // look, and a spurious one costs a cheap indexed read. A missed one
        // costs a stale canvas.
        self.announce(revision);
        Ok(ids)
    }

    // ── Outbox ───────────────────────────────────────────────────────────

    /// Effects that have not been delivered.
    pub fn pending_effects(&self) -> Result<Vec<OutboxRow>, MemoryError> {
        let conn = self.conn.lock().map_err(|_| MemoryError::Storage {
            detail: "the memory graph was left locked".into(),
        })?;
        let mut statement = conn
            .prepare(
                "SELECT outbox_id, target, idempotency_key, payload, created_at
                 FROM agent_memory_outbox WHERE delivered_at IS NULL ORDER BY outbox_id",
            )
            .map_err(storage)?;
        let rows = statement
            .query_map([], |row| {
                Ok(OutboxRow {
                    outbox_id: row.get(0)?,
                    target: row.get(1)?,
                    idempotency_key: row.get(2)?,
                    payload: row.get(3)?,
                    created_at: row.get(4)?,
                })
            })
            .map_err(storage)?
            .filter_map(Result::ok)
            .collect();
        Ok(rows)
    }

    /// Marks one effect delivered. Safe to call twice.
    pub fn mark_delivered(&self, outbox_id: i64, at: &str) -> Result<(), MemoryError> {
        let conn = self.conn.lock().map_err(|_| MemoryError::Storage {
            detail: "the memory graph was left locked".into(),
        })?;
        conn.execute(
            "UPDATE agent_memory_outbox SET delivered_at = ?1
             WHERE outbox_id = ?2 AND delivered_at IS NULL",
            params![at, outbox_id],
        )
        .map_err(storage)?;
        Ok(())
    }

    // ── Reading ──────────────────────────────────────────────────────────

    /// One item, without any authorisation. Internal use only.
    fn raw(&self, item_id: &str) -> Result<Option<MemoryItem>, MemoryError> {
        let conn = self.conn.lock().map_err(|_| MemoryError::Storage {
            detail: "the memory graph was left locked".into(),
        })?;
        let body: Option<String> = conn
            .query_row(
                "SELECT body FROM agent_memory_items WHERE item_id = ?1",
                params![item_id],
                |row| row.get(0),
            )
            .optional()
            .map_err(storage)?;
        Ok(match body {
            Some(text) => Some(serde_json::from_str(&text).map_err(storage)?),
            None => None,
        })
    }

    fn all_raw(&self) -> Result<Vec<MemoryItem>, MemoryError> {
        let conn = self.conn.lock().map_err(|_| MemoryError::Storage {
            detail: "the memory graph was left locked".into(),
        })?;
        let mut statement = conn
            .prepare("SELECT body FROM agent_memory_items")
            .map_err(storage)?;
        let rows: Vec<MemoryItem> = statement
            .query_map([], |row| row.get::<_, String>(0))
            .map_err(storage)?
            .filter_map(Result::ok)
            .filter_map(|text| serde_json::from_str(&text).ok())
            .collect();
        Ok(rows)
    }

    /// The items this reader may see in this scope.
    ///
    /// ## The authorised set is resolved first, and everything else works
    /// inside it
    ///
    /// Not as an optimisation. A traversal that walked every edge and filtered
    /// the *result* would still have told the reader how many neighbours a node
    /// has, which of them are contradictions, and what the labels on the way
    /// were — each of which is a disclosure about rows they may not read.
    /// [`Self::neighbours`] and [`Self::count`] take this set rather than
    /// re-deriving one.
    pub fn snapshot(
        &self,
        session: &Session,
        scope: &MemoryScope,
        project_id: Option<&str>,
    ) -> Result<Vec<MemoryItem>, MemoryError> {
        let candidates: Vec<MemoryItem> = {
            let conn = self.conn.lock().map_err(|_| MemoryError::Storage {
                detail: "the memory graph was left locked".into(),
            })?;
            // The scope is bound into the statement rather than filtered after
            // the fetch, the way `knowledge::graph::store` binds
            // `owner_user_id`.
            let mut statement = conn
                .prepare("SELECT body FROM agent_memory_items WHERE scope_key = ?1")
                .map_err(storage)?;
            let rows: Vec<String> = statement
                .query_map(params![scope.key()], |row| row.get::<_, String>(0))
                .map_err(storage)?
                .filter_map(Result::ok)
                .collect();
            rows.into_iter()
                .filter_map(|text| serde_json::from_str::<MemoryItem>(&text).ok())
                .collect()
        };

        Ok(candidates
            .into_iter()
            .filter(|item| item.readable_by(session, project_id))
            .collect())
    }

    /// Neighbours of `item_id`, **within** an already-authorised set.
    ///
    /// Takes the set rather than a session on purpose: the caller has already
    /// resolved what may be seen, and an edge to something outside it is not
    /// reported at all — not as a redacted entry, which would still disclose
    /// that something is there.
    pub fn neighbours(
        &self,
        item_id: &str,
        authorised: &[MemoryItem],
    ) -> Result<Vec<MemoryEdge>, MemoryError> {
        let visible: BTreeSet<&str> = authorised
            .iter()
            .map(|item| item.item_id.as_str())
            .collect();
        if !visible.contains(item_id) {
            return Ok(Vec::new());
        }

        let rows: Vec<(String, String, String, String, String, String, String)> = {
            let conn = self.conn.lock().map_err(|_| MemoryError::Storage {
                detail: "the memory graph was left locked".into(),
            })?;
            let mut statement = conn
                .prepare(
                    "SELECT edge_id, from_item, to_item, kind, agent_id, scope_key, created_at
                     FROM agent_memory_edges WHERE from_item = ?1 OR to_item = ?1",
                )
                .map_err(storage)?;
            let collected: Vec<_> = statement
                .query_map(params![item_id], |row| {
                    Ok((
                        row.get(0)?,
                        row.get(1)?,
                        row.get(2)?,
                        row.get(3)?,
                        row.get(4)?,
                        row.get(5)?,
                        row.get(6)?,
                    ))
                })
                .map_err(storage)?
                .filter_map(Result::ok)
                .collect();
            collected
        };

        Ok(rows
            .into_iter()
            .filter(|(_, from, to, ..)| {
                visible.contains(from.as_str()) && visible.contains(to.as_str())
            })
            .filter_map(|(id, from, to, kind, agent, scope_key, created)| {
                Some(MemoryEdge {
                    edge_id: id,
                    from_item: from,
                    to_item: to,
                    kind: EdgeKind::parse(&kind)?,
                    agent_id: agent,
                    scope: scope_from_key(&scope_key)?,
                    created_at: created,
                })
            })
            .collect())
    }

    /// How many items of each kind this reader may see.
    ///
    /// Computed from the authorised set, never with `SELECT COUNT(*)`. A count
    /// taken before authorisation is a disclosure: "there are 14 facts in this
    /// task" tells a reader about rows they cannot read.
    pub fn count(authorised: &[MemoryItem]) -> HashMap<&'static str, usize> {
        let mut out = HashMap::new();
        for item in authorised {
            *out.entry(item.kind.as_str()).or_insert(0) += 1;
        }
        out
    }

    /// Installs the thing to tell when the graph moves.
    ///
    /// Replaces whatever was there. One application, one watcher; a second
    /// caller wanting notifications is a sign that something should be fanning
    /// out on the other side of this, not that this should hold a list.
    pub fn on_change(&self, watcher: Arc<dyn Fn(i64) + Send + Sync>) {
        if let Ok(mut held) = self.watcher.lock() {
            *held = Some(watcher);
        }
    }

    /// Announces a committed change. Called *after* the transaction, never
    /// inside it.
    ///
    /// That order is the commit-before-publish rule, and it is one line of
    /// code: a notification sent from inside the transaction would reach a
    /// reader that then queried and found nothing, because the write it was
    /// told about had not landed yet — and on a rollback it would have been
    /// told about a write that never happened at all.
    ///
    /// The watcher is copied out before it is called, so the lock is not held
    /// while somebody else's code runs.
    fn announce(&self, revision: i64) {
        let watcher = self
            .watcher
            .lock()
            .ok()
            .and_then(|held| held.as_ref().map(Arc::clone));
        if let Some(watcher) = watcher {
            watcher(revision);
        }
    }

    /// The changefeed position right now.
    pub fn graph_revision(&self) -> Result<i64, MemoryError> {
        let conn = self.conn.lock().map_err(|_| MemoryError::Storage {
            detail: "the memory graph was left locked".into(),
        })?;
        Ok(Self::latest_revision(&conn).unwrap_or(0))
    }

    // ── The changefeed ───────────────────────────────────────────────────

    /// The graph as it stands, with the cursor that continues it.
    ///
    /// ## One transaction, because two reads are a lost change
    ///
    /// The items, the edges and the revision are read inside a single
    /// transaction. Reading them separately is the classic subscribe-after-
    /// snapshot bug: a write landing between the two is in neither the snapshot
    /// nor the replay, and nothing afterwards can tell it happened. Taken
    /// together, `cursor` names exactly the last change already folded into
    /// `items`, so a reader that subscribes from it misses nothing and repeats
    /// only what it can recognise by id.
    ///
    /// SQLite's default deferred transaction takes its read snapshot at the
    /// first statement and holds it, which is the isolation this needs; the
    /// connection mutex is held throughout, so no writer can interleave.
    ///
    /// ## Authorisation first, geometry never
    ///
    /// The authorised item set is resolved before edges are looked at, and an
    /// edge survives only if *both* its ends are in that set — the rule
    /// [`Self::neighbours`] already applies. An edge with one hidden end is
    /// dropped entirely rather than returned with the end blanked: a stub still
    /// says "there is something over there and this contradicts it", which is a
    /// disclosure about a row the reader may not read.
    pub fn snapshot_at(
        &self,
        session: &Session,
        scope: &MemoryScope,
        project_id: Option<&str>,
    ) -> Result<MemorySnapshot, MemoryError> {
        let mut conn = self.conn.lock().map_err(|_| MemoryError::Storage {
            detail: "the memory graph was left locked".into(),
        })?;
        let transaction = conn.transaction().map_err(storage)?;
        let scope_key = scope.key();

        let items: Vec<MemoryItem> = {
            let mut statement = transaction
                .prepare("SELECT body FROM agent_memory_items WHERE scope_key = ?1")
                .map_err(storage)?;
            let bodies: Vec<String> = statement
                .query_map(params![scope_key], |row| row.get::<_, String>(0))
                .map_err(storage)?
                .filter_map(Result::ok)
                .collect();
            bodies
                .into_iter()
                .filter_map(|text| serde_json::from_str::<MemoryItem>(&text).ok())
                .filter(|item| item.readable_by(session, project_id))
                .collect()
        };

        let visible: BTreeSet<&str> = items.iter().map(|item| item.item_id.as_str()).collect();

        let edges: Vec<MemoryEdge> = {
            let mut statement = transaction
                .prepare(
                    "SELECT edge_id, from_item, to_item, kind, agent_id, scope_key, created_at
                     FROM agent_memory_edges WHERE scope_key = ?1",
                )
                .map_err(storage)?;
            let rows: Vec<(String, String, String, String, String, String, String)> = statement
                .query_map(params![scope_key], |row| {
                    Ok((
                        row.get(0)?,
                        row.get(1)?,
                        row.get(2)?,
                        row.get(3)?,
                        row.get(4)?,
                        row.get(5)?,
                        row.get(6)?,
                    ))
                })
                .map_err(storage)?
                .filter_map(Result::ok)
                .collect();
            rows.into_iter()
                .filter(|(_, from, to, ..)| {
                    visible.contains(from.as_str()) && visible.contains(to.as_str())
                })
                .filter_map(|(id, from, to, kind, agent, key, created)| {
                    Some(MemoryEdge {
                        edge_id: id,
                        from_item: from,
                        to_item: to,
                        kind: EdgeKind::parse(&kind)?,
                        agent_id: agent,
                        scope: scope_from_key(&key)?,
                        created_at: created,
                    })
                })
                .collect()
        };

        let cursor = Self::latest_revision(&transaction).unwrap_or(0);
        transaction.commit().map_err(storage)?;

        Ok(MemorySnapshot {
            items,
            edges,
            cursor,
            in_context: Vec::new(),
            context_revision: None,
        })
    }

    /// Several scopes read together, with the one cursor they were read at.
    ///
    /// ## Why this exists beside [`Self::snapshot`]
    ///
    /// The context compiler used to call `snapshot` for the task and label the
    /// result with a revision it had read *separately* — so a commit landing
    /// between the two reads was in the rows and not in the cursor, or the
    /// other way round, and the manifest recorded a position the rows did not
    /// come from (plan §3, finding 1). And it read only the task: the project's
    /// knowledge, the person's preferences and activated procedures were never
    /// consulted at all.
    ///
    /// Here every scope is read inside one transaction and the cursor is read
    /// inside the same one, the way [`Self::snapshot_at`] does it for a single
    /// scope. `cursor` names exactly the last change folded into the rows.
    ///
    /// ## Authorisation per scope, before anything else
    ///
    /// Each item is checked with [`MemoryItem::readable_by`], which applies the
    /// scope's own rule — a user scope is one person's, a workspace scope is its
    /// project's — and then the item's ACL. A scope the reader may not see
    /// contributes nothing, and nothing about it: no count, no id.
    pub fn snapshot_scopes(
        &self,
        session: &Session,
        scopes: &[MemoryScope],
        project_id: Option<&str>,
    ) -> Result<(Vec<MemoryItem>, i64), MemoryError> {
        let mut conn = self.conn.lock().map_err(|_| MemoryError::Storage {
            detail: "the memory graph was left locked".into(),
        })?;
        let transaction = conn.transaction().map_err(storage)?;
        let mut items: Vec<MemoryItem> = Vec::new();
        {
            let mut statement = transaction
                .prepare("SELECT body FROM agent_memory_items WHERE scope_key = ?1")
                .map_err(storage)?;
            for scope in scopes {
                let bodies: Vec<String> = statement
                    .query_map(params![scope.key()], |row| row.get::<_, String>(0))
                    .map_err(storage)?
                    .filter_map(Result::ok)
                    .collect();
                items.extend(
                    bodies
                        .into_iter()
                        .filter_map(|text| serde_json::from_str::<MemoryItem>(&text).ok())
                        .filter(|item| item.readable_by(session, project_id)),
                );
            }
        }
        let cursor = Self::latest_revision(&transaction).unwrap_or(0);
        transaction.commit().map_err(storage)?;
        Ok((items, cursor))
    }

    /// Everything that happened after `cursor`, as this reader may see it.
    ///
    /// ## Why the current row is resolved rather than the logged one
    ///
    /// The log records *that* a subject changed, not what it changed to. Each
    /// entry is answered by reading the row as it stands now and deciding
    /// afresh whether this reader may see it. Two things fall out of that, both
    /// wanted:
    ///
    /// - **Revocation works.** An item whose ACL was narrowed resolves to
    ///   unreadable and the reader is sent [`FeedChange::ItemDropped`], so a
    ///   cached copy is removed from the canvas. A feed that replayed stored
    ///   payloads would cheerfully re-deliver the content that was just
    ///   withdrawn.
    /// - **A burst collapses.** An item written five times is resolved once;
    ///   only its latest revision is on the wire. Coalescing is by subject id,
    ///   keeping the highest revision, which is safe precisely because the
    ///   payload is current state rather than a diff.
    ///
    /// ## Gap recovery
    ///
    /// A cursor below the retention floor cannot be caught up by replay, and
    /// saying so is the whole point: returning the rows that happen to survive
    /// would leave the reader holding items deleted while it was away, with
    /// nothing anywhere to correct it. It gets [`ChangeBatch::reset`] and takes
    /// a fresh snapshot.
    ///
    /// ## Backpressure
    ///
    /// At most `limit` log rows are scanned, capped at [`MAX_BATCH`]. `has_more`
    /// says whether to come straight back rather than wait for the next push,
    /// so a long catch-up costs one round trip per batch instead of one per
    /// write — and a reader that is falling behind simply asks less often. The
    /// backend never pushes rows, so it cannot outrun anybody.
    pub fn changes_since(
        &self,
        session: &Session,
        scope: &MemoryScope,
        project_id: Option<&str>,
        cursor: i64,
        limit: usize,
    ) -> Result<ChangeBatch, MemoryError> {
        let limit = limit.clamp(1, MAX_BATCH);
        let conn = self.conn.lock().map_err(|_| MemoryError::Storage {
            detail: "the memory graph was left locked".into(),
        })?;
        let scope_key = scope.key();
        let latest = Self::latest_revision(&conn).unwrap_or(0);

        let floor: i64 = conn
            .query_row(
                "SELECT floor FROM agent_memory_feed_floor WHERE id = 1",
                [],
                |row| row.get(0),
            )
            .optional()
            .map_err(storage)?
            .unwrap_or(0);
        if cursor < floor {
            return Ok(ChangeBatch::reset_at(latest));
        }

        let scanned: Vec<(i64, String, String, String)> = {
            let mut statement = conn
                .prepare(
                    "SELECT revision, item_id, at, subject_kind FROM agent_memory_log
                     WHERE scope_key = ?1 AND revision > ?2
                     ORDER BY revision LIMIT ?3",
                )
                .map_err(storage)?;
            let collected: Vec<_> = statement
                .query_map(params![scope_key, cursor, limit as i64], |row| {
                    Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
                })
                .map_err(storage)?
                .filter_map(Result::ok)
                .collect();
            collected
        };

        // Nothing in this scope. The cursor still advances to the head of the
        // log: the rows in between belong to other scopes and will never
        // concern this reader, and leaving the cursor behind would make every
        // future poll rescan them.
        let Some(&(last_scanned, ..)) = scanned.last() else {
            return Ok(ChangeBatch::caught_up_at(latest.max(cursor)));
        };

        let has_more: bool = conn
            .query_row(
                "SELECT 1 FROM agent_memory_log
                 WHERE scope_key = ?1 AND revision > ?2 LIMIT 1",
                params![scope_key, last_scanned],
                |_| Ok(true),
            )
            .optional()
            .map_err(storage)?
            .unwrap_or(false);

        // Coalesce to the last mention of each subject, keeping the order the
        // log put them in.
        let mut newest: HashMap<&str, usize> = HashMap::new();
        for (index, (_, subject, _, _)) in scanned.iter().enumerate() {
            newest.insert(subject.as_str(), index);
        }

        // Whether an item is readable, memoised: an edge asks about both of its
        // ends, and a batch of contradictions asks about the same hub node over
        // and over.
        let mut readable: HashMap<String, bool> = HashMap::new();
        let mut entries: Vec<FeedEntry> = Vec::new();

        for (index, (revision, subject, at, subject_kind)) in scanned.iter().enumerate() {
            if newest.get(subject.as_str()) != Some(&index) {
                continue;
            }
            let change = match Subject::parse(subject_kind) {
                Subject::Item => {
                    match Self::read_item(&conn, subject)? {
                        Some(item) if item.readable_by(session, project_id) => {
                            readable.insert(subject.clone(), true);
                            FeedChange::ItemChanged {
                                item: Box::new(item),
                            }
                        }
                        // Gone, tombstoned, or no longer this reader's. The
                        // three are answered identically on purpose — see the
                        // note on `FeedChange::ItemDropped`.
                        _ => {
                            readable.insert(subject.clone(), false);
                            FeedChange::ItemDropped {
                                item_id: subject.clone(),
                            }
                        }
                    }
                }
                Subject::Edge => match Self::read_edge(&conn, subject)? {
                    Some(edge) => {
                        let mut ends_visible = true;
                        for end in [&edge.from_item, &edge.to_item] {
                            let visible = match readable.get(end) {
                                Some(known) => *known,
                                None => {
                                    let seen = Self::read_item(&conn, end)?
                                        .is_some_and(|item| {
                                            item.readable_by(session, project_id)
                                        });
                                    readable.insert(end.clone(), seen);
                                    seen
                                }
                            };
                            if !visible {
                                ends_visible = false;
                                break;
                            }
                        }
                        if ends_visible {
                            FeedChange::EdgeChanged {
                                edge: Box::new(edge),
                            }
                        } else {
                            FeedChange::EdgeDropped {
                                edge_id: subject.clone(),
                            }
                        }
                    }
                    None => FeedChange::EdgeDropped {
                        edge_id: subject.clone(),
                    },
                },
            };
            entries.push(FeedEntry {
                revision: *revision,
                change,
                at: at.clone(),
            });
        }

        Ok(ChangeBatch {
            entries,
            cursor: last_scanned,
            has_more,
            reset: false,
        })
    }

    /// One item by id, with no authorisation applied.
    ///
    /// Takes a borrowed connection rather than locking, because every caller is
    /// already inside the lock — `self.raw` would deadlock on the same
    /// non-reentrant mutex.
    fn read_item(conn: &Connection, item_id: &str) -> Result<Option<MemoryItem>, MemoryError> {
        let body: Option<String> = conn
            .query_row(
                "SELECT body FROM agent_memory_items WHERE item_id = ?1",
                params![item_id],
                |row| row.get(0),
            )
            .optional()
            .map_err(storage)?;
        Ok(match body {
            Some(text) => serde_json::from_str(&text).ok(),
            None => None,
        })
    }

    /// One edge by id, with no authorisation applied.
    fn read_edge(conn: &Connection, edge: &str) -> Result<Option<MemoryEdge>, MemoryError> {
        let row: Option<(String, String, String, String, String, String)> = conn
            .query_row(
                "SELECT from_item, to_item, kind, agent_id, scope_key, created_at
                 FROM agent_memory_edges WHERE edge_id = ?1",
                params![edge],
                |row| {
                    Ok((
                        row.get(0)?,
                        row.get(1)?,
                        row.get(2)?,
                        row.get(3)?,
                        row.get(4)?,
                        row.get(5)?,
                    ))
                },
            )
            .optional()
            .map_err(storage)?;
        Ok(
            row.and_then(|(from, to, kind, agent, key, created)| {
                Some(MemoryEdge {
                    edge_id: edge.to_string(),
                    from_item: from,
                    to_item: to,
                    kind: EdgeKind::parse(&kind)?,
                    agent_id: agent,
                    scope: scope_from_key(&key)?,
                    created_at: created,
                })
            }),
        )
    }
}

fn scope_from_key(key: &str) -> Option<MemoryScope> {
    let (kind, value) = key.split_once(':')?;
    Some(match kind {
        "task" => MemoryScope::Task {
            task_id: value.to_string(),
        },
        "workspace" => MemoryScope::Workspace {
            project_id: value.to_string(),
        },
        "user" => MemoryScope::User {
            user_id: value.to_string(),
        },
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent_runtime::memory::Acl;
    use crate::identity::Role;
    use crate::knowledge::graph::runtime_memory::tests::{item, model, operator, receipt, session};
    use crate::knowledge::graph::runtime_memory::{ArtifactRef, MemoryKind, SourceRef};
    use crate::policy::Classification;

    fn graph() -> MemoryGraph {
        MemoryGraph::in_memory().expect("opens")
    }

    /// Two agents establish the same claim at the same moment. Both are kept,
    /// attributed separately, and neither is silently lost.
    #[test]
    fn two_agents_writing_the_same_claim_both_survive() {
        let graph = graph();

        let mut first = item(MemoryKind::Fact, receipt());
        first.agent_id = "ag-a".into();
        let mut second = item(MemoryKind::Fact, receipt());
        second.agent_id = "ag-b".into();
        assert_eq!(first.content, second.content);

        let a = graph.commit(first.clone(), None, &[]).expect("commits");
        let b = graph.commit(second.clone(), None, &[]).expect("commits");

        assert_ne!(a.item_id, b.item_id, "one agent's claim replaced the other's");
        assert!(!a.duplicate && !b.duplicate);

        let seen = graph
            .snapshot(&session("priya", vec![Role::Employee]), &first.scope, None)
            .expect("reads");
        assert_eq!(seen.len(), 2);
        let authors: BTreeSet<&str> = seen.iter().map(|i| i.agent_id.as_str()).collect();
        assert!(authors.contains("ag-a") && authors.contains("ag-b"));
    }

    /// A writer that read revision 1 and writes against a record now at 2 is
    /// told, rather than overwriting whatever moved it.
    #[test]
    fn a_stale_write_is_a_named_conflict_not_a_silent_overwrite() {
        let graph = graph();
        let original = item(MemoryKind::Fact, operator());
        graph.commit(original.clone(), None, &[]).expect("commits");

        let mut second = original.clone();
        second.content = "changed by the first writer".into();
        graph.commit(second, Some(1), &[]).expect("commits");

        let mut stale = original.clone();
        stale.content = "changed by the second writer".into();
        let refusal = graph
            .commit(stale, Some(1), &[])
            .expect_err("a stale write must be refused");
        assert_eq!(
            refusal,
            MemoryError::RevisionConflict {
                item_id: original.item_id.clone(),
                expected: 1,
                actual: 2,
            }
        );
        assert!(refusal.explain().contains("Re-read"));
    }

    /// The writer never learned whether its commit landed, so it sends it
    /// again. Exactly one item exists.
    #[test]
    fn a_duplicate_commit_after_a_lost_acknowledgement_changes_nothing() {
        let graph = graph();
        let mut once = item(MemoryKind::ToolObservation, receipt());
        once.idempotency_key = Some("run-1:tc-1:create_docx".into());

        let first = graph.commit(once.clone(), None, &[]).expect("commits");
        assert!(!first.duplicate);

        // The retry carries a different item id, as a fresh attempt would.
        let mut retry = once.clone();
        retry.item_id = crate::knowledge::graph::runtime_memory::item_id();
        let second = graph.commit(retry, None, &[]).expect("accepts the retry");

        assert!(second.duplicate, "the retry was applied a second time");
        assert_eq!(second.item_id, first.item_id);
        assert_eq!(
            graph.all_raw().expect("reads").len(),
            1,
            "the same observation was recorded twice"
        );
    }

    /// The item and the cross-store effect are written together or not at all,
    /// and a crash before delivery leaves the effect to be retried.
    #[test]
    fn an_effect_survives_a_crash_between_commit_and_delivery() {
        let graph = graph();
        let fact = item(MemoryKind::Fact, operator());
        graph
            .commit(
                fact.clone(),
                None,
                &[(
                    "conversations".into(),
                    "run-1:fact-1".into(),
                    r#"{"pin":"mi-1"}"#.into(),
                )],
            )
            .expect("commits");

        // — the process dies here, before anything drained the outbox —

        let pending = graph.pending_effects().expect("reads");
        assert_eq!(pending.len(), 1, "the effect was lost with the process");
        assert_eq!(pending[0].target, "conversations");
        assert_eq!(pending[0].idempotency_key, "run-1:fact-1");

        graph
            .mark_delivered(pending[0].outbox_id, "2026-01-02T00:00:00Z")
            .expect("delivers");
        assert!(graph.pending_effects().expect("reads").is_empty());

        // Marking it again is harmless — a deliverer that crashed after the
        // receiver accepted it retries, and must not resurrect the row.
        graph
            .mark_delivered(pending[0].outbox_id, "2026-01-03T00:00:00Z")
            .expect("is idempotent");
        assert!(graph.pending_effects().expect("reads").is_empty());
    }

    /// A correction supersedes without erasing, and the disagreement stays
    /// visible.
    #[test]
    fn a_correction_supersedes_and_keeps_what_it_corrected() {
        let graph = graph();
        let mut wrong = item(MemoryKind::Fact, model());
        wrong.content = "The design pressure is 150 PSI.".into();
        graph.commit(wrong.clone(), None, &[]).expect("commits");

        let mut correction = item(MemoryKind::Correction, operator());
        correction.content = "The design pressure is 10 bar.".into();
        graph
            .correct(correction.clone(), &wrong.item_id)
            .expect("corrects");

        let superseded = graph
            .raw(&wrong.item_id)
            .expect("reads")
            .expect("the corrected item is still there");
        assert_eq!(superseded.status, ItemStatus::Superseded);
        assert_eq!(
            superseded.content, "The design pressure is 150 PSI.",
            "the corrected content was erased"
        );

        let authorised = graph
            .snapshot(&session("priya", vec![Role::Employee]), &wrong.scope, None)
            .expect("reads");
        let edges = graph
            .neighbours(&wrong.item_id, &authorised)
            .expect("traverses");
        assert!(edges.iter().any(|edge| edge.kind == EdgeKind::Supersedes));
    }

    /// And a model cannot undo a person.
    #[test]
    fn a_model_cannot_supersede_an_operators_correction() {
        let graph = graph();
        let correction = item(MemoryKind::Correction, operator());
        graph
            .commit(correction.clone(), None, &[])
            .expect("commits");

        let refusal = graph
            .correct(item(MemoryKind::Fact, model()), &correction.item_id)
            .expect_err("a model overrode a person");
        assert!(matches!(refusal, MemoryError::NotPermitted { .. }));
        assert!(refusal.explain().contains("contradiction"));
    }

    /// Removing a source invalidates what was believed on the strength of it,
    /// and says which items so a cache can drop them.
    #[test]
    fn removing_a_source_invalidates_what_depended_on_it() {
        let graph = graph();
        let mut grounded = item(MemoryKind::Fact, receipt());
        grounded.sources = vec![SourceRef {
            sha256: "ab12".into(),
            locator: "page 12".into(),
            extraction_revision: None,
        }];
        graph.commit(grounded.clone(), None, &[]).expect("commits");

        let unrelated = item(MemoryKind::Fact, receipt());
        graph.commit(unrelated.clone(), None, &[]).expect("commits");

        let affected = graph
            .invalidate_source("ab12", "2026-02-01T00:00:00Z")
            .expect("invalidates");
        assert_eq!(affected, vec![grounded.item_id.clone()]);

        let after = graph.raw(&grounded.item_id).expect("reads").expect("kept");
        assert_eq!(after.status, ItemStatus::Rejected);
        assert!(
            !after.status.usable_as_evidence(),
            "an answer could still be built on a revoked source"
        );

        let untouched = graph.raw(&unrelated.item_id).expect("reads").expect("kept");
        assert_ne!(untouched.status, ItemStatus::Rejected);
    }

    // ── Authorisation ────────────────────────────────────────────────────

    /// The set is resolved before anything traverses, counts or labels.
    #[test]
    fn traversal_counts_and_labels_all_work_inside_the_authorised_set() {
        let graph = graph();

        let mut mine = item(MemoryKind::Fact, operator());
        mine.scope = MemoryScope::Workspace {
            project_id: "unit-four".into(),
        };
        mine.acl = Acl::for_classification(Classification::Internal, Some("unit-four"));

        // Somebody else's, in the same project and at the same classification.
        //
        // The separating axis is the ACL's owner, not the classification:
        // in this product every classification clears both active roles (see
        // `Classification::cleared_roles`), so a test that relied on
        // classification alone would prove nothing about the filter.
        let mut theirs = item(MemoryKind::Fact, operator());
        theirs.scope = MemoryScope::Workspace {
            project_id: "unit-four".into(),
        };
        theirs.acl = Acl {
            owner: Some("someone-else".into()),
            ..Acl::for_classification(Classification::Internal, Some("unit-four"))
        };

        graph.commit(mine.clone(), None, &[]).expect("commits");
        graph.commit(theirs.clone(), None, &[]).expect("commits");
        graph
            .link(MemoryEdge {
                edge_id: edge_id(&mine.item_id, EdgeKind::Supports, &theirs.item_id),
                from_item: mine.item_id.clone(),
                to_item: theirs.item_id.clone(),
                kind: EdgeKind::Supports,
                agent_id: "ag-1".into(),
                scope: mine.scope.clone(),
                created_at: "2026-01-01T00:00:00Z".into(),
            })
            .expect("links");

        let employee = session("priya", vec![Role::Employee]);
        let authorised = graph
            .snapshot(&employee, &mine.scope, Some("unit-four"))
            .expect("reads");

        // The financial item is not in the set at all.
        assert_eq!(authorised.len(), 1);
        assert_eq!(authorised[0].item_id, mine.item_id);

        // The count is taken from the set, so it cannot disclose the other row.
        let counts = MemoryGraph::count(&authorised);
        assert_eq!(counts.get("fact"), Some(&1));

        // And the edge to it is not reported — not even as a redacted entry,
        // which would still say something is there.
        let edges = graph
            .neighbours(&mine.item_id, &authorised)
            .expect("traverses");
        assert!(
            edges.is_empty(),
            "an edge leaked the existence of an unauthorised item"
        );
    }

    #[test]
    fn a_reader_on_another_project_sees_nothing_of_this_one() {
        let graph = graph();
        let mut confined = item(MemoryKind::Fact, operator());
        confined.scope = MemoryScope::Workspace {
            project_id: "unit-four".into(),
        };
        confined.acl = Acl::for_classification(Classification::Internal, Some("unit-four"));
        graph.commit(confined.clone(), None, &[]).expect("commits");

        let reader = session("priya", vec![Role::Employee]);
        assert!(graph
            .snapshot(&reader, &confined.scope, Some("unit-five"))
            .expect("reads")
            .is_empty());
        assert!(
            graph
                .snapshot(&reader, &confined.scope, None)
                .expect("reads")
                .is_empty(),
            "naming no project read a confined item"
        );
    }

    #[test]
    fn a_tombstoned_item_is_not_returned_and_its_row_remains() {
        let graph = graph();
        let doomed = item(MemoryKind::Fact, operator());
        graph.commit(doomed.clone(), None, &[]).expect("commits");
        graph
            .tombstone(&doomed.item_id, "2026-03-01T00:00:00Z")
            .expect("tombstones");

        let seen = graph
            .snapshot(&session("priya", vec![Role::Employee]), &doomed.scope, None)
            .expect("reads");
        assert!(seen.is_empty());

        let row = graph
            .raw(&doomed.item_id)
            .expect("reads")
            .expect("the row is still there so edges resolve");
        assert_eq!(row.status, ItemStatus::Tombstoned);
    }

    // ── Durability ───────────────────────────────────────────────────────

    /// Memory outlives the run that wrote it. Nothing here is keyed by a model
    /// or cleared when one unloads.
    #[test]
    fn memory_survives_a_restart_of_the_process() {
        let dir = tempfile::tempdir().expect("temp dir");
        let written = {
            let graph = MemoryGraph::open(dir.path()).expect("opens");
            let fact = item(MemoryKind::Fact, operator());
            graph.commit(fact.clone(), None, &[]).expect("commits");
            fact
        };

        let reopened = MemoryGraph::open(dir.path()).expect("reopens");
        let seen = reopened
            .snapshot(
                &session("priya", vec![Role::Employee]),
                &written.scope,
                None,
            )
            .expect("reads");
        assert_eq!(seen.len(), 1);
        assert_eq!(seen[0].item_id, written.item_id);
    }

    /// The changefeed cursor is monotonic, which a `MAX(updated_at)` timestamp
    /// is not — two writes in the same second would be indistinguishable.
    #[test]
    fn the_graph_revision_advances_with_every_committed_change() {
        let graph = graph();
        assert_eq!(graph.graph_revision().expect("reads"), 0);

        let first = graph
            .commit(item(MemoryKind::Fact, operator()), None, &[])
            .expect("commits");
        let second = graph
            .commit(item(MemoryKind::Fact, operator()), None, &[])
            .expect("commits");
        assert!(second.graph_revision > first.graph_revision);
        assert_eq!(
            graph.graph_revision().expect("reads"),
            second.graph_revision
        );
    }

    /// Exact bytes go in and exact bytes come out.
    #[test]
    fn source_and_artifact_versions_round_trip_exactly() {
        let graph = graph();
        let mut cited = item(MemoryKind::Fact, receipt());
        cited.sources = vec![SourceRef {
            sha256: "ab12cd34".into(),
            locator: "page 12".into(),
            extraction_revision: Some("rev-3".into()),
        }];
        cited.artifacts = vec![ArtifactRef {
            artifact_id: "art-9".into(),
            revision: 3,
            sha256: "ffee".into(),
        }];
        graph.commit(cited.clone(), None, &[]).expect("commits");

        let back = graph.raw(&cited.item_id).expect("reads").expect("there");
        assert_eq!(back.sources, cited.sources);
        assert_eq!(back.artifacts, cited.artifacts);
        assert_eq!(back.artifacts[0].revision, 3);
    }

    // ── The changefeed ───────────────────────────────────────────────────

    /// An ACL an ordinary employee does not satisfy.
    ///
    /// Narrowed by role rather than by [`Classification`]: every classification
    /// this deployment defines clears both active roles, so changing one would
    /// revoke nothing and the test would pass without testing anything.
    fn admins_only() -> Acl {
        Acl {
            cleared_roles: vec![Role::Administrator],
            project_id: None,
            owner: None,
        }
    }

    /// Reads the whole feed from a cursor, batch by batch, the way a reconnecting
    /// client does. Returns what it ended up holding, and where it got to.
    fn drain(
        graph: &MemoryGraph,
        session: &Session,
        scope: &MemoryScope,
        from: i64,
        batch_size: usize,
    ) -> (Vec<FeedEntry>, i64, bool) {
        let mut cursor = from;
        let mut all = Vec::new();
        let mut reset = false;
        for _ in 0..64 {
            let batch = graph
                .changes_since(session, scope, None, cursor, batch_size)
                .expect("reads the feed");
            if batch.reset {
                reset = true;
                cursor = batch.cursor;
                break;
            }
            all.extend(batch.entries);
            cursor = batch.cursor;
            if !batch.has_more {
                break;
            }
        }
        (all, cursor, reset)
    }

    /// The bug this whole design exists to prevent: a write that lands between
    /// the snapshot and the subscription, and is in neither.
    ///
    /// The snapshot's cursor is read in the same transaction as its contents,
    /// so subscribing from it cannot skip the write that happened next.
    #[test]
    fn a_write_between_snapshot_and_subscribe_is_not_lost() {
        let graph = graph();
        let reader = session("priya", vec![Role::Employee]);
        let scope = item(MemoryKind::Fact, operator()).scope;

        let early = item(MemoryKind::Fact, operator());
        graph.commit(early, None, &[]).expect("commits");

        let snapshot = graph.snapshot_at(&reader, &scope, None).expect("reads");
        assert_eq!(snapshot.items.len(), 1);

        // The window. In a broken design this write is in neither half.
        let mut late = item(MemoryKind::Fact, operator());
        late.content = "written in the window".into();
        graph.commit(late.clone(), None, &[]).expect("commits");

        let (entries, _, reset) = drain(&graph, &reader, &scope, snapshot.cursor, MAX_BATCH);
        assert!(!reset);
        let arrived: Vec<&str> = entries
            .iter()
            .filter_map(|entry| match &entry.change {
                FeedChange::ItemChanged { item } => Some(item.content.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(
            arrived,
            vec!["written in the window"],
            "the write between snapshot and subscribe was lost"
        );
    }

    /// Reconnect after a missed revision: a client that was away and comes back
    /// with an old cursor is caught up, and holds what a fresh snapshot holds.
    #[test]
    fn a_reader_that_missed_revisions_catches_up_to_the_same_graph() {
        let graph = graph();
        let reader = session("priya", vec![Role::Employee]);
        let scope = item(MemoryKind::Fact, operator()).scope;

        let snapshot = graph.snapshot_at(&reader, &scope, None).expect("reads");
        let disconnected_at = snapshot.cursor;

        // Twelve writes the reader never saw.
        let mut written = Vec::new();
        for n in 0..12 {
            let mut fresh = item(MemoryKind::Fact, operator());
            fresh.content = format!("fact {n}");
            graph.commit(fresh.clone(), None, &[]).expect("commits");
            written.push(fresh.item_id);
        }

        // It comes back holding nothing but an old cursor, and drains in small
        // batches — the backpressure path, not one giant message.
        let (entries, cursor, reset) = drain(&graph, &reader, &scope, disconnected_at, 5);
        assert!(!reset, "the log still covered this cursor");

        let caught_up: BTreeSet<&str> = entries
            .iter()
            .filter_map(|entry| match &entry.change {
                FeedChange::ItemChanged { item } => Some(item.item_id.as_str()),
                _ => None,
            })
            .collect();
        let expected: BTreeSet<&str> = written.iter().map(String::as_str).collect();
        assert_eq!(caught_up, expected, "replay did not reconstruct the graph");

        // And it is now level with the truth: a further drain yields nothing.
        let (again, _, _) = drain(&graph, &reader, &scope, cursor, MAX_BATCH);
        assert!(again.is_empty(), "the reader was not actually caught up");
    }

    /// An item written five times arrives once, at its latest revision.
    ///
    /// Not a nicety: without it, a model correcting a fact in a loop makes the
    /// canvas redraw the same node once per write.
    #[test]
    fn a_burst_of_writes_to_one_item_collapses_to_its_latest_revision() {
        let graph = graph();
        let reader = session("priya", vec![Role::Employee]);
        let mut subject = item(MemoryKind::Fact, operator());
        let scope = subject.scope.clone();

        let snapshot = graph.snapshot_at(&reader, &scope, None).expect("reads");
        graph.commit(subject.clone(), None, &[]).expect("commits");
        for n in 1..5 {
            subject.content = format!("revision {n}");
            graph
                .commit(subject.clone(), Some(n), &[])
                .expect("commits");
        }

        let (entries, _, _) = drain(&graph, &reader, &scope, snapshot.cursor, MAX_BATCH);
        assert_eq!(entries.len(), 1, "a burst was replayed write by write");
        match &entries[0].change {
            FeedChange::ItemChanged { item } => {
                assert_eq!(item.revision, 5);
                assert_eq!(item.content, "revision 4");
            }
            other => panic!("expected the item, got {other:?}"),
        }
    }

    /// Revocation reaches a cache. An item whose clearance narrows is not
    /// merely absent from later reads — the reader is told to drop it.
    #[test]
    fn narrowing_an_items_clearance_tells_the_reader_to_drop_it() {
        let graph = graph();
        let reader = session("priya", vec![Role::Employee]);

        let mut open = item(MemoryKind::Fact, operator());
        open.classification = Classification::Internal;
        open.acl = Acl::for_classification(Classification::Internal, None);
        let scope = open.scope.clone();
        graph.commit(open.clone(), None, &[]).expect("commits");

        let snapshot = graph.snapshot_at(&reader, &scope, None).expect("reads");
        assert_eq!(snapshot.items.len(), 1, "the reader could see it to begin with");

        // Narrowed to a clearance this employee does not hold.
        let mut narrowed = open.clone();
        narrowed.acl = admins_only();
        graph.commit(narrowed, Some(1), &[]).expect("commits");

        let (entries, ..) = drain(&graph, &reader, &scope, snapshot.cursor, MAX_BATCH);
        assert_eq!(entries.len(), 1);
        match &entries[0].change {
            FeedChange::ItemDropped { item_id } => assert_eq!(item_id, &open.item_id),
            other => panic!("a revoked item was not dropped — it arrived as {other:?}"),
        }

        // And a fresh snapshot agrees: nothing, not a redacted stub.
        let after = graph.snapshot_at(&reader, &scope, None).expect("reads");
        assert!(after.items.is_empty());
        assert!(after.edges.is_empty());
    }

    /// The label and the content go with the permission. A dropped item must
    /// not carry the thing the reader just lost the right to read.
    #[test]
    fn a_drop_never_carries_the_content_it_revoked() {
        let graph = graph();
        let reader = session("priya", vec![Role::Employee]);
        let mut secret = item(MemoryKind::Fact, operator());
        secret.content = "the design pressure is 10 bar".into();
        let scope = secret.scope.clone();
        graph.commit(secret.clone(), None, &[]).expect("commits");

        let snapshot = graph.snapshot_at(&reader, &scope, None).expect("reads");
        let mut narrowed = secret.clone();
        narrowed.acl = admins_only();
        graph.commit(narrowed, Some(1), &[]).expect("commits");

        let (entries, ..) = drain(&graph, &reader, &scope, snapshot.cursor, MAX_BATCH);
        let wire = serde_json::to_string(&entries).expect("serialises");
        assert!(
            !wire.contains("10 bar"),
            "the revoked content was on the wire: {wire}"
        );
    }

    /// An edge whose far end the reader may not see is not reported at all —
    /// not as a stub, which would still disclose that something is over there.
    #[test]
    fn an_edge_into_the_dark_is_dropped_rather_than_stubbed() {
        let graph = graph();
        let reader = session("priya", vec![Role::Employee]);

        let visible = item(MemoryKind::Fact, operator());
        let mut hidden = item(MemoryKind::Fact, operator());
        hidden.scope = visible.scope.clone();
        hidden.content = "a restricted finding".into();
        hidden.acl = admins_only();
        let scope = visible.scope.clone();

        graph.commit(visible.clone(), None, &[]).expect("commits");
        graph.commit(hidden.clone(), None, &[]).expect("commits");

        let snapshot = graph.snapshot_at(&reader, &scope, None).expect("reads");
        assert_eq!(snapshot.items.len(), 1, "only the visible end is readable");

        graph
            .link(MemoryEdge {
                edge_id: edge_id(&visible.item_id, EdgeKind::Contradicts, &hidden.item_id),
                from_item: visible.item_id.clone(),
                to_item: hidden.item_id.clone(),
                kind: EdgeKind::Contradicts,
                agent_id: "ag-a".into(),
                scope: scope.clone(),
                created_at: "2026-01-01T00:00:00Z".into(),
            })
            .expect("links");

        let (entries, ..) = drain(&graph, &reader, &scope, snapshot.cursor, MAX_BATCH);
        assert_eq!(entries.len(), 1, "expected exactly the edge's change");
        assert!(
            matches!(entries[0].change, FeedChange::EdgeDropped { .. }),
            "an edge into a hidden item was delivered: {:?}",
            entries[0].change
        );
        let after = graph.snapshot_at(&reader, &scope, None).expect("reads");
        assert!(after.edges.is_empty());
    }

    /// A link does reach the feed when both ends are visible. This is the
    /// regression the `link` change fixed: an edge used to land in the table
    /// and never be announced.
    #[test]
    fn a_link_between_visible_items_reaches_the_feed() {
        let graph = graph();
        let reader = session("priya", vec![Role::Employee]);
        let a = item(MemoryKind::Fact, operator());
        let mut b = item(MemoryKind::Fact, operator());
        b.scope = a.scope.clone();
        let scope = a.scope.clone();
        graph.commit(a.clone(), None, &[]).expect("commits");
        graph.commit(b.clone(), None, &[]).expect("commits");

        let snapshot = graph.snapshot_at(&reader, &scope, None).expect("reads");
        let link = MemoryEdge {
            edge_id: edge_id(&a.item_id, EdgeKind::Supports, &b.item_id),
            from_item: a.item_id.clone(),
            to_item: b.item_id.clone(),
            kind: EdgeKind::Supports,
            agent_id: "ag-a".into(),
            scope: scope.clone(),
            created_at: "2026-01-01T00:00:00Z".into(),
        };
        graph.link(link.clone()).expect("links");
        // Written twice: idempotent in the table, and it must not announce the
        // second one or every retry becomes a redraw.
        graph.link(link).expect("links again");

        let (entries, ..) = drain(&graph, &reader, &scope, snapshot.cursor, MAX_BATCH);
        assert_eq!(entries.len(), 1, "a repeat link was announced twice");
        assert!(matches!(entries[0].change, FeedChange::EdgeChanged { .. }));
    }

    /// A cursor the log can no longer cover is answered with "start again",
    /// never with the rows that happen to survive.
    #[test]
    fn a_cursor_below_the_retention_floor_asks_for_a_fresh_snapshot() {
        let graph = graph();
        let reader = session("priya", vec![Role::Employee]);
        let scope = item(MemoryKind::Fact, operator()).scope;
        graph
            .commit(item(MemoryKind::Fact, operator()), None, &[])
            .expect("commits");

        {
            let conn = graph.conn.lock().expect("locks");
            conn.execute("UPDATE agent_memory_feed_floor SET floor = 5 WHERE id = 1", [])
                .expect("sets the floor");
        }

        let batch = graph
            .changes_since(&reader, &scope, None, 0, MAX_BATCH)
            .expect("reads");
        assert!(batch.reset, "a cursor below the floor was silently continued");
        assert!(batch.entries.is_empty());
    }

    /// Re-reading from the same cursor returns the same thing. A reader that
    /// retries after a dropped reply must not skip or double-count.
    #[test]
    fn reading_the_same_cursor_twice_is_the_same_answer() {
        let graph = graph();
        let reader = session("priya", vec![Role::Employee]);
        let scope = item(MemoryKind::Fact, operator()).scope;
        let snapshot = graph.snapshot_at(&reader, &scope, None).expect("reads");
        graph
            .commit(item(MemoryKind::Fact, operator()), None, &[])
            .expect("commits");

        let first = graph
            .changes_since(&reader, &scope, None, snapshot.cursor, MAX_BATCH)
            .expect("reads");
        let second = graph
            .changes_since(&reader, &scope, None, snapshot.cursor, MAX_BATCH)
            .expect("reads");
        assert_eq!(first, second, "the feed is not replayable from a cursor");
    }

    /// One scope's writes never appear in another's feed, and never in its
    /// snapshot. The scope is bound into the query, not filtered afterwards.
    #[test]
    fn another_scopes_writes_stay_out_of_this_feed() {
        let graph = graph();
        let reader = session("priya", vec![Role::Employee]);
        let mine = item(MemoryKind::Fact, operator());
        let scope = mine.scope.clone();
        let mut theirs = item(MemoryKind::Fact, operator());
        theirs.scope = MemoryScope::Task {
            task_id: "some-other-task".into(),
        };

        let snapshot = graph.snapshot_at(&reader, &scope, None).expect("reads");
        graph.commit(theirs, None, &[]).expect("commits");
        graph.commit(mine.clone(), None, &[]).expect("commits");

        let (entries, ..) = drain(&graph, &reader, &scope, snapshot.cursor, MAX_BATCH);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].change.subject(), mine.item_id);
    }

    /// An idle scope still advances its cursor past other scopes' traffic,
    /// rather than rescanning the same empty range forever.
    #[test]
    fn an_idle_scope_still_advances_its_cursor() {
        let graph = graph();
        let reader = session("priya", vec![Role::Employee]);
        let scope = MemoryScope::Task {
            task_id: "quiet".into(),
        };
        let before = graph
            .changes_since(&reader, &scope, None, 0, MAX_BATCH)
            .expect("reads");

        for _ in 0..3 {
            graph
                .commit(item(MemoryKind::Fact, operator()), None, &[])
                .expect("commits");
        }

        let after = graph
            .changes_since(&reader, &scope, None, before.cursor, MAX_BATCH)
            .expect("reads");
        assert!(after.entries.is_empty());
        assert!(after.cursor > before.cursor, "the idle cursor did not move");
        assert!(!after.reset);
    }
}
