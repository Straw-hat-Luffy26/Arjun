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

use super::receipts::{OutboxConsumer, ReceiptLedger};
use super::runtime_feed::{ChangeBatch, FeedChange, FeedEntry, MemorySnapshot, Subject, MAX_BATCH};
use super::runtime_memory::{
    admit, edge_id, may_supersede, Authority, Basis, EdgeKind, ItemStatus, MemoryEdge, MemoryItem,
    MemoryScope, Provenance, ReceiptVerdict, MEMORY_SCHEMA_VERSION,
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

/// What [`MemoryGraph::commit_within`] did.
enum Written {
    /// The same write had already been applied; nothing changed.
    Duplicate(Committed),
    /// A new revision was written.
    Fresh(Committed),
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
    /// How many deliveries have been tried and failed.
    #[serde(default)]
    pub attempts: i64,
    /// Why the last one failed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_error: Option<String>,
}

/// What one pass of the outbox deliverer did.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DeliveryReport {
    pub delivered: usize,
    pub failed: usize,
    /// Rows naming a target nothing was offered to deliver to. Left pending,
    /// not dropped: a consumer that is registered later still gets them.
    pub no_consumer: usize,
}

/// What a migrated record's write did.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum MigratedWrite {
    /// First time across.
    Inserted,
    /// Already here, with the same content. A re-run changes nothing.
    Unchanged,
    /// Already here, and the legacy record has changed since: a new revision.
    Updated,
    /// It was rolled back, and is being brought back.
    Restored,
}

/// One immutable revision of an item, as it was written.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ItemVersion {
    pub item_id: String,
    pub revision: u64,
    /// The changefeed position this revision was written at.
    pub graph_revision: i64,
    pub item: MemoryItem,
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
    /// Where a tool receipt is resolved before it may admit anything.
    ///
    /// `None` is a graph with no event log in reach, and then *no* receipt
    /// admits: an item claiming one stays a proposal and says why. There is no
    /// fallback that trusts the shape of a receipt, because the shape is
    /// exactly what anybody can forge. See [`super::receipts`].
    receipts: std::sync::RwLock<Option<Arc<dyn ReceiptLedger>>>,
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
            receipts: std::sync::RwLock::new(None),
        })
    }

    /// An in-memory graph, for tests.
    pub fn in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()?;
        Self::prepare(&conn)?;
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
            watcher: Mutex::new(None),
            receipts: std::sync::RwLock::new(None),
        })
    }

    /// Installs the event log receipts are resolved against.
    pub fn set_receipts(&self, ledger: Arc<dyn ReceiptLedger>) {
        if let Ok(mut held) = self.receipts.write() {
            *held = Some(ledger);
        }
    }

    /// The same, as a builder.
    pub fn with_receipts(self, ledger: Arc<dyn ReceiptLedger>) -> Self {
        self.set_receipts(ledger);
        self
    }

    /// What the event log says about the receipt this item claims.
    ///
    /// Asked *outside* the graph's transaction: the event log is its own
    /// connection with its own lock, and holding this one across it would make
    /// the two stores' locks an ordering nobody wrote down.
    fn verdict_for(&self, item: &MemoryItem) -> ReceiptVerdict {
        let Provenance::ToolReceipt {
            run_id,
            tool,
            event_seq,
            output_sha256,
        } = &item.provenance
        else {
            return ReceiptVerdict::NotAReceipt;
        };
        let Some(output) = output_sha256.as_deref() else {
            return ReceiptVerdict::Refused {
                because: "it names no output hash".to_string(),
            };
        };
        let ledger = self.receipts.read().ok().and_then(|held| held.clone());
        match ledger {
            None => ReceiptVerdict::Refused {
                because: "no event log is in reach to check it against".to_string(),
            },
            Some(ledger) => match ledger.verify(run_id, *event_seq, tool, output) {
                Ok(()) => ReceiptVerdict::Verified,
                Err(because) => ReceiptVerdict::Refused { because },
            },
        }
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

        // ── P02: history, dependencies, delivery, migration ──────────────
        //
        // All additive. Nothing here drops, rewrites or deletes a row that
        // was already written; a database opened by the previous build keeps
        // every row it had, and gains the columns and tables below.
        for (statement, what) in [
            (
                "ALTER TABLE agent_memory_log ADD COLUMN item_revision INTEGER",
                "the changefeed's item revision column",
            ),
            (
                "ALTER TABLE agent_memory_outbox ADD COLUMN attempts INTEGER NOT NULL DEFAULT 0",
                "the outbox's attempt count",
            ),
            (
                "ALTER TABLE agent_memory_outbox ADD COLUMN last_error TEXT",
                "the outbox's last error",
            ),
        ] {
            if let Err(error) = conn.execute(statement, []) {
                if !error.to_string().contains("duplicate column name") {
                    return Err(error).with_context(|| format!("{what} could not be added"));
                }
            }
        }

        conn.execute_batch(
            "
            -- Every revision of every item, as it was written, never updated.
            --
            -- What a historical read and a replay read from. `agent_memory_items`
            -- holds only the latest body, so a context manifest that recorded
            -- 'graph revision 12' could not be reproduced once the item moved to
            -- revision 13 -- and the compiler was labelling latest rows with an
            -- older cursor. This is the immutable half.
            CREATE TABLE IF NOT EXISTS agent_memory_versions (
                item_id        TEXT NOT NULL,
                revision       INTEGER NOT NULL,
                graph_revision INTEGER NOT NULL,
                scope_key      TEXT NOT NULL,
                status         TEXT NOT NULL,
                body           TEXT NOT NULL,
                PRIMARY KEY (item_id, revision)
            );
            CREATE INDEX IF NOT EXISTS agent_memory_versions_scope
                ON agent_memory_versions (scope_key, graph_revision);

            -- Which items rest on which, so a change reaches what depends on it
            -- without scanning every body.
            CREATE TABLE IF NOT EXISTS agent_memory_dependencies (
                item_id    TEXT NOT NULL,
                depends_on TEXT NOT NULL,
                PRIMARY KEY (item_id, depends_on)
            );
            CREATE INDEX IF NOT EXISTS agent_memory_dependencies_on
                ON agent_memory_dependencies (depends_on);

            -- One row per migration or rollback pass. A pass with no
            -- `finished_at` was interrupted; running it again is safe because
            -- every migrated write is addressed by what it came from.
            CREATE TABLE IF NOT EXISTS agent_memory_migration_runs (
                run_id      TEXT PRIMARY KEY,
                source      TEXT NOT NULL,
                action      TEXT NOT NULL,
                started_at  TEXT NOT NULL,
                finished_at TEXT,
                report      TEXT
            );

            -- Items written before versions were kept get their current body as
            -- the one version known. What their earlier revisions said was never
            -- stored, and is not invented here.
            INSERT OR IGNORE INTO agent_memory_versions
                (item_id, revision, graph_revision, scope_key, status, body)
            SELECT i.item_id, i.revision,
                   COALESCE((SELECT MAX(l.revision) FROM agent_memory_log l
                             WHERE l.item_id = i.item_id AND l.subject_kind = 'item'), 0),
                   i.scope_key, i.status, i.body
              FROM agent_memory_items i;
            ",
        )?;
        Ok(())
    }

    // ── One revision, written whole ─────────────────────────────────────

    /// Writes one revision of an item inside the caller's transaction: the
    /// current row, its immutable version, its dependency edges and its
    /// changefeed entry. Returns the changefeed position.
    ///
    /// The single place an item is written. Every path that changes an item --
    /// a commit, a supersede, a tombstone, a source invalidation, a revocation,
    /// staleness -- comes through here, so none of them can change a row
    /// without a new revision, a version a replay can read, and a feed entry a
    /// subscriber sees. The previous code updated `status` and `body` in place
    /// on four of those paths and left the revision where it was, so an
    /// optimistic writer holding "revision 3" could not tell the item had been
    /// superseded under it.
    fn write_revision(
        transaction: &Connection,
        item: &MemoryItem,
        change: &str,
    ) -> Result<i64, MemoryError> {
        let body = serde_json::to_string(item).map_err(storage)?;
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
                "INSERT INTO agent_memory_log
                     (item_id, scope_key, change, at, subject_kind, item_revision)
                 VALUES (?1, ?2, ?3, ?4, 'item', ?5)",
                params![item.item_id, scope_key, change, item.updated_at, item.revision as i64],
            )
            .map_err(storage)?;
        let graph_revision = transaction.last_insert_rowid();
        transaction
            .execute(
                "INSERT INTO agent_memory_versions
                     (item_id, revision, graph_revision, scope_key, status, body)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                params![
                    item.item_id,
                    item.revision as i64,
                    graph_revision,
                    scope_key,
                    item.status.as_str(),
                    body,
                ],
            )
            .map_err(storage)?;
        transaction
            .execute(
                "DELETE FROM agent_memory_dependencies WHERE item_id = ?1",
                params![item.item_id],
            )
            .map_err(storage)?;
        for dependency in &item.depends_on {
            transaction
                .execute(
                    "INSERT OR IGNORE INTO agent_memory_dependencies (item_id, depends_on)
                     VALUES (?1, ?2)",
                    params![item.item_id, dependency.item_id],
                )
                .map_err(storage)?;
        }
        Ok(graph_revision)
    }

    /// The items that rest on `item_id`: those that pinned it as a dependency,
    /// and those linked to it as derived from or citing it.
    fn dependents_within(transaction: &Connection, item_id: &str) -> Result<Vec<String>, MemoryError> {
        let mut statement = transaction
            .prepare(
                "SELECT item_id FROM agent_memory_dependencies WHERE depends_on = ?1
                 UNION
                 SELECT from_item FROM agent_memory_edges
                  WHERE to_item = ?1 AND kind IN ('derivedFrom', 'cites')",
            )
            .map_err(storage)?;
        let found = statement
            .query_map(params![item_id], |row| row.get::<_, String>(0))
            .map_err(storage)?
            .filter_map(Result::ok)
            .collect();
        Ok(found)
    }

    /// Marks everything that rests on `roots`, transitively, as stale.
    ///
    /// Each gets a new revision -- a stale item is a changed item, and a reader
    /// holding the old one must be told -- and nothing is deleted. Items already
    /// superseded, rejected, tombstoned or stale keep that more specific answer,
    /// and the walk still continues through them, so a chain does not stop at a
    /// link that happened to be withdrawn already.
    fn propagate_staleness(
        transaction: &Connection,
        roots: &[String],
        because: &str,
        at: &str,
    ) -> Result<Vec<String>, MemoryError> {
        let mut queue: std::collections::VecDeque<String> = roots.iter().cloned().collect();
        let mut seen: BTreeSet<String> = roots.iter().cloned().collect();
        let mut marked = Vec::new();
        while let Some(changed) = queue.pop_front() {
            for dependent in Self::dependents_within(transaction, &changed)? {
                if !seen.insert(dependent.clone()) {
                    continue;
                }
                if let Some(mut item) = Self::read_item(transaction, &dependent)? {
                    if item.status.can_go_stale() {
                        item.status = ItemStatus::Stale;
                        item.revision += 1;
                        item.updated_at = at.to_string();
                        Self::write_revision(
                            transaction,
                            &item,
                            &format!("stale: {because} ({changed})"),
                        )?;
                        marked.push(dependent.clone());
                    }
                }
                queue.push_back(dependent);
            }
        }
        Ok(marked)
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
        item: MemoryItem,
        expected_revision: Option<u64>,
        effects: &[(String, String, String)],
    ) -> Result<Committed, MemoryError> {
        // The receipt is resolved before the graph is locked -- see
        // `verdict_for` for why the two stores' locks are never held together.
        let verdict = self.verdict_for(&item);

        let mut conn = self.conn.lock().map_err(|_| MemoryError::Storage {
            detail: "the memory graph was left locked by a failed write".into(),
        })?;
        let transaction = conn.transaction().map_err(storage)?;

        let committed = match Self::commit_within(&transaction, item, expected_revision, &verdict)? {
            Written::Duplicate(committed) => return Ok(committed),
            Written::Fresh(committed) => committed,
        };

        for (target, key, payload) in effects {
            transaction
                .execute(
                    "INSERT INTO agent_memory_outbox
                         (target, idempotency_key, payload, created_at)
                     VALUES (?1, ?2, ?3, ?4)",
                    params![target, key, payload, chrono::Utc::now().to_rfc3339()],
                )
                .map_err(storage)?;
        }

        transaction.commit().map_err(storage)?;
        // The connection lock goes before the watcher runs. A watcher that
        // reached back into the graph would otherwise deadlock on a mutex this
        // frame is still holding, and "notifying somebody wedges the writer" is
        // a failure that only shows up under load.
        drop(conn);
        self.announce(committed.graph_revision);
        Ok(committed)
    }

    /// The whole of a commit's decision, inside a transaction the caller owns.
    ///
    /// Shared by [`Self::commit`] and [`Self::correct`], so a correction is one
    /// transaction rather than a commit followed by a separate supersede --
    /// which is what it was, and a crash between the two left a correction
    /// standing beside the fact it corrected, both current.
    fn commit_within(
        transaction: &Connection,
        mut item: MemoryItem,
        expected_revision: Option<u64>,
        verdict: &ReceiptVerdict,
    ) -> Result<Written, MemoryError> {
        let outcome = admit(&item, verdict);
        item.status = outcome.status;
        // Computed, never taken from the writer. See `Basis`.
        item.basis = Some(Basis::of(&item.provenance));
        let mut because = outcome.because;

        // A key already applied is the same write arriving twice.
        if let Some(key) = item.idempotency_key.as_deref() {
            let held: Option<(String, i64, String)> = transaction
                .query_row(
                    "SELECT item_id, revision, status FROM agent_memory_items
                      WHERE idempotency_key = ?1",
                    params![key],
                    |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
                )
                .optional()
                .map_err(storage)?;
            if let Some((item_id, revision, status)) = held {
                let graph_revision = Self::latest_revision(transaction).unwrap_or(0);
                return Ok(Written::Duplicate(Committed {
                    item_id,
                    revision: revision as u64,
                    graph_revision,
                    duplicate: true,
                    // What is stored, not what this attempt would have been:
                    // a retry does not re-decide a settled write.
                    status: ItemStatus::parse(&status).unwrap_or(outcome.status),
                    because,
                }));
            }
        }

        let existing = Self::read_item(transaction, &item.item_id)?;
        let existing_revision = existing.as_ref().map(|held| held.revision);

        match (existing_revision, expected_revision) {
            (Some(actual), Some(expected)) if actual != expected => {
                return Err(MemoryError::RevisionConflict {
                    item_id: item.item_id,
                    expected,
                    actual,
                })
            }
            // A writer that expected nothing and found something is the same
            // conflict: it believed it was creating this item.
            (Some(actual), None) => {
                return Err(MemoryError::RevisionConflict {
                    item_id: item.item_id,
                    expected: 0,
                    actual,
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

        // Explicit per-record authority. A record the graph holds as a copy of
        // a legacy store is written only by the migration that copies it; a
        // second writer here would race the legacy writer that is still live.
        if let Some(held) = &existing {
            if let Authority::Legacy { store } = &held.authority {
                if !matches!(item.provenance, Provenance::Migrated { .. }) {
                    return Err(MemoryError::NotPermitted {
                        because: format!(
                            "{} is a copy of a record in {store}, which is still where it is \
                             written. Change it there; the graph follows at the next migration \
                             pass, and becomes the authority when {store} is cut over.",
                            held.item_id
                        ),
                    });
                }
            }
        }

        // ── Revalidate the inputs, and inherit their restrictions ────────
        //
        // Every pinned dependency is read *now*, in this transaction. An input
        // that moved on, was withdrawn or disappeared while the work was in
        // flight makes this result stale rather than silently current; and a
        // derived record is never less restricted than what it was derived
        // from. Plan §6.
        let mut stale_because: Vec<String> = Vec::new();
        for dependency in item.depends_on.clone() {
            match Self::read_item(transaction, &dependency.item_id)? {
                None => stale_because.push(format!("its input {} no longer exists", dependency.item_id)),
                Some(input) => {
                    Self::inherit_restrictions(&mut item, &input)?;
                    if input.status.withdrawn() {
                        stale_because.push(format!(
                            "its input {} is now {}",
                            input.item_id,
                            input.status.as_str()
                        ));
                    } else if input.revision != dependency.revision {
                        stale_because.push(format!(
                            "its input {} moved from revision {} to {} while it was being worked out",
                            input.item_id, dependency.revision, input.revision
                        ));
                    }
                }
            }
        }
        if !stale_because.is_empty() && item.status.can_go_stale() {
            item.status = ItemStatus::Stale;
            because = format!("{because}; published as stale because {}", stale_because.join("; "));
        }

        item.revision = existing_revision.map(|held| held + 1).unwrap_or(1);
        let graph_revision = Self::write_revision(
            transaction,
            &item,
            if existing.is_some() { "updated" } else { "created" },
        )?;

        Ok(Written::Fresh(Committed {
            item_id: item.item_id,
            revision: item.revision,
            graph_revision,
            duplicate: false,
            status: item.status,
            because,
        }))
    }

    /// Narrows `item` to be no more visible than `input`.
    ///
    /// The restrictive combination, field by field: the cleared roles are the
    /// intersection; a project or an owner the input is confined to confines
    /// the result too, and two different ones cannot be combined at all; a
    /// classification above `Internal` is carried; and anybody revoked from the
    /// input is revoked from what was derived from it.
    fn inherit_restrictions(item: &mut MemoryItem, input: &MemoryItem) -> Result<(), MemoryError> {
        item.acl
            .cleared_roles
            .retain(|role| input.acl.cleared_roles.contains(role));
        match (&item.acl.project_id, &input.acl.project_id) {
            (Some(mine), Some(theirs)) if mine != theirs => {
                return Err(MemoryError::NotPermitted {
                    because: format!(
                        "this combines material confined to project {mine} with material confined \
                         to project {theirs}, and nothing may be readable in both"
                    ),
                })
            }
            (None, Some(theirs)) => item.acl.project_id = Some(theirs.clone()),
            _ => {}
        }
        match (&item.acl.owner, &input.acl.owner) {
            (Some(mine), Some(theirs)) if mine != theirs => {
                return Err(MemoryError::NotPermitted {
                    because: "this combines two different people's own material".to_string(),
                })
            }
            (None, Some(theirs)) => item.acl.owner = Some(theirs.clone()),
            _ => {}
        }
        if item.classification == crate::policy::Classification::Internal
            && input.classification != crate::policy::Classification::Internal
        {
            item.classification = input.classification;
        }
        for reader in &input.revoked_readers {
            if !item.revoked_readers.contains(reader) {
                item.revoked_readers.push(reader.clone());
            }
        }
        Ok(())
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
    ///
    /// ## One transaction
    ///
    /// The correction, the supersede, the edge and the staleness of everything
    /// derived from the corrected item land together or not at all. They were
    /// two transactions, and a crash between them left the correction and the
    /// fact it corrected both current, with nothing marked stale.
    ///
    /// `expected_revision` is the revision of the item being corrected that the
    /// corrector read. A correction of something that has moved on since is a
    /// [`MemoryError::RevisionConflict`], for the same reason a commit is.
    pub fn correct(
        &self,
        correction: MemoryItem,
        supersedes_item: &str,
    ) -> Result<Committed, MemoryError> {
        self.correct_at(correction, supersedes_item, None)
    }

    /// [`Self::correct`], checked against the revision the corrector read.
    pub fn correct_at(
        &self,
        correction: MemoryItem,
        supersedes_item: &str,
        expected_revision: Option<u64>,
    ) -> Result<Committed, MemoryError> {
        let verdict = self.verdict_for(&correction);
        let mut conn = self.conn.lock().map_err(|_| MemoryError::Storage {
            detail: "the memory graph was left locked by a failed write".into(),
        })?;
        let transaction = conn.transaction().map_err(storage)?;

        let mut existing = Self::read_item(&transaction, supersedes_item)?.ok_or(
            MemoryError::NotFound {
                item_id: supersedes_item.to_string(),
            },
        )?;
        if let Some(expected) = expected_revision {
            if existing.revision != expected {
                return Err(MemoryError::RevisionConflict {
                    item_id: supersedes_item.to_string(),
                    expected,
                    actual: existing.revision,
                });
            }
        }
        if let Authority::Legacy { store } = &existing.authority {
            return Err(MemoryError::NotPermitted {
                because: format!(
                    "{supersedes_item} is a copy of a record in {store}; correct it there until \
                     {store} is cut over to the graph"
                ),
            });
        }
        may_supersede(&correction, &existing)
            .map_err(|because| MemoryError::NotPermitted { because })?;

        let mut correction = correction;
        correction.supersedes = Some(supersedes_item.to_string());
        let at = correction.updated_at.clone();
        let committed = match Self::commit_within(&transaction, correction.clone(), None, &verdict)? {
            // Corrected before, by this same write. Nothing more to do.
            Written::Duplicate(committed) => return Ok(committed),
            Written::Fresh(committed) => committed,
        };

        existing.status = ItemStatus::Superseded;
        existing.revision += 1;
        existing.updated_at = at.clone();
        Self::write_revision(&transaction, &existing, "superseded")?;

        let id = edge_id(&correction.item_id, EdgeKind::Supersedes, supersedes_item);
        let inserted = transaction
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
                    at,
                ],
            )
            .map_err(storage)?;
        if inserted > 0 {
            transaction
                .execute(
                    "INSERT INTO agent_memory_log
                         (item_id, scope_key, change, at, subject_kind)
                     VALUES (?1, ?2, 'linked', ?3, 'edge')",
                    params![id, correction.scope.key(), at],
                )
                .map_err(storage)?;
        }

        // What was built on the corrected item is no longer standing on
        // anything a person agrees with.
        Self::propagate_staleness(
            &transaction,
            &[supersedes_item.to_string()],
            "an input was corrected",
            &at,
        )?;

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
    ///
    /// A new revision, so every reader holding the old one is told; its earlier
    /// versions stay in the history. What was derived from it goes stale.
    pub fn tombstone(&self, item_id: &str, at: &str) -> Result<(), MemoryError> {
        let mut conn = self.conn.lock().map_err(|_| MemoryError::Storage {
            detail: "the memory graph was left locked by a failed write".into(),
        })?;
        let transaction = conn.transaction().map_err(storage)?;
        let Some(mut item) = Self::read_item(&transaction, item_id)? else {
            return Err(MemoryError::NotFound {
                item_id: item_id.to_string(),
            });
        };
        item.status = ItemStatus::Tombstoned;
        item.revision += 1;
        item.updated_at = at.to_string();
        Self::write_revision(&transaction, &item, "tombstoned")?;
        Self::propagate_staleness(&transaction, &[item_id.to_string()], "an input was removed", at)?;
        let revision = Self::latest_revision(&transaction).unwrap_or(0);
        transaction.commit().map_err(storage)?;
        drop(conn);
        self.announce(revision);
        Ok(())
    }

    /// Withdraws one person's access to one item -- and to everything derived
    /// from it, which inherits the restriction.
    ///
    /// A per-reader revocation: narrower than changing the item's ACL, and it
    /// wins over every role that person holds. A new revision, so the
    /// changefeed tells that reader's window to drop its copy
    /// ([`FeedChange::ItemDropped`] carries no content).
    pub fn revoke_reader(&self, item_id: &str, user_id: &str, at: &str) -> Result<Vec<String>, MemoryError> {
        let mut conn = self.conn.lock().map_err(|_| MemoryError::Storage {
            detail: "the memory graph was left locked by a failed write".into(),
        })?;
        let transaction = conn.transaction().map_err(storage)?;
        if Self::read_item(&transaction, item_id)?.is_none() {
            return Err(MemoryError::NotFound {
                item_id: item_id.to_string(),
            });
        }

        let mut queue = std::collections::VecDeque::from([item_id.to_string()]);
        let mut seen: BTreeSet<String> = BTreeSet::from([item_id.to_string()]);
        let mut revoked = Vec::new();
        while let Some(next) = queue.pop_front() {
            if let Some(mut item) = Self::read_item(&transaction, &next)? {
                if !item.revoked_readers.iter().any(|held| held == user_id) {
                    item.revoked_readers.push(user_id.to_string());
                    item.revision += 1;
                    item.updated_at = at.to_string();
                    Self::write_revision(&transaction, &item, "readerRevoked")?;
                    revoked.push(next.clone());
                }
            }
            for dependent in Self::dependents_within(&transaction, &next)? {
                if seen.insert(dependent.clone()) {
                    queue.push_back(dependent);
                }
            }
        }

        let revision = Self::latest_revision(&transaction).unwrap_or(0);
        transaction.commit().map_err(storage)?;
        drop(conn);
        self.announce(revision);
        Ok(revoked)
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
            .filter(|item| item.is_readable() && !matches!(item.status, ItemStatus::Rejected))
            .collect();

        let mut conn = self.conn.lock().map_err(|_| MemoryError::Storage {
            detail: "the memory graph was left locked by a failed write".into(),
        })?;
        let transaction = conn.transaction().map_err(storage)?;
        let mut ids = Vec::new();
        for item in affected {
            // Re-read inside the transaction: the list above was taken without
            // the lock, and the revision written must follow the one stored.
            let Some(mut item) = Self::read_item(&transaction, &item.item_id)? else {
                continue;
            };
            item.status = ItemStatus::Rejected;
            item.revision += 1;
            item.updated_at = at.to_string();
            Self::write_revision(&transaction, &item, "sourceInvalidated")?;
            ids.push(item.item_id);
        }
        // And what rested on those, transitively.
        Self::propagate_staleness(&transaction, &ids, "a source it rests on was withdrawn", at)?;
        let revision = Self::latest_revision(&transaction).unwrap_or(0);
        transaction.commit().map_err(storage)?;
        drop(conn);
        // Announced even when nothing matched: the watcher is a hint to go and
        // look, and a spurious one costs a cheap indexed read. A missed one
        // costs a stale canvas.
        self.announce(revision);
        Ok(ids)
    }

    /// Marks stale every item whose evidence came from a source version that
    /// has since been revised (P07).
    ///
    /// Stale, not rejected: the claim was true of the version it read, and a
    /// revised SOP may well say the same thing. What a reader must not do is
    /// treat it as established about the current version without looking, and
    /// the stale label says exactly that. Withdrawal and revocation are
    /// [`Self::invalidate_source`].
    pub fn source_revised(&self, sha256: &str, at: &str) -> Result<Vec<String>, MemoryError> {
        let affected: Vec<MemoryItem> = self
            .all_raw()?
            .into_iter()
            .filter(|item| item.sources.iter().any(|source| source.sha256 == sha256))
            .filter(|item| item.is_readable() && item.status.can_go_stale())
            .collect();

        let mut conn = self.conn.lock().map_err(|_| MemoryError::Storage {
            detail: "the memory graph was left locked by a failed write".into(),
        })?;
        let transaction = conn.transaction().map_err(storage)?;
        let mut ids = Vec::new();
        for item in affected {
            let Some(mut item) = Self::read_item(&transaction, &item.item_id)? else {
                continue;
            };
            if !item.status.can_go_stale() {
                continue;
            }
            item.status = ItemStatus::Stale;
            item.revision += 1;
            item.updated_at = at.to_string();
            Self::write_revision(&transaction, &item, "stale: the source version it cites was revised")?;
            ids.push(item.item_id);
        }
        Self::propagate_staleness(&transaction, &ids, "a source it rests on was revised", at)?;
        let revision = Self::latest_revision(&transaction).unwrap_or(0);
        transaction.commit().map_err(storage)?;
        drop(conn);
        self.announce(revision);
        Ok(ids)
    }

    /// A bounded breadth-first walk from `start`, entirely inside an
    /// authorised set (P07's `memory.neighbours`).
    ///
    /// Built on [`Self::neighbours`], so an edge to anything outside the set is
    /// never followed, never counted and never named — the walk cannot reveal
    /// that a hidden item exists by the shape of what it returns. `start`
    /// outside the set returns nothing, exactly as a start that does not exist.
    ///
    /// Returns the items reached (the start first, then in walk order) and the
    /// edges walked, each once. `depth` is capped at 2 and the item count at
    /// `max_items`; ties are broken by edge id so two walks of one graph agree.
    pub fn neighbourhood(
        &self,
        start: &str,
        authorised: &[MemoryItem],
        depth: u32,
        max_items: usize,
        kinds: Option<&[EdgeKind]>,
    ) -> Result<(Vec<MemoryItem>, Vec<MemoryEdge>), MemoryError> {
        let by_id: HashMap<&str, &MemoryItem> =
            authorised.iter().map(|item| (item.item_id.as_str(), item)).collect();
        let Some(first) = by_id.get(start) else {
            return Ok((Vec::new(), Vec::new()));
        };
        let depth = depth.clamp(1, 2);
        let mut items: Vec<MemoryItem> = vec![(*first).clone()];
        let mut seen: BTreeSet<String> = BTreeSet::from([start.to_string()]);
        let mut edges: Vec<MemoryEdge> = Vec::new();
        let mut walked: BTreeSet<String> = BTreeSet::new();
        let mut frontier: Vec<String> = vec![start.to_string()];
        for _ in 0..depth {
            let mut next: Vec<String> = Vec::new();
            for node in &frontier {
                let mut found = self.neighbours(node, authorised)?;
                found.sort_by(|a, b| a.edge_id.cmp(&b.edge_id));
                for edge in found {
                    if kinds.is_some_and(|kinds| !kinds.contains(&edge.kind)) {
                        continue;
                    }
                    let other = if edge.from_item == *node { &edge.to_item } else { &edge.from_item };
                    if !seen.contains(other) {
                        if items.len() >= max_items {
                            continue;
                        }
                        let Some(item) = by_id.get(other.as_str()) else {
                            continue;
                        };
                        seen.insert(other.clone());
                        items.push((*item).clone());
                        next.push(other.clone());
                    }
                    if walked.insert(edge.edge_id.clone()) {
                        edges.push(edge);
                    }
                }
            }
            frontier = next;
            if frontier.is_empty() {
                break;
            }
        }
        // Only edges whose both ends were returned: a cut at `max_items` must
        // not leave an edge pointing at an item the caller was not given.
        edges.retain(|edge| seen.contains(&edge.from_item) && seen.contains(&edge.to_item));
        Ok((items, edges))
    }

    // ── Outbox ───────────────────────────────────────────────────────────

    /// Effects that have not been delivered.
    pub fn pending_effects(&self) -> Result<Vec<OutboxRow>, MemoryError> {
        let conn = self.conn.lock().map_err(|_| MemoryError::Storage {
            detail: "the memory graph was left locked".into(),
        })?;
        let mut statement = conn
            .prepare(
                "SELECT outbox_id, target, idempotency_key, payload, created_at, attempts, last_error
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
                    attempts: row.get(5)?,
                    last_error: row.get(6)?,
                })
            })
            .map_err(storage)?
            .filter_map(Result::ok)
            .collect();
        Ok(rows)
    }

    /// Records that a delivery was tried and did not land. The row stays
    /// pending; the next pass -- in this process or the next one -- tries it
    /// again, and the consumer's idempotency makes that safe.
    pub fn record_delivery_failure(&self, outbox_id: i64, error: &str) -> Result<(), MemoryError> {
        let conn = self.conn.lock().map_err(|_| MemoryError::Storage {
            detail: "the memory graph was left locked".into(),
        })?;
        conn.execute(
            "UPDATE agent_memory_outbox SET attempts = attempts + 1, last_error = ?1
             WHERE outbox_id = ?2 AND delivered_at IS NULL",
            params![error, outbox_id],
        )
        .map_err(storage)?;
        Ok(())
    }

    /// Delivers every pending effect it has a consumer for.
    ///
    /// Run at start-up -- which is the "after restart" half of recovery -- and
    /// after a publication. A row is marked delivered only *after* its
    /// consumer accepted it; a crash between the two leaves it pending, and the
    /// next pass delivers it again, which the consumer recognises by key. So
    /// every row is applied exactly once however many times it is delivered.
    pub fn deliver_pending(
        &self,
        consumers: &[&dyn OutboxConsumer],
        at: &str,
    ) -> Result<DeliveryReport, MemoryError> {
        let mut report = DeliveryReport::default();
        for row in self.pending_effects()? {
            let Some(consumer) = consumers.iter().find(|consumer| consumer.target() == row.target)
            else {
                report.no_consumer += 1;
                continue;
            };
            match consumer.deliver(&row) {
                Ok(()) => {
                    self.mark_delivered(row.outbox_id, at)?;
                    report.delivered += 1;
                }
                Err(error) => {
                    log::warn!(
                        "[memory-graph] outbox row {} to {} was not delivered: {error}",
                        row.outbox_id,
                        row.target
                    );
                    self.record_delivery_failure(row.outbox_id, &error)?;
                    report.failed += 1;
                }
            }
        }
        Ok(report)
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

    /// One item, without any authorisation. For tests that inspect the row a
    /// write left behind; production code reads inside its own transaction.
    #[cfg(test)]
    fn raw(&self, item_id: &str) -> Result<Option<MemoryItem>, MemoryError> {
        let conn = self.conn.lock().map_err(|_| MemoryError::Storage {
            detail: "the memory graph was left locked".into(),
        })?;
        Self::read_item(&conn, item_id)
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

    /// The graph as it stood at `cursor`, read from the immutable versions.
    ///
    /// ## Why this exists
    ///
    /// Plan P02: "never label latest rows with an old cursor." A context
    /// compiled at graph revision 12 and replayed after the graph moved to 20
    /// must see what revision 12 held, not what is there now under 12's label.
    /// Each item is read at its latest version written at or before `cursor`;
    /// an item created after it is absent; an edge is present if it was linked
    /// at or before `cursor` and both its ends are.
    ///
    /// ## Current authorisation still decides
    ///
    /// A historical read is not a way round a revocation. An item is returned
    /// only if the reader may see it *now* and could see it *then*: an item
    /// since tombstoned, or withdrawn from this reader, is not handed back in
    /// its old form.
    ///
    /// A cursor ahead of the graph is refused. Labelling the present with a
    /// future position would claim to have seen writes that have not happened.
    pub fn snapshot_as_of(
        &self,
        session: &Session,
        scope: &MemoryScope,
        project_id: Option<&str>,
        cursor: i64,
    ) -> Result<MemorySnapshot, MemoryError> {
        let mut conn = self.conn.lock().map_err(|_| MemoryError::Storage {
            detail: "the memory graph was left locked".into(),
        })?;
        let transaction = conn.transaction().map_err(storage)?;
        let head = Self::latest_revision(&transaction).unwrap_or(0);
        if cursor > head {
            return Err(MemoryError::NotPermitted {
                because: format!(
                    "the graph is at revision {head}, so it cannot be read as of {cursor}: that \
                     would label the present with a position nothing has reached"
                ),
            });
        }
        let scope_key = scope.key();

        let historical: Vec<MemoryItem> = {
            let mut statement = transaction
                .prepare(
                    "SELECT v.body FROM agent_memory_versions v
                       JOIN (SELECT item_id, MAX(revision) AS revision
                               FROM agent_memory_versions
                              WHERE scope_key = ?1 AND graph_revision <= ?2
                              GROUP BY item_id) latest
                         ON latest.item_id = v.item_id AND latest.revision = v.revision",
                )
                .map_err(storage)?;
            let bodies: Vec<String> = statement
                .query_map(params![scope_key, cursor], |row| row.get::<_, String>(0))
                .map_err(storage)?
                .filter_map(Result::ok)
                .collect();
            bodies
                .into_iter()
                .filter_map(|text| serde_json::from_str::<MemoryItem>(&text).ok())
                .collect()
        };

        let mut items = Vec::new();
        for item in historical {
            let now_readable = Self::read_item(&transaction, &item.item_id)?
                .is_some_and(|current| current.readable_by(session, project_id));
            if now_readable && item.readable_by(session, project_id) {
                items.push(item);
            }
        }

        let visible: BTreeSet<&str> = items.iter().map(|item| item.item_id.as_str()).collect();
        let edges: Vec<MemoryEdge> = {
            let mut statement = transaction
                .prepare(
                    "SELECT e.edge_id, e.from_item, e.to_item, e.kind, e.agent_id, e.scope_key,
                            e.created_at
                       FROM agent_memory_edges e
                      WHERE e.scope_key = ?1
                        AND EXISTS (SELECT 1 FROM agent_memory_log l
                                     WHERE l.item_id = e.edge_id AND l.subject_kind = 'edge'
                                       AND l.revision <= ?2)",
                )
                .map_err(storage)?;
            let rows: Vec<(String, String, String, String, String, String, String)> = statement
                .query_map(params![scope_key, cursor], |row| {
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
        transaction.commit().map_err(storage)?;

        Ok(MemorySnapshot {
            items,
            edges,
            cursor,
            in_context: Vec::new(),
            context_revision: None,
        })
    }

    /// Every revision of one item this reader may see, oldest first.
    ///
    /// Empty when the reader may not see the item as it stands now -- the same
    /// rule as [`Self::snapshot_as_of`], and the same answer for "no such item",
    /// so the history cannot be used to discover what exists.
    pub fn versions_of(
        &self,
        session: &Session,
        item_id: &str,
        project_id: Option<&str>,
    ) -> Result<Vec<ItemVersion>, MemoryError> {
        let conn = self.conn.lock().map_err(|_| MemoryError::Storage {
            detail: "the memory graph was left locked".into(),
        })?;
        let current_visible = Self::read_item(&conn, item_id)?
            .is_some_and(|current| current.readable_by(session, project_id));
        if !current_visible {
            return Ok(Vec::new());
        }
        let mut statement = conn
            .prepare(
                "SELECT revision, graph_revision, body FROM agent_memory_versions
                  WHERE item_id = ?1 ORDER BY revision",
            )
            .map_err(storage)?;
        let rows: Vec<(i64, i64, String)> = statement
            .query_map(params![item_id], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))
            .map_err(storage)?
            .filter_map(Result::ok)
            .collect();
        Ok(rows
            .into_iter()
            .filter_map(|(revision, graph_revision, body)| {
                let item: MemoryItem = serde_json::from_str(&body).ok()?;
                Some(ItemVersion {
                    item_id: item_id.to_string(),
                    revision: revision as u64,
                    graph_revision,
                    item,
                })
            })
            .collect())
    }

    // ── Migration support ────────────────────────────────────────────────

    /// Writes one record brought across from a legacy store.
    ///
    /// Addressed by the item id the migration derives from the legacy record,
    /// so it is restartable by construction: a re-run finds what the last run
    /// wrote and changes nothing when the content is the same. A legacy record
    /// that changed since is a new revision -- the graph follows its authority
    /// -- and one that was rolled back is restored. Nothing is ever deleted.
    pub fn upsert_migrated(&self, mut item: MemoryItem) -> Result<MigratedWrite, MemoryError> {
        if !matches!(item.provenance, Provenance::Migrated { .. }) {
            return Err(MemoryError::NotPermitted {
                because: "only a migrated record may be written through the migration path"
                    .to_string(),
            });
        }
        let outcome = admit(&item, &ReceiptVerdict::NotAReceipt);
        item.status = outcome.status;
        item.basis = Some(Basis::of(&item.provenance));

        let mut conn = self.conn.lock().map_err(|_| MemoryError::Storage {
            detail: "the memory graph was left locked by a failed write".into(),
        })?;
        let transaction = conn.transaction().map_err(storage)?;
        let written = match Self::read_item(&transaction, &item.item_id)? {
            None => {
                item.revision = 1;
                Self::write_revision(&transaction, &item, "migrated")?;
                MigratedWrite::Inserted
            }
            Some(held) if held.status == ItemStatus::Tombstoned => {
                item.revision = held.revision + 1;
                Self::write_revision(&transaction, &item, "migrationRestored")?;
                MigratedWrite::Restored
            }
            Some(held) if held.content == item.content && held.sources == item.sources
                && held.artifacts == item.artifacts =>
            {
                MigratedWrite::Unchanged
            }
            Some(held) => {
                item.revision = held.revision + 1;
                Self::write_revision(&transaction, &item, "migrationResynced")?;
                Self::propagate_staleness(
                    &transaction,
                    &[item.item_id.clone()],
                    "its legacy source changed",
                    &item.updated_at,
                )?;
                MigratedWrite::Updated
            }
        };
        let revision = Self::latest_revision(&transaction).unwrap_or(0);
        transaction.commit().map_err(storage)?;
        drop(conn);
        if written != MigratedWrite::Unchanged {
            self.announce(revision);
        }
        Ok(written)
    }

    /// Rolls back one legacy source: every record migrated from it is
    /// tombstoned, as a new revision, and nothing is deleted.
    ///
    /// The legacy store was never touched by the migration and is still where
    /// those records are written, so rolling back returns the deployment to
    /// the state before the migration ran. Re-running the migration restores
    /// them ([`MigratedWrite::Restored`]).
    pub fn retire_migrated(&self, legacy_store: &str, at: &str) -> Result<Vec<String>, MemoryError> {
        self.retire_migrated_where(legacy_store, |_| true, at)
    }

    /// Retires the records migrated from `legacy_store` whose legacy id `pick`
    /// selects -- the orphans a pass finds when a legacy record has gone.
    pub fn retire_migrated_where(
        &self,
        legacy_store: &str,
        pick: impl Fn(&str) -> bool,
        at: &str,
    ) -> Result<Vec<String>, MemoryError> {
        let targets: Vec<String> = self
            .all_raw()?
            .into_iter()
            .filter(|item| {
                matches!(&item.provenance, Provenance::Migrated { legacy_store: store, legacy_id }
                    if store == legacy_store && pick(legacy_id))
                    && item.status != ItemStatus::Tombstoned
            })
            .map(|item| item.item_id)
            .collect();
        if targets.is_empty() {
            return Ok(targets);
        }
        let mut conn = self.conn.lock().map_err(|_| MemoryError::Storage {
            detail: "the memory graph was left locked by a failed write".into(),
        })?;
        let transaction = conn.transaction().map_err(storage)?;
        for id in &targets {
            if let Some(mut item) = Self::read_item(&transaction, id)? {
                item.status = ItemStatus::Tombstoned;
                item.revision += 1;
                item.updated_at = at.to_string();
                Self::write_revision(&transaction, &item, "migrationRolledBack")?;
            }
        }
        Self::propagate_staleness(&transaction, &targets, "its migrated source was rolled back", at)?;
        let revision = Self::latest_revision(&transaction).unwrap_or(0);
        transaction.commit().map_err(storage)?;
        drop(conn);
        self.announce(revision);
        Ok(targets)
    }

    /// Opens a migration pass in the run ledger.
    pub fn begin_migration_run(&self, source: &str, action: &str, at: &str) -> Result<String, MemoryError> {
        let run_id = format!("mig-{}", uuid::Uuid::new_v4());
        let conn = self.conn.lock().map_err(|_| MemoryError::Storage {
            detail: "the memory graph was left locked".into(),
        })?;
        conn.execute(
            "INSERT INTO agent_memory_migration_runs (run_id, source, action, started_at)
             VALUES (?1, ?2, ?3, ?4)",
            params![run_id, source, action, at],
        )
        .map_err(storage)?;
        Ok(run_id)
    }

    /// Closes a migration pass with what it did.
    pub fn finish_migration_run(&self, run_id: &str, report: &str, at: &str) -> Result<(), MemoryError> {
        let conn = self.conn.lock().map_err(|_| MemoryError::Storage {
            detail: "the memory graph was left locked".into(),
        })?;
        conn.execute(
            "UPDATE agent_memory_migration_runs SET finished_at = ?1, report = ?2 WHERE run_id = ?3",
            params![at, report, run_id],
        )
        .map_err(storage)?;
        Ok(())
    }

    /// Migration passes that started and never finished.
    pub fn unfinished_migration_runs(&self) -> Result<Vec<(String, String, String)>, MemoryError> {
        let conn = self.conn.lock().map_err(|_| MemoryError::Storage {
            detail: "the memory graph was left locked".into(),
        })?;
        let mut statement = conn
            .prepare(
                "SELECT run_id, source, action FROM agent_memory_migration_runs
                  WHERE finished_at IS NULL ORDER BY started_at",
            )
            .map_err(storage)?;
        let rows = statement
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))
            .map_err(storage)?
            .filter_map(Result::ok)
            .collect();
        Ok(rows)
    }

    /// Items migrated from one legacy store, whatever their status, keyed by
    /// the legacy id they came from. What a verification compares against.
    pub fn migrated_from(&self, legacy_store: &str) -> Result<HashMap<String, MemoryItem>, MemoryError> {
        Ok(self
            .all_raw()?
            .into_iter()
            .filter_map(|item| match &item.provenance {
                Provenance::Migrated {
                    legacy_store: store,
                    legacy_id,
                } if store == legacy_store => Some((legacy_id.clone(), item)),
                _ => None,
            })
            .collect())
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
    MemoryScope::from_key(key)
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
