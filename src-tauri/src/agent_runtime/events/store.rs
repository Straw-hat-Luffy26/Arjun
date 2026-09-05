//! The durable store: ordered appends, snapshots, and recovery after a restart.
//!
//! ## Why SQLite and not more JSON files
//!
//! The task record is a file per run, written once when the run ends. That is
//! the right shape for a document nobody appends to. It is the wrong shape for
//! a stream: appending to a JSON file means rewriting it, rewriting it means a
//! window in which it is neither the old one nor the new one, and doing that on
//! every tool call means that window is open most of the time.
//!
//! What is actually needed is: many small ordered writes, a reader that can ask
//! for "everything after seq 12", and an ending that is either wholly there or
//! wholly not. That is a transaction, and the database is already open beside
//! the audit log.
//!
//! ## The two guarantees, and how each is kept
//!
//! - **Atomic.** Every append is one `BEGIN IMMEDIATE` transaction that reads
//!   the run's tail and writes the next row. A crash between the two leaves
//!   nothing, never a row with a sequence number that was already taken.
//! - **Ordered.** `seq` is assigned inside that transaction and constrained
//!   `UNIQUE (run_id, seq)`. Two writers racing cannot both take the same
//!   number: one of them loses the constraint and retries. The order is a
//!   property of the storage engine, not of the callers behaving.
//!
//! Rows are append-only by trigger, for the same reason the audit log is: a
//! history that can be quietly rewritten is not a history.

use std::collections::BTreeSet;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use rusqlite::{params, Connection, OptionalExtension};
use serde_json::{json, Value};

use super::checkpoint::{NotResumable, RunCheckpoint};
use super::approvals::{self, ApprovalStatus, DurableApproval};
use super::lease::{self, Held, Lease};
use super::idempotency::{self, EffectLookup, EffectStatus, RecordedOutcome};
use super::machine::RunState;
use super::model::{
    payload_hash, EventDraft, TaskEvent, TaskEventType, UnreadableEvent, SCHEMA_VERSION,
    SYSTEM_ACTOR,
};
use super::projection::{fold, TaskSnapshot};
use super::MAX_RECOVERY_ATTEMPTS;

#[cfg(test)]
#[path = "durability_tests.rs"]
mod durability_tests;

/// Why an append did not happen.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AppendError {
    /// This event id is already in the log. Not an error the caller has to
    /// handle as a failure — the event it was trying to write is there, which
    /// is the outcome it wanted.
    Duplicate { event_id: String, seq: i64 },
    /// The run already has a terminal event. Nothing may follow one: a run that
    /// has both `run_cancelled` and, later, `run_completed` describes two
    /// different endings and lets a reader pick.
    AlreadyEnded {
        run_id: String,
        ending: TaskEventType,
    },
    Storage(String),
}

impl std::fmt::Display for AppendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AppendError::Duplicate { event_id, seq } => write!(
                f,
                "event {event_id} is already recorded at sequence {seq}; it was not written again"
            ),
            AppendError::AlreadyEnded { run_id, ending } => write!(
                f,
                "run {run_id} already ended with {}; nothing may be recorded after it",
                ending.as_str()
            ),
            AppendError::Storage(detail) => write!(f, "the task event could not be written: {detail}"),
        }
    }
}

impl From<rusqlite::Error> for AppendError {
    fn from(error: rusqlite::Error) -> Self {
        AppendError::Storage(error.to_string())
    }
}

/// A run's events, plus the ones that could not be read.
#[derive(Debug, Clone, Default)]
pub struct EventPage {
    pub events: Vec<TaskEvent>,
    pub unreadable: Vec<UnreadableEvent>,
}

impl EventPage {
    /// The highest sequence number seen, readable or not.
    pub fn last_seq(&self) -> i64 {
        let read = self.events.last().map(|event| event.seq).unwrap_or(0);
        let broken = self
            .unreadable
            .iter()
            .map(|event| event.seq)
            .max()
            .unwrap_or(0);
        read.max(broken)
    }
}

/// The durable record of what runs are doing and have done.
pub struct TaskEventLog {
    pub(super) conn: Arc<Mutex<Connection>>,
}

/// How long a writer waits for another connection to let go of the database.
///
/// The file is shared with the audit log, the knowledge index and the
/// credential store. Without this, a run's event append fails outright the
/// first time a document is indexed at the same moment.
const BUSY_TIMEOUT: Duration = Duration::from_secs(5);

impl TaskEventLog {
    /// Opens the log beside the rest of the application data.
    pub fn open(app_data_dir: &Path) -> Result<Self, String> {
        std::fs::create_dir_all(app_data_dir)
            .map_err(|error| format!("could not create {}: {error}", app_data_dir.display()))?;
        let conn = Connection::open(app_data_dir.join("sarathi.db"))
            .map_err(|error| format!("could not open the task event log: {error}"))?;
        Self::from_connection(conn)
    }

    /// An in-memory log, for tests and for a deployment where the database
    /// could not be opened at all — a run with an unrecorded history is better
    /// than no run, and the caller logs the degradation.
    pub fn in_memory() -> Result<Self, String> {
        let conn = Connection::open_in_memory()
            .map_err(|error| format!("could not open an in-memory task event log: {error}"))?;
        Self::from_connection(conn)
    }

    fn from_connection(conn: Connection) -> Result<Self, String> {
        conn.busy_timeout(BUSY_TIMEOUT)
            .map_err(|error| format!("could not set the busy timeout: {error}"))?;
        conn.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS task_events (
                event_id       TEXT PRIMARY KEY,
                run_id         TEXT NOT NULL,
                seq            INTEGER NOT NULL,
                event_type     TEXT NOT NULL,
                at             TEXT NOT NULL,
                actor          TEXT NOT NULL,
                schema_version INTEGER NOT NULL,
                payload        TEXT NOT NULL,
                payload_hash   TEXT NOT NULL,
                UNIQUE (run_id, seq)
            );

            CREATE INDEX IF NOT EXISTS task_events_run_idx ON task_events(run_id, seq);
            CREATE INDEX IF NOT EXISTS task_events_type_idx ON task_events(run_id, event_type);

            -- Append-only, enforced by the storage engine rather than by
            -- convention. A history that can be quietly rewritten proves
            -- nothing about what a run did.
            CREATE TRIGGER IF NOT EXISTS task_events_is_append_only_update
            BEFORE UPDATE ON task_events
            BEGIN
                SELECT RAISE(ABORT, 'task_events is append-only: rows cannot be modified');
            END;

            CREATE TRIGGER IF NOT EXISTS task_events_is_append_only_delete
            BEFORE DELETE ON task_events
            BEGIN
                SELECT RAISE(ABORT, 'task_events is append-only: rows cannot be deleted');
            END;

            -- The snapshot is a fold of the events with the sequence number it
            -- was folded up to. It is updated in place on purpose: it is a
            -- cache, and the events above are what it can always be rebuilt
            -- from.
            CREATE TABLE IF NOT EXISTS task_snapshots (
                run_id         TEXT PRIMARY KEY,
                seq            INTEGER NOT NULL,
                status         TEXT NOT NULL,
                schema_version INTEGER NOT NULL,
                updated_at     TEXT NOT NULL,
                state          TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS task_snapshots_status_idx ON task_snapshots(status);

            -- The point a run can be continued from. One row per run: the
            -- newest safe point is the only one worth keeping, and choosing
            -- between several at resume time has no defensible answer other
            -- than the newest one. Updated in place, guarded by sequence in
            -- `save_checkpoint` so a late write from a dying process cannot
            -- move the point backwards.
            CREATE TABLE IF NOT EXISTS run_checkpoints (
                run_id          TEXT PRIMARY KEY,
                attempt_id      TEXT NOT NULL,
                last_event_seq  INTEGER NOT NULL,
                state           TEXT NOT NULL,
                at              TEXT NOT NULL,
                schema_version  INTEGER NOT NULL,
                checkpoint_hash TEXT NOT NULL,
                body            TEXT NOT NULL
            );
            ",
        )
        .map_err(|error| format!("could not prepare the task event schema: {error}"))?;

        idempotency::prepare(&conn)
            .map_err(|error| format!("could not prepare the tool effect schema: {error}"))?;

        lease::prepare(&conn)
            .map_err(|error| format!("could not prepare the run lease schema: {error}"))?;

        // Everything the baseline above does not cover, applied in order and
        // recorded in the file. Runs after the baseline on purpose: a database
        // in the field already has the baseline tables and no version number,
        // so the migrations must be able to assume the baseline is present.
        let version = super::migrations::apply(&conn)
            .map_err(|error| format!("could not migrate the task event schema: {error}"))?;
        log::debug!("[tasks] task event schema is at version {version}");

        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    fn lock(&self) -> Result<std::sync::MutexGuard<'_, Connection>, AppendError> {
        self.conn
            .lock()
            .map_err(|_| AppendError::Storage("the task event log is poisoned".to_string()))
    }

    /// Appends one event and returns it as stored.
    ///
    /// The sequence number is assigned here, inside the transaction, and is the
    /// only thing about the event the caller does not choose.
    pub fn append(&self, draft: EventDraft) -> Result<TaskEvent, AppendError> {
        self.append_inner(draft, false, None)
    }

    /// Appends an event that is allowed to follow an ending.
    ///
    /// There is exactly one of these: a person saying what actually happened to
    /// a side effect that was in flight when the process died. It is deliberately
    /// not a general escape hatch — the rule that nothing follows an ending is
    /// what stops a run having two accounts of how it finished, and this does
    /// not compete with the ending. It describes something the ending left
    /// unresolved.
    ///
    /// The ending itself is never rewritten; the reconciliation is a later
    /// event that a reader sees *after* it.
    pub fn append_past_ending(&self, draft: EventDraft) -> Result<TaskEvent, AppendError> {
        self.append_inner(draft, true, None)
    }

    fn append_inner(
        &self,
        draft: EventDraft,
        past_ending: bool,
        expected_lease: Option<&Lease>,
    ) -> Result<TaskEvent, AppendError> {
        let payload = serde_json::to_string(&draft.payload)
            .map_err(|error| AppendError::Storage(error.to_string()))?;
        let hash = payload_hash(&draft.payload);
        let at = draft.at.to_rfc3339();

        let conn = self.lock()?;
        conn.execute_batch("BEGIN IMMEDIATE")?;

        let result = (|| -> Result<TaskEvent, AppendError> {
            if let Some(expected) = expected_lease {
                let current = lease::holder(&conn, &draft.run_id, chrono::Utc::now())?;
                if expected.run_id != draft.run_id || !current.is_some_and(|held| {
                    held.owner == expected.owner && held.fence_token == expected.fence_token
                        && held.live_at(chrono::Utc::now())
                }) {
                    return Err(AppendError::Storage("the execution lease is no longer current".into()));
                }
            }
            // Presented again after an ambiguous failure. The event is there,
            // so say where rather than writing a second copy of it.
            if let Some(seq) = existing_seq(&conn, &draft.event_id)? {
                return Err(AppendError::Duplicate {
                    event_id: draft.event_id.clone(),
                    seq,
                });
            }

            if !past_ending {
                if let Some(ending) = ending_of(&conn, &draft.run_id)? {
                    return Err(AppendError::AlreadyEnded {
                        run_id: draft.run_id.clone(),
                        ending,
                    });
                }
            }

            let seq: i64 = conn.query_row(
                "SELECT COALESCE(MAX(seq), 0) + 1 FROM task_events WHERE run_id = ?1",
                params![draft.run_id],
                |row| row.get(0),
            )?;

            conn.execute(
                "INSERT INTO task_events
                    (event_id, run_id, seq, event_type, at, actor, schema_version, payload, payload_hash)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
                params![
                    draft.event_id,
                    draft.run_id,
                    seq,
                    draft.event_type.as_str(),
                    at,
                    draft.actor,
                    SCHEMA_VERSION,
                    payload,
                    hash,
                ],
            )?;

            Ok(TaskEvent {
                run_id: draft.run_id.clone(),
                event_id: draft.event_id.clone(),
                seq,
                event_type: draft.event_type,
                at: at.clone(),
                actor: draft.actor.clone(),
                schema_version: SCHEMA_VERSION,
                payload: draft.payload.clone(),
                payload_hash: hash.clone(),
            })
        })();

        match &result {
            Ok(_) => conn.execute_batch("COMMIT")?,
            // Rolled back on every failure, including the duplicate: the point
            // of the duplicate check is that nothing is written.
            Err(_) => {
                let _ = conn.execute_batch("ROLLBACK");
            }
        }
        result
    }

    /// Appends and folds the result into the run's snapshot in one step.
    ///
    /// The ordinary way to write. Keeping the two together means there is no
    /// call site that records an event and forgets the state it implies, which
    /// is the way a cache and its source drift apart.
    pub fn record(&self, draft: EventDraft) -> Result<TaskEvent, AppendError> {
        let run_id = draft.run_id.clone();
        let event = self.append(draft)?;
        if let Err(error) = self.advance_snapshot(&run_id, &event) {
            // The snapshot is rebuildable from the events, which are written.
            // Losing it costs a slower read, not a lost run.
            log::warn!("[tasks] the snapshot for run {run_id} could not be updated: {error}");
        }
        Ok(event)
    }

    /// Check the live writer and append under the same SQLite write transaction.
    pub fn record_fenced(&self, draft: EventDraft, lease: &Lease) -> Result<TaskEvent, AppendError> {
        let run_id = draft.run_id.clone();
        let event = self.append_inner(draft, false, Some(lease))?;
        if let Err(error) = self.advance_snapshot(&run_id, &event) {
            log::warn!("[tasks] the snapshot for run {run_id} could not be updated: {error}");
        }
        Ok(event)
    }

    /// Every event of a run after `after_seq`, in order.
    ///
    /// A row that cannot be read is reported rather than dropped or fatal. One
    /// corrupt event should cost its own line in the trace, not the run's whole
    /// recoverable history.
    pub fn events_since(&self, run_id: &str, after_seq: i64) -> Result<EventPage, String> {
        let conn = self
            .conn
            .lock()
            .map_err(|_| "the task event log is poisoned".to_string())?;
        read_since(&conn, run_id, after_seq)
    }

    /// The latest state of a run, without replaying its whole history.
    ///
    /// Reads the stored snapshot and folds only what has happened since. Falls
    /// back to a full rebuild when there is no snapshot, or when the stored one
    /// will not parse — a broken cache is a slow read, never a wrong answer.
    pub fn snapshot(&self, run_id: &str) -> Result<Option<TaskSnapshot>, String> {
        let conn = self
            .conn
            .lock()
            .map_err(|_| "the task event log is poisoned".to_string())?;
        snapshot_within(&conn, run_id)
    }

    /// Folds a run's whole history, ignoring any stored snapshot.
    pub fn rebuild(&self, run_id: &str) -> Result<Option<TaskSnapshot>, String> {
        let conn = self
            .conn
            .lock()
            .map_err(|_| "the task event log is poisoned".to_string())?;
        rebuild_within(&conn, run_id)
    }

    /// Writes a snapshot, refusing to move it backwards.
    ///
    /// Two windows watching one run both fold and both save. Without the
    /// guard the slower one would overwrite the faster one's work with an
    /// older view, and the screen would appear to go backwards.
    pub fn save_snapshot(&self, snapshot: &TaskSnapshot) -> Result<(), String> {
        let conn = self
            .conn
            .lock()
            .map_err(|_| "the task event log is poisoned".to_string())?;
        save_within(&conn, snapshot)
    }

    fn advance_snapshot(&self, run_id: &str, event: &TaskEvent) -> Result<(), String> {
        let conn = self
            .conn
            .lock()
            .map_err(|_| "the task event log is poisoned".to_string())?;
        let mut snapshot = snapshot_within(&conn, run_id)?.unwrap_or_else(|| {
            let mut fresh = TaskSnapshot::empty(run_id);
            fresh.started_at = event.at.clone();
            fresh
        });
        snapshot.apply(event);
        save_within(&conn, &snapshot)
    }

    /// Every run this machine knows about, newest first.
    pub fn snapshots(&self) -> Result<Vec<TaskSnapshot>, String> {
        let conn = self
            .conn
            .lock()
            .map_err(|_| "the task event log is poisoned".to_string())?;
        let mut statement = conn
            .prepare("SELECT run_id, state FROM task_snapshots ORDER BY updated_at DESC")
            .map_err(|error| error.to_string())?;
        let rows = statement
            .query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })
            .map_err(|error| error.to_string())?;

        let mut snapshots = Vec::new();
        for row in rows.flatten() {
            let (run_id, state) = row;
            match serde_json::from_str::<TaskSnapshot>(&state) {
                Ok(snapshot) => snapshots.push(snapshot),
                // A snapshot row that will not parse is a cache line, not a
                // record. Rebuild it from the events rather than dropping the
                // run off the listing.
                Err(_) => {
                    if let Ok(Some(rebuilt)) = rebuild_within(&conn, &run_id) {
                        snapshots.push(rebuilt);
                    }
                }
            }
        }
        Ok(snapshots)
    }

    /// The runs that are still going as far as the record is concerned.
    pub fn running(&self) -> Result<Vec<TaskSnapshot>, String> {
        Ok(self
            .snapshots()?
            .into_iter()
            .filter(|snapshot| !snapshot.state.is_terminal())
            .collect())
    }

    /// Closes off runs that were going when the process went away.
    ///
    /// Called once at start, before anything else can write, and in an order
    /// that matters:
    ///
    /// 1. **Every side effect still marked `pending` becomes `unknown`.** The
    ///    process that was performing it is gone, so a `pending` row at this
    ///    moment is exactly a side effect nobody can account for. Each one gets
    ///    a `tool_effect_unknown` event against its run.
    /// 2. **Every run with no ending is ended** as `run_degraded`.
    ///
    /// The order is what makes the second step honest: by the time a run is
    /// closed off, its unknown effects are already in its history, so the
    /// snapshot says *why* a person is needed rather than only that they are.
    ///
    /// A run with events and no ending cannot be resumed — the loop carrying it
    /// is gone. Without this the Tasks screen shows a run that has been
    /// "running" since last Tuesday next to one that is running now.
    ///
    /// Returns the runs it closed.
    pub fn recover_interrupted(&self, actor: &str) -> Result<Vec<String>, String> {
        for effect in self.strand_pending_effects()? {
            let draft = EventDraft::idempotent(
                &effect.run_id,
                TaskEventType::ToolEffectUnknown,
                actor,
                &effect.idempotency_key,
            )
            .with(json!({
                "idempotencyKey": effect.idempotency_key,
                "tool": effect.tool,
                "target": effect.target,
                "toolCallId": effect.idempotency_key,
                "note": "in flight when the process went away; nobody can say whether it took effect",
            }));
            // Written before the run is ended, so it lands while the run is
            // still open. After the ending it would be refused.
            if let Err(error) = self.record(draft) {
                log::warn!(
                    "[tasks] run {}: an unknown side effect could not be recorded: {error}",
                    effect.run_id
                );
            }
        }

        let unfinished = self.unfinished_runs()?;
        let mut recovered = Vec::new();

        for run_id in unfinished {
            let claim = match self.claim_run(&run_id, &format!("recovery:{}",uuid::Uuid::new_v4()), chrono::Duration::seconds(60), chrono::Utc::now())? {
                Ok(claim) => claim,
                Err(_) => continue,
            };
            let _guard = RecoveryClaim { log: self, claim };
            // Whether this run could be picked up again, from the durable facts
            // this side owns. Deliberately not the whole answer: the authority
            // on resumability is `checkpoint::resumable_against`, which
            // re-derives the policy, plan and workspace hashes against the
            // world as it is now and is re-run by `agent_resume_run` before any
            // work happens. This is the cheaper question of whether it is worth
            // offering at all.
            //
            // Before this, every interrupted run was marked degraded — and
            // `DegradedNeedsHuman` is terminal, so `resumable_against` refused
            // it as `AlreadyEnded`. Startup was therefore closing off precisely
            // the runs resumption exists for.
            let snapshot = self.snapshot(&run_id)?;
            let attempts = snapshot
                .as_ref()
                .map(|snapshot| snapshot.recovery_attempts)
                .unwrap_or(0);
            let unsettled = !self.effect_obligations(&run_id)?.0.is_empty();
            let has_checkpoint = matches!(self.checkpoint(&run_id), Ok(Some(_)));

            // An effect nobody settled stops this outright. Continuing would
            // either repeat it or assume it worked, and nothing here can tell
            // which — the same rule `agent_resume_run` applies.
            let worth_offering = has_checkpoint && !unsettled && attempts < MAX_RECOVERY_ATTEMPTS;

            if worth_offering {
                let draft = EventDraft::idempotent(
                    &run_id,
                    TaskEventType::RecoveryStarted,
                    actor,
                    &format!("recovery-{}",attempts + 1),
                )
                .with(json!({
                    "attempt": attempts + 1,
                    "maxAttempts": MAX_RECOVERY_ATTEMPTS,
                    "recoveredAt": chrono::Utc::now().to_rfc3339(),
                }));
                match self.record(draft) {
                    Ok(_) => {
                        recovered.push(run_id);
                        continue;
                    }
                    Err(AppendError::AlreadyEnded { .. }) | Err(AppendError::Duplicate { .. }) => {
                        continue
                    }
                    Err(error) => {
                        log::warn!(
                            "[tasks] run {run_id} could not be offered for recovery: {error}"
                        );
                        // Falls through to being closed off, which is the safe
                        // direction: a run nothing recorded as recoverable must
                        // not be left looking live.
                    }
                }
            }

            // A run that has used up its attempts says so before it is closed
            // off, so a recovery loop is visible as one rather than as a run
            // that merely kept being interrupted.
            if attempts >= MAX_RECOVERY_ATTEMPTS {
                let _ = self.record(
                    EventDraft::idempotent(
                        &run_id,
                        TaskEventType::RecoveryFailed,
                        actor,
                        "recovery-exhausted",
                    )
                    .with(json!({
                        "attempts": attempts,
                        "maxAttempts": MAX_RECOVERY_ATTEMPTS,
                    })),
                );
            }

            // Everything else is closed off exactly as before. The sentence
            // says which of the reasons applies, because "somebody needs to
            // look" is not actionable without it.
            let why = if unsettled {
                "A side effect was in flight when the application closed, and nobody can say whether it took effect. That has to be settled before this run is continued or relied on."
            } else if attempts >= MAX_RECOVERY_ATTEMPTS {
                "This run has already been picked up as many times as it is allowed to be, and was interrupted again each time. Somebody needs to look at it."
            } else if !has_checkpoint {
                "Interrupted: the application closed while this was still running, and it never reached a point it could be continued from. Somebody needs to look at this before it is relied on."
            } else {
                "Interrupted: the application closed while this was still running. Somebody needs to look at this before it is relied on."
            };

            let draft =
                EventDraft::idempotent(&run_id, TaskEventType::RunDegraded, actor, "recovery")
                    .with(json!({
                        "failure": why,
                        "recoveredAt": chrono::Utc::now().to_rfc3339(),
                    }));
            match self.record(draft) {
                Ok(_) => recovered.push(run_id),
                // Already ended between the query and the write — the ordinary
                // race with a run finishing as the application starts.
                Err(AppendError::AlreadyEnded { .. }) | Err(AppendError::Duplicate { .. }) => {}
                Err(error) => {
                    log::warn!("[tasks] run {run_id} could not be closed off: {error}");
                }
            }
        }
        Ok(recovered)
    }

    /// Promotes every still-`pending` side effect to `unknown`.
    fn strand_pending_effects(&self) -> Result<Vec<RecordedOutcome>, String> {
        let conn = self
            .conn
            .lock()
            .map_err(|_| "the task event log is poisoned".to_string())?;
        idempotency::promote_pending_to_unknown(&conn).map_err(|error| error.to_string())
    }

    /// Runs that have events but no ending.
    fn unfinished_runs(&self) -> Result<Vec<String>, String> {
        let conn = self
            .conn
            .lock()
            .map_err(|_| "the task event log is poisoned".to_string())?;
        let terminals: Vec<&str> = TERMINAL_TYPES.iter().map(|kind| kind.as_str()).collect();
        let placeholders = vec!["?"; terminals.len()].join(", ");
        let sql = format!(
            "SELECT DISTINCT run_id FROM task_events AS outer_events
             WHERE NOT EXISTS (
                 SELECT 1 FROM task_events AS ending
                 WHERE ending.run_id = outer_events.run_id
                   AND ending.event_type IN ({placeholders})
             )
             ORDER BY run_id"
        );
        let mut statement = conn.prepare(&sql).map_err(|error| error.to_string())?;
        let rows = statement
            .query_map(rusqlite::params_from_iter(terminals), |row| {
                row.get::<_, String>(0)
            })
            .map_err(|error| error.to_string())?;
        Ok(rows.flatten().collect())
    }

    // -- Idempotency ------------------------------------------------------

    /// Records the intent to perform a side effect, before it is performed.
    ///
    /// The reply says whether the caller may go ahead. Only
    /// [`EffectLookup::Fresh`] means yes; everything else is either a result to
    /// replay or a reason to refuse. See [`super::idempotency`] for why the
    /// intent is written first.
    pub fn begin_effect(
        &self,
        run_id: &str,
        key: &str,
        tool: &str,
        args_fingerprint: &str,
        target: &str,
    ) -> EffectLookup {
        let Ok(conn) = self.conn.lock() else {
            return EffectLookup::Unavailable {
                reason: "The task effect ledger could not be locked. No action was started.".into(),
            };
        };
        match idempotency::begin(&conn, run_id, key, tool, args_fingerprint, target) {
            Ok(lookup) => lookup,
            Err(error) => {
                log::warn!("[tasks] run {run_id}: the intent to {tool} was not recorded: {error}");
                EffectLookup::Unavailable {
                    reason: "The tool intent could not be recorded durably. No action was started.".into(),
                }
            }
        }
    }

    /// Settles an effect that was begun.
    pub fn settle_effect(
        &self,
        run_id: &str,
        key: &str,
        outcome: &Result<String, String>,
    ) -> Result<(), String> {
        let conn = self.conn.lock().map_err(|_| "The task effect ledger could not be locked.")?;
        idempotency::settle(&conn, run_id, key, outcome).map_err(|error| {
            log::error!("[tasks] run {run_id}: a side effect could not be settled: {error}");
            "The action may have happened, but its result could not be recorded. Reconciliation is required before continuing.".into()
        })
    }

    /// Records what a person found out about an unknown side effect.
    ///
    /// Returns false when there was no unknown effect under that key — which is
    /// an ordinary race (somebody else reconciled it first), not a failure.
    pub fn reconcile_effect(
        &self,
        run_id: &str,
        key: &str,
        happened: bool,
        by: &str,
    ) -> Result<bool, String> {
        let changed = {
            let conn = self
                .conn
                .lock()
                .map_err(|_| "the task event log is poisoned".to_string())?;
            idempotency::reconcile(&conn, run_id, key, happened, by)
                .map_err(|error| error.to_string())?
        };
        if changed == 0 {
            return Ok(false);
        }

        // Recorded against the run even though the run has ended. This is the
        // one thing that may follow a terminal event, because it is a statement
        // *about* the ending rather than a competing account of it — so it goes
        // through `append_past_ending` rather than `record`, which refuses.
        let draft =
            EventDraft::idempotent(run_id, TaskEventType::ToolEffectReconciled, by, key).with(
                json!({
                    "idempotencyKey": key,
                    "happened": happened,
                    "reconciledBy": by,
                }),
            );
        match self.append_past_ending(draft) {
            Ok(event) => {
                if let Err(error) = self.advance_snapshot(run_id, &event) {
                    log::warn!("[tasks] run {run_id}: the snapshot was not updated: {error}");
                }
            }
            Err(AppendError::Duplicate { .. }) => {}
            Err(error) => log::warn!("[tasks] run {run_id}: reconciliation not recorded: {error}"),
        }
        Ok(true)
    }

    // -- Leases -----------------------------------------------------------

    /// Claims the right to advance a run.
    ///
    /// `Err(Held)` means something else is working it. The caller must not
    /// proceed: the event log would accept both, and the work would be done
    /// twice.
    pub fn claim_run(
        &self,
        run_id: &str,
        owner: &str,
        term: chrono::Duration,
        now: chrono::DateTime<chrono::Utc>,
    ) -> Result<Result<Lease, Held>, String> {
        let conn = self
            .conn
            .lock()
            .map_err(|_| "the task event log is poisoned".to_string())?;
        lease::acquire(&conn, run_id, owner, term, now).map_err(|error| error.to_string())
    }

    /// Extends a claim. `false` means the caller no longer holds it and must
    /// stop.
    pub fn renew_claim(
        &self,
        run_id: &str,
        owner: &str,
        fence_token: i64,
        term: chrono::Duration,
        now: chrono::DateTime<chrono::Utc>,
    ) -> Result<bool, String> {
        let conn = self
            .conn
            .lock()
            .map_err(|_| "the task event log is poisoned".to_string())?;
        lease::renew(&conn, run_id, owner, fence_token, term, now)
            .map_err(|error| error.to_string())
    }

    /// Gives a claim up. Token-checked, so a straggler cannot release a lease
    /// somebody else now holds.
    pub fn release_claim(
        &self,
        run_id: &str,
        owner: &str,
        fence_token: i64,
    ) -> Result<bool, String> {
        let conn = self
            .conn
            .lock()
            .map_err(|_| "the task event log is poisoned".to_string())?;
        lease::release(&conn, run_id, owner, fence_token).map_err(|error| error.to_string())
    }

    /// Who is working this run, if anybody. A lapsed claim reads as nobody.
    pub fn run_holder(
        &self,
        run_id: &str,
        now: chrono::DateTime<chrono::Utc>,
    ) -> Result<Option<Lease>, String> {
        let conn = self
            .conn
            .lock()
            .map_err(|_| "the task event log is poisoned".to_string())?;
        lease::holder(&conn, run_id, now).map_err(|error| error.to_string())
    }

    // -- Approvals --------------------------------------------------------

    /// Writes an approval request down before anybody is asked.
    ///
    /// Called when the request is raised rather than when it is decided, which
    /// is the whole point: a process that dies while somebody is deciding must
    /// leave the question behind, not just the answer it never got.
    pub fn record_approval(&self, approval: &DurableApproval) -> Result<(), String> {
        let conn = self
            .conn
            .lock()
            .map_err(|_| "the task event log is poisoned".to_string())?;
        approvals::record(&conn, approval).map_err(|error| error.to_string())
    }

    /// Records what a person decided.
    ///
    /// `false` means it was already decided and this decision did not take. The
    /// first answer stands: it is the one somebody gave with the run stopped in
    /// front of them.
    pub fn resolve_approval(
        &self,
        approval_id: &str,
        status: ApprovalStatus,
        by: &str,
        resolution: Option<&str>,
        at: chrono::DateTime<chrono::Utc>,
    ) -> Result<bool, String> {
        let conn = self
            .conn
            .lock()
            .map_err(|_| "the task event log is poisoned".to_string())?;
        approvals::resolve(&conn, approval_id, status, by, resolution, at)
            .map_err(|error| error.to_string())
    }

    /// One approval by id.
    pub fn approval(&self, approval_id: &str) -> Result<Option<DurableApproval>, String> {
        let conn = self
            .conn
            .lock()
            .map_err(|_| "the task event log is poisoned".to_string())?;
        approvals::get(&conn, approval_id).map_err(|error| error.to_string())
    }

    /// Everything still undecided, oldest first.
    ///
    /// What a restart has to put back in front of somebody, and the reason the
    /// table exists.
    pub fn pending_approvals(&self) -> Result<Vec<DurableApproval>, String> {
        let conn = self
            .conn
            .lock()
            .map_err(|_| "the task event log is poisoned".to_string())?;
        approvals::pending(&conn).map_err(|error| error.to_string())
    }

    /// Every approval raised for one run, decided or not.
    pub fn approvals_for_run(&self, run_id: &str) -> Result<Vec<DurableApproval>, String> {
        let conn = self
            .conn
            .lock()
            .map_err(|_| "the task event log is poisoned".to_string())?;
        approvals::for_run(&conn, run_id).map_err(|error| error.to_string())
    }

    /// How a run ended, if it has.
    ///
    /// One indexed row, so it is cheap enough to ask before every tool call —
    /// which is what makes a cancellation stop the run at a boundary rather
    /// than whenever the child process happens to notice.
    pub fn ending(&self, run_id: &str) -> Option<TaskEventType> {
        let conn = self.conn.lock().ok()?;
        ending_of(&conn, run_id).ok().flatten()
    }

    /// Every side effect nobody can account for, oldest first.
    ///
    /// What the screen that asks a person to reconcile reads.
    /// Writes the point this run can be continued from.
    ///
    /// Sequence-guarded: a checkpoint whose `last_event_seq` is behind the one
    /// already stored is dropped rather than written. Two writers racing at the
    /// end of a run — the loop settling a tool and the shutdown path recording
    /// the ending — would otherwise let whichever finished last decide the
    /// resume point, and the later writer is not always the further-along one.
    ///
    /// Returns whether the row was actually moved, so a caller that needs to
    /// know its checkpoint landed can tell that from a no-op.
    pub fn save_checkpoint(&self, checkpoint: &RunCheckpoint) -> Result<bool, String> {
        // Sealed before it is written, never trusted from the caller. A
        // checkpoint whose hash was computed elsewhere is a checkpoint whose
        // hash proves nothing about what is in this row.
        if !checkpoint.is_intact() {
            return Err(
                "the checkpoint does not match its own hash and was not written".to_string(),
            );
        }
        let body = serde_json::to_string(checkpoint)
            .map_err(|error| format!("the checkpoint could not be written: {error}"))?;

        let conn = self
            .conn
            .lock()
            .map_err(|_| "the task event log is poisoned".to_string())?;

        let held: Option<i64> = conn
            .query_row(
                "SELECT last_event_seq FROM run_checkpoints WHERE run_id = ?1",
                [&checkpoint.run_id],
                |row| row.get(0),
            )
            .optional()
            .map_err(|error| error.to_string())?;

        if let Some(held) = held {
            if held > checkpoint.last_event_seq {
                return Ok(false);
            }
        }

        conn.execute(
            "INSERT INTO run_checkpoints
                 (run_id, attempt_id, last_event_seq, state, at, schema_version, checkpoint_hash, body)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
             ON CONFLICT(run_id) DO UPDATE SET
                 attempt_id = excluded.attempt_id,
                 last_event_seq = excluded.last_event_seq,
                 state = excluded.state,
                 at = excluded.at,
                 schema_version = excluded.schema_version,
                 checkpoint_hash = excluded.checkpoint_hash,
                 body = excluded.body",
            rusqlite::params![
                checkpoint.run_id,
                checkpoint.attempt_id,
                checkpoint.last_event_seq,
                checkpoint.state.as_str(),
                checkpoint.at,
                checkpoint.schema_version,
                checkpoint.checkpoint_hash,
                body,
            ],
        )
        .map_err(|error| format!("the checkpoint could not be saved: {error}"))?;
        Ok(true)
    }

    /// The point this run can be continued from, if there is one.
    ///
    /// A row that will not parse, or whose hash does not match its body, is
    /// reported as a refusal rather than as absence. The two are different: no
    /// checkpoint means the run was never safe to continue, and a broken one
    /// means somebody should know the record was damaged.
    pub fn checkpoint(&self, run_id: &str) -> Result<Option<RunCheckpoint>, NotResumable> {
        let conn = self.conn.lock().map_err(|_| NotResumable::UnreadableCheckpoint {
            detail: "the task event log is poisoned".to_string(),
        })?;

        let body: Option<String> = conn
            .query_row(
                "SELECT body FROM run_checkpoints WHERE run_id = ?1",
                [run_id],
                |row| row.get(0),
            )
            .optional()
            .map_err(|error| NotResumable::UnreadableCheckpoint {
                detail: error.to_string(),
            })?;

        let Some(body) = body else {
            return Ok(None);
        };

        let checkpoint: RunCheckpoint = serde_json::from_str(&body).map_err(|error| {
            NotResumable::UnreadableCheckpoint {
                detail: error.to_string(),
            }
        })?;
        if !checkpoint.is_intact() {
            return Err(NotResumable::CorruptCheckpoint);
        }
        Ok(Some(checkpoint))
    }

    /// Forgets a run's resume point.
    ///
    /// Called when a run reaches an ending that leaves nothing to continue. The
    /// events stay; only the invitation to carry on goes.
    pub fn clear_checkpoint(&self, run_id: &str) -> Result<(), String> {
        let conn = self
            .conn
            .lock()
            .map_err(|_| "the task event log is poisoned".to_string())?;
        conn.execute("DELETE FROM run_checkpoints WHERE run_id = ?1", [run_id])
            .map_err(|error| format!("the checkpoint could not be cleared: {error}"))?;
        Ok(())
    }

    pub fn unknown_effects(&self) -> Result<Vec<RecordedOutcome>, String> {
        let conn = self
            .conn
            .lock()
            .map_err(|_| "the task event log is poisoned".to_string())?;
        idempotency::all_with_status(&conn, EffectStatus::Unknown)
            .map_err(|error| error.to_string())
    }

    pub fn begin_effect_fenced(&self, claim: &Lease, key: &str, tool: &str, fingerprint: &str, target: &str) -> EffectLookup {
        let result = (|| -> Result<EffectLookup,String> {
            let conn=self.conn.lock().map_err(|_| "Effect store unavailable.")?;
            let tx=rusqlite::Transaction::new_unchecked(&conn,rusqlite::TransactionBehavior::Immediate).map_err(|_| "Effect transaction unavailable.")?;
            let held=lease::holder(&tx,&claim.run_id,chrono::Utc::now()).map_err(|_| "Effect lease unavailable.")?;
            if !held.is_some_and(|held| held.owner==claim.owner && held.fence_token==claim.fence_token) { return Err("The effect belongs to an expired attempt.".into()); }
            if ending_of(&tx,&claim.run_id).map_err(|_| "Run ending unavailable.")?.is_some() { return Err("The run has ended.".into()); }
            let result=idempotency::begin(&tx,&claim.run_id,key,tool,fingerprint,target).map_err(|_| "The effect intent could not be committed.")?;
            tx.commit().map_err(|_| "The effect intent could not be committed.")?;
            Ok(result)
        })();
        result.unwrap_or_else(|reason| EffectLookup::Unavailable { reason })
    }

    pub fn settle_effect_fenced(&self, claim: &Lease, key: &str, outcome: &Result<String,String>) -> Result<(),String> {
        let conn=self.conn.lock().map_err(|_| "Effect store unavailable.")?;
        let tx=rusqlite::Transaction::new_unchecked(&conn,rusqlite::TransactionBehavior::Immediate).map_err(|_| "Effect transaction unavailable.")?;
        let held=lease::holder(&tx,&claim.run_id,chrono::Utc::now()).map_err(|_| "Effect lease unavailable.")?;
        if !held.is_some_and(|held| held.owner==claim.owner && held.fence_token==claim.fence_token) { return Err("The effect result belongs to an expired attempt.".into()); }
        idempotency::settle(&tx,&claim.run_id,key,outcome).map_err(|_| "The effect result could not be committed.")?;
        tx.commit().map_err(|_| "The effect result could not be committed.".into())
    }

    /// Read unsettled effects and approvals in one SQLite statement, not from a
    /// possibly stale UI snapshot. Pending effects also block completion.
    pub fn completion_obligations(&self, run_id: &str) -> Result<(Vec<String>, usize), String> {
        self.read_obligations(run_id,true)
    }

    pub fn effect_obligations(&self, run_id: &str) -> Result<(Vec<String>, usize), String> {
        self.read_obligations(run_id,false)
    }

    fn read_obligations(&self, run_id: &str, include_operations: bool) -> Result<(Vec<String>, usize), String> {
        let conn = self.conn.lock().map_err(|_| "durable obligations are unavailable".to_string())?;
        let read = || -> rusqlite::Result<(Vec<String>, usize)> {
            let mut statement = conn.prepare(
                "SELECT 'effect', idempotency_key FROM task_tool_effects
                   WHERE run_id = ?1 AND status IN ('pending', 'unknown')
                 UNION ALL
                 SELECT 'approval', approval_id FROM run_approvals
                   WHERE run_id = ?1 AND status = 'pending'
                 UNION ALL
                 SELECT 'effect', operation_id FROM run_tool_operations
                   WHERE run_id = ?1 AND ?2 AND status IN ('proposed','running','unknown')",
            )?;
            let rows = statement.query_map(params![run_id,include_operations], |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)))?;
            let mut effects = Vec::new();
            let mut approvals = 0;
            for row in rows {
                let (kind, id) = row?;
                if kind == "effect" { effects.push(id); } else { approvals += 1; }
            }
            Ok((effects, approvals))
        };
        read().map_err(|_| "durable obligations could not be read; reconciliation is required".into())
    }
}

struct RecoveryClaim<'a> { log: &'a TaskEventLog, claim: Lease }
impl Drop for RecoveryClaim<'_> {
    fn drop(&mut self) { let _=self.log.release_claim(&self.claim.run_id,&self.claim.owner,self.claim.fence_token); }
}

/// The endings a run can have. Kept next to the query that uses them so a new
/// terminal event type cannot be added without this list being in view.
const TERMINAL_TYPES: &[TaskEventType] = &[
    TaskEventType::RunCompleted,
    TaskEventType::RunFailed,
    TaskEventType::RunCancelled,
    TaskEventType::RunStoppedByBudget,
    TaskEventType::RunStoppedByLength,
    TaskEventType::RunStoppedByPolicy,
    TaskEventType::RunDegraded,
    // Schema 1 spellings. Still terminal when read back, so a run ended by an
    // older build is not swept up again by recovery.
    TaskEventType::RunTimedOut,
    TaskEventType::RunInterrupted,
];

fn existing_seq(conn: &Connection, event_id: &str) -> Result<Option<i64>, rusqlite::Error> {
    conn.query_row(
        "SELECT seq FROM task_events WHERE event_id = ?1",
        params![event_id],
        |row| row.get(0),
    )
    .map(Some)
    .or_else(|error| match error {
        rusqlite::Error::QueryReturnedNoRows => Ok(None),
        other => Err(other),
    })
}

pub(super) fn ending_of(conn: &Connection, run_id: &str) -> Result<Option<TaskEventType>, rusqlite::Error> {
    let terminals: Vec<&str> = TERMINAL_TYPES.iter().map(|kind| kind.as_str()).collect();
    let placeholders = vec!["?"; terminals.len()].join(", ");
    let sql = format!(
        "SELECT event_type FROM task_events
          WHERE run_id = ? AND event_type IN ({placeholders})
          ORDER BY seq LIMIT 1"
    );
    let mut bound: Vec<&dyn rusqlite::ToSql> = vec![&run_id];
    for kind in &terminals {
        bound.push(kind);
    }
    conn.query_row(&sql, bound.as_slice(), |row| row.get::<_, String>(0))
        .map(|raw| TaskEventType::from_str(&raw))
        .or_else(|error| match error {
            rusqlite::Error::QueryReturnedNoRows => Ok(None),
            other => Err(other),
        })
}

fn read_since(conn: &Connection, run_id: &str, after_seq: i64) -> Result<EventPage, String> {
    let mut statement = conn
        .prepare(
            "SELECT event_id, seq, event_type, at, actor, schema_version, payload, payload_hash
               FROM task_events WHERE run_id = ?1 AND seq > ?2 ORDER BY seq",
        )
        .map_err(|error| error.to_string())?;

    let rows = statement
        .query_map(params![run_id, after_seq], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, i64>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, String>(4)?,
                row.get::<_, i64>(5)?,
                row.get::<_, String>(6)?,
                row.get::<_, String>(7)?,
            ))
        })
        .map_err(|error| error.to_string())?;

    let mut page = EventPage::default();
    let mut seen: BTreeSet<i64> = BTreeSet::new();

    for row in rows.flatten() {
        let (event_id, seq, raw_type, at, actor, schema_version, payload, stored_hash) = row;

        // Two rows claiming one position cannot both be believed. The
        // constraint makes this unreachable through this code; it is checked
        // anyway because the file is on somebody's disk.
        if !seen.insert(seq) {
            page.unreadable.push(UnreadableEvent {
                seq,
                event_id,
                problem: "two events claim this position in the run".to_string(),
            });
            continue;
        }

        let Some(event_type) = TaskEventType::from_str(&raw_type) else {
            page.unreadable.push(UnreadableEvent {
                seq,
                event_id,
                problem: format!("{raw_type:?} is not an event type this version understands"),
            });
            continue;
        };

        if schema_version as u32 > SCHEMA_VERSION {
            page.unreadable.push(UnreadableEvent {
                seq,
                event_id,
                problem: format!(
                    "written in event format {schema_version}, and this version reads {SCHEMA_VERSION}"
                ),
            });
            continue;
        }

        let Ok(parsed) = serde_json::from_str::<Value>(&payload) else {
            page.unreadable.push(UnreadableEvent {
                seq,
                event_id,
                problem: "the payload is not readable JSON".to_string(),
            });
            continue;
        };

        // The seal is over the payload as it was written. A mismatch means the
        // row was changed underneath us, which is exactly what the seal is for
        // — and a changed payload is not evidence of anything, so it is not
        // folded into the state.
        let recomputed = payload_hash(&parsed);
        if recomputed != stored_hash {
            page.unreadable.push(UnreadableEvent {
                seq,
                event_id,
                problem: "the payload does not match its recorded hash".to_string(),
            });
            continue;
        }

        page.events.push(TaskEvent {
            run_id: run_id.to_string(),
            event_id,
            seq,
            event_type,
            at,
            actor,
            schema_version: schema_version as u32,
            payload: parsed,
            payload_hash: stored_hash,
        });
    }

    Ok(page)
}

fn rebuild_within(conn: &Connection, run_id: &str) -> Result<Option<TaskSnapshot>, String> {
    let page = read_since(conn, run_id, 0)?;
    if page.events.is_empty() && page.unreadable.is_empty() {
        return Ok(None);
    }
    Ok(Some(fold(run_id, &page.events, &page.unreadable)))
}

fn snapshot_within(conn: &Connection, run_id: &str) -> Result<Option<TaskSnapshot>, String> {
    let stored: Option<String> = conn
        .query_row(
            "SELECT state FROM task_snapshots WHERE run_id = ?1",
            params![run_id],
            |row| row.get(0),
        )
        .ok();

    let Some(mut snapshot) = stored
        .as_deref()
        .and_then(|state| serde_json::from_str::<TaskSnapshot>(state).ok())
    else {
        return rebuild_within(conn, run_id);
    };

    // Only what has happened since. This is the whole reason a snapshot exists.
    let page = read_since(conn, run_id, snapshot.seq)?;
    for event in &page.events {
        snapshot.apply(event);
    }
    for broken in page.unreadable {
        if !snapshot.unreadable_events.contains(&broken) {
            if broken.seq > snapshot.seq {
                snapshot.seq = broken.seq;
            }
            snapshot.unreadable_events.push(broken);
        }
    }
    Ok(Some(snapshot))
}

fn save_within(conn: &Connection, snapshot: &TaskSnapshot) -> Result<(), String> {
    let state = serde_json::to_string(snapshot).map_err(|error| error.to_string())?;
    conn.execute(
        "INSERT INTO task_snapshots (run_id, seq, status, schema_version, updated_at, state)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6)
         ON CONFLICT(run_id) DO UPDATE SET
             seq = excluded.seq,
             status = excluded.status,
             schema_version = excluded.schema_version,
             updated_at = excluded.updated_at,
             state = excluded.state
         WHERE excluded.seq >= task_snapshots.seq",
        params![
            snapshot.run_id,
            snapshot.seq,
            snapshot.state.as_str(),
            snapshot.schema_version,
            if snapshot.updated_at.is_empty() {
                chrono::Utc::now().to_rfc3339()
            } else {
                snapshot.updated_at.clone()
            },
            state,
        ],
    )
    .map_err(|error| format!("the task snapshot could not be saved: {error}"))?;
    Ok(())
}

/// The actor recorded when recovery closes a run off.
pub const RECOVERY_ACTOR: &str = SYSTEM_ACTOR;

/// Whether a state means the run can still be watched.
pub fn is_live(state: RunState) -> bool {
    !state.is_terminal()
}
