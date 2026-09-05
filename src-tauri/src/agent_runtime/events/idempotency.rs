//! Doing a side-effecting thing at most once, and knowing when you cannot tell.
//!
//! ## The failure this exists for
//!
//! A run asks for `create_docx`. The document is written. Before the result
//! reaches the model, the process dies. On the next start the run is recovered
//! and something asks for `create_docx` again.
//!
//! Nothing upstream can prevent that. The grant ledger ([`super::super::grants`])
//! is in memory and dies with the process, which is correct for what it defends
//! against — a *live* runtime replaying an authorisation — and useless here,
//! because after a restart there is no ledger to have spent. The plan's repeat
//! limit counts calls within one run in memory, and is likewise gone.
//!
//! So the record of "this exact side effect already happened" has to live where
//! the restart does not reach it: on disk, keyed by something both attempts
//! compute the same way.
//!
//! ## Why an intent is written before the effect
//!
//! Recording the outcome *after* the tool runs closes the easy half of the
//! problem and leaves the hard half open. If the process dies **during** the
//! write, there is no outcome row — so the next attempt sees nothing, concludes
//! the effect never happened, and does it again. On a half-written approval
//! note that is the worst available answer.
//!
//! So two writes, not one. `pending` goes down before the tool is touched, and
//! is settled to `succeeded` or `failed` afterwards. A `pending` row still
//! standing at the next start is proof that something was in flight when the
//! lights went out, and it is promoted to **`unknown`** rather than being
//! cleared or assumed either way.
//!
//! ## What `unknown` means, and why nothing automatic follows it
//!
//! It means: a side effect may or may not have happened, and this program
//! cannot find out. Re-running would risk doing it twice; assuming it happened
//! would risk never doing it at all. Both are worse than stopping, so an
//! unknown effect is refused and a person is asked what actually occurred. That
//! is the whole of [`EffectLookup::Unknown`], and it is the reason a run
//! interrupted mid-tool ends in `degraded_needs_human` rather than being
//! quietly retried.
//!
//! ## Why only side-effecting tools
//!
//! A second `search_documents` costs a little time and returns the same rows;
//! collapsing it would only hide a model going in circles from the repeat limit
//! that is supposed to catch that. A second `create_docx` overwrites somebody's
//! deliverable. Only the second kind is recorded here.

use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::orchestrator::tools::ToolName;

use super::model::{canonical, digest};

/// Whether running this tool twice does something twice.
///
/// `validate_artifact` reads and reports; `search_documents`, `read_scoped_file`
/// and `run_calculation` are pure with respect to the workspace. The rest write
/// files or run code.
pub const fn is_side_effecting(tool: ToolName) -> bool {
    matches!(
        tool,
        ToolName::WriteScopedFile
            | ToolName::CreateDocx
            | ToolName::CreateXlsx
            // A briefing deck is a file written to disk, exactly like the other
            // two documents. It was missing from this list, so a run that
            // produced one and was then interrupted had nothing recorded to
            // stop the resumption producing a second.
            | ToolName::CreatePptx
            | ToolName::ExecuteCode
    )
}

/// The key a call is remembered under when the runtime does not supply one.
///
/// Deterministic over the call itself, so the attempt before the crash and the
/// attempt after it agree without either knowing the other happened.
pub fn derive_key(run_id: &str, tool: &str, args: &Value) -> String {
    digest(&format!("{run_id}\u{1f}{tool}\u{1f}{}", canonical(args)))
}

/// A fingerprint of the arguments, kept so a key presented with *different*
/// arguments is a conflict rather than a hit.
pub fn args_fingerprint(args: &Value) -> String {
    digest(&canonical(args))
}

/// Where a recorded side effect stands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum EffectStatus {
    /// Begun, and not yet settled. Either in flight right now, or interrupted
    /// and not yet promoted to `Unknown` by recovery.
    Pending,
    Succeeded,
    Failed,
    /// In flight when the process went away. Needs a person.
    Unknown,
}

impl EffectStatus {
    pub const fn as_str(self) -> &'static str {
        match self {
            EffectStatus::Pending => "pending",
            EffectStatus::Succeeded => "succeeded",
            EffectStatus::Failed => "failed",
            EffectStatus::Unknown => "unknown",
        }
    }

    pub fn from_str(raw: &str) -> Option<Self> {
        Some(match raw {
            "pending" => EffectStatus::Pending,
            "succeeded" => EffectStatus::Succeeded,
            "failed" => EffectStatus::Failed,
            "unknown" => EffectStatus::Unknown,
            _ => return None,
        })
    }
}

/// What happened the first time a key was used.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct RecordedOutcome {
    pub idempotency_key: String,
    pub run_id: String,
    pub tool: String,
    pub args_fingerprint: String,
    pub status: EffectStatus,
    /// What the tool reported, trimmed. These are short confirmations —
    /// "Wrote note.docx" — because the tools recorded here report what they
    /// did rather than what they read.
    pub result: String,
    /// What it acted on, as a reference a person can go and look at. A file
    /// name, never contents.
    pub target: String,
    pub at: String,
}

impl RecordedOutcome {
    pub fn succeeded(&self) -> bool {
        self.status == EffectStatus::Succeeded
    }

    /// The replayed result, as the tool would have returned it.
    pub fn replay(&self) -> Result<String, String> {
        if self.succeeded() {
            Ok(self.result.clone())
        } else {
            Err(self.result.clone())
        }
    }

    /// What a person is told about an effect nobody can account for.
    pub fn unknown_refusal(&self) -> String {
        format!(
            "This action was interrupted while it was happening, so nobody can say whether it \
             took effect. {} may or may not have been written. It has not been attempted again, \
             because repeating it could do it twice. Check {} and record what you find before \
             this task is relied on.",
            self.target, self.target
        )
    }
}

/// Why a presented key could not be used as a hit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KeyConflict {
    /// The key was used before for the same tool with different arguments.
    /// Refused rather than replayed: returning the earlier result would be
    /// answering a question nobody asked.
    DifferentArguments,
    /// The key was used before for a different tool entirely.
    DifferentTool { first: String },
}

impl std::fmt::Display for KeyConflict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KeyConflict::DifferentArguments => write!(
                f,
                "this call reuses an idempotency key that was recorded for the same tool with \
                 different arguments, so the earlier result does not answer it. Refused."
            ),
            KeyConflict::DifferentTool { first } => write!(
                f,
                "this call reuses an idempotency key that was recorded for {first}. Refused."
            ),
        }
    }
}

/// What the store found when a side effect asked to begin.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EffectLookup {
    /// Never seen. The intent is now recorded and the tool may run.
    Fresh,
    /// Already settled. Return the recorded outcome; do not run the tool.
    Settled(RecordedOutcome),
    /// Another attempt at this exact effect is in flight right now.
    ///
    /// Refused rather than queued: two writers to one path in one run is a
    /// defect, and serialising them would hide it while still producing
    /// whichever result finished last.
    InFlight(RecordedOutcome),
    /// Interrupted, outcome unknowable. Refused until a person says what
    /// happened. See the module note.
    Unknown(RecordedOutcome),
    /// The key describes a different call.
    Conflict(KeyConflict),
    /// The ledger could not establish a durable intent. Execution is forbidden.
    Unavailable { reason: String },
}

/// Longest tool result kept against a key. Matches the task record's limit, so
/// a replayed result reads exactly like the recorded one.
const RESULT_CHARS: usize = 400;

pub(super) fn prepare(conn: &Connection) -> rusqlite::Result<()> {
    conn.execute_batch(
        "
        CREATE TABLE IF NOT EXISTS task_tool_effects (
            run_id            TEXT NOT NULL,
            idempotency_key   TEXT NOT NULL,
            tool              TEXT NOT NULL,
            args_fingerprint  TEXT NOT NULL,
            status            TEXT NOT NULL,
            outcome           TEXT NOT NULL,
            result            TEXT NOT NULL,
            target            TEXT NOT NULL DEFAULT '',
            at                TEXT NOT NULL,
            PRIMARY KEY (run_id, idempotency_key)
        );

        CREATE INDEX IF NOT EXISTS task_tool_effects_status_idx
            ON task_tool_effects(status);
        ",
    )?;

    // Schema 1 wrote this table without `status` or `target`. Added in place
    // rather than migrated through a rebuild: the rows are the record of side
    // effects that really happened, and re-deriving them is not possible.
    add_column_if_missing(conn, "status", "TEXT NOT NULL DEFAULT 'succeeded'")?;
    add_column_if_missing(conn, "target", "TEXT NOT NULL DEFAULT ''")?;
    Ok(())
}

/// Adds a column, ignoring the error that means it is already there.
///
/// SQLite has no `ADD COLUMN IF NOT EXISTS`, and reading `PRAGMA table_info`
/// first is the same number of round trips with more to get wrong.
fn add_column_if_missing(conn: &Connection, name: &str, decl: &str) -> rusqlite::Result<()> {
    match conn.execute_batch(&format!(
        "ALTER TABLE task_tool_effects ADD COLUMN {name} {decl};"
    )) {
        Ok(()) => Ok(()),
        Err(error) if error.to_string().contains("duplicate column name") => Ok(()),
        Err(error) => Err(error),
    }
}

/// Records the intent to perform a side effect, before it is performed.
///
/// Returns what was already known about this key. Only [`EffectLookup::Fresh`]
/// means the caller may go ahead.
pub(super) fn begin(
    conn: &Connection,
    run_id: &str,
    key: &str,
    tool: &str,
    args_fingerprint: &str,
    target: &str,
) -> rusqlite::Result<EffectLookup> {
    // One statement, so two callers racing cannot both see "nothing there" and
    // both proceed. The loser gets zero rows changed and reads what the winner
    // wrote.
    let inserted = conn.execute(
        "INSERT OR IGNORE INTO task_tool_effects
            (run_id, idempotency_key, tool, args_fingerprint, status, outcome, result, target, at)
         VALUES (?1, ?2, ?3, ?4, 'pending', 'pending', '', ?5, ?6)",
        params![
            run_id,
            key,
            tool,
            args_fingerprint,
            target,
            chrono::Utc::now().to_rfc3339(),
        ],
    )?;

    if inserted == 1 {
        return Ok(EffectLookup::Fresh);
    }

    let Some(existing) = recall(conn, run_id, key)? else {
        // An ignored insertion is not proof that an intent exists.
        return Err(rusqlite::Error::QueryReturnedNoRows);
    };

    if let Err(conflict) = matches(&existing, tool, args_fingerprint) {
        return Ok(EffectLookup::Conflict(conflict));
    }

    Ok(match existing.status {
        EffectStatus::Succeeded | EffectStatus::Failed => EffectLookup::Settled(existing),
        EffectStatus::Pending => EffectLookup::InFlight(existing),
        EffectStatus::Unknown => EffectLookup::Unknown(existing),
    })
}

/// Settles an effect that was begun. Only moves a `pending` row.
pub(super) fn settle(
    conn: &Connection,
    run_id: &str,
    key: &str,
    outcome: &Result<String, String>,
) -> rusqlite::Result<()> {
    let (status, text) = match outcome {
        Ok(text) => (EffectStatus::Succeeded, text),
        Err(reason) => (EffectStatus::Failed, reason),
    };
    let trimmed: String = text.chars().take(RESULT_CHARS).collect();
    let changed = conn.execute(
        "UPDATE task_tool_effects
            SET status = ?1, outcome = ?1, result = ?2, at = ?3
          WHERE run_id = ?4 AND idempotency_key = ?5 AND status = 'pending'",
        params![
            status.as_str(),
            trimmed,
            chrono::Utc::now().to_rfc3339(),
            run_id,
            key,
        ],
    )?;
    if changed != 1 {
        return Err(rusqlite::Error::QueryReturnedNoRows);
    }
    Ok(())
}

/// Promotes every still-`pending` effect to `unknown`.
///
/// Called once at start, before anything else can write. A `pending` row at
/// this moment cannot be in flight — the process that was performing it is
/// gone — so it is exactly the set of side effects nobody can account for.
pub(super) fn promote_pending_to_unknown(
    conn: &Connection,
) -> rusqlite::Result<Vec<RecordedOutcome>> {
    let tx = rusqlite::Transaction::new_unchecked(conn, rusqlite::TransactionBehavior::Immediate)?;
    let mut stranded = Vec::new();
    for effect in all_with_status(&tx, EffectStatus::Pending)? {
        if super::lease::holder(&tx,&effect.run_id,chrono::Utc::now())?.is_none() { stranded.push(effect); }
    }
    if stranded.is_empty() {
        return Ok(stranded);
    }
    for effect in &stranded {
        tx.execute("UPDATE task_tool_effects SET status = 'unknown' WHERE run_id=?1 AND idempotency_key=?2 AND status='pending'",params![effect.run_id,effect.idempotency_key])?;
    }
    tx.commit()?;
    Ok(stranded
        .into_iter()
        .map(|effect| RecordedOutcome {
            status: EffectStatus::Unknown,
            ..effect
        })
        .collect())
}

/// Records what a person found out about an unknown effect.
///
/// `happened` is their assertion, not a measurement, so it is stored as one:
/// the result text names who said so. Only an `unknown` row moves — a settled
/// effect is already accounted for, and letting a person overwrite it would
/// make the record editable.
pub(super) fn reconcile(
    conn: &Connection,
    run_id: &str,
    key: &str,
    happened: bool,
    by: &str,
) -> rusqlite::Result<usize> {
    let status = if happened {
        EffectStatus::Succeeded
    } else {
        EffectStatus::Failed
    };
    let note = if happened {
        format!("{by} confirmed this took effect after it was interrupted.")
    } else {
        format!("{by} confirmed this did not take effect before it was interrupted.")
    };
    conn.execute(
        "UPDATE task_tool_effects
            SET status = ?1, outcome = ?1, result = ?2, at = ?3
          WHERE run_id = ?4 AND idempotency_key = ?5 AND status = 'unknown'",
        params![
            status.as_str(),
            note,
            chrono::Utc::now().to_rfc3339(),
            run_id,
            key,
        ],
    )
}

pub(super) fn recall(
    conn: &Connection,
    run_id: &str,
    key: &str,
) -> rusqlite::Result<Option<RecordedOutcome>> {
    conn.query_row(
        "SELECT tool, args_fingerprint, status, result, target, at
           FROM task_tool_effects WHERE run_id = ?1 AND idempotency_key = ?2",
        params![run_id, key],
        |row| {
            let raw: String = row.get(2)?;
            Ok(RecordedOutcome {
                idempotency_key: key.to_string(),
                run_id: run_id.to_string(),
                tool: row.get(0)?,
                args_fingerprint: row.get(1)?,
                // An unreadable status is treated as unknown, which is the
                // conservative reading: it refuses rather than replaying.
                status: EffectStatus::from_str(&raw).unwrap_or(EffectStatus::Unknown),
                result: row.get(3)?,
                target: row.get(4)?,
                at: row.get(5)?,
            })
        },
    )
    .map(Some)
    .or_else(|error| match error {
        rusqlite::Error::QueryReturnedNoRows => Ok(None),
        other => Err(other),
    })
}

/// Every effect in a given state, for recovery and for the screen that lists
/// what needs a person.
pub(super) fn all_with_status(
    conn: &Connection,
    status: EffectStatus,
) -> rusqlite::Result<Vec<RecordedOutcome>> {
    let mut statement = conn.prepare(
        "SELECT run_id, idempotency_key, tool, args_fingerprint, status, result, target, at
           FROM task_tool_effects WHERE status = ?1 ORDER BY at",
    )?;
    let rows = statement.query_map(params![status.as_str()], |row| {
        let raw: String = row.get(4)?;
        Ok(RecordedOutcome {
            run_id: row.get(0)?,
            idempotency_key: row.get(1)?,
            tool: row.get(2)?,
            args_fingerprint: row.get(3)?,
            status: EffectStatus::from_str(&raw).unwrap_or(EffectStatus::Unknown),
            result: row.get(5)?,
            target: row.get(6)?,
            at: row.get(7)?,
        })
    })?;
    rows.collect()
}

/// Whether a recorded outcome answers the call now being made.
pub fn matches(
    recorded: &RecordedOutcome,
    tool: &str,
    args_fingerprint: &str,
) -> Result<(), KeyConflict> {
    if recorded.tool != tool {
        return Err(KeyConflict::DifferentTool {
            first: recorded.tool.clone(),
        });
    }
    if recorded.args_fingerprint != args_fingerprint {
        return Err(KeyConflict::DifferentArguments);
    }
    Ok(())
}
