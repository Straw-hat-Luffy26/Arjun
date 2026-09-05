//! Schema versions, applied in order and recorded in the database.
//!
//! ## Why this exists
//!
//! The baseline schema is created with `CREATE TABLE IF NOT EXISTS`, which is
//! the right thing for a table that has never changed and says nothing at all
//! about one that has. When a column had to be added, the answer was
//! `idempotency::add_column_if_missing` — an `ALTER TABLE` that swallows the
//! "duplicate column name" error and tries again on every open.
//!
//! That works, and it does not scale past a couple of columns. It cannot drop
//! anything, cannot backfill, cannot reorder, and cannot tell you what version
//! a database in the field is on: every open re-runs every change and infers
//! "already done" from an error string. A database that fails half way through
//! a multi-statement change is left in a state nothing describes.
//!
//! So changes are numbered, applied in one transaction, and the number is
//! written into the file itself. `PRAGMA user_version` is SQLite's own slot for
//! exactly this, is transactional, and costs no table of our own.
//!
//! ## The rules
//!
//! - **Append only.** A migration that has shipped is never edited, because a
//!   database in the field has already applied it and will not apply it again.
//!   Change it by adding the next one.
//! - **One transaction.** Every pending migration commits together or none of
//!   them does. There is no half-migrated database to reason about.
//! - **Idempotent statements anyway.** `IF NOT EXISTS` throughout, so a
//!   database that predates this runner and already has a table — created by
//!   the baseline batch — is not an error.
//!
//! The baseline schema in [`super::store`] is deliberately *not* migration 0.
//! It is still executed on every open, unchanged, because databases in the
//! field already have those tables and no version number recorded. Treating it
//! as a migration would mean deciding what version an existing file is on, and
//! there is no honest way to answer that.

use rusqlite::Connection;

/// One schema change, in the order it must be applied.
struct Migration {
    /// What it does, for the log line and for reading this list.
    name: &'static str,
    sql: &'static str,
}

/// Every schema change since the baseline, oldest first.
///
/// The index is the version: after applying all of these, `user_version` is
/// `MIGRATIONS.len()`.
const MIGRATIONS: &[Migration] = &[Migration {
    name: "run_approvals",
    // Approvals were an in-memory `Mutex<Vec<_>>` and died with the process, so
    // a run waiting on a person at the moment of a crash lost both the question
    // and the answer. `args_fingerprint` is stored beside the arguments on
    // purpose: an approval authorises a specific call, and the check at resume
    // time is that the call has not changed since a person looked at it.
    sql: "
        CREATE TABLE IF NOT EXISTS run_approvals (
            approval_id       TEXT PRIMARY KEY,
            run_id            TEXT NOT NULL,
            tool              TEXT NOT NULL,
            target            TEXT NOT NULL DEFAULT '',
            args_fingerprint  TEXT NOT NULL,
            arguments         TEXT NOT NULL,
            reason            TEXT NOT NULL DEFAULT '',
            status            TEXT NOT NULL,
            allowed_decisions TEXT NOT NULL DEFAULT '[]',
            created_at        TEXT NOT NULL,
            expires_at        TEXT,
            resolved_at       TEXT,
            resolved_by       TEXT,
            resolution        TEXT
        );

        CREATE INDEX IF NOT EXISTS run_approvals_run_idx
            ON run_approvals(run_id, status);

        CREATE INDEX IF NOT EXISTS run_approvals_status_idx
            ON run_approvals(status);
    ",
}, Migration {
    name: "durable_context_v1",
    sql: include_str!("migrations/002_durable_context.sql"),
}, Migration {
    name: "tool_operations_v1",
    sql: include_str!("migrations/003_tool_operations.sql"),
}];

/// Applies every migration the database has not had, and returns the version
/// it is now on.
///
/// A database already at or past the last version is left completely alone —
/// no statements run, no transaction is opened.
pub(super) fn apply(conn: &Connection) -> rusqlite::Result<u32> {
    let current: u32 = conn.query_row("PRAGMA user_version", [], |row| row.get(0))?;
    let target = MIGRATIONS.len() as u32;
    if current >= target {
        return Ok(current);
    }

    let tx = conn.unchecked_transaction()?;
    for migration in &MIGRATIONS[current as usize..] {
        tx.execute_batch(migration.sql).map_err(|error| {
            log::error!(
                "[tasks] schema migration {:?} failed; the database is unchanged: {error}",
                migration.name
            );
            error
        })?;
    }
    // `PRAGMA` takes no parameters, and `target` is a list length rather than
    // anything a caller supplies.
    tx.execute_batch(&format!("PRAGMA user_version = {target};"))?;
    tx.commit()?;
    Ok(target)
}

#[cfg(test)]
mod tests {
    use super::{apply, MIGRATIONS};
    use rusqlite::Connection;

    fn version(conn: &Connection) -> u32 {
        conn.query_row("PRAGMA user_version", [], |row| row.get(0))
            .expect("user_version is always readable")
    }

    #[test]
    fn a_fresh_database_lands_on_the_latest_version() {
        let conn = Connection::open_in_memory().expect("an in-memory database");
        assert_eq!(version(&conn), 0);
        assert_eq!(
            apply(&conn).expect("migrations apply"),
            MIGRATIONS.len() as u32
        );
        assert_eq!(version(&conn), MIGRATIONS.len() as u32);
    }

    #[test]
    fn applying_twice_changes_nothing_the_second_time() {
        let conn = Connection::open_in_memory().expect("an in-memory database");
        apply(&conn).expect("first apply");
        let after_first = version(&conn);
        apply(&conn).expect("second apply is a no-op");
        assert_eq!(version(&conn), after_first);
    }

    /// The table the first migration exists for is actually there, and takes a
    /// row. A migration that runs and leaves nothing usable is the failure this
    /// guards against.
    #[test]
    fn the_approvals_table_exists_and_accepts_a_row() {
        let conn = Connection::open_in_memory().expect("an in-memory database");
        apply(&conn).expect("migrations apply");
        conn.execute(
            "INSERT INTO run_approvals
                (approval_id, run_id, tool, args_fingerprint, arguments, status, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            rusqlite::params![
                "approval-1",
                "run-1",
                "create_docx",
                "fingerprint",
                "{}",
                "pending",
                "2026-09-04T00:00:00Z"
            ],
        )
        .expect("the row inserts");

        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM run_approvals", [], |row| row.get(0))
            .expect("the count reads");
        assert_eq!(count, 1);
    }

    /// A database that predates this runner already has the baseline tables and
    /// no version recorded. Migrating it must not trip over what is already
    /// there.
    #[test]
    fn a_database_that_predates_the_runner_migrates_cleanly() {
        let conn = Connection::open_in_memory().expect("an in-memory database");
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS run_approvals (
                 approval_id       TEXT PRIMARY KEY,
                 run_id            TEXT NOT NULL,
                 tool              TEXT NOT NULL,
                 target            TEXT NOT NULL DEFAULT '',
                 args_fingerprint  TEXT NOT NULL,
                 arguments         TEXT NOT NULL,
                 reason            TEXT NOT NULL DEFAULT '',
                 status            TEXT NOT NULL,
                 allowed_decisions TEXT NOT NULL DEFAULT '[]',
                 created_at        TEXT NOT NULL,
                 expires_at        TEXT,
                 resolved_at       TEXT,
                 resolved_by       TEXT,
                 resolution        TEXT
             );",
        )
        .expect("the pre-existing table is created");

        assert_eq!(
            apply(&conn).expect("migrations apply"),
            MIGRATIONS.len() as u32
        );
    }
}
