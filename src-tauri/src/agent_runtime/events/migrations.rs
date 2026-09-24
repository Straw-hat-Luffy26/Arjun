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
const MIGRATIONS: &[Migration] = &[
    Migration {
        name: "run_approvals",
        // Approvals were an in-memory `Mutex<Vec<_>>` and died with the process,
        // so a run waiting on a person at the moment of a crash lost both the
        // question and the answer. `args_fingerprint` is stored beside the
        // arguments on purpose: an approval authorises a specific call, and the
        // check at resume time is that the call has not changed since a person
        // looked at it.
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
    },
    Migration {
        name: "agent_model_transitions",
        // Changing the model an agent is bound to used to be a field in an
        // edit form, so there was nothing to record and nowhere to record it.
        // This is the ledger: one row per handoff, holding where it got to and
        // enough hashes to answer afterwards what moved.
        //
        // `settled_at IS NULL` is the definition of "still going", and the
        // partial unique index below makes "at most one open handoff per agent"
        // a property of the storage engine rather than a check somebody
        // remembers to write. Two administrators pressing the button at once is
        // the ordinary case for a screen that lists agents, and two concurrent
        // handoffs of the same agent would race over one registry row.
        //
        // `run_id` is nullable on purpose: reassigning an idle agent is a real
        // transition with no run to hand over, and inventing a run id for it
        // would put a task in the task list that nobody started.
        sql: "
        CREATE TABLE IF NOT EXISTS agent_model_transitions (
            transition_id   TEXT PRIMARY KEY,
            agent_id        TEXT NOT NULL,
            run_id          TEXT,
            phase           TEXT NOT NULL,
            outcome         TEXT NOT NULL,
            from_model_id   TEXT NOT NULL,
            to_model_id     TEXT NOT NULL,
            requested_by    TEXT NOT NULL,
            requested_at    TEXT NOT NULL,
            settled_at      TEXT,
            schema_version  INTEGER NOT NULL,
            body            TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS agent_model_transitions_agent_idx
            ON agent_model_transitions(agent_id, requested_at);

        CREATE INDEX IF NOT EXISTS agent_model_transitions_run_idx
            ON agent_model_transitions(run_id);

        CREATE UNIQUE INDEX IF NOT EXISTS agent_model_transitions_one_open
            ON agent_model_transitions(agent_id)
            WHERE settled_at IS NULL;
    ",
    },
    // P05: the orchestrator's plan and the jobs it dispatched.
    //
    // Plan versions are append-only for the reason task events are: a plan
    // whose history can be rewritten cannot show that a step was once marked
    // failed, and "the model cannot erase a completed receipt" would be a
    // promise about one code path rather than about the file.
    //
    // Jobs are updated in place, guarded by status in `plans.rs`: a job moves
    // queued -> running -> one ending, and an ending is never overwritten.
    Migration {
        name: "task_plans_and_jobs",
        sql: "
        CREATE TABLE IF NOT EXISTS task_plan_versions (
            run_id       TEXT NOT NULL,
            version      INTEGER NOT NULL,
            author       TEXT NOT NULL,
            reason       TEXT NOT NULL,
            created_at   TEXT NOT NULL,
            body         TEXT NOT NULL,
            body_sha256  TEXT NOT NULL,
            PRIMARY KEY (run_id, version)
        );

        CREATE TRIGGER IF NOT EXISTS task_plan_versions_append_only_update
        BEFORE UPDATE ON task_plan_versions
        BEGIN
            SELECT RAISE(ABORT, 'task_plan_versions is append-only: rows cannot be modified');
        END;

        CREATE TRIGGER IF NOT EXISTS task_plan_versions_append_only_delete
        BEFORE DELETE ON task_plan_versions
        BEGIN
            SELECT RAISE(ABORT, 'task_plan_versions is append-only: rows cannot be deleted');
        END;

        CREATE TABLE IF NOT EXISTS delegated_jobs (
            job_id              TEXT PRIMARY KEY,
            run_id              TEXT NOT NULL,
            step_id             TEXT NOT NULL,
            role                TEXT NOT NULL,
            mode                TEXT NOT NULL,
            attempt             INTEGER NOT NULL,
            child_id            TEXT NOT NULL,
            status              TEXT NOT NULL,
            definition_id       TEXT NOT NULL DEFAULT '',
            definition_version  INTEGER,
            definition_origin   TEXT NOT NULL DEFAULT '',
            model_id            TEXT,
            parent_model_id     TEXT,
            lease               TEXT NOT NULL DEFAULT '',
            objective_sha256    TEXT NOT NULL,
            deadline_at         TEXT NOT NULL,
            created_at          TEXT NOT NULL,
            settled_at          TEXT,
            result              TEXT,
            result_hash         TEXT
        );

        CREATE INDEX IF NOT EXISTS delegated_jobs_run_idx
            ON delegated_jobs(run_id, created_at);

        CREATE INDEX IF NOT EXISTS delegated_jobs_status_idx
            ON delegated_jobs(status);
    ",
    },
];

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
