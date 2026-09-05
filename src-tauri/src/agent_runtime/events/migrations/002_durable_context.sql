-- Additive, forward-only migration. No DOWN: dropping these append-only tables
-- would destroy recovery evidence. A rollback keeps them and runs the old app.
CREATE TABLE IF NOT EXISTS run_context_messages (
    run_id TEXT NOT NULL,
    seq INTEGER NOT NULL CHECK (seq > 0),
    entry_id TEXT NOT NULL,
    body TEXT NOT NULL,
    body_hash TEXT NOT NULL,
    at TEXT NOT NULL,
    PRIMARY KEY (run_id, seq),
    UNIQUE (run_id, entry_id)
);
CREATE TABLE IF NOT EXISTS run_context_commits (
    run_id TEXT NOT NULL,
    revision INTEGER NOT NULL CHECK (revision > 0),
    commit_id TEXT NOT NULL,
    request_hash TEXT NOT NULL,
    fence_token INTEGER NOT NULL,
    body TEXT NOT NULL,
    body_hash TEXT NOT NULL,
    at TEXT NOT NULL,
    PRIMARY KEY (run_id, revision),
    UNIQUE (run_id, commit_id)
);
CREATE TRIGGER IF NOT EXISTS context_messages_no_update
BEFORE UPDATE ON run_context_messages BEGIN SELECT RAISE(ABORT, 'context history is append-only'); END;
CREATE TRIGGER IF NOT EXISTS context_messages_no_delete
BEFORE DELETE ON run_context_messages BEGIN SELECT RAISE(ABORT, 'context history is append-only'); END;
CREATE TRIGGER IF NOT EXISTS context_commits_no_update
BEFORE UPDATE ON run_context_commits BEGIN SELECT RAISE(ABORT, 'context checkpoints are append-only'); END;
CREATE TRIGGER IF NOT EXISTS context_commits_no_delete
BEFORE DELETE ON run_context_commits BEGIN SELECT RAISE(ABORT, 'context checkpoints are append-only'); END;
