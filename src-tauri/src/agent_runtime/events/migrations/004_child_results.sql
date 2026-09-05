CREATE TABLE run_child_results (
    run_id TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    policy_hash TEXT NOT NULL,
    body TEXT NOT NULL,
    body_hash TEXT NOT NULL,
    PRIMARY KEY (run_id, idempotency_key)
);
CREATE TRIGGER child_results_no_update BEFORE UPDATE ON run_child_results
BEGIN SELECT RAISE(ABORT, 'child results are immutable'); END;
CREATE TRIGGER child_results_no_delete BEFORE DELETE ON run_child_results
BEGIN SELECT RAISE(ABORT, 'child results are immutable'); END;
