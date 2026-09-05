-- Full private tool receipts. Keep these records on downgrade.
CREATE TABLE IF NOT EXISTS run_tool_operations (
    run_id TEXT NOT NULL,
    operation_id TEXT NOT NULL,
    message_seq INTEGER NOT NULL,
    tool_call_id TEXT NOT NULL,
    tool TEXT NOT NULL,
    arguments TEXT NOT NULL,
    args_hash TEXT NOT NULL,
    class TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('proposed','running','succeeded','failed','unknown')),
    fence_token INTEGER NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    base_revision INTEGER NOT NULL,
    result TEXT,
    result_hash TEXT,
    core_state TEXT,
    core_hash TEXT,
    provider_request_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (run_id, operation_id),
    UNIQUE (run_id, message_seq, tool_call_id)
);
CREATE UNIQUE INDEX IF NOT EXISTS one_running_operation_per_run
ON run_tool_operations(run_id) WHERE status = 'running';
CREATE TRIGGER IF NOT EXISTS operation_identity_immutable
BEFORE UPDATE OF run_id,operation_id,message_seq,tool_call_id,tool,arguments,args_hash,class,created_at ON run_tool_operations
BEGIN SELECT RAISE(ABORT, 'operation intent is immutable'); END;
CREATE TRIGGER IF NOT EXISTS settled_operation_immutable
BEFORE UPDATE ON run_tool_operations WHEN OLD.status IN ('succeeded','failed')
BEGIN SELECT RAISE(ABORT, 'settled operation is immutable'); END;
CREATE TRIGGER IF NOT EXISTS operations_no_delete
BEFORE DELETE ON run_tool_operations BEGIN SELECT RAISE(ABORT, 'operation history is retained'); END;
