//! Fault injection against the same storage entry points used by the executor.
use super::*;

#[test]
fn a_replaced_worker_cannot_commit_a_terminal_event() {
    let store = TaskEventLog::in_memory().unwrap();
    let now = chrono::Utc::now();
    store.record(EventDraft::new("r", TaskEventType::RunCreated, "operator")).unwrap();
    let old = store.claim_run("r", "worker-1", chrono::Duration::minutes(1), now).unwrap().unwrap();
    store.release_claim("r", &old.owner, old.fence_token).unwrap();
    let current = store.claim_run("r", "worker-2", chrono::Duration::minutes(1), now).unwrap().unwrap();
    assert!(store.record_fenced(EventDraft::new("r", TaskEventType::RunCompleted, "operator"), &old).is_err());
    assert_eq!(store.events_since("r", 0).unwrap().events.len(), 1);
    store.record_fenced(EventDraft::new("r", TaskEventType::RunCompleted, "operator"), &current).unwrap();
    assert_eq!(store.snapshot("r").unwrap().unwrap().state, RunState::Completed);
}

#[test]
fn an_unwritable_intent_never_authorizes_an_effect() {
    let store = TaskEventLog::in_memory().expect("store");
    store.conn.lock().expect("lock").execute_batch(
        "CREATE TRIGGER fail_intent BEFORE INSERT ON task_tool_effects
         BEGIN SELECT RAISE(ABORT, 'injected disk failure'); END;",
    ).expect("install fault");

    assert!(
        !matches!(store.begin_effect("run", "operation", "workspace.write_text", "args", "note.txt"), EffectLookup::Fresh),
        "a tool must not run when its intent could not be persisted"
    );
}

#[test]
fn a_poisoned_store_never_authorizes_an_effect() {
    let store = TaskEventLog::in_memory().expect("store");
    let connection = Arc::clone(&store.conn);
    let _ = std::thread::spawn(move || {
        let _held = connection.lock().expect("lock");
        panic!("injected worker failure");
    }).join();

    assert!(
        !matches!(store.begin_effect("run", "operation", "workspace.write_text", "args", "note.txt"), EffectLookup::Fresh),
        "a poisoned store is not evidence that an operation is fresh"
    );
}

#[test]
fn an_ignored_insert_without_a_row_is_not_a_fresh_intent() {
    let store = TaskEventLog::in_memory().expect("store");
    store.conn.lock().expect("lock").execute_batch(
        "CREATE TRIGGER ignore_intent BEFORE INSERT ON task_tool_effects
         BEGIN SELECT RAISE(IGNORE); END;",
    ).expect("install fault");
    assert!(matches!(
        store.begin_effect("run", "operation", "workspace.write_text", "args", "note.txt"),
        EffectLookup::Unavailable { .. }
    ));
}

#[test]
fn a_failed_result_commit_stays_unsettled_and_must_be_reconciled() {
    let store = TaskEventLog::in_memory().expect("store");
    assert_eq!(store.begin_effect("run", "operation", "workspace.write_text", "args", "note.txt"), EffectLookup::Fresh);
    store.conn.lock().expect("lock").execute_batch(
        "CREATE TRIGGER fail_settlement BEFORE UPDATE ON task_tool_effects
         BEGIN SELECT RAISE(ABORT, 'injected result-write failure'); END;",
    ).expect("install fault");
    assert!(store.settle_effect("run", "operation", &Ok("Wrote note.txt".into())).is_err());
    assert!(matches!(store.begin_effect("run", "operation", "workspace.write_text", "args", "note.txt"), EffectLookup::InFlight(_)));
    store.conn.lock().expect("lock").execute_batch("DROP TRIGGER fail_settlement").expect("restore writes");
    store.strand_pending_effects().expect("restart reconciliation");
    assert!(matches!(store.begin_effect("run", "operation", "workspace.write_text", "args", "note.txt"), EffectLookup::Unknown(_)));
}

#[test]
fn settlement_without_a_pending_intent_is_refused() {
    let store = TaskEventLog::in_memory().expect("store");
    assert!(store.settle_effect("run", "missing", &Ok("not recorded".into())).is_err());
    store.begin_effect("run", "operation", "workspace.write_text", "args", "note.txt");
    store.settle_effect("run", "operation", &Ok("first result".into())).expect("settled");
    assert!(store.settle_effect("run", "operation", &Ok("replacement result".into())).is_err());
    match store.begin_effect("run", "operation", "workspace.write_text", "args", "note.txt") {
        EffectLookup::Settled(recorded) => assert_eq!(recorded.result, "first result"),
        other => panic!("expected original result, got {other:?}"),
    }
}
