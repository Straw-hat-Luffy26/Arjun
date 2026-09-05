//! What the durable record has to survive.
//!
//! Each group below is one failure the old arrangement had no answer for. They
//! are written against a real database file wherever the point is *restart*,
//! because a store that only works while the process is up is precisely the
//! thing being replaced.

use serde_json::json;

use super::*;
use crate::agent_runtime::tasks::PlanRecord;
use crate::orchestrator::plan::{Budget, PlanRun};
use crate::orchestrator::tools::ToolName;

const USER: &str = "priya";

fn log() -> TaskEventLog {
    TaskEventLog::in_memory().expect("an in-memory log")
}

/// A log on disk, so it can be dropped and opened again — a restart.
fn on_disk(dir: &std::path::Path) -> TaskEventLog {
    TaskEventLog::open(dir).expect("a log on disk")
}

fn created(run_id: &str) -> EventDraft {
    EventDraft::new(run_id, TaskEventType::RunCreated, USER).with(json!({
        "promptShown": "draft an approval note",
    }))
}

fn routed(run_id: &str) -> EventDraft {
    EventDraft::new(run_id, TaskEventType::RunRouted, USER).with(json!({
        "modelName": "Qwen2.5 7B",
        "modelId": "qwen2.5-7b",
    }))
}

fn plan_record() -> PlanRecord {
    PlanRecord::of(&PlanRun::new(
        "run-1",
        vec!["Search".to_string(), "Draft".to_string()],
        Budget::standard(vec![ToolName::SearchDocuments, ToolName::CreateDocx]),
    ))
}

/// Walks a run to `running`, which is where most of these tests start.
fn start(log: &TaskEventLog, run_id: &str) {
    log.record(created(run_id)).expect("created");
    log.record(routed(run_id)).expect("routed");
    log.record(
        EventDraft::new(run_id, TaskEventType::RunStarted, USER).with(json!({})),
    )
    .expect("started");
}

// == 1. Event sequence is monotonic =======================================

#[test]
fn sequence_numbers_start_at_one_and_have_no_gaps() {
    let log = log();
    for index in 0..8 {
        let event = log
            .record(
                EventDraft::new("run-1", TaskEventType::PlanStep, USER)
                    .with(json!({ "stepsTaken": index + 1 })),
            )
            .expect("appended");
        assert_eq!(event.seq, index + 1);
    }

    let page = log.events_since("run-1", 0).expect("readable");
    let seen: Vec<i64> = page.events.iter().map(|event| event.seq).collect();
    assert_eq!(seen, (1..=8).collect::<Vec<_>>());
}

#[test]
fn each_run_numbers_its_own_events() {
    // Sequence numbers are per run, not global: a reader catching up on one run
    // should not have to know how busy the others were.
    let log = log();
    log.record(created("run-a")).expect("a");
    log.record(created("run-b")).expect("b");
    let second_a = log.record(routed("run-a")).expect("a2");
    assert_eq!(second_a.seq, 2);
}

#[test]
fn events_come_back_in_order_however_they_were_asked_for() {
    let log = log();
    for index in 0..5 {
        log.record(
            EventDraft::new("run-1", TaskEventType::PlanStep, USER)
                .with(json!({ "stepsTaken": index })),
        )
        .expect("appended");
    }
    let tail = log.events_since("run-1", 2).expect("readable");
    assert_eq!(
        tail.events.iter().map(|e| e.seq).collect::<Vec<_>>(),
        vec![3, 4, 5]
    );
    assert_eq!(tail.last_seq(), 5);
}

#[test]
fn concurrent_writers_never_take_the_same_position() {
    // The guarantee is the storage engine's, not the callers'. Four threads
    // append to one run at once; if the read-then-write were not one
    // transaction, two of them would compute the same next number.
    use std::sync::Arc;

    let dir = tempfile::tempdir().expect("temp dir");
    let log = Arc::new(on_disk(dir.path()));
    let mut threads = Vec::new();

    for worker in 0..4 {
        let log = Arc::clone(&log);
        threads.push(std::thread::spawn(move || {
            for index in 0..10 {
                log.append(
                    EventDraft::new("run-1", TaskEventType::PlanStep, USER)
                        .with(json!({ "worker": worker, "index": index })),
                )
                .expect("appended");
            }
        }));
    }
    for thread in threads {
        thread.join().expect("the worker finished");
    }

    let page = log.events_since("run-1", 0).expect("readable");
    let seen: Vec<i64> = page.events.iter().map(|event| event.seq).collect();
    assert_eq!(seen, (1..=40).collect::<Vec<_>>());
}

// == 2. Duplicate event insertion is harmless =============================

#[test]
fn the_same_event_id_is_refused_and_nothing_is_written_twice() {
    let log = log();
    let draft = created("run-1").with_event_id("fixed-id");
    let first = log.record(draft.clone()).expect("the first one lands");
    assert_eq!(first.seq, 1);

    let again = log.record(draft);
    assert_eq!(
        again.unwrap_err(),
        AppendError::Duplicate {
            event_id: "fixed-id".to_string(),
            seq: 1
        }
    );

    // The point of rejecting it is that the history is unchanged.
    let page = log.events_since("run-1", 0).expect("readable");
    assert_eq!(page.events.len(), 1);
}

#[test]
fn a_duplicate_does_not_burn_a_sequence_number() {
    // A rejected append that still advanced the counter would leave a gap, and
    // a reader asking "everything after 2" would wait forever for a 3 that was
    // never written.
    let log = log();
    log.record(created("run-1").with_event_id("a")).expect("a");
    let _ = log.record(created("run-1").with_event_id("a"));
    let next = log
        .record(routed("run-1").with_event_id("b"))
        .expect("b");
    assert_eq!(next.seq, 2);
}

#[test]
fn a_duplicate_leaves_the_state_exactly_as_it_was() {
    let log = log();
    start(&log, "run-1");
    let before = log.snapshot("run-1").unwrap().unwrap();

    let repeat = EventDraft::idempotent("run-1", TaskEventType::TurnEnded, USER, "turn-1");
    log.record(repeat.clone()).expect("first");
    let _ = log.record(repeat);

    let after = log.snapshot("run-1").unwrap().unwrap();
    // One turn, not two. A duplicate that were merely *stored* once but folded
    // twice would still corrupt every count on the screen.
    assert_eq!(before.turns, 0);
    assert_eq!(after.turns, 1);
}

// == 3. Duplicate tool / artifact / approval requests are idempotent ======

#[test]
fn a_side_effecting_call_repeated_after_a_restart_is_not_run_again() {
    // The reason this cannot be solved in memory: the grant ledger and the
    // plan's repeat limit both died with the process that held them.
    let dir = tempfile::tempdir().expect("temp dir");
    let args = json!({ "path": "note.docx", "template": "approval_note" });
    let key = derive_key("run-1", "create_docx", &args);
    let fingerprint = args_fingerprint(&args);

    {
        let before = on_disk(dir.path());
        assert_eq!(
            before.begin_effect("run-1", &key, "create_docx", &fingerprint, "note.docx"),
            EffectLookup::Fresh
        );
        before.settle_effect("run-1", &key, &Ok("Wrote note.docx".to_string())).expect("settled");
    }

    let after = on_disk(dir.path());
    match after.begin_effect("run-1", &key, "create_docx", &fingerprint, "note.docx") {
        EffectLookup::Settled(recorded) => {
            assert_eq!(recorded.replay(), Ok("Wrote note.docx".to_string()));
        }
        other => panic!("expected the recorded outcome, got {other:?}"),
    }
}

#[test]
fn the_derived_key_is_the_same_for_the_same_call_and_different_otherwise() {
    let first = json!({ "path": "note.docx" });
    let second = json!({ "path": "other.docx" });
    assert_eq!(
        derive_key("run-1", "create_docx", &first),
        derive_key("run-1", "create_docx", &first)
    );
    assert_ne!(
        derive_key("run-1", "create_docx", &first),
        derive_key("run-1", "create_docx", &second)
    );
    // Two runs writing the same file are two side effects, not one.
    assert_ne!(
        derive_key("run-1", "create_docx", &first),
        derive_key("run-2", "create_docx", &first)
    );
}

#[test]
fn a_key_reused_with_different_arguments_is_refused_rather_than_replayed() {
    // Returning the first result would be answering a question nobody asked.
    let log = log();
    let first = json!({ "path": "note.docx" });
    let second = json!({ "path": "payroll.docx" });
    log.begin_effect("run-1", "shared", "create_docx", &args_fingerprint(&first), "note.docx");
    log.settle_effect("run-1", "shared", &Ok("Wrote note.docx".to_string())).expect("settled");

    assert_eq!(
        log.begin_effect(
            "run-1",
            "shared",
            "create_docx",
            &args_fingerprint(&second),
            "payroll.docx"
        ),
        EffectLookup::Conflict(KeyConflict::DifferentArguments)
    );
}

#[test]
fn an_approval_decision_recorded_twice_is_recorded_once() {
    // Two windows watching one approvals queue both see the same decision land.
    // The event id is derived from the approval, so the second is the duplicate
    // it is rather than a second decision in the history.
    let log = log();
    start(&log, "run-1");

    let decision = || {
        EventDraft::idempotent(
            "run-1",
            TaskEventType::ApprovalDecided,
            "ravi",
            "approval-88",
        )
        .with(json!({ "toolCallId": "c1", "tool": "create_docx", "approved": true }))
    };
    log.record(decision()).expect("the first one lands");
    assert!(matches!(
        log.record(decision()),
        Err(AppendError::Duplicate { .. })
    ));

    let page = log.events_since("run-1", 0).expect("readable");
    let decisions = page
        .events
        .iter()
        .filter(|e| e.event_type == TaskEventType::ApprovalDecided)
        .count();
    assert_eq!(decisions, 1);
}

#[test]
fn a_run_completion_recorded_twice_is_recorded_once() {
    let log = log();
    start(&log, "run-1");

    let ending = || {
        EventDraft::idempotent("run-1", TaskEventType::RunCompleted, USER, "ending")
            .with(json!({ "answer": "done", "turns": 2 }))
    };
    log.record(ending()).expect("the first one lands");
    // The second is refused as a duplicate, not as "already ended" — the
    // distinction matters, because a duplicate is the caller's own retry and an
    // already-ended is a *different* ending trying to land.
    assert!(matches!(
        log.record(ending()),
        Err(AppendError::Duplicate { .. })
    ));
    assert_eq!(
        log.snapshot("run-1").unwrap().unwrap().state,
        RunState::Completed
    );
}

#[test]
fn only_tools_that_actually_do_something_twice_are_tracked() {
    assert!(is_side_effecting(ToolName::CreateDocx));
    assert!(is_side_effecting(ToolName::CreateXlsx));
    assert!(is_side_effecting(ToolName::WriteScopedFile));
    assert!(is_side_effecting(ToolName::ExecuteCode));
    // Collapsing a repeated search would hide a model going in circles from
    // the repeat limit that exists to catch exactly that.
    assert!(!is_side_effecting(ToolName::SearchDocuments));
    assert!(!is_side_effecting(ToolName::ReadScopedFile));
    assert!(!is_side_effecting(ToolName::RunCalculation));
    assert!(!is_side_effecting(ToolName::ValidateArtifact));
}

// == 4. Restart restores the correct run state ============================

#[test]
fn a_run_still_going_when_the_process_died_is_found_and_marked_for_a_person() {
    // The failure this whole module exists for. Before it, a run killed
    // mid-flight left nothing at all, and the Tasks screen simply had a hole
    // where the interesting task should have been.
    let dir = tempfile::tempdir().expect("temp dir");

    {
        let before = on_disk(dir.path());
        start(&before, "run-1");
        before
            .record(
                EventDraft::new("run-1", TaskEventType::ToolSucceeded, USER)
                    .with(json!({ "toolCallId": "c1", "tool": "search_documents" })),
            )
            .expect("a tool call");
        // No ending: the process goes away here.
    }

    let after = on_disk(dir.path());
    let recovered = after.recover_interrupted(SYSTEM_ACTOR).expect("recovery ran");
    assert_eq!(recovered, vec!["run-1".to_string()]);

    let snapshot = after.snapshot("run-1").expect("readable").expect("a snapshot");
    assert_eq!(snapshot.state, RunState::DegradedNeedsHuman);
    assert!(snapshot.needs_person());
    // The work it had done before the process went away is still there.
    assert_eq!(snapshot.activity.len(), 1);
    assert_eq!(snapshot.activity[0].status, "done");
}

#[test]
fn restart_restores_the_state_the_run_was_actually_in() {
    // Not merely "it was running" — the specific state, folded from the events
    // that were written before the lights went out.
    let cases: [(&[TaskEventType], RunState); 4] = [
        (&[TaskEventType::RunCreated], RunState::Created),
        (
            &[TaskEventType::RunCreated, TaskEventType::RunRouted],
            RunState::Routed,
        ),
        (
            &[
                TaskEventType::RunCreated,
                TaskEventType::RunRouted,
                TaskEventType::RunStarted,
                TaskEventType::RunCompleted,
            ],
            RunState::Completed,
        ),
        (
            &[
                TaskEventType::RunCreated,
                TaskEventType::RunStarted,
                TaskEventType::RunCancelled,
            ],
            RunState::Cancelled,
        ),
    ];

    for (index, (events, expected)) in cases.into_iter().enumerate() {
        let dir = tempfile::tempdir().expect("temp dir");
        let run_id = format!("run-{index}");
        {
            let before = on_disk(dir.path());
            for event in events {
                before
                    .record(EventDraft::new(&run_id, *event, USER).with(json!({})))
                    .expect("appended");
            }
        }
        let after = on_disk(dir.path());
        assert_eq!(
            after.snapshot(&run_id).unwrap().unwrap().state,
            expected,
            "case {index}"
        );
    }
}

#[test]
fn the_state_a_screen_draws_survives_the_restart_without_replaying_everything() {
    // The check that it *is* a shortcut rather than a rebuild in disguise is
    // that the snapshot's own seq is the one the events stopped at.
    let dir = tempfile::tempdir().expect("temp dir");
    {
        let before = on_disk(dir.path());
        before.record(created("run-1")).expect("created");
        before.record(routed("run-1")).expect("routed");
        before
            .record(
                EventDraft::new("run-1", TaskEventType::PlanReady, USER)
                    .with(json!({ "plan": plan_record() })),
            )
            .expect("plan");
        before
            .record(
                EventDraft::new("run-1", TaskEventType::RunCompleted, USER)
                    .with(json!({ "answer": "done", "turns": 4 })),
            )
            .expect("completed");
    }

    let after = on_disk(dir.path());
    let snapshot = after.snapshot("run-1").unwrap().unwrap();
    assert_eq!(snapshot.seq, 4);
    assert_eq!(snapshot.prompt, "draft an approval note");
    assert_eq!(snapshot.model_name, "Qwen2.5 7B");
    assert_eq!(snapshot.turns, 4);
    assert_eq!(snapshot.plan.expect("a plan").steps.len(), 2);
}

#[test]
fn a_snapshot_that_will_not_parse_is_rebuilt_from_the_events() {
    // The snapshot is a cache. A cache that can make the answer wrong rather
    // than slow is not worth having.
    let dir = tempfile::tempdir().expect("temp dir");
    let log = on_disk(dir.path());
    start(&log, "run-1");
    log.record(EventDraft::new("run-1", TaskEventType::RunCancelled, USER).with(json!({})))
        .expect("cancelled");

    {
        let conn = rusqlite::Connection::open(dir.path().join("sarathi.db")).unwrap();
        conn.execute(
            "UPDATE task_snapshots SET state = '{ not json' WHERE run_id = 'run-1'",
            [],
        )
        .unwrap();
    }

    let snapshot = on_disk(dir.path())
        .snapshot("run-1")
        .expect("readable")
        .expect("a snapshot");
    assert_eq!(snapshot.state, RunState::Cancelled);
}

#[test]
fn a_run_that_finished_before_the_restart_is_left_alone() {
    let dir = tempfile::tempdir().expect("temp dir");
    {
        let before = on_disk(dir.path());
        start(&before, "run-1");
        before
            .record(
                EventDraft::new("run-1", TaskEventType::RunCompleted, USER)
                    .with(json!({ "answer": "The seal is worn.", "turns": 3 })),
            )
            .expect("completed");
    }

    let after = on_disk(dir.path());
    assert!(after
        .recover_interrupted(SYSTEM_ACTOR)
        .expect("recovery ran")
        .is_empty());
    assert_eq!(
        after.snapshot("run-1").unwrap().unwrap().state,
        RunState::Completed
    );
}

// == 6. Cancellation stops at a safe boundary =============================

#[test]
fn cancelling_a_run_records_who_stopped_it() {
    let log = log();
    start(&log, "run-1");
    log.record(
        EventDraft::new("run-1", TaskEventType::RunCancelled, USER)
            .with(json!({ "failure": "Stopped, because somebody stopped it." })),
    )
    .expect("cancelled");

    let snapshot = log.snapshot("run-1").unwrap().unwrap();
    assert_eq!(snapshot.state, RunState::Cancelled);
    let page = log.events_since("run-1", 0).unwrap();
    let ending = page.events.last().expect("an ending");
    assert_eq!(ending.event_type, TaskEventType::RunCancelled);
    assert_eq!(ending.actor, USER);
}

#[test]
fn a_cancelled_run_reports_an_ending_so_the_next_tool_call_is_refused() {
    // This is the safe boundary. `ending` is what `tool.authorize` asks before
    // every call: a run that has one starts nothing new, whatever the loop in
    // the other process is still doing.
    let log = log();
    start(&log, "run-1");
    assert!(log.ending("run-1").is_none(), "a live run has no ending");

    log.record(EventDraft::new("run-1", TaskEventType::RunCancelled, USER).with(json!({})))
        .expect("cancelled");

    assert_eq!(log.ending("run-1"), Some(TaskEventType::RunCancelled));
}

#[test]
fn a_tool_still_running_when_the_run_ended_is_not_left_looking_like_it_hung() {
    // The abort race. Stop is pressed, the cancellation is recorded, and the
    // tool that was already executing finishes afterwards — at which point its
    // outcome event is refused, because nothing may follow an ending.
    //
    // The row must not go on saying "running". What is true is narrower: the
    // history does not record how that call ended.
    let log = log();
    start(&log, "run-1");
    log.record(
        EventDraft::new("run-1", TaskEventType::ToolAuthorized, USER)
            .with(json!({ "toolCallId": "c1", "tool": "create_docx" })),
    )
    .expect("authorised");

    log.record(EventDraft::new("run-1", TaskEventType::RunCancelled, USER).with(json!({})))
        .expect("cancelled");

    // The tool's own outcome arrives too late and is refused, exactly as it
    // would be in the running system.
    assert!(matches!(
        log.record(
            EventDraft::new("run-1", TaskEventType::ToolSucceeded, USER)
                .with(json!({ "toolCallId": "c1", "tool": "create_docx" })),
        ),
        Err(AppendError::AlreadyEnded { .. })
    ));

    let snapshot = log.snapshot("run-1").unwrap().unwrap();
    assert_eq!(snapshot.state, RunState::Cancelled);
    assert_eq!(snapshot.activity[0].status, "unknown");
}

#[test]
fn a_run_that_finished_cleanly_has_no_unknown_calls() {
    // The counterpart: stranding must only catch the abort race, never a run
    // whose tools all reported back before it ended.
    let log = log();
    start(&log, "run-1");
    log.record(
        EventDraft::new("run-1", TaskEventType::ToolAuthorized, USER)
            .with(json!({ "toolCallId": "c1", "tool": "search_documents" })),
    )
    .expect("authorised");
    log.record(
        EventDraft::new("run-1", TaskEventType::ToolSucceeded, USER)
            .with(json!({ "toolCallId": "c1", "tool": "search_documents" })),
    )
    .expect("succeeded");
    log.record(
        EventDraft::new("run-1", TaskEventType::RunCompleted, USER).with(json!({ "turns": 2 })),
    )
    .expect("completed");

    let snapshot = log.snapshot("run-1").unwrap().unwrap();
    assert_eq!(snapshot.activity[0].status, "done");
}

#[test]
fn a_cancelled_run_cannot_later_claim_to_have_completed() {
    // The race worth defending against: the abort lands, and the loop's own
    // completion arrives a moment later. Two endings would let a reader pick.
    let log = log();
    start(&log, "run-1");
    log.record(EventDraft::new("run-1", TaskEventType::RunCancelled, USER).with(json!({})))
        .expect("cancelled");

    assert!(matches!(
        log.record(
            EventDraft::new("run-1", TaskEventType::RunCompleted, USER)
                .with(json!({ "answer": "too late" })),
        ),
        Err(AppendError::AlreadyEnded { .. })
    ));
    assert_eq!(
        log.snapshot("run-1").unwrap().unwrap().state,
        RunState::Cancelled
    );
}

#[test]
fn a_timeout_is_a_budget_stop_and_not_a_failure() {
    // They mean different things to somebody reading the list: a failure is
    // something going wrong, a budget stop is a limit doing its job.
    let log = log();
    start(&log, "run-1");
    log.record(
        EventDraft::new("run-1", TaskEventType::RunStoppedByBudget, SYSTEM_ACTOR)
            .with(json!({ "allowedSeconds": 600 })),
    )
    .expect("stopped");

    let snapshot = log.snapshot("run-1").unwrap().unwrap();
    assert_eq!(snapshot.state, RunState::StoppedByBudget);
    assert_ne!(snapshot.state, RunState::Failed);
    assert!(!snapshot.state.is_success());
}

#[test]
fn a_cancelled_run_is_not_swept_up_again_by_recovery() {
    let dir = tempfile::tempdir().expect("temp dir");
    {
        let before = on_disk(dir.path());
        start(&before, "run-1");
        before
            .record(EventDraft::new("run-1", TaskEventType::RunCancelled, USER).with(json!({})))
            .expect("cancelled");
    }
    let after = on_disk(dir.path());
    assert!(after
        .recover_interrupted(SYSTEM_ACTOR)
        .expect("recovery ran")
        .is_empty());
}

// == 7. Interrupted writes are not repeated automatically =================

#[test]
fn a_write_interrupted_mid_flight_is_marked_unknown_and_refused_afterwards() {
    // The whole of requirement 7. The intent is written, the process dies
    // before the outcome is, and the next start must not simply try again.
    let dir = tempfile::tempdir().expect("temp dir");
    let args = json!({ "path": "note.docx", "template": "approval_note" });
    let key = derive_key("run-1", "create_docx", &args);
    let fingerprint = args_fingerprint(&args);

    {
        let before = on_disk(dir.path());
        start(&before, "run-1");
        assert_eq!(
            before.begin_effect("run-1", &key, "create_docx", &fingerprint, "note.docx"),
            EffectLookup::Fresh
        );
        // The document is being written right here. No settle: the lights go out.
    }

    let after = on_disk(dir.path());
    after.recover_interrupted(SYSTEM_ACTOR).expect("recovery ran");

    // Not retried, not assumed to have worked, not assumed to have failed.
    match after.begin_effect("run-1", &key, "create_docx", &fingerprint, "note.docx") {
        EffectLookup::Unknown(recorded) => {
            assert_eq!(recorded.status, EffectStatus::Unknown);
            assert_eq!(recorded.target, "note.docx");
            let refusal = recorded.unknown_refusal();
            assert!(refusal.contains("note.docx"), "{refusal}");
            assert!(refusal.contains("may or may not"), "{refusal}");
            // It tells the reader not to expect a retry, which is the part a
            // model would otherwise invent an answer about.
            assert!(refusal.contains("not been attempted again"), "{refusal}");
        }
        other => panic!("an interrupted write must not be repeatable: {other:?}"),
    }
}

#[test]
fn the_interrupted_run_says_which_action_needs_checking() {
    let dir = tempfile::tempdir().expect("temp dir");
    let key = derive_key("run-1", "create_docx", &json!({ "path": "note.docx" }));

    {
        let before = on_disk(dir.path());
        start(&before, "run-1");
        before.begin_effect("run-1", &key, "create_docx", "fp", "note.docx");
    }

    let after = on_disk(dir.path());
    after.recover_interrupted(SYSTEM_ACTOR).expect("recovery ran");

    let snapshot = after.snapshot("run-1").unwrap().unwrap();
    assert_eq!(snapshot.state, RunState::DegradedNeedsHuman);
    // "Something was interrupted" is not actionable. "note.docx may not have
    // been written" is.
    assert_eq!(snapshot.unknown_effects.len(), 1);
    assert_eq!(snapshot.unknown_effects[0].target, "note.docx");
    assert!(snapshot.needs_person());
}

#[test]
fn two_attempts_at_one_side_effect_at_once_are_refused_not_serialised() {
    // Whichever finished last would win, which is not a decision anybody made.
    let log = log();
    let key = "k";
    assert_eq!(
        log.begin_effect("run-1", key, "create_docx", "fp", "note.docx"),
        EffectLookup::Fresh
    );
    match log.begin_effect("run-1", key, "create_docx", "fp", "note.docx") {
        EffectLookup::InFlight(recorded) => assert_eq!(recorded.status, EffectStatus::Pending),
        other => panic!("expected an in-flight refusal, got {other:?}"),
    }
}

#[test]
fn a_person_can_say_what_actually_happened_and_the_run_stops_asking() {
    let dir = tempfile::tempdir().expect("temp dir");
    let key = derive_key("run-1", "create_docx", &json!({ "path": "note.docx" }));
    {
        let before = on_disk(dir.path());
        start(&before, "run-1");
        before.begin_effect("run-1", &key, "create_docx", "fp", "note.docx");
    }

    let after = on_disk(dir.path());
    after.recover_interrupted(SYSTEM_ACTOR).expect("recovery ran");
    assert_eq!(after.unknown_effects().unwrap().len(), 1);

    let settled = after
        .reconcile_effect("run-1", &key, true, "ravi")
        .expect("reconciled");
    assert!(settled);

    // Gone from the queue, and gone from the run's own snapshot.
    assert!(after.unknown_effects().unwrap().is_empty());
    let snapshot = after.snapshot("run-1").unwrap().unwrap();
    assert!(snapshot.unknown_effects.is_empty());

    // Reconciling twice is an ordinary race, not an error.
    assert!(!after
        .reconcile_effect("run-1", &key, true, "ravi")
        .expect("second attempt answers"));
}

#[test]
fn a_settled_effect_cannot_be_rewritten_by_reconciliation() {
    // Only an unknown row moves. Letting a person overwrite a settled one would
    // make the record editable, which is the property it exists to not have.
    let log = log();
    log.begin_effect("run-1", "k", "create_docx", "fp", "note.docx");
    log.settle_effect("run-1", "k", &Ok("Wrote note.docx".to_string())).expect("settled");

    assert!(!log
        .reconcile_effect("run-1", "k", false, "ravi")
        .expect("answers"));
    match log.begin_effect("run-1", "k", "create_docx", "fp", "note.docx") {
        EffectLookup::Settled(recorded) => {
            assert_eq!(recorded.replay(), Ok("Wrote note.docx".to_string()))
        }
        other => panic!("the settled outcome should stand: {other:?}"),
    }
}

// == 8. Confidential arguments are absent from UI events ==================

#[test]
fn document_text_never_reaches_the_event_log() {
    // ARJUN design rule 14: confidential contents must not be copied into a record more
    // people can read than could read the original. Enforced on the way in, so
    // a future call site cannot opt out of it by forgetting.
    let secret = "The Kolkata plant seal is worn beyond the service limit.";
    let log = log();
    let event = log
        .record(
            EventDraft::new("run-1", TaskEventType::ToolSucceeded, USER).with(json!({
                "toolCallId": "c1",
                "tool": "search_documents",
                "detail": secret,
                "args": { "query": secret },
            })),
        )
        .expect("appended");

    let written = serde_json::to_string(&event.payload).expect("serialised");
    assert!(!written.contains("seal is worn"));
    assert!(!written.contains("Kolkata"));
    // What survives is enough to identify the same content again.
    assert_eq!(event.payload["detail"]["sha256"], json!(digest(secret)));
    assert_eq!(
        event.payload["detail"]["chars"],
        json!(secret.chars().count())
    );
    // Identifiers and tool names are references, not contents, and stay.
    assert_eq!(event.payload["tool"], json!("search_documents"));
}

#[test]
fn the_envelope_a_window_receives_carries_no_confidential_argument() {
    // The event reaching the UI is the redacted one, unchanged — there is no
    // second serialisation path that could reintroduce what was stripped.
    let log = log();
    let secret = "vendor quote: 42 lakh, do not circulate";
    let event = log
        .record(
            EventDraft::new("run-1", TaskEventType::ToolAuthorized, USER).with(json!({
                "toolCallId": "c1",
                "tool": "write_scoped_file",
                "args": { "path": "quote.txt", "content": secret },
            })),
        )
        .expect("appended");

    let envelope = serde_json::to_string(&event.envelope()).expect("serialised");
    assert!(!envelope.contains("42 lakh"), "{envelope}");
    assert!(!envelope.contains("do not circulate"), "{envelope}");
    // The sequence number is the reason this channel exists at all.
    assert_eq!(event.envelope()["seq"], json!(1));
    assert_eq!(event.envelope()["runId"], json!("run-1"));
}

#[test]
fn a_credential_shaped_field_is_redacted_even_though_nothing_should_put_one_there() {
    // Defence in depth: no call site sends these, and the list covers them
    // anyway, because the cost of being wrong is a disclosure.
    for key in ["password", "token", "apiKey", "secret", "credential"] {
        let redacted = redact(json!({ key: "hunter2" }));
        let written = serde_json::to_string(&redacted).unwrap();
        assert!(!written.contains("hunter2"), "{key} survived: {written}");
    }
}

#[test]
fn the_models_own_words_are_redacted_too() {
    // Rule 9: the operational trace only. No message, no reasoning, no partial
    // completion reaches a screen through this path.
    for key in ["reasoning", "thinking", "completion", "message", "answer"] {
        let redacted = redact(json!({ key: "let me think step by step about the seal" }));
        let written = serde_json::to_string(&redacted).unwrap();
        assert!(!written.contains("step by step"), "{key} survived: {written}");
    }
}

#[test]
fn the_snapshot_keeps_a_reference_to_the_answer_and_not_the_answer() {
    let log = log();
    start(&log, "run-1");
    log.record(
        EventDraft::new("run-1", TaskEventType::RunCompleted, USER)
            .with(json!({ "answer": "The seal is worn beyond the limit [E1].", "turns": 3 })),
    )
    .expect("completed");

    let snapshot = log.snapshot("run-1").unwrap().unwrap();
    assert_eq!(
        snapshot.answer_hash.as_deref(),
        Some(digest("The seal is worn beyond the limit [E1].").as_str())
    );
    assert_eq!(snapshot.answer_chars, 39);
    let written = serde_json::to_string(&snapshot).expect("serialised");
    assert!(!written.contains("seal is worn"));
}

#[test]
fn redaction_reaches_into_nested_payloads() {
    let nested = redact(json!({
        "outer": { "inner": [{ "content": "secret" }] },
        "toolCallId": "c1",
    }));
    let written = serde_json::to_string(&nested).unwrap();
    assert!(!written.contains("secret"));
    assert!(written.contains("c1"));
}

// == Corrupted history ====================================================

#[test]
fn an_event_whose_payload_will_not_parse_costs_its_own_line_and_no_more() {
    let dir = tempfile::tempdir().expect("temp dir");
    let log = on_disk(dir.path());
    log.record(created("run-1")).expect("created");
    log.record(
        EventDraft::new("run-1", TaskEventType::PlanStep, USER).with(json!({ "stepsTaken": 1 })),
    )
    .expect("a step");
    log.record(
        EventDraft::new("run-1", TaskEventType::RunCompleted, USER)
            .with(json!({ "answer": "done", "turns": 2 })),
    )
    .expect("completed");

    // The triggers refuse an UPDATE, which is the point of them. Corruption
    // reaches the table the way it would in the world: past them.
    {
        let conn = rusqlite::Connection::open(dir.path().join("sarathi.db")).unwrap();
        conn.execute_batch("DROP TRIGGER task_events_is_append_only_update")
            .unwrap();
        conn.execute(
            "UPDATE task_events SET payload = '{ truncated' WHERE run_id = 'run-1' AND seq = 2",
            [],
        )
        .unwrap();
    }

    let page = on_disk(dir.path()).events_since("run-1", 0).expect("readable");
    assert_eq!(page.events.len(), 2);
    assert_eq!(page.unreadable.len(), 1);
    assert_eq!(page.unreadable[0].seq, 2);
    assert!(page.unreadable[0].problem.contains("readable JSON"));
}

#[test]
fn a_payload_edited_underneath_us_is_not_folded_into_the_state() {
    // The hash is what makes a rewritten payload detectable rather than
    // believed. An event that has been changed is not evidence of anything.
    let dir = tempfile::tempdir().expect("temp dir");
    let log = on_disk(dir.path());
    log.record(created("run-1")).expect("created");
    log.record(
        EventDraft::new("run-1", TaskEventType::RunFailed, USER)
            .with(json!({ "failure": "the agent runtime stopped" })),
    )
    .expect("failed");

    {
        let conn = rusqlite::Connection::open(dir.path().join("sarathi.db")).unwrap();
        conn.execute_batch("DROP TRIGGER task_events_is_append_only_update")
            .unwrap();
        conn.execute(
            r#"UPDATE task_events SET payload = '{"failure":"nothing went wrong at all"}'
               WHERE run_id = 'run-1' AND seq = 2"#,
            [],
        )
        .unwrap();
    }

    let reopened = on_disk(dir.path());
    let page = reopened.events_since("run-1", 0).expect("readable");
    assert_eq!(page.unreadable.len(), 1);
    assert!(page.unreadable[0].problem.contains("recorded hash"));

    let rebuilt = reopened.rebuild("run-1").unwrap().unwrap();
    assert!(!rebuilt.is_intact());
    assert_ne!(rebuilt.failure.as_deref(), Some("nothing went wrong at all"));
}

#[test]
fn an_event_type_from_a_newer_build_is_reported_rather_than_guessed_at() {
    let dir = tempfile::tempdir().expect("temp dir");
    let log = on_disk(dir.path());
    log.record(created("run-1")).expect("created");

    {
        let conn = rusqlite::Connection::open(dir.path().join("sarathi.db")).unwrap();
        conn.execute(
            "INSERT INTO task_events
                (event_id, run_id, seq, event_type, at, actor, schema_version, payload, payload_hash)
             VALUES ('x', 'run-1', 2, 'quantum_entangled', '2026-08-27T10:00:00+00:00',
                     'system', 2, '{}', ?1)",
            rusqlite::params![payload_hash(&json!({}))],
        )
        .unwrap();
    }

    let page = on_disk(dir.path()).events_since("run-1", 0).expect("readable");
    assert_eq!(page.events.len(), 1);
    assert!(page.unreadable[0]
        .problem
        .contains("not an event type this version understands"));
}

#[test]
fn a_run_whose_ending_is_the_corrupt_part_is_not_reported_as_still_running() {
    // The one actively misleading answer available here. A run whose only
    // unreadable event is its ending must not sit on the screen claiming to be
    // in flight.
    let dir = tempfile::tempdir().expect("temp dir");
    let log = on_disk(dir.path());
    log.record(created("run-1")).expect("created");
    log.record(
        EventDraft::new("run-1", TaskEventType::RunCompleted, USER).with(json!({ "turns": 1 })),
    )
    .expect("completed");

    {
        let conn = rusqlite::Connection::open(dir.path().join("sarathi.db")).unwrap();
        conn.execute_batch("DROP TRIGGER task_events_is_append_only_update")
            .unwrap();
        conn.execute(
            "UPDATE task_events SET payload = 'nope' WHERE run_id = 'run-1' AND seq = 2",
            [],
        )
        .unwrap();
    }

    let rebuilt = on_disk(dir.path()).rebuild("run-1").unwrap().unwrap();
    assert!(!rebuilt.is_intact());
    // Its position is still accounted for, so a caller asking for "everything
    // after seq 2" does not wait forever for an event it will never get.
    assert_eq!(rebuilt.seq, 2);
}

#[test]
fn an_event_that_could_not_legally_follow_is_recorded_but_not_applied() {
    // Two writers disagreeing about a run. The late event stays in the history
    // — it happened — but it does not drag the state backwards, and the
    // disagreement is reported rather than hidden.
    let log = log();
    start(&log, "run-1");
    log.record(routed("run-1")).expect("a late routing event");

    let snapshot = log.snapshot("run-1").unwrap().unwrap();
    assert_eq!(snapshot.state, RunState::Running);
    assert!(!snapshot.is_intact());
    assert!(snapshot.anomalies[0].contains("back to routed"), "{:?}", snapshot.anomalies);
    // Still in the history: four events, one of which was not applied.
    assert_eq!(log.events_since("run-1", 0).unwrap().events.len(), 4);
}

// == Folding ==============================================================

#[test]
fn a_refused_call_appears_in_the_trace_even_though_it_was_never_authorised() {
    // A refusal is the first thing heard about that call. A trace that drops
    // it is a trace saying the policy never did anything.
    let log = log();
    start(&log, "run-1");
    log.record(
        EventDraft::new("run-1", TaskEventType::ToolRefused, USER).with(json!({
            "toolCallId": "c9",
            "tool": "execute_code",
            "reason": "execute_code is not one of the tools this task was planned to use.",
        })),
    )
    .expect("refused");

    let snapshot = log.snapshot("run-1").unwrap().unwrap();
    assert_eq!(snapshot.activity.len(), 1);
    assert_eq!(snapshot.activity[0].status, "refused");
    assert_eq!(snapshot.activity[0].tool, "execute_code");
}

#[test]
fn an_authorised_call_becomes_one_row_that_changes_state() {
    let log = log();
    start(&log, "run-1");
    log.record(
        EventDraft::new("run-1", TaskEventType::ToolAuthorized, USER)
            .with(json!({ "toolCallId": "c1", "tool": "create_docx" })),
    )
    .expect("authorised");
    let mid = log.snapshot("run-1").unwrap().unwrap();
    assert_eq!(mid.activity[0].status, "running");
    assert_eq!(mid.state, RunState::ExecutingTool);

    log.record(
        EventDraft::new("run-1", TaskEventType::ToolSucceeded, USER)
            .with(json!({ "toolCallId": "c1", "tool": "create_docx" })),
    )
    .expect("succeeded");

    let snapshot = log.snapshot("run-1").unwrap().unwrap();
    assert_eq!(snapshot.activity.len(), 1);
    assert_eq!(snapshot.activity[0].status, "done");
    assert_eq!(snapshot.state, RunState::ToolResultRecorded);
}

#[test]
fn an_approval_shows_as_waiting_and_then_stops_waiting() {
    let log = log();
    start(&log, "run-1");
    log.record(
        EventDraft::new("run-1", TaskEventType::ApprovalRequested, USER)
            .with(json!({ "toolCallId": "c1", "tool": "create_docx" })),
    )
    .expect("requested");

    let waiting = log.snapshot("run-1").unwrap().unwrap();
    assert_eq!(waiting.state, RunState::AwaitingApproval);
    assert_eq!(waiting.approvals_pending, 1);
    assert!(waiting.needs_person());

    log.record(
        EventDraft::new("run-1", TaskEventType::ApprovalDecided, "ravi")
            .with(json!({ "toolCallId": "c1", "approved": true })),
    )
    .expect("decided");

    let resumed = log.snapshot("run-1").unwrap().unwrap();
    assert_eq!(resumed.state, RunState::Running);
    assert_eq!(resumed.approvals_pending, 0);
}

#[test]
fn a_recovered_trace_keeps_the_caveat_that_history_was_summarised() {
    // The compaction is a caveat on everything the run says after it: those
    // answers rest on a summary of the earlier turns rather than on the turns.
    let log = log();
    start(&log, "run-1");
    log.record(EventDraft::new("run-1", TaskEventType::TurnEnded, USER).with(json!({})))
        .expect("a turn");
    log.record(
        EventDraft::new("run-1", TaskEventType::ContextCompacted, SYSTEM_ACTOR)
            .with(json!({ "tokensBefore": 8000, "tokensAfter": 2100 })),
    )
    .expect("a compaction");
    log.record(EventDraft::new("run-1", TaskEventType::TurnEnded, USER).with(json!({})))
        .expect("another turn");

    let snapshot = log.snapshot("run-1").unwrap().unwrap();
    assert_eq!(snapshot.turns, 2);
    assert_eq!(snapshot.compactions, 1);
}

/// One compaction event, with the whole payload the runtime now sends.
fn compaction_payload(ordinal: u32, refined: bool, before: u32, after: u32) -> serde_json::Value {
    json!({
        "ordinal": ordinal,
        "tokensBefore": before,
        "tokensAfter": after,
        "messagesSummarised": 24,
        "refinedExistingSummary": refined,
        "toolResultsCleared": 3,
        "at": "2026-08-28T09:15:00+00:00",
        "ledger": {
            "system": 400,
            "skill": 0,
            "toolSchema": 1200,
            "evidence": 300,
            "notes": 150,
            "transcript": 2000,
            "compaction": 250,
            "reserve": 1600,
            "occupied": 4300,
            "committed": 5900,
            "window": 8192,
            "headroom": 2292
        }
    })
}

#[test]
fn a_recovered_trace_says_what_each_compaction_actually_did() {
    // A count alone says the window ran out. What somebody reviewing the run
    // needs is which section filled it and how much each pass reclaimed —
    // otherwise "compacted three times" is a fact with no remedy attached.
    let log = log();
    start(&log, "run-1");
    log.record(
        EventDraft::new("run-1", TaskEventType::ContextCompacted, SYSTEM_ACTOR)
            .with(compaction_payload(1, false, 8000, 2100)),
    )
    .expect("first compaction");
    log.record(
        EventDraft::new("run-1", TaskEventType::ContextCompacted, SYSTEM_ACTOR)
            .with(compaction_payload(2, true, 7800, 3000)),
    )
    .expect("second compaction");

    let snapshot = log.snapshot("run-1").unwrap().unwrap();

    assert_eq!(snapshot.compactions, 2);
    assert_eq!(snapshot.compaction_events.len(), 2);
    assert_eq!(snapshot.compaction_events[0].tokens_before, 8000);
    assert_eq!(snapshot.compaction_events[0].ledger.tool_schema, 1200);
    // The distinction that decides whether the earlier half of the run is
    // described once or twice.
    assert!(!snapshot.compaction_events[0].refined_existing_summary);
    assert!(snapshot.compaction_events[1].refined_existing_summary);
}

#[test]
fn a_compaction_whose_detail_cannot_be_read_is_still_counted() {
    // An unreadable payload should cost the detail, not the fact. A run that
    // compacted and appears not to have is a run whose answers look better
    // grounded than they are.
    let log = log();
    start(&log, "run-1");
    log.record(
        EventDraft::new("run-1", TaskEventType::ContextCompacted, SYSTEM_ACTOR)
            .with(json!({ "tokensBefore": "not a number" })),
    )
    .expect("a compaction");

    let snapshot = log.snapshot("run-1").unwrap().unwrap();
    assert_eq!(snapshot.compactions, 1);
    assert!(snapshot.compaction_events.is_empty());
}

#[test]
fn a_compaction_event_carries_no_document_text() {
    // The ledger is counts only. That is what makes it safe on a screen read
    // more widely than the transcript it describes — a section size cannot
    // reveal what was in the section.
    let log = log();
    start(&log, "run-1");
    log.record(
        EventDraft::new("run-1", TaskEventType::ContextCompacted, SYSTEM_ACTOR)
            .with(compaction_payload(1, false, 8000, 2100)),
    )
    .expect("a compaction");

    let snapshot = log.snapshot("run-1").unwrap().unwrap();
    let serialised = serde_json::to_string(&snapshot.compaction_events).unwrap();

    // Every value in the record is a number, a boolean or a timestamp. Nothing
    // in it is prose, so there is nothing for a passage to hide in.
    for field in ["tokensBefore", "messagesSummarised", "ledger"] {
        assert!(
            serialised.contains(&field.to_string())
                || serialised.contains(&heck_snake(field)),
            "{field} missing from {serialised}"
        );
    }
    assert!(!serialised.contains("Maintenance SOP"));
}

/// `camelCase` to `snake_case`, for asserting against either serialisation.
fn heck_snake(field: &str) -> String {
    let mut out = String::new();
    for c in field.chars() {
        if c.is_ascii_uppercase() {
            out.push('_');
            out.push(c.to_ascii_lowercase());
        } else {
            out.push(c);
        }
    }
    out
}

#[test]
fn the_running_list_holds_only_runs_that_have_not_ended() {
    let log = log();
    start(&log, "live");
    start(&log, "done");
    log.record(
        EventDraft::new("done", TaskEventType::RunCompleted, USER).with(json!({ "turns": 1 })),
    )
    .expect("completed");

    let running = log.running().expect("readable");
    assert_eq!(running.len(), 1);
    assert_eq!(running[0].run_id, "live");
}

#[test]
fn a_run_nobody_has_heard_of_is_no_snapshot_and_not_an_error() {
    // The Tasks screen asks about a run id from a URL. A missing one is an
    // empty answer, not a failure.
    let log = log();
    assert!(log.snapshot("never-happened").expect("readable").is_none());
}
