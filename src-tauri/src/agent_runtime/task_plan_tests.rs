//! The plan's rules, without a runtime: patches against versions, receipts the
//! model cannot write, settlement decided by the backend, and the gate.

use serde_json::json;

use super::*;

const AT: &str = "2026-09-24T10:00:00Z";

/// A receipt source that knows one successful `artifact.create_approval_note`
/// call at sequence 7.
struct Ledger;

impl ReceiptSource for Ledger {
    fn tool_succeeded(&self, run_id: &str, event_seq: i64) -> Option<ToolReceipt> {
        (run_id == "run-1" && event_seq == 7).then(|| ToolReceipt {
            tool: "artifact.create_approval_note".into(),
            output_sha256: Some("ab".repeat(32)),
        })
    }
}

fn create() -> TaskPlan {
    let patch = parse_patch(&json!({
        "base_version": 0,
        "operations": [
            {"op": "set_goal", "text": "An approval note on the seal failure"},
            {"op": "add_constraint", "text": "Every pressure in bar"},
            {"op": "add_step", "id": "find", "title": "Find the seal records", "kind": "delegate",
             "role": "knowledge-retriever", "acceptance": ["childCompleted", "memoryPublished"]},
            {"op": "add_step", "id": "check", "title": "Re-derive the figures", "kind": "delegate",
             "role": "calculation-checker", "depends_on": ["find"], "inputs": ["step:find"]},
            {"op": "add_step", "id": "write", "title": "Write the note", "kind": "direct",
             "depends_on": ["check"], "acceptance": ["toolSucceeded:artifact.create_approval_note"]},
        ]
    }))
    .expect("the patch parses");
    apply_patch("run-1", None, &patch, &Ledger, AT).expect("the plan is created")
}

fn settlement(job: &str, status: &str, findings: usize, evidenced: usize) -> JobSettlement {
    JobSettlement {
        job_id: job.into(),
        status: status.into(),
        findings,
        evidenced,
        receipts: 0,
        published: if findings > 0 {
            vec![StepRef::Memory { item_id: "mi-1".into(), revision: 1 }]
        } else {
            Vec::new()
        },
        artifacts: Vec::new(),
        artifacts_accepted: None,
        result_hash: format!("hash-{job}"),
        event_seq: Some(11),
        summary: format!("{status} with {findings} finding(s)"),
        missing: Vec::new(),
        superseded: false,
    }
}

#[test]
fn a_plan_is_created_at_version_one_with_its_goal_and_ordered_steps() {
    let plan = create();
    assert_eq!(plan.version, 1);
    assert_eq!(plan.author, Author::Model);
    assert_eq!(plan.steps.len(), 3);
    assert_eq!(plan.steps[0].repair.max_attempts, DEFAULT_ATTEMPTS);
    // Only the step with no prerequisites is ready.
    let ready: Vec<&str> = plan.ready().iter().map(|step| step.id.as_str()).collect();
    assert_eq!(ready, vec!["find"]);
    assert!(project(&plan).contains("Every pressure in bar"));
}

#[test]
fn a_patch_against_an_old_version_is_refused_and_changes_nothing() {
    let plan = create();
    let stale = parse_patch(&json!({
        "base_version": 0,
        "operations": [{"op": "add_constraint", "text": "late"}]
    }))
    .unwrap();
    assert_eq!(
        apply_patch("run-1", Some(&plan), &stale, &Ledger, AT),
        Err(Refusal::Stale { base: 0, current: 1 })
    );
}

#[test]
fn a_cycle_and_a_dangling_dependency_are_refused() {
    let plan = create();
    let cycle = parse_patch(&json!({
        "base_version": 1,
        "operations": [{"op": "update_step", "id": "find", "depends_on": ["write"]}]
    }))
    .unwrap();
    let refused = apply_patch("run-1", Some(&plan), &cycle, &Ledger, AT).unwrap_err();
    assert!(refused.explain().contains("cycle"), "{}", refused.explain());

    let dangling = parse_patch(&json!({
        "base_version": 1,
        "operations": [{"op": "add_step", "id": "x", "title": "x", "kind": "delegate",
                        "role": "knowledge-retriever", "depends_on": ["nope"]}]
    }))
    .unwrap();
    assert!(apply_patch("run-1", Some(&plan), &dangling, &Ledger, AT).is_err());
}

#[test]
fn a_patch_with_an_unknown_field_or_op_is_refused_rather_than_half_applied() {
    assert!(parse_patch(&json!({
        "base_version": 1,
        "operations": [{"op": "add_step", "id": "x", "title": "x", "kind": "delegate",
                        "role": "r", "status": "completed"}]
    }))
    .unwrap_err()
    .contains("status"));
    assert!(parse_patch(&json!({
        "base_version": 1,
        "operations": [{"op": "complete_step", "id": "find"}]
    }))
    .is_err());
}

#[test]
fn the_model_cannot_mark_a_step_done_or_erase_a_completed_receipt() {
    let plan = create();
    let started = job_started(&plan, "find", "job-1", AT).unwrap();
    let done = job_settled(&started, "find", &settlement("job-1", "completed", 2, 2), AT).unwrap();
    assert_eq!(done.step("find").unwrap().status, StepStatus::Completed);
    assert_eq!(done.step("find").unwrap().receipts.len(), 1);

    // Skipping, editing and reopening a completed step are all refused.
    for operation in [
        json!({"op": "skip_step", "id": "find", "reason": "not needed"}),
        json!({"op": "update_step", "id": "find", "title": "renamed"}),
        json!({"op": "reopen_step", "id": "find", "reason": "again"}),
    ] {
        let patch = parse_patch(&json!({"base_version": done.version, "operations": [operation]})).unwrap();
        assert!(
            apply_patch("run-1", Some(&done), &patch, &Ledger, AT).is_err(),
            "{operation} was allowed on a completed step"
        );
    }
    // And no operation exists that writes a status or a receipt.
    assert!(parse_patch(&json!({
        "base_version": done.version,
        "operations": [{"op": "update_step", "id": "check", "receipts": []}]
    }))
    .is_err());
}

#[test]
fn a_direct_step_settles_only_on_a_receipt_the_event_log_holds() {
    let plan = create();
    let forged = parse_patch(&json!({
        "base_version": 1,
        "operations": [{"op": "claim_receipt", "id": "write", "event_seq": 8}]
    }))
    .unwrap();
    assert!(apply_patch("run-1", Some(&plan), &forged, &Ledger, AT).is_err());

    let real = parse_patch(&json!({
        "base_version": 1,
        "operations": [{"op": "claim_receipt", "id": "write", "event_seq": 7}]
    }))
    .unwrap();
    let claimed = apply_patch("run-1", Some(&plan), &real, &Ledger, AT).unwrap();
    let step = claimed.step("write").unwrap();
    assert_eq!(step.status, StepStatus::Completed);
    assert_eq!(step.receipts[0].event_seq, Some(7));

    // The same receipt cannot be claimed for a delegate step.
    let wrong = parse_patch(&json!({
        "base_version": 1,
        "operations": [{"op": "claim_receipt", "id": "find", "event_seq": 7}]
    }))
    .unwrap();
    assert!(apply_patch("run-1", Some(&plan), &wrong, &Ledger, AT).is_err());
}

#[test]
fn a_completed_status_with_nothing_cited_is_partial_not_done() {
    let plan = create();
    let started = job_started(&plan, "find", "job-1", AT).unwrap();
    let settled = job_settled(&started, "find", &settlement("job-1", "completed", 3, 0), AT).unwrap();
    let step = settled.step("find").unwrap();
    assert_eq!(step.status, StepStatus::Partial);
    assert!(step.note.as_deref().unwrap().contains("no cited finding"));
    // A dependent step does not become ready on a partial result.
    assert!(settled.ready().iter().all(|ready| ready.id != "check"));
}

#[test]
fn the_same_failure_twice_blocks_the_step_even_with_attempts_left() {
    let mut plan = create();
    let patch = parse_patch(&json!({
        "base_version": 1,
        "operations": [{"op": "update_step", "id": "find", "max_attempts": 3}]
    }))
    .unwrap();
    plan = apply_patch("run-1", Some(&plan), &patch, &Ledger, AT).unwrap();

    plan = job_started(&plan, "find", "job-1", AT).unwrap();
    plan = job_settled(&plan, "find", &settlement("job-1", "failed", 0, 0), AT).unwrap();
    assert_eq!(plan.step("find").unwrap().status, StepStatus::Failed);

    let reopen = parse_patch(&json!({
        "base_version": plan.version,
        "operations": [{"op": "reopen_step", "id": "find", "reason": "try again"}]
    }))
    .unwrap();
    plan = apply_patch("run-1", Some(&plan), &reopen, &Ledger, AT).unwrap();
    plan = job_started(&plan, "find", "job-2", AT).unwrap();
    plan = job_settled(&plan, "find", &settlement("job-2", "failed", 0, 0), AT).unwrap();
    let step = plan.step("find").unwrap();
    assert_eq!(step.status, StepStatus::Blocked);
    assert_eq!(step.repair.attempts, 2);
    assert!(step.note.as_deref().unwrap().contains("no progress"));

    // And the model cannot reopen it for a third identical try.
    let again = parse_patch(&json!({
        "base_version": plan.version,
        "operations": [{"op": "reopen_step", "id": "find", "reason": "once more"}]
    }))
    .unwrap();
    assert!(apply_patch("run-1", Some(&plan), &again, &Ledger, AT).is_err());
}

#[test]
fn attempts_are_bounded_even_when_each_failure_differs() {
    let mut plan = create();
    for (attempt, detail) in ["timed_out", "failed"].iter().enumerate() {
        if attempt > 0 {
            let reopen = parse_patch(&json!({
                "base_version": plan.version,
                "operations": [{"op": "reopen_step", "id": "find", "reason": "retry"}]
            }))
            .unwrap();
            plan = apply_patch("run-1", Some(&plan), &reopen, &Ledger, AT).unwrap();
        }
        let job = format!("job-{attempt}");
        plan = job_started(&plan, "find", &job, AT).unwrap();
        plan = job_settled(&plan, "find", &settlement(&job, detail, 0, 0), AT).unwrap();
    }
    let step = plan.step("find").unwrap();
    assert_eq!(step.status, StepStatus::Blocked);
    assert!(step.note.as_deref().unwrap().contains("attempts used"), "{:?}", step.note);
}

#[test]
fn a_correction_reopens_a_completed_step_and_keeps_its_receipts() {
    let mut plan = create();
    plan = job_started(&plan, "find", "job-1", AT).unwrap();
    plan = job_settled(&plan, "find", &settlement("job-1", "completed", 1, 1), AT).unwrap();
    assert_eq!(plan.step("find").unwrap().status, StepStatus::Completed);

    let (corrected, running) = correction_recorded(&plan, "Use the 2019 revision", "alice", AT);
    assert!(running.is_empty());
    assert_eq!(corrected.corrections.len(), 1);
    assert!(project(&corrected).contains("Use the 2019 revision"));

    let reopen = parse_patch(&json!({
        "base_version": corrected.version,
        "operations": [{"op": "reopen_step", "id": "find", "reason": "the correction changes the source"}]
    }))
    .unwrap();
    let reopened = apply_patch("run-1", Some(&corrected), &reopen, &Ledger, AT).unwrap();
    let step = reopened.step("find").unwrap();
    assert_eq!(step.status, StepStatus::Pending);
    assert_eq!(step.receipts.len(), 1, "the completed receipt is kept");
}

#[test]
fn a_job_superseded_by_a_correction_is_not_charged() {
    let mut plan = create();
    plan = job_started(&plan, "find", "job-1", AT).unwrap();
    let (corrected, running) = correction_recorded(&plan, "Only the east train", "alice", AT);
    assert_eq!(running, vec!["job-1".to_string()]);
    let mut stopped = settlement("job-1", "cancelled", 0, 0);
    stopped.superseded = true;
    let settled = job_settled(&corrected, "find", &stopped, AT).unwrap();
    let step = settled.step("find").unwrap();
    assert_eq!(step.status, StepStatus::Pending);
    assert_eq!(step.repair.attempts, 0);
}

#[test]
fn an_interrupted_writer_needs_a_person_and_an_interrupted_reader_can_be_retried() {
    let plan = job_started(&create(), "find", "job-1", AT).unwrap();
    let reader = job_interrupted(&plan, "find", "job-1", false, AT).unwrap();
    assert_eq!(reader.step("find").unwrap().status, StepStatus::Failed);
    let writer = job_interrupted(&plan, "find", "job-1", true, AT).unwrap();
    assert_eq!(writer.step("find").unwrap().status, StepStatus::Blocked);
}

#[test]
fn the_gate_names_every_unfinished_step_and_the_missing_review() {
    let mut plan = create();
    let add_review = parse_patch(&json!({
        "base_version": 1,
        "operations": [{"op": "add_step", "id": "review", "title": "Independent review",
                        "kind": "review", "depends_on": ["write"]}]
    }))
    .unwrap();
    plan = apply_patch("run-1", Some(&plan), &add_review, &Ledger, AT).unwrap();
    let gate = gate(&plan);
    assert_eq!(gate.completed, 0);
    assert_eq!(gate.unfinished.len(), 4);
    assert!(gate.needs_review);
    assert!(!gate.reviewed);

    let reviewed = review_settled(&plan, "review", "rev-1", true, "passed", None, AT).unwrap();
    assert!(super::gate(&reviewed).reviewed);
}

#[test]
fn a_stored_plan_round_trips_and_its_digest_is_stable() {
    let plan = create();
    let text = serde_json::to_string(&plan).unwrap();
    let back: TaskPlan = serde_json::from_str(&text).unwrap();
    assert_eq!(back, plan);
    assert_eq!(back.digest(), plan.digest());
}

#[test]
fn a_step_already_running_does_not_start_a_second_job() {
    let started = job_started(&create(), "find", "job-1", AT).unwrap();
    assert!(job_started(&started, "find", "job-2", AT).is_none());
    assert_eq!(started.step("find").unwrap().job_id.as_deref(), Some("job-1"));
}
