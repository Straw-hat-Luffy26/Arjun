//! The production execution/completion path, independent of the desktop shell.
//! Routing, authenticated restoration and conversation UI remain in the command.
//! Both fresh and resumed tasks run through this driver; a runtime's `completed`
//! response is only a proposal until receipts, artifacts and the answer pass.

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use chrono::Utc;
use serde_json::{json, Value};

use super::{
    artifacts, completion, events, outcome::RunOutcome, planning, retrieval, tasks, AgentRuntime,
};
use crate::artifacts::{verify, Evidence, Grounding, VerificationReport};
use crate::commands::agent::{RunCalculations, RunPlans, RunToolCalls};
use crate::knowledge::SearchResult;
use crate::orchestrator::calculation::CalculationRecord;

pub struct TaskDriver<'a> {
    pub run_id: &'a str,
    pub prompt: &'a str,
    pub actor: &'a str,
    pub lease: &'a events::Lease,
    pub lease_lost: &'a AtomicBool,
    pub events: &'a events::TaskEventLog,
    pub health: &'a super::audit_health::AuditHealth,
    pub plans: &'a RunPlans,
    pub passages: &'a retrieval::RunPassages,
    pub calculations: &'a RunCalculations,
    pub produced: &'a artifacts::RunArtifacts,
    pub calls: &'a RunToolCalls,
}

pub struct DrivenTask {
    pub response: Result<Value, String>,
    pub outcome: RunOutcome,
    pub answer: String,
    pub turns: u32,
    pub plan: tasks::PlanRecord,
    pub verification: Option<VerificationReport>,
    pub completion: completion::CompletionVerification,
    pub artifacts: Vec<artifacts::ArtifactReport>,
    pub passages: Vec<SearchResult>,
    pub calculations: Vec<CalculationRecord>,
    pub calls: Vec<tasks::ToolCallRecord>,
    pub finished_at: chrono::DateTime<Utc>,
    pub record_failure: Option<String>,
}

impl TaskDriver<'_> {
    pub async fn run(
        &self,
        runtime: &AgentRuntime,
        params: Value,
        allowed: Duration,
        on_verifying: impl FnOnce(usize),
        emit: impl Fn(Value),
    ) -> DrivenTask {
        let (response, mut outcome) =
            match tokio::time::timeout(allowed, runtime.request("run.start", params)).await {
                Ok(Ok(value)) => {
                    let outcome =
                        RunOutcome::from_runtime(&value).unwrap_or_else(|| RunOutcome::Failed {
                            detail: "The runtime finished without saying how the run ended.".into(),
                        });
                    (Ok(value), outcome)
                }
                Ok(Err(error)) => {
                    let detail = error.to_string();
                    (Err(detail.clone()), RunOutcome::from_rpc_error(detail))
                }
                Err(_) => {
                    let _ = runtime
                        .request("run.abort", json!({"runId": self.run_id}))
                        .await;
                    let detail = format!(
                        "Stopped: it ran past the {} minutes this task was allowed.",
                        allowed.as_secs() / 60
                    );
                    (Err(detail.clone()), RunOutcome::BudgetStopped { detail })
                }
            };
        if self.lease_lost.load(Ordering::Acquire) {
            outcome = RunOutcome::NeedsReview {
                detail: "The execution lease was lost; this attempt cannot declare completion."
                    .into(),
            };
        }
        let answer = response
            .as_ref()
            .ok()
            .and_then(|v| v["text"].as_str())
            .unwrap_or_default()
            .to_string();
        let turns = response
            .as_ref()
            .ok()
            .and_then(|v| v["turns"].as_u64())
            .unwrap_or(0) as u32;
        let has_answer = !answer.trim().is_empty();
        let derived = planning::derive(self.prompt);
        let mut evidence_errors = Vec::new();
        match self.events.children_for_run(self.run_id) {
            Ok(children) if children.iter().any(|c| !c.result.is_complete()) => evidence_errors
                .push(
                    "A delegated specialist did not complete; its result requires review.".into(),
                ),
            Err(error) => evidence_errors.push(error),
            _ => {}
        }
        // An unreadable table is uncertainty, never an empty successful ledger.
        let mut plan = match self
            .plans
            .lock()
            .ok()
            .and_then(|t| t.get(self.run_id).map(tasks::PlanRecord::of))
        {
            Some(plan) => plan,
            None => {
                evidence_errors.push("The task plan could not be read.".to_string());
                tasks::PlanRecord::of(&planning::plan_for(self.run_id, self.prompt))
            }
        };
        let passages = match self.passages.lock() {
            Ok(table) => table.get(self.run_id).cloned().unwrap_or_default(),
            Err(_) => {
                evidence_errors.push("The evidence table is unavailable.".into());
                Vec::new()
            }
        };
        let calculations = match self.calculations.lock() {
            Ok(table) => table.get(self.run_id).cloned().unwrap_or_default(),
            Err(_) => {
                evidence_errors.push("The calculation table is unavailable.".into());
                Vec::new()
            }
        };
        let calls = match self.calls.lock() {
            Ok(table) => table.get(self.run_id).cloned().unwrap_or_default(),
            Err(_) => {
                evidence_errors.push("The tool history is unavailable.".into());
                Vec::new()
            }
        };
        if calls.iter().any(|call| {
            call.tool == crate::orchestrator::tools::ToolName::AgentDelegateReadonly.as_str()
                && call.outcome != tasks::CallOutcome::Succeeded
        }) {
            evidence_errors.push(
                "A requested specialist handoff failed or was refused; completion requires review."
                    .into(),
            );
        }
        let artifacts = match self.produced.lock() {
            Ok(table) => table
                .get(self.run_id)
                .into_iter()
                .flatten()
                .map(artifacts::check)
                .collect::<Vec<_>>(),
            Err(_) => {
                evidence_errors.push("The artifact table is unavailable.".into());
                Vec::new()
            }
        };
        let grounding = if derived
            .budget
            .permitted_tools
            .iter()
            .any(|t| t.is_retrieval())
        {
            Grounding::OrganisationRecord
        } else {
            Grounding::GeneralKnowledge
        };
        if has_answer && !matches!(outcome, RunOutcome::Paused { .. }) {
            on_verifying(answer.chars().count());
        }
        let verification = has_answer.then(|| {
            verify(
                &answer,
                &Evidence {
                    grounding,
                    passages: &passages,
                    calculations: &calculations,
                    unread_pages: &[],
                },
            )
        });
        if plan.steps.len() != derived.steps.len()
            || plan
                .steps
                .iter()
                .zip(&derived.steps)
                .any(|(a, b)| a.intent != b.intent)
        {
            evidence_errors.push("The recorded plan differs from the task's fixed plan.".into());
        } else {
            let succeeded = calls
                .iter()
                .filter(|c| c.outcome == tasks::CallOutcome::Succeeded)
                .map(|c| c.tool.clone())
                .collect::<Vec<_>>();
            plan.settle(
                &derived.steps,
                &succeeded,
                has_answer,
                verification.is_some(),
            );
        }
        let (unknown_effects, pending_approvals) =
            match self.events.completion_obligations(self.run_id) {
                Ok(obligations) => obligations,
                Err(error) => {
                    evidence_errors.push(error);
                    (Vec::new(), 0)
                }
            };
        let finished_at = Utc::now();
        let completion = completion::verify(
            &completion::CompletionInputs {
                evidence_error: (!evidence_errors.is_empty()).then(|| evidence_errors.join(" ")),
                failure: outcome.detail().map(str::to_string),
                unfinished_steps: plan.unfinished().len(),
                unknown_effects,
                pending_approvals,
                artifacts: artifacts
                    .iter()
                    .map(|a| (a.name.clone(), a.sound))
                    .collect(),
                grounding_ready: verification.as_ref().map(VerificationReport::is_ready),
                has_answer,
            },
            finished_at,
        );
        outcome = completion.enforce_outcome(outcome);
        let record_failure = if matches!(outcome, RunOutcome::Paused { .. }) { None } else { match self.events.record_fenced(
            events::EventDraft::new(self.run_id, events::TaskEventType::CompletionVerified, self.actor).with(json!({
                "passed": completion.passed(), "outcome": completion.outcome.as_str(),
                "verifierVersion": completion.verifier_version, "verifiedAt": completion.verified_at,
                "criteria": completion.criteria.iter().map(|c| json!({
                    "criterionId": c.criterion_id, "status": c.status.as_str(), "evidence": c.evidence,
                })).collect::<Vec<_>>(),
            })), self.lease,
        ) {
            Ok(event) => { emit(event.envelope()); None }
            Err(error) => {
                let detail = format!("The completion-verification record could not be saved: {error}");
                self.health.writes_failed(detail.clone());
                if outcome.is_success() { outcome = RunOutcome::NeedsReview { detail: detail.clone() }; }
                Some(detail)
            }
        } };
        plan.ended(outcome.detail());
        DrivenTask {
            response,
            outcome,
            answer,
            turns,
            plan,
            verification,
            completion,
            artifacts,
            passages,
            calculations,
            calls,
            finished_at,
            record_failure,
        }
    }
}

/// Publish the task file and the authoritative fenced ending together. The shell
/// and headless recovery tests use the same ordering and duplicate handling.
pub fn publish(
    dir: &std::path::Path,
    record: &tasks::TaskRecord,
    events: &events::TaskEventLog,
    lease: &events::Lease,
    ending_payload: Value,
    emit: impl Fn(Value),
) -> Result<RunOutcome, String> {
    let mut record = record.clone();
    record.children = events.children_for_run(&record.run_id)?;
    let outcome = record
        .outcome
        .as_ref()
        .ok_or("The task has no typed outcome.")?;
    if matches!(outcome, RunOutcome::Paused { .. }) {
        let saved = events.load_context(&record.run_id)?.ok_or("A pause requires a durable context boundary.")?;
        if saved.view.phase != events::context::ContextPhase::ModelReady || !events.effect_obligations(&record.run_id)?.0.is_empty() {
            return Err("The task has no settled model boundary for pausing.".into());
        }
        record.completion_verification = None;
        tasks::save(dir, &record)?;
        let event = events.record_fenced(events::EventDraft::idempotent(
            &record.run_id, events::TaskEventType::RunPaused, &record.user_id,
            &format!("pause:{}", lease.fence_token),
        ).with(ending_payload), lease).map_err(|error| error.to_string())?;
        emit(event.envelope());
        return Ok(outcome.clone());
    }
    tasks::save_with_ending(dir, &record, || {
        let draft = events::EventDraft::idempotent(
            &record.run_id,
            outcome.event_type(),
            &record.user_id,
            "ending",
        )
        .with(ending_payload);
        match events.record_fenced(draft, lease) {
            Ok(event) => emit(event.envelope()),
            Err(events::AppendError::Duplicate { .. })
            | Err(events::AppendError::AlreadyEnded { .. }) => {}
            Err(error) => return Err(error.to_string()),
        }
        events
            .snapshot(&record.run_id)?
            .as_ref()
            .and_then(RunOutcome::from_snapshot)
            .ok_or_else(|| "The authoritative terminal state could not be confirmed.".into())
    })
}
