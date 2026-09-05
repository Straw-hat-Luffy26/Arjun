//! The record a finished run leaves behind.
//!
//! ARJUN design rule: a Tasks surface showing every task that has run, with
//! its plan, the models it chose and why, the evidence it retrieved, and the
//! artifacts it produced. PS 26117 does not ask for this surface; it is how
//! ARJUN makes an agentic run checkable after the fact. Until now a run left nothing at all: the answer went
//! to whoever was watching, and everything that made it checkable — the plan it
//! was held to, the passages it stood on, the working behind its figures — was
//! dropped when the process moved on.
//!
//! ## Why this is a file and not a row in the audit log
//!
//! The audit log already records that a run happened and which model took it,
//! under access control and hash-chained. It is deliberately not the place for a
//! task's whole contents: an append-only chain is expensive to write a 200 KB
//! answer into, and the value of the chain is that entries are small, ordered
//! and tamper-evident rather than that they hold everything.
//!
//! So the two are complementary and say so: the audit entry is the evidence
//! that the run occurred, and this is the readable record of what it did.
//!
//! ## Why it is written even when the run failed
//!
//! A run that stopped at its step budget, or was aborted, or whose answer did
//! not pass verification is exactly the run somebody will want to look at
//! afterwards. Saving only the successes would make the Tasks screen a list of
//! good news, which is worse than no screen.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::artifacts::VerificationReport;
use crate::knowledge::SearchResult;
use crate::orchestrator::calculation::CalculationRecord;
use crate::orchestrator::plan::{PlanRun, PlanStep, StopReason};
use crate::registry::router::RoutingDecision;
use crate::serving::Endpoint;

use super::artifacts::ArtifactReport;

/// One planned step, and whether the run left behind the evidence for it.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PlanStepRecord {
    pub ordinal: u32,
    /// What the step was for, in the person's terms.
    pub intent: String,
    /// True only when the evidence named by `settled_by` actually exists.
    pub done: bool,
    /// What would settle this step, in words. Shown on an unfinished step so
    /// the gap says what is missing rather than only that something is.
    pub settled_by: String,
}

/// The plan as it stood when the run ended.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PlanRecord {
    pub steps: Vec<PlanStepRecord>,
    pub max_steps: u32,
    pub max_duration_seconds: u64,
    /// Tool names, as the model would have had to write them.
    pub permitted_tools: Vec<String>,
    pub repeat_limit: u32,
    pub steps_taken: u32,
    /// Absent while a run is still going.
    pub stop_reason: Option<StopReason>,
    /// The stop reason as a sentence, so the UI does not have to rebuild it.
    pub stopped_because: String,
}

impl PlanRecord {
    pub fn of(plan: &PlanRun) -> Self {
        let stop_reason = plan.stopped().cloned();
        Self {
            steps: plan
                .steps
                .iter()
                .map(|step| PlanStepRecord {
                    ordinal: step.ordinal,
                    intent: step.intent.clone(),
                    // Nothing is finished until its evidence exists. See
                    // `settle`, which is the only thing that sets this.
                    done: false,
                    settled_by: String::new(),
                })
                .collect(),
            max_steps: plan.budget.max_steps,
            max_duration_seconds: plan.budget.max_duration.as_secs(),
            permitted_tools: plan
                .budget
                .permitted_tools
                .iter()
                .map(|tool| tool.as_str().to_string())
                .collect(),
            repeat_limit: plan.budget.repeat_limit,
            steps_taken: plan.steps_taken(),
            stopped_because: stop_reason
                .as_ref()
                .map(StopReason::explain)
                .unwrap_or_else(|| "Still running.".to_string()),
            stop_reason,
        }
    }

    /// Decides which planned steps the run actually carried out.
    ///
    /// Against evidence the run left behind, never against the model's account
    /// of itself. A step wanting a `create_docx` is finished when a `create_docx`
    /// call succeeded; a model that says it wrote the document and did not call
    /// the tool leaves that step unfinished, which is the entire point.
    ///
    /// `specs` comes from [`super::planning::derive`], which is deterministic
    /// over the prompt — so re-deriving here gives back exactly the plan this
    /// run was held to, with no need to have carried it around.
    pub fn settle(
        &mut self,
        specs: &[super::planning::StepSpec],
        succeeded_tools: &[String],
        has_answer: bool,
        verified: bool,
    ) {
        use super::planning::Satisfies;

        for (step, spec) in self.steps.iter_mut().zip(specs.iter()) {
            step.settled_by = spec.satisfied_by.describe();
            step.done = match &spec.satisfied_by {
                Satisfies::Tool(tool) => succeeded_tools.iter().any(|ran| ran == tool.as_str()),
                Satisfies::Answer => has_answer,
                Satisfies::Verification => verified,
            };
        }
    }

    /// Steps that were planned and never reached.
    ///
    /// Shown rather than hidden: somebody who can see what was skipped can
    /// decide whether the partial answer is usable, and somebody who cannot has
    /// to assume the worst.
    pub fn unfinished(&self) -> Vec<&PlanStepRecord> {
        self.steps.iter().filter(|step| !step.done).collect()
    }

    /// Records how the run actually ended, once that is known.
    ///
    /// The plan itself only knows the endings *it* caused — out of steps, out
    /// of time, going in circles. A loop that simply finished, or a runtime
    /// that fell over, ends the run without the plan hearing about it, and a
    /// record still saying "Still running." after the fact would be a plain
    /// untruth on the screen somebody checks the work on.
    ///
    /// Deliberately does not mark the planned steps done. Whether the model
    /// actually did each thing it set out to do is not something reaching the
    /// end of the loop establishes, and the artifacts and the verification
    /// below it are the evidence for that question.
    pub fn ended(&mut self, failure: Option<&str>) {
        if self.stop_reason.is_some() {
            // The plan stopped it, and that reason is the more specific one.
            return;
        }
        match failure {
            Some(detail) => {
                self.stopped_because = StopReason::Failed {
                    detail: detail.to_string(),
                }
                .explain();
                self.stop_reason = Some(StopReason::Failed {
                    detail: detail.to_string(),
                });
            }
            None => {
                self.stopped_because = format!(
                    "Finished, using {} of the {} tool calls it was allowed.",
                    self.steps_taken, self.max_steps
                );
                self.stop_reason = Some(StopReason::Completed);
            }
        }
    }
}

/// How one tool call ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum CallOutcome {
    Succeeded,
    /// The tool ran and failed — a missing file, an expression that would not
    /// evaluate. The model reads the reason and can try something else.
    Failed,
    /// The gateway, the plan or a person said no before it ran. Not a fault:
    /// a trace full of refusals is the policy working, and drawing it as a
    /// failure would teach people to ignore the ones that matter.
    Refused,
}

/// One tool call the run made.
///
/// Arguments are deliberately absent. They can carry whole documents, the audit
/// log already holds them under access control, and a second copy in a file the
/// Tasks screen reads would be a second place to leak them from. What is kept is
/// what somebody reviewing the work needs: which tool, how it went, and what it
/// said.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolCallRecord {
    pub tool: String,
    pub outcome: CallOutcome,
    /// What the tool reported, trimmed. The model's own result text, so a
    /// reviewer sees what the model saw rather than a summary of it.
    pub detail: String,
    /// RFC 3339, UTC.
    pub at: String,
}

/// Longest tool result kept per call.
const DETAIL_CHARS: usize = 400;

impl ToolCallRecord {
    pub fn new(tool: &str, outcome: CallOutcome, detail: &str) -> Self {
        let trimmed: String = detail.chars().take(DETAIL_CHARS).collect();
        Self {
            tool: tool.to_string(),
            outcome,
            detail: if detail.chars().count() > DETAIL_CHARS {
                format!("{trimmed}…")
            } else {
                trimmed
            },
            at: chrono::Utc::now().to_rfc3339(),
        }
    }
}

/// One thing a person was asked to allow during the run, and what they said.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ApprovalRecord {
    pub id: String,
    pub tool: String,
    /// What it would act on.
    pub target: String,
    /// The arguments as the approver read them.
    pub arguments: Vec<String>,
    pub consequences: String,
    pub requested_at: String,
    /// `approved`, `rejected`, or `pending` for one nobody answered.
    pub state: String,
    pub decided_by: Option<String>,
    pub decided_at: Option<String>,
    /// Present on a rejection: a refusal without a reason is not a decision.
    pub because: Option<String>,
}

/// One passage the run stood on, as its citation marker refers to it.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EvidenceRecord {
    /// The number in `[E3]`.
    pub marker: usize,
    pub citation: String,
    pub document_name: String,
    pub page: u32,
    /// Trimmed. The whole passage belongs in the knowledge base, not in every
    /// task record that happened to retrieve it.
    pub excerpt: String,
}

/// Longest excerpt kept per passage.
const EXCERPT_CHARS: usize = 600;

impl EvidenceRecord {
    fn of(marker: usize, passage: &SearchResult) -> Self {
        let excerpt: String = passage.text.chars().take(EXCERPT_CHARS).collect();
        Self {
            marker,
            citation: passage.citation(),
            document_name: passage.document_name.clone(),
            page: passage.page,
            excerpt: if passage.text.chars().count() > EXCERPT_CHARS {
                format!("{excerpt}…")
            } else {
                excerpt
            },
        }
    }
}

/// How the context window was divided at a moment in a run.
///
/// The answer to the question an operator has when a run compacts four times in
/// twenty turns: *what filled it?* A compaction count alone says the window ran
/// out; this says whether it was the tool schemas, one enormous tool result, or
/// simply a long conversation — and only one of those three has a remedy that
/// does not degrade the run.
///
/// Mirrors `ContextLedgerSnapshot` in the runtime. The counts are the runtime's
/// own, not re-derived here: two estimators would disagree, and the one that
/// matters is the one compaction actually decided on.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ContextLedgerRecord {
    pub system: u32,
    pub skill: u32,
    pub tool_schema: u32,
    pub evidence: u32,
    pub notes: u32,
    pub transcript: u32,
    pub compaction: u32,
    /// Held back for the model's output and the summarisation request. Not
    /// occupied — committed, which is the same thing for deciding whether the
    /// next turn fits.
    pub reserve: u32,
    /// Everything except `reserve`.
    pub occupied: u32,
    /// `occupied + reserve`. What the next turn has to fit inside.
    pub committed: u32,
    /// The model's window. Zero when the runtime was not told one.
    pub window: u32,
    /// `window - committed`. Negative would mean the next turn does not fit;
    /// stored signed so a reader is not told a shortfall is a surplus.
    pub headroom: i64,
}

/// One time a run's older history was replaced by a summary.
///
/// Kept on the record, not only counted, because "compacted three times" and
/// "compacted three times, and the third pass had to summarise ninety messages
/// to claw back four hundred tokens" are different runs, and only the second
/// tells somebody the task was too large for the model it was given.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompactionRecord {
    /// Which compaction of this run, 1-based.
    pub ordinal: u32,
    /// RFC 3339, UTC.
    pub at: String,
    pub tokens_before: u32,
    pub tokens_after: u32,
    /// Messages now represented by the summary rather than sent whole.
    pub messages_summarised: u32,
    /// True when this pass extended the summary already held.
    ///
    /// A `false` on anything but the first pass would mean the run started a
    /// second summary and the earlier half of its history is described twice or
    /// not at all. Recorded so that is visible rather than inferred.
    pub refined_existing_summary: bool,
    /// Raw tool results replaced by an evidence reference, cumulatively.
    pub tool_results_cleared: u32,
    /// Where the window stood after this pass.
    pub ledger: ContextLedgerRecord,
}

/// Everything worth keeping about one finished run.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TaskRecord {
    /// Full private specialist packets/results, persisted under this parent.
    #[serde(default)]
    pub children: Vec<super::events::children::ChildRecord>,
    pub run_id: String,
    pub prompt: String,
    /// RFC 3339, UTC.
    pub started_at: String,
    pub finished_at: String,
    pub duration_seconds: u64,
    /// Who asked. The audit log holds the authoritative attribution; this is
    /// here so the Tasks screen can show it without reading that log.
    pub user_id: String,
    pub routing: RoutingDecision,
    pub endpoint: Endpoint,
    pub plan: PlanRecord,
    pub answer: String,
    pub turns: u32,
    /// Absent when the run produced no answer to check.
    pub verification: Option<VerificationReport>,
    /// The independent run-level gate; old records remain readable but are not
    /// promoted to verified success when this evidence is absent.
    #[serde(default)]
    pub completion_verification: Option<super::completion::CompletionVerification>,
    pub artifacts: Vec<ArtifactReport>,
    pub evidence: Vec<EvidenceRecord>,
    pub calculations: Vec<CalculationRecord>,
    /// Every tool the run called, in order, with how each one went.
    pub tool_calls: Vec<ToolCallRecord>,
    /// Everything a person was asked to allow during the run, and what they
    /// said. Kept even when nobody answered — an unanswered request is the
    /// reason a run stalled, and a record omitting it explains nothing.
    pub approvals: Vec<ApprovalRecord>,
    /// Set when the run ended badly, in the words shown to the person.
    pub failure: Option<String>,
    /// How the run ended, typed.
    ///
    /// The authority for what happened, and the only field that separates a
    /// finished answer from a fragment cut off at the output cap — the text of
    /// the two is indistinguishable. `failure` carries the same information as
    /// a sentence for display; this carries it as a value a screen can filter
    /// and count on.
    ///
    /// Defaulted so records written before this existed still load. Those
    /// records read as `None`, which is honestly "not recorded" rather than a
    /// guess at which ending they had.
    #[serde(default)]
    pub outcome: Option<crate::agent_runtime::outcome::RunOutcome>,
    /// Every time the run's history was replaced by a summary, in order.
    ///
    /// Defaulted so records written before this existed still load. A run that
    /// never compacted has an empty list, which is the truthful answer and is
    /// distinguishable from a record that predates the field only by its date.
    #[serde(default)]
    pub compactions: Vec<CompactionRecord>,
    /// The run's bounded notes as they finished.
    ///
    /// What a resumption reads. Absent on a record written before this existed,
    /// and on a run that recorded nothing — the two are not distinguished here,
    /// because in both cases there is nothing to resume from.
    #[serde(default)]
    pub working_notes: Option<super::memory::RunMemory>,
    /// Where the context window stood when the run ended.
    #[serde(default)]
    pub context_ledger: Option<ContextLedgerRecord>,
}

impl TaskRecord {
    /// Turns retrieved passages into the record's evidence list.
    pub fn evidence_from(passages: &[SearchResult]) -> Vec<EvidenceRecord> {
        passages
            .iter()
            .enumerate()
            .map(|(i, passage)| EvidenceRecord::of(i + 1, passage))
            .collect()
    }

    /// Whether this task can be handed on as it stands.
    pub fn is_ready(&self) -> bool {
        self.failure.is_none()
            && self.outcome.as_ref().is_some_and(super::outcome::RunOutcome::is_success)
            && self.completion_verification.as_ref().is_some_and(super::completion::CompletionVerification::passed)
            && self
                .verification
                .as_ref()
                .map(VerificationReport::is_ready)
                .unwrap_or(true)
            && self.artifacts.iter().all(|artifact| artifact.sound)
            // A run that never reached half its plan is not ready however
            // confident its answer reads.
            && self.plan.unfinished().is_empty()
    }
}

/// A row on the Tasks screen. Small enough that listing a thousand of them does
/// not mean reading a thousand answers off disk.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TaskSummary {
    pub run_id: String,
    /// Who ran it. Carried on the summary so the listing can be filtered to
    /// what the reader may see without opening every record.
    pub user_id: String,
    pub prompt: String,
    pub started_at: String,
    pub finished_at: String,
    pub duration_seconds: u64,
    pub model_name: String,
    pub intent: String,
    pub turns: u32,
    pub artifact_count: usize,
    pub evidence_count: usize,
    pub tool_call_count: usize,
    /// Steps planned but never reached. Non-zero is the signal to look.
    pub unfinished_steps: usize,
    /// Approval requests nobody answered.
    pub approvals_pending: usize,
    /// Times the run's older history was replaced by a summary so it could
    /// continue. Non-zero on a short task is the signal that the routed model's
    /// window is too small for the work it is being given.
    pub compaction_count: usize,
    pub stopped_because: String,
    pub ready: bool,
    pub failure: Option<String>,
    /// Where the run stands.
    ///
    /// Read off the durable event history where there is one, and derived from
    /// the record otherwise. It is the only thing on this row that can say
    /// *degraded* — a run the process took down with it writes no record at
    /// all, so before the event history existed those runs simply did not
    /// appear on this screen.
    pub state: super::events::RunState,
    /// True while the run is still going. The row is drawn differently and is
    /// not offered as something to hand on.
    pub live: bool,
}

impl From<&TaskRecord> for TaskSummary {
    fn from(record: &TaskRecord) -> Self {
        Self {
            state: if record.failure.is_some() {
                super::events::RunState::Failed
            } else {
                super::events::RunState::Completed
            },
            live: false,
            run_id: record.run_id.clone(),
            user_id: record.user_id.clone(),
            prompt: record.prompt.clone(),
            started_at: record.started_at.clone(),
            finished_at: record.finished_at.clone(),
            duration_seconds: record.duration_seconds,
            model_name: record.routing.model_name.clone(),
            intent: record.routing.intent.clone(),
            turns: record.turns,
            artifact_count: record.artifacts.len(),
            evidence_count: record.evidence.len(),
            tool_call_count: record.tool_calls.len(),
            unfinished_steps: record.plan.unfinished().len(),
            approvals_pending: record
                .approvals
                .iter()
                .filter(|approval| approval.state == "pending")
                .count(),
            compaction_count: record.compactions.len(),
            stopped_because: record.plan.stopped_because.clone(),
            ready: record.is_ready(),
            failure: record.failure.clone(),
        }
    }
}

/// A row for a run that has no record yet — one still going, or one the process
/// took down with it.
///
/// The counterpart to [`TaskSummary::from`], which reads a finished run's JSON
/// record. Both produce the same row so the Tasks screen does not have to know
/// which kind it is looking at; what differs is how much is knowable. A run
/// still in flight has no duration, no verification and no evidence count yet,
/// and this reports those as zero rather than inventing them.
pub fn summary_of(snapshot: &super::events::TaskSnapshot) -> TaskSummary {
    let live = !snapshot.state.is_terminal();
    let plan = snapshot.plan.as_ref();
    TaskSummary {
        run_id: snapshot.run_id.clone(),
        user_id: snapshot.actor.clone(),
        prompt: snapshot.prompt.clone(),
        started_at: snapshot.started_at.clone(),
        // A run that has not ended has not ended. Reporting `updated_at` here
        // would put a finish time on the screen for a task that is still going.
        finished_at: if live {
            String::new()
        } else {
            snapshot.updated_at.clone()
        },
        duration_seconds: duration_between(&snapshot.started_at, &snapshot.updated_at),
        model_name: snapshot.model_name.clone(),
        intent: String::new(),
        turns: snapshot.turns,
        artifact_count: snapshot.artifacts.len(),
        evidence_count: 0,
        tool_call_count: snapshot.activity.len(),
        unfinished_steps: plan
            .map(|plan| plan.unfinished().len())
            .unwrap_or_default(),
        approvals_pending: snapshot.approvals_pending,
        compaction_count: snapshot.compactions as usize,
        stopped_because: snapshot
            .stopped_because
            .clone()
            .unwrap_or_else(|| snapshot.state.describe().to_string()),
        // Nothing without a record is ready to hand on: there is no answer to
        // check, no artifact re-opened, and no verification.
        ready: false,
        failure: snapshot.failure.clone(),
        state: snapshot.state,
        live,
    }
}

/// Seconds between two RFC 3339 instants, or zero if either will not parse.
fn duration_between(from: &str, to: &str) -> u64 {
    let parse = |raw: &str| chrono::DateTime::parse_from_rfc3339(raw).ok();
    match (parse(from), parse(to)) {
        (Some(from), Some(to)) => (to - from).num_seconds().max(0) as u64,
        _ => 0,
    }
}

/// Where task records live under the application's data directory.
pub fn directory(app_data_dir: &Path) -> PathBuf {
    app_data_dir.join("tasks")
}

/// Publish success only after the controller has committed and read the ending.
/// A crash or failed terminal write leaves an inspectable, non-successful record.
/// The callback returns the authoritative ending, including a winning cancellation.
pub fn save_with_ending(
    app_data_dir: &Path,
    record: &TaskRecord,
    finish: impl FnOnce() -> Result<super::outcome::RunOutcome, String>,
) -> Result<super::outcome::RunOutcome, String> {
    let mut saved = record.clone();
    if saved.outcome.as_ref().is_some_and(super::outcome::RunOutcome::is_success) {
        let pending = super::outcome::RunOutcome::NeedsReview {
            detail: "Final publication has not been confirmed.".into(),
        };
        saved.failure = pending.detail().map(str::to_string);
        saved.outcome = Some(pending);
    }
    save(app_data_dir, &saved)?;
    let outcome = finish()?;
    saved.failure = outcome.detail().map(str::to_string);
    saved.plan.ended(saved.failure.as_deref());
    saved.outcome = Some(outcome.clone());
    save(app_data_dir, &saved)?;
    Ok(outcome)
}

/// Writes one record.
///
/// Written to a temporary name and renamed into place, so a crash midway
/// through leaves the previous record rather than half of a new one. A task
/// list that can contain a truncated entry is a task list whose every reader
/// has to defend against truncated entries.
pub fn save(app_data_dir: &Path, record: &TaskRecord) -> Result<PathBuf, String> {
    let dir = directory(app_data_dir);
    std::fs::create_dir_all(&dir)
        .map_err(|error| format!("the task record directory could not be created: {error}"))?;

    let path = dir.join(format!("{}.json", record.run_id));
    let temporary = dir.join(format!("{}.json.writing", record.run_id));
    let body = serde_json::to_vec_pretty(record)
        .map_err(|error| format!("the task record could not be written: {error}"))?;

    std::fs::write(&temporary, body)
        .map_err(|error| format!("the task record could not be written: {error}"))?;
    std::fs::rename(&temporary, &path)
        .map_err(|error| format!("the task record could not be saved: {error}"))?;
    Ok(path)
}

/// Reads one record back. `owner_user_id` is the per-user isolation
/// boundary from TODO 2: a request from a different user returns
/// `Err` (the same shape as a missing file, so the UI sees a
/// "task not found" rather than a 403-style leak). `None` is the
/// unrestricted form, used by tests and the audit log.
pub fn load(
    app_data_dir: &Path,
    run_id: &str,
    owner_user_id: Option<&str>,
) -> Result<TaskRecord, String> {
    // Checked as a single component: a run id is a UUID this process generated,
    // but a task id arriving from the UI should not be able to name a path.
    let name = format!("{run_id}.json");
    if Path::new(&name).components().count() != 1 {
        return Err(format!("{run_id:?} is not a task id."));
    }
    let path = directory(app_data_dir).join(name);
    let body = std::fs::read(&path)
        .map_err(|error| format!("that task's record could not be read: {error}"))?;
    let record: TaskRecord = serde_json::from_slice(&body)
        .map_err(|error| format!("that task's record could not be understood: {error}"))?;
    if let Some(owner) = owner_user_id {
        if record.user_id != owner {
            return Err(format!("that task was not found for {owner}."));
        }
    }
    Ok(record)
}

/// Every task, newest first. `owner_user_id` is the per-user filter;
/// `None` returns every record (used by tests and the audit log).
///
/// A record that will not parse is skipped rather than failing the listing. One
/// unreadable file from an older build should cost its own row, not the screen.
pub fn list(app_data_dir: &Path, owner_user_id: Option<&str>) -> Vec<TaskSummary> {
    let Ok(entries) = std::fs::read_dir(directory(app_data_dir)) else {
        // No directory yet simply means no task has been run.
        return Vec::new();
    };

    let mut summaries: Vec<TaskSummary> = entries
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|extension| extension == "json"))
        .filter_map(|path| std::fs::read(&path).ok())
        .filter_map(|body| serde_json::from_slice::<TaskRecord>(&body).ok())
        .filter(|record| match owner_user_id {
            Some(owner) => record.user_id == owner,
            None => true,
        })
        .map(|record| TaskSummary::from(&record))
        .collect();

    // On the finish time, so a task that ran for an hour does not sort above one
    // started after it finished. RFC 3339 in UTC sorts lexically by instant.
    summaries.sort_by(|a, b| b.finished_at.cmp(&a.finished_at));
    summaries
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::agent_runtime::artifacts::Kind;
    use crate::artifacts::verifier::Finding;
    use crate::artifacts::{Severity, Standing};
    use crate::orchestrator::plan::Budget;
    use crate::orchestrator::tools::ToolName;
    use crate::registry::{ModelRole, Runtime};

    /// A whole, realistic task record.
    ///
    /// `pub(crate)` because the durability tests in `agent_runtime::tests` need
    /// a real record to drive `save` against a disk that will not take it, and
    /// rebuilding one there would mean maintaining a second fixture that drifts
    /// from this one.
    pub(crate) fn record(run_id: &str, finished_at: &str) -> TaskRecord {
        TaskRecord {
            children: Vec::new(),
            run_id: run_id.to_string(),
            prompt: "draft an approval note".to_string(),
            started_at: "2026-08-27T10:00:00+00:00".to_string(),
            finished_at: finished_at.to_string(),
            duration_seconds: 42,
            user_id: "priya".to_string(),
            routing: RoutingDecision {
                model_id: "qwen2.5-7b".to_string(),
                model_name: "Qwen2.5 7B".to_string(),
                role: ModelRole::Reasoning,
                intent: "reasoning".to_string(),
                confidence: 0.8,
                used_fallback: false,
                reasons: vec!["it fits in VRAM".to_string()],
                gpu_plan_summary: "all layers on GPU".to_string(),
                fully_on_gpu: true,
            },
            endpoint: Endpoint {
                base_url: "http://127.0.0.1:8080/v1".to_string(),
                served_model_id: "qwen2.5-7b".to_string(),
                managed: true,
                runtime: Runtime::LlamaCpp,
            },
            plan: PlanRecord::of(&PlanRun::new(
                run_id,
                vec!["Search".to_string()],
                Budget::standard(vec![ToolName::SearchDocuments]),
            )),
            answer: "The seal is worn beyond the limit [E1].".to_string(),
            turns: 3,
            compactions: Vec::new(),
            working_notes: None,
            context_ledger: None,
            verification: None,
            completion_verification: None,
            artifacts: Vec::new(),
            evidence: Vec::new(),
            calculations: Vec::new(),
            tool_calls: Vec::new(),
            approvals: Vec::new(),
            failure: None,
            outcome: Some(crate::agent_runtime::outcome::RunOutcome::Completed),
        }
    }

    #[test]
    fn a_saved_task_reads_back_as_what_was_written() {
        let dir = tempfile::tempdir().expect("temp dir");
        let written = record("run-1", "2026-08-27T10:00:42+00:00");
        save(dir.path(), &written).expect("saved");

        let read = load(dir.path(), "run-1", None).expect("loaded");
        assert_eq!(read.prompt, written.prompt);
        assert_eq!(read.routing.model_name, "Qwen2.5 7B");
        assert_eq!(read.plan.permitted_tools, vec!["knowledge.search_authorized"]);
    }

    #[test]
    fn a_failed_terminal_commit_cannot_leave_a_successful_task_record() {
        let dir = tempfile::tempdir().unwrap();
        let candidate = record("r", "2026-08-27T10:00:42+00:00");
        let result = save_with_ending(dir.path(), &candidate, || {
            let provisional = load(dir.path(), "r", None).unwrap();
            assert!(!provisional.outcome.unwrap().is_success());
            Err("injected terminal-write failure".into())
        });
        assert!(result.is_err());
        assert!(!load(dir.path(), "r", None).unwrap().outcome.unwrap().is_success());
    }

    #[test]
    fn a_cancellation_winning_the_terminal_race_is_saved_instead_of_model_success() {
        let dir = tempfile::tempdir().unwrap();
        let candidate = record("r", "2026-08-27T10:00:42+00:00");
        let ending = super::super::outcome::RunOutcome::Aborted { detail: "Stopped by operator".into() };
        assert_eq!(save_with_ending(dir.path(), &candidate, || Ok(ending.clone())).unwrap(), ending);
        let saved = load(dir.path(), "r", None).unwrap();
        assert_eq!(saved.outcome, Some(ending));
        assert_eq!(saved.failure.as_deref(), Some("Stopped by operator"));
    }

    #[test]
    fn an_unwritable_provisional_record_does_not_commit_a_successful_ending() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(directory(dir.path()), "not a directory").unwrap();
        let candidate = record("r", "2026-08-27T10:00:42+00:00");
        assert!(save_with_ending(dir.path(), &candidate, || panic!("must not declare success")).is_err());
    }

    #[test]
    fn tasks_list_newest_first() {
        let dir = tempfile::tempdir().expect("temp dir");
        save(dir.path(), &record("old", "2026-08-27T09:00:00+00:00")).expect("saved");
        save(dir.path(), &record("new", "2026-08-27T11:00:00+00:00")).expect("saved");

        let listed = list(dir.path(), None);
        assert_eq!(listed.len(), 2);
        assert_eq!(listed[0].run_id, "new");
    }

    #[test]
    fn no_tasks_yet_is_an_empty_list_and_not_an_error() {
        // The Tasks screen opens before anything has ever been run.
        let dir = tempfile::tempdir().expect("temp dir");
        assert!(list(dir.path(), None).is_empty());
    }

    #[test]
    fn one_unreadable_record_does_not_cost_the_whole_listing() {
        let dir = tempfile::tempdir().expect("temp dir");
        save(dir.path(), &record("good", "2026-08-27T09:00:00+00:00")).expect("saved");
        std::fs::write(directory(dir.path()).join("broken.json"), b"{ not json")
            .expect("wrote the broken one");

        let listed = list(dir.path(), None);
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].run_id, "good");
    }

    /// A record written before the durable event history existed still reads.
    ///
    /// Written as raw JSON rather than through [`save`] on purpose: serialising
    /// the current struct and reading it back would pass however the struct
    /// changed, which is precisely the regression this is meant to catch. This
    /// is the exact field set older builds wrote.
    #[test]
    fn a_record_from_before_the_event_history_still_lists_and_opens() {
        let dir = tempfile::tempdir().expect("temp dir");
        std::fs::create_dir_all(directory(dir.path())).expect("the directory");
        std::fs::write(
            directory(dir.path()).join("old-run.json"),
            r#"{
              "runId": "old-run",
              "prompt": "draft an approval note",
              "startedAt": "2026-08-20T10:00:00+00:00",
              "finishedAt": "2026-08-20T10:00:42+00:00",
              "durationSeconds": 42,
              "userId": "priya",
              "routing": {
                "modelId": "qwen2.5-7b", "modelName": "Qwen2.5 7B", "role": "reasoning",
                "intent": "reasoning", "confidence": 0.8, "usedFallback": false,
                "reasons": ["it fits in VRAM"], "gpuPlanSummary": "all layers on GPU",
                "fullyOnGpu": true
              },
              "endpoint": {
                "baseUrl": "http://127.0.0.1:8080/v1", "servedModelId": "qwen2.5-7b",
                "managed": true, "runtime": "llamaCpp"
              },
              "plan": {
                "steps": [], "maxSteps": 12, "maxDurationSeconds": 600,
                "permittedTools": ["search_documents"], "repeatLimit": 3, "stepsTaken": 2,
                "stopReason": {"reason": "completed"}, "stoppedBecause": "Finished."
              },
              "answer": "The seal is worn beyond the limit [E1].",
              "turns": 3,
              "verification": null,
              "artifacts": [],
              "evidence": [],
              "calculations": [],
              "toolCalls": [],
              "approvals": [],
              "failure": null
            }"#,
        )
        .expect("the old record");

        let listed = list(dir.path(), None);
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].run_id, "old-run");
        assert_eq!(listed[0].model_name, "Qwen2.5 7B");
        // The fields the summary grew are derived, so an old record gets a
        // truthful value for them rather than a missing one.
        assert_eq!(listed[0].state, crate::agent_runtime::events::RunState::Completed);
        assert!(!listed[0].live);

        let opened = load(dir.path(), "old-run", None).expect("it opens");
        assert_eq!(opened.answer, "The seal is worn beyond the limit [E1].");
    }

    #[test]
    fn an_old_record_that_failed_summarises_as_failed_rather_than_finished() {
        let dir = tempfile::tempdir().expect("temp dir");
        let mut written = record("run-1", "2026-08-27T10:00:42+00:00");
        written.failure = Some("the agent runtime stopped".to_string());
        save(dir.path(), &written).expect("saved");

        assert_eq!(
            list(dir.path(), None)[0].state,
            crate::agent_runtime::events::RunState::Failed
        );
    }

    #[test]
    fn a_run_with_no_record_yet_summarises_from_what_is_known() {
        // The row an interrupted run gets. Everything the record would have
        // supplied is absent, and the summary says so rather than inventing it.
        let mut snapshot = crate::agent_runtime::events::TaskSnapshot::empty("run-1");
        snapshot.state = crate::agent_runtime::events::RunState::DegradedNeedsHuman;
        snapshot.prompt = "draft an approval note".to_string();
        snapshot.actor = "priya".to_string();
        snapshot.started_at = "2026-08-27T10:00:00+00:00".to_string();
        snapshot.updated_at = "2026-08-27T10:00:30+00:00".to_string();
        snapshot.failure = Some("Interrupted: the application closed.".to_string());

        let summary = summary_of(&snapshot);
        assert_eq!(summary.user_id, "priya");
        assert_eq!(summary.duration_seconds, 30);
        assert!(!summary.live);
        // Nothing without a record is ready to hand on: there is no answer to
        // check, no artifact re-opened, and no verification.
        assert!(!summary.ready);
    }

    #[test]
    fn a_run_still_going_is_not_given_a_finish_time_it_does_not_have() {
        let mut snapshot = crate::agent_runtime::events::TaskSnapshot::empty("run-1");
        snapshot.started_at = "2026-08-27T10:00:00+00:00".to_string();
        snapshot.updated_at = "2026-08-27T10:00:30+00:00".to_string();

        let summary = summary_of(&snapshot);
        assert!(summary.live);
        assert!(summary.finished_at.is_empty());
    }

    #[test]
    fn a_task_id_cannot_name_a_path() {
        let dir = tempfile::tempdir().expect("temp dir");
        assert!(load(dir.path(), "../../secrets", None).is_err());
    }

    #[test]
    fn a_task_whose_draft_needs_review_is_not_ready() {
        let mut written = record("run-1", "2026-08-27T10:00:42+00:00");
        written.verification = Some(VerificationReport {
            coverage: Default::default(),
            standing: Standing::NeedsReview {
                blocking: 1,
                advisory: 0,
            },
            findings: vec![Finding {
                severity: Severity::Blocking,
                detail: "cites a passage that was never retrieved".to_string(),
                excerpt: Some("[E9]".to_string()),
            }],
            citations_resolved: 0,
            figures_checked: 0,
        });

        assert!(!written.is_ready());
    }

    #[test]
    fn a_task_whose_document_did_not_open_is_not_ready() {
        // The artifact is the deliverable. A run that produced a corrupt one has
        // not finished, whatever its answer says.
        let mut written = record("run-1", "2026-08-27T10:00:42+00:00");
        written.artifacts = vec![ArtifactReport {
            name: "note.docx".to_string(),
            path: "/runs/run-1/note.docx".to_string(),
            kind: Kind::Document,
            template: Some("approval_note".to_string()),
            bytes: 12,
            sound: false,
            detail: "Does not open as a Word document.".to_string(),
            problems: vec!["the document is missing word/document.xml".to_string()],
            produced_at: "2026-08-27T10:00:30+00:00".to_string(),
        }];

        assert!(!written.is_ready());
    }

    #[test]
    fn a_failed_run_is_still_saved_and_still_listed() {
        // The run somebody most wants to look at afterwards is the one that went
        // wrong. A Tasks screen showing only successes is worse than none.
        let dir = tempfile::tempdir().expect("temp dir");
        let mut written = record("run-1", "2026-08-27T10:00:42+00:00");
        written.failure = Some("the agent runtime stopped".to_string());
        save(dir.path(), &written).expect("saved");

        let listed = list(dir.path(), None);
        assert_eq!(listed.len(), 1);
        assert!(!listed[0].ready);
        assert_eq!(
            listed[0].failure.as_deref(),
            Some("the agent runtime stopped")
        );
    }

    // -----------------------------------------------------------------------
    // Per-user isolation tests (TODO 2 of the 7-step plan).
    // -----------------------------------------------------------------------

    #[test]
    fn a_user_does_not_see_another_users_tasks_in_the_listing() {
        let dir = tempfile::tempdir().expect("temp dir");
        let mut by_priya = record("priya-1", "2026-08-27T10:00:00+00:00");
        by_priya.user_id = "modeladmin".to_string(); // S. Kulkarni
        let mut by_ravi = record("ravi-1", "2026-08-27T11:00:00+00:00");
        by_ravi.user_id = "admin".to_string(); // R. Nair
        save(dir.path(), &by_priya).expect("saved priya");
        save(dir.path(), &by_ravi).expect("saved ravi");

        // S. Kulkarni asks for her own tasks: she sees one.
        let priya_view = list(dir.path(), Some("modeladmin"));
        assert_eq!(priya_view.len(), 1);
        assert_eq!(priya_view[0].run_id, "priya-1");
        // R. Nair asks for his own: he sees the other one.
        let ravi_view = list(dir.path(), Some("admin"));
        assert_eq!(ravi_view.len(), 1);
        assert_eq!(ravi_view[0].run_id, "ravi-1");
        // The unrestricted form returns both, for the audit log.
        let all = list(dir.path(), None);
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn a_user_cannot_load_another_users_task_record() {
        let dir = tempfile::tempdir().expect("temp dir");
        let mut by_priya = record("priya-1", "2026-08-27T10:00:00+00:00");
        by_priya.user_id = "modeladmin".to_string();
        save(dir.path(), &by_priya).expect("saved");

        // R. Nair asks for S. Kulkarni's task by id. The store
        // refuses at the function boundary, before any payload
        // is read into the response.
        let result = load(dir.path(), "priya-1", Some("admin"));
        assert!(
            result.is_err(),
            "loading a stranger's task must fail at the store boundary"
        );
        // S. Kulkarni can still load her own.
        let result = load(dir.path(), "priya-1", Some("modeladmin"));
        assert!(result.is_ok());
    }
}

#[cfg(test)]
mod ending_tests {
    use super::tests::record;
    use super::*;
    use crate::orchestrator::plan::Budget;
    use crate::orchestrator::tools::ToolName;

    #[test]
    fn a_run_that_simply_finished_does_not_still_say_it_is_running() {
        // The plan only learns about the endings it caused. Without this, a
        // successful task reads "Still running." on the screen somebody checks
        // the work on.
        let mut written = record("run-1", "2026-08-27T10:00:42+00:00");
        written.plan.ended(None);

        assert_eq!(written.plan.stop_reason, Some(StopReason::Completed));
        assert!(written.plan.stopped_because.contains("Finished"));
        assert!(!written.plan.stopped_because.contains("Still running"));
    }

    #[test]
    fn a_run_that_fell_over_records_why() {
        let mut written = record("run-1", "2026-08-27T10:00:42+00:00");
        written.plan.ended(Some("the agent runtime stopped"));

        assert!(written.plan.stopped_because.contains("the agent runtime stopped"));
    }

    #[test]
    fn the_plans_own_reason_wins_over_the_generic_one() {
        // "Stopped after 12 of 12 permitted steps" is a more useful thing to
        // read than "Finished", and it is the one that is actually true.
        let mut plan = PlanRun::new(
            "run-1",
            vec!["Search".to_string()],
            Budget::standard(vec![ToolName::SearchDocuments]),
        );
        plan.complete();
        let mut planned = PlanRecord::of(&plan);
        let already = planned.stopped_because.clone();

        planned.ended(Some("something else entirely"));
        assert_eq!(planned.stopped_because, already);
    }

    #[test]
    fn reaching_the_end_does_not_tick_the_planned_steps_off() {
        // A run finishing is not evidence that each thing it set out to do was
        // done. The artifacts and the verification are the evidence for that,
        // and a checklist that ticks itself would contradict them.
        let mut written = record("run-1", "2026-08-27T10:00:42+00:00");
        written.plan.ended(None);

        assert!(written.plan.steps.iter().all(|step| !step.done));
    }

    #[test]
    fn a_summary_carries_who_ran_it_so_the_listing_can_be_scoped() {
        let written = record("run-1", "2026-08-27T10:00:42+00:00");
        assert_eq!(TaskSummary::from(&written).user_id, "priya");
    }
}
