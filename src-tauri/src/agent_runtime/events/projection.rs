//! The latest state of a task, folded from its events.
//!
//! ## Why a snapshot exists at all
//!
//! The event log is the truth, and replaying it is how the truth is recovered.
//! But the Tasks screen opens on a list, and a list that replays every event of
//! every run to draw a row is a list that gets slower for the rest of the
//! product's life. So each run keeps one row of folded state, updated as the
//! run goes, and the screen reads that.
//!
//! The snapshot is therefore a **cache with a sequence number on it**, not a
//! second source of truth. It records the seq it was folded up to; anything
//! after that is folded on top when it is read, and a snapshot that cannot be
//! parsed at all is thrown away and rebuilt from the events. There is no state
//! the snapshot can be in that the events cannot correct.
//!
//! ## What it deliberately does not carry
//!
//! Not the answer, not any passage, and nothing the model said. A finished
//! run's answer is in its task record, which is access-controlled and already
//! exists; an unfinished run has no answer to carry. What is kept is a hash and
//! a length, which is enough to tell a reader that an answer landed and enough
//! to notice it changing.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::agent_runtime::tasks::PlanRecord;

use super::machine::{advance, RunState, Transition};
use super::model::{TaskEvent, TaskEventType, UnreadableEvent, SCHEMA_VERSION};

/// One thing the run did, in the order it did it.
///
/// Mirrors what the live trace shows, so a run recovered after a remount looks
/// like the one that was never interrupted rather than like a different screen.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct ActivityRecord {
    /// The runtime's id for the call, so a later outcome updates the right row.
    pub tool_call_id: String,
    pub tool: String,
    /// `running`, `done`, `failed`, `refused`, `replayed` or `unknown`.
    pub status: String,
    pub at: String,
}

/// A side effect nobody can account for, carried on the snapshot so the screen
/// can say what needs looking at without a second query.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct UnknownEffect {
    pub idempotency_key: String,
    pub tool: String,
    /// A reference — a file name — never contents.
    pub target: String,
    pub at: String,
}

/// Everything the UI needs to draw a task without replaying its history.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TaskSnapshot {
    pub run_id: String,
    /// The last event folded in. A caller that already holds a snapshot asks
    /// for events after this and applies them itself.
    pub seq: i64,
    pub schema_version: u32,
    pub state: RunState,
    /// RFC 3339, UTC.
    pub started_at: String,
    pub updated_at: String,
    /// When the run must stop, if it has a deadline. RFC 3339, UTC.
    pub deadline: Option<String>,
    /// Who started it.
    pub actor: String,
    /// What was asked, in the person's own words. Their own text, shown back to
    /// them — the confidentiality rule is about copying *documents* into a
    /// wider-read record, and a task with no prompt on it cannot be identified.
    pub prompt: String,
    pub model_name: String,
    pub classification: Option<String>,
    /// Absent until the plan is published, which is before the first turn.
    pub plan: Option<PlanRecord>,
    pub activity: Vec<ActivityRecord>,
    pub turns: u32,
    /// Conversation this turn could not carry, if any did not fit.
    ///
    /// `None` means the whole thread was sent. See `tasks::HistoryTrim` for why
    /// this is not folded in with the compactions: a compaction says what it
    /// replaced, and this is history that never reached the model at all.
    pub history_trim: Option<super::super::tasks::HistoryTrim>,
    /// Times older history was replaced by a summary so the run could continue.
    pub compactions: u32,
    /// What each of those compactions actually did.
    ///
    /// The count says the window ran out; these say what filled it, how much
    /// was reclaimed, and whether each pass refined the summary already held or
    /// started a new one. A run that reports three compactions and cannot say
    /// any of that is a run nobody can diagnose after the fact — which is the
    /// position the Tasks screen was in before this existed.
    ///
    /// Folded from the durable events, so a trace read after a restart shows
    /// the same list a watched one did.
    pub compaction_events: Vec<super::super::tasks::CompactionRecord>,
    /// Names only. The files are in the run's workspace; this is a reference.
    pub artifacts: Vec<String>,
    pub approvals_pending: usize,
    /// Times this run read a scope of memory it was entitled to.
    /// Resume points written during this run.
    pub checkpoints_taken: u32,
    /// Resume points that could not be written. Non-zero means the run cannot be
    /// continued from as far along as it actually got.
    pub checkpoint_failures: u32,
    /// Times a person picked this task up again.
    pub resumptions: u32,
    /// How many times the process has tried to pick this run back up by itself.
    ///
    /// `serde(default)` because snapshots written before this field existed are
    /// still in the database, and a cache that refuses to deserialise is a
    /// cache that silently rebuilds every run on every list.
    #[serde(default)]
    pub recovery_attempts: u32,
    /// How many of those gave up. The ceiling on automatic recovery is counted
    /// against `recovery_attempts`; this is what tells a person it kept failing.
    #[serde(default)]
    pub recovery_failures: u32,
    /// What the completion check concluded, once it has run. `None` means it
    /// has not — deliberately three-valued, because "not checked" and "checked
    /// and failed" must never collapse into the same thing.
    #[serde(default)]
    pub completion_verified: Option<bool>,
    pub memory_reads: u32,
    /// Skills whose instructions were loaded into this run's context.
    #[serde(default)]
    pub skills_loaded: u32,
    /// Skills the run asked for and was refused.
    #[serde(default)]
    pub skills_refused: u32,
    /// Facts promoted into project memory under an approval.
    pub memory_promotions: u32,
    /// Memory operations refused. Non-zero is worth a look: it means the run
    /// asked for something policy would not give it.
    pub memory_refusals: u32,
    /// Lifecycle checks that refused something.
    ///
    /// Non-zero means a deployment's own rule stopped this run doing something
    /// the product's rules would have allowed — which is the one case where
    /// reading the run's trace alone would leave somebody puzzled.
    pub hook_blocks: u32,
    /// Bounded workers this run started.
    pub subagents_started: u32,
    pub subagents_finished: u32,
    /// How many of those did not complete — failed, timed out or were stopped.
    /// Kept apart from the total because a fan-out where three of four workers
    /// timed out is a different run from one where all four finished.
    pub subagents_incomplete: u32,
    /// Side effects that were in flight when the process went away.
    /// Non-empty is why a run is `degraded_needs_human`.
    pub unknown_effects: Vec<UnknownEffect>,
    /// Set when the plan stopped the run, in the words shown to the person.
    pub stopped_because: Option<String>,
    /// Set when it ended badly.
    pub failure: Option<String>,
    /// A reference to the answer, not the answer. See the module note.
    pub answer_hash: Option<String>,
    pub answer_chars: usize,
    /// Events that were on disk and could not be read while folding this.
    /// Non-empty means the history has a hole in it, and the screen says so.
    pub unreadable_events: Vec<UnreadableEvent>,
    /// Events that could not legally follow the state they arrived in.
    ///
    /// Recorded in the history and not applied. Surfaced rather than hidden: an
    /// event out of order means two writers disagreed about a run, and that is
    /// worth somebody knowing.
    pub anomalies: Vec<String>,
}

impl TaskSnapshot {
    /// An empty snapshot for a run nothing is yet known about.
    pub fn empty(run_id: &str) -> Self {
        Self {
            run_id: run_id.to_string(),
            seq: 0,
            schema_version: SCHEMA_VERSION,
            state: RunState::Created,
            started_at: String::new(),
            updated_at: String::new(),
            deadline: None,
            actor: String::new(),
            prompt: String::new(),
            model_name: String::new(),
            classification: None,
            plan: None,
            activity: Vec::new(),
            turns: 0,
            history_trim: None,
            compactions: 0,
            checkpoints_taken: 0,
            checkpoint_failures: 0,
            resumptions: 0,
            recovery_attempts: 0,
            recovery_failures: 0,
            completion_verified: None,
            memory_reads: 0,
            skills_loaded: 0,
            skills_refused: 0,
            memory_promotions: 0,
            memory_refusals: 0,
            hook_blocks: 0,
            compaction_events: Vec::new(),
            artifacts: Vec::new(),
            approvals_pending: 0,
            subagents_started: 0,
            subagents_finished: 0,
            subagents_incomplete: 0,
            unknown_effects: Vec::new(),
            stopped_because: None,
            failure: None,
            answer_hash: None,
            answer_chars: 0,
            unreadable_events: Vec::new(),
            anomalies: Vec::new(),
        }
    }

    /// Whether the history behind this snapshot is complete and consistent.
    pub fn is_intact(&self) -> bool {
        self.unreadable_events.is_empty() && self.anomalies.is_empty()
    }

    /// Whether a person has to do something before this run can be relied on.
    pub fn needs_person(&self) -> bool {
        self.state.needs_person() || !self.unknown_effects.is_empty()
    }

    /// Folds one event in, in place.
    ///
    /// Out-of-order and already-seen events are ignored rather than rejected: a
    /// caller catching up after a reconnect will legitimately re-send what it
    /// already had, and making that an error would push the retry logic into
    /// every caller.
    pub fn apply(&mut self, event: &TaskEvent) {
        if event.seq <= self.seq {
            return;
        }
        self.seq = event.seq;
        self.updated_at = event.at.clone();

        // The state moves first, and independently of the payload. An event
        // that cannot legally follow is recorded as an anomaly and its payload
        // is still read: a late `run_routed` should not move the state, but the
        // model name it carries is not thereby wrong.
        match advance(self.state, event.event_type) {
            Transition::To(next) => self.state = next,
            Transition::Stays => {}
            Transition::Illegal { reason } => {
                let note = format!("seq {}: {reason}", event.seq);
                if !self.anomalies.contains(&note) {
                    self.anomalies.push(note);
                }
                // A terminal state is absorbing in both directions: nothing
                // after an ending is read at all, because the ending is the
                // one true account of how the run finished.
                if self.state.is_terminal() {
                    return;
                }
            }
        }

        self.absorb(event);
    }

    /// Reads whatever the payload has to say, without touching the state.
    fn absorb(&mut self, event: &TaskEvent) {
        let text = |key: &str| -> Option<String> {
            event
                .payload
                .get(key)
                .and_then(Value::as_str)
                .map(str::to_string)
        };
        let count = |key: &str| -> Option<u64> { event.payload.get(key).and_then(Value::as_u64) };

        match event.event_type {
            TaskEventType::RunCreated => {
                self.started_at = event.at.clone();
                self.actor = event.actor.clone();
                // `promptShown` rather than `prompt`: the redaction hashes
                // anything called `prompt`, and this is the person's own words
                // being shown back to them on their own machine. A task list
                // where every row reads as a hash identifies nothing.
                if let Some(prompt) = text("promptShown") {
                    self.prompt = prompt;
                }
                if let Some(deadline) = text("deadline") {
                    self.deadline = Some(deadline);
                }
            }
            TaskEventType::RunClassified => self.classification = text("classification"),
            TaskEventType::RunRouted => {
                if let Some(model) = text("modelName") {
                    self.model_name = model;
                }
            }
            TaskEventType::PlanReady => {
                self.plan = event
                    .payload
                    .get("plan")
                    .and_then(|plan| serde_json::from_value::<PlanRecord>(plan.clone()).ok());
            }
            TaskEventType::RunStarted => {
                if let Some(deadline) = text("deadline") {
                    self.deadline = Some(deadline);
                }
            }
            TaskEventType::PlanStep => {
                if let (Some(plan), Some(taken)) = (self.plan.as_mut(), count("stepsTaken")) {
                    // The plan's own count, never one incremented here: an
                    // event that never arrived would leave a local counter
                    // permanently and invisibly wrong.
                    plan.steps_taken = taken as u32;
                }
            }
            TaskEventType::PlanStopped => self.stopped_because = text("reason"),
            // Counted from the events rather than incremented locally, so a
            // recovered trace and a watched one agree.
            TaskEventType::TurnEnded => self.turns += 1,
            TaskEventType::ContextTrimmed => {
                // Last one wins. A run is one chat turn, so there is at most
                // one of these — but a resumed attempt fits the window again
                // against whatever model is serving now, and the later reading
                // is the one that describes what the model actually saw.
                self.history_trim = serde_json::from_value::<
                    super::super::tasks::HistoryTrim,
                >(event.payload.clone())
                .ok();
            }
            TaskEventType::ContextCompacted => {
                self.compactions += 1;
                // The count is incremented whether or not the payload can be
                // read. A compaction that happened is a fact about the run, and
                // an unreadable payload should cost its detail, not its
                // existence.
                if let Ok(mut record) = serde_json::from_value::<
                    super::super::tasks::CompactionRecord,
                >(event.payload.clone())
                {
                    // The envelope's own instant, not one inside the payload:
                    // the payload crossed a process boundary and the envelope
                    // was stamped where the row was written.
                    record.at = event.at.clone();
                    if record.ordinal == 0 {
                        record.ordinal = self.compactions;
                    }
                    self.compaction_events.push(record);
                }
            }

            TaskEventType::ToolAuthorized => self.begin_call(event, "running"),
            TaskEventType::ToolRefused => self.settle_call(event, "refused"),
            TaskEventType::ToolSucceeded => self.settle_call(event, "done"),
            TaskEventType::ToolFailed => self.settle_call(event, "failed"),
            TaskEventType::ToolReplayed => self.settle_call(event, "replayed"),
            TaskEventType::ToolEffectPending => {}
            TaskEventType::ToolEffectUnknown => {
                self.settle_call(event, "unknown");
                let effect = UnknownEffect {
                    idempotency_key: text("idempotencyKey").unwrap_or_default(),
                    tool: text("tool").unwrap_or_else(|| "unknown".to_string()),
                    target: text("target").unwrap_or_default(),
                    at: event.at.clone(),
                };
                if !self.unknown_effects.contains(&effect) {
                    self.unknown_effects.push(effect);
                }
            }
            TaskEventType::ToolEffectReconciled => {
                if let Some(key) = text("idempotencyKey") {
                    self.unknown_effects
                        .retain(|effect| effect.idempotency_key != key);
                }
            }
            TaskEventType::ArtifactProduced => {
                if let Some(name) = text("name") {
                    if !self.artifacts.contains(&name) {
                        self.artifacts.push(name);
                    }
                }
            }

            // Counted, not quoted. The payloads carry hashes and counts by
            // construction (see `memory_api`), and folding a value out of one
            // into a snapshot the Tasks list reads would undo that.
            TaskEventType::CheckpointTaken => self.checkpoints_taken += 1,
            // Counted separately and prominently: a run with failed checkpoints
            // is a run whose resume point is behind where it actually got to.
            TaskEventType::CheckpointFailed => self.checkpoint_failures += 1,
            TaskEventType::RunResumed => self.resumptions += 1,
            // Counted, so a trace can say a run's instructions were added to
            // part-way through. A refusal is counted too: a skill asked for and
            // denied is a thing the run tried to do.
            TaskEventType::SkillLoaded => self.skills_loaded += 1,
            TaskEventType::SkillRefused => self.skills_refused += 1,
            TaskEventType::MemoryRecalled => self.memory_reads += 1,
            TaskEventType::MemoryPromoted => self.memory_promotions += 1,
            TaskEventType::MemoryRefused => self.memory_refusals += 1,
            TaskEventType::MemoryForgotten => {}

            TaskEventType::ApprovalRequested => self.approvals_pending += 1,
            TaskEventType::ApprovalDecided => {
                self.approvals_pending = self.approvals_pending.saturating_sub(1)
            }

            TaskEventType::HookEvaluated => {
                if event.payload.get("blocked").and_then(Value::as_bool) == Some(true) {
                    self.hook_blocks += 1;
                }
            }

            TaskEventType::VerificationStarted => {}
            // What the completion check concluded. Read off the payload rather
            // than inferred from the run reaching an ending: a run can end
            // without ever being checked, and that is not a pass.
            TaskEventType::CompletionVerified => {
                self.completion_verified = event
                    .payload
                    .get("passed")
                    .and_then(Value::as_bool)
                    .or(Some(false));
            }

            // Counted so a recovery loop is visible as one. A run whose
            // attempts climb without an ending is the shape of a run being
            // picked up and dropped repeatedly.
            TaskEventType::RecoveryStarted => self.recovery_attempts += 1,
            TaskEventType::RecoveryFailed => self.recovery_failures += 1,

            // Observations that carry counts and digests rather than state.
            // The turn count already comes from `turn_ended`; adding a second
            // source for it would let the two disagree.
            TaskEventType::ModelRequested
            | TaskEventType::ModelResponded
            // Paired with `context_compacted`, which is what is counted.
            | TaskEventType::CompactionStarted
            | TaskEventType::WaitStarted
            | TaskEventType::WaitCompleted
            | TaskEventType::RunPaused => {}

            TaskEventType::SubagentStarted => self.subagents_started += 1,
            TaskEventType::SubagentStopped => {
                self.subagents_finished += 1;
                // Counted separately from the total, because a fan-out where
                // three of four workers timed out is a very different run from
                // one where all four finished, and a single count cannot say so.
                if event.payload.get("complete").and_then(Value::as_bool) != Some(true) {
                    self.subagents_incomplete += 1;
                }
            }

            TaskEventType::RunCompleted => {
                self.answer_hash = event
                    .payload
                    .get("answer")
                    .and_then(|answer| answer.get("sha256"))
                    .and_then(Value::as_str)
                    .map(str::to_string);
                self.answer_chars = event
                    .payload
                    .get("answer")
                    .and_then(|answer| answer.get("chars"))
                    .and_then(Value::as_u64)
                    .unwrap_or(0) as usize;
                if let Some(turns) = count("turns") {
                    self.turns = turns as u32;
                }
            }
            // Both an answer and a caveat. A run cut off at the output cap
            // produced real text worth keeping — and the one thing a reader
            // must not be allowed to assume about it is that it is whole, so
            // the fragment and the reason are recorded together.
            TaskEventType::RunStoppedByLength => {
                self.answer_hash = event
                    .payload
                    .get("answer")
                    .and_then(|answer| answer.get("sha256"))
                    .and_then(Value::as_str)
                    .map(str::to_string);
                self.answer_chars = event
                    .payload
                    .get("answer")
                    .and_then(|answer| answer.get("chars"))
                    .and_then(Value::as_u64)
                    .unwrap_or(0) as usize;
                if let Some(turns) = count("turns") {
                    self.turns = turns as u32;
                }
                self.failure =
                    Some(text("failure").unwrap_or_else(|| self.state.describe().to_string()));
                self.strand_running_calls();
            }
            TaskEventType::RunFailed
            | TaskEventType::RunCancelled
            | TaskEventType::RunStoppedByBudget
            | TaskEventType::RunStoppedByPolicy
            | TaskEventType::RunDegraded
            | TaskEventType::RunTimedOut
            | TaskEventType::RunInterrupted => {
                self.failure =
                    Some(text("failure").unwrap_or_else(|| self.state.describe().to_string()));
                self.strand_running_calls();
            }
        }
    }

    /// Marks any call still showing as running when the run ended.
    ///
    /// The case this exists for is the abort race: a person presses stop, the
    /// cancellation is recorded, and a tool that was already executing finishes
    /// afterwards — at which point its outcome event is refused, because
    /// nothing may follow an ending. That is the right rule, and it leaves the
    /// call's row saying "running" forever.
    ///
    /// Which would be a lie of the worst kind: it looks like a tool that hung.
    /// The truth is narrower and worth saying exactly — the event history does
    /// not record how this call ended. For a call that touched nothing, that is
    /// the whole story. For one that wrote something, the effect table knows,
    /// and says so separately through `unknown_effects`.
    fn strand_running_calls(&mut self) {
        for call in self.activity.iter_mut().filter(|call| call.status == "running") {
            call.status = "unknown".to_string();
        }
    }

    fn begin_call(&mut self, event: &TaskEvent, status: &str) {
        let Some(id) = event.payload.get("toolCallId").and_then(Value::as_str) else {
            return;
        };
        if self.activity.iter().any(|item| item.tool_call_id == id) {
            return;
        }
        self.activity.push(ActivityRecord {
            tool_call_id: id.to_string(),
            tool: event
                .payload
                .get("tool")
                .and_then(Value::as_str)
                .unwrap_or("unknown")
                .to_string(),
            status: status.to_string(),
            at: event.at.clone(),
        });
    }

    fn settle_call(&mut self, event: &TaskEvent, status: &str) {
        let Some(id) = event.payload.get("toolCallId").and_then(Value::as_str) else {
            return;
        };
        match self.activity.iter_mut().find(|item| item.tool_call_id == id) {
            // A refusal never gets an authorize event, so its outcome is the
            // first thing heard about the call. Recorded rather than dropped:
            // a trace missing every refusal is a trace that says the policy
            // never did anything.
            None => self.begin_call(event, status),
            Some(item) => {
                item.status = status.to_string();
                item.at = event.at.clone();
            }
        }
    }
}

/// Folds a run's events into the state the UI draws.
pub fn fold(run_id: &str, events: &[TaskEvent], unreadable: &[UnreadableEvent]) -> TaskSnapshot {
    let mut snapshot = TaskSnapshot::empty(run_id);
    for event in events {
        snapshot.apply(event);
    }
    snapshot.unreadable_events = unreadable.to_vec();
    // An unreadable event may have been the terminal one. Saying "running" for
    // a run whose ending is the part that will not parse would be the one
    // actively misleading answer available here.
    if let Some(worst) = unreadable.iter().map(|item| item.seq).max() {
        if worst > snapshot.seq {
            snapshot.seq = worst;
        }
    }
    snapshot
}
