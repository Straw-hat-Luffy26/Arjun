//! A plan the model cannot extend, and a run that knows when to stop.
//!
//! ARJUN design rule 19: *"The plan includes a maximum number of steps, maximum execution
//! time, permitted tools, permitted files, model budget, and stop conditions.
//! The model is not allowed to extend the plan indefinitely."*
//!
//! And Part C, on failure behaviour: *"Agent loop repeats → Stop at the
//! step/time/tool budget and show the incomplete plan."*
//!
//! Those two sentences describe the difference between an agent and a runaway
//! process. An agent that cannot stop is not more capable than one that can —
//! it is a machine that will read the same document forty times while somebody
//! watches a spinner, and then produce nothing.
//!
//! ## The budget is set before the model sees anything
//!
//! Every limit here is fixed when the plan is created and is never adjusted by
//! anything the model emits. A model that could raise its own step budget has no
//! budget; a model that could add a tool to its permitted set has no tool
//! policy. Both are asked for by the problem statement precisely because both
//! are the obvious shortcut.
//!
//! ## Stopping is a result, not a failure
//!
//! A run that exhausts its budget returns what it managed, what it did not, and
//! why it stopped. Showing an incomplete plan honestly is far more useful than
//! either pretending it finished or discarding the work.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use thiserror::Error;

/// What can go wrong when the orchestrator or planner mutates a
/// plan. Kept narrow: a step that does not exist is the only
/// mutation that can fail by name. Other invariants are checked by
/// the type system.
#[derive(Debug, Error, PartialEq, Eq, Clone)]
pub enum PlanError {
    #[error("no step with ordinal {ordinal} exists in this plan")]
    NoSuchStep { ordinal: u32 },
}

/// A milestone step that has just been completed. Returned from
/// [`PlanRun::record_step`] so the executor can pause the run for a
/// human gate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MilestoneHit {
    pub ordinal: u32,
    pub checkpoint_id: Option<String>,
    pub intent: String,
}

use serde::{Deserialize, Serialize};

use super::tools::{ToolCall, ToolName};

/// Limits fixed when the plan is made.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Budget {
    /// Most steps this task may take.
    pub max_steps: u32,
    /// Wall-clock ceiling for the whole task.
    #[serde(with = "duration_seconds")]
    pub max_duration: Duration,
    /// Tools this task may use. A tool outside this set is refused even when the
    /// user holds the permission — the plan is narrower than the person.
    pub permitted_tools: Vec<ToolName>,
    /// How many times the same call may repeat before it is treated as a loop.
    pub repeat_limit: u32,
}

mod duration_seconds {
    use serde::{Deserialize, Deserializer, Serializer};
    use std::time::Duration;

    pub fn serialize<S: Serializer>(value: &Duration, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_u64(value.as_secs())
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Duration, D::Error> {
        Ok(Duration::from_secs(u64::deserialize(d)?))
    }
}

impl Budget {
    /// Sensible limits for an ordinary desk task.
    ///
    /// Twelve steps is enough for read → extract → search → compare → calculate
    /// → draft → validate with room to recover from a couple of mistakes, and
    /// few enough that a loop is caught before a person gives up waiting.
    pub fn standard(permitted_tools: Vec<ToolName>) -> Self {
        Self {
            max_steps: 12,
            max_duration: Duration::from_secs(10 * 60),
            permitted_tools,
            repeat_limit: 3,
        }
    }

    pub fn permits(&self, tool: ToolName) -> bool {
        self.permitted_tools.contains(&tool)
    }
}

/// One intended step, written when the plan is made.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PlanStep {
    pub ordinal: u32,
    /// What this step is for, in the user's terms.
    pub intent: String,
    pub done: bool,
    /// If true, finishing this step is a checkpoint. The executor
    /// pauses the run and emits `TaskState::MilestoneReached` so the
    /// UI can ask a person to confirm before the next leg of work
    /// starts. ARJUN calls this an "evidence-anchored decision
    /// point" — the model says "I think we are here" and a human
    /// signs off. The term is ours; PS 26117 does not ask for this.
    #[serde(default)]
    pub milestone: bool,
    /// Stable identifier the parent plan wrote when the plan was
    /// drafted. The UI uses this to address the gate ("approve
    /// milestone `mtn-2`"); the resume path uses it to know which
    /// checkpoint was the last acknowledged one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checkpoint_id: Option<String>,
}

/// Why a run ended.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", tag = "reason")]
pub enum StopReason {
    /// Every step finished.
    Completed,
    /// The step budget ran out.
    StepsExhausted { taken: u32, allowed: u32 },
    /// Every remaining step is held by a call that is already under way.
    ///
    /// Deliberately distinct from [`Self::StepsExhausted`], and deliberately
    /// *not* a halt. The runtime authorises read-only tools in parallel, so a
    /// turn of four searches against three remaining steps has one call arrive
    /// to find the budget fully committed and nothing yet spent. That call is
    /// refused; the run is not ended. If one of the three in flight is then
    /// refused by the gateway its slot comes back, and the model can try again
    /// with what it learned from the other results.
    ///
    /// Ending the run here would mean a model that asked for several things at
    /// once was punished for it, which is the opposite of what running them in
    /// parallel is for.
    StepsInFlight { in_flight: u32, allowed: u32 },
    /// The clock ran out.
    TimeExhausted { allowed_seconds: u64 },
    /// The same call kept coming back — the agent is going in circles.
    Looping { tool: String, repeats: u32 },
    /// Waiting on a person. Not a failure; the run resumes when they answer.
    AwaitingApproval { tool: String },
    /// A person was shown a milestone and declined to continue past it.
    ///
    /// Deliberately distinct from [`Self::Failed`]: nothing went wrong. Somebody
    /// looked at the work so far and decided the next leg should not start, and
    /// a record calling that a failure would report a working control as a
    /// defect. Everything completed before the gate stands.
    MilestoneRejected {
        checkpoint_id: String,
        /// The step's own words, copied in so the record says what was declined
        /// without re-reading the plan.
        intent: String,
    },
    /// A step failed and the plan cannot continue past it.
    Failed { detail: String },
}

impl StopReason {
    /// Whether the task got where it was going.
    pub fn is_success(&self) -> bool {
        matches!(self, StopReason::Completed)
    }

    /// What to tell the person, phrased so an incomplete run is legible rather
    /// than alarming.
    ///
    /// Always a complete sentence. Several variants embed a detail written
    /// elsewhere, which may or may not be punctuated, and a status line that
    /// sometimes trails off reads as though the message itself was truncated.
    pub fn explain(&self) -> String {
        let mut text = self.body();
        if !text.ends_with('.') && !text.ends_with('!') && !text.ends_with('?') {
            text.push('.');
        }
        text
    }

    fn body(&self) -> String {
        match self {
            StopReason::Completed => "Finished.".to_string(),
            StopReason::StepsExhausted { taken, allowed } => format!(
                "Stopped after {taken} of {allowed} permitted steps. The work below is what was \
                 completed; the remaining steps were not attempted."
            ),
            StopReason::StepsInFlight { in_flight, allowed } => format!(
                "Not started: {in_flight} of {allowed} permitted steps are already under way, so                  there was no room for this call. Wait for the results you asked for and decide                  what to do with them."
            ),
            StopReason::TimeExhausted { allowed_seconds } => format!(
                "Stopped after {} minutes, the time allowed for one task. The work below is what \
                 was completed.",
                allowed_seconds / 60
            ),
            StopReason::Looping { tool, repeats } => format!(
                "Stopped: the same {tool} call was attempted {repeats} times without progress, so \
                 the task was going in circles rather than getting closer to an answer."
            ),
            StopReason::AwaitingApproval { tool } => {
                format!("Waiting for you to approve the request to {tool}.")
            }
            StopReason::MilestoneRejected { intent, .. } => format!(
                "Stopped at your decision not to continue past \"{intent}\". The work completed \
                 before that point is below and has been kept."
            ),
            StopReason::Failed { detail } => format!("Stopped: {detail}"),
        }
    }
}

/// A plan being carried out.
pub struct PlanRun {
    pub task_id: String,
    pub steps: Vec<PlanStep>,
    pub budget: Budget,
    started_at: Instant,
    steps_taken: u32,
    /// How many times each distinct call has been seen.
    seen: HashMap<String, u32>,
    stopped: Option<StopReason>,
    /// Slots held by calls that have been authorised and have not yet settled.
    ///
    /// Keyed by the tool-call id the runtime supplied, which is unique per call
    /// and is what makes settling and releasing address one lease rather than
    /// "the most recent". See [`PlanRun::reserve_at`].
    leases: HashMap<String, Lease>,
}

/// Application-owned execution counters. Grants/Instant values never survive a
/// restart. Unsettled reservations are charged conservatively on restoration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PlanProgress {
    pub task_id: String,
    pub budget: Budget,
    pub steps: Vec<PlanStep>,
    pub elapsed_millis: u64,
    pub steps_taken: u32,
    pub seen: HashMap<String, u32>,
    pub stopped: Option<StopReason>,
    pub reserved_calls: u32,
}

/// A slot in the budget, held between authorisation and settlement.
#[derive(Debug, Clone)]
struct Lease {
    /// When it was taken, so an abandoned one can be reclaimed.
    at: Instant,
    /// The tool it was taken for. Kept for the diagnostic, not for the
    /// decision — the id is what identifies the lease.
    tool: String,
}

/// How long an authorised call may hold its slot before it is reclaimed.
///
/// A lease is normally settled or released within milliseconds of the tool
/// finishing. It is not settled at all when the loop authorised a call and then
/// never ran it — the model changed its mind between the batch being approved
/// and the batch being executed, the turn was interrupted, the process
/// stumbled. Nothing tells this side that happened.
///
/// Without a ceiling those slots would be held for the life of the run, and a
/// run that abandoned three calls would quietly lose three steps of budget it
/// never spent. Five minutes is comfortably longer than the longest tool
/// timeout in `orchestrator::tools` (two minutes, for producing a document) and
/// far shorter than the plan's own ten-minute ceiling, so a reclaimed lease is
/// always one nothing is waiting on.
const LEASE_TTL: Duration = Duration::from_secs(5 * 60);

/// Whether the run may take another step, and if not, why.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Continuation {
    Proceed,
    Stop(StopReason),
}

impl PlanRun {
    pub fn checkpoint_progress(&self) -> PlanProgress {
        PlanProgress {
            task_id: self.task_id.clone(), budget: self.budget.clone(), steps: self.steps.clone(),
            elapsed_millis: self.started_at.elapsed().as_millis().min(u64::MAX as u128) as u64,
            steps_taken: self.steps_taken, seen: self.seen.clone(), stopped: self.stopped.clone(),
            reserved_calls: self.leases.len().min(u32::MAX as usize) as u32,
        }
    }

    pub fn restore_progress(&mut self, saved: &PlanProgress) -> Result<(), String> {
        if saved.task_id != self.task_id || serde_json::to_value(&saved.budget).ok() != serde_json::to_value(&self.budget).ok()
            || saved.steps.len() != self.steps.len()
            || saved.steps.iter().zip(&self.steps).any(|(old, new)| old.ordinal != new.ordinal || old.intent != new.intent)
            || saved.steps_taken.saturating_add(saved.reserved_calls) > self.budget.max_steps
            || saved.elapsed_millis > self.budget.max_duration.as_millis().min(u64::MAX as u128) as u64 {
            return Err("The saved plan no longer matches its authority or has exhausted its budget.".into());
        }
        self.started_at = Instant::now().checked_sub(Duration::from_millis(saved.elapsed_millis))
            .ok_or_else(|| "The saved execution clock cannot be restored.".to_string())?;
        self.steps = saved.steps.clone();
        self.steps_taken = saved.steps_taken.saturating_add(saved.reserved_calls);
        self.seen = saved.seen.clone();
        self.stopped = saved.stopped.clone();
        self.leases.clear();
        Ok(())
    }

    pub fn new(task_id: impl Into<String>, steps: Vec<String>, budget: Budget) -> Self {
        Self {
            task_id: task_id.into(),
            steps: steps
                .into_iter()
                .enumerate()
                .map(|(i, intent)| PlanStep {
                    ordinal: i as u32 + 1,
                    intent,
                    done: false,
                    milestone: false,
                    checkpoint_id: None,
                })
                .collect(),
            budget,
            started_at: Instant::now(),
            steps_taken: 0,
            seen: HashMap::new(),
            stopped: None,
            leases: HashMap::new(),
        }
    }

    /// Marks an existing step as a milestone checkpoint.
    ///
    /// Call this when the plan is first drafted, before the run
    /// starts. Marking a step *during* a run is also supported, in
    /// case the model discovers partway through that the next leg of
    /// work is a decision the user should make. The change is local
    /// to this `PlanRun`; persistence is the caller's job.
    pub fn mark_milestone(
        &mut self,
        ordinal: u32,
        checkpoint_id: impl Into<String>,
    ) -> Result<(), PlanError> {
        let step = self
            .steps
            .iter_mut()
            .find(|s| s.ordinal == ordinal)
            .ok_or(PlanError::NoSuchStep { ordinal })?;
        step.milestone = true;
        step.checkpoint_id = Some(checkpoint_id.into());
        Ok(())
    }

    /// Returns the checkpoint id of the most recently completed
    /// milestone, if any. Used by the resume path to know which gate
    /// the human has already approved.
    pub fn last_acknowledged_checkpoint(&self) -> Option<&str> {
        self.steps
            .iter()
            .filter(|s| s.done && s.milestone)
            .filter_map(|s| s.checkpoint_id.as_deref())
            .next_back()
    }

    /// For tests and for resuming a run that waited on a person.
    pub fn started_at(&mut self, when: Instant) {
        self.started_at = when;
    }

    pub fn steps_taken(&self) -> u32 {
        self.steps_taken
    }

    pub fn stopped(&self) -> Option<&StopReason> {
        self.stopped.as_ref()
    }

    /// Steps that were planned but never reached.
    ///
    /// Shown rather than hidden: a person who can see what was skipped can
    /// decide whether the partial answer is usable, and one who cannot has to
    /// assume the worst.
    pub fn unfinished(&self) -> Vec<&PlanStep> {
        self.steps.iter().filter(|s| !s.done).collect()
    }

    /// Steps committed to: spent, plus reserved by calls still in flight.
    ///
    /// The figure the budget is judged against. `steps_taken` alone is what a
    /// concurrent batch could all read before any of them had settled.
    pub fn steps_committed(&self) -> u32 {
        self.steps_taken
            .saturating_add(self.leases.len() as u32)
    }

    /// How many authorised calls are holding a slot right now.
    pub fn leases_held(&self) -> usize {
        self.leases.len()
    }

    /// The tools of the calls currently holding a slot, sorted.
    ///
    /// What a person watching a parallel batch wants when the step counter
    /// jumps: which calls are in flight, not merely how many. Sorted so the
    /// same set reads the same way twice.
    pub fn leases_outstanding(&self) -> Vec<String> {
        let mut tools: Vec<String> = self
            .leases
            .values()
            .map(|lease| lease.tool.clone())
            .collect();
        tools.sort();
        tools
    }

    /// Takes a slot in the budget for a call that is about to be authorised.
    ///
    /// ## Why reservation and not a check
    ///
    /// [`Self::may_call`] only *asked* whether there was room, and the slot was
    /// not spent until the tool had finished running. The runtime executes
    /// read-only tools in parallel, so a turn with four searches put four
    /// authorisation requests through this while `steps_taken` was unchanged:
    /// all four read the same figure, all four were told there was room, all
    /// four were granted, and the budget ended three steps over. The ceiling
    /// that exists to stop a model going in circles could be passed by a model
    /// that simply asked for several things at once.
    ///
    /// So the slot is taken here, under the same lock as the check, before any
    /// grant is issued. Everything after this either settles the lease
    /// ([`Self::settle`]) or gives it back ([`Self::release`]).
    ///
    /// `now` is passed in so the reclaim of abandoned leases can be driven in a
    /// test without sleeping; [`Self::reserve`] supplies the wall clock.
    pub fn reserve_at(
        &mut self,
        tool_call_id: &str,
        call: &ToolCall,
        now: Instant,
    ) -> Continuation {
        // Slots held by calls that were authorised and never run. Reclaimed
        // before the ceiling is judged, so an abandoned batch costs the run
        // nothing beyond the wait. See [`LEASE_TTL`].
        self.leases
            .retain(|_, lease| now.duration_since(lease.at) < LEASE_TTL);

        // A lease is per call. A second authorisation for the same tool-call id
        // is the same call being asked about twice — a retry after an ambiguous
        // failure, say — and must not take a second slot.
        if let Some(existing) = self.leases.get_mut(tool_call_id) {
            existing.at = now;
            return Continuation::Proceed;
        }

        match self.admits(call) {
            Continuation::Proceed => {
                self.leases.insert(
                    tool_call_id.to_string(),
                    Lease {
                        at: now,
                        tool: call.tool.clone(),
                    },
                );
                Continuation::Proceed
            }
            stop => stop,
        }
    }

    /// [`Self::reserve_at`] against the wall clock.
    pub fn reserve(&mut self, tool_call_id: &str, call: &ToolCall) -> Continuation {
        self.reserve_at(tool_call_id, call, Instant::now())
    }

    /// Gives a reserved slot back, for a call that will not run.
    ///
    /// The gateway refused it after the plan admitted it, a person declined the
    /// approval, the grant was never redeemed. None of those spent anything, so
    /// none of them should cost a step — a budget that charged for refused
    /// calls would let a policy the model kept running into exhaust the run.
    ///
    /// Returns whether a lease was actually held. Releasing twice is harmless
    /// and answers `false` the second time.
    pub fn release(&mut self, tool_call_id: &str) -> bool {
        self.leases.remove(tool_call_id).is_some()
    }

    /// Turns a reserved slot into a spent step.
    ///
    /// Returns whether this call settled *now*. A second settlement for the
    /// same tool-call id answers `false` and changes nothing, which is what
    /// makes "a completed call cannot be counted twice" a property of this type
    /// rather than a rule every call site has to remember. It is also what
    /// stops a replayed effect and its original both being charged.
    pub fn settle(&mut self, tool_call_id: &str) -> bool {
        if self.leases.remove(tool_call_id).is_none() {
            return false;
        }
        self.steps_taken = self.steps_taken.saturating_add(1);
        true
    }

    /// Checks whether the run may make this call.
    ///
    /// Kept for callers that drive a plan a step at a time and settle
    /// immediately — the checklist-style path in this module's own tests. The
    /// agent path goes through [`Self::reserve_at`], which asks the same
    /// questions and takes the slot while it still holds the answer.
    pub fn may_call(&mut self, call: &ToolCall) -> Continuation {
        self.admits(call)
    }

    /// The questions themselves, asked once and used by both callers.
    fn admits(&mut self, call: &ToolCall) -> Continuation {
        if let Some(reason) = &self.stopped {
            return Continuation::Stop(reason.clone());
        }

        // Time first: an overrunning task should stop even if it has steps left.
        if self.started_at.elapsed() >= self.budget.max_duration {
            return self.halt(StopReason::TimeExhausted {
                allowed_seconds: self.budget.max_duration.as_secs(),
            });
        }

        // Genuinely spent: the budget is gone and the run ends here.
        if self.steps_taken >= self.budget.max_steps {
            return self.halt(StopReason::StepsExhausted {
                taken: self.steps_taken,
                allowed: self.budget.max_steps,
            });
        }

        // Spent *or reserved*. A batch of four searches authorised together
        // holds four slots, and the fourth must see the first three — but this
        // is contention, not exhaustion, so the call is refused and the run
        // carries on. A slot released by a refused call becomes available
        // again, and halting here would have ended the run over a queue.
        if self.steps_committed() >= self.budget.max_steps {
            return Continuation::Stop(StopReason::StepsInFlight {
                in_flight: self.leases.len() as u32,
                allowed: self.budget.max_steps,
            });
        }

        // A tool outside the plan is refused even when the person could use it
        // elsewhere. The plan is narrower than the permission, deliberately.
        let Some(tool) = ToolName::from_str(&call.tool) else {
            return self.halt(StopReason::Failed {
                detail: format!("the model asked for a tool that does not exist: {:?}", call.tool),
            });
        };

        if !self.budget.permits(tool) {
            return self.halt(StopReason::Failed {
                detail: format!(
                    "{} is not among the tools this task was allowed to use",
                    tool.as_str()
                ),
            });
        }

        // Loop detection. Keyed on the whole call, so re-reading a *different*
        // file is progress while re-reading the same one is not.
        let fingerprint = format!("{}::{}", call.tool, call.arguments);
        let repeats = self.seen.entry(fingerprint).or_insert(0);
        *repeats += 1;
        if *repeats > self.budget.repeat_limit {
            let repeats = *repeats;
            return self.halt(StopReason::Looping {
                tool: tool.as_str().to_string(),
                repeats,
            });
        }

        Continuation::Proceed
    }

    /// Records that a step ran.
    ///
    /// Returns the checkpoint id of any milestone that was just
    /// completed, so the executor can pause for a human. The step is
    /// always marked done; the milestone flag is a separate bit on
    /// the same step.
    pub fn record_step(&mut self) -> Option<MilestoneHit> {
        self.steps_taken += 1;
        let hit = self
            .steps
            .iter_mut()
            .find(|s| !s.done)
            .and_then(|s| {
                s.done = true;
                if s.milestone {
                    Some(MilestoneHit {
                        ordinal: s.ordinal,
                        checkpoint_id: s.checkpoint_id.clone(),
                        intent: s.intent.clone(),
                    })
                } else {
                    None
                }
            });
        hit
    }

    /// Records that a tool call was spent, without claiming a step is finished.
    ///
    /// The distinction matters wherever one planned step takes more than one
    /// call. [`Self::record_step`] advances the checklist on every call, which
    /// is right when the caller drives the plan a step at a time. On the agent
    /// path a model may search four times to satisfy one step, and ticking four
    /// steps off would tell an operator the document had been produced and
    /// checked when nothing of the sort had happened.
    ///
    /// So the budget is spent either way, and only the claim differs.
    pub fn record_call(&mut self) {
        self.steps_taken += 1;
    }

    /// Ends the run because a person has to answer.
    ///
    /// Not a failure — the budget is preserved so the run continues from here
    /// once they do.
    pub fn await_approval(&mut self, tool: ToolName) -> StopReason {
        let reason = StopReason::AwaitingApproval {
            tool: tool.describe().to_string(),
        };
        self.stopped = Some(reason.clone());
        reason
    }

    /// Clears a pause so the run can carry on after an approval.
    pub fn resume(&mut self) {
        if matches!(self.stopped, Some(StopReason::AwaitingApproval { .. })) {
            self.stopped = None;
        }
    }

    /// Ends the run because a person declined a milestone.
    ///
    /// The steps already marked done stay done: this is a decision to stop
    /// here, not a reason to discard what the run had produced. Returns the
    /// reason so the caller can record and emit the same value the plan now
    /// holds, rather than reconstructing it.
    ///
    /// `None` when this plan has no step by that checkpoint id — a gate the
    /// plan does not have is a request to refuse rather than to guess at.
    pub fn reject_milestone(&mut self, checkpoint_id: &str) -> Option<StopReason> {
        let step = self
            .steps
            .iter()
            .find(|step| step.checkpoint_id.as_deref() == Some(checkpoint_id))?;
        let reason = StopReason::MilestoneRejected {
            checkpoint_id: checkpoint_id.to_string(),
            intent: step.intent.clone(),
        };
        self.stopped = Some(reason.clone());
        Some(reason)
    }

    /// The step carrying a checkpoint id, if the plan has one.
    ///
    /// Used by the acknowledgement path to attribute a decision to the step it
    /// was actually about — the ordinal and the intent both go into the durable
    /// record, and neither can be invented.
    pub fn step_at_checkpoint(&self, checkpoint_id: &str) -> Option<&PlanStep> {
        self.steps
            .iter()
            .find(|step| step.checkpoint_id.as_deref() == Some(checkpoint_id))
    }

    pub fn complete(&mut self) -> StopReason {
        for step in &mut self.steps {
            step.done = true;
        }
        let reason = StopReason::Completed;
        self.stopped = Some(reason.clone());
        reason
    }

    fn halt(&mut self, reason: StopReason) -> Continuation {
        self.stopped = Some(reason.clone());
        Continuation::Stop(reason)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn tools() -> Vec<ToolName> {
        vec![
            ToolName::SearchDocuments,
            ToolName::ReadScopedFile,
            ToolName::RunCalculation,
        ]
    }

    fn run() -> PlanRun {
        PlanRun::new(
            "task-1",
            vec![
                "Read the inspection report".into(),
                "Find the relevant SOP".into(),
                "Calculate the deviation".into(),
            ],
            Budget::standard(tools()),
        )
    }

    fn search(query: &str) -> ToolCall {
        ToolCall::new("search_documents", json!({ "query": query }))
    }

    #[test]
    fn a_run_within_its_budget_proceeds() {
        let mut run = run();
        assert_eq!(run.may_call(&search("wall thickness")), Continuation::Proceed);
    }

    #[test]
    fn the_step_budget_stops_the_run_and_says_how_far_it_got() {
        let mut run = PlanRun::new(
            "task-1",
            vec!["one".into()],
            Budget {
                max_steps: 2,
                ..Budget::standard(tools())
            },
        );

        for i in 0..2 {
            assert_eq!(run.may_call(&search(&format!("q{i}"))), Continuation::Proceed);
            run.record_step();
        }

        match run.may_call(&search("q3")) {
            Continuation::Stop(StopReason::StepsExhausted { taken, allowed }) => {
                assert_eq!((taken, allowed), (2, 2));
            }
            other => panic!("expected the step budget to stop it, got {other:?}"),
        }
    }

    #[test]
    fn the_time_budget_stops_the_run_even_with_steps_left() {
        let mut run = PlanRun::new(
            "task-1",
            vec!["one".into()],
            Budget {
                max_duration: Duration::from_secs(60),
                ..Budget::standard(tools())
            },
        );
        // Pretend the task started well over an hour ago.
        run.started_at(Instant::now() - Duration::from_secs(3700));

        assert!(matches!(
            run.may_call(&search("anything")),
            Continuation::Stop(StopReason::TimeExhausted { .. })
        ));
    }

    /// The plan is narrower than the person: a tool the user could use elsewhere
    /// is still refused if this task was not given it.
    #[test]
    fn a_tool_outside_the_plan_is_refused_even_when_the_user_holds_it() {
        let mut run = run();
        let call = ToolCall::new(
            "execute_code",
            json!({ "language": "python", "source": "print(1)" }),
        );

        match run.may_call(&call) {
            Continuation::Stop(StopReason::Failed { detail }) => {
                assert!(detail.contains("not among the tools"), "{detail}");
            }
            other => panic!("expected a refusal, got {other:?}"),
        }
    }

    #[test]
    fn a_tool_that_does_not_exist_stops_the_run() {
        let mut run = run();
        let call = ToolCall::new("delete_everything", json!({}));
        assert!(matches!(
            run.may_call(&call),
            Continuation::Stop(StopReason::Failed { .. })
        ));
    }

    // ── Loops ────────────────────────────────────────────────────────────

    /// The failure this exists to catch: an agent reading the same document
    /// forty times while somebody watches a spinner.
    #[test]
    fn repeating_the_same_call_is_detected_as_a_loop() {
        let mut run = run();
        let call = search("wall thickness");

        for _ in 0..3 {
            assert_eq!(run.may_call(&call), Continuation::Proceed);
        }

        match run.may_call(&call) {
            Continuation::Stop(StopReason::Looping { tool, repeats }) => {
                assert_eq!(tool, "knowledge.search_authorized");
                assert_eq!(repeats, 4);
            }
            other => panic!("expected a loop to be caught, got {other:?}"),
        }
    }

    /// Re-reading a *different* file is progress; only the identical call is not.
    #[test]
    fn different_calls_to_the_same_tool_are_not_a_loop() {
        let mut run = run();
        for i in 0..8 {
            assert_eq!(
                run.may_call(&search(&format!("query {i}"))),
                Continuation::Proceed,
                "distinct queries should not look like a loop"
            );
        }
    }

    // ── Stopping honestly ────────────────────────────────────────────────

    #[test]
    fn an_incomplete_run_reports_the_steps_it_never_reached() {
        let mut run = run();
        run.record_step();

        let unfinished = run.unfinished();
        assert_eq!(unfinished.len(), 2);
        assert_eq!(unfinished[0].intent, "Find the relevant SOP");
    }

    #[test]
    fn a_completed_run_has_nothing_unfinished() {
        let mut run = run();
        assert!(run.complete().is_success());
        assert!(run.unfinished().is_empty());
    }

    #[test]
    fn every_stop_reason_explains_itself_in_plain_words() {
        let reasons = [
            StopReason::Completed,
            StopReason::StepsExhausted { taken: 12, allowed: 12 },
            StopReason::TimeExhausted { allowed_seconds: 600 },
            StopReason::Looping { tool: "search_documents".into(), repeats: 4 },
            StopReason::AwaitingApproval { tool: "write a file".into() },
            StopReason::Failed { detail: "the sandbox refused".into() },
        ];

        for reason in reasons {
            let text = reason.explain();
            assert!(!text.is_empty());
            assert!(text.ends_with('.'), "{text:?} should read as a sentence");
        }
    }

    #[test]
    fn the_step_exhausted_message_tells_a_person_what_they_have() {
        let text = StopReason::StepsExhausted { taken: 12, allowed: 12 }.explain();
        assert!(text.contains("what was completed"));
        assert!(text.contains("not attempted"));
    }

    // ── Approval pauses rather than fails ────────────────────────────────

    #[test]
    fn waiting_for_approval_is_not_a_failure_and_the_run_resumes() {
        let mut run = run();
        let reason = run.await_approval(ToolName::WriteScopedFile);

        assert!(!reason.is_success());
        assert!(matches!(
            run.may_call(&search("anything")),
            Continuation::Stop(StopReason::AwaitingApproval { .. })
        ));

        run.resume();
        assert_eq!(run.may_call(&search("anything")), Continuation::Proceed);
    }

    /// Resuming must not clear a stop that was not an approval pause.
    #[test]
    fn resuming_does_not_revive_a_run_that_ran_out_of_steps() {
        let mut run = PlanRun::new(
            "task-1",
            vec!["one".into()],
            Budget { max_steps: 0, ..Budget::standard(tools()) },
        );
        assert!(matches!(run.may_call(&search("q")), Continuation::Stop(_)));

        run.resume();
        assert!(
            matches!(run.may_call(&search("q")), Continuation::Stop(_)),
            "an exhausted run must stay stopped"
        );
    }

    /// Nothing the model emits may widen the budget.
    #[test]
    fn the_budget_is_fixed_when_the_plan_is_made() {
        let budget = Budget::standard(tools());
        let run = PlanRun::new("task-1", vec!["one".into()], budget.clone());

        // The only way to change limits is to build a different plan.
        assert_eq!(run.budget.max_steps, budget.max_steps);
        assert_eq!(run.budget.permitted_tools, budget.permitted_tools);
    }

    // ── Milestone checkpoints ────────────────────────────────────────

    #[test]
    fn marking_a_step_as_milestone_stores_the_checkpoint_id() {
        let mut run = run();
        run.mark_milestone(2, "mtn-sop-look-up").unwrap();

        let step = run.steps.iter().find(|s| s.ordinal == 2).unwrap();
        assert!(step.milestone);
        assert_eq!(step.checkpoint_id.as_deref(), Some("mtn-sop-look-up"));
    }

    #[test]
    fn marking_an_unknown_step_is_an_error_not_a_panic() {
        let mut run = run();
        let err = run.mark_milestone(99, "nope").unwrap_err();
        assert!(matches!(err, PlanError::NoSuchStep { ordinal: 99 }));
    }

    #[test]
    fn completing_a_milestone_records_the_hit_with_its_intent() {
        let mut run = run();
        run.mark_milestone(1, "mtn-survey").unwrap();

        let hit = run.record_step();
        assert_eq!(
            hit,
            Some(MilestoneHit {
                ordinal: 1,
                checkpoint_id: Some("mtn-survey".to_string()),
                intent: "Read the inspection report".to_string(),
            }),
        );
    }

    #[test]
    fn completing_a_non_milestone_records_no_hit() {
        let mut run = run();
        let hit = run.record_step();
        assert!(hit.is_none());
    }

    #[test]
    fn last_acknowledged_checkpoint_returns_the_most_recent_done_milestone() {
        let mut run = run();
        run.mark_milestone(1, "mtn-1").unwrap();
        run.mark_milestone(3, "mtn-3").unwrap();

        // First step done is a milestone.
        let hit1 = run.record_step();
        assert_eq!(hit1.unwrap().checkpoint_id.as_deref(), Some("mtn-1"));
        // Second step is not a milestone.
        let hit2 = run.record_step();
        assert!(hit2.is_none());
        // Third step done is also a milestone.
        let hit3 = run.record_step();
        assert_eq!(hit3.unwrap().checkpoint_id.as_deref(), Some("mtn-3"));

        assert_eq!(run.last_acknowledged_checkpoint(), Some("mtn-3"));
    }

    /// A person who declines a gate stops the run. What was already finished
    /// stays finished: this is a decision to go no further, not a reason to
    /// throw away the work that led to the decision.
    #[test]
    fn rejecting_a_milestone_stops_the_run_and_keeps_the_completed_work() {
        let mut run = run();
        run.mark_milestone(1, "mtn-1").unwrap();
        run.record_step();

        let reason = run.reject_milestone("mtn-1").expect("the plan has that gate");

        assert!(matches!(
            reason,
            StopReason::MilestoneRejected { ref checkpoint_id, .. } if checkpoint_id == "mtn-1"
        ));
        assert!(matches!(
            run.stopped(),
            Some(StopReason::MilestoneRejected { .. })
        ));
        assert!(
            run.steps.iter().any(|step| step.done),
            "the step finished before the gate has to survive the rejection"
        );
        assert!(
            !reason.is_success(),
            "declining to continue is not a completed task"
        );
    }

    /// Not a failure. A record that called a person's decision a defect would
    /// misreport a working control.
    #[test]
    fn a_rejected_milestone_reads_as_a_decision_not_a_fault() {
        let mut run = run();
        run.mark_milestone(1, "mtn-1").unwrap();
        run.record_step();
        let reason = run.reject_milestone("mtn-1").unwrap();

        let sentence = reason.explain();
        assert!(sentence.ends_with('.'), "{sentence}");
        assert!(
            !sentence.to_lowercase().contains("failed"),
            "a decision must not be reported as a failure: {sentence}"
        );
    }

    /// A gate the plan does not have is refused rather than guessed at — the
    /// ordinal and intent it would record cannot be invented.
    #[test]
    fn rejecting_a_checkpoint_the_plan_does_not_have_is_refused() {
        let mut run = run();
        run.mark_milestone(1, "mtn-1").unwrap();

        assert!(run.reject_milestone("mtn-nope").is_none());
        assert!(run.stopped().is_none(), "a refused request changes nothing");
    }

    #[test]
    fn a_checkpoint_id_resolves_to_the_step_that_carries_it() {
        let mut run = run();
        run.mark_milestone(2, "mtn-2").unwrap();

        let step = run.step_at_checkpoint("mtn-2").expect("registered");
        assert_eq!(step.ordinal, 2);
        assert!(run.step_at_checkpoint("mtn-1").is_none());
    }

    #[test]
    fn last_acknowledged_checkpoint_is_none_before_anything_completes() {
        let run = run();
        assert_eq!(run.last_acknowledged_checkpoint(), None);
    }
}

/// The budget under a batch of calls that arrive together.
///
/// ## The defect
///
/// `may_call` only *asked* whether there was room; the slot was not spent until
/// the tool had finished running. The agent runtime executes read-only tools in
/// parallel — a document task typically wants several searches at once — so a
/// turn with four searches put four authorisations through the check while
/// `steps_taken` was unchanged. All four read the same figure, all four were
/// told there was room, all four were granted, and the run ended three steps
/// past its ceiling.
///
/// The ceiling exists to stop a model going in circles. It could be passed by a
/// model that simply asked for several things at once.
#[cfg(test)]
mod reservations {
    use super::*;

    fn call(tool: &str, query: &str) -> ToolCall {
        ToolCall::new(tool, serde_json::json!({ "query": query }))
    }

    /// A plan with `max_steps` to spend and every tool permitted.
    fn plan_with(max_steps: u32) -> PlanRun {
        PlanRun::new(
            "run-1",
            vec!["do the work".to_string()],
            Budget {
                max_steps,
                max_duration: Duration::from_secs(600),
                permitted_tools: ToolName::ALL.to_vec(),
                // High enough that loop detection cannot be what refuses these;
                // each test below is about the step ceiling alone.
                repeat_limit: 100,
            },
        )
    }

    #[test]
    fn several_calls_contending_for_one_slot_admit_exactly_one() {
        // The whole defect, in one assertion. Four calls arrive before any of
        // them has settled, and there is room for one.
        let mut plan = plan_with(1);
        let admitted = ["tc-1", "tc-2", "tc-3", "tc-4"]
            .into_iter()
            .filter(|id| {
                plan.reserve(id, &call("search_documents", "seal specification"))
                    == Continuation::Proceed
            })
            .count();
        assert_eq!(admitted, 1, "the ceiling was passed by asking all at once");
        assert_eq!(plan.leases_held(), 1);
        assert_eq!(plan.steps_committed(), 1);
    }

    #[test]
    fn a_parallel_batch_cannot_commit_more_than_the_budget_allows() {
        // The realistic shape: a turn of four searches against a budget with
        // three steps left.
        let mut plan = plan_with(3);
        let admitted = (0..4)
            .filter(|i| {
                plan.reserve(
                    &format!("tc-{i}"),
                    &call("search_documents", &format!("query {i}")),
                ) == Continuation::Proceed
            })
            .count();
        assert_eq!(admitted, 3);
        assert_eq!(plan.steps_committed(), 3);
        // And settling them moves the accounting from reserved to spent
        // without changing the total.
        for i in 0..3 {
            assert!(plan.settle(&format!("tc-{i}")));
        }
        assert_eq!(plan.steps_taken(), 3);
        assert_eq!(plan.leases_held(), 0);
        assert_eq!(plan.steps_committed(), 3);
    }

    #[test]
    fn a_slot_is_taken_before_any_grant_could_be_issued() {
        // Stated as a property of the ordering rather than of the count: the
        // second call sees the first one's slot, even though nothing has run.
        let mut plan = plan_with(2);
        assert_eq!(
            plan.reserve("tc-1", &call("search_documents", "a")),
            Continuation::Proceed
        );
        assert_eq!(
            plan.steps_taken(),
            0,
            "nothing has run, so nothing is spent yet"
        );
        assert_eq!(
            plan.steps_committed(),
            1,
            "but the slot is committed, which is what the next call must see"
        );
    }

    #[test]
    fn a_refused_call_gives_its_slot_back() {
        // The gateway refused it after the plan admitted it. Nothing ran, so
        // nothing should be charged — a budget that paid for refused calls
        // would let a policy the model kept running into exhaust the run.
        let mut plan = plan_with(1);
        assert_eq!(
            plan.reserve("tc-1", &call("search_documents", "a")),
            Continuation::Proceed
        );
        assert!(plan.release("tc-1"));
        assert_eq!(plan.steps_committed(), 0);
        // And the slot is genuinely available again.
        assert_eq!(
            plan.reserve("tc-2", &call("search_documents", "b")),
            Continuation::Proceed
        );
    }

    #[test]
    fn releasing_twice_is_harmless_and_frees_nothing_extra() {
        let mut plan = plan_with(2);
        plan.reserve("tc-1", &call("search_documents", "a"));
        assert!(plan.release("tc-1"));
        assert!(!plan.release("tc-1"), "there was nothing left to release");
        assert_eq!(plan.steps_committed(), 0);
    }

    #[test]
    fn a_completed_call_cannot_be_charged_twice() {
        // Structural rather than a rule each call site has to remember: the
        // second settlement finds no lease and changes nothing.
        let mut plan = plan_with(5);
        plan.reserve("tc-1", &call("search_documents", "a"));
        assert!(plan.settle("tc-1"));
        assert_eq!(plan.steps_taken(), 1);
        assert!(!plan.settle("tc-1"), "the same call settled twice");
        assert_eq!(plan.steps_taken(), 1);
    }

    #[test]
    fn a_call_that_never_reserved_a_slot_is_not_charged_to_the_budget() {
        // A grant issued outside the plan should not be able to spend the
        // plan's budget by settling against it.
        let mut plan = plan_with(5);
        assert!(!plan.settle("tc-never-authorised"));
        assert_eq!(plan.steps_taken(), 0);
    }

    #[test]
    fn asking_twice_about_one_call_does_not_take_two_slots() {
        // A retry after an ambiguous failure presents the same tool-call id.
        // It is the same call, and it holds one slot.
        let mut plan = plan_with(2);
        assert_eq!(
            plan.reserve("tc-1", &call("search_documents", "a")),
            Continuation::Proceed
        );
        assert_eq!(
            plan.reserve("tc-1", &call("search_documents", "a")),
            Continuation::Proceed
        );
        assert_eq!(plan.leases_held(), 1);
        assert_eq!(plan.steps_committed(), 1);
    }

    #[test]
    fn an_abandoned_reservation_is_reclaimed_rather_than_held_forever() {
        // The loop authorised a call and never ran it — the model changed its
        // mind between the batch being approved and the batch being executed.
        // Nothing tells this side that happened, so the slot is reclaimed on
        // age rather than held for the life of the run.
        let start = Instant::now();
        let mut plan = plan_with(1);
        assert_eq!(
            plan.reserve_at("tc-abandoned", &call("search_documents", "a"), start),
            Continuation::Proceed
        );
        // Still inside the lease's life: the slot is genuinely taken.
        let soon = start + Duration::from_secs(30);
        assert!(matches!(
            plan.reserve_at("tc-2", &call("search_documents", "b"), soon),
            Continuation::Stop(_)
        ));

        // Long past it.
        let later = start + Duration::from_secs(6 * 60);
        assert_eq!(
            plan.reserve_at("tc-3", &call("search_documents", "c"), later),
            Continuation::Proceed
        );
        assert_eq!(plan.leases_held(), 1, "only the live one is held");
        assert_eq!(
            plan.steps_taken(),
            0,
            "an abandoned call ran nothing and is charged nothing"
        );
    }

    #[test]
    fn the_refusal_names_what_is_committed_not_merely_what_has_finished() {
        // A model told "0 of 1 steps taken" while its own batch holds the last
        // slot would have no idea what stopped it.
        let mut plan = plan_with(1);
        plan.reserve("tc-1", &call("search_documents", "a"));
        let Continuation::Stop(reason) = plan.reserve("tc-2", &call("search_documents", "b"))
        else {
            panic!("the second call must be stopped");
        };
        let StopReason::StepsInFlight { in_flight, allowed } = reason else {
            panic!("stopped for the wrong reason: {reason:?}");
        };
        assert_eq!(in_flight, 1);
        assert_eq!(allowed, 1);
        // And the run is not over: contention is a queue, not a budget spent.
        assert!(
            plan.stopped().is_none(),
            "a call refused for want of a free slot must not end the run"
        );
    }

    #[test]
    fn reservations_are_still_subject_to_every_other_check() {
        // The reservation is taken *after* the plan has agreed, not instead of
        // it. A tool outside the plan takes no slot.
        let mut plan = PlanRun::new(
            "run-1",
            vec!["do the work".to_string()],
            Budget {
                max_steps: 5,
                max_duration: Duration::from_secs(600),
                permitted_tools: vec![ToolName::SearchDocuments],
                repeat_limit: 100,
            },
        );
        assert!(matches!(
            plan.reserve("tc-1", &call("write_scoped_file", "a")),
            Continuation::Stop(_)
        ));
        assert_eq!(plan.leases_held(), 0);
        assert_eq!(plan.steps_committed(), 0);
    }
}
