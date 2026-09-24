//! How the runtime writes what it is doing into the durable record.
//!
//! The glue between [`super`], which decides and performs tool calls, and
//! [`events`], which stores an ordered history of them. Kept apart from both:
//! the decision path should not grow a second concern, and the event store
//! should not learn what a `RuntimeDeps` is.
//!
//! ## Why this duplicates the in-memory tables
//!
//! `RuntimeDeps` already accumulates a run's tool calls, passages and
//! artifacts, and the task record is built from those when the run ends. Every
//! one of those tables dies with the process. So does the account of the run
//! they were going to produce.
//!
//! What is written here is the same information going somewhere that survives.
//! The duplication is the point: one copy is convenient and fast and is lost on
//! a crash, and the other is the one a window that reopens can read.

use std::path::Path;
use std::sync::Arc;

use serde_json::{json, Value};

use super::events;
use super::{CallParams, RuntimeDeps};
use crate::orchestrator::tools::ToolName;

impl RuntimeDeps {
    /// Who to attribute a record to right now.
    ///
    /// Falls back to `system` rather than refusing: an event that could not be
    /// attributed is still worth having, and the tool call it describes has
    /// already been through a gateway that *does* refuse when nobody is signed
    /// in.
    pub(super) fn actor(&self) -> String {
        self.session
            .read()
            .ok()
            .and_then(|guard| guard.as_ref().map(|session| session.user.id.clone()))
            .unwrap_or_else(|| events::SYSTEM_ACTOR.to_string())
    }

    /// Writes one event into the run's durable history.
    ///
    /// Best-effort by design, and the asymmetry with `RuntimeDeps::publish` is deliberate:
    /// a dropped UI event costs a progress line, a dropped durable event costs
    /// a line in the history a restart reads. So this one is logged when it
    /// fails, and it still does not stop the run — a task that refuses to
    /// proceed because its history could not be written would trade a
    /// recoverable gap for a certain loss.
    pub(super) fn remember(&self, run_id: &str, event_type: events::TaskEventType, payload: Value) {
        let draft = events::EventDraft::new(run_id, event_type, self.actor()).with(payload);
        match self.events.record(draft) {
            // Published only once it is on disk, and carrying the sequence
            // number the row was given. A client that receives these in order
            // can tell a gap from a quiet moment; one that received them before
            // the write could be told about an event that never landed.
            Ok(event) => (self.emit_durable)(event.envelope()),
            // The run ended while a tool call was still in flight — an ordinary
            // race after an abort, not a fault.
            Err(events::AppendError::AlreadyEnded { .. })
            | Err(events::AppendError::Duplicate { .. }) => {}
            Err(error) => {
                log::warn!("[tasks] run {run_id}: {error}");
            }
        }
    }
}

impl RuntimeDeps {
    /// Writes the point this run could be continued from, right now.
    ///
    /// Called after every tool result and either side of a side-effecting call.
    /// A checkpoint is cheap — one small row, overwritten — and the alternative
    /// is a run that can only be resumed from wherever the last expensive
    /// milestone happened to be.
    ///
    /// ## Why a failure here is returned rather than logged
    ///
    /// Everything else in this module is best-effort: a dropped progress event
    /// costs a line on a screen. This is not that. A run that believes it
    /// checkpointed and did not will later be *offered* for resumption from a
    /// point that does not exist, and whoever accepts that offer gets a
    /// continuation built on a record nobody wrote. So the caller is told, and
    /// the caller decides whether it is safe to go on.
    ///
    /// Returns `Ok(false)` when the run has no seed — started before checkpoints
    /// existed, or never fully started. That is not a failure: there is nothing
    /// to checkpoint against, and inventing a world to record would be worse
    /// than recording nothing.
    /// ## Why this no longer takes the notes
    ///
    /// It used to, and every production caller passed `RunMemory::default()` —
    /// not out of carelessness but because the tool path genuinely has no way
    /// to reach the loop's notes: they live in the child process. So the one
    /// record a recovery reads said the run had no goal, no evidence and no
    /// completed effects, and a resumption started the work again.
    ///
    /// The notes now come from the seed, where
    /// [`crate::agent_runtime::state_commit`] puts them after checking every
    /// claim that could excuse work. A checkpoint therefore carries the last
    /// state Rust actually agreed to, and there is no parameter through which a
    /// caller can accidentally write an empty one.
    pub(super) fn checkpoint(
        &self,
        run_id: &str,
        state: events::RunState,
    ) -> Result<bool, String> {
        let seed = {
            let Ok(seeds) = self.checkpoints.lock() else {
                return Err("the checkpoint table was left locked by a failed write".to_string());
            };
            match seeds.get(run_id) {
                Some(seed) => seed.clone(),
                None => return Ok(false),
            }
        };
        let notes = seed.committed_notes.clone();
        let manifest = seed.manifest.clone();

        // The sequence this checkpoint is taken after, read from the history
        // rather than counted here: a local counter and the durable log
        // disagreeing is exactly the drift a resumption would act on.
        let last_seq = self
            .events
            .events_since(run_id, 0)
            .map(|page| page.last_seq())
            .unwrap_or(0);

        // Effects nobody has settled. Read fresh every time, because an effect
        // becoming unknown is the single fact that must stop a resumption, and a
        // stale copy of this list is a copy that says a run is safe.
        let unknown: Vec<String> = self
            .events
            .unknown_effects()
            .unwrap_or_default()
            .into_iter()
            .filter(|effect| effect.run_id == run_id)
            .map(|effect| effect.idempotency_key.clone())
            .collect();

        let checkpoint = seed.checkpoint(run_id, state, last_seq, notes, None, manifest, unknown);
        crate::agent_runtime::resume::checkpoint_now(&self.events, &checkpoint)
    }

    /// Takes a checkpoint, and records the failure to take one.
    ///
    /// For the call sites where stopping the run is not the right answer but
    /// silently continuing is not either: the failure becomes a durable event,
    /// so a later reader sees the gap rather than inferring it from a resume
    /// point further back than it should be.
    pub(super) fn checkpoint_or_note(&self, run_id: &str, state: events::RunState) {
        if let Err(error) = self.checkpoint(run_id, state) {
            log::error!("[tasks] run {run_id}: the checkpoint could not be written: {error}");
            self.remember(
                run_id,
                events::TaskEventType::CheckpointFailed,
                json!({ "detail": error }),
            );
        }
    }
}

/// Records a refusal.
///
/// One place rather than five, because a refusal that reaches the model and not
/// the history is the failure mode that makes a trace say the policy never did
/// anything.
pub(super) fn remember_refusal(deps: &Arc<RuntimeDeps>, call: &CallParams, reason: &str) {
    deps.remember(
        &call.run_id,
        events::TaskEventType::ToolRefused,
        json!({
            "toolCallId": call.tool_call_id,
            "tool": call.tool,
            "reason": reason,
        }),
    );
}

/// Refuses a call, records the refusal, and gives back its budget slot.
///
/// The release belongs here rather than at each refusal site for the same
/// reason the recording does: this is the one place a refusal is produced, and
/// a slot held by a call that was refused is a step the run pays for and never
/// takes. The plan admits a call *before* the gateway, the hooks and the
/// approval queue have their say, so every refusal after that point is
/// releasing a reservation it took a moment ago.
///
/// Harmless where no reservation was taken — a call refused before the plan saw
/// it holds nothing, and releasing nothing answers `false` and changes nothing.
pub(super) fn refused(deps: &Arc<RuntimeDeps>, call: &CallParams, reason: String) -> Value {
    // Every refusal, named, beside the `[tool]` line `execute` writes when a
    // call actually runs. Without this a refused call and a call that was never
    // made look identical from outside the process, and the difference is the
    // whole question when a tool appears to do nothing.
    log::warn!(
        "[tool] run={} {} refused: {reason}",
        call.run_id,
        call.tool
    );
    remember_refusal(deps, call, &reason);
    // And into the in-memory table the task record is built from. `remember_refusal`
    // above writes the durable event, which is what a screen reads after a
    // restart; this is what the record written at the end of the run holds, and
    // until now only successful and failed calls reached it.
    super::record_refusal(deps, &call.run_id, &call.tool, &reason);
    super::release_reservation(deps, &call.run_id, &call.tool_call_id);
    json!({ "outcome": "refuse", "reason": reason })
}

/// A refusal that may also be the end of the turn.
///
/// `terminal` is set when the plan has halted — its budget is spent, it caught
/// the run looping, or its time is up — and every call after this one would be
/// refused with the same sentence. The loop reads it and stops rather than
/// trying again, which is what it did before: each retry emitted events, the
/// events rearmed the stall guard, and a run with nothing left to spend ran to
/// its thirty-minute deadline producing refusals.
pub(super) fn refused_terminally(
    deps: &Arc<RuntimeDeps>,
    call: &CallParams,
    reason: String,
    terminal: bool,
) -> Value {
    let mut verdict = refused(deps, call, reason);
    if terminal {
        if let Some(object) = verdict.as_object_mut() {
            object.insert("terminal".to_string(), Value::Bool(true));
        }
    }
    verdict
}

/// Writes how a tool call went into the run's durable history.
///
/// Separate from `super::record_call`, which fills the in-memory table the task
/// record is built from at the end. The two look redundant and are not: the
/// in-memory one dies with the process, and this one is what a screen reads
/// after a restart. A run interrupted halfway still shows the calls it made.
pub(super) fn remember_outcome(
    deps: &Arc<RuntimeDeps>,
    call: &CallParams,
    tool: ToolName,
    resolved_path: Option<&Path>,
    outcome: &Result<String, String>,
) {
    let (event_type, payload) = match outcome {
        // `detail` is redacted on the way in — a search result carries the
        // passage it found, and that is exactly what must not be copied here.
        Ok(text) => (
            events::TaskEventType::ToolSucceeded,
            json!({
                "toolCallId": call.tool_call_id,
                "tool": tool.as_str(),
                "detail": text,
            }),
        ),
        Err(reason) => (
            events::TaskEventType::ToolFailed,
            json!({
                "toolCallId": call.tool_call_id,
                "tool": tool.as_str(),
                "reason": reason,
            }),
        ),
    };
    deps.remember(&call.run_id, event_type, payload);

    // Checkpointed after the outcome is recorded and before the loop is told,
    // so the resume point never claims a tool settled that the history does not
    // also show settling.
    deps.checkpoint_or_note(&call.run_id, events::RunState::ToolResultRecorded);

    // The file is a reference: its name, so the Tasks screen can list what a
    // run produced without opening anything.
    if outcome.is_ok() {
        if let Some(name) = resolved_path
            .filter(|_| matches!(tool, ToolName::CreateDocx | ToolName::CreateXlsx | ToolName::WriteScopedFile))
            .and_then(|path| path.file_name())
            .and_then(|name| name.to_str())
        {
            deps.remember(
                &call.run_id,
                events::TaskEventType::ArtifactProduced,
                json!({ "name": name, "tool": tool.as_str() }),
            );
        }
    }
}

/// Keeps the two loop events a recovered trace would otherwise be missing.
///
/// Everything else the loop publishes is progress — a message part, a tool
/// starting — and this side already records the tool calls themselves from the
/// authorisation path, which is the account that cannot be dropped. These two
/// have no other source: the turn count and the compactions are only ever
/// announced by the loop.
/// The compaction's source range, copied field by field for the reason the
/// rest of that payload is.
fn source_range(range: Option<&Value>) -> Value {
    let Some(range) = range else {
        return Value::Null;
    };
    let calls: Vec<Value> = range
        .get("toolCalls")
        .and_then(Value::as_array)
        .map(|calls| {
            calls
                .iter()
                .filter_map(Value::as_str)
                .take(256)
                .map(|id| Value::String(id.chars().take(96).collect()))
                .collect()
        })
        .unwrap_or_default();
    json!({
        "from": range.get("from").and_then(Value::as_u64),
        "to": range.get("to").and_then(Value::as_u64),
        "toolCalls": calls,
    })
}

pub(super) fn remember_loop_event(deps: &Arc<RuntimeDeps>, params: &Value) {
    let Some(run_id) = params.get("runId").and_then(Value::as_str) else {
        return;
    };
    let Some(event) = params.get("event") else {
        return;
    };
    match event.get("type").and_then(Value::as_str) {
        Some("turn_end") => deps.remember(run_id, events::TaskEventType::TurnEnded, json!({})),
        // Copied field by field rather than forwarded whole. The loop's event
        // carries a `type` this side has already branched on, and a payload that
        // grows silently with whatever a future runtime adds is a payload
        // nobody has checked for document text. Naming each field means a new
        // one has to be added here, by somebody who can see what it holds.
        Some("context_compacted") => deps.remember(
            run_id,
            events::TaskEventType::ContextCompacted,
            json!({
                "ordinal": event.get("ordinal").cloned().unwrap_or(Value::Null),
                "tokensBefore": event.get("tokensBefore").cloned().unwrap_or(Value::Null),
                "tokensAfter": event.get("tokensAfter").cloned().unwrap_or(Value::Null),
                "messagesSummarised": event
                    .get("messagesSummarised")
                    .cloned()
                    .unwrap_or(Value::Null),
                "refinedExistingSummary": event
                    .get("refinedExistingSummary")
                    .cloned()
                    .unwrap_or(Value::Bool(false)),
                "toolResultsCleared": event
                    .get("toolResultsCleared")
                    .cloned()
                    .unwrap_or(Value::Null),
                // Which transcript positions the summary now stands in for, and
                // the tool-call ids among them: positions and opaque ids, never
                // message text. Lets a reader say exactly which of the run's own
                // records the model was reading *about* rather than reading.
                "sourceRange": source_range(event.get("sourceRange")),
                // Counts only. The ledger says how many tokens each section
                // held, never what was in them, so it is safe in a record read
                // more widely than the transcript it describes.
                "ledger": ledger_counts(event.get("ledger")),
                // Stamped by the store when the row is written; carried here so
                // a payload read on its own is not undated.
                "at": chrono::Utc::now().to_rfc3339(),
            }),
        ),
        _ => {}
    }
}


/// The ledger's section counts, flattened and stripped of anything else.
///
/// The runtime sends a nested snapshot; the record holds a flat row of numbers.
/// Rebuilt rather than forwarded so a field the runtime adds later cannot ride
/// into the durable record unexamined — every number below is one somebody
/// chose to keep.
fn ledger_counts(ledger: Option<&Value>) -> Value {
    let Some(ledger) = ledger else {
        return Value::Null;
    };
    let section = |name: &str| {
        ledger
            .get("sections")
            .and_then(|sections| sections.get(name))
            .and_then(Value::as_u64)
            .unwrap_or(0)
    };
    let top = |name: &str| ledger.get(name).and_then(Value::as_i64).unwrap_or(0);

    json!({
        "system": section("system"),
        "skill": section("skill"),
        "toolSchema": section("toolSchema"),
        "evidence": section("evidence"),
        "notes": section("notes"),
        "transcript": section("transcript"),
        "compaction": section("compaction"),
        "reserve": section("reserve"),
        "occupied": top("occupied"),
        "committed": top("committed"),
        "window": top("window"),
        "headroom": top("headroom"),
    })
}

impl RuntimeDeps {
    /// What Rust itself recorded about this run, for checking a proposal.
    ///
    /// Read fresh on every commit. A cached copy is a copy that can say a tool
    /// succeeded when the row saying so was never written.
    pub(super) fn rust_facts(
        &self,
        run_id: &str,
    ) -> crate::agent_runtime::state_commit::RustFacts {
        use crate::agent_runtime::state_commit::RustFacts;

        let mut facts = RustFacts::default();

        // Tool receipts, from the durable log. This is the half a proposal
        // cannot forge: the row was written by `remember_outcome` after the
        // gateway and the tool had both agreed.
        if let Ok(page) = self.events.events_since(run_id, 0) {
            for event in &page.events {
                match event.event_type {
                    events::TaskEventType::ToolSucceeded => {
                        if let Some(tool) =
                            event.payload.get("tool").and_then(serde_json::Value::as_str)
                        {
                            *facts.succeeded_tools.entry(tool.to_string()).or_default() += 1;
                        }
                    }
                    events::TaskEventType::ArtifactProduced => {
                        let name = event.payload.get("name").and_then(serde_json::Value::as_str);
                        let tool = event.payload.get("tool").and_then(serde_json::Value::as_str);
                        if let (Some(name), Some(tool)) = (name, tool) {
                            facts
                                .produced_artifacts
                                .push((name.to_string(), tool.to_string()));
                        }
                    }
                    _ => {}
                }
            }
        }

        // The markers this run's searches actually handed out. `retrieval`
        // numbers them from one against this table, so its length is the
        // highest marker that can mean anything.
        facts.evidence_count =
            crate::agent_runtime::retrieval::for_run(&self.passages, run_id).len() as u32;

        facts.calculation_count = self
            .calculations
            .lock()
            .ok()
            .and_then(|table| table.get(run_id).map(|rows| rows.len() as u32))
            .unwrap_or(0);

        facts.artifact_names = crate::agent_runtime::artifacts::for_run(&self.produced, run_id)
            .into_iter()
            .map(|produced| produced.name)
            .collect();

        // Milestones are a person's signature, written by
        // `agent_acknowledge_milestone`. They are carried on the accepted notes
        // and are never taken from a proposal.
        facts.milestones = self
            .checkpoints
            .lock()
            .ok()
            .and_then(|seeds| seeds.get(run_id).map(|seed| seed.committed_notes.milestones.clone()))
            .unwrap_or_default();

        facts
    }

    /// Accepts a state proposal from the loop, or explains why it cannot.
    ///
    /// The safe-boundary commit. Everything the loop knows and Rust does not —
    /// the goal, where the plan has reached, what it intends next — arrives
    /// here; everything Rust knows and the loop could be wrong about is checked
    /// against the record before any of it is written. See
    /// [`crate::agent_runtime::state_commit`] for the rules and why each one is
    /// there.
    ///
    /// The checkpoint is written *after* the notes are accepted and in the same
    /// call, so a run cannot be left having agreed state that its resume point
    /// does not carry.
    pub(super) fn commit_state(
        &self,
        proposal: &crate::agent_runtime::state_commit::StateProposal,
    ) -> crate::agent_runtime::state_commit::CommitOutcome {
        use crate::agent_runtime::state_commit::{self, CommitOutcome};

        let last_event_seq = self
            .events
            .events_since(&proposal.run_id, 0)
            .map(|page| page.last_seq())
            .unwrap_or(0);

        let refuse = |because: String| CommitOutcome {
            accepted: false,
            refused_because: Some(because),
            last_event_seq,
            corrections: Vec::new(),
            notes: crate::agent_runtime::memory::RunMemory::default(),
        };

        if proposal.commit_version != state_commit::STATE_COMMIT_VERSION {
            return refuse(format!(
                "this build writes state commits at version {}, and the proposal is version {}; \
                 applying half a record it does not understand is how a resumption acts on a \
                 state nobody wrote",
                state_commit::STATE_COMMIT_VERSION,
                proposal.commit_version
            ));
        }

        // The seed is the attempt. A commit naming a different one is a
        // straggler from a worker that outlived a restart, and letting it write
        // would move the resume point of the attempt now running back to a
        // state a dead process believed.
        let seed = {
            let Ok(seeds) = self.checkpoints.lock() else {
                return refuse(
                    "the checkpoint table was left locked by a failed write".to_string(),
                );
            };
            match seeds.get(&proposal.run_id) {
                Some(seed) => seed.clone(),
                None => {
                    return refuse(format!(
                        "run {} has no checkpoint seed on this side, so there is no attempt for \
                         this commit to belong to",
                        proposal.run_id
                    ))
                }
            }
        };
        if seed.attempt_id != proposal.attempt_id {
            return refuse(format!(
                "this commit belongs to attempt {}, and the attempt running is {}; a commit from \
                 an earlier attempt cannot move this one's resume point",
                proposal.attempt_id, seed.attempt_id
            ));
        }

        let facts = self.rust_facts(&proposal.run_id);
        let (notes, corrections) = state_commit::validate(proposal, &facts, &seed.committed_notes);

        // Said out loud, one line each. A correction is the record disagreeing
        // with the run's own account of itself, and the serious ones are the
        // cases that would have caused duplicate work.
        for correction in &corrections {
            if correction.is_serious() {
                log::warn!(
                    "[state] run {}: {}",
                    proposal.run_id,
                    correction.explain()
                );
            } else {
                log::info!(
                    "[state] run {}: {}",
                    proposal.run_id,
                    correction.explain()
                );
            }
        }

        if let Ok(mut seeds) = self.checkpoints.lock() {
            if let Some(held) = seeds.get_mut(&proposal.run_id) {
                // Re-checked under the lock. Between the read above and here a
                // resumption could have replaced the seed, and writing this
                // attempt's notes onto that one is the race this guards.
                if held.attempt_id == proposal.attempt_id {
                    held.committed_notes = notes.clone();
                }
            }
        }

        self.checkpoint_or_note(&proposal.run_id, proposal.state);

        CommitOutcome {
            accepted: true,
            refused_because: None,
            last_event_seq,
            corrections,
            notes,
        }
    }
}
