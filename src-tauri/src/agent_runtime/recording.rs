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
    pub(super) fn checkpoint(
        &self,
        run_id: &str,
        state: events::RunState,
        notes: crate::agent_runtime::memory::RunMemory,
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

        // The sequence this checkpoint is taken after, read from the history
        // rather than counted here: a local counter and the durable log
        // disagreeing is exactly the drift a resumption would act on.
        let last_seq = self
            .events
            .events_since(run_id, 0)
            .map(|page| page.last_seq())?;

        // Effects nobody has settled. Read fresh every time, because an effect
        // becoming unknown is the single fact that must stop a resumption, and a
        // stale copy of this list is a copy that says a run is safe.
        let unknown = self.events.effect_obligations(run_id)?.0;
        let notes = if notes == crate::agent_runtime::memory::RunMemory::default() {
            self.events.checkpoint(run_id).map_err(|error| error.explain())?.map(|saved| saved.notes).unwrap_or(notes)
        } else { notes };

        let checkpoint = seed.checkpoint(run_id, state, last_seq, notes, None, unknown);
        crate::agent_runtime::resume::checkpoint_now(&self.events, &checkpoint)
    }

    /// Takes a checkpoint, and records the failure to take one.
    ///
    /// For the call sites where stopping the run is not the right answer but
    /// silently continuing is not either: the failure becomes a durable event,
    /// so a later reader sees the gap rather than inferring it from a resume
    /// point further back than it should be.
    pub(super) fn checkpoint_or_note(
        &self,
        run_id: &str,
        state: events::RunState,
        notes: crate::agent_runtime::memory::RunMemory,
    ) {
        if let Err(error) = self.checkpoint(run_id, state, notes) {
            self.audit_health.writes_failed(&error);
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
    remember_refusal(deps, call, &reason);
    super::release_reservation(deps, &call.run_id, &call.tool_call_id);
    json!({ "outcome": "refuse", "reason": reason })
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
    deps.checkpoint_or_note(
        &call.run_id,
        events::RunState::ToolResultRecorded,
        crate::agent_runtime::memory::RunMemory::default(),
    );

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
