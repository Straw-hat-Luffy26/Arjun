//! What a run is doing, published while it is doing it.
//!
//! ## Why this exists
//!
//! [`crate::commands::agent::agent_start_run`] does a great deal before the
//! agent loop is handed anything: it reads attachments through the OCR model,
//! probes the GPU, routes to a model, and — on a cold start — waits for a
//! `llama-server` to load several gigabytes of weights. None of that produced
//! an event, so the chat surface showed a motionless "Thinking" pill from the
//! moment the person pressed enter until the first token arrived.
//!
//! The durable event log confirms the shape of it: on one measured run the
//! events recorded between `run_started` and `turn_ended` were none at all,
//! across 122 seconds.
//!
//! ## The contract
//!
//! A stage is emitted **only when the work it names is actually starting**.
//! There are no timers, no interpolated percentages, and no stage standing in
//! for "something is probably happening". A stage that cannot be determined is
//! simply not emitted, and the surface keeps showing the last one it was told.
//!
//! Every stage carries the identifiers the chat surface routes on
//! ([`StageTag`]). The `runId` on the envelope is the caller's correlation id
//! until the server has issued its own — the same rule `plan_ready` follows —
//! so a stage emitted before the run exists still reaches the cell waiting for
//! it, and cannot reach any other.
//!
//! ## What is never on this channel
//!
//! The model's private reasoning. There is deliberately no stage for it here:
//! reasoning happens inside the agent loop, where the translator in
//! `agent-runtime/src/run.ts` turns it into a `model_thinking` event carrying
//! a duration and a character count and no text at all. That translator is the
//! only place the reasoning stream is read, and this module never sees it.
//!
//! Not to be confused with [`super::memory::RunStage`], which is a step in the
//! *model's own* plan as it reports it in its working notes. This module is
//! about the application's work, not the model's.

use serde::{Deserialize, Serialize};
use serde_json::json;
use tauri::{AppHandle, Emitter};

use super::AGENT_EVENT;

/// A stage of a run, named for the work rather than for a progress bar.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Stage {
    /// The command was entered and the caller's permission checked. The first
    /// thing that happens, so it is the first thing the surface can show.
    Accepted,
    /// An attachment is being read. Carries the file and, when the reader
    /// knows them, the page it is on and how many there are.
    ReadingAttachment,
    /// Every attachment has been read and folded into the prompt.
    AttachmentsRead,
    /// The pages that were read are being cut into retrievable passages.
    ///
    /// Real work with a real duration, and the step that makes a document
    /// larger than the window survivable — so it gets a line of its own rather
    /// than disappearing into the gap between "read the attachment" and
    /// "choosing a model". Carries the pages and passages actually produced.
    IndexingDocument,
    /// Passages are being chosen to fit the window the server actually holds.
    ///
    /// Emitted once, after the model is known, carrying how many passages the
    /// documents hold and how many this turn could afford. The difference
    /// between those two numbers is the thing a person is owed and used to be
    /// told nothing about.
    SelectingContext,
    /// Hardware is being inspected and a model chosen for this prompt.
    Routing,
    /// A model was chosen. Carries its name and role.
    Routed,
    /// Weights are being loaded. Emitted only when the server for this model
    /// is not already answering — a warm server goes straight to
    /// [`Stage::ModelReady`].
    LoadingModel,
    /// The model server is answering. Carries whether it was already warm and
    /// how long the wait was.
    ModelReady,
    /// The workspace and the plan for this run are being fixed.
    Planning,
    /// The prompt has been handed to the agent loop.
    Generating,
    /// The answer is being checked against the evidence the run gathered.
    Verifying,
    /// The run is over, one way or another.
    Complete,
}

impl Stage {
    /// The wire name. Kept next to the variant so the two cannot drift.
    pub fn as_str(self) -> &'static str {
        match self {
            Stage::Accepted => "accepted",
            Stage::ReadingAttachment => "readingAttachment",
            Stage::AttachmentsRead => "attachmentsRead",
            Stage::IndexingDocument => "indexingDocument",
            Stage::SelectingContext => "selectingContext",
            Stage::Routing => "routing",
            Stage::Routed => "routed",
            Stage::LoadingModel => "loadingModel",
            Stage::ModelReady => "modelReady",
            Stage::Planning => "planning",
            Stage::Generating => "generating",
            Stage::Verifying => "verifying",
            Stage::Complete => "complete",
        }
    }

    /// Every stage, for tests and for anything that needs to enumerate them.
    pub const ALL: [Stage; 13] = [
        Stage::Accepted,
        Stage::ReadingAttachment,
        Stage::AttachmentsRead,
        Stage::IndexingDocument,
        Stage::SelectingContext,
        Stage::Routing,
        Stage::Routed,
        Stage::LoadingModel,
        Stage::ModelReady,
        Stage::Planning,
        Stage::Generating,
        Stage::Verifying,
        Stage::Complete,
    ];
}

/// The identifiers every stage is stamped with.
///
/// Held together rather than passed as four arguments because dropping one of
/// them is exactly the defect this is meant to prevent: a stage that reaches
/// the wrong cell is worse than no stage at all.
#[derive(Debug, Clone)]
pub struct StageTag {
    /// The caller's own run id, echoed on every stage so a surface can match
    /// before the server has issued its id.
    pub correlation_id: Option<String>,
    /// The assistant `Message` the answer will be written into.
    pub message_id: Option<String>,
    /// The conversation that message belongs to.
    pub conversation_id: Option<String>,
    /// The server's run id, once there is one.
    pub run_id: Option<String>,
}

impl StageTag {
    /// A tag for a run that has not been given a server id yet.
    pub fn new(
        correlation_id: Option<String>,
        message_id: Option<String>,
        conversation_id: Option<String>,
    ) -> Self {
        Self {
            correlation_id,
            message_id,
            conversation_id,
            run_id: None,
        }
    }

    /// Records the server's run id, once `agent_start_run` has minted one.
    pub fn with_run_id(&mut self, run_id: &str) {
        self.run_id = Some(run_id.to_string());
    }

    /// What the envelope's `runId` should be.
    ///
    /// The server's id once it exists, and the caller's correlation id before
    /// that. Both are ids the waiting reducer already knows, which is what
    /// makes a stage routable at every point in the run.
    pub fn envelope_run_id(&self) -> String {
        self.run_id
            .clone()
            .or_else(|| self.correlation_id.clone())
            .unwrap_or_default()
    }
}

/// Publishes stages, and times them.
///
/// Holds the instant the run was accepted so every stage can carry how long
/// after the button press it happened. That number is both the measurement the
/// performance work needs and the "12s" the person reads, taken from one clock
/// rather than two that can disagree.
pub struct StageReporter {
    app: AppHandle,
    tag: StageTag,
    started: std::time::Instant,
}

impl StageReporter {
    pub fn new(app: AppHandle, tag: StageTag) -> Self {
        Self {
            app,
            tag,
            started: std::time::Instant::now(),
        }
    }

    /// Milliseconds since the run was accepted.
    pub fn elapsed_ms(&self) -> u64 {
        self.started.elapsed().as_millis() as u64
    }

    /// The tag, so `agent_start_run` can stamp the server run id onto it once
    /// it has minted one.
    pub fn tag_mut(&mut self) -> &mut StageTag {
        &mut self.tag
    }

    /// The tag, for work on other channels that has to name the same turn.
    ///
    /// The document reader emits its own per-page progress on
    /// `attachment:progress`, and that line belongs to this turn as much as
    /// any stage does. Handing it the same tag is what stops a page counter
    /// from landing on a conversation that did not attach anything.
    pub fn tag(&self) -> &StageTag {
        &self.tag
    }

    /// Emits one stage with no extra detail.
    pub fn stage(&self, stage: Stage) {
        self.emit(stage, json!({}));
    }

    /// Emits one stage carrying detail the caller actually measured.
    ///
    /// `detail` is merged into the event object, so a field name here becomes
    /// a field on the wire. Nothing is invented: every caller passes values it
    /// read from the work it has just done.
    pub fn stage_with(&self, stage: Stage, detail: serde_json::Value) {
        self.emit(stage, detail);
    }

    fn emit(&self, stage: Stage, detail: serde_json::Value) {
        let elapsed = self.elapsed_ms();
        let mut event = json!({
            "type": "run_stage",
            "stage": stage.as_str(),
            "elapsedMs": elapsed,
            "correlationId": self.tag.correlation_id,
            "messageId": self.tag.message_id,
            "conversationId": self.tag.conversation_id,
        });
        if let (Some(map), Some(extra)) = (event.as_object_mut(), detail.as_object()) {
            for (key, value) in extra {
                map.insert(key.clone(), value.clone());
            }
        }
        // Logged as well as emitted. The emit reaches a window that is open
        // now; the log is what a timing measurement is read back from
        // afterwards, including for a run nobody was watching.
        log::info!(
            "[stage] run={} stage={} at={}ms detail={}",
            self.tag.envelope_run_id(),
            stage.as_str(),
            elapsed,
            detail
        );
        let _ = self.app.emit(
            AGENT_EVENT,
            json!({ "runId": self.tag.envelope_run_id(), "event": event }),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_stage_has_a_distinct_wire_name() {
        let mut names: Vec<&str> = Stage::ALL.iter().map(|s| s.as_str()).collect();
        names.sort_unstable();
        let count = names.len();
        names.dedup();
        assert_eq!(names.len(), count, "two stages share a wire name");
    }

    #[test]
    fn the_envelope_falls_back_to_the_callers_id_until_the_server_has_one() {
        let mut tag = StageTag::new(
            Some("corr-1".into()),
            Some("a-1".into()),
            Some("c-1".into()),
        );
        assert_eq!(tag.envelope_run_id(), "corr-1");
        tag.with_run_id("server-1");
        assert_eq!(tag.envelope_run_id(), "server-1");
    }

    #[test]
    fn a_run_with_neither_id_produces_an_empty_envelope_rather_than_a_panic() {
        let tag = StageTag::new(None, Some("a-1".into()), None);
        assert_eq!(tag.envelope_run_id(), "");
    }
}
