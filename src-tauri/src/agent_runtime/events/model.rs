//! What one durable task event is, and what may be written into it.
//!
//! A run used to leave behind exactly one thing: a JSON file written after it
//! ended. That is enough to review a finished task and nothing at all for the
//! two cases that matter most — a run still going when the window remounts, and
//! a run that was going when the process died. Both look identical from the
//! outside (no file), and both used to be unrecoverable.
//!
//! So the record is now written *as the run happens*, one ordered event at a
//! time. This module is the vocabulary: the event types, the envelope every
//! event carries, and — the part with teeth — the redaction that decides what
//! is allowed into a payload in the first place.
//!
//! ## Why redaction is here and not at the call sites
//!
//! ARJUN design rule 14 says confidential contents must not be copied into a record more
//! people can read than could read the original. A rule enforced at the call
//! sites is a rule that holds until somebody adds a call site. So every payload
//! goes through [`redact`] on its way into an [`EventDraft`], and the fields
//! that carry document text come out as a hash and a length. The event says
//! *that* a passage was retrieved and which one it was; it does not say what
//! the passage said.
//!
//! The same rule covers the model's own words. Nothing here carries a message,
//! a reasoning trace or a partial completion — the operational trace is plan,
//! decision, tool, evidence, approval, artifact and verifier status, and that
//! is all an event may hold.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

/// The shape events are written in.
///
/// Bumped when a payload's meaning changes, never when a field is added. Stored
/// on every row so a reader built for an older shape can say "this event is
/// newer than I am" instead of silently misreading it.
///
/// - **1** — the first durable history: run start, plan, tools, endings.
/// - **2** — explicit run states. Adds the lifecycle events the state machine
///   in [`super::machine`] needs, and the tool-effect events that make an
///   interrupted side effect recoverable.
pub const SCHEMA_VERSION: u32 = 2;

/// What happened. Coarse on purpose — the detail belongs in the payload, and a
/// type per detail would make every reader a match arm longer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum TaskEventType {
    // -- Lifecycle --------------------------------------------------------
    /// The run was accepted. Always the first event of a run.
    RunCreated,
    /// The sensitivity of the material is known, so routing can be narrowed.
    RunClassified,
    /// A model was chosen, with the reasons that led there.
    RunRouted,
    /// The plan it will be held to, fixed before the model was told anything.
    PlanReady,
    /// The loop began.
    RunStarted,

    // -- Progress ---------------------------------------------------------
    /// A step spent.
    PlanStep,
    /// The plan will permit nothing further. The *ending* event that follows
    /// says which kind of ending it was; this only says the plan is done.
    PlanStopped,
    /// One model turn finished.
    TurnEnded,
    /// Older history was replaced by a summary so the run could continue.
    ///
    /// Kept durably rather than only shown live, because it is a caveat on
    /// everything the run says afterwards: those answers rest on a summary of
    /// the earlier turns rather than on the turns themselves. A recovered trace
    /// that quietly dropped this would overstate its own grounding.
    ContextCompacted,
    /// The turn could not carry all of its own conversation, so the oldest of
    /// it was left out before the model was called.
    ///
    /// Durable for the same reason as `ContextCompacted`, and it is the
    /// stronger of the two: a compaction replaces history with a summary and
    /// tells the model it did, while this simply does not send it. An answer
    /// given afterwards rests on less than the thread holds and says nothing
    /// about the gap, so a trace that dropped this would overstate what the
    /// model was looking at.
    ContextTrimmed,

    // -- Tools ------------------------------------------------------------
    /// The gateway allowed a call and issued a grant.
    ToolAuthorized,
    /// The gateway, the plan or a person said no before the tool ran.
    ToolRefused,
    ToolSucceeded,
    ToolFailed,
    /// A side-effecting call arrived twice under one idempotency key. The
    /// recorded outcome was returned and the tool was *not* run again.
    ToolReplayed,
    /// A side-effecting call is about to happen. Written **before** the effect,
    /// so that a process which dies mid-call leaves evidence it was trying.
    ToolEffectPending,
    /// A side effect whose outcome nobody knows: it was in flight when the
    /// process went away. Needs a person before its key can be reused.
    ToolEffectUnknown,
    /// A person said what actually happened to an unknown side effect.
    ToolEffectReconciled,
    ArtifactProduced,

    // -- Subagents --------------------------------------------------------
    /// A bounded worker began, with the manifest of what it was permitted.
    SubagentStarted,
    /// It ended. The status says which way, and a failure, a timeout and a
    /// cancellation are each named rather than folded into one ending.
    SubagentStopped,

    // -- Checkpoints ------------------------------------------------------
    /// A resume point was written. Carries the sequence and the attempt, never
    /// the notes themselves.
    CheckpointTaken,
    /// A resume point could not be written. Recorded so a later reader sees the
    /// gap rather than inferring it from a resume point further back than it
    /// should be.
    CheckpointFailed,
    /// A person asked for a stopped run to be continued, and it was.
    RunResumed,

    // -- Memory -----------------------------------------------------------
    /// A run read a scope of memory it was entitled to. Carries counts and key
    /// hashes; never the values, because this record is read by people who are
    /// not cleared for the material it describes.
    MemoryRecalled,
    /// A memory operation was refused, with the fixed reason it was refused for.
    MemoryRefused,
    /// A skill's instructions were loaded into a run's context.
    ///
    /// Written when the body actually reaches the model, not when a skill is
    /// merely listed. The payload names the skill, its version and the hash the
    /// bytes were checked against, plus what the load did to the run's tool
    /// set — a skill can only ever narrow it, and the record says by how much.
    SkillLoaded,
    /// A skill was asked for and refused, with the fixed reason.
    SkillRefused,

    /// A run-scope fact was promoted into a project's memory under an approval.
    /// Carries the binding's hashes so the decision stays checkable.
    MemoryPromoted,
    /// An item was removed, or lapsed past its retention.
    MemoryForgotten,

    // -- Approval ---------------------------------------------------------
    ApprovalRequested,
    ApprovalDecided,

    // -- Hooks ------------------------------------------------------------
    /// A deterministic check ran at a lifecycle point.
    ///
    /// Written only when a check had something to say — a refusal, a failure,
    /// or a note. A hook that passed silently is the overwhelmingly common case
    /// and recording it would bury the ones that matter in a run's own noise.
    ///
    /// The payload carries the point, the hook's name, and its bounded reason.
    /// Never the material the check was about: this record is read by people
    /// who are not cleared for what the run was handling.
    HookEvaluated,

    // -- Model turns ------------------------------------------------------
    /// A request was put to the model. Carries counts and a digest, never the
    /// prompt: ARJUN design rule 14 keeps message content out of a record more
    /// people can read than could read the original.
    ModelRequested,
    /// The model answered. Same rule: how much came back, not what it said.
    ModelResponded,

    // -- Compaction -------------------------------------------------------
    /// Compaction is beginning. Paired with `context_compacted`, which reports
    /// that it finished — before this existed only the finish was recorded, so
    /// a run that died *during* compaction left no trace of having tried.
    CompactionStarted,

    // -- Waiting on something outside the run -----------------------------
    /// The run is waiting on something no part of ARJUN controls.
    WaitStarted,
    /// What it was waiting for happened.
    WaitCompleted,

    // -- Recovery ---------------------------------------------------------
    /// A recovery attempt is starting. Distinct from `run_resumed`, which is a
    /// person choosing to continue: this is the process deciding to.
    RecoveryStarted,
    /// Recovery was attempted and did not succeed.
    RecoveryFailed,

    // -- Verification -----------------------------------------------------
    /// The answer is being checked against the evidence the run actually holds.
    VerificationStarted,
    /// The completion check finished, and this is what it concluded. Carries
    /// the per-criterion verdicts, so "it said it was done" and "it was checked
    /// to be done" are different rows in the history.
    CompletionVerified,

    // -- Pausing ----------------------------------------------------------
    /// A person stopped the run without ending it.
    RunPaused,

    // -- Endings ----------------------------------------------------------
    /// The loop finished and an answer was produced.
    RunCompleted,
    /// The run ended badly. `failure` in the payload is the sentence shown.
    RunFailed,
    /// A person stopped it.
    RunCancelled,
    /// It reached the limit the plan set — steps, time, or going in circles.
    RunStoppedByBudget,
    /// The model ran into the output cap for one turn, so the answer stops
    /// mid-way.
    ///
    /// Its own ending rather than a completion or a failure. The model did
    /// what it was asked and the deployment's cap stopped it — and the
    /// difference is invisible in the text, because a cut-off answer reads
    /// exactly like a short one.
    RunStoppedByLength,
    /// It needed to do something it is not permitted to do.
    RunStoppedByPolicy,
    /// It was still going when the process went away, and the next start found
    /// it. Written by recovery, never by the run itself — a run cannot record
    /// its own sudden death.
    RunDegraded,

    // -- Read-only legacy -------------------------------------------------
    /// Schema 1's spelling for what is now [`Self::RunStoppedByBudget`].
    /// Readable so an older database still folds; never written.
    RunTimedOut,
    /// Schema 1's spelling for what is now [`Self::RunDegraded`].
    RunInterrupted,
}

impl TaskEventType {
    /// Stable database spelling. Written out rather than derived from the
    /// variant name so renaming a variant cannot rewrite history.
    pub const fn as_str(self) -> &'static str {
        match self {
            TaskEventType::RunCreated => "run_created",
            TaskEventType::RunClassified => "run_classified",
            TaskEventType::RunRouted => "run_routed",
            TaskEventType::PlanReady => "plan_ready",
            TaskEventType::RunStarted => "run_started",
            TaskEventType::PlanStep => "plan_step",
            TaskEventType::PlanStopped => "plan_stopped",
            TaskEventType::TurnEnded => "turn_ended",
            TaskEventType::ContextCompacted => "context_compacted",
            TaskEventType::ContextTrimmed => "context_trimmed",
            TaskEventType::ToolAuthorized => "tool_authorized",
            TaskEventType::ToolRefused => "tool_refused",
            TaskEventType::ToolSucceeded => "tool_succeeded",
            TaskEventType::ToolFailed => "tool_failed",
            TaskEventType::ToolReplayed => "tool_replayed",
            TaskEventType::ToolEffectPending => "tool_effect_pending",
            TaskEventType::ToolEffectUnknown => "tool_effect_unknown",
            TaskEventType::ToolEffectReconciled => "tool_effect_reconciled",
            TaskEventType::ArtifactProduced => "artifact_produced",
            TaskEventType::SubagentStarted => "subagent_started",
            TaskEventType::SubagentStopped => "subagent_stopped",
            TaskEventType::CheckpointTaken => "checkpoint_taken",
            TaskEventType::CheckpointFailed => "checkpoint_failed",
            TaskEventType::RunResumed => "run_resumed",
            TaskEventType::SkillLoaded => "skill_loaded",
            TaskEventType::SkillRefused => "skill_refused",
            TaskEventType::MemoryRecalled => "memory_recalled",
            TaskEventType::MemoryRefused => "memory_refused",
            TaskEventType::MemoryPromoted => "memory_promoted",
            TaskEventType::MemoryForgotten => "memory_forgotten",
            TaskEventType::ApprovalRequested => "approval_requested",
            TaskEventType::ApprovalDecided => "approval_decided",
            TaskEventType::HookEvaluated => "hook_evaluated",
            TaskEventType::ModelRequested => "model_requested",
            TaskEventType::ModelResponded => "model_responded",
            TaskEventType::CompactionStarted => "compaction_started",
            TaskEventType::WaitStarted => "wait_started",
            TaskEventType::WaitCompleted => "wait_completed",
            TaskEventType::RecoveryStarted => "recovery_started",
            TaskEventType::RecoveryFailed => "recovery_failed",
            TaskEventType::VerificationStarted => "verification_started",
            TaskEventType::CompletionVerified => "completion_verified",
            TaskEventType::RunPaused => "run_paused",
            TaskEventType::RunCompleted => "run_completed",
            TaskEventType::RunFailed => "run_failed",
            TaskEventType::RunCancelled => "run_cancelled",
            TaskEventType::RunStoppedByBudget => "run_stopped_by_budget",
            TaskEventType::RunStoppedByLength => "run_stopped_by_length",
            TaskEventType::RunStoppedByPolicy => "run_stopped_by_policy",
            TaskEventType::RunDegraded => "run_degraded",
            TaskEventType::RunTimedOut => "run_timed_out",
            TaskEventType::RunInterrupted => "run_interrupted",
        }
    }

    pub fn from_str(raw: &str) -> Option<Self> {
        Some(match raw {
            "run_created" => TaskEventType::RunCreated,
            "run_classified" => TaskEventType::RunClassified,
            "run_routed" => TaskEventType::RunRouted,
            "plan_ready" => TaskEventType::PlanReady,
            "run_started" => TaskEventType::RunStarted,
            "plan_step" => TaskEventType::PlanStep,
            "plan_stopped" => TaskEventType::PlanStopped,
            "turn_ended" => TaskEventType::TurnEnded,
            "context_compacted" => TaskEventType::ContextCompacted,
            "context_trimmed" => TaskEventType::ContextTrimmed,
            "tool_authorized" => TaskEventType::ToolAuthorized,
            "tool_refused" => TaskEventType::ToolRefused,
            "tool_succeeded" => TaskEventType::ToolSucceeded,
            "tool_failed" => TaskEventType::ToolFailed,
            "tool_replayed" => TaskEventType::ToolReplayed,
            "tool_effect_pending" => TaskEventType::ToolEffectPending,
            "tool_effect_unknown" => TaskEventType::ToolEffectUnknown,
            "tool_effect_reconciled" => TaskEventType::ToolEffectReconciled,
            "artifact_produced" => TaskEventType::ArtifactProduced,
            "subagent_started" => TaskEventType::SubagentStarted,
            "subagent_stopped" => TaskEventType::SubagentStopped,
            "checkpoint_taken" => TaskEventType::CheckpointTaken,
            "checkpoint_failed" => TaskEventType::CheckpointFailed,
            "run_resumed" => TaskEventType::RunResumed,
            "skill_loaded" => TaskEventType::SkillLoaded,
            "skill_refused" => TaskEventType::SkillRefused,
            "memory_recalled" => TaskEventType::MemoryRecalled,
            "memory_refused" => TaskEventType::MemoryRefused,
            "memory_promoted" => TaskEventType::MemoryPromoted,
            "memory_forgotten" => TaskEventType::MemoryForgotten,
            "approval_requested" => TaskEventType::ApprovalRequested,
            "approval_decided" => TaskEventType::ApprovalDecided,
            "hook_evaluated" => TaskEventType::HookEvaluated,
            "model_requested" => TaskEventType::ModelRequested,
            "model_responded" => TaskEventType::ModelResponded,
            "compaction_started" => TaskEventType::CompactionStarted,
            "wait_started" => TaskEventType::WaitStarted,
            "wait_completed" => TaskEventType::WaitCompleted,
            "recovery_started" => TaskEventType::RecoveryStarted,
            "recovery_failed" => TaskEventType::RecoveryFailed,
            "verification_started" => TaskEventType::VerificationStarted,
            "completion_verified" => TaskEventType::CompletionVerified,
            "run_paused" => TaskEventType::RunPaused,
            "run_completed" => TaskEventType::RunCompleted,
            "run_failed" => TaskEventType::RunFailed,
            "run_cancelled" => TaskEventType::RunCancelled,
            "run_stopped_by_budget" => TaskEventType::RunStoppedByBudget,
            "run_stopped_by_length" => TaskEventType::RunStoppedByLength,
            "run_stopped_by_policy" => TaskEventType::RunStoppedByPolicy,
            "run_degraded" => TaskEventType::RunDegraded,
            "run_timed_out" => TaskEventType::RunTimedOut,
            "run_interrupted" => TaskEventType::RunInterrupted,
            _ => return None,
        })
    }

    /// Whether this event ends the run. A run whose history has no terminal
    /// event is a run that is still going — which is how recovery finds the
    /// ones the process took down with it.
    pub const fn is_terminal(self) -> bool {
        matches!(
            self,
            TaskEventType::RunCompleted
                | TaskEventType::RunFailed
                | TaskEventType::RunCancelled
                | TaskEventType::RunStoppedByBudget
                | TaskEventType::RunStoppedByLength
                | TaskEventType::RunStoppedByPolicy
                | TaskEventType::RunDegraded
                | TaskEventType::RunTimedOut
                | TaskEventType::RunInterrupted
        )
    }
}

#[cfg(test)]
mod vocabulary_tests {
    use super::TaskEventType;

    /// The database spelling of the new ending, pinned.
    ///
    /// Written out rather than derived, so a rename of the variant cannot
    /// rewrite history — and a test rather than a comment, so the rule is
    /// enforced rather than merely stated.
    #[test]
    fn the_length_ending_round_trips_through_its_database_spelling() {
        let event = TaskEventType::RunStoppedByLength;
        assert_eq!(event.as_str(), "run_stopped_by_length");
        assert_eq!(TaskEventType::from_str("run_stopped_by_length"), Some(event));
        assert!(
            event.is_terminal(),
            "a run cut off at the output cap has ended; recovery must not sweep it up again"
        );
    }
}

/// One event as it is stored and as it is read back.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TaskEvent {
    pub run_id: String,
    /// Unique across every run. Supplied by the writer so a retry after an
    /// ambiguous failure presents the same id and is rejected as the duplicate
    /// it is, rather than landing twice. See [`EventDraft::idempotent`].
    pub event_id: String,
    /// Monotonic within the run, starting at 1, no gaps. Assigned by the store
    /// inside the same transaction as the insert.
    pub seq: i64,
    pub event_type: TaskEventType,
    /// RFC 3339, UTC.
    pub at: String,
    /// Who caused it — a user id, or `system` for the application itself.
    pub actor: String,
    pub schema_version: u32,
    /// Redacted. See [`redact`].
    pub payload: Value,
    /// SHA-256 over the canonical form of `payload`. Lets a reader tell a
    /// payload that was rewritten underneath it from one that was not.
    pub payload_hash: String,
}

impl TaskEvent {
    /// How a durable event travels to a window.
    ///
    /// Carries the sequence number, which is the whole point: a client that
    /// receives seq 14 having last applied seq 12 knows it missed one, and can
    /// ask for a snapshot rather than drawing a trace with a hole in it. The
    /// best-effort loop stream has no such number and cannot offer that.
    ///
    /// The payload is the redacted one, unchanged — there is no second
    /// redaction pass here, because there is nothing that could have been added
    /// since [`EventDraft::with`] ran.
    pub fn envelope(&self) -> Value {
        json!({
            "runId": self.run_id,
            "seq": self.seq,
            "eventId": self.event_id,
            "eventType": self.event_type,
            "at": self.at,
            "actor": self.actor,
            "schemaVersion": self.schema_version,
            "payload": self.payload,
        })
    }
}

/// An event that was on disk and could not be read.
///
/// Kept and reported rather than dropped silently: a history with a hole in it
/// is still usable, but only if the screen reading it knows the hole is there.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct UnreadableEvent {
    pub seq: i64,
    pub event_id: String,
    /// What is wrong with it, in words.
    pub problem: String,
}

/// An event on its way in.
#[derive(Debug, Clone)]
pub struct EventDraft {
    pub run_id: String,
    pub event_id: String,
    pub event_type: TaskEventType,
    pub at: DateTime<Utc>,
    pub actor: String,
    pub payload: Value,
}

/// The actor recorded when the application itself caused something.
pub const SYSTEM_ACTOR: &str = "system";

impl EventDraft {
    /// A new event with a fresh id and the current time.
    pub fn new(
        run_id: impl Into<String>,
        event_type: TaskEventType,
        actor: impl Into<String>,
    ) -> Self {
        Self {
            run_id: run_id.into(),
            event_id: uuid::Uuid::new_v4().to_string(),
            event_type,
            at: Utc::now(),
            actor: actor.into(),
            payload: Value::Object(Map::new()),
        }
    }

    /// A new event whose id is derived from what it is about.
    ///
    /// This is what makes a class of event idempotent rather than merely
    /// deduplicable: two writers describing the same thing — the same approval
    /// decision, the same run finishing, the same tool call settling — compute
    /// the same id without coordinating, and the second one is refused as the
    /// duplicate it is.
    ///
    /// `discriminator` must identify the thing uniquely within the run: an
    /// approval id, a tool-call id, or a constant for a once-per-run event.
    pub fn idempotent(
        run_id: impl Into<String>,
        event_type: TaskEventType,
        actor: impl Into<String>,
        discriminator: &str,
    ) -> Self {
        let run_id = run_id.into();
        let event_id = digest(&format!(
            "{run_id}\u{1f}{}\u{1f}{discriminator}",
            event_type.as_str()
        ));
        Self {
            event_id,
            ..Self::new(run_id, event_type, actor)
        }
    }

    /// Attaches a payload, redacted on the way in.
    ///
    /// There is deliberately no way to attach one that is not redacted. A
    /// caller who genuinely needs the contents kept has the task record and the
    /// workspace for that; the event stream is not the place, and making it
    /// impossible here is cheaper than reviewing every future call site.
    pub fn with(mut self, payload: Value) -> Self {
        self.payload = redact(payload);
        self
    }

    /// Uses a caller-chosen event id.
    pub fn with_event_id(mut self, event_id: impl Into<String>) -> Self {
        self.event_id = event_id.into();
        self
    }

    /// Sets the moment it happened, for events reconstructed after the fact.
    pub fn at(mut self, at: DateTime<Utc>) -> Self {
        self.at = at;
        self
    }
}

/// Payload keys whose values may carry the contents of somebody's documents,
/// the model's own words, or a credential.
///
/// Redacted wherever they appear, at any depth. The list is deliberately
/// generous — a field wrongly hashed costs a line of debugging, a field wrongly
/// kept costs a disclosure — and it is matched case-insensitively because the
/// two sides of this wire disagree about camelCase.
const CONFIDENTIAL_KEYS: &[&str] = &[
    "answer",
    "apikey",
    "args",
    "arguments",
    "body",
    "completion",
    "content",
    "credential",
    "detail",
    "excerpt",
    "expression",
    "key",
    "message",
    "passage",
    "password",
    "prompt",
    "query",
    "reasoning",
    "result",
    "secret",
    "source",
    "text",
    "thinking",
    "token",
];

/// Replaces every confidential-looking value with a hash of itself.
///
/// Structure is preserved so the shape of an event stays readable: a redacted
/// string becomes `{"sha256": "…", "chars": 812}`, which still tells a reader
/// that something was there and how much of it, and still lets two events be
/// compared for being about the same content.
pub fn redact(value: Value) -> Value {
    match value {
        Value::Object(fields) => Value::Object(
            fields
                .into_iter()
                .map(|(key, value)| {
                    if is_confidential(&key) {
                        (key, digest_of(&value))
                    } else {
                        (key, redact(value))
                    }
                })
                .collect(),
        ),
        Value::Array(items) => Value::Array(items.into_iter().map(redact).collect()),
        other => other,
    }
}

/// Whether a payload key names something that must not be stored in the clear.
pub fn is_confidential(key: &str) -> bool {
    CONFIDENTIAL_KEYS
        .iter()
        .any(|blocked| key.eq_ignore_ascii_case(blocked))
}

/// The stand-in a redacted value is replaced by.
fn digest_of(value: &Value) -> Value {
    match value {
        Value::Null => Value::Null,
        Value::String(text) => json!({
            "sha256": digest(text),
            "chars": text.chars().count(),
        }),
        other => {
            let canonical = canonical(other);
            json!({
                "sha256": digest(&canonical),
                "chars": canonical.chars().count(),
            })
        }
    }
}

/// SHA-256 of some text, hex.
pub fn digest(text: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    format!("{:x}", hasher.finalize())
}

/// One JSON value, written the same way every time.
///
/// Object keys are sorted explicitly rather than relying on `serde_json`'s map
/// being ordered: that depends on a Cargo feature, and a hash whose stability
/// depends on a feature flag somebody else can turn on is not a stable hash.
pub fn canonical(value: &Value) -> String {
    match value {
        Value::Object(fields) => {
            let mut keys: Vec<&String> = fields.keys().collect();
            keys.sort();
            let body: Vec<String> = keys
                .into_iter()
                .map(|key| format!("{}:{}", Value::String(key.clone()), canonical(&fields[key])))
                .collect();
            format!("{{{}}}", body.join(","))
        }
        Value::Array(items) => {
            let body: Vec<String> = items.iter().map(canonical).collect();
            format!("[{}]", body.join(","))
        }
        other => other.to_string(),
    }
}

/// The seal over a payload.
pub fn payload_hash(payload: &Value) -> String {
    digest(&canonical(payload))
}
