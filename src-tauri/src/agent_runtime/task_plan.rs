//! The orchestrator's plan: typed steps a run is held to, changed only by
//! validated patches and settled only by receipts.
//!
//! ## Why this is not `orchestrator::plan::PlanRun`
//!
//! `PlanRun` is the *budget* a run is held to — how many tool calls, for how
//! long, which tools — fixed by [`super::planning`] before the model is told
//! anything. It answers "may this call happen?". This answers a different
//! question: "what is the work, which parts of it are done, and how do we
//! know?". A coordinator (Spark, in the target deployment) writes it; nothing it
//! writes can widen the budget above, and nothing it writes can mark work done.
//!
//! ## Who may change what
//!
//! | Field | Model (via `task.plan_update`) | Backend |
//! |---|---|---|
//! | goal | once, when the plan is created | — |
//! | constraints, decisions, questions | append | — |
//! | corrections | never | a person's steer, recorded as it arrives |
//! | steps | add; edit or skip one that has not settled | status, outputs, receipts, attempts |
//! | receipts | never | appended when a job, tool or review settles |
//!
//! So "a model cannot erase completed receipts" is structural rather than a
//! check: no patch operation can reach the `receipts` list, a completed step
//! can be neither edited nor skipped, and a step reopened after a correction
//! keeps every receipt it had.
//!
//! ## Why patches carry a base version
//!
//! Because two writers change the plan: the coordinator, between its turns, and
//! the backend, whenever a job it dispatched settles. A patch written against
//! version 3 when version 4 already recorded a failed child would otherwise
//! silently overwrite the failure. So each patch names the version it was
//! written against, and a stale one is refused with the current plan attached —
//! the model re-reads and re-decides, which is what it should do anyway.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

/// Most steps one plan may hold. A coordinator on a 4B model that has planned
/// more than this has planned a different task, and a longer list would not fit
/// the context projection it is re-read through every round.
pub const MAX_STEPS: usize = 16;
/// The most attempts a step may be given, retries included.
pub const MAX_ATTEMPTS: u32 = 3;
/// What a step is given when the patch does not say.
pub const DEFAULT_ATTEMPTS: u32 = 2;
/// The same failure this many times in a row is no progress, and the step stops
/// being retried even with attempts left.
pub const NO_PROGRESS_LIMIT: u32 = 2;
/// Longest text any one field may hold.
pub const MAX_TEXT: usize = 600;
/// Longest list any one field may hold.
pub const MAX_LIST: usize = 12;
/// Most operations one patch may carry.
pub const MAX_OPERATIONS: usize = 24;

// ─── The plan ──────────────────────────────────────────────────────────────

/// One version of a run's plan.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TaskPlan {
    pub run_id: String,
    /// 1 for the first version. Versions are consecutive and never reused.
    pub version: u32,
    pub goal: String,
    #[serde(default)]
    pub constraints: Vec<String>,
    /// A person's corrections, oldest first. Written only by the backend.
    #[serde(default)]
    pub corrections: Vec<Correction>,
    pub steps: Vec<Step>,
    #[serde(default)]
    pub decisions: Vec<DecisionNote>,
    #[serde(default)]
    pub questions: Vec<Question>,
    /// Who wrote this version.
    pub author: Author,
    /// Why this version exists, in a line.
    pub reason: String,
    pub created_at: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", tag = "kind")]
pub enum Author {
    /// The coordinating model, through `task.plan_update`.
    Model,
    /// This process, on a settled job, tool or review.
    Backend,
    /// A person, through a steer.
    Operator { user_id: String },
}

impl Author {
    pub fn as_str(&self) -> &'static str {
        match self {
            Author::Model => "model",
            Author::Backend => "backend",
            Author::Operator { .. } => "operator",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Step {
    /// Chosen by the coordinator: lowercase letters, digits and dashes.
    pub id: String,
    pub title: String,
    pub kind: StepKind,
    /// The agent role a `delegate` step is for. Checked against the registry
    /// when the job is dispatched, not here.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(default)]
    pub depends_on: Vec<String>,
    pub acceptance: Vec<Acceptance>,
    #[serde(default)]
    pub inputs: Vec<StepRef>,
    pub status: StepStatus,
    /// What the step produced, as exact references. Backend-written.
    #[serde(default)]
    pub outputs: Vec<StepRef>,
    /// The evidence behind every status change. Backend-written, append-only.
    #[serde(default)]
    pub receipts: Vec<Receipt>,
    pub repair: RepairBudget,
    /// The job working on this step now, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub job_id: Option<String>,
    /// The plan version in which the step last reached `completed` or
    /// `awaitingReview`. A correction newer than this is what may reopen it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub settled_version: Option<u32>,
    /// Why the step is where it is, when that is not obvious.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum StepKind {
    /// Done by the coordinator itself with its own tools; settled by claiming a
    /// tool receipt.
    Direct,
    /// Handed to a specialist as a job.
    Delegate,
    /// An independent review of other steps' outputs.
    Review,
}

impl StepKind {
    pub fn as_str(self) -> &'static str {
        match self {
            StepKind::Direct => "direct",
            StepKind::Delegate => "delegate",
            StepKind::Review => "review",
        }
    }
    fn parse(raw: &str) -> Option<Self> {
        Some(match raw {
            "direct" => StepKind::Direct,
            "delegate" => StepKind::Delegate,
            "review" => StepKind::Review,
            _ => return None,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum StepStatus {
    Pending,
    /// A job is working on it.
    Running,
    /// The job finished and the step still waits on an independent review.
    AwaitingReview,
    /// Every acceptance criterion is met, each by a receipt.
    Completed,
    /// Some of the work is real and some is not; the missing part is named.
    Partial,
    /// Cannot go on without something this deployment lacks, a person, or a
    /// different approach — including "the same failure twice".
    Blocked,
    Failed,
    Cancelled,
    /// The coordinator decided it is not needed, and said why.
    Skipped,
}

impl StepStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            StepStatus::Pending => "pending",
            StepStatus::Running => "running",
            StepStatus::AwaitingReview => "awaitingReview",
            StepStatus::Completed => "completed",
            StepStatus::Partial => "partial",
            StepStatus::Blocked => "blocked",
            StepStatus::Failed => "failed",
            StepStatus::Cancelled => "cancelled",
            StepStatus::Skipped => "skipped",
        }
    }

    /// Whether the step's outcome is fixed until somebody reopens it.
    pub fn is_settled(self) -> bool {
        !matches!(self, StepStatus::Pending | StepStatus::Running)
    }

    /// Whether a later step may rely on it.
    pub fn satisfies_dependents(self) -> bool {
        matches!(self, StepStatus::Completed)
    }
}

/// What must be true, by receipt, before a step counts as done.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", tag = "kind")]
pub enum Acceptance {
    /// A job for this step completed with at least one cited finding, or an
    /// artifact at an exact version.
    ChildCompleted,
    /// Every artifact the step produced passed format-aware validation to
    /// `accepted` (P04). A reopen alone is not acceptance.
    ArtifactAccepted,
    /// An independent review of this step's outputs passed.
    ReviewPassed,
    /// The job published at least one item to the task's shared memory.
    MemoryPublished,
    /// A direct step: the named tool succeeded in this run, and the receipt was
    /// claimed for it.
    ToolSucceeded { tool: String },
}

impl Acceptance {
    pub fn describe(&self) -> String {
        match self {
            Acceptance::ChildCompleted => "a specialist completed it with cited findings".into(),
            Acceptance::ArtifactAccepted => "its artifacts passed validation".into(),
            Acceptance::ReviewPassed => "an independent review passed".into(),
            Acceptance::MemoryPublished => "its result is in shared memory".into(),
            Acceptance::ToolSucceeded { tool } => format!("{tool} succeeded (receipt claimed)"),
        }
    }

    fn parse(value: &Value) -> Result<Self, String> {
        let raw = value
            .as_str()
            .map(str::to_string)
            .or_else(|| value.get("kind").and_then(Value::as_str).map(str::to_string))
            .ok_or("each acceptance entry is a name such as \"childCompleted\"")?;
        Ok(match raw.as_str() {
            "childCompleted" | "child_completed" => Acceptance::ChildCompleted,
            "artifactAccepted" | "artifact_accepted" => Acceptance::ArtifactAccepted,
            "reviewPassed" | "review_passed" => Acceptance::ReviewPassed,
            "memoryPublished" | "memory_published" => Acceptance::MemoryPublished,
            other => {
                // `toolSucceeded:create_docx`, or `{kind: toolSucceeded, tool}`.
                let tool = other
                    .strip_prefix("toolSucceeded:")
                    .or_else(|| other.strip_prefix("tool_succeeded:"))
                    .map(str::to_string)
                    .or_else(|| {
                        (other == "toolSucceeded" || other == "tool_succeeded")
                            .then(|| value.get("tool").and_then(Value::as_str).map(str::to_string))
                            .flatten()
                    });
                match tool {
                    Some(tool) if crate::orchestrator::tools::ToolName::from_str(&tool).is_some() => {
                        Acceptance::ToolSucceeded { tool }
                    }
                    Some(tool) => return Err(format!("{tool:?} is not a tool this runtime knows")),
                    None => {
                        return Err(format!(
                            "{other:?} is not an acceptance criterion; use childCompleted, \
                             artifactAccepted, reviewPassed, memoryPublished or \
                             toolSucceeded:<tool>"
                        ))
                    }
                }
            }
        })
    }
}

/// An exact reference to something a step reads or produced.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", tag = "kind")]
pub enum StepRef {
    Artifact { artifact_id: String, version: u32 },
    Memory { item_id: String, revision: u64 },
    /// Whatever another step produced; resolved when this one is dispatched.
    Step { step_id: String },
    Document { sha256: String },
}

impl StepRef {
    pub fn describe(&self) -> String {
        match self {
            StepRef::Artifact { artifact_id, version } => format!("{artifact_id}@{version}"),
            StepRef::Memory { item_id, revision } => format!("{item_id}#r{revision}"),
            StepRef::Step { step_id } => format!("output of {step_id}"),
            StepRef::Document { sha256 } => format!("document {}", &sha256[..sha256.len().min(12)]),
        }
    }

    /// `art-1@2`, `mi-x#r3`, `step:s1`, `doc:<sha256>`.
    fn parse(raw: &str) -> Result<Self, String> {
        let raw = raw.trim();
        if let Some(step_id) = raw.strip_prefix("step:") {
            return Ok(StepRef::Step { step_id: step_id.trim().to_string() });
        }
        if let Some(sha256) = raw.strip_prefix("doc:") {
            let sha256 = sha256.trim();
            if sha256.len() == 64 && sha256.chars().all(|c| c.is_ascii_hexdigit()) {
                return Ok(StepRef::Document { sha256: sha256.to_ascii_lowercase() });
            }
            return Err(format!("{raw:?} does not name a document by its sha-256"));
        }
        if let Some((item_id, revision)) = raw.split_once("#r") {
            let revision: u64 = revision
                .parse()
                .map_err(|_| format!("{raw:?}: the revision after #r must be a whole number"))?;
            if revision == 0 || item_id.trim().is_empty() {
                return Err(format!("{raw:?} does not name a memory item at a revision"));
            }
            return Ok(StepRef::Memory { item_id: item_id.trim().to_string(), revision });
        }
        if let Some((artifact_id, version)) = raw.rsplit_once('@') {
            let version: u32 = version
                .parse()
                .map_err(|_| format!("{raw:?}: the version after @ must be a whole number"))?;
            if version == 0 || artifact_id.trim().is_empty() {
                return Err(format!("{raw:?} does not name an artifact at a version"));
            }
            return Ok(StepRef::Artifact { artifact_id: artifact_id.trim().to_string(), version });
        }
        Err(format!(
            "{raw:?} is not an exact reference; use art-…@N, mi-…#rN, step:<id> or doc:<sha256>"
        ))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Receipt {
    pub kind: ReceiptKind,
    /// The job id, the event sequence, or the review id.
    pub reference: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub event_seq: Option<i64>,
    /// The result hash (job) or output hash (tool) the event recorded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hash: Option<String>,
    /// The status the receipt reports, as the thing that wrote it spelled it.
    pub status: String,
    /// A specialist's one-line summary, or the tool's name.
    pub summary: String,
    pub at: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ReceiptKind {
    Job,
    Tool,
    Review,
}

/// How many times a step may be tried, and whether the tries are going anywhere.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RepairBudget {
    pub max_attempts: u32,
    pub attempts: u32,
    /// Consecutive settled attempts that ended with the same signature.
    pub no_progress: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_signature: Option<String>,
}

impl RepairBudget {
    fn new(max_attempts: u32) -> Self {
        Self { max_attempts, attempts: 0, no_progress: 0, last_signature: None }
    }
    pub fn remaining(&self) -> u32 {
        self.max_attempts.saturating_sub(self.attempts)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Correction {
    pub id: String,
    pub text: String,
    pub by: String,
    pub at: String,
    /// The plan version the correction arrived against.
    pub against_version: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DecisionNote {
    pub id: String,
    pub text: String,
    /// What the decision rests on: job ids, `art-…@N`, `mi-…#rN`, `[E3]`.
    /// References, not restated facts.
    #[serde(default)]
    pub evidence: Vec<String>,
    pub at: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Question {
    pub id: String,
    pub text: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub answer: Option<String>,
}

impl TaskPlan {
    pub fn step(&self, id: &str) -> Option<&Step> {
        self.steps.iter().find(|step| step.id == id)
    }

    fn step_mut(&mut self, id: &str) -> Option<&mut Step> {
        self.steps.iter_mut().find(|step| step.id == id)
    }

    /// SHA-256 over the canonical body, so a stored version can be checked.
    pub fn digest(&self) -> String {
        let body = serde_json::to_value(self).unwrap_or(Value::Null);
        let mut hasher = Sha256::new();
        hasher.update(crate::agent_runtime::events::canonical(&body).as_bytes());
        format!("{:x}", hasher.finalize())
    }

    /// The next version, empty of authorship until the caller sets it.
    fn successor(&self, author: Author, reason: impl Into<String>, at: &str) -> TaskPlan {
        let mut next = self.clone();
        next.version = self.version + 1;
        next.author = author;
        next.reason = reason.into();
        next.created_at = at.to_string();
        next
    }

    /// Steps whose prerequisites are all complete and that nothing is working on.
    pub fn ready(&self) -> Vec<&Step> {
        self.steps
            .iter()
            .filter(|step| step.status == StepStatus::Pending)
            .filter(|step| self.unmet_dependencies(step).is_empty())
            .collect()
    }

    /// The prerequisites of `step` that are not complete, by id and status.
    pub fn unmet_dependencies(&self, step: &Step) -> Vec<(String, StepStatus)> {
        step.depends_on
            .iter()
            .filter_map(|id| {
                let status = self.step(id).map(|dep| dep.status).unwrap_or(StepStatus::Failed);
                (!status.satisfies_dependents()).then(|| (id.clone(), status))
            })
            .collect()
    }
}

// ─── Patches from the model ────────────────────────────────────────────────

/// A change to the plan, as the coordinating model asked for it.
#[derive(Debug, Clone, PartialEq)]
pub struct Patch {
    pub base_version: u32,
    pub operations: Vec<Operation>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Operation {
    SetGoal { text: String },
    AddConstraint { text: String },
    AddStep {
        id: String,
        title: String,
        kind: StepKind,
        role: Option<String>,
        depends_on: Vec<String>,
        acceptance: Vec<Acceptance>,
        inputs: Vec<StepRef>,
        max_attempts: Option<u32>,
    },
    UpdateStep {
        id: String,
        title: Option<String>,
        depends_on: Option<Vec<String>>,
        acceptance: Option<Vec<Acceptance>>,
        inputs: Option<Vec<StepRef>>,
        max_attempts: Option<u32>,
    },
    SkipStep { id: String, reason: String },
    /// Back to pending after a failure, a block, a cancellation, a partial
    /// result — or, only after a correction arrived, a completion.
    ReopenStep { id: String, reason: String },
    /// Settles a direct step on a tool receipt this run holds.
    ClaimReceipt { id: String, event_seq: i64 },
    RecordDecision { text: String, evidence: Vec<String> },
    AddQuestion { text: String },
    ResolveQuestion { id: String, answer: String },
}

/// Why a patch was not applied.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Refusal {
    /// The patch was written against an older version.
    Stale { base: u32, current: u32 },
    /// The patch itself is malformed or asks for something not permitted.
    Invalid(String),
}

impl Refusal {
    pub fn explain(&self) -> String {
        match self {
            Refusal::Stale { base, current } => format!(
                "This patch was written against plan version {base}, and the plan is now at \
                 version {current}. Nothing was changed. Read the current plan below and write \
                 the patch against version {current}."
            ),
            Refusal::Invalid(reason) => format!("{reason} Nothing was changed."),
        }
    }
}

/// A tool receipt the event log holds, as `ClaimReceipt` needs to check it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolReceipt {
    pub tool: String,
    pub output_sha256: Option<String>,
}

/// Reads the durable record, so a claim is checked rather than believed.
pub trait ReceiptSource {
    /// A `tool_succeeded` event in this run at this sequence, if there is one.
    fn tool_succeeded(&self, run_id: &str, event_seq: i64) -> Option<ToolReceipt>;
}

fn text_field(value: &Value, key: &str, required: bool) -> Result<Option<String>, String> {
    match value.get(key) {
        None | Some(Value::Null) if !required => Ok(None),
        None | Some(Value::Null) => Err(format!("`{key}` is required.")),
        Some(Value::String(text)) => {
            let text = text.trim();
            if text.is_empty() {
                return Err(format!("`{key}` is empty."));
            }
            if text.chars().count() > MAX_TEXT {
                return Err(format!("`{key}` is longer than {MAX_TEXT} characters; say it shorter."));
            }
            Ok(Some(text.to_string()))
        }
        Some(_) => Err(format!("`{key}` must be text.")),
    }
}

fn list_field(value: &Value, key: &str) -> Result<Option<Vec<Value>>, String> {
    match value.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Array(items)) if items.len() <= MAX_LIST => Ok(Some(items.clone())),
        Some(Value::Array(_)) => Err(format!("`{key}` holds more than {MAX_LIST} entries.")),
        Some(_) => Err(format!("`{key}` must be a list.")),
    }
}

fn strings(items: Vec<Value>, key: &str) -> Result<Vec<String>, String> {
    items
        .into_iter()
        .map(|item| match item {
            Value::String(text) if !text.trim().is_empty() && text.chars().count() <= MAX_TEXT => {
                Ok(text.trim().to_string())
            }
            _ => Err(format!("every entry in `{key}` must be short, non-empty text.")),
        })
        .collect()
}

fn valid_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 24
        && id.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
}

fn attempts_field(value: &Value) -> Result<Option<u32>, String> {
    match value.get("max_attempts").or_else(|| value.get("maxAttempts")) {
        None | Some(Value::Null) => Ok(None),
        Some(raw) => match raw.as_u64() {
            Some(n) if (1..=MAX_ATTEMPTS as u64).contains(&n) => Ok(Some(n as u32)),
            _ => Err(format!("`max_attempts` must be between 1 and {MAX_ATTEMPTS}.")),
        },
    }
}

fn refs_field(value: &Value, key: &str) -> Result<Option<Vec<StepRef>>, String> {
    list_field(value, key)?
        .map(|items| {
            strings(items, key)?
                .iter()
                .map(|raw| StepRef::parse(raw))
                .collect::<Result<Vec<_>, _>>()
        })
        .transpose()
}

fn acceptance_field(value: &Value) -> Result<Option<Vec<Acceptance>>, String> {
    list_field(value, "acceptance")?
        .map(|items| items.iter().map(Acceptance::parse).collect::<Result<Vec<_>, _>>())
        .transpose()
}

/// Reads a patch from `task.plan_update`'s arguments.
///
/// Strict on purpose: an operation with a field it does not recognise is
/// refused rather than applied without it, for the reason the gateway refuses
/// undeclared arguments — a silently dropped `depends_on` is a plan that runs
/// in the wrong order.
pub fn parse_patch(arguments: &Value) -> Result<Patch, String> {
    let base_version = arguments
        .get("base_version")
        .and_then(Value::as_u64)
        .ok_or("`base_version` is required: 0 to create the plan, otherwise the version you read.")?
        as u32;
    let operations = arguments
        .get("operations")
        .and_then(Value::as_array)
        .ok_or("`operations` is required: a list of changes, each with an `op`.")?;
    if operations.is_empty() {
        return Err("`operations` is empty, so there is nothing to change.".into());
    }
    if operations.len() > MAX_OPERATIONS {
        return Err(format!("A patch may carry at most {MAX_OPERATIONS} operations."));
    }

    let mut parsed = Vec::with_capacity(operations.len());
    for (index, operation) in operations.iter().enumerate() {
        let op = operation
            .get("op")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("operation {} has no `op`.", index + 1))?;
        let allowed: &[&str] = match op {
            "set_goal" | "add_constraint" | "add_question" => &["op", "text"],
            "add_step" => &[
                "op", "id", "title", "kind", "role", "depends_on", "acceptance", "inputs",
                "max_attempts",
            ],
            "update_step" => &["op", "id", "title", "depends_on", "acceptance", "inputs", "max_attempts"],
            "skip_step" | "reopen_step" => &["op", "id", "reason"],
            "claim_receipt" => &["op", "id", "event_seq"],
            "record_decision" => &["op", "text", "evidence"],
            "resolve_question" => &["op", "id", "answer"],
            other => {
                return Err(format!(
                    "operation {}: {other:?} is not an operation. Use set_goal, add_constraint, \
                     add_step, update_step, skip_step, reopen_step, claim_receipt, \
                     record_decision, add_question or resolve_question.",
                    index + 1
                ))
            }
        };
        if let Some(fields) = operation.as_object() {
            if let Some(unknown) = fields.keys().find(|key| !allowed.contains(&key.as_str())) {
                return Err(format!(
                    "operation {} ({op}) has no {unknown:?} field; it takes {}.",
                    index + 1,
                    allowed[1..].join(", ")
                ));
            }
        }
        let at = |error: String| format!("operation {} ({op}): {error}", index + 1);
        let id = || -> Result<String, String> {
            let id = text_field(operation, "id", true).map_err(at)?.unwrap_or_default();
            if !valid_id(&id) {
                return Err(at(format!(
                    "{id:?} is not a step id; use up to 24 lowercase letters, digits and dashes."
                )));
            }
            Ok(id)
        };
        parsed.push(match op {
            "set_goal" => Operation::SetGoal {
                text: text_field(operation, "text", true).map_err(at)?.unwrap_or_default(),
            },
            "add_constraint" => Operation::AddConstraint {
                text: text_field(operation, "text", true).map_err(at)?.unwrap_or_default(),
            },
            "add_question" => Operation::AddQuestion {
                text: text_field(operation, "text", true).map_err(at)?.unwrap_or_default(),
            },
            "add_step" => {
                let kind_raw = text_field(operation, "kind", true).map_err(at)?.unwrap_or_default();
                let kind = StepKind::parse(&kind_raw)
                    .ok_or_else(|| at(format!("{kind_raw:?} is not a step kind; use direct, delegate or review.")))?;
                let role = text_field(operation, "role", false).map_err(at)?;
                if kind == StepKind::Delegate && role.is_none() {
                    return Err(at("a delegate step names the `role` that will do it.".into()));
                }
                let acceptance = acceptance_field(operation).map_err(at)?.unwrap_or_default();
                Operation::AddStep {
                    id: id()?,
                    title: text_field(operation, "title", true).map_err(at)?.unwrap_or_default(),
                    kind,
                    role,
                    depends_on: list_field(operation, "depends_on")
                        .and_then(|items| items.map(|items| strings(items, "depends_on")).transpose())
                        .map_err(at)?
                        .unwrap_or_default(),
                    acceptance,
                    inputs: refs_field(operation, "inputs").map_err(at)?.unwrap_or_default(),
                    max_attempts: attempts_field(operation).map_err(at)?,
                }
            }
            "update_step" => Operation::UpdateStep {
                id: id()?,
                title: text_field(operation, "title", false).map_err(at)?,
                depends_on: list_field(operation, "depends_on")
                    .and_then(|items| items.map(|items| strings(items, "depends_on")).transpose())
                    .map_err(at)?,
                acceptance: acceptance_field(operation).map_err(at)?,
                inputs: refs_field(operation, "inputs").map_err(at)?,
                max_attempts: attempts_field(operation).map_err(at)?,
            },
            "skip_step" => Operation::SkipStep {
                id: id()?,
                reason: text_field(operation, "reason", true).map_err(at)?.unwrap_or_default(),
            },
            "reopen_step" => Operation::ReopenStep {
                id: id()?,
                reason: text_field(operation, "reason", true).map_err(at)?.unwrap_or_default(),
            },
            "claim_receipt" => Operation::ClaimReceipt {
                id: id()?,
                event_seq: operation
                    .get("event_seq")
                    .and_then(Value::as_i64)
                    .filter(|seq| *seq > 0)
                    .ok_or_else(|| at("`event_seq` is the sequence number of the tool's receipt.".into()))?,
            },
            "record_decision" => Operation::RecordDecision {
                text: text_field(operation, "text", true).map_err(at)?.unwrap_or_default(),
                evidence: list_field(operation, "evidence")
                    .and_then(|items| items.map(|items| strings(items, "evidence")).transpose())
                    .map_err(at)?
                    .unwrap_or_default(),
            },
            "resolve_question" => Operation::ResolveQuestion {
                id: text_field(operation, "id", true).map_err(at)?.unwrap_or_default(),
                answer: text_field(operation, "answer", true).map_err(at)?.unwrap_or_default(),
            },
            _ => unreachable!("every op was matched above"),
        });
    }
    Ok(Patch { base_version, operations: parsed })
}

/// Applies a model's patch to the current plan, or says why not.
///
/// `current` is `None` only when the run has no plan yet, and then the patch
/// must be written against version 0 and must set the goal.
pub fn apply_patch(
    run_id: &str,
    current: Option<&TaskPlan>,
    patch: &Patch,
    receipts: &dyn ReceiptSource,
    at: &str,
) -> Result<TaskPlan, Refusal> {
    let current_version = current.map(|plan| plan.version).unwrap_or(0);
    if patch.base_version != current_version {
        return Err(Refusal::Stale { base: patch.base_version, current: current_version });
    }
    let invalid = |reason: String| Refusal::Invalid(reason);

    let mut next = match current {
        Some(plan) => plan.successor(Author::Model, String::new(), at),
        None => TaskPlan {
            run_id: run_id.to_string(),
            version: 1,
            goal: String::new(),
            constraints: Vec::new(),
            corrections: Vec::new(),
            steps: Vec::new(),
            decisions: Vec::new(),
            questions: Vec::new(),
            author: Author::Model,
            reason: String::new(),
            created_at: at.to_string(),
        },
    };
    let latest_correction_version = next.corrections.last().map(|c| c.against_version);
    let mut summary: Vec<String> = Vec::new();

    for operation in &patch.operations {
        match operation {
            Operation::SetGoal { text } => {
                if current.is_some() && !next.goal.is_empty() {
                    return Err(invalid(
                        "The goal is set once, when the plan is created. A change of goal comes \
                         from the person, as a correction; record your reading of it as a \
                         decision instead."
                            .into(),
                    ));
                }
                next.goal = text.clone();
                summary.push("set the goal".into());
            }
            Operation::AddConstraint { text } => {
                if next.constraints.len() >= MAX_LIST {
                    return Err(invalid(format!("The plan already holds {MAX_LIST} constraints.")));
                }
                next.constraints.push(text.clone());
                summary.push("added a constraint".into());
            }
            Operation::AddStep { id, title, kind, role, depends_on, acceptance, inputs, max_attempts } => {
                if next.step(id).is_some() {
                    return Err(invalid(format!("There is already a step {id:?}.")));
                }
                if next.steps.len() >= MAX_STEPS {
                    return Err(invalid(format!(
                        "A plan holds at most {MAX_STEPS} steps; skip or merge some first."
                    )));
                }
                let acceptance = if acceptance.is_empty() {
                    default_acceptance(*kind)
                } else {
                    acceptance.clone()
                };
                check_acceptance(*kind, &acceptance).map_err(|reason| invalid(format!("Step {id}: {reason}")))?;
                next.steps.push(Step {
                    id: id.clone(),
                    title: title.clone(),
                    kind: *kind,
                    role: role.clone(),
                    depends_on: depends_on.clone(),
                    acceptance,
                    inputs: inputs.clone(),
                    status: StepStatus::Pending,
                    outputs: Vec::new(),
                    receipts: Vec::new(),
                    repair: RepairBudget::new(max_attempts.unwrap_or(DEFAULT_ATTEMPTS)),
                    job_id: None,
                    settled_version: None,
                    note: None,
                });
                summary.push(format!("added {id}"));
            }
            Operation::UpdateStep { id, title, depends_on, acceptance, inputs, max_attempts } => {
                let step = next
                    .step_mut(id)
                    .ok_or_else(|| invalid(format!("There is no step {id:?}.")))?;
                if !matches!(step.status, StepStatus::Pending) {
                    return Err(invalid(format!(
                        "Step {id} is {}; only a pending step can be edited. Reopen it first if \
                         it failed, or add a new step.",
                        step.status.as_str()
                    )));
                }
                if let Some(title) = title {
                    step.title = title.clone();
                }
                if let Some(depends_on) = depends_on {
                    step.depends_on = depends_on.clone();
                }
                if let Some(acceptance) = acceptance {
                    let acceptance = if acceptance.is_empty() {
                        default_acceptance(step.kind)
                    } else {
                        acceptance.clone()
                    };
                    check_acceptance(step.kind, &acceptance)
                        .map_err(|reason| invalid(format!("Step {id}: {reason}")))?;
                    step.acceptance = acceptance;
                }
                if let Some(inputs) = inputs {
                    step.inputs = inputs.clone();
                }
                if let Some(max) = max_attempts {
                    if *max < step.repair.attempts {
                        return Err(invalid(format!(
                            "Step {id} has already used {} attempt(s).",
                            step.repair.attempts
                        )));
                    }
                    step.repair.max_attempts = *max;
                }
                summary.push(format!("edited {id}"));
            }
            Operation::SkipStep { id, reason } => {
                let step = next
                    .step_mut(id)
                    .ok_or_else(|| invalid(format!("There is no step {id:?}.")))?;
                match step.status {
                    StepStatus::Completed | StepStatus::AwaitingReview => {
                        return Err(invalid(format!(
                            "Step {id} has settled with receipts and cannot be skipped; its \
                             receipts stay in the plan."
                        )))
                    }
                    StepStatus::Running => {
                        return Err(invalid(format!(
                            "Step {id} has a job running; cancel it with agent.cancel first."
                        )))
                    }
                    _ => {}
                }
                step.status = StepStatus::Skipped;
                step.note = Some(reason.clone());
                summary.push(format!("skipped {id}"));
            }
            Operation::ReopenStep { id, reason } => {
                let step = next
                    .step_mut(id)
                    .ok_or_else(|| invalid(format!("There is no step {id:?}.")))?;
                match step.status {
                    StepStatus::Failed
                    | StepStatus::Partial
                    | StepStatus::Cancelled
                    | StepStatus::Skipped => {}
                    StepStatus::Blocked if step.repair.no_progress >= NO_PROGRESS_LIMIT => {
                        return Err(invalid(format!(
                            "Step {id} failed the same way {} times in a row. Retrying the same \
                             job will not help: change its inputs or objective in a new step, \
                             or report it unfinished.",
                            step.repair.no_progress
                        )))
                    }
                    StepStatus::Blocked => {}
                    StepStatus::Completed | StepStatus::AwaitingReview => {
                        // Only a correction that arrived after the step settled
                        // can reopen it. Its receipts stay: a reopened step is
                        // redone, not rewritten.
                        let settled = step.settled_version.unwrap_or(0);
                        let corrected_after = latest_correction_version
                            .is_some_and(|version| version > settled);
                        if !corrected_after {
                            return Err(invalid(format!(
                                "Step {id} is complete. A completed step is reopened only after \
                                 a person's correction arrives; its receipts are kept either way."
                            )));
                        }
                    }
                    StepStatus::Pending | StepStatus::Running => {
                        return Err(invalid(format!(
                            "Step {id} is {}; there is nothing to reopen.",
                            step.status.as_str()
                        )))
                    }
                }
                let was_complete = matches!(step.status, StepStatus::Completed | StepStatus::AwaitingReview);
                if step.repair.remaining() == 0 && !was_complete {
                    return Err(invalid(format!(
                        "Step {id} has used all {} of its attempts. Report it unfinished, or add \
                         a different step.",
                        step.repair.max_attempts
                    )));
                }
                if was_complete {
                    // Redoing a completed step after a correction is new work,
                    // not a repair of a failure, so it gets its budget back.
                    step.repair.attempts = 0;
                    step.repair.no_progress = 0;
                    step.repair.last_signature = None;
                }
                step.status = StepStatus::Pending;
                step.job_id = None;
                step.note = Some(format!("reopened: {reason}"));
                summary.push(format!("reopened {id}"));
            }
            Operation::ClaimReceipt { id, event_seq } => {
                let version = next.version;
                let step = next
                    .step_mut(id)
                    .ok_or_else(|| invalid(format!("There is no step {id:?}.")))?;
                if step.kind != StepKind::Direct {
                    return Err(invalid(format!(
                        "Step {id} is a {} step; it is settled by its job or review, not by a \
                         claimed receipt.",
                        step.kind.as_str()
                    )));
                }
                if step.status == StepStatus::Completed {
                    return Err(invalid(format!("Step {id} is already complete.")));
                }
                let receipt = receipts.tool_succeeded(run_id, *event_seq).ok_or_else(|| {
                    invalid(format!(
                        "Event {event_seq} is not a successful tool call in this run, so it \
                         cannot settle step {id}."
                    ))
                })?;
                let wanted: Vec<&str> = step
                    .acceptance
                    .iter()
                    .filter_map(|criterion| match criterion {
                        Acceptance::ToolSucceeded { tool } => Some(tool.as_str()),
                        _ => None,
                    })
                    .collect();
                if !wanted.contains(&receipt.tool.as_str()) {
                    return Err(invalid(format!(
                        "Event {event_seq} is a {} receipt; step {id} is settled by {}.",
                        receipt.tool,
                        wanted.join(" or ")
                    )));
                }
                if step.receipts.iter().any(|held| held.event_seq == Some(*event_seq)) {
                    return Err(invalid(format!("Event {event_seq} is already claimed for {id}.")));
                }
                step.receipts.push(Receipt {
                    kind: ReceiptKind::Tool,
                    reference: event_seq.to_string(),
                    event_seq: Some(*event_seq),
                    hash: receipt.output_sha256.clone(),
                    status: "succeeded".into(),
                    summary: receipt.tool.clone(),
                    at: at.to_string(),
                });
                // The step's tool criteria are met when each named tool has a
                // claimed receipt. A review or an accepted artifact asked for on
                // top holds it at `awaitingReview` until `task.request_review`
                // passes.
                let satisfied = step.acceptance.iter().all(|criterion| match criterion {
                    Acceptance::ToolSucceeded { tool } => step
                        .receipts
                        .iter()
                        .any(|held| held.kind == ReceiptKind::Tool && &held.summary == tool),
                    _ => true,
                });
                let wants_review = step.acceptance.iter().any(|criterion| {
                    matches!(criterion, Acceptance::ReviewPassed | Acceptance::ArtifactAccepted)
                });
                if satisfied {
                    step.status = if wants_review {
                        StepStatus::AwaitingReview
                    } else {
                        StepStatus::Completed
                    };
                    step.settled_version = Some(version);
                    step.note = None;
                }
                summary.push(format!("claimed event {event_seq} for {id}"));
            }
            Operation::RecordDecision { text, evidence } => {
                if next.decisions.len() >= MAX_LIST * 2 {
                    return Err(invalid("The plan already holds its limit of decisions.".into()));
                }
                let id = format!("d{}", next.decisions.len() + 1);
                next.decisions.push(DecisionNote {
                    id: id.clone(),
                    text: text.clone(),
                    evidence: evidence.clone(),
                    at: at.to_string(),
                });
                summary.push(format!("recorded decision {id}"));
            }
            Operation::AddQuestion { text } => {
                if next.questions.len() >= MAX_LIST {
                    return Err(invalid(format!("The plan already holds {MAX_LIST} questions.")));
                }
                let id = format!("q{}", next.questions.len() + 1);
                next.questions.push(Question { id: id.clone(), text: text.clone(), answer: None });
                summary.push(format!("asked {id}"));
            }
            Operation::ResolveQuestion { id, answer } => {
                let question = next
                    .questions
                    .iter_mut()
                    .find(|question| &question.id == id)
                    .ok_or_else(|| invalid(format!("There is no question {id:?}.")))?;
                question.answer = Some(answer.clone());
                summary.push(format!("answered {id}"));
            }
        }
    }

    if next.goal.trim().is_empty() {
        return Err(invalid(
            "A plan starts with its goal: include {\"op\": \"set_goal\", \"text\": …} when \
             creating it."
                .into(),
        ));
    }
    validate_graph(&next).map_err(invalid)?;
    next.reason = summary.join("; ");
    Ok(next)
}

fn default_acceptance(kind: StepKind) -> Vec<Acceptance> {
    match kind {
        StepKind::Delegate => vec![Acceptance::ChildCompleted],
        StepKind::Review => vec![Acceptance::ReviewPassed],
        // A direct step with no named tool has nothing to settle it; the check
        // below refuses it rather than invent one.
        StepKind::Direct => Vec::new(),
    }
}

fn check_acceptance(kind: StepKind, acceptance: &[Acceptance]) -> Result<(), String> {
    match kind {
        StepKind::Direct => {
            // Settled by the receipts of the tools the coordinator calls, and
            // optionally held for a review or an accepted artifact on top.
            let tools = acceptance
                .iter()
                .filter(|criterion| matches!(criterion, Acceptance::ToolSucceeded { .. }))
                .count();
            let others_allowed = acceptance.iter().all(|criterion| {
                matches!(
                    criterion,
                    Acceptance::ToolSucceeded { .. }
                        | Acceptance::ReviewPassed
                        | Acceptance::ArtifactAccepted
                )
            });
            if tools == 0 || !others_allowed {
                return Err(
                    "a direct step is settled by the receipts of the tools you call for it; \
                     give its acceptance as toolSucceeded:<tool>, optionally with reviewPassed \
                     or artifactAccepted."
                        .into(),
                );
            }
        }
        StepKind::Delegate => {
            if acceptance.iter().any(|criterion| matches!(criterion, Acceptance::ToolSucceeded { .. })) {
                return Err("a delegate step is settled by its job, not by your own tool receipts.".into());
            }
        }
        StepKind::Review => {
            if acceptance != [Acceptance::ReviewPassed] {
                return Err("a review step is settled by its review; its acceptance is reviewPassed.".into());
            }
        }
    }
    Ok(())
}

/// Every dependency names a step, nothing depends on itself, and there is no
/// cycle. A plan with a cycle has a step that can never become ready, and the
/// run would report it "pending" forever.
fn validate_graph(plan: &TaskPlan) -> Result<(), String> {
    for step in &plan.steps {
        for dependency in &step.depends_on {
            if dependency == &step.id {
                return Err(format!("Step {} depends on itself.", step.id));
            }
            if plan.step(dependency).is_none() {
                return Err(format!("Step {} depends on {dependency:?}, which is not a step.", step.id));
            }
        }
        for input in &step.inputs {
            if let StepRef::Step { step_id } = input {
                if plan.step(step_id).is_none() {
                    return Err(format!("Step {} reads the output of {step_id:?}, which is not a step.", step.id));
                }
                if !step.depends_on.contains(step_id) {
                    return Err(format!(
                        "Step {} reads the output of {step_id} without depending on it; add it to \
                         depends_on so it waits.",
                        step.id
                    ));
                }
            }
        }
    }
    // Kahn's algorithm over the dependency edges.
    let mut remaining: Vec<&Step> = plan.steps.iter().collect();
    let mut placed: Vec<&str> = Vec::new();
    while !remaining.is_empty() {
        let before = remaining.len();
        remaining.retain(|step| {
            if step.depends_on.iter().all(|dep| placed.contains(&dep.as_str())) {
                placed.push(step.id.as_str());
                false
            } else {
                true
            }
        });
        if remaining.len() == before {
            return Err(format!(
                "The steps {} depend on each other in a cycle, so none of them could ever start.",
                remaining.iter().map(|step| step.id.as_str()).collect::<Vec<_>>().join(", ")
            ));
        }
    }
    Ok(())
}

// ─── Settlement by the backend ─────────────────────────────────────────────

/// What a dispatched job came back with, reduced to what a step needs.
#[derive(Debug, Clone, PartialEq)]
pub struct JobSettlement {
    pub job_id: String,
    /// The manager's status, spelled as `ChildStatus::as_str` spells it.
    pub status: String,
    pub findings: usize,
    /// Findings citing a source passage.
    pub evidenced: usize,
    /// Durable tool receipts the child's work rests on (a calculation the
    /// engine ran, a file re-opened). A finding backed by one is checkable
    /// even with no passage to cite.
    pub receipts: usize,
    pub published: Vec<StepRef>,
    pub artifacts: Vec<StepRef>,
    /// Whether every artifact above passed validation to `accepted`. `None`
    /// when there were no artifacts.
    pub artifacts_accepted: Option<bool>,
    pub result_hash: String,
    pub event_seq: Option<i64>,
    /// One line, the specialist's own account.
    pub summary: String,
    /// What it did not do, or the prerequisite it lacked.
    pub missing: Vec<String>,
    /// Whether it was stopped because a correction superseded it. Such an
    /// attempt is not charged against the step's budget.
    pub superseded: bool,
}

/// A step's job started.
pub fn job_started(plan: &TaskPlan, step_id: &str, job_id: &str, at: &str) -> Option<TaskPlan> {
    let mut next = plan.successor(Author::Backend, format!("job {job_id} started for {step_id}"), at);
    let step = next.step_mut(step_id)?;
    // Only a pending step starts a job: a second call racing the first for the
    // same step finds it running and is refused.
    if step.status != StepStatus::Pending {
        return None;
    }
    step.status = StepStatus::Running;
    step.job_id = Some(job_id.to_string());
    step.repair.attempts += 1;
    step.note = None;
    Some(next)
}

/// A step's job settled. Decides the step's status from receipts, never from
/// the specialist's own description of its work.
pub fn job_settled(plan: &TaskPlan, step_id: &str, settled: &JobSettlement, at: &str) -> Option<TaskPlan> {
    let mut next = plan.successor(
        Author::Backend,
        format!("job {} settled {} for {step_id}", settled.job_id, settled.status),
        at,
    );
    let version = next.version;
    let step = next.step_mut(step_id)?;
    if step.job_id.as_deref() != Some(settled.job_id.as_str()) {
        // A job the step has moved on from (reopened, re-dispatched). Its
        // receipt is still recorded; its status decides nothing.
        step.receipts.push(receipt_of(settled, at));
        next.reason = format!("late settlement of job {} for {step_id} recorded", settled.job_id);
        return Some(next);
    }
    step.job_id = None;
    step.receipts.push(receipt_of(settled, at));
    step.outputs.extend(settled.published.iter().cloned());
    step.outputs.extend(settled.artifacts.iter().cloned());
    step.outputs.dedup();

    if settled.superseded {
        step.repair.attempts = step.repair.attempts.saturating_sub(1);
        step.status = StepStatus::Pending;
        step.note = Some("stopped because a correction arrived; it will be dispatched again".into());
        return Some(next);
    }

    let completed = settled.status == "completed";
    let cited = settled.findings > 0 && (settled.evidenced > 0 || settled.receipts > 0);
    let produced = !settled.artifacts.is_empty();
    let mut unmet: Vec<String> = Vec::new();
    let mut awaiting_review = false;
    for criterion in &step.acceptance {
        match criterion {
            Acceptance::ChildCompleted if !(completed && (cited || produced)) => unmet.push(if completed {
                "the specialist said it finished but returned no cited finding and no artifact".into()
            } else {
                format!("the specialist ended {}", settled.status)
            }),
            Acceptance::MemoryPublished if settled.published.is_empty() => {
                unmet.push("nothing was published to shared memory".into())
            }
            Acceptance::ArtifactAccepted => match settled.artifacts_accepted {
                Some(true) => {}
                Some(false) => unmet.push("an artifact did not pass validation to accepted".into()),
                None => unmet.push("no artifact was produced to validate".into()),
            },
            Acceptance::ReviewPassed => awaiting_review = true,
            _ => {}
        }
    }

    let signature = signature_of(settled);
    if !completed || !unmet.is_empty() {
        if step.repair.last_signature.as_deref() == Some(signature.as_str()) {
            step.repair.no_progress += 1;
        } else {
            step.repair.no_progress = 1;
        }
    } else {
        step.repair.no_progress = 0;
    }
    step.repair.last_signature = Some(signature);

    step.status = match settled.status.as_str() {
        "completed" if unmet.is_empty() && awaiting_review => StepStatus::AwaitingReview,
        "completed" if unmet.is_empty() => StepStatus::Completed,
        // Finished by its own account and short by ours. Kept, not accepted.
        "completed" | "partial" => StepStatus::Partial,
        "cancelled" => StepStatus::Cancelled,
        "blocked" | "refused" => StepStatus::Blocked,
        _ => StepStatus::Failed,
    };
    if matches!(step.status, StepStatus::Completed | StepStatus::AwaitingReview) {
        step.settled_version = Some(version);
    }
    let mut notes = unmet;
    notes.extend(settled.missing.iter().cloned());
    if matches!(step.status, StepStatus::Failed | StepStatus::Partial) {
        if step.repair.no_progress >= NO_PROGRESS_LIMIT {
            step.status = StepStatus::Blocked;
            notes.push(format!(
                "no progress: the same outcome {} times in a row",
                step.repair.no_progress
            ));
        } else if step.repair.remaining() == 0 {
            step.status = StepStatus::Blocked;
            notes.push(format!("all {} attempts used", step.repair.max_attempts));
        }
    }
    step.note = (!notes.is_empty()).then(|| notes.join("; "));
    Some(next)
}

/// A review of `step_id`'s outputs settled.
pub fn review_settled(
    plan: &TaskPlan,
    step_id: &str,
    review_id: &str,
    passed: bool,
    summary: &str,
    event_seq: Option<i64>,
    at: &str,
) -> Option<TaskPlan> {
    let mut next = plan.successor(
        Author::Backend,
        format!("review {review_id} of {step_id} {}", if passed { "passed" } else { "failed" }),
        at,
    );
    let version = next.version;
    let step = next.step_mut(step_id)?;
    step.receipts.push(Receipt {
        kind: ReceiptKind::Review,
        reference: review_id.to_string(),
        event_seq,
        hash: None,
        status: if passed { "passed" } else { "failed" }.into(),
        summary: summary.chars().take(240).collect(),
        at: at.to_string(),
    });
    match (step.kind, step.status, passed) {
        (StepKind::Review, _, true) => {
            step.status = StepStatus::Completed;
            step.settled_version = Some(version);
            step.note = None;
        }
        (StepKind::Review, _, false) => {
            step.status = StepStatus::Failed;
            step.note = Some(summary.chars().take(240).collect());
        }
        (_, StepStatus::AwaitingReview, true) => {
            step.status = StepStatus::Completed;
            step.settled_version = Some(version);
            step.note = None;
        }
        (_, StepStatus::AwaitingReview, false) => {
            step.status = StepStatus::Partial;
            step.note = Some(format!("review failed: {}", summary.chars().take(200).collect::<String>()));
        }
        // A review of a step that was not waiting for one is recorded as a
        // receipt and changes nothing else.
        _ => {}
    }
    Some(next)
}

/// A person's correction, recorded against the plan.
///
/// Steps with a job running are marked for re-dispatch by the caller, which
/// cancels the jobs; this only records the correction and says which steps are
/// affected.
pub fn correction_recorded(plan: &TaskPlan, text: &str, by: &str, at: &str) -> (TaskPlan, Vec<String>) {
    let mut next = plan.successor(Author::Operator { user_id: by.to_string() }, "a correction arrived", at);
    let id = format!("c{}", next.corrections.len() + 1);
    next.corrections.push(Correction {
        id,
        text: text.chars().take(MAX_TEXT).collect(),
        by: by.to_string(),
        at: at.to_string(),
        against_version: next.version,
    });
    let running = next
        .steps
        .iter()
        .filter(|step| step.status == StepStatus::Running)
        .filter_map(|step| step.job_id.clone())
        .collect();
    (next, running)
}

/// Settles a running step whose job this process no longer holds — the process
/// that ran it went away.
///
/// A read-only job is safe to try again, so it becomes `failed` and the
/// coordinator may reopen it within its budget. A writer job's effects are
/// unknown, so it becomes `blocked` for a person to reconcile.
pub fn job_interrupted(plan: &TaskPlan, step_id: &str, job_id: &str, writer: bool, at: &str) -> Option<TaskPlan> {
    let mut next = plan.successor(Author::Backend, format!("job {job_id} for {step_id} was interrupted"), at);
    let step = next.step_mut(step_id)?;
    if step.job_id.as_deref() != Some(job_id) || step.status != StepStatus::Running {
        return None;
    }
    step.job_id = None;
    step.receipts.push(Receipt {
        kind: ReceiptKind::Job,
        reference: job_id.to_string(),
        event_seq: None,
        hash: None,
        status: "interrupted".into(),
        summary: "the process running this job stopped before it settled".into(),
        at: at.to_string(),
    });
    if writer {
        step.status = StepStatus::Blocked;
        step.note = Some(
            "interrupted while it could write; what it did is unknown and a person must \
             reconcile it before it is run again"
                .into(),
        );
    } else {
        step.status = StepStatus::Failed;
        step.note = Some("interrupted by a restart; it read only, so it can be reopened".into());
    }
    Some(next)
}

fn receipt_of(settled: &JobSettlement, at: &str) -> Receipt {
    Receipt {
        kind: ReceiptKind::Job,
        reference: settled.job_id.clone(),
        event_seq: settled.event_seq,
        hash: Some(settled.result_hash.clone()),
        status: settled.status.clone(),
        summary: settled.summary.chars().take(240).collect(),
        at: at.to_string(),
    }
}

/// What makes two failures "the same": the status and what was missing, not
/// the wording around them or the time.
fn signature_of(settled: &JobSettlement) -> String {
    let mut missing = settled.missing.clone();
    missing.sort();
    let mut hasher = Sha256::new();
    hasher.update(settled.status.as_bytes());
    hasher.update(b"\x1f");
    hasher.update(settled.findings.to_string().as_bytes());
    for item in missing {
        hasher.update(b"\x1f");
        hasher.update(item.to_lowercase().as_bytes());
    }
    if settled.findings == 0 {
        // A failure with no findings is characterised by its summary.
        hasher.update(b"\x1e");
        hasher.update(normalise(&settled.summary).as_bytes());
    }
    format!("{:x}", hasher.finalize())[..16].to_string()
}

fn normalise(text: &str) -> String {
    text.chars()
        .filter(|c| !c.is_ascii_digit())
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

// ─── Reading it back ───────────────────────────────────────────────────────

/// The plan as the coordinator re-reads it every round: objective, corrections,
/// steps with status, pending decisions and questions, receipts and specialist
/// summaries, exact references. Never artifact bodies.
pub fn project(plan: &TaskPlan) -> String {
    let mut out = format!("Plan v{} — goal: {}\n", plan.version, plan.goal);
    for correction in &plan.corrections {
        out.push_str(&format!("Correction {} (outranks the plan): {}\n", correction.id, correction.text));
    }
    for constraint in &plan.constraints {
        out.push_str(&format!("Constraint: {constraint}\n"));
    }
    for step in &plan.steps {
        let role = step.role.as_deref().map(|role| format!(" [{role}]")).unwrap_or_default();
        let after = if step.depends_on.is_empty() {
            String::new()
        } else {
            format!(" after {}", step.depends_on.join(","))
        };
        out.push_str(&format!(
            "- {} {}{role}{after}: {} — {} (attempts {}/{})\n",
            step.id,
            step.kind.as_str(),
            step.title,
            step.status.as_str(),
            step.repair.attempts,
            step.repair.max_attempts
        ));
        if let Some(job) = &step.job_id {
            out.push_str(&format!("    job {job} running\n"));
        }
        if let Some(note) = &step.note {
            out.push_str(&format!("    note: {note}\n"));
        }
        if let Some(receipt) = step.receipts.last() {
            out.push_str(&format!(
                "    last receipt: {} {} {} — {}\n",
                match receipt.kind {
                    ReceiptKind::Job => "job",
                    ReceiptKind::Tool => "tool event",
                    ReceiptKind::Review => "review",
                },
                receipt.reference,
                receipt.status,
                receipt.summary
            ));
        }
        if !step.outputs.is_empty() {
            out.push_str(&format!(
                "    outputs: {}\n",
                step.outputs.iter().map(StepRef::describe).collect::<Vec<_>>().join(", ")
            ));
        }
    }
    for decision in &plan.decisions {
        let evidence = if decision.evidence.is_empty() {
            " (no evidence named)".to_string()
        } else {
            format!(" (rests on {})", decision.evidence.join(", "))
        };
        out.push_str(&format!("Decision {}: {}{evidence}\n", decision.id, decision.text));
    }
    for question in plan.questions.iter().filter(|question| question.answer.is_none()) {
        out.push_str(&format!("Open question {}: {}\n", question.id, question.text));
    }
    let ready: Vec<&str> = plan.ready().iter().map(|step| step.id.as_str()).collect();
    if !ready.is_empty() {
        out.push_str(&format!("Ready to start: {}\n", ready.join(", ")));
    }
    out
}

/// Whether the plan says the task is done, and why not when it is not.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PlanGate {
    pub version: u32,
    pub steps: usize,
    pub completed: usize,
    /// `id (status: note)` for every step that is neither complete nor skipped.
    pub unfinished: Vec<String>,
    pub running: Vec<String>,
    /// Whether any step's acceptance asks for a review or an accepted artifact.
    pub needs_review: bool,
    /// Whether every such step has a passed review receipt.
    pub reviewed: bool,
    pub corrections: usize,
}

pub fn gate(plan: &TaskPlan) -> PlanGate {
    let mut unfinished = Vec::new();
    let mut running = Vec::new();
    for step in &plan.steps {
        match step.status {
            StepStatus::Completed | StepStatus::Skipped => {}
            StepStatus::Running => {
                running.push(step.id.clone());
                unfinished.push(format!("{} (running)", step.id));
            }
            other => unfinished.push(match &step.note {
                Some(note) => format!("{} ({}: {note})", step.id, other.as_str()),
                None => format!("{} ({})", step.id, other.as_str()),
            }),
        }
    }
    let reviewed_steps: Vec<&Step> = plan
        .steps
        .iter()
        .filter(|step| step.status != StepStatus::Skipped)
        .filter(|step| step.kind == StepKind::Review || step.acceptance.contains(&Acceptance::ReviewPassed))
        .collect();
    let needs_review = !reviewed_steps.is_empty()
        || plan.steps.iter().any(|step| step.acceptance.contains(&Acceptance::ArtifactAccepted));
    let reviewed = !reviewed_steps.is_empty()
        && reviewed_steps.iter().all(|step| {
            step.receipts
                .iter()
                .any(|receipt| receipt.kind == ReceiptKind::Review && receipt.status == "passed")
        });
    PlanGate {
        version: plan.version,
        steps: plan.steps.len(),
        completed: plan.steps.iter().filter(|step| step.status == StepStatus::Completed).count(),
        unfinished,
        running,
        needs_review,
        reviewed,
        corrections: plan.corrections.len(),
    }
}

#[cfg(test)]
#[path = "task_plan_tests.rs"]
mod tests;
