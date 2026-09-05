//! What a child sends back, and why it cannot quietly claim success.
//!
//! ## Status is not derived from the payload
//!
//! Requirement 8: a child's failure, timeout or cancellation must be visible to
//! the parent and must never be silently converted into success.
//!
//! The way that goes wrong is subtle. A worker times out having found two of
//! the four passages it was after; the parent sees two passages, folds them in,
//! and the run continues as though the retrieval finished. Nothing lied — the
//! two passages are real — and the answer is nonetheless built on a search that
//! did not complete.
//!
//! So [`ChildStatus`] is a field the manager sets from what actually happened,
//! not something inferred from whether `findings` is empty. A timed-out child
//! **may** carry findings, and [`ChildResult::is_complete`] is still false, and
//! the parent has to decide what to do about a partial result rather than being
//! handed one that looks whole.
//!
//! ## Compact, and referenced
//!
//! Findings carry evidence *references*, not passages. The parent already has a
//! way to resolve a marker; sending the text back would duplicate it into a
//! second place, under a second set of clearance assumptions.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::profile::SchemaKind;

/// How a child ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ChildStatus {
    /// It finished the work it was given.
    Completed,
    /// It ran and could not finish. `detail` on the result says why.
    Failed,
    /// It reached its deadline. May carry partial findings, and is still not a
    /// success.
    TimedOut,
    /// The parent stopped it, or the run it belonged to ended.
    Cancelled,
    /// It was never started: the policy refused it. Distinct from `Failed`
    /// because nothing went wrong — the answer was no.
    Refused,
}

impl ChildStatus {
    pub const fn as_str(self) -> &'static str {
        match self {
            ChildStatus::Completed => "completed",
            ChildStatus::Failed => "failed",
            ChildStatus::TimedOut => "timed_out",
            ChildStatus::Cancelled => "cancelled",
            ChildStatus::Refused => "refused",
        }
    }

    /// Only one value means the work is done.
    pub const fn is_complete(self) -> bool {
        matches!(self, ChildStatus::Completed)
    }

    /// What a parent should say about a child that ended this way.
    pub const fn describe(self) -> &'static str {
        match self {
            ChildStatus::Completed => "finished",
            ChildStatus::Failed => "did not finish; treat anything it returned as incomplete",
            ChildStatus::TimedOut => {
                "ran out of time; anything it returned is partial and the rest was not looked at"
            }
            ChildStatus::Cancelled => "was stopped before it finished",
            ChildStatus::Refused => "was not started, because it was not permitted",
        }
    }
}

/// A passage or file a finding rests on. A reference, never the text.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EvidenceRef {
    /// The marker the parent knows this passage by, where there is one.
    pub marker: Option<usize>,
    /// The document's content hash.
    pub document_sha256: String,
    pub page: Option<u32>,
    /// A short citation string, for a person reading the trace.
    pub citation: String,
}

/// One thing a child established.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Finding {
    /// What was found, in the child's words. Short: this is a result, not a
    /// report, and the parent writes the report.
    pub statement: String,
    /// What it rests on. A finding with no evidence is permitted and is
    /// reported as such — a worker that found nothing has found nothing, and
    /// inventing a citation for it would be worse.
    pub evidence: Vec<EvidenceRef>,
}

/// What a child hands back.
/// `PartialEq` and deliberately not `Eq`: `confidence` is a float, and two
/// results being "equal" is a comparison for tests rather than a key.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ChildResult {
    pub child_id: String,
    pub profile: String,
    /// Set by the manager from what happened. Never inferred from the payload.
    pub status: ChildStatus,
    /// The shape this was asked for. A result whose schema does not match the
    /// packet's is refused by the parent rather than folded in.
    pub schema: SchemaKind,
    pub findings: Vec<Finding>,
    /// How sure the child is, 0.0 to 1.0. Not a model probability — a worker's
    /// own account of how much it had to infer.
    pub confidence: f32,
    /// What it could not establish. Named individually, because "some
    /// uncertainty" is not actionable and "page 4 could not be read" is.
    pub uncertainty: Vec<String>,
    /// Present when the status is not `Completed`, in words a person reads.
    pub detail: Option<String>,
    /// How many turns it actually took.
    pub turns_used: u32,
    /// SHA-256 over the findings and status, so the parent's record of what
    /// came back can be checked against the child's.
    pub result_hash: String,
    pub finished_at: DateTime<Utc>,
}

impl ChildResult {
    /// A result for a child that produced findings and finished.
    pub fn completed(
        child_id: impl Into<String>,
        profile: impl Into<String>,
        schema: SchemaKind,
        findings: Vec<Finding>,
        confidence: f32,
        uncertainty: Vec<String>,
        turns_used: u32,
    ) -> Self {
        Self::sealed(
            child_id.into(),
            profile.into(),
            ChildStatus::Completed,
            schema,
            findings,
            confidence.clamp(0.0, 1.0),
            uncertainty,
            None,
            turns_used,
        )
    }

    /// A result for a child that did not finish.
    ///
    /// Takes whatever findings it had. Keeping them is right — two passages
    /// found before a timeout are two real passages — and the status is what
    /// stops them being read as a completed search.
    pub fn ended(
        child_id: impl Into<String>,
        profile: impl Into<String>,
        status: ChildStatus,
        schema: SchemaKind,
        findings: Vec<Finding>,
        detail: impl Into<String>,
        turns_used: u32,
    ) -> Self {
        debug_assert!(
            !status.is_complete(),
            "`ended` is for a child that did not finish; use `completed`"
        );
        Self::sealed(
            child_id.into(),
            profile.into(),
            status,
            schema,
            findings,
            // A child that did not finish is not confident. Set here rather
            // than taken from the worker, so a worker cannot report high
            // confidence in a result it did not finish producing.
            0.0,
            Vec::new(),
            Some(detail.into()),
            turns_used,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn sealed(
        child_id: String,
        profile: String,
        status: ChildStatus,
        schema: SchemaKind,
        findings: Vec<Finding>,
        confidence: f32,
        uncertainty: Vec<String>,
        detail: Option<String>,
        turns_used: u32,
    ) -> Self {
        let result_hash = hash_of(&status, schema, &findings);
        Self {
            child_id,
            profile,
            status,
            schema,
            findings,
            confidence,
            uncertainty,
            detail,
            turns_used,
            result_hash,
            finished_at: Utc::now(),
        }
    }

    /// Whether the parent may treat this as the work being done.
    pub fn is_complete(&self) -> bool {
        self.status.is_complete()
    }

    /// Whether this answers the packet it came back from.
    ///
    /// Checked by the parent rather than trusted: a result whose schema does
    /// not match what was asked for is a worker answering a different question,
    /// and folding it in would put an extraction where a calculation belongs.
    pub fn answers(&self, packet: &super::packet::ChildTaskPacket) -> bool {
        self.child_id == packet.child_id && self.schema == packet.required_schema
            && self.profile == packet.profile
            && self.result_hash == hash_of(&self.status, self.schema, &self.findings)
            && self.confidence.is_finite() && (0.0..=1.0).contains(&self.confidence)
            && self.turns_used <= packet.limits.max_turns
            && serde_json::to_vec(&(&self.findings, &self.uncertainty, &self.detail))
                .is_ok_and(|body| body.len() <= packet.limits.max_output_tokens as usize * 4)
    }

    /// One line for the parent's trace.
    pub fn describe(&self) -> String {
        let head = format!("{} {}", self.profile, self.status.describe());
        match (&self.detail, self.findings.len()) {
            (Some(detail), 0) => format!("{head}: {detail}"),
            (Some(detail), n) => format!("{head}: {detail} ({n} partial finding(s) kept)"),
            (None, n) => format!("{head}, with {n} finding(s)"),
        }
    }
}

/// The seal over what a child established.
///
/// Over the status as well as the findings, so a record that kept the findings
/// and changed the status does not match — which is exactly the alteration
/// requirement 8 is about.
fn hash_of(status: &ChildStatus, schema: SchemaKind, findings: &[Finding]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(status.as_str().as_bytes());
    hasher.update(b"\x1f");
    hasher.update(schema.as_str().as_bytes());
    hasher.update(b"\x1f");
    for finding in findings {
        hasher.update(finding.statement.as_bytes());
        hasher.update(b"\x1e");
        for evidence in &finding.evidence {
            hasher.update(evidence.document_sha256.as_bytes());
            hasher.update(b"\x1d");
        }
        hasher.update(b"\x1f");
    }
    format!("{:x}", hasher.finalize())
}
