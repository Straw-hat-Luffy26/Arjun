//! What to do when a model will not load, decided without inventing anything.
//!
//! ## The failure this exists to prevent
//!
//! A model that fails to load has always ended the turn with the loader's own
//! sentence and nothing else: no record of *why* in terms anyone could act on,
//! no statement of what the run could do instead, and — worse, on the paths
//! that did try to recover — the temptation to answer the load failure by
//! quietly loading something smaller. On an 8 GB card the smaller thing to hand
//! is usually the same model at a lower quantisation, and that is precisely the
//! substitution this deployment's policy forbids: Spark is qualified at Q8, the
//! 9B models at Q4, and an answer produced by a Q2 re-quantisation of the model
//! that was asked for is not an answer from the model that was asked for.
//!
//! So a load failure produces three things here, all of them facts:
//!
//! 1. **What kind of failure it was** ([`LoadFailureKind`]) — read off the
//!    serving error, never guessed from a timing.
//! 2. **Whether the task survived it** — always yes. Nothing in this module
//!    touches the task, its notes or its effects; the caller's checkpoint is
//!    what the resumption continues from.
//! 3. **An honest decision** ([`FallbackDecision`]) — retry later, use a
//!    *different* model that is qualified for the same role at its own policy
//!    quantisation, or blocked. Never "the same model, fewer bits".
//!
//! ## Why mid-task never switches models here
//!
//! A run that has already done work under one model and is continued under
//! another has been handed to a stranger, with a different tokenizer, window
//! and template. That is a model *handoff*, and it has its own states, checks
//! and record in [`crate::agent_runtime::model_transition`]. A load failure is
//! not the place to perform one silently, so a failure mid-task is always
//! [`FallbackDecision::RetryLater`]: the run is suspended with its state intact
//! and resumes on the model it was on once that model can be served again.

use serde::{Deserialize, Serialize};

use crate::registry::{ModelEntry, ModelRole};

/// Parameter count, in billions, at or below which the chat-model pool is
/// served at 8 bits.
///
/// The deployment's rule is "8-bit around 4B; 4-bit above 4B" (plan §4). 4.4
/// rather than 4.0 because the models the rule was written for are not round:
/// Nemotron Nano 4B is 3.97B and Spark X2.5 is 4.0B, while Gemma 4 E4B —
/// which the plan explicitly places on the 4-bit side — is 4.5B effective.
pub const EIGHT_BIT_UP_TO_B: f32 = 4.4;

/// How many bits a quantisation label stores per weight, coarsely.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", tag = "kind")]
pub enum QuantClass {
    /// `Q<n>_…`, read off the label.
    Bits { bits: u8 },
    /// `F16`, `BF16`, `F32`.
    Float { bits: u8 },
    /// No label, or one that names no width (`unknown`, `mixed`).
    Unknown,
}

impl QuantClass {
    /// Reads a GGUF quantisation label.
    ///
    /// Tolerant of the decorations converters put around the width —
    /// `UD-Q4_K_XL`, `IQ4_XS`, `q8_0` — because what the policy cares about is
    /// the width, and every one of those spellings states it. A label that
    /// states no width is [`QuantClass::Unknown`], not a guess.
    pub fn parse(label: Option<&str>) -> Self {
        let Some(raw) = label else {
            return QuantClass::Unknown;
        };
        let upper = raw.trim().to_ascii_uppercase();
        if upper.contains("BF16") || upper.contains("F16") {
            return QuantClass::Float { bits: 16 };
        }
        if upper.contains("F32") {
            return QuantClass::Float { bits: 32 };
        }
        // The first `Q` followed directly by digits. `IQ4_XS` and `UD-Q4_K_XL`
        // both have one; `Q` alone in a word does not.
        let bytes = upper.as_bytes();
        for (index, byte) in bytes.iter().enumerate() {
            if *byte != b'Q' {
                continue;
            }
            let digits: String = upper[index + 1..]
                .chars()
                .take_while(char::is_ascii_digit)
                .collect();
            if let Ok(bits) = digits.parse::<u8>() {
                if (1..=16).contains(&bits) {
                    return QuantClass::Bits { bits };
                }
            }
        }
        QuantClass::Unknown
    }

    pub fn label(self) -> String {
        match self {
            QuantClass::Bits { bits } => format!("{bits}-bit"),
            QuantClass::Float { bits } => format!("F{bits}"),
            QuantClass::Unknown => "unknown width".to_string(),
        }
    }
}

/// Whether an entry sits where the deployment's Q8/Q4 rule says it should.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", tag = "kind")]
pub enum PolicyConformance {
    /// Its width is the one its size calls for.
    Conforms { expected_bits: u8 },
    /// Its width is not. Recorded, never corrected here — changing an
    /// installed model's quantisation is a decision, not a repair.
    Nonconforming { expected_bits: u8, actual: String },
    /// Its label states no width, so conformance cannot be shown.
    Unknown { expected_bits: u8 },
    /// The rule does not cover this model: the OCR package and retrieval
    /// models are preserved at their inventoried configuration (plan §4).
    Exempt { because: String },
}

impl PolicyConformance {
    /// Whether this entry may be chosen *automatically*.
    ///
    /// Only a conforming or exempt entry. An unknown width is not conformance,
    /// and choosing it automatically would be claiming a policy property
    /// nobody established.
    pub fn eligible_automatically(&self) -> bool {
        matches!(
            self,
            PolicyConformance::Conforms { .. } | PolicyConformance::Exempt { .. }
        )
    }
}

/// The Q8/Q4 rule, applied to one entry.
pub fn conformance(entry: &ModelEntry) -> PolicyConformance {
    let chat_pool = entry.roles.iter().any(|role| {
        matches!(
            role,
            ModelRole::Reasoning | ModelRole::Coding | ModelRole::Vision
        )
    });
    if !chat_pool {
        return PolicyConformance::Exempt {
            because: "the Q8/Q4 rule covers the chat-model pool; OCR, embedding, reranking and \
                      relation-extraction models are preserved at their inventoried configuration"
                .to_string(),
        };
    }
    let expected_bits = if entry.parameters_b > 0.0 && entry.parameters_b <= EIGHT_BIT_UP_TO_B {
        8
    } else {
        4
    };
    match QuantClass::parse(entry.quantization.as_deref()) {
        QuantClass::Bits { bits } if bits == expected_bits => {
            PolicyConformance::Conforms { expected_bits }
        }
        QuantClass::Unknown => PolicyConformance::Unknown { expected_bits },
        other => PolicyConformance::Nonconforming {
            expected_bits,
            actual: other.label(),
        },
    }
}

/// The weights an entry is a quantisation *of*, as far as the registry says.
///
/// Two entries with the same base are the same model at different widths —
/// `unlimited-ocr-q6-k` and `unlimited-ocr-q4-k-m` — and a fallback between
/// them is the substitution the policy forbids. Derived by removing the
/// quantisation label from the id, which is how every entry this product ships
/// and every file the scanner names is spelled.
pub fn base_identity(entry: &ModelEntry) -> String {
    let mut id = entry.id.to_ascii_lowercase();
    // The declared label first, in both separator conventions.
    let mut labels: Vec<String> = Vec::new();
    if let Some(label) = entry.quantization.as_deref() {
        let lower = label.to_ascii_lowercase();
        labels.push(lower.clone());
        labels.push(lower.replace('_', "-"));
        labels.push(lower.replace('-', "_"));
    }
    for label in labels {
        if !label.is_empty() && label != "unknown" {
            id = id.replace(&label, "");
        }
    }
    // Then any remaining width token, so an entry whose declared label is
    // missing or wrong still collapses onto its siblings.
    let tokens: Vec<&str> = id
        .split(|c: char| c == '-' || c == '_' || c == '.')
        .filter(|token| !token.is_empty())
        .collect();
    let kept: Vec<&str> = tokens
        .into_iter()
        .filter(|token| !is_width_token(token))
        .collect();
    kept.join("-")
}

fn is_width_token(token: &str) -> bool {
    let token = token.trim_start_matches("ud").trim_start_matches('i');
    if matches!(token, "f16" | "bf16" | "f32" | "k" | "s" | "m" | "l" | "xl" | "xs" | "xxs") {
        return true;
    }
    token.starts_with('q') && token[1..].chars().next().is_some_and(|c| c.is_ascii_digit())
}

/// Why a model did not come up.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", tag = "kind")]
pub enum LoadFailureKind {
    /// The device ran out of memory loading it, or the plan said it would.
    OutOfMemory,
    /// The weights, a projector, or part of the file are missing.
    FilesMissing,
    /// This machine's `llama-server` cannot serve the architecture.
    UnsupportedRuntime,
    /// It was started and never answered a readiness probe.
    NeverReady,
    /// The server process could not be started at all.
    LaunchFailed,
    /// Anything else, with the loader's own words.
    Other,
}

impl LoadFailureKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            LoadFailureKind::OutOfMemory => "outOfMemory",
            LoadFailureKind::FilesMissing => "filesMissing",
            LoadFailureKind::UnsupportedRuntime => "unsupportedRuntime",
            LoadFailureKind::NeverReady => "neverReady",
            LoadFailureKind::LaunchFailed => "launchFailed",
            LoadFailureKind::Other => "other",
        }
    }

    /// Whether the same model might load if asked again later, unchanged.
    ///
    /// Memory pressure and a slow first load pass; a missing file or an
    /// architecture the binary does not know do not pass on their own.
    pub fn may_pass_by_waiting(&self) -> bool {
        matches!(
            self,
            LoadFailureKind::OutOfMemory | LoadFailureKind::NeverReady | LoadFailureKind::Other
        )
    }
}

/// What a load failure was, in the loader's words and in a class.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LoadFailure {
    pub model_id: String,
    pub kind: LoadFailureKind,
    /// The loader's own sentence, bounded.
    pub detail: String,
}

/// Markers of an allocation failure, as llama.cpp and the CUDA/Vulkan runtimes
/// word them. Matched case-insensitively against the loader's text.
const OOM_MARKERS: &[&str] = &[
    "out of memory",
    "cudamalloc failed",
    "failed to allocate",
    "unable to allocate",
    "vk_error_out_of_device_memory",
    "oom",
];

impl LoadFailure {
    /// Classifies a serving error.
    pub fn from_serving(model_id: &str, error: &crate::serving::ServingError) -> Self {
        use crate::serving::ServingError as E;
        let detail = bounded(&error.to_string());
        let kind = match error {
            E::WontFit { .. } => LoadFailureKind::OutOfMemory,
            E::WeightsMissing { .. } | E::WeightsIncomplete { .. } | E::ProjectorMissing { .. } => {
                LoadFailureKind::FilesMissing
            }
            E::UnsupportedRuntime(_) | E::NeedsExternalEndpoint { .. } => {
                LoadFailureKind::UnsupportedRuntime
            }
            E::NeverReady { detail: inner, .. } if mentions_oom(inner) => {
                LoadFailureKind::OutOfMemory
            }
            E::NeverReady { .. } => LoadFailureKind::NeverReady,
            E::LaunchFailed(inner) if mentions_oom(inner) => LoadFailureKind::OutOfMemory,
            E::LaunchFailed(_) | E::NoPort(_) => LoadFailureKind::LaunchFailed,
        };
        Self {
            model_id: model_id.to_string(),
            kind,
            detail,
        }
    }

    /// Classifies a failure known only as text — a transport error from a
    /// server that died mid-request, say.
    pub fn from_text(model_id: &str, text: &str) -> Self {
        Self {
            model_id: model_id.to_string(),
            kind: if mentions_oom(text) {
                LoadFailureKind::OutOfMemory
            } else {
                LoadFailureKind::Other
            },
            detail: bounded(text),
        }
    }
}

fn mentions_oom(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    OOM_MARKERS.iter().any(|marker| {
        if *marker == "oom" {
            // Only as a word: "room" and "bloom" are not allocation failures.
            lower
                .split(|c: char| !c.is_ascii_alphanumeric())
                .any(|word| word == "oom")
        } else {
            lower.contains(marker)
        }
    })
}

fn bounded(text: &str) -> String {
    const LIMIT: usize = 400;
    if text.chars().count() <= LIMIT {
        return text.to_string();
    }
    let cut: String = text.chars().take(LIMIT).collect();
    format!("{cut}…")
}

/// Where the run stood when its model would not load.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum RunPhase {
    /// Nothing has been done under any model yet. A different qualified model
    /// may take the turn, and the record says it did.
    FreshTurn,
    /// Work has been done. The run keeps its model; see the module header.
    MidTask,
    /// A delegated job whose model was pinned when it was dispatched. It is not
    /// substituted here; the parent decides whether to dispatch again.
    PinnedJob,
}

/// What happens next.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", tag = "decision")]
pub enum FallbackDecision {
    /// Suspend and resume on the same model later. The task is intact.
    RetryLater { because: String },
    /// A different model, qualified for the same role at its own policy width.
    UseAlternative { model_id: String, because: String },
    /// Nothing on this machine can take the work until something changes.
    Blocked { because: String },
}

impl FallbackDecision {
    pub fn as_str(&self) -> &'static str {
        match self {
            FallbackDecision::RetryLater { .. } => "retryLater",
            FallbackDecision::UseAlternative { .. } => "useAlternative",
            FallbackDecision::Blocked { .. } => "blocked",
        }
    }

    pub fn because(&self) -> &str {
        match self {
            FallbackDecision::RetryLater { because }
            | FallbackDecision::UseAlternative { because, .. }
            | FallbackDecision::Blocked { because } => because,
        }
    }

    /// Whether the task can continue later without anybody changing anything
    /// but the machine's load. Every decision leaves the task intact; this says
    /// whether waiting alone can be enough.
    pub fn recoverable(&self) -> bool {
        !matches!(self, FallbackDecision::Blocked { .. })
    }
}

/// Decides what a load failure means for the run.
///
/// `candidates` are the registry's entries; the failed one may be among them
/// and is never chosen. The first acceptable alternative in the order given
/// wins, so the caller's routing preference is respected.
pub fn decide(
    failed: &ModelEntry,
    failure: &LoadFailure,
    phase: RunPhase,
    role: ModelRole,
    candidates: &[ModelEntry],
) -> FallbackDecision {
    let what = format!(
        "{} could not be loaded ({}): {}",
        failed.id,
        failure.kind.as_str(),
        failure.detail
    );

    if phase == RunPhase::MidTask {
        return FallbackDecision::RetryLater {
            because: format!(
                "{what}. This task has already done work under {model}, so it is suspended with \
                 its notes, receipts and artifacts intact and resumes on {model} once it can be \
                 served. Moving it to a different model is a handoff, with its own checks, and is \
                 not done silently here.",
                model = failed.id
            ),
        };
    }

    if phase == RunPhase::PinnedJob {
        let retry = failure.kind.may_pass_by_waiting();
        let because = format!(
            "{what}. This job's model was pinned when it was dispatched, so nothing else was \
             loaded in its place and nothing was done. {}",
            if retry {
                "The parent may dispatch it again once the machine has more memory free."
            } else {
                "The model files or the runtime have to be fixed before it can run."
            }
        );
        return if retry {
            FallbackDecision::RetryLater { because }
        } else {
            FallbackDecision::Blocked { because }
        };
    }

    let failed_base = base_identity(failed);
    let mut refused: Vec<String> = Vec::new();
    for candidate in candidates {
        if candidate.id == failed.id || !candidate.enabled || !candidate.serves(role) {
            continue;
        }
        if base_identity(candidate) == failed_base {
            // The forbidden substitution, named so the record shows it was
            // considered and why it was not taken.
            refused.push(format!(
                "{} is {} at a different quantisation, which this deployment does not \
                 substitute",
                candidate.id, failed.id
            ));
            continue;
        }
        let conformance = conformance(candidate);
        if !conformance.eligible_automatically() {
            refused.push(format!(
                "{} does not meet the Q8/Q4 rule ({conformance:?})",
                candidate.id
            ));
            continue;
        }
        return FallbackDecision::UseAlternative {
            model_id: candidate.id.clone(),
            because: format!(
                "{what}. Nothing had been done yet, so {} — registered for the same role and \
                 served at its own policy quantisation — takes the turn instead. The quantisation \
                 of {} was not changed.",
                candidate.id, failed.id
            ),
        };
    }

    let considered = if refused.is_empty() {
        "no other enabled model is registered for this role".to_string()
    } else {
        refused.join("; ")
    };
    if failure.kind.may_pass_by_waiting() {
        FallbackDecision::RetryLater {
            because: format!(
                "{what}. No alternative was taken: {considered}. The turn can be asked again once \
                 the machine has more memory free."
            ),
        }
    } else {
        FallbackDecision::Blocked {
            because: format!(
                "{what}. No alternative was taken: {considered}. This will not pass by waiting — \
                 the model files or the runtime have to be fixed first."
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::tests::entry;

    fn quantised(id: &str, params: f32, quant: &str, roles: Vec<ModelRole>) -> ModelEntry {
        let mut e = entry(id, params, roles);
        e.quantization = Some(quant.to_string());
        e
    }

    #[test]
    fn widths_are_read_off_every_spelling_this_product_meets() {
        assert_eq!(QuantClass::parse(Some("Q8_0")), QuantClass::Bits { bits: 8 });
        assert_eq!(QuantClass::parse(Some("Q4_K_S")), QuantClass::Bits { bits: 4 });
        assert_eq!(QuantClass::parse(Some("UD-Q4_K_XL")), QuantClass::Bits { bits: 4 });
        assert_eq!(QuantClass::parse(Some("IQ4_XS")), QuantClass::Bits { bits: 4 });
        assert_eq!(QuantClass::parse(Some("q6_k")), QuantClass::Bits { bits: 6 });
        assert_eq!(QuantClass::parse(Some("F16")), QuantClass::Float { bits: 16 });
        assert_eq!(QuantClass::parse(Some("unknown")), QuantClass::Unknown);
        assert_eq!(QuantClass::parse(None), QuantClass::Unknown);
    }

    /// The inventory's own models, against the rule.
    #[test]
    fn the_q8_q4_rule_is_applied_by_size_and_never_assumed() {
        let spark = quantised("Spark-X2.5-4B-Q8_0", 4.0, "Q8_0", vec![ModelRole::Reasoning]);
        assert_eq!(conformance(&spark), PolicyConformance::Conforms { expected_bits: 8 });

        let qwen = quantised("Qwen3.5-9B-Q4_K_S", 9.0, "Q4_K_S", vec![ModelRole::Coding]);
        assert_eq!(conformance(&qwen), PolicyConformance::Conforms { expected_bits: 4 });

        // P00 found Nemotron installed at Q4_K_M. Recorded as nonconforming,
        // not corrected.
        let nemotron = quantised(
            "NVIDIA-Nemotron3-Nano-4B-Q4_K_M",
            3.97,
            "Q4_K_M",
            vec![ModelRole::Reasoning],
        );
        assert!(matches!(
            conformance(&nemotron),
            PolicyConformance::Nonconforming { expected_bits: 8, .. }
        ));
        assert!(!conformance(&nemotron).eligible_automatically());

        // Two Gemma rows say `unknown`. That is not conformance.
        let gemma = quantised("gemma-4-E4B-it", 4.5, "unknown", vec![ModelRole::Vision]);
        assert_eq!(conformance(&gemma), PolicyConformance::Unknown { expected_bits: 4 });
        assert!(!conformance(&gemma).eligible_automatically());

        // OCR is outside the chat-model rule.
        let ocr = quantised("unlimited-ocr-q6-k", 3.0, "Q6_K", vec![ModelRole::DocumentOcr]);
        assert!(matches!(conformance(&ocr), PolicyConformance::Exempt { .. }));
    }

    #[test]
    fn two_widths_of_one_model_share_a_base_and_two_models_do_not() {
        let q6 = quantised("unlimited-ocr-q6-k", 3.0, "Q6_K", vec![ModelRole::DocumentOcr]);
        let q4 = quantised("unlimited-ocr-q4-k-m", 3.0, "Q4_K_M", vec![ModelRole::DocumentOcr]);
        assert_eq!(base_identity(&q6), base_identity(&q4));

        let q8 = quantised("Qwen3.5-9B-Q8_0", 9.0, "Q8_0", vec![ModelRole::Coding]);
        let q4k = quantised("Qwen3.5-9B-Q4_K_S", 9.0, "Q4_K_S", vec![ModelRole::Coding]);
        assert_eq!(base_identity(&q8), base_identity(&q4k));

        let spark = quantised("Spark-X2.5-4B-Q8_0", 4.0, "Q8_0", vec![ModelRole::Reasoning]);
        assert_ne!(base_identity(&spark), base_identity(&q8));
    }

    /// The load-bearing test: a failed model is never replaced by itself at
    /// fewer bits, even when that is the only other thing registered.
    #[test]
    fn a_failed_load_never_falls_back_to_a_lower_quantisation_of_the_same_model() {
        let q8 = quantised("Qwen3.5-9B-Q4_K_S", 9.0, "Q4_K_S", vec![ModelRole::Coding]);
        let q2 = quantised("Qwen3.5-9B-Q2_K", 9.0, "Q2_K", vec![ModelRole::Coding]);
        let failure = LoadFailure {
            model_id: q8.id.clone(),
            kind: LoadFailureKind::OutOfMemory,
            detail: "cudaMalloc failed: out of memory".into(),
        };
        let decision = decide(
            &q8,
            &failure,
            RunPhase::FreshTurn,
            ModelRole::Coding,
            &[q8.clone(), q2.clone()],
        );
        assert!(
            !matches!(&decision, FallbackDecision::UseAlternative { model_id, .. } if model_id == &q2.id),
            "fell back to a re-quantisation: {decision:?}"
        );
        assert!(matches!(decision, FallbackDecision::RetryLater { .. }));
        assert!(decision.because().contains("different quantisation"));
    }

    #[test]
    fn a_fresh_turn_may_move_to_a_different_conforming_model_and_says_so() {
        let qwen = quantised("Qwen3.5-9B-Q4_K_S", 9.0, "Q4_K_S", vec![ModelRole::Reasoning]);
        let spark = quantised("Spark-X2.5-4B-Q8_0", 4.0, "Q8_0", vec![ModelRole::Reasoning]);
        let failure = LoadFailure::from_text(&qwen.id, "CUDA error: out of memory");
        assert_eq!(failure.kind, LoadFailureKind::OutOfMemory);

        let decision = decide(
            &qwen,
            &failure,
            RunPhase::FreshTurn,
            ModelRole::Reasoning,
            &[qwen.clone(), spark.clone()],
        );
        match &decision {
            FallbackDecision::UseAlternative { model_id, because } => {
                assert_eq!(model_id, &spark.id);
                assert!(because.contains("quantisation of Qwen3.5-9B-Q4_K_S was not changed"));
            }
            other => panic!("expected an alternative, got {other:?}"),
        }
    }

    #[test]
    fn mid_task_the_run_keeps_its_model_and_is_told_it_can_resume() {
        let qwen = quantised("Qwen3.5-9B-Q4_K_S", 9.0, "Q4_K_S", vec![ModelRole::Reasoning]);
        let spark = quantised("Spark-X2.5-4B-Q8_0", 4.0, "Q8_0", vec![ModelRole::Reasoning]);
        let failure = LoadFailure::from_text(&qwen.id, "never became ready");
        let decision = decide(
            &qwen,
            &failure,
            RunPhase::MidTask,
            ModelRole::Reasoning,
            &[qwen.clone(), spark],
        );
        assert!(matches!(decision, FallbackDecision::RetryLater { .. }));
        assert!(decision.recoverable());
        assert!(decision.because().contains("intact"));
    }

    #[test]
    fn a_pinned_job_is_never_substituted_even_when_an_alternative_exists() {
        let qwen = quantised("Qwen3.5-9B-Q4_K_S", 9.0, "Q4_K_S", vec![ModelRole::Reasoning]);
        let spark = quantised("Spark-X2.5-4B-Q8_0", 4.0, "Q8_0", vec![ModelRole::Reasoning]);
        let failure = LoadFailure::from_text(&qwen.id, "out of memory");
        let decision = decide(
            &qwen,
            &failure,
            RunPhase::PinnedJob,
            ModelRole::Reasoning,
            &[qwen.clone(), spark],
        );
        assert!(matches!(decision, FallbackDecision::RetryLater { .. }), "{decision:?}");
        assert!(decision.because().contains("pinned"));
    }

    #[test]
    fn a_missing_file_with_no_alternative_is_blocked_not_retried() {
        let qwen = quantised("Qwen3.5-9B-Q4_K_S", 9.0, "Q4_K_S", vec![ModelRole::Coding]);
        let failure = LoadFailure {
            model_id: qwen.id.clone(),
            kind: LoadFailureKind::FilesMissing,
            detail: "no weights".into(),
        };
        let decision = decide(&qwen, &failure, RunPhase::FreshTurn, ModelRole::Coding, &[qwen.clone()]);
        assert!(matches!(decision, FallbackDecision::Blocked { .. }));
        assert!(!decision.recoverable());
    }

    #[test]
    fn a_nonconforming_or_unknown_width_is_not_an_automatic_alternative() {
        let qwen = quantised("Qwen3.5-9B-Q4_K_S", 9.0, "Q4_K_S", vec![ModelRole::Reasoning]);
        let nemotron = quantised(
            "NVIDIA-Nemotron3-Nano-4B-Q4_K_M",
            3.97,
            "Q4_K_M",
            vec![ModelRole::Reasoning],
        );
        let gemma = quantised("gemma-4-12b-it-UD-Q4_K_XL", 12.0, "unknown", vec![ModelRole::Reasoning]);
        let failure = LoadFailure::from_text(&qwen.id, "out of memory");
        let decision = decide(
            &qwen,
            &failure,
            RunPhase::FreshTurn,
            ModelRole::Reasoning,
            &[qwen.clone(), nemotron, gemma],
        );
        assert!(matches!(decision, FallbackDecision::RetryLater { .. }), "{decision:?}");
        assert!(decision.because().contains("Q8/Q4 rule"));
    }

    #[test]
    fn serving_errors_are_classified_from_what_the_loader_said() {
        use crate::serving::ServingError;
        let oom = ServingError::NeverReady {
            model: "m".into(),
            base_url: "http://127.0.0.1:1/v1".into(),
            detail: "ggml_backend_cuda_buffer_type_alloc_buffer: allocating 5.1 GiB: cudaMalloc failed: out of memory".into(),
        };
        assert_eq!(LoadFailure::from_serving("m", &oom).kind, LoadFailureKind::OutOfMemory);

        let slow = ServingError::NeverReady {
            model: "m".into(),
            base_url: "http://127.0.0.1:1/v1".into(),
            detail: "timed out".into(),
        };
        assert_eq!(LoadFailure::from_serving("m", &slow).kind, LoadFailureKind::NeverReady);

        let missing = ServingError::WeightsMissing {
            model: "m".into(),
            path: "/nowhere.gguf".into(),
        };
        assert_eq!(LoadFailure::from_serving("m", &missing).kind, LoadFailureKind::FilesMissing);

        // "room" is not an allocation failure.
        assert_eq!(
            LoadFailure::from_text("m", "no room at the inn").kind,
            LoadFailureKind::Other
        );
    }
}
