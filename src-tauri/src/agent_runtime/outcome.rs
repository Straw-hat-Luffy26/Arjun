//! How a run ended, as one typed value.
//!
//! ## Why this exists
//!
//! [`crate::commands::agent::agent_start_run`] used to decide a run's ending
//! from the shape of the JSON-RPC reply:
//!
//! ```text
//! Ok(Ok(value)) => (Ok(value), TaskEventType::RunCompleted)
//! ```
//!
//! That reads "the request resolved" as "the task succeeded", and the two are
//! not the same question. Every ordinary ending of an agent loop resolves that
//! request: an operator pressing stop, a provider answering 503, the model
//! running into its output cap mid-sentence. All three were recorded as
//! `run_completed`, listed as finished on the Tasks screen, and shown in the
//! chat as an answer. A fragment cut off at the token limit is the worst of
//! them, because it looks exactly like a short answer.
//!
//! So the ending is now a value the loop reports and this side classifies,
//! never a thing inferred from transport success.
//!
//! ## Who decides which
//!
//! Both sides know things the other does not, and the split is deliberate:
//!
//! - The **runtime** knows what the loop did: whether the model stopped of its
//!   own accord, hit the output cap, errored, or was aborted — and, because it
//!   records the cause at the point the abort is requested, whether an abort
//!   was an operator or the run's own deadline.
//! - The **core** knows what it decided: that it timed the run out, that the
//!   plan or the gateway refused what the run needed, that a person cancelled
//!   it. A refusal never reaches the model as an exception — it is a tool
//!   result the model reads — so the runtime cannot see a policy stop at all.
//!
//! [`RunOutcome::from_runtime`] reads the first. [`RunOutcome::from_rpc_error`]
//! and the call site read the second. Where they disagree the core wins, since
//! a run the core stopped did not end the way the loop thought it was ending.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::events::TaskEventType;

/// The ending of a run.
///
/// Every variant except [`RunOutcome::Completed`] carries the one sentence a
/// person is shown. Bounded and safe to display: it is the loop's or the core's
/// own wording, never a tool result and never model output.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum RunOutcome {
    /// The loop finished. The controller must pass its independent completion gate.
    Completed,
    /// A required check or durable effect cannot be settled safely.
    NeedsReview { detail: String },
    /// The provider or the loop errored.
    Failed { detail: String },
    /// A person, or the core shutting down, stopped it.
    Aborted { detail: String },
    /// The model hit the output cap. The answer is a fragment.
    ///
    /// Kept separate from `Completed` because the difference is invisible in
    /// the text: a cut-off answer reads as a short one, and an engineer acting
    /// on half a specification is the failure this distinction prevents.
    LengthLimited { detail: String },
    /// It ran past the time or the steps its plan allowed.
    BudgetStopped { detail: String },
    /// It needed to do something it is not permitted to do.
    PolicyStopped { detail: String },
}

impl RunOutcome {
    /// The stored terminal state wins a race with a worker's completion claim.
    pub fn from_snapshot(snapshot: &super::events::TaskSnapshot) -> Option<Self> {
        use super::events::RunState;
        let detail = snapshot.failure.clone().unwrap_or_else(|| snapshot.state.describe().into());
        Some(match snapshot.state {
            RunState::Completed => Self::Completed,
            RunState::Cancelled => Self::Aborted { detail },
            RunState::Failed => Self::Failed { detail },
            RunState::StoppedByBudget => Self::BudgetStopped { detail },
            RunState::StoppedByLength => Self::LengthLimited { detail },
            RunState::StoppedByPolicy => Self::PolicyStopped { detail },
            RunState::DegradedNeedsHuman => Self::NeedsReview { detail },
            _ => return None,
        })
    }

    /// The wire spelling, matching the TypeScript `RunOutcomeKind` union.
    pub const fn kind(&self) -> &'static str {
        match self {
            RunOutcome::Completed => "completed",
            RunOutcome::NeedsReview { .. } => "needsReview",
            RunOutcome::Failed { .. } => "failed",
            RunOutcome::Aborted { .. } => "aborted",
            RunOutcome::LengthLimited { .. } => "lengthLimited",
            RunOutcome::BudgetStopped { .. } => "budgetStopped",
            RunOutcome::PolicyStopped { .. } => "policyStopped",
        }
    }

    /// The terminal event this ending is recorded as.
    ///
    /// A length-limited run gets its own event rather than being folded into
    /// `run_completed` or `run_failed`. It is neither: the model did what it
    /// was asked and the deployment's cap stopped it, and both other spellings
    /// misdescribe that to whoever reads the history.
    pub const fn event_type(&self) -> TaskEventType {
        match self {
            RunOutcome::Completed => TaskEventType::RunCompleted,
            RunOutcome::NeedsReview { .. } => TaskEventType::RunDegraded,
            RunOutcome::Failed { .. } => TaskEventType::RunFailed,
            RunOutcome::Aborted { .. } => TaskEventType::RunCancelled,
            RunOutcome::LengthLimited { .. } => TaskEventType::RunStoppedByLength,
            RunOutcome::BudgetStopped { .. } => TaskEventType::RunStoppedByBudget,
            RunOutcome::PolicyStopped { .. } => TaskEventType::RunStoppedByPolicy,
        }
    }

    /// The sentence to show, or `None` when the run needs no explanation.
    pub fn detail(&self) -> Option<&str> {
        match self {
            RunOutcome::Completed => None,
            RunOutcome::Failed { detail }
            | RunOutcome::NeedsReview { detail }
            | RunOutcome::Aborted { detail }
            | RunOutcome::LengthLimited { detail }
            | RunOutcome::BudgetStopped { detail }
            | RunOutcome::PolicyStopped { detail } => Some(detail.as_str()),
        }
    }

    /// Whether the run finished the work it set out to do.
    ///
    /// Only `Completed`. A run stopped by its budget produced whatever it
    /// produced, and calling that success would make the budget decorative —
    /// the same rule `RunState::is_success` applies.
    pub const fn is_success(&self) -> bool {
        matches!(self, RunOutcome::Completed)
    }

    /// Whether an answer this run produced should be shown as final.
    ///
    /// A cut-off answer is worth showing and must not be presented as complete,
    /// which is why this is a separate question from [`Self::is_success`].
    pub const fn answer_is_whole(&self) -> bool {
        matches!(self, RunOutcome::Completed)
    }

    /// Reads the runtime's typed outcome out of a `run.start` result.
    ///
    /// Returns `None` for a reply that carries no outcome at all — an older
    /// runtime, or a malformed one. The caller decides what to do about that;
    /// this deliberately does not default to `Completed`, because defaulting to
    /// success on a message it could not understand is the whole bug.
    pub fn from_runtime(value: &Value) -> Option<Self> {
        let outcome = value.get("outcome")?;
        let kind = outcome.get("kind")?.as_str()?;
        let detail = outcome
            .get("detail")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        Some(match kind {
            "completed" => RunOutcome::Completed,
            "needsReview" => RunOutcome::NeedsReview {
                detail: fallback(detail, "A required check could not be completed safely."),
            },
            "failed" => RunOutcome::Failed {
                detail: fallback(detail, "The run did not finish."),
            },
            "aborted" => RunOutcome::Aborted {
                detail: fallback(detail, "Stopped before it finished."),
            },
            "lengthLimited" => RunOutcome::LengthLimited {
                detail: fallback(
                    detail,
                    "Stopped: the answer reached the output limit for one turn, so it is cut off.",
                ),
            },
            "budgetStopped" => RunOutcome::BudgetStopped {
                detail: fallback(detail, "Stopped: it reached the limit its plan set."),
            },
            "policyStopped" => RunOutcome::PolicyStopped {
                detail: fallback(
                    detail,
                    "Stopped: it needed to do something it is not permitted to do.",
                ),
            },
            // An unrecognised kind from a runtime this build does not know is
            // not an answer. Reported as a failure rather than guessed at.
            other => RunOutcome::Failed {
                detail: format!("The runtime reported an ending this build does not know: {other}"),
            },
        })
    }

    /// Classifies a `run.start` request that came back as an error.
    ///
    /// A run the gateway or the plan stopped is not a fault, and a list that
    /// paints it the same colour as one teaches people to skip the row that
    /// actually broke. Read from the refusal's own wording, because every
    /// refusal path produces a sentence and none of them produces a code.
    pub fn from_rpc_error(detail: String) -> Self {
        let stopped_by_policy = detail.contains("not permitted")
            || detail.contains("was not approved")
            || detail.contains("is not cleared");
        let stopped_by_budget = detail.contains("permitted steps")
            || detail.contains("going in circles")
            || detail.contains("time allowed")
            || detail.contains("time budget");
        if stopped_by_policy {
            RunOutcome::PolicyStopped { detail }
        } else if stopped_by_budget {
            RunOutcome::BudgetStopped { detail }
        } else {
            RunOutcome::Failed { detail }
        }
    }
}

fn fallback(detail: String, when_empty: &str) -> String {
    if detail.trim().is_empty() {
        when_empty.to_string()
    } else {
        detail
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn a_resolved_reply_is_not_automatically_a_completion() {
        // The defect in one assertion: a reply that arrived, carrying an
        // ending that is not "completed", must not be read as one.
        for (kind, expected) in [
            ("failed", TaskEventType::RunFailed),
            ("aborted", TaskEventType::RunCancelled),
            ("lengthLimited", TaskEventType::RunStoppedByLength),
            ("budgetStopped", TaskEventType::RunStoppedByBudget),
            ("policyStopped", TaskEventType::RunStoppedByPolicy),
        ] {
            let reply = json!({ "text": "half an answer", "outcome": { "kind": kind } });
            let outcome = RunOutcome::from_runtime(&reply).expect("outcome parses");
            assert_eq!(outcome.event_type(), expected, "for {kind}");
            assert!(!outcome.is_success(), "{kind} must not count as success");
            assert!(outcome.detail().is_some(), "{kind} must say why");
        }
    }

    #[test]
    fn a_completed_reply_maps_to_run_completed_and_needs_no_excuse() {
        let reply = json!({ "text": "the answer", "outcome": { "kind": "completed" } });
        let outcome = RunOutcome::from_runtime(&reply).expect("outcome parses");
        assert_eq!(outcome, RunOutcome::Completed);
        assert_eq!(outcome.event_type(), TaskEventType::RunCompleted);
        assert!(outcome.is_success());
        assert_eq!(outcome.detail(), None);
    }

    #[test]
    fn a_reply_with_no_outcome_is_not_read_as_success() {
        // An older or malformed runtime. Absent, not completed: defaulting to
        // success on a message this side could not understand is the bug.
        assert_eq!(RunOutcome::from_runtime(&json!({ "text": "x" })), None);
    }

    #[test]
    fn an_unknown_ending_is_a_failure_rather_than_a_guess() {
        let reply = json!({ "outcome": { "kind": "transcended" } });
        let outcome = RunOutcome::from_runtime(&reply).expect("outcome parses");
        assert_eq!(outcome.event_type(), TaskEventType::RunFailed);
    }

    #[test]
    fn the_length_cap_is_its_own_ending_and_never_a_completion() {
        let reply = json!({
            "text": "The seal specification is ",
            "outcome": { "kind": "lengthLimited", "detail": "cut off mid-way" },
        });
        let outcome = RunOutcome::from_runtime(&reply).expect("outcome parses");
        assert_eq!(outcome.kind(), "lengthLimited");
        assert!(!outcome.answer_is_whole());
        assert_eq!(outcome.detail(), Some("cut off mid-way"));
    }

    #[test]
    fn a_refused_run_is_a_policy_stop_and_not_a_fault() {
        let outcome =
            RunOutcome::from_rpc_error("that write is not permitted outside the workspace".into());
        assert_eq!(outcome.event_type(), TaskEventType::RunStoppedByPolicy);
    }

    #[test]
    fn a_run_out_of_steps_is_a_budget_stop_and_not_a_fault() {
        for wording in [
            "stopped after 12 of 12 permitted steps",
            "the same call repeated: going in circles",
            "it ran past the time allowed",
        ] {
            let outcome = RunOutcome::from_rpc_error(wording.into());
            assert_eq!(
                outcome.event_type(),
                TaskEventType::RunStoppedByBudget,
                "for {wording}"
            );
        }
    }

    #[test]
    fn anything_else_on_the_error_path_is_a_failure() {
        let outcome = RunOutcome::from_rpc_error("the runtime went away".into());
        assert_eq!(outcome.event_type(), TaskEventType::RunFailed);
    }

    #[test]
    fn the_wire_spelling_matches_the_typescript_union() {
        let fixtures: Vec<Value> = serde_json::from_str(include_str!("../../../contracts/run-outcomes.json")).unwrap();
        for fixture in fixtures {
            let outcome: RunOutcome = serde_json::from_value(fixture.clone()).expect("shared wire contract");
            assert_eq!(serde_json::to_value(&outcome).unwrap(), fixture);
            assert_eq!(outcome.is_success(), outcome.kind() == "completed");
        }
        // The frontend narrows on these exact strings. A rename on this side
        // that is not made on that one is a silently unhandled state.
        let cases = [
            (RunOutcome::Completed, "completed"),
            (
                RunOutcome::Failed {
                    detail: "x".into(),
                },
                "failed",
            ),
            (
                RunOutcome::Aborted {
                    detail: "x".into(),
                },
                "aborted",
            ),
            (
                RunOutcome::LengthLimited {
                    detail: "x".into(),
                },
                "lengthLimited",
            ),
            (
                RunOutcome::BudgetStopped {
                    detail: "x".into(),
                },
                "budgetStopped",
            ),
            (
                RunOutcome::PolicyStopped {
                    detail: "x".into(),
                },
                "policyStopped",
            ),
        ];
        for (outcome, expected) in cases {
            assert_eq!(outcome.kind(), expected);
            let encoded = serde_json::to_value(&outcome).expect("serialises");
            assert_eq!(encoded.get("kind").and_then(Value::as_str), Some(expected));
        }
    }
}
