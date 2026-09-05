//! Whether a run actually finished, decided from what it left behind.
//!
//! ## Why the model's word is not the input
//!
//! A run ends when the loop stops, and the loop stops when the model stops
//! producing tool calls. That is a statement about the model's behaviour and
//! not about the task. A model that has lost the thread stops exactly as a
//! model that has finished does, and the answer it leaves reads the same way in
//! both cases — confidently.
//!
//! So nothing here reads the answer's claims about itself. Every criterion is
//! checked against a record some other part of the system wrote for its own
//! reasons: the plan's own step ledger, the effect ledger, the approval ledger,
//! files re-opened from disk, the grounding report.
//!
//! ## Why there are three outcomes and not two
//!
//! `tasks::is_ready` was a `bool`, and a `bool` has to put "this failed" and
//! "nobody can tell whether this worked" in the same bucket. Those are the two
//! situations an operator most needs told apart: the first is a result, the
//! second is a question.
//!
//! An unknown side effect is the clearest case. A document may or may not have
//! been written; the run is not failed, and it is certainly not done. It is
//! [`Outcome::NeedsReview`], and it says which criterion could not be settled.
//!
//! ## Why the criteria are named and versioned
//!
//! A verdict that cannot be re-derived is an opinion. Each criterion carries a
//! stable id and the evidence it was decided from, and the record carries the
//! version of the checker that produced it — so a run verified last month can
//! be told apart from one verified after the rules changed, rather than both
//! reading as "verified".

use serde::{Deserialize, Serialize};

/// Bumped whenever a criterion is added, removed, or its meaning changes.
///
/// Stored on every record. Two runs that both say "passed" under different
/// versions were not held to the same thing, and nothing else in the record
/// would say so.
pub const VERIFIER_VERSION: u32 = 2;

/// How one criterion came out.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum CriterionStatus {
    Passed,
    Failed,
    /// The check could not be made. Never treated as a pass.
    Unknown,
    /// The criterion does not apply to this run — no artifacts were asked for,
    /// so "the artifacts are sound" is not a bar it has to clear.
    NotApplicable,
}

impl CriterionStatus {
    pub const fn as_str(self) -> &'static str {
        match self {
            CriterionStatus::Passed => "passed",
            CriterionStatus::Failed => "failed",
            CriterionStatus::Unknown => "unknown",
            CriterionStatus::NotApplicable => "not_applicable",
        }
    }
}

/// One thing checked, and what it was decided from.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Criterion {
    /// Stable across versions. The thing a later reader matches on.
    pub criterion_id: String,
    pub status: CriterionStatus,
    /// What the verdict was read from — a count, a name, a state. Never a
    /// passage, an answer, or anything a document said.
    pub evidence: String,
}

/// What the run may be called.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Outcome {
    /// Every required criterion passed.
    Succeeded,
    /// Something could not be checked, or an effect is ambiguous.
    NeedsReview,
    /// A criterion was checked and did not pass.
    Failed,
}

impl Outcome {
    pub const fn as_str(self) -> &'static str {
        match self {
            Outcome::Succeeded => "succeeded",
            Outcome::NeedsReview => "needs_review",
            Outcome::Failed => "failed",
        }
    }
}

/// The completed check.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompletionVerification {
    pub outcome: Outcome,
    pub criteria: Vec<Criterion>,
    /// RFC 3339, UTC.
    pub verified_at: String,
    pub verifier_version: u32,
}

impl CompletionVerification {
    /// The model can propose completion, but cannot override this gate. Preserve
    /// cancellations and explicit controller stops instead of reclassifying them.
    pub fn enforce_outcome(&self, reported: super::outcome::RunOutcome) -> super::outcome::RunOutcome {
        use super::outcome::RunOutcome;
        if !reported.is_success() {
            return reported;
        }
        match self.outcome {
            Outcome::Succeeded => RunOutcome::Completed,
            Outcome::Failed => RunOutcome::Failed { detail: self.explain() },
            Outcome::NeedsReview => RunOutcome::NeedsReview { detail: self.explain() },
        }
    }

    pub fn passed(&self) -> bool {
        self.outcome == Outcome::Succeeded
    }

    /// The criteria that stopped this being a pass, for the sentence an
    /// operator reads.
    pub fn blocking(&self) -> Vec<&Criterion> {
        self.criteria
            .iter()
            .filter(|criterion| {
                matches!(
                    criterion.status,
                    CriterionStatus::Failed | CriterionStatus::Unknown
                )
            })
            .collect()
    }

    pub fn explain(&self) -> String {
        match self.outcome {
            Outcome::Succeeded => "Every check this run had to pass, passed.".to_string(),
            _ => {
                let reasons = self
                    .blocking()
                    .iter()
                    .map(|criterion| format!("{} ({})", criterion.criterion_id, criterion.evidence))
                    .collect::<Vec<_>>()
                    .join("; ");
                format!(
                    "This run is not finished: {reasons}. Somebody needs to look before it is \
                     relied on."
                )
            }
        }
    }
}

/// Everything the check reads. Gathered by the caller, never by this module —
/// a checker that fetches its own inputs cannot be tested against inputs it
/// would refuse, and the refusals are the point.
#[derive(Debug, Clone, Default)]
pub struct CompletionInputs {
    /// A failed read must not turn missing durable obligations into an empty list.
    pub evidence_error: Option<String>,
    /// A failure the run itself reported.
    pub failure: Option<String>,
    /// Plan steps the run never reached.
    pub unfinished_steps: usize,
    /// Pending OR unknown side effects. Each is an idempotency key.
    pub unknown_effects: Vec<String>,
    /// Approvals raised and never decided.
    pub pending_approvals: usize,
    /// Artifacts produced, and whether each re-opened soundly.
    pub artifacts: Vec<(String, bool)>,
    /// Whether the grounding report says the answer is usable. `None` when
    /// nothing checked it.
    pub grounding_ready: Option<bool>,
    /// Whether the run produced any answer at all.
    pub has_answer: bool,
}

/// Decides whether the run may be called finished.
///
/// Pure. Every criterion is evaluated and recorded even once the outcome is
/// settled, because an operator looking at a failed run wants to know which of
/// the other checks also failed, not just the first one.
pub fn verify(
    inputs: &CompletionInputs,
    at: chrono::DateTime<chrono::Utc>,
) -> CompletionVerification {
    let mut criteria = Vec::new();

    criteria.push(Criterion {
        criterion_id: "evidence.available".into(),
        status: if inputs.evidence_error.is_none() { CriterionStatus::Passed } else { CriterionStatus::Unknown },
        evidence: inputs.evidence_error.clone().unwrap_or_else(|| "durable obligations were read successfully".into()),
    });

    // The run's own report of itself. Not the model's word about the task —
    // this is set when the loop or the provider reported an error.
    criteria.push(match &inputs.failure {
        None => Criterion {
            criterion_id: "run.no_reported_failure".into(),
            status: CriterionStatus::Passed,
            evidence: "the run reported no failure".into(),
        },
        Some(failure) => Criterion {
            criterion_id: "run.no_reported_failure".into(),
            status: CriterionStatus::Failed,
            evidence: failure.clone(),
        },
    });

    criteria.push(Criterion {
        criterion_id: "plan.every_step_reached".into(),
        status: if inputs.unfinished_steps == 0 {
            CriterionStatus::Passed
        } else {
            CriterionStatus::Failed
        },
        evidence: format!("{} plan step(s) never reached", inputs.unfinished_steps),
    });

    // The one that most often turns a "finished" run into a question. An effect
    // nobody settled means a document may or may not exist, and no amount of
    // reading the answer will say which.
    criteria.push(Criterion {
        criterion_id: "effects.none_unknown".into(),
        status: if inputs.unknown_effects.is_empty() {
            CriterionStatus::Passed
        } else {
            CriterionStatus::Unknown
        },
        evidence: if inputs.unknown_effects.is_empty() {
            "every side effect settled".into()
        } else {
            format!(
                "{} side effect(s) nobody could settle: {}",
                inputs.unknown_effects.len(),
                inputs.unknown_effects.join(", ")
            )
        },
    });

    // A run holding an undecided approval has not finished; it stopped.
    criteria.push(Criterion {
        criterion_id: "approvals.none_pending".into(),
        status: if inputs.pending_approvals == 0 {
            CriterionStatus::Passed
        } else {
            CriterionStatus::Unknown
        },
        evidence: format!("{} approval(s) still undecided", inputs.pending_approvals),
    });

    // Re-opened from disk by the caller, never taken on the model's word: a
    // document that was written and then corrupted passes every test of the
    // code that wrote it.
    let unsound: Vec<&str> = inputs
        .artifacts
        .iter()
        .filter(|(_, sound)| !sound)
        .map(|(name, _)| name.as_str())
        .collect();
    criteria.push(Criterion {
        criterion_id: "artifacts.all_sound".into(),
        status: if inputs.artifacts.is_empty() {
            CriterionStatus::NotApplicable
        } else if unsound.is_empty() {
            CriterionStatus::Passed
        } else {
            CriterionStatus::Failed
        },
        evidence: if inputs.artifacts.is_empty() {
            "the run produced no files".into()
        } else if unsound.is_empty() {
            format!("{} file(s) re-opened and sound", inputs.artifacts.len())
        } else {
            format!("could not be re-opened: {}", unsound.join(", "))
        },
    });

    criteria.push(match inputs.grounding_ready {
        // No answer is nothing to check. Reporting "nothing to verify" as a
        // pass is the one misleading outcome available here.
        None => Criterion {
            criterion_id: "answer.grounded".into(),
            status: if inputs.has_answer {
                CriterionStatus::Unknown
            } else {
                CriterionStatus::NotApplicable
            },
            evidence: if inputs.has_answer {
                "there is an answer and it was not checked".into()
            } else {
                "the run produced no answer".into()
            },
        },
        Some(true) => Criterion {
            criterion_id: "answer.grounded".into(),
            status: CriterionStatus::Passed,
            evidence: "the answer's claims resolve to evidence the run holds".into(),
        },
        Some(false) => Criterion {
            criterion_id: "answer.grounded".into(),
            status: CriterionStatus::Failed,
            evidence: "the answer makes claims the run's evidence does not support".into(),
        },
    });

    // Uncertainty needs reconciliation even when another criterion failed.
    // Calling an ambiguous external effect merely failed can invite a retry.
    let outcome = if criteria
        .iter()
        .any(|criterion| criterion.status == CriterionStatus::Unknown)
    {
        Outcome::NeedsReview
    } else if criteria
        .iter()
        .any(|criterion| criterion.status == CriterionStatus::Failed)
    {
        Outcome::Failed
    } else {
        Outcome::Succeeded
    };

    CompletionVerification {
        outcome,
        criteria,
        verified_at: at.to_rfc3339(),
        verifier_version: VERIFIER_VERSION,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clean() -> CompletionInputs {
        CompletionInputs {
            evidence_error: None,
            failure: None,
            unfinished_steps: 0,
            unknown_effects: Vec::new(),
            pending_approvals: 0,
            artifacts: vec![("approval-note.docx".into(), true)],
            grounding_ready: Some(true),
            has_answer: true,
        }
    }

    fn verdict(inputs: CompletionInputs) -> CompletionVerification {
        verify(&inputs, chrono::Utc::now())
    }

    #[test]
    fn a_run_that_did_everything_is_a_pass() {
        let report = verdict(clean());
        assert_eq!(report.outcome, Outcome::Succeeded);
        assert!(report.passed());
        assert!(report.blocking().is_empty());
        assert_eq!(report.verifier_version, VERIFIER_VERSION);
    }

    /// The headline rule: the model stopping is not the model finishing.
    #[test]
    fn an_unknown_side_effect_stops_a_run_being_called_finished() {
        let report = verdict(CompletionInputs {
            unknown_effects: vec!["effect-1".into()],
            ..clean()
        });
        assert_eq!(report.outcome, Outcome::NeedsReview);
        assert!(!report.passed());
        assert_eq!(report.blocking()[0].criterion_id, "effects.none_unknown");
    }

    #[test]
    fn an_undecided_approval_stops_a_run_being_called_finished() {
        let report = verdict(CompletionInputs {
            pending_approvals: 1,
            ..clean()
        });
        assert_eq!(report.outcome, Outcome::NeedsReview);
        assert_eq!(report.blocking()[0].criterion_id, "approvals.none_pending");
    }

    #[test]
    fn a_file_that_will_not_re_open_fails_the_run() {
        let report = verdict(CompletionInputs {
            artifacts: vec![("approval-note.docx".into(), false)],
            ..clean()
        });
        assert_eq!(report.outcome, Outcome::Failed);
        assert_eq!(report.blocking()[0].criterion_id, "artifacts.all_sound");
    }

    #[test]
    fn plan_steps_never_reached_fail_the_run() {
        let report = verdict(CompletionInputs {
            unfinished_steps: 2,
            ..clean()
        });
        assert_eq!(report.outcome, Outcome::Failed);
    }

    #[test]
    fn a_reported_failure_fails_the_run() {
        let report = verdict(CompletionInputs {
            failure: Some("the provider answered 503".into()),
            ..clean()
        });
        assert_eq!(report.outcome, Outcome::Failed);
        assert!(report.explain().contains("503"));
    }

    /// A failed check does not remove the need to reconcile an uncertain effect.
    #[test]
    fn an_uncertainty_requires_review_even_when_a_check_failed() {
        let report = verdict(CompletionInputs {
            unfinished_steps: 1,
            unknown_effects: vec!["effect-1".into()],
            ..clean()
        });
        assert_eq!(report.outcome, Outcome::NeedsReview);
        assert_eq!(report.blocking().len(), 2, "both are still reported");
    }

    /// A run with no answer has nothing to ground, and that is not a pass by
    /// omission — but it is also not a failure of grounding.
    #[test]
    fn no_answer_is_not_a_grounding_pass() {
        let report = verdict(CompletionInputs {
            grounding_ready: None,
            has_answer: false,
            artifacts: Vec::new(),
            ..clean()
        });
        let grounding = report
            .criteria
            .iter()
            .find(|criterion| criterion.criterion_id == "answer.grounded")
            .expect("the criterion is always reported");
        assert_eq!(grounding.status, CriterionStatus::NotApplicable);
    }

    /// An answer that was never checked is not an answer that passed.
    #[test]
    fn an_unchecked_answer_needs_review() {
        let report = verdict(CompletionInputs {
            grounding_ready: None,
            has_answer: true,
            ..clean()
        });
        assert_eq!(report.outcome, Outcome::NeedsReview);
    }

    /// Every criterion is reported every time, so a reader can see what was
    /// checked rather than only what failed.
    #[test]
    fn every_criterion_is_always_reported() {
        assert_eq!(verdict(clean()).criteria.len(), 7);
        assert_eq!(
            verdict(CompletionInputs {
                failure: Some("stopped".into()),
                ..clean()
            })
            .criteria
            .len(),
            7
        );
    }

    #[test]
    fn false_model_completion_is_rejected_by_the_terminal_gate() {
        use super::super::outcome::RunOutcome;
        let report = verdict(CompletionInputs { unfinished_steps: 1, ..clean() });
        assert!(matches!(report.enforce_outcome(RunOutcome::Completed), RunOutcome::Failed { .. }));
        let uncertain = verdict(CompletionInputs { unknown_effects: vec!["intent-1".into()], ..clean() });
        assert_eq!(uncertain.enforce_outcome(RunOutcome::Completed).kind(), "needsReview");
        assert_eq!(verdict(clean()).enforce_outcome(RunOutcome::Completed), RunOutcome::Completed);
    }

    #[test]
    fn completion_checks_do_not_erase_an_operator_cancellation() {
        use super::super::outcome::RunOutcome;
        let cancelled = RunOutcome::Aborted { detail: "operator stopped".into() };
        assert_eq!(verdict(clean()).enforce_outcome(cancelled.clone()), cancelled);
    }

    #[test]
    fn unavailable_obligations_are_not_an_empty_successful_ledger() {
        let report = verdict(CompletionInputs { evidence_error: Some("ledger unavailable".into()), ..clean() });
        assert_eq!(report.outcome, Outcome::NeedsReview);
        assert_eq!(report.blocking()[0].criterion_id, "evidence.available");
    }
}
