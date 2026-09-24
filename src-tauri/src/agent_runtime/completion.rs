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
pub const VERIFIER_VERSION: u32 = 1;

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
    /// A failure the run itself reported.
    pub failure: Option<String>,
    /// Plan steps the run never reached.
    pub unfinished_steps: usize,
    /// Side effects nobody could settle. Each is an idempotency key.
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
    /// Every worker this run delegated to, and what it actually handed back.
    ///
    /// ## Why a child's own word is not enough
    ///
    /// Because a worker's status is the manager's, but its *payload* is its
    /// own, and the failure this guards against is a child returning
    /// `Completed` with nothing behind it. A retrieval that found no passages
    /// and a retrieval that returned six sentences citing nothing both look
    /// like success from the outside; a parent that folded either in as done
    /// would be writing an answer on a search it cannot show anybody.
    ///
    /// So the parent checks the *contract*: did the child hand back the shape
    /// it was asked for, with evidence or artifacts behind it, and did it
    /// publish anything a later step can actually read. See [`ChildOutcome`].
    pub children: Vec<ChildOutcome>,
    /// The orchestrator's plan as it stood at the end, when the run kept one
    /// (P05). `None` for a run that never planned -- a simple question needs
    /// no plan, and is not failed for lacking one.
    ///
    /// The plan's statuses are written only from receipts (settled jobs,
    /// claimed tool events, reviews), so reading them here is reading the
    /// backend's record, not the coordinator's account of its work.
    pub task_plan: Option<super::task_plan::PlanGate>,
}

/// What one delegated worker came back with, as the parent sees it.
///
/// Deliberately not the `ChildResult` itself: this is the parent's reading of
/// it, and the fields are the questions the parent has to answer rather than
/// everything the child said.
#[derive(Debug, Clone, PartialEq)]
pub struct ChildOutcome {
    pub child_id: String,
    pub profile: String,
    /// What the manager recorded — never what the worker claimed. See
    /// `subagents::result::ChildStatus`.
    pub status: String,
    /// True only for the one status that means the work is done.
    pub complete: bool,
    /// What the parent asked it to hand back, in the parent's words. Empty when
    /// the dispatch named nothing, which is itself worth seeing.
    pub deliverable: String,
    /// How many findings it returned.
    pub findings: usize,
    /// How many of those carry an evidence reference.
    ///
    /// The number that matters. A worker reporting six findings and citing
    /// nothing has produced six sentences, and a parent that repeated them
    /// would be citing the worker rather than a source.
    pub evidenced: usize,
    /// Items it committed to the task's shared memory, by id.
    ///
    /// The strongest form of the contract, because a later step can go and read
    /// them. A child that published nothing handed nothing to its siblings,
    /// whatever its result said.
    pub published: Vec<String>,
    /// Whether a later step depends on this one having finished.
    ///
    /// A failed worker nobody was waiting for is a degraded run; a failed
    /// worker a later step needed is a blocked one, and the two must not read
    /// the same.
    pub blocking: bool,
}

impl ChildOutcome {
    /// Whether this worker handed back what it was asked for.
    ///
    /// Three things, and all three are required: the manager recorded it as
    /// finished, it returned something, and what it returned rests on evidence
    /// somebody can go and read. A worker that found nothing and *said so* is
    /// handled separately — see [`Self::found_nothing`] — because "no source
    /// says this" is a legitimate deliverable and an uncited claim is not.
    pub fn honoured(&self) -> bool {
        self.complete && self.findings > 0 && self.evidenced > 0
    }

    /// Whether this worker finished and legitimately found nothing.
    ///
    /// Told apart from a broken one by the status, which the manager sets from
    /// what happened: a child that ran to completion and returned no findings
    /// genuinely searched and genuinely found none.
    pub fn found_nothing(&self) -> bool {
        self.complete && self.findings == 0
    }

    /// The sentence the parent's report carries for this worker.
    pub fn explain(&self) -> String {
        if self.honoured() {
            return format!(
                "{} finished with {} finding(s), {} of them evidenced, and published {} item(s) \
                 to the task's shared memory",
                self.profile,
                self.findings,
                self.evidenced,
                self.published.len()
            );
        }
        if self.found_nothing() {
            return format!(
                "{} finished and found nothing, which is an answer rather than a failure",
                self.profile
            );
        }
        if !self.complete {
            return format!(
                "{} did not finish ({}), so anything it returned is incomplete{}",
                self.profile,
                self.status,
                if self.blocking {
                    " and a later step depends on it"
                } else {
                    ""
                }
            );
        }
        format!(
            "{} reported {} finding(s) and none of them cite anything, so there is nothing a \
             reader could check them against",
            self.profile, self.findings
        )
    }
}

/// Every worker this run delegated to, read back off the durable record.
///
/// ## Why this is read from the log and not held in memory
///
/// Because the parent's completion check has to survive the thing it is
/// checking. A run that was interrupted and resumed has a fresh process with an
/// empty map, and a verifier reading that map would conclude the run delegated
/// to nobody — which would let a task whose only real work was done by a child
/// that died pass as complete.
///
/// So the pair of events the manager writes is the record: `subagent_started`
/// carries who the child was and what it was asked for, and `subagent_stopped`
/// carries what the manager recorded about how it ended. A child with a start
/// and no stop is one that never came back, and is reported as exactly that
/// rather than being left out.
pub fn children_of(events: &super::events::TaskEventLog, run_id: &str) -> Vec<ChildOutcome> {
    use std::collections::BTreeMap;

    let Ok(page) = events.events_since(run_id, 0) else {
        return Vec::new();
    };

    // Started first, so a stop has something to attach to.
    let mut started: BTreeMap<String, ChildOutcome> = BTreeMap::new();
    let mut order: Vec<String> = Vec::new();
    for event in &page.events {
        if event.event_type != super::events::TaskEventType::SubagentStarted {
            continue;
        }
        let payload = &event.payload;
        let Some(child_id) = payload.get("childId").and_then(|v| v.as_str()) else {
            continue;
        };
        order.push(child_id.to_string());
        started.insert(
            child_id.to_string(),
            ChildOutcome {
                child_id: child_id.to_string(),
                profile: payload
                    .get("profile")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
                    .to_string(),
                // Until a stop is seen, the honest reading of a child is that it
                // has not come back. A start with no stop stays here.
                status: "no result recorded".to_string(),
                complete: false,
                deliverable: payload
                    .get("deliverable")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string(),
                findings: 0,
                evidenced: 0,
                published: Vec::new(),
                // A child another child was told to wait for is one a later step
                // depends on, and the packet's requirement is where that is
                // recorded. Anything else is a fan-out nobody is blocked on.
                blocking: payload
                    .get("requirement")
                    .and_then(|requirement| requirement.get("mode"))
                    .and_then(|mode| mode.as_str())
                    .is_some_and(|mode| mode != "latest"),
            },
        );
    }

    for event in &page.events {
        if event.event_type != super::events::TaskEventType::SubagentStopped {
            continue;
        }
        let payload = &event.payload;
        let Some(child_id) = payload.get("childId").and_then(|v| v.as_str()) else {
            continue;
        };
        let Some(outcome) = started.get_mut(child_id) else {
            continue;
        };
        outcome.status = payload
            .get("status")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();
        // The manager's own reading, not the worker's. See
        // `subagents::result::ChildStatus::is_complete`.
        outcome.complete = payload
            .get("complete")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        outcome.findings = payload
            .get("findings")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        outcome.evidenced = payload
            .get("evidenced")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        outcome.published = payload
            .get("published")
            .and_then(|v| v.as_array())
            .map(|items| {
                items
                    .iter()
                    .filter_map(|item| item.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default();
    }

    order
        .into_iter()
        .filter_map(|child_id| started.remove(&child_id))
        .collect()
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

    // -- The workers this run delegated to ------------------------------
    //
    // A child saying "done" is not enough, and this is where that is enforced.
    // The manager already refuses to let a worker *set* its own status; these
    // two criteria go further and check the payload, because a status of
    // `completed` with six uncited sentences behind it is exactly what a parent
    // must not fold into an answer.
    if !inputs.children.is_empty() {
        let broken: Vec<&ChildOutcome> = inputs
            .children
            .iter()
            .filter(|child| !child.honoured() && !child.found_nothing())
            .collect();
        criteria.push(Criterion {
            criterion_id: "children.contract_honoured".into(),
            status: if broken.is_empty() {
                CriterionStatus::Passed
            } else {
                CriterionStatus::Failed
            },
            evidence: if broken.is_empty() {
                format!(
                    "{} delegated worker(s) handed back what they were asked for",
                    inputs.children.len()
                )
            } else {
                broken
                    .iter()
                    .map(|child| child.explain())
                    .collect::<Vec<_>>()
                    .join("; ")
            },
        });

        // A worker a later step was waiting for is a different failure from one
        // nobody needed. Named separately so an operator reading a blocked run
        // is sent to the dependency rather than to the answer.
        let blocked: Vec<&ChildOutcome> = inputs
            .children
            .iter()
            .filter(|child| child.blocking && !child.complete)
            .collect();
        criteria.push(Criterion {
            criterion_id: "children.none_blocking".into(),
            status: if blocked.is_empty() {
                CriterionStatus::Passed
            } else {
                CriterionStatus::Failed
            },
            evidence: if blocked.is_empty() {
                "no dependent step is waiting on a worker that did not finish".into()
            } else {
                format!(
                    "{} step(s) depend on worker(s) that did not finish: {}",
                    blocked.len(),
                    blocked
                        .iter()
                        .map(|child| format!("{} ({})", child.profile, child.status))
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            },
        });
    }

    // -- The orchestrator's plan ------------------------------------------
    //
    // "Done" is the plan's steps settled by receipts and, where a deliverable
    // is involved, an independent review that passed -- never the coordinator
    // saying it finished.
    if let Some(gate) = &inputs.task_plan {
        let unsettled: Vec<&String> = gate
            .unfinished
            .iter()
            .filter(|step| !step.contains("(running)"))
            .collect();
        criteria.push(Criterion {
            criterion_id: "taskplan.steps_accepted".into(),
            status: if unsettled.is_empty() {
                CriterionStatus::Passed
            } else {
                CriterionStatus::Failed
            },
            evidence: if unsettled.is_empty() {
                format!(
                    "{} of {} plan step(s) complete by receipt at plan version {}",
                    gate.completed, gate.steps, gate.version
                )
            } else {
                format!(
                    "plan version {} is unfinished: {}",
                    gate.version,
                    unsettled.iter().map(|step| step.as_str()).collect::<Vec<_>>().join("; ")
                )
            },
        });
        criteria.push(Criterion {
            criterion_id: "taskplan.no_job_running".into(),
            status: if gate.running.is_empty() {
                CriterionStatus::Passed
            } else {
                CriterionStatus::Unknown
            },
            evidence: if gate.running.is_empty() {
                "no delegated job was still running".into()
            } else {
                format!(
                    "step(s) {} still had a job running when the run ended",
                    gate.running.join(", ")
                )
            },
        });
        criteria.push(Criterion {
            criterion_id: "taskplan.independent_review".into(),
            status: match (gate.needs_review, gate.reviewed) {
                (false, _) => CriterionStatus::NotApplicable,
                (true, true) => CriterionStatus::Passed,
                (true, false) => CriterionStatus::Failed,
            },
            evidence: match (gate.needs_review, gate.reviewed) {
                (false, _) => "no step asks for a review or an accepted artifact".into(),
                (true, true) => "every step that asks for review has a passed review receipt".into(),
                (true, false) => {
                    "a step asks for review or an accepted artifact and no independent review \
                     passed"
                        .into()
                }
            },
        });
    }

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

    // Failure outranks uncertainty: a run with both a failed check and an
    // unsettled one has been shown to be wrong, and "somebody should look"
    // would understate that.
    let outcome = if criteria
        .iter()
        .any(|criterion| criterion.status == CriterionStatus::Failed)
    {
        Outcome::Failed
    } else if criteria
        .iter()
        .any(|criterion| criterion.status == CriterionStatus::Unknown)
    {
        Outcome::NeedsReview
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
            failure: None,
            unfinished_steps: 0,
            unknown_effects: Vec::new(),
            children: Vec::new(),
            pending_approvals: 0,
            artifacts: vec![("approval-note.docx".into(), true)],
            grounding_ready: Some(true),
            has_answer: true,
            task_plan: None,
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

    /// Being shown to be wrong outranks not being able to tell.
    #[test]
    fn a_failure_outranks_an_uncertainty() {
        let report = verdict(CompletionInputs {
            unfinished_steps: 1,
            unknown_effects: vec!["effect-1".into()],
            ..clean()
        });
        assert_eq!(report.outcome, Outcome::Failed);
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
        assert_eq!(verdict(clean()).criteria.len(), 6);
        assert_eq!(
            verdict(CompletionInputs {
                failure: Some("stopped".into()),
                ..clean()
            })
            .criteria
            .len(),
            6
        );
    }
}
