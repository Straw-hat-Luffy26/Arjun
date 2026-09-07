//! Whether a cheaper model may take a child's work.
//!
//! ## The temptation, and why it needs a check
//!
//! Extraction and retrieval are the two roles where a small model is genuinely
//! attractive: they are mechanical, they run often, and a 3B model that can do
//! them frees the 7B for the reasoning that actually needs it.
//!
//! They are also the two roles where a small model failing is hardest to
//! notice. A retrieval worker that quietly returns three passages instead of
//! five does not error; it returns three passages. The parent writes a
//! well-sourced answer that is missing the finding that mattered.
//!
//! So requirement 9: a cheaper model is used for these roles **only if
//! certification data says it is reliable for that role**, and the check is
//! here rather than in the router, because the router's question is "which
//! model fits this prompt" and this one is "may we spend less on this".
//!
//! ## What the certification data is worth
//!
//! Less than its name suggests, and this module should not pretend otherwise.
//! `PackageCertification` carries per-capability scores produced by a runner
//! that is not in this repository, and the pack shipped with the product has
//! fourteen of its seventeen sub-scores set to exactly the same value. That is
//! not the shape of measurement.
//!
//! So this module treats the data as **necessary and not sufficient**: an
//! absent or low score refuses the cheaper model, and a high score permits it
//! rather than recommending it. [`Decision::reason`] always names what the
//! decision rested on, so an operator reading a run manifest can see that it
//! rested on a number from a pack rather than on evidence from this site.

use serde::{Deserialize, Serialize};

use crate::model_recommendation::certified_catalog::{CertificationTier, PackageCertification};
use crate::registry::{ModelEntry, ModelRole};

/// The score a model must reach on the capability a role depends on.
///
/// One number rather than a per-role table, because the underlying data does
/// not support a finer distinction and inventing one would give the appearance
/// of calibration nobody performed.
pub const RELIABLE_AT: f64 = 85.0;

/// Roles where a cheaper model may be considered at all.
///
/// Deliberately short. Reasoning and coding are where a small model's failures
/// are visible and expensive; extraction and retrieval are where they are
/// cheap and, checked, safe.
pub const CHEAPER_ELIGIBLE: &[ModelRole] = &[ModelRole::DocumentOcr, ModelRole::Embedding];

/// Which model a child will use, and why.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Decision {
    pub model_id: String,
    pub role: ModelRole,
    /// True when this is a smaller model chosen instead of the run's own.
    pub cheaper_than_parent: bool,
    /// What the decision rested on, in words, for the run manifest.
    pub reason: String,
    /// The certification tier, where there was one. `None` means no
    /// certification data existed for this model at all.
    pub tier: Option<CertificationTier>,
    /// The score the decision turned on, where one applied.
    pub score: Option<f64>,
}

/// Why no cheaper model was used.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Refused {
    /// The role is not one where a cheaper model is considered.
    RoleNotEligible,
    /// No certification data for this model.
    Uncertified,
    /// Certified, and not well enough for this role.
    BelowFloor,
    /// Certified as experimental, whatever its numbers say.
    Experimental,
}

impl Refused {
    pub fn explain(&self, model_id: &str, role: ModelRole) -> String {
        match self {
            Refused::RoleNotEligible => format!(
                "a smaller model is not considered for {} work; the failures are too easy to miss",
                role.label()
            ),
            Refused::Uncertified => format!(
                "{model_id} has no certification data, so nothing establishes it is reliable for \
                 {} work",
                role.label()
            ),
            Refused::BelowFloor => format!(
                "{model_id} is certified below the {RELIABLE_AT} floor for {} work",
                role.label()
            ),
            Refused::Experimental => format!(
                "{model_id} is certified as experimental, which is not a basis for using it \
                 unattended"
            ),
        }
    }
}

/// The capability score a role actually depends on.
///
/// Named per role rather than averaged, because an average hides the one that
/// matters: a model with excellent reasoning and poor JSON reliability is
/// useless as a worker that must return a typed result.
fn score_for(role: ModelRole, certification: &PackageCertification) -> Option<f64> {
    let scores = &certification.numeric_scores;
    Some(match role {
        // Reading a document and returning what is on it: getting the shape
        // right matters as much as reading it, because a result the parent
        // cannot parse is a result the parent does not have.
        ModelRole::DocumentOcr => scores.json_reliability.min(scores.instruction_following),
        // Finding the right passages, and not inventing ones.
        ModelRole::Embedding => scores.hallucination_rate.min(scores.instruction_following),
        ModelRole::Reasoning => scores.reasoning_quality,
        ModelRole::Coding => scores.coding_ability,
        ModelRole::Vision => return None,
        ModelRole::Rerank => scores.instruction_following,
        // A relation extractor's only failure mode that matters is asserting a
        // link the passage does not support. Format compliance is enforced by
        // the decoder, not by the model choosing to comply.
        ModelRole::RelationExtraction => scores.hallucination_rate,
    })
}

/// Whether a model may be used for a role on the strength of its certification.
pub fn is_reliable_for(
    certification: Option<&PackageCertification>,
    role: ModelRole,
) -> Result<(f64, CertificationTier), Refused> {
    let Some(certification) = certification else {
        return Err(Refused::Uncertified);
    };
    if matches!(certification.tier, CertificationTier::Experimental) {
        return Err(Refused::Experimental);
    }
    let Some(score) = score_for(role, certification) else {
        return Err(Refused::Uncertified);
    };
    if score < RELIABLE_AT {
        return Err(Refused::BelowFloor);
    }
    Ok((score, certification.tier.clone()))
}

/// Picks the model a child will use.
///
/// `parent_model` is what the run is already using — always a legal answer, and
/// the answer whenever the cheaper path is refused. A child is never left
/// without a model because a smaller one was not certified.
pub fn choose(
    role: ModelRole,
    parent_model: &str,
    candidates: &[(&ModelEntry, Option<&PackageCertification>)],
) -> Decision {
    let fall_back = |reason: String| Decision {
        model_id: parent_model.to_string(),
        role,
        cheaper_than_parent: false,
        reason,
        tier: None,
        score: None,
    };

    if !CHEAPER_ELIGIBLE.contains(&role) {
        return fall_back(format!(
            "Using the run's own model: {}.",
            Refused::RoleNotEligible.explain(parent_model, role)
        ));
    }

    // Smallest first: the point of the exercise is to spend less, and a
    // candidate that clears the floor is good enough by definition.
    let mut ordered: Vec<&(&ModelEntry, Option<&PackageCertification>)> = candidates.iter().collect();
    ordered.sort_by(|a, b| a.0.weights_bytes.cmp(&b.0.weights_bytes));

    let mut why_not: Vec<String> = Vec::new();
    for (entry, certification) in ordered {
        if entry.id == parent_model {
            continue;
        }
        match is_reliable_for(*certification, role) {
            Ok((score, tier)) => {
                return Decision {
                    model_id: entry.id.clone(),
                    role,
                    cheaper_than_parent: true,
                    reason: format!(
                        "{} is certified {:?} and scores {score:.1} on the capability {} work \
                         depends on, above the {RELIABLE_AT} floor. Note that this rests on the \
                         certification pack rather than on measurement at this site.",
                        entry.id,
                        tier,
                        role.label()
                    ),
                    tier: Some(tier),
                    score: Some(score),
                }
            }
            Err(refused) => why_not.push(refused.explain(&entry.id, role)),
        }
    }

    fall_back(if why_not.is_empty() {
        format!(
            "Using the run's own model: no smaller model is registered for {} work.",
            role.label()
        )
    } else {
        format!("Using the run's own model: {}.", why_not.join("; "))
    })
}
