//! The live [`ContentSource`]: what a model actually supplied, plus the
//! repairs that can be made without asking it again.
//!
//! ## The shape of the problem
//!
//! [`super::production::produce`] was written around a synchronous content
//! source — ask the model, render, re-open, hand the renderer's objections
//! back, ask again. It has always worked and nothing has ever called it,
//! because in this product the model is not something a tool handler can ask.
//! The handler *is* the model's call: by the time `create_docx` runs, the model
//! has already spoken and is waiting for an answer.
//!
//! So there are two loops, not one, and they differ in kind:
//!
//! - **Inner, here.** Deterministic repairs that need no new information — a
//!   field the model put under the heading rather than the key, a plural where
//!   the template wants a singular, whitespace where it wants text. These are
//!   transcription errors, and correcting them in code is both safe and free.
//! - **Outer, the model's own next turn.** When the inner loop runs out, the
//!   tool returns the renderer's objections *by field name* and the model tries
//!   again. That is the loop the `production` module docs describe.
//!
//! What this does **not** do is invent content. A required field nobody
//! supplied stays missing, and the outcome says so. Filling it with something
//! plausible is the one failure this whole subsystem exists to prevent.
//!
//! ## What going through `produce` gains
//!
//! The single-attempt path this replaces rendered once, checked once, and on
//! failure returned a sentence to the model — leaving a broken file on disk
//! under the name the person had been told to expect. Through `produce`:
//!
//! - every attempt is written as its own revision (`note.r1.docx`,
//!   `note.r2.docx`) and none overwrites another, so "what did it get wrong the
//!   first time" has an answer on disk;
//! - the draft's standing is settled by the verifier *before* the file is
//!   stamped, so a document the run's evidence cannot support is marked DRAFT
//!   rather than presented as finished;
//! - attempts are bounded by `MAX_ATTEMPTS`, and the failure is a sentence a
//!   person can act on.

use std::collections::BTreeMap;

use super::production::{CompositionRequest, ContentSource};

/// The model's fields, with deterministic repair between attempts.
pub struct ModelSupplied {
    /// Exactly what the model sent, untouched. Kept so attempt 2 repairs the
    /// original rather than compounding attempt 1's edits.
    supplied: BTreeMap<String, String>,
}

impl ModelSupplied {
    pub fn new(supplied: BTreeMap<String, String>) -> Self {
        Self { supplied }
    }
}

/// `Recommendations` and `recommendation ` both mean `recommendation`.
fn canonical(key: &str) -> String {
    let collapsed: String = key
        .trim()
        .to_lowercase()
        .chars()
        .filter(char::is_ascii_alphanumeric)
        .collect();
    // A trailing `s` is the difference between what a template asks for and
    // what a model writes about half the time.
    collapsed.strip_suffix('s').map(str::to_string).unwrap_or(collapsed)
}

impl ContentSource for ModelSupplied {
    fn compose(&mut self, request: &CompositionRequest) -> Result<BTreeMap<String, String>, String> {
        // Attempt 1 is what the model said, with nothing done to it. If that
        // renders, no repair was needed and none is applied.
        if request.attempt == 1 {
            return Ok(self.supplied.clone());
        }

        // Attempt 2 onwards: match the supplied keys to the template's by their
        // canonical form, and by the heading the template prints. This recovers
        // `Findings` -> `findings`, `recommendations` -> `recommendation`, and
        // ` assumptions ` -> `assumptions`.
        let mut repaired: BTreeMap<String, String> = BTreeMap::new();
        let mut used: Vec<String> = Vec::new();

        for field in &request.fields {
            let wanted = canonical(&field.key);
            let heading = canonical(&field.heading);
            let found = self.supplied.iter().find(|(key, value)| {
                if value.trim().is_empty() {
                    return false;
                }
                let candidate = canonical(key);
                candidate == wanted || candidate == heading
            });
            if let Some((key, value)) = found {
                repaired.insert(field.key.clone(), value.trim().to_string());
                used.push(key.clone());
            }
        }

        // Anything the model sent that no field claimed is carried through
        // under its own name. The renderer ignores what it does not recognise,
        // and dropping it here would silently lose content on a retry.
        for (key, value) in &self.supplied {
            if !used.contains(key) && !repaired.contains_key(key) && !value.trim().is_empty() {
                repaired.insert(key.clone(), value.trim().to_string());
            }
        }

        if repaired.is_empty() {
            return Err(
                "Every field supplied was empty, so there is nothing to correct. Write the \
                 document's content and produce it again."
                    .to_string(),
            );
        }

        // If repair changed nothing, say so rather than rendering the same
        // bytes again and reporting the same failure as though it were new.
        if request.attempt > 2 && repaired == self.supplied {
            return Err(format!(
                "Nothing could be corrected without new content. What is still missing: {}.",
                request.corrections.join("; ")
            ));
        }

        Ok(repaired)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::artifacts::production::{
        CompositionField, CompositionRequest, COMPOSITION_TEMPERATURE,
    };

    fn request(attempt: usize) -> CompositionRequest {
        CompositionRequest {
            template: "approval_note".to_string(),
            fields: vec![
                CompositionField {
                    key: "findings".to_string(),
                    heading: "Findings".to_string(),
                    required: true,
                },
                CompositionField {
                    key: "recommendation".to_string(),
                    heading: "Recommendation".to_string(),
                    required: true,
                },
            ],
            temperature: COMPOSITION_TEMPERATURE,
            corrections: vec!["The \"recommendation\" field was required".to_string()],
            attempt,
        }
    }

    fn supplied(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
        pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect()
    }

    #[test]
    fn the_first_attempt_is_exactly_what_the_model_said() {
        let given = supplied(&[("findings", "The seal was worn.")]);
        let mut source = ModelSupplied::new(given.clone());
        assert_eq!(source.compose(&request(1)).expect("composes"), given);
    }

    /// The transcription error this exists for. A model that writes
    /// `Recommendations` has not failed to do the work — it named the box
    /// differently, and asking it again wastes a turn.
    #[test]
    fn a_plural_is_matched_to_the_field_the_template_wants() {
        let mut source = ModelSupplied::new(supplied(&[
            ("findings", "The seal was worn."),
            ("Recommendations", "Replace the seal at the next outage."),
        ]));
        let out = source.compose(&request(2)).expect("composes");
        assert_eq!(
            out.get("recommendation").map(String::as_str),
            Some("Replace the seal at the next outage.")
        );
    }

    #[test]
    fn a_heading_is_matched_to_its_key() {
        let mut source = ModelSupplied::new(supplied(&[
            ("Findings", "The seal was worn."),
            ("Recommendation", "Replace it."),
        ]));
        let out = source.compose(&request(2)).expect("composes");
        assert_eq!(out.get("findings").map(String::as_str), Some("The seal was worn."));
        assert_eq!(out.get("recommendation").map(String::as_str), Some("Replace it."));
    }

    #[test]
    fn whitespace_is_not_content() {
        let mut source = ModelSupplied::new(supplied(&[
            ("findings", "Real text."),
            ("recommendation", "   \n  "),
        ]));
        let out = source.compose(&request(2)).expect("composes");
        assert!(
            !out.contains_key("recommendation"),
            "a field of spaces must stay missing so the renderer names it"
        );
    }

    /// The line this must never cross.
    #[test]
    fn a_missing_field_is_never_invented() {
        let mut source = ModelSupplied::new(supplied(&[("findings", "The seal was worn.")]));
        let out = source.compose(&request(2)).expect("composes");
        assert!(
            !out.contains_key("recommendation"),
            "the repair invented a recommendation nobody wrote: {out:?}"
        );
    }

    #[test]
    fn content_under_an_unrecognised_name_is_carried_through_not_dropped() {
        let mut source = ModelSupplied::new(supplied(&[
            ("findings", "Real text."),
            ("annexure", "Something the template does not know about."),
        ]));
        let out = source.compose(&request(2)).expect("composes");
        assert!(out.contains_key("annexure"), "a retry must not lose content: {out:?}");
    }

    #[test]
    fn nothing_at_all_is_an_honest_refusal() {
        let mut source = ModelSupplied::new(supplied(&[("findings", "  ")]));
        let error = source.compose(&request(2)).expect_err("must refuse");
        assert!(error.contains("nothing to correct"), "{error}");
    }

    /// Repeating a rendering that already failed is not a retry.
    #[test]
    fn a_repair_that_changes_nothing_stops_rather_than_looping() {
        let mut source = ModelSupplied::new(supplied(&[("findings", "Real text.")]));
        let error = source.compose(&request(3)).expect_err("must refuse");
        assert!(error.contains("without new content"), "{error}");
    }
}
