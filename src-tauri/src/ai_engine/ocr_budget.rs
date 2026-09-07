//! Estimating a document's size, and naming how much of it a turn carried.
//!
//! ## What this module used to do, and why it stopped
//!
//! It decided how much of an OCR'd document went into the prompt, and the
//! answer was always *a prefix*: `plan` sized an allowance and `take_tokens`
//! cut the text to it. The header here used to describe that as a rule, with
//! a section admitting the flaw — "the part that is included is the
//! beginning of the document, not the part most relevant to the question".
//!
//! That flaw was the whole defect. A question about page 31 of 40 was
//! answered from pages 1-17, confidently, and the only marker was a sentence
//! the model was free to ignore. It is now handled where it should be, by
//! chunking the document and choosing passages:
//! [`crate::agent_runtime::doc_pipeline`].
//!
//! `plan` and `take_tokens` are gone rather than deprecated. A public
//! function that truncates a document to fit is a hazard sitting in the tree
//! waiting for a caller, and this product has already been bitten by exactly
//! that.
//!
//! ## What is left, and who uses it
//!
//! - [`estimate_tokens`] — the `chars / 4` estimate. Still the first
//!   approximation everywhere a size is needed before a tokeniser can be
//!   asked. It is labelled an estimate wherever it travels, and where the
//!   answer has to be right it is replaced by a count from the server:
//!   [`crate::serving::probe::count_tokens`].
//! - [`InjectionStrategy`] — the vocabulary the context meter shows a person
//!   for how much of a document entered a turn. The values are unchanged,
//!   because what they mean to a reader is unchanged; only the thing that
//!   decides between them moved.

use serde::{Deserialize, Serialize};

/// The share of free budget one document's text may occupy.
pub const FULL_INCLUSION_SHARE: f64 = 0.5;

/// The floor under a document's allowance, in tokens.
///
/// Below this, injecting "the document" means injecting three sentences of it.
/// A model reading that answers confidently from a fragment nobody has marked
/// as a fragment, which is worse than being told the document did not fit.
pub const MINIMUM_USEFUL_ALLOWANCE: u32 = 512;

/// Characters per token, for the pre-call estimate.
///
/// Four is the usual figure for English prose and is wrong for dense tables,
/// which is exactly what OCR output often is. It is labelled an estimate
/// wherever it travels, and the reconciliation on the next model turn replaces
/// it with what was actually charged. See
/// [`crate::ai_engine::token_reconciliation`].
const CHARS_PER_TOKEN: usize = 4;

/// A character-count estimate of a document's token cost.
///
/// Deliberately not called `count`: nothing here counted anything.
pub fn estimate_tokens(text: &str) -> u32 {
    let chars = text.chars().count();
    if chars == 0 {
        return 0;
    }
    ((chars / CHARS_PER_TOKEN) as u32).max(1)
}

/// What will be done with one document's text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum InjectionStrategy {
    /// The whole text goes into the prompt.
    Full,
    /// Only as much of the document as fits goes in; the rest is left out.
    Chunked,
    /// None of it fits. The document is named, nothing more.
    ReferenceOnly,
}

impl InjectionStrategy {
    pub fn label(self) -> &'static str {
        match self {
            InjectionStrategy::Full => "in full",
            InjectionStrategy::Chunked => "in part",
            InjectionStrategy::ReferenceOnly => "by reference only",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_empty_document_estimates_to_zero_not_one() {
        assert_eq!(estimate_tokens(""), 0);
        assert_eq!(estimate_tokens("a"), 1, "a non-empty document is never zero");
    }

    /// The estimate is `chars / 4` and nothing more. Pinned because three
    /// separate budgets are built on it, and a change here silently moves all
    /// of them.
    #[test]
    fn the_estimate_is_four_characters_to_the_token() {
        assert_eq!(estimate_tokens(&"x".repeat(400)), 100);
        assert_eq!(estimate_tokens(&"x".repeat(4_000)), 1_000);
    }

    /// Counted in characters, not bytes. OCR output is full of degree signs,
    /// dashes and Devanagari, and estimating a Hindi page by its byte length
    /// would overstate it threefold and starve the turn that carried it.
    #[test]
    fn the_estimate_counts_characters_rather_than_bytes() {
        let devanagari = "पंप".repeat(100); // 300 chars, 900 bytes
        assert_eq!(devanagari.chars().count(), 300);
        assert_eq!(estimate_tokens(&devanagari), 75);
    }

    /// The words a person reads on the context meter's row for a document.
    /// Shown verbatim, so they are pinned here rather than left to a caller.
    #[test]
    fn each_strategy_reads_as_a_phrase_a_person_understands() {
        assert_eq!(InjectionStrategy::Full.label(), "in full");
        assert_eq!(InjectionStrategy::Chunked.label(), "in part");
        assert_eq!(InjectionStrategy::ReferenceOnly.label(), "by reference only");
    }

    /// The guard on the defect this module used to contain.
    ///
    /// `take_tokens` cut a document to a prefix, and every caller that used it
    /// produced an answer drawn from the front of a document regardless of
    /// where the answer was. It is gone, and this asserts that the file has
    /// not quietly grown it back — a compile-time check would be better, but
    /// there is no way to assert the *absence* of a function, and a reviewer
    /// reading this test will know why one was refused.
    #[test]
    fn this_module_no_longer_contains_a_way_to_truncate_a_document() {
        let source = include_str!("ocr_budget.rs");
        // A definition, not a mention: the header above explains why the
        // function was removed and has to be free to name it.
        let banned = concat!("fn ", "take", "_tokens");
        let uses: Vec<&str> = source
            .lines()
            .filter(|line| line.contains(banned))
            .filter(|line| !line.contains("concat!"))
            .collect();
        assert!(
            uses.is_empty(),
            "the prefix-truncating helper is back: {uses:?}. Passage selection lives in \
             `agent_runtime::doc_pipeline`; a document that does not fit is chunked and \
             chosen from, never cut to its first N tokens."
        );
    }
}
