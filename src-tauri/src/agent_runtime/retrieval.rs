//! What a run actually retrieved, kept for the whole run.
//!
//! A single search returns six passages and then forgets them. That is enough
//! for the model, which has them in its context, and not enough for anything
//! else: [`crate::artifacts::verifier`] resolves each `[En]` citation in the
//! final answer against the passages the task really retrieved, and it cannot
//! do that against passages nobody kept.
//!
//! So searches on the agent path go through here. Every hit is recorded against
//! the run, numbered once, and the number it was given is the number the model
//! is told to cite. Two things follow that are worth stating plainly:
//!
//! - **A marker means one passage for the life of the run.** The list only
//!   grows, and a passage found again keeps the number it already had.
//! - **One run cannot cite another's evidence.** The table is keyed by run id,
//!   the same way workspaces and calculations already are, so an invented
//!   `[E9]` in a run that retrieved four passages is caught rather than
//!   resolved against somebody else's search.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::knowledge::SearchResult;
use crate::orchestrator::runner::render_passages;

/// Passages retrieved so far, keyed by run id.
pub type RunPassages = Arc<Mutex<HashMap<String, Vec<SearchResult>>>>;

/// Records one search's hits against the run and renders them for the model.
///
/// A passage retrieved twice keeps the marker it was given the first time. The
/// model searching again with different wording is ordinary — it should not
/// make the same page of the same document into two pieces of evidence, because
/// the verifier would then count one source as corroborating itself.
pub fn record(passages: &RunPassages, run_id: &str, query: &str, hits: &[SearchResult]) -> String {
    let mut table = match passages.lock() {
        Ok(table) => table,
        Err(_) => {
            // The evidence table is how a citation is checked. Rendering the
            // passages anyway, with numbers that will resolve to nothing, would
            // produce a draft whose citations silently fail verification — so
            // this is said here, where the model can still act on it.
            return format!(
                "The passages for {query:?} were found but could not be recorded as this task's \
                 evidence, so nothing retrieved now can be cited. Report this rather than \
                 answering from it."
            );
        }
    };
    let recorded = table.entry(run_id.to_string()).or_default();

    let markers: Vec<usize> = hits
        .iter()
        .map(|hit| {
            match recorded
                .iter()
                .position(|kept| kept.chunk_id == hit.chunk_id)
            {
                Some(index) => index + 1,
                None => {
                    recorded.push(hit.clone());
                    recorded.len()
                }
            }
        })
        .collect();

    let marked: Vec<(usize, &SearchResult)> =
        markers.into_iter().zip(hits.iter()).collect();
    render_passages(query, &marked)
}

/// Records passages against the run and returns the marker each was given.
///
/// For a caller that renders the passages itself (`knowledge.hybrid_search`
/// shows excerpts and labelled scores rather than whole passages). The same
/// table and the same rule as [`record`]: a passage found again keeps its
/// marker, so the verifier resolves `[En]` to one passage for the whole run.
pub fn record_markers(passages: &RunPassages, run_id: &str, hits: &[SearchResult]) -> Option<Vec<usize>> {
    let mut table = passages.lock().ok()?;
    let recorded = table.entry(run_id.to_string()).or_default();
    Some(
        hits.iter()
            .map(|hit| match recorded.iter().position(|kept| kept.chunk_id == hit.chunk_id) {
                Some(index) => index + 1,
                None => {
                    recorded.push(hit.clone());
                    recorded.len()
                }
            })
            .collect(),
    )
}

/// Reads a named page range and records it as this run's evidence.
///
/// ## Why this exists rather than a "read the document" tool
///
/// A search returns a passage and the page it sits on. Often that is enough;
/// when it is not — a table split across a page break, a clause that continues
/// overleaf — the model needs the pages around it. The obvious way to serve that
/// is to let it read the document, and on this workbench a document is a
/// 200-page drawing set. Pasted into an 8k window it does not give the model
/// more context: it ends the run, because the inference server refuses a prompt
/// at or over its window.
///
/// So the unit of "load more" is a page range, and the range the model asked for
/// is the range it gets. That is what makes progressive disclosure work here —
/// the run holds markers, and pulls back only the region it turns out to need.
///
/// ## What it shares with `record`
///
/// Everything that matters. The passages go into the same numbered table under
/// the same rule — a passage found again keeps the marker it already had — so a
/// page pulled back after a search does not become a second piece of evidence
/// corroborating the first.
pub fn record_region(
    passages: &RunPassages,
    run_id: &str,
    document_name: &str,
    from_page: u32,
    to_page: u32,
    hits: &[SearchResult],
) -> String {
    let described = describe_region(document_name, from_page, to_page);
    let rendered = record(passages, run_id, &described, hits);
    if hits.is_empty() {
        // The empty rendering already names what was asked for, so a second
        // header would repeat it.
        return rendered;
    }
    // Said explicitly, because the rendering itself does not. A model handed
    // passages with no statement of which pages they came from will cite the
    // range it asked for rather than the range it received — and those differ
    // whenever a page holds nothing indexable.
    format!("Read {described}.

{rendered}")
}

/// How a page range reads in a sentence.
fn describe_region(document_name: &str, from_page: u32, to_page: u32) -> String {
    if from_page == to_page {
        format!("page {from_page} of {document_name}")
    } else {
        format!("pages {from_page} to {to_page} of {document_name}")
    }
}

/// Everything this run retrieved, in the order its markers refer to.
pub fn for_run(passages: &RunPassages, run_id: &str) -> Vec<SearchResult> {
    passages
        .lock()
        .ok()
        .and_then(|table| table.get(run_id).cloned())
        .unwrap_or_default()
}

/// Drops a finished run's evidence.
///
/// Called when the run ends and its report has been written. Holding every
/// passage of every run for the life of the session would grow without bound,
/// and the ones worth keeping are already in the saved task record.
pub fn forget(passages: &RunPassages, run_id: &str) {
    if let Ok(mut table) = passages.lock() {
        table.remove(run_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge::Retrieval;
    use crate::policy::Classification;

    fn passage(chunk_id: &str, text: &str) -> SearchResult {
        SearchResult {
            chunk_id: chunk_id.to_string(),
            document_sha256: "sha".to_string(),
            document_name: "Maintenance SOP".to_string(),
            text: text.to_string(),
            page: 4,
            section_path: Vec::new(),
            classification: Classification::Internal,
            score: 1.0,
            retrieval: Retrieval::Keyword,
        }
    }

    #[test]
    fn markers_carry_on_across_searches_rather_than_restarting() {
        let table: RunPassages = Arc::default();
        let first = record(&table, "r", "seal", &[passage("a", "one"), passage("b", "two")]);
        let second = record(&table, "r", "wear", &[passage("c", "three")]);

        assert!(first.contains("[E1]"), "{first}");
        assert!(first.contains("[E2]"), "{first}");
        // Not [E1] again: the verifier resolves markers against the run's whole
        // evidence list, so a second search starting over would make one number
        // mean two different passages.
        assert!(second.contains("[E3]"), "{second}");
        assert_eq!(for_run(&table, "r").len(), 3);
    }

    #[test]
    fn the_same_passage_found_twice_keeps_its_first_marker() {
        let table: RunPassages = Arc::default();
        record(&table, "r", "seal", &[passage("a", "one"), passage("b", "two")]);
        let again = record(&table, "r", "seal wear", &[passage("b", "two")]);

        assert!(again.contains("[E2]"), "{again}");
        // Counted once. Otherwise one source appears to corroborate itself.
        assert_eq!(for_run(&table, "r").len(), 2);
    }

    #[test]
    fn one_runs_evidence_is_not_anothers() {
        let table: RunPassages = Arc::default();
        record(&table, "r1", "seal", &[passage("a", "one")]);

        assert!(for_run(&table, "r2").is_empty());
    }

    #[test]
    fn finding_nothing_records_nothing_and_says_so() {
        let table: RunPassages = Arc::default();
        let out = record(&table, "r", "unicorns", &[]);

        assert!(out.contains("do not assert it"), "{out}");
        assert!(for_run(&table, "r").is_empty());
    }

    #[test]
    fn a_page_pulled_back_later_keeps_the_marker_it_already_had() {
        // The whole point of loading a region rather than a document: the model
        // asks for more of what it already cited, and the thing it cited does
        // not thereby become two pieces of evidence.
        let table: RunPassages = Arc::default();
        record(&table, "r", "seal", &[passage("a", "one")]);
        let more = record_region(&table, "r", "Maintenance SOP", 4, 4, &[passage("a", "one")]);

        assert!(more.contains("[E1]"), "{more}");
        assert_eq!(for_run(&table, "r").len(), 1);
    }

    #[test]
    fn a_loaded_region_says_which_pages_it_came_from() {
        let table: RunPassages = Arc::default();
        let out = record_region(&table, "r", "Maintenance SOP", 11, 13, &[passage("b", "two")]);

        assert!(out.contains("pages 11 to 13"), "{out}");
        assert!(out.contains("Maintenance SOP"), "{out}");
    }

    #[test]
    fn a_region_that_holds_nothing_says_so_rather_than_returning_silence() {
        // A model told nothing came back asks for a wider range. A model told
        // nothing at all assumes the page was blank and writes that down.
        let table: RunPassages = Arc::default();
        let out = record_region(&table, "r", "Maintenance SOP", 900, 901, &[]);

        assert!(out.contains("do not assert it"), "{out}");
        assert!(for_run(&table, "r").is_empty());
    }

    #[test]
    fn a_finished_run_stops_being_held() {
        let table: RunPassages = Arc::default();
        record(&table, "r", "seal", &[passage("a", "one")]);
        forget(&table, "r");

        assert!(for_run(&table, "r").is_empty());
    }
}
