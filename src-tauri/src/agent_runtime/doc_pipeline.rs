//! Making a document larger than the window answerable anyway.
//!
//! ## The failure this removes
//!
//! A turn carrying a document was assembled like this: read every page, glue
//! the text together, and if the result did not fit the model's window, take
//! the first N tokens of it and send those. The rest was not sent, and — before
//! [`super::documents`] existed — not kept either. Two things followed.
//!
//! The first is the one an operator sees:
//!
//! ```text
//! 400 request (8590 tokens) exceeds the available context size (8192 tokens)
//! ```
//!
//! That is not a rounding error in the budget. The budget was computed against
//! the window the *registry* declares while the server had been started with a
//! smaller one — see [`crate::serving::Endpoint::context_tokens`]. The turn was
//! trimmed to fit a window that did not exist.
//!
//! The second is quieter and worse: a prefix is not a summary. "The first 43%
//! of this document was included" means a question about page 31 of 40 is
//! answered from pages 1-17, confidently, with nothing on the answer saying
//! which pages it could actually see.
//!
//! ## What replaces it
//!
//! Chunk, select, cite — the ordinary shape of a retrieval pipeline, with the
//! two properties this product needs bolted on.
//!
//! 1. **Every page is chunked**, through [`crate::knowledge::chunking`] — the
//!    same structure-aware cut the collection pipeline uses, so tables stay
//!    whole and every chunk carries the heading trail above it and the page it
//!    came from.
//! 2. **Chunks are stored**, in [`super::documents`], beside the page text and
//!    under the same owner/conversation isolation. This is the part that makes
//!    the window irrelevant to what survives: the document is preserved whole,
//!    for the life of the conversation, whatever any single turn could afford.
//! 3. **A turn selects** what fits ([`select`]) — by relevance for a pointed
//!    question, by even coverage for a whole-document one.
//! 4. **What was left out is named**, with the pages it was on, so the model
//!    can fetch it: [`super::documents::DocumentStore::pages`] for a page
//!    range, `document.search` for finding a passage by its content.
//!
//! ## What this deliberately does not do
//!
//! It does not summarise a document down before storing it. The evidence an
//! answer rests on must be quotable back to the page it came from, and a
//! paraphrase produced by a 4B model at index time is not that. Selection
//! chooses among verbatim chunks; nothing is rewritten.
//!
//! It does not claim a document was fully read when a page produced no text.
//! [`Completeness`] carries the failures by page number and the prompt says so
//! in words.

use std::collections::{BTreeSet, HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::knowledge::chunking::{Chunk, ChunkKind};

/// Characters per token, matching [`crate::ai_engine::ocr_budget`].
///
/// The same estimate, deliberately, so two parts of one turn cannot disagree
/// about how large the same text is. It is an estimate everywhere it appears,
/// and where the answer has to be right it is replaced by a count from the
/// server — see [`crate::serving::probe::count_tokens`].
const CHARS_PER_TOKEN: usize = 4;

/// How much of a document's share a whole-document question spends on even
/// coverage rather than on the best matches.
///
/// A question like "what is this document about" has no distinctive terms to
/// rank against, so ranking returns essentially arbitrary chunks clustered
/// wherever the question's few common words happen to fall. Sampling across the
/// document instead is both more useful and more honest about what it is doing.
const COVERAGE_SHARE: f64 = 0.7;

/// The share of the budget pinned documents divide between themselves.
///
/// A pin is a person saying "keep this one". Honouring it by merely visiting
/// the document first is not honouring it at all: the budget is handed out in
/// turn and whatever is left goes to the last document, so a pinned document
/// visited first ended up with less of the turn than an unpinned one visited
/// last. It gets a reserved majority instead.
///
/// Not all of it. A pin is "prefer this", not "and discard everything else" —
/// a person who pins a drawing and then attaches an invoice still expects the
/// invoice to be read.
const PINNED_SHARE: f64 = 0.7;

/// Tokens charged for the label above each chunk — the page and heading trail.
///
/// Charged rather than ignored. A selection of forty chunks whose labels were
/// free is a selection that overspends by forty labels, and the overspend
/// arrives as a 400 from the server rather than as a rounding error.
const LABEL_TOKENS: u32 = 12;

/// How many failed pages are listed by number before the list is capped.
const MAX_NAMED_FAILURES: usize = 40;

/// Words too common to tell two chunks apart.
///
/// Deliberately short. A long stop list starts removing terms that are
/// meaningless in prose and decisive in a specification — "shall", "minimum",
/// "not" — and the cost of keeping a weak term is far lower than the cost of
/// dropping a decisive one.
const STOP_WORDS: &[&str] = &[
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "does", "for", "from", "how", "i", "in",
    "is", "it", "its", "me", "my", "of", "on", "or", "please", "tell", "that", "the", "their",
    "them", "there", "these", "they", "this", "to", "was", "were", "what", "when", "where",
    "which", "who", "why", "with", "you", "your",
];

/// Phrases that mean "all of it" rather than "this bit of it".
///
/// Matched against the question to choose between ranking and coverage. A false
/// positive costs a broader selection than the question needed; a false
/// negative costs a summary written from whichever pages happened to rank. That
/// asymmetry is why the list is generous.
const WHOLE_DOCUMENT_CUES: &[&str] = &[
    "summarise",
    "summarize",
    "summary",
    "overview",
    "whole document",
    "entire document",
    "the document",
    "this document",
    "all pages",
    "every page",
    "overall",
    "in general",
    "what is this",
    "what's this",
    "tell me about",
    "key points",
    "main points",
    "gist",
];

/// An estimate of a string's token cost. See [`CHARS_PER_TOKEN`].
pub fn estimate_tokens(text: &str) -> u32 {
    let chars = text.chars().count();
    if chars == 0 {
        return 0;
    }
    ((chars / CHARS_PER_TOKEN) as u32).max(1)
}

/// Proof about what happened to one document, in numbers that can be checked.
///
/// Every field is counted from work that actually ran. This is what lets the
/// product say "all 42 pages were processed" without that being a claim
/// somebody hopes is true: [`Self::fully_processed`] is false whenever a page
/// produced no text or a chunk could not be stored, and the prompt and the UI
/// both say so rather than rounding it away.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct Completeness {
    /// Pages the reader said the document has.
    pub pages_total: u32,
    /// Pages that produced text.
    pub pages_extracted: u32,
    /// Page numbers that produced nothing, in order.
    ///
    /// Named rather than counted, because "which page failed" is the question
    /// asked next and a count cannot answer it. Bounded by
    /// [`MAX_NAMED_FAILURES`]; past that the count still tells the truth.
    pub pages_failed: Vec<u32>,
    /// Chunks the cut produced across every page.
    pub chunks_total: u32,
    /// Chunks written to the store.
    ///
    /// Equal to `chunks_total` unless storing failed, which is the only way a
    /// chunk can exist and not be retrievable later.
    pub chunks_stored: u32,
    /// Estimated tokens across the whole extraction. See [`CHARS_PER_TOKEN`].
    pub extracted_tokens: u32,
    /// Characters across the whole extraction. Counted, not estimated.
    pub extracted_chars: u64,
    /// True when the *reader* stopped early — a workbook past its row cap.
    ///
    /// A property of the document that no amount of re-reading pages fixes, and
    /// the one case where "every page was processed" and "the whole document
    /// was processed" are different sentences.
    pub source_truncated: bool,
}

impl Completeness {
    /// Whether every page produced text and every chunk was stored.
    ///
    /// Deliberately strict, and deliberately not called `ok`. A scan with one
    /// blank page is *not* fully processed, and the difference between "blank"
    /// and "unreadable" is not knowable here — so this reports the fact and
    /// leaves the judgement to a person who can look at the page.
    pub fn fully_processed(&self) -> bool {
        self.pages_failed.is_empty()
            && self.chunks_stored == self.chunks_total
            && !self.source_truncated
    }

    /// The one-line account, in the words shown to a person.
    pub fn summary(&self) -> String {
        let mut line = format!(
            "{} of {} page{} read, {} section{} indexed, about {} tokens",
            self.pages_extracted,
            self.pages_total,
            if self.pages_total == 1 { "" } else { "s" },
            self.chunks_stored,
            if self.chunks_stored == 1 { "" } else { "s" },
            self.extracted_tokens
        );
        if !self.pages_failed.is_empty() {
            line.push_str(&format!(
                "; no text from page{} {}",
                if self.pages_failed.len() == 1 { "" } else { "s" },
                join_pages(&self.pages_failed)
            ));
        }
        if self.source_truncated {
            line.push_str("; the reader stopped before the end of the file");
        }
        line
    }
}

/// Counts one document's extraction and its cut.
///
/// `pages_total` is what the reader reported the document has, which is not
/// necessarily how many produced text — that difference is the whole point.
pub fn measure(
    pages_total: u32,
    page_text: &[(u32, String)],
    chunks: &[Chunk],
    stored: u32,
    source_truncated: bool,
) -> Completeness {
    let with_text: BTreeSet<u32> = page_text
        .iter()
        .filter(|(_, text)| !text.trim().is_empty())
        .map(|(page, _)| *page)
        .collect();
    let mut failed: Vec<u32> = (1..=pages_total)
        .filter(|page| !with_text.contains(page))
        .collect();
    failed.truncate(MAX_NAMED_FAILURES);

    let chars: u64 = page_text
        .iter()
        .map(|(_, text)| text.chars().count() as u64)
        .sum();
    Completeness {
        pages_total,
        pages_extracted: with_text.len() as u32,
        pages_failed: failed,
        chunks_total: chunks.len() as u32,
        chunks_stored: stored,
        extracted_tokens: (chars / CHARS_PER_TOKEN as u64).min(u64::from(u32::MAX)) as u32,
        extracted_chars: chars,
        source_truncated,
    }
}

/// `[3]` reads "3"; `[3, 4, 5, 9]` reads "3-5 and 9".
///
/// Written out because a list of forty page numbers in a prompt is noise the
/// model has to parse, while "no text from pages 12-31" is one fact.
pub fn join_pages(pages: &[u32]) -> String {
    if pages.is_empty() {
        return String::new();
    }
    let mut runs: Vec<(u32, u32)> = Vec::new();
    for page in pages {
        match runs.last_mut() {
            Some(run) if run.1 + 1 == *page => run.1 = *page,
            _ => runs.push((*page, *page)),
        }
    }
    let parts: Vec<String> = runs
        .iter()
        .map(|(from, to)| {
            if from == to {
                from.to_string()
            } else {
                format!("{from}-{to}")
            }
        })
        .collect();
    match parts.len() {
        1 => parts[0].clone(),
        2 => format!("{} and {}", parts[0], parts[1]),
        _ => format!(
            "{} and {}",
            parts[..parts.len() - 1].join(", "),
            parts[parts.len() - 1]
        ),
    }
}

/// One document as the selector sees it.
pub struct Candidate<'a> {
    pub sha256: &'a str,
    pub name: &'a str,
    pub chunks: &'a [Chunk],
    /// Highest page number the document has, for the "and the rest is on pages
    /// X-Y" line.
    pub pages: u32,
    /// True when the person pinned this document. A pinned document is served
    /// first and is the last thing dropped.
    pub pinned: bool,
}

/// Why one chunk is in this turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum SelectionReason {
    /// Matched terms in the question.
    Relevance,
    /// Included so the document is covered evenly rather than from the front.
    Coverage,
    /// Small enough that the whole document fitted, so nothing was chosen.
    WholeDocument,
}

/// One chunk chosen for a turn.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Selected {
    pub document_sha256: String,
    pub document_name: String,
    pub chunk_id: String,
    pub ordinal: u32,
    pub page: u32,
    pub section_path: Vec<String>,
    pub text: String,
    pub tokens: u32,
    /// Reported so the meter can say which passages were chosen for relevance
    /// and which for coverage, rather than implying all of them were matches.
    pub reason: SelectionReason,
}

/// What one document could not fit into this turn.
///
/// Never "the document was truncated". The text is in the store, whole; this
/// says which parts of it are not in *this* turn and how to ask for them. That
/// distinction is the difference between losing evidence and paging through it.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Omission {
    pub sha256: String,
    pub name: String,
    pub chunks_total: u32,
    pub chunks_included: u32,
    /// Pages with at least one included chunk, in order.
    pub pages_included: Vec<u32>,
    /// Pages with no included chunk, in order.
    pub pages_omitted: Vec<u32>,
    pub pages: u32,
}

/// What one turn's documents contributed, and what they could not.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Selection {
    pub chosen: Vec<Selected>,
    /// Per document: what was left out of *this turn* and where to find it.
    pub omitted: Vec<Omission>,
    /// Estimated tokens across [`Self::chosen`].
    pub tokens: u32,
    /// The budget this selection was fitted to.
    pub budget: u32,
}

impl Selection {
    /// True when every chunk of every document made it in.
    pub fn complete(&self) -> bool {
        self.omitted
            .iter()
            .all(|o| o.chunks_included >= o.chunks_total)
    }

    /// Chunks left out of this turn, across every document.
    pub fn omitted_chunks(&self) -> u32 {
        self.omitted
            .iter()
            .map(|o| o.chunks_total.saturating_sub(o.chunks_included))
            .sum()
    }
}

/// Whether the question is about the document as a whole.
///
/// See [`WHOLE_DOCUMENT_CUES`] for why this leans towards saying yes.
pub fn asks_about_the_whole_document(question: &str) -> bool {
    let lowered = question.to_lowercase();
    WHOLE_DOCUMENT_CUES.iter().any(|cue| lowered.contains(cue))
}

/// The searchable terms in a question.
fn terms(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|token| token.len() > 1)
        .filter(|token| !STOP_WORDS.contains(token))
        .map(|token| token.to_string())
        .collect()
}

/// Every word in a chunk, heading trail included.
fn words_of(chunk: &Chunk) -> Vec<String> {
    let mut haystack = chunk.section_path.join(" ").to_lowercase();
    haystack.push(' ');
    haystack.push_str(&chunk.text.to_lowercase());
    haystack
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(|w| w.to_string())
        .collect()
}

/// How well one chunk answers the question, higher is better.
///
/// A deliberately simple lexical score — term frequency damped by repetition,
/// rare terms weighted above common ones, and the heading trail searched
/// alongside the body. It is not an embedding model and does not pretend to be:
/// nothing on an air-gapped workstation guarantees an embedding model is
/// installed, and a scorer that silently degraded to nothing when one was
/// missing would be worse than one that never claimed to be semantic.
///
/// What it is good at is what this product is actually asked: tag numbers,
/// clause numbers, part names, units. `PV-2201` is one token, and it is either
/// in the chunk or it is not.
fn score(
    chunk: &Chunk,
    query: &[String],
    document_frequency: &HashMap<String, u32>,
    total: u32,
) -> f64 {
    if query.is_empty() {
        return 0.0;
    }
    let words = words_of(chunk);
    if words.is_empty() {
        return 0.0;
    }

    let mut counts: HashMap<&str, u32> = HashMap::new();
    for word in &words {
        *counts.entry(word.as_str()).or_insert(0) += 1;
    }

    let mut total_score = 0.0;
    for term in query {
        let frequency = *counts.get(term.as_str()).unwrap_or(&0);
        if frequency == 0 {
            continue;
        }
        // Inverse document frequency: a term in every chunk separates nothing,
        // and a term in one chunk is the answer.
        let in_chunks = f64::from(*document_frequency.get(term).unwrap_or(&1));
        let idf = ((f64::from(total) + 1.0) / (in_chunks + 0.5)).ln().max(0.0);
        // Saturating term frequency. Ten mentions is not ten times the evidence
        // of one, and without damping a long chunk that repeats a word beats the
        // short chunk that defines it.
        let tf = f64::from(frequency) / (f64::from(frequency) + 1.5);
        total_score += idf * tf;
    }
    // A table matching the question is usually *the* answer in this domain —
    // the thickness table, the torque table — so it is not penalised for being
    // long the way a rambling prose chunk is.
    if chunk.kind == ChunkKind::Table && total_score > 0.0 {
        total_score *= 1.15;
    }
    total_score
}

/// Ranks chunks against a question, best first, dropping non-matches.
///
/// The same scorer [`select`] uses, exposed because `document.search` must
/// rank the way the turn ranked. Two scorers would mean a model told "this
/// passage was not relevant enough to include" finding it by searching for the
/// same words, or worse, not finding it.
///
/// Frequencies are computed over the chunks passed in, which is the right
/// corpus for both callers: what makes a term distinctive is how it is spread
/// across the material actually on offer.
pub fn rank_chunks<'a>(question: &str, chunks: &'a [Chunk]) -> Vec<(f64, &'a Chunk)> {
    let query = terms(question);
    if query.is_empty() {
        return Vec::new();
    }
    let mut frequency: HashMap<String, u32> = HashMap::new();
    for chunk in chunks {
        let present: HashSet<String> = words_of(chunk).into_iter().collect();
        for term in &query {
            if present.contains(term) {
                *frequency.entry(term.clone()).or_insert(0) += 1;
            }
        }
    }
    let total = chunks.len() as u32;
    let mut ranked: Vec<(f64, &Chunk)> = chunks
        .iter()
        .map(|chunk| (score(chunk, &query, &frequency, total), chunk))
        .filter(|(value, _)| *value > 0.0)
        .collect();
    ranked.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.ordinal.cmp(&b.1.ordinal))
    });
    ranked
}

/// How many chunks mention each query term, across everything on offer.
fn document_frequencies(
    candidates: &[Candidate<'_>],
    query: &[String],
) -> (HashMap<String, u32>, u32) {
    let mut frequency: HashMap<String, u32> = HashMap::new();
    let mut total = 0u32;
    for candidate in candidates {
        for chunk in candidate.chunks {
            total += 1;
            let present: HashSet<String> = words_of(chunk).into_iter().collect();
            for term in query {
                if present.contains(term) {
                    *frequency.entry(term.clone()).or_insert(0) += 1;
                }
            }
        }
    }
    (frequency, total)
}

/// Chooses what goes into this turn.
///
/// ## The rules, in order
///
/// 1. **Everything, if everything fits.** A one-page invoice is not improved by
///    being retrieved from itself, and a model that can see the whole document
///    should.
/// 2. Otherwise the budget is split between documents, pinned first. A turn
///    carrying three documents must not spend all of itself on the first.
/// 3. Within a document, chunks are chosen by relevance, by coverage, or by
///    both — see [`COVERAGE_SHARE`].
/// 4. Whatever is chosen is emitted **in document order**, never in score
///    order. A model reading page 31 before page 4 draws conclusions about
///    sequence that are not in the document.
///
/// `budget` is in tokens and is what is genuinely free for document text this
/// turn: the caller has already subtracted the question, the system prompt, the
/// conversation history and the reply reserve.
pub fn select(question: &str, candidates: &[Candidate<'_>], budget: u32) -> Selection {
    let mut chosen: Vec<Selected> = Vec::new();
    let mut omitted: Vec<Omission> = Vec::new();

    if candidates.is_empty() || budget == 0 {
        for candidate in candidates {
            omitted.push(omission(candidate, &[]));
        }
        return Selection {
            chosen,
            omitted,
            tokens: 0,
            budget,
        };
    }

    let query = terms(question);
    let whole = asks_about_the_whole_document(question) || query.is_empty();
    let (frequency, total_chunks) = document_frequencies(candidates, &query);

    // Pinned documents first, then in the order the caller gave — which is the
    // order they were attached.
    let mut order: Vec<usize> = (0..candidates.len()).collect();
    order.sort_by_key(|i| !candidates[*i].pinned);

    // Two pools, so a pin is a reservation rather than a place in a queue. See
    // [`PINNED_SHARE`]. Within a pool the budget is split evenly and earlier
    // documents leave what they cannot spend to later ones — an even split
    // alone wastes the budget whenever one document is a single page.
    let mut pinned_left = candidates.iter().filter(|c| c.pinned).count() as u32;
    let mut plain_left = candidates.len() as u32 - pinned_left;
    let mut pinned_pool = if pinned_left == 0 {
        0
    } else if plain_left == 0 {
        budget
    } else {
        (f64::from(budget) * PINNED_SHARE) as u32
    };
    let mut plain_pool = budget.saturating_sub(pinned_pool);
    let mut handed_over = false;

    for index in order {
        let candidate = &candidates[index];
        let share = if candidate.pinned {
            let share = if pinned_left <= 1 {
                pinned_pool
            } else {
                pinned_pool / pinned_left
            };
            pinned_left = pinned_left.saturating_sub(1);
            share
        } else {
            // Whatever the pinned documents did not need is not wasted; it
            // joins the pool the rest are drawing from. Done once, on the first
            // unpinned document, because by then every pinned one has spent.
            if !handed_over {
                plain_pool = plain_pool.saturating_add(pinned_pool);
                pinned_pool = 0;
                handed_over = true;
            }
            let share = if plain_left <= 1 {
                plain_pool
            } else {
                plain_pool / plain_left
            };
            plain_left = plain_left.saturating_sub(1);
            share
        };

        let picked = pick_within(candidate, &query, whole, share, &frequency, total_chunks);
        let spent: u32 = picked.iter().map(|c| c.tokens).sum();
        if candidate.pinned {
            pinned_pool = pinned_pool.saturating_sub(spent);
        } else {
            plain_pool = plain_pool.saturating_sub(spent);
        }
        omitted.push(omission(candidate, &picked));
        chosen.extend(picked);
    }

    // Document order, then reading order within a document. See rule 4.
    chosen.sort_by(|a, b| {
        a.document_sha256
            .cmp(&b.document_sha256)
            .then(a.ordinal.cmp(&b.ordinal))
    });

    let tokens = chosen.iter().map(|c| c.tokens).sum();
    Selection {
        chosen,
        omitted,
        tokens,
        budget,
    }
}

/// The token price of one chunk, label included.
fn cost(chunk: &Chunk) -> u32 {
    estimate_tokens(&chunk.text).saturating_add(LABEL_TOKENS)
}

/// Chooses within one document, against one document's share of the budget.
fn pick_within(
    candidate: &Candidate<'_>,
    query: &[String],
    whole: bool,
    share: u32,
    frequency: &HashMap<String, u32>,
    total_chunks: u32,
) -> Vec<Selected> {
    let everything: u32 = candidate.chunks.iter().map(cost).sum();

    // Rule 1.
    if everything <= share {
        return candidate
            .chunks
            .iter()
            .map(|chunk| selected(candidate, chunk, SelectionReason::WholeDocument, cost(chunk)))
            .collect();
    }

    let mut taken: Vec<Selected> = Vec::new();
    let mut spent = 0u32;
    let mut used: HashSet<u32> = HashSet::new();

    // Coverage: chunks spread evenly across the document by position, so a
    // summary is written from the whole of it rather than from the front.
    let coverage_budget = if whole {
        (f64::from(share) * COVERAGE_SHARE) as u32
    } else {
        0
    };
    if coverage_budget > 0 {
        for index in evenly_spaced(candidate.chunks, coverage_budget) {
            let chunk = &candidate.chunks[index];
            let price = cost(chunk);
            if spent + price > coverage_budget {
                continue;
            }
            spent += price;
            used.insert(chunk.ordinal);
            taken.push(selected(candidate, chunk, SelectionReason::Coverage, price));
        }
    }

    // Relevance: the best of what is left, until the share is spent.
    let mut ranked: Vec<(f64, &Chunk)> = candidate
        .chunks
        .iter()
        .filter(|chunk| !used.contains(&chunk.ordinal))
        .map(|chunk| (score(chunk, query, frequency, total_chunks), chunk))
        .filter(|(value, _)| *value > 0.0)
        .collect();
    ranked.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.ordinal.cmp(&b.1.ordinal))
    });

    for (_, chunk) in ranked {
        let price = cost(chunk);
        if spent + price > share {
            continue;
        }
        spent += price;
        used.insert(chunk.ordinal);
        taken.push(selected(candidate, chunk, SelectionReason::Relevance, price));
    }

    // Whatever relevance did not spend is spent on coverage.
    //
    // Ranking stops when it runs out of *matches*, not when it runs out of
    // budget, and on a pointed question about a long document that is almost
    // immediately: "what is the code on page 101" matches page 101 and nothing
    // else, takes one passage, and returns a turn holding a tenth of what the
    // window could carry. Measured on the validation set, three of eight
    // documents were answered wrongly for exactly this reason — and on the
    // 200-page one the single affordable passage was the *filler* half of page
    // 101 rather than the half with the code on it.
    //
    // An unspent budget is a window the operator paid for and did not get. So
    // the remainder is filled with passages spread across the document, which
    // is also the right shape of guess: the neighbours of a match are where the
    // rest of its clause lives, and a spread is what makes a second, unrelated
    // part of the question answerable at all.
    //
    // This subsumes the old "nothing matched at all" case. A document whose
    // terms appear nowhere still contributes a spread, so the model can see it
    // is the wrong document rather than being told only that one exists.
    if spent < share {
        for index in evenly_spaced(candidate.chunks, share - spent) {
            let chunk = &candidate.chunks[index];
            if used.contains(&chunk.ordinal) {
                continue;
            }
            let price = cost(chunk);
            if spent + price > share {
                continue;
            }
            spent += price;
            used.insert(chunk.ordinal);
            taken.push(selected(candidate, chunk, SelectionReason::Coverage, price));
        }
    }

    taken
}

/// Indices spread evenly across a document, as many as the budget allows.
///
/// Even by *position*, not by page: a document whose first page is dense and
/// whose last thirty are sparse should still be sampled across all of it. The
/// first and last chunks are always among them — front matter and whatever the
/// document concludes with are both places a summary has to look.
fn evenly_spaced(chunks: &[Chunk], budget: u32) -> Vec<usize> {
    let count = chunks.len();
    if count == 0 || budget == 0 {
        return Vec::new();
    }
    let average = {
        let total: u32 = chunks.iter().map(cost).sum();
        (total / count as u32).max(1)
    };
    let affordable = ((budget / average) as usize).clamp(1, count);
    if affordable >= count {
        return (0..count).collect();
    }
    if affordable == 1 {
        return vec![0];
    }
    (0..affordable)
        .map(|i| i * (count - 1) / (affordable - 1))
        .collect()
}

fn selected(
    candidate: &Candidate<'_>,
    chunk: &Chunk,
    reason: SelectionReason,
    tokens: u32,
) -> Selected {
    Selected {
        document_sha256: candidate.sha256.to_string(),
        document_name: candidate.name.to_string(),
        chunk_id: chunk.id.clone(),
        ordinal: chunk.ordinal,
        page: chunk.page,
        section_path: chunk.section_path.clone(),
        text: chunk.text.clone(),
        tokens,
        reason,
    }
}

fn omission(candidate: &Candidate<'_>, picked: &[Selected]) -> Omission {
    let included: BTreeSet<u32> = picked.iter().map(|c| c.page).collect();
    let all: BTreeSet<u32> = candidate.chunks.iter().map(|c| c.page).collect();
    Omission {
        sha256: candidate.sha256.to_string(),
        name: candidate.name.to_string(),
        chunks_total: candidate.chunks.len() as u32,
        chunks_included: picked.len() as u32,
        pages_included: included.iter().copied().collect(),
        pages_omitted: all.difference(&included).copied().collect(),
        pages: candidate.pages,
    }
}

/// Renders a selection as prompt text.
///
/// Every chunk is labelled with its page and heading trail, because that is
/// what makes it citable — and because a model handed four passages with no
/// markers between them reads them as continuous prose and will happily invent
/// the transition.
pub fn render(selection: &Selection, completeness: &HashMap<String, Completeness>) -> String {
    let mut out = String::new();
    let mut current = String::new();

    for chunk in &selection.chosen {
        if chunk.document_sha256 != current {
            if !current.is_empty() {
                out.push('\n');
            }
            current = chunk.document_sha256.clone();
            out.push_str(&format!("--- {} ---\n", chunk.document_name));
            if let Some(counted) = completeness.get(&chunk.document_sha256) {
                out.push_str(&format!("({})\n", counted.summary()));
            }
        }
        let label = if chunk.section_path.is_empty() {
            format!("[page {}]", chunk.page)
        } else {
            format!("[page {} — {}]", chunk.page, chunk.section_path.join(" › "))
        };
        out.push_str(&label);
        out.push('\n');
        out.push_str(chunk.text.trim());
        out.push_str("\n\n");
    }

    for omission in &selection.omitted {
        if omission.chunks_included >= omission.chunks_total {
            continue;
        }
        // The sentence that replaces silent truncation. It names what is not
        // here, says the text still exists, and says how to reach it — all
        // three, because a model told only the first apologises instead of
        // asking.
        out.push_str(&format!(
            "[{} — {} of {} sections are shown above. The rest of the document was read and \
             stored; it is not lost. Pages not shown here: {}. Use document.search to find a \
             passage by its content, or document.read_pages to read a page range.]\n",
            omission.name,
            omission.chunks_included,
            omission.chunks_total,
            if omission.pages_omitted.is_empty() {
                "none".to_string()
            } else {
                join_pages(&omission.pages_omitted)
            }
        ));
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const DOC: &str = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
    const OTHER: &str = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";

    fn chunk(ordinal: u32, page: u32, text: &str) -> Chunk {
        Chunk {
            id: format!("c{ordinal}"),
            document_sha256: DOC.to_string(),
            ordinal,
            text: text.to_string(),
            page,
            section_path: Vec::new(),
            kind: ChunkKind::Prose,
            char_count: text.len() as u32,
        }
    }

    fn candidate(chunks: &[Chunk]) -> Candidate<'_> {
        Candidate {
            sha256: DOC,
            name: "report.pdf",
            chunks,
            pages: chunks.iter().map(|c| c.page).max().unwrap_or(0),
            pinned: false,
        }
    }

    #[test]
    fn a_document_that_fits_goes_in_whole() {
        let chunks = vec![chunk(0, 1, "short"), chunk(1, 2, "also short")];
        let picked = select("anything", &[candidate(&chunks)], 10_000);
        assert_eq!(picked.chosen.len(), 2);
        assert!(picked.complete(), "nothing should be reported as omitted");
    }

    #[test]
    fn a_document_larger_than_the_budget_still_answers_from_the_right_page() {
        // Forty pages of filler with the answer on page 31.
        let mut chunks: Vec<Chunk> = (0..40)
            .map(|i| {
                chunk(
                    i,
                    i + 1,
                    &format!("Routine paragraph about general matters, number {i}. ").repeat(20),
                )
            })
            .collect();
        chunks[30] = chunk(30, 31, "The flange gasket torque is 47 Nm on revision C.");

        let picked = select("what is the flange gasket torque", &[candidate(&chunks)], 600);
        assert!(
            picked.chosen.iter().any(|c| c.page == 31),
            "the page holding the answer was not selected: {:?}",
            picked.chosen.iter().map(|c| c.page).collect::<Vec<_>>()
        );
        assert!(!picked.complete(), "a 40-page document did not fit 600 tokens");
    }

    #[test]
    fn the_selection_never_exceeds_its_budget() {
        let chunks: Vec<Chunk> = (0..200)
            .map(|i| {
                chunk(
                    i,
                    i / 4 + 1,
                    &"dense table row with many numbers 1 2 3 4 5 ".repeat(30),
                )
            })
            .collect();
        for budget in [200u32, 1_000, 4_000, 9_999] {
            let picked = select("torque", &[candidate(&chunks)], budget);
            assert!(
                picked.tokens <= budget,
                "a selection spent {} of a {budget}-token budget",
                picked.tokens
            );
        }
    }

    #[test]
    fn selected_passages_arrive_in_reading_order() {
        let mut chunks: Vec<Chunk> = (0..30)
            .map(|i| chunk(i, i + 1, &format!("filler {i} ").repeat(40)))
            .collect();
        // Put the strongest match late, so score order and reading order differ.
        chunks[25] = chunk(25, 26, "torque torque torque");
        chunks[3] = chunk(3, 4, "torque once");
        let picked = select("torque", &[candidate(&chunks)], 800);
        let ordinals: Vec<u32> = picked.chosen.iter().map(|c| c.ordinal).collect();
        let mut sorted = ordinals.clone();
        sorted.sort_unstable();
        assert_eq!(ordinals, sorted, "passages were not in reading order");
    }

    #[test]
    fn a_whole_document_question_is_answered_from_across_the_document() {
        let chunks: Vec<Chunk> = (0..60)
            .map(|i| {
                chunk(
                    i,
                    i + 1,
                    &format!("Section {i} discusses a distinct subject. ").repeat(20),
                )
            })
            .collect();
        let picked = select("summarise this document", &[candidate(&chunks)], 1_500);
        let pages: Vec<u32> = picked.chosen.iter().map(|c| c.page).collect();
        assert!(pages.iter().any(|p| *p <= 5), "no early page: {pages:?}");
        assert!(
            pages.iter().any(|p| *p >= 55),
            "no late page — a summary was written from the front: {pages:?}"
        );
    }

    #[test]
    fn what_did_not_fit_is_named_rather_than_dropped_quietly() {
        let chunks: Vec<Chunk> = (0..40)
            .map(|i| chunk(i, i + 1, &format!("page {i} content ").repeat(60)))
            .collect();
        let picked = select("torque", &[candidate(&chunks)], 500);
        let rendered = render(&picked, &HashMap::new());
        assert!(
            rendered.contains("was read and stored; it is not lost"),
            "the prompt did not say the rest survives: {rendered}"
        );
        assert!(
            rendered.contains("document.search"),
            "the prompt did not say how to reach the rest"
        );
    }

    #[test]
    fn every_passage_carries_the_page_it_came_from() {
        let chunks = vec![chunk(0, 7, "something")];
        let picked = select("something", &[candidate(&chunks)], 10_000);
        let rendered = render(&picked, &HashMap::new());
        assert!(rendered.contains("[page 7]"), "no page label: {rendered}");
    }

    #[test]
    fn a_page_that_produced_no_text_is_counted_as_a_failure_not_as_a_success() {
        let pages = vec![(1u32, "text".to_string()), (3u32, "text".to_string())];
        let counted = measure(4, &pages, &[], 0, false);
        assert_eq!(counted.pages_extracted, 2);
        assert_eq!(counted.pages_failed, vec![2, 4]);
        assert!(!counted.fully_processed());
        assert!(counted.summary().contains("no text from pages 2 and 4"));
    }

    #[test]
    fn a_document_read_end_to_end_reports_itself_complete() {
        let pages: Vec<(u32, String)> = (1..=3).map(|p| (p, format!("page {p}"))).collect();
        let chunks = vec![chunk(0, 1, "a"), chunk(1, 2, "b")];
        let counted = measure(3, &pages, &chunks, 2, false);
        assert!(counted.fully_processed());
        assert_eq!(counted.chunks_stored, counted.chunks_total);
    }

    #[test]
    fn a_reader_that_stopped_early_is_never_reported_as_complete() {
        let pages: Vec<(u32, String)> = (1..=3).map(|p| (p, format!("page {p}"))).collect();
        let counted = measure(3, &pages, &[], 0, true);
        assert!(
            !counted.fully_processed(),
            "a file the reader cut short cannot be called fully processed"
        );
        assert!(counted.summary().contains("stopped before the end"));
    }

    #[test]
    fn page_runs_read_as_ranges() {
        assert_eq!(join_pages(&[3]), "3");
        assert_eq!(join_pages(&[3, 4, 5]), "3-5");
        assert_eq!(join_pages(&[3, 4, 5, 9]), "3-5 and 9");
        assert_eq!(join_pages(&[1, 4, 7]), "1, 4 and 7");
    }

    #[test]
    fn two_documents_both_reach_the_turn() {
        let first: Vec<Chunk> = (0..30)
            .map(|i| chunk(i, i + 1, &"alpha ".repeat(200)))
            .collect();
        let mut second: Vec<Chunk> = (0..30)
            .map(|i| chunk(i, i + 1, &"beta ".repeat(200)))
            .collect();
        for piece in &mut second {
            piece.document_sha256 = OTHER.to_string();
        }
        let a = Candidate {
            sha256: DOC,
            name: "first.pdf",
            chunks: &first,
            pages: 30,
            pinned: false,
        };
        let b = Candidate {
            sha256: OTHER,
            name: "second.pdf",
            chunks: &second,
            pages: 30,
            pinned: false,
        };
        let picked = select("alpha beta", &[a, b], 2_000);
        let names: HashSet<&str> = picked
            .chosen
            .iter()
            .map(|c| c.document_name.as_str())
            .collect();
        assert!(
            names.contains("first.pdf") && names.contains("second.pdf"),
            "one document took the whole turn: {names:?}"
        );
    }

    /// The defect the validation table exposed: relevance stops at the last
    /// match, not at the last token, so a pointed question about a long
    /// document used to return one passage and leave the window empty.
    #[test]
    fn a_pointed_question_still_spends_the_budget_it_was_given() {
        // One passage mentions "torque"; the other ninety-nine do not. Ranking
        // alone therefore takes exactly one and stops.
        let mut chunks: Vec<Chunk> = (0..100)
            .map(|i| chunk(i, i + 1, &format!("Routine paragraph {i}. ").repeat(12)))
            .collect();
        chunks[50] = chunk(50, 51, "The torque figure is 47 Nm.");

        let picked = select("what is the torque figure", &[candidate(&chunks)], 3_000);
        assert!(
            picked.chosen.iter().any(|c| c.page == 51),
            "the matching passage was dropped"
        );
        assert!(
            picked.tokens > 3_000 / 2,
            "only {} of a 3,000-token budget was used; the rest of the window was left empty",
            picked.tokens
        );
        assert!(
            picked.tokens <= 3_000,
            "the fill pass overspent: {} of 3,000",
            picked.tokens
        );
    }

    #[test]
    fn a_question_whose_words_appear_nowhere_still_shows_the_document() {
        let chunks: Vec<Chunk> = (0..20)
            .map(|i| chunk(i, i + 1, &format!("hydraulic schedule {i} ").repeat(40)))
            .collect();
        let picked = select("zzzqqq unrelated", &[candidate(&chunks)], 900);
        assert!(
            !picked.chosen.is_empty(),
            "a document contributed nothing at all, so the model cannot even see it is the wrong one"
        );
    }

    #[test]
    fn a_pinned_document_is_served_before_the_others() {
        let big: Vec<Chunk> = (0..40)
            .map(|i| chunk(i, i + 1, &"filler ".repeat(200)))
            .collect();
        let mut pinned_chunks: Vec<Chunk> = (0..40)
            .map(|i| chunk(i, i + 1, &"filler ".repeat(200)))
            .collect();
        for piece in &mut pinned_chunks {
            piece.document_sha256 = OTHER.to_string();
        }
        let unpinned = Candidate {
            sha256: DOC,
            name: "ordinary.pdf",
            chunks: &big,
            pages: 40,
            pinned: false,
        };
        let pinned = Candidate {
            sha256: OTHER,
            name: "pinned.pdf",
            chunks: &pinned_chunks,
            pages: 40,
            pinned: true,
        };
        let picked = select("filler", &[unpinned, pinned], 1_200);
        let pinned_share: u32 = picked
            .chosen
            .iter()
            .filter(|c| c.document_name == "pinned.pdf")
            .map(|c| c.tokens)
            .sum();
        let other_share: u32 = picked
            .chosen
            .iter()
            .filter(|c| c.document_name == "ordinary.pdf")
            .map(|c| c.tokens)
            .sum();
        assert!(
            pinned_share >= other_share,
            "the pinned document got less of the turn than the unpinned one: \
             {pinned_share} vs {other_share}"
        );
    }
}
