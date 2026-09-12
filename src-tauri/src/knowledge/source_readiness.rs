//! Whether a notebook source can actually be read, and if not, why not.
//!
//! ## The bug this exists to make impossible
//!
//! A notebook source was "there" if `notebook_documents` had a row for it. That
//! row is a membership record and nothing more — it says somebody added a file,
//! not that the file's text is on disk, not that this notebook was ever told
//! about it, and not that the reader got anything off the page.
//!
//! Two paths added sources and only one of them wrote the record that makes the
//! text readable. [`crate::commands::notebook::notebook_add_documents`] records
//! a sighting under the notebook's own synthetic conversation id;
//! `notebook.add_source` — the tool a chat turn uses to file an attachment into
//! a notebook — wrote the membership row and no sighting at all. So the
//! notebook listed the document, the screen counted it as a source, retrieval
//! asked the document store for it under `notebook:{id}`, got nothing, and
//! reported "its extracted text is missing".
//!
//! That is one of at least four situations that all read as "missing" before
//! this module: never stored, stored and corrupt, stored but not associated
//! with this notebook, and stored with no text on the page. They have four
//! different repairs — one of which is a single click that re-reads nothing,
//! because the text was on disk the whole time — and a screen that cannot tell
//! them apart can only offer a shrug.
//!
//! ## Reported, never inferred
//!
//! Every field here comes from work that was actually done: `Completeness` is
//! written at extraction time from pages that were actually processed, and the
//! chunk counts are the chunks in the same file as the page text. Nothing here
//! estimates, and nothing rounds a partial read up to a whole one. A source
//! with two unreadable pages out of forty is [`SourceState::PartiallyReady`]
//! and says which two.

use serde::{Deserialize, Serialize};

use crate::agent_runtime::documents::{ExtractedDocument, SourcePresence};
use crate::knowledge::graph::NotebookDocument;

/// How far along one source is, from added to answerable.
///
/// The transient states at the top are reported by an ingestion job that is
/// running right now; the settled states below are worked out from what is on
/// disk by [`assess`]. They share one enum because the screen shows one badge
/// per source, and a person does not care which half of the system knows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum SourceState {
    /// Accepted and waiting for a reader. Nothing has been opened yet.
    Queued,
    /// The bytes are being read off disk and identified.
    Reading,
    /// A parser or the OCR model is working through it.
    Extracting,
    /// The page carries no text layer and reading it needs the local
    /// OCR/vision path. Whether that path is *available* is a separate fact —
    /// see [`ReadinessReason::VisionUnavailable`].
    NeedsVision,
    /// Text is out and the passages are being cut and stored.
    Indexing,
    /// Readable, complete, and usable as evidence.
    Ready,
    /// Readable, and not all of it. Some pages, rows or slides are missing and
    /// the reasons say which.
    PartiallyReady,
    /// Nothing can be read from it as it stands. Usually repairable.
    Failed,
    /// Reading it would need a capability this machine does not have.
    ///
    /// Distinct from [`Self::Failed`] because there is nothing to click: the
    /// answer is to install the parser or the model, not to retry.
    Unavailable,
}

impl SourceState {
    /// Whether a question may draw evidence from this source.
    ///
    /// [`Self::PartiallyReady`] is included — a partly-read document is real
    /// evidence for what it does contain — but every caller that includes one
    /// must carry its reasons through to the answer, which is what
    /// [`SourceReadiness::limitation`] is for.
    pub fn usable_as_evidence(self) -> bool {
        matches!(self, SourceState::Ready | SourceState::PartiallyReady)
    }

    /// Whether this source is still being worked on.
    ///
    /// The screen polls while this is true and stops when it is not, rather
    /// than polling for ever on a source that failed.
    pub fn in_progress(self) -> bool {
        matches!(
            self,
            SourceState::Queued
                | SourceState::Reading
                | SourceState::Extracting
                | SourceState::Indexing
        )
    }

    pub fn as_str(self) -> &'static str {
        match self {
            SourceState::Queued => "queued",
            SourceState::Reading => "reading",
            SourceState::Extracting => "extracting",
            SourceState::NeedsVision => "needsVision",
            SourceState::Indexing => "indexing",
            SourceState::Ready => "ready",
            SourceState::PartiallyReady => "partiallyReady",
            SourceState::Failed => "failed",
            SourceState::Unavailable => "unavailable",
        }
    }
}

/// Precisely what is wrong, in a form the screen can act on.
///
/// One variant per *distinct repair*. Two situations that call for the same
/// button are the same reason; two that call for different buttons are not, no
/// matter how similar they read.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", tag = "kind")]
pub enum ReadinessReason {
    /// The notebook has a membership row and the document store was never told
    /// this notebook may read the text. The text itself is on disk.
    ///
    /// The bug in the module docs, as a state a person can see and fix.
    NotAssociated,
    /// No extraction was ever written for this content address.
    MissingExtraction,
    /// The document store could not be read at all.
    ///
    /// Kept apart from [`Self::MissingExtraction`] because the two lead
    /// somewhere opposite: this one is retried, that one is re-uploaded, and
    /// telling somebody to supply a file again because a volume was briefly
    /// unavailable wastes their time and loses the real fault.
    StoreUnavailable { problem: String },
    /// An extraction exists and this build of ARJUN cannot parse it.
    IncompatibleExtraction { problem: String },
    /// The original bytes are gone, so it cannot be read again.
    MissingOriginal,
    /// The file is encrypted and no password was supplied.
    PasswordProtected,
    /// Reading this format needs a parser or converter that is not installed.
    MissingParser { needed: String },
    /// A legacy-format conversion was attempted and did not succeed.
    ConversionFailure { tool: String, problem: String },
    /// The reader opened it and got no text at all.
    NoTextExtracted,
    /// The content is images, and text needs the local OCR/vision path.
    RequiresVision,
    /// It needs vision and this machine has no vision model loaded.
    VisionUnavailable,
    /// Named pages produced nothing. `total` is how many there are in all, so
    /// "3 of 40" can be said without a second lookup.
    UnreadablePages { pages: Vec<u32>, total: u32 },
    /// The reader stopped before the end — a workbook past its row cap, a deck
    /// past its slide cap. No amount of re-reading pages fixes this one.
    PartialExtraction { read: u32, total: u32 },
    /// Text was extracted and the passages were not stored.
    NotIndexed { stored: u32, expected: u32 },
}

impl ReadinessReason {
    /// One sentence, meant for a person looking at the source list.
    pub fn describe(&self) -> String {
        match self {
            ReadinessReason::NotAssociated => {
                "Its text is on this machine, but this notebook was never given access to it. \
                 Repairing takes a moment and re-reads nothing."
                    .to_string()
            }
            ReadinessReason::MissingExtraction => {
                "Nothing was ever extracted from this file. Add it again to read it.".to_string()
            }
            ReadinessReason::StoreUnavailable { problem } => format!(
                "This machine's document store could not be read, so whether this source \
                 has text is not known: {problem}"
            ),
            ReadinessReason::IncompatibleExtraction { problem } => {
                format!("Its stored text cannot be read by this version of ARJUN: {problem}")
            }
            ReadinessReason::MissingOriginal => {
                "The original file is no longer on this machine, so it cannot be read again. \
                 Add it again from wherever it is kept."
                    .to_string()
            }
            ReadinessReason::PasswordProtected => {
                "The file is password-protected, so nothing could be opened.".to_string()
            }
            ReadinessReason::MissingParser { needed } => {
                format!(
                    "Reading this format needs {needed}, which is not installed on this machine."
                )
            }
            ReadinessReason::ConversionFailure { tool, problem } => {
                format!("{tool} could not convert this file: {problem}")
            }
            ReadinessReason::NoTextExtracted => {
                "The reader opened it and found no text.".to_string()
            }
            ReadinessReason::RequiresVision => {
                "Its content is images, so reading it needs the local OCR/vision model."
                    .to_string()
            }
            ReadinessReason::VisionUnavailable => {
                "Its content is images and no vision model is loaded on this machine, so there \
                 is no local way to read it."
                    .to_string()
            }
            ReadinessReason::UnreadablePages { pages, total } => {
                let named: Vec<String> = pages.iter().take(8).map(u32::to_string).collect();
                let rest = pages.len().saturating_sub(named.len());
                let list = if rest > 0 {
                    format!("{} and {rest} more", named.join(", "))
                } else {
                    named.join(", ")
                };
                format!(
                    "{} of {total} pages produced no text (page {list}).",
                    pages.len()
                )
            }
            ReadinessReason::PartialExtraction { read, total } => format!(
                "The reader stopped after {read} of {total}, so the rest of this file is not \
                 indexed at all."
            ),
            ReadinessReason::NotIndexed { stored, expected } => format!(
                "{stored} of {expected} passages were stored, so part of this source cannot be \
                 retrieved."
            ),
        }
    }

    /// What a person can do about this from the source list.
    ///
    /// Drives whether a repair control is drawn. A reason that is not
    /// repairable gets no button, because a button that cannot work is worse
    /// than no button.
    pub fn repair(&self) -> Option<Repair> {
        match self {
            ReadinessReason::NotAssociated => Some(Repair::Associate),
            ReadinessReason::IncompatibleExtraction { .. }
            | ReadinessReason::NoTextExtracted
            | ReadinessReason::UnreadablePages { .. }
            | ReadinessReason::NotIndexed { .. }
            | ReadinessReason::RequiresVision
            | ReadinessReason::ConversionFailure { .. } => Some(Repair::Reprocess),
            ReadinessReason::MissingExtraction
            | ReadinessReason::MissingOriginal
            | ReadinessReason::PartialExtraction { .. } => Some(Repair::AddAgain),
            // Retried, not re-uploaded. Nothing is known to be wrong with the
            // file itself.
            ReadinessReason::StoreUnavailable { .. } => Some(Repair::Retry),
            ReadinessReason::PasswordProtected
            | ReadinessReason::MissingParser { .. }
            | ReadinessReason::VisionUnavailable => None,
        }
    }
}

/// What the screen may offer for a reason.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Repair {
    /// One click, no re-reading: record the sighting this notebook is missing.
    Associate,
    /// Read the retained original again, at the best available settings.
    Reprocess,
    /// The bytes are gone; the person has to supply the file again.
    AddAgain,
    /// Nothing is known to be wrong with the file — ask the store again.
    Retry,
}

/// One source, and everything known about whether it can be read.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SourceReadiness {
    pub document_sha256: String,
    pub document_name: String,
    pub state: SourceState,
    /// Every distinct thing wrong with it. Empty for [`SourceState::Ready`].
    pub reasons: Vec<ReadinessReason>,
    /// Which local path produced the text: `pdf-text`, `pdf-scan`, `xlsx`…
    /// `None` when nothing has been extracted.
    pub extraction_kind: Option<String>,
    /// The extraction revision, so a citation can name what it read.
    pub source_revision: Option<String>,
    pub pages_total: u32,
    pub pages_with_text: u32,
    pub passages: u32,
    /// The repair the screen should offer, if any — the first one suggested by
    /// any of the reasons.
    pub repair: Option<Repair>,
}

impl SourceReadiness {
    /// The sentence an answer carries when this source was in scope.
    ///
    /// `None` for a source that was read completely: an answer must not be
    /// padded with a limitation that says nothing was wrong.
    pub fn limitation(&self) -> Option<String> {
        if self.state == SourceState::Ready {
            return None;
        }
        let reasons: Vec<String> = self.reasons.iter().map(ReadinessReason::describe).collect();
        if reasons.is_empty() {
            return Some(format!(
                "{}: {} — no further detail was recorded.",
                self.document_name,
                self.state.as_str()
            ));
        }
        Some(format!("{}: {}", self.document_name, reasons.join(" ")))
    }
}

/// What this machine can currently do, for deciding between "needs vision" and
/// "cannot be read here".
///
/// Passed in rather than looked up, so the decision is testable without a model
/// registry and so no path can answer it by assuming.
#[derive(Debug, Clone, Copy, Default)]
pub struct LocalCapabilities {
    /// A vision or document-OCR model is loaded and could read an image page.
    pub vision: bool,
}

/// Extraction kinds whose content is pictures of text rather than text.
fn is_image_only(kind: &str) -> bool {
    matches!(kind, "image" | "pdf-scan" | "drawing" | "photo")
}

/// The extraction revision a passage from this document would be cited at.
///
/// Extraction time plus page count: the same document re-read at a better OCR
/// stop produces a different revision, which is what stops an old citation
/// resolving against new words.
pub fn revision_of(document: &ExtractedDocument) -> String {
    format!("{}#{}", document.extracted_at, document.page_text.len())
}

/// Works out one source's readiness from what is actually on disk.
///
/// Takes a [`SourcePresence`] rather than the store on purpose: authorisation
/// happened when the presence was produced, and a function that could re-read
/// the store could re-read it with the wrong owner.
pub fn assess(
    member: &NotebookDocument,
    presence: &SourcePresence,
    capabilities: LocalCapabilities,
) -> SourceReadiness {
    let blank = |state: SourceState, reasons: Vec<ReadinessReason>| {
        let repair = reasons.iter().find_map(ReadinessReason::repair);
        SourceReadiness {
            document_sha256: member.document_sha256.clone(),
            document_name: member.document_name.clone(),
            state,
            reasons,
            extraction_kind: None,
            source_revision: None,
            pages_total: 0,
            pages_with_text: 0,
            passages: 0,
            repair,
        }
    };

    let document = match presence {
        SourcePresence::Absent => {
            return blank(SourceState::Failed, vec![ReadinessReason::MissingExtraction])
        }
        SourcePresence::Unavailable { problem } => {
            // `Unavailable`, not `Failed`: nothing has been established about
            // this source, so the screen must not offer a repair that assumes
            // the file is at fault.
            return blank(
                SourceState::Unavailable,
                vec![ReadinessReason::StoreUnavailable {
                    problem: problem.clone(),
                }],
            )
        }
        SourcePresence::Unreadable { problem } => {
            return blank(
                SourceState::Failed,
                vec![ReadinessReason::IncompatibleExtraction {
                    problem: problem.clone(),
                }],
            )
        }
        SourcePresence::NotAssociated(document) => {
            // The text exists and this notebook may not read it yet. Reported
            // with the counts from the document sitting there, so the screen
            // can say what repairing it will recover.
            let mut readiness = blank(SourceState::Failed, vec![ReadinessReason::NotAssociated]);
            readiness.extraction_kind = Some(document.kind.clone());
            readiness.pages_total = document.completeness.pages_total.max(document.pages);
            readiness.pages_with_text = document.completeness.pages_extracted;
            readiness.passages = document.chunks.len() as u32;
            return readiness;
        }
        SourcePresence::Present(document) => document,
    };

    let completeness = &document.completeness;
    let pages_total = completeness.pages_total.max(document.pages);
    let pages_with_text = completeness.pages_extracted;
    let mut reasons: Vec<ReadinessReason> = Vec::new();

    let settled = |state: SourceState, reasons: Vec<ReadinessReason>, passages: u32| {
        let repair = reasons.iter().find_map(ReadinessReason::repair);
        SourceReadiness {
            document_sha256: member.document_sha256.clone(),
            document_name: member.document_name.clone(),
            state,
            reasons,
            extraction_kind: Some(document.kind.clone()),
            source_revision: Some(revision_of(document)),
            pages_total,
            pages_with_text,
            passages,
            repair,
        }
    };

    // Nothing readable came out at all.
    if document.chunks.is_empty() {
        if document.page_text.is_empty() {
            let state = if !is_image_only(&document.kind) {
                reasons.push(ReadinessReason::NoTextExtracted);
                SourceState::Failed
            } else if capabilities.vision {
                reasons.push(ReadinessReason::RequiresVision);
                SourceState::NeedsVision
            } else {
                reasons.push(ReadinessReason::RequiresVision);
                reasons.push(ReadinessReason::VisionUnavailable);
                SourceState::Unavailable
            };
            return settled(state, reasons, 0);
        }
        // Text on the pages and no passages: the cut or the store failed. The
        // text is there, so reprocessing can recover it.
        reasons.push(ReadinessReason::NotIndexed {
            stored: 0,
            expected: completeness.chunks_total.max(1),
        });
        return settled(SourceState::Failed, reasons, 0);
    }

    // Readable. Now: all of it, or some of it?
    if !completeness.pages_failed.is_empty() {
        reasons.push(ReadinessReason::UnreadablePages {
            pages: completeness.pages_failed.clone(),
            total: pages_total,
        });
        if is_image_only(&document.kind) && !capabilities.vision {
            reasons.push(ReadinessReason::VisionUnavailable);
        }
    }
    if completeness.source_truncated || document.truncated {
        reasons.push(ReadinessReason::PartialExtraction {
            read: pages_with_text,
            total: pages_total.max(pages_with_text),
        });
    }
    if completeness.chunks_stored < completeness.chunks_total {
        reasons.push(ReadinessReason::NotIndexed {
            stored: completeness.chunks_stored,
            expected: completeness.chunks_total,
        });
    }

    let state = if reasons.is_empty() {
        SourceState::Ready
    } else {
        SourceState::PartiallyReady
    };
    settled(state, reasons, document.chunks.len() as u32)
}

/// The counts a screen needs to stop contradicting itself.
///
/// Three numbers, not two. The old preview reported "selected" and "total" and
/// listed unreadable sources separately, which is how "1 of 1 will be read"
/// came to sit directly above a warning that the one source could not be read.
/// Whether a source is *selected* and whether it can be *read* are independent
/// facts, and a screen carrying only one of them will eventually say both
/// things at once.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReadinessCounts {
    /// Sources in the notebook.
    pub total: u32,
    /// Sources this question is allowed to use.
    pub selected: u32,
    /// Selected sources with usable text — the only number that may be said
    /// out loud as "will be read".
    pub readable: u32,
    /// Selected sources that are readable and incomplete.
    pub partial: u32,
    /// Selected sources that cannot be read at all.
    pub blocked: u32,
    /// Selected sources still being worked on.
    pub in_progress: u32,
}

impl ReadinessCounts {
    pub fn tally(total: u32, selected: &[SourceReadiness]) -> Self {
        let mut counts = ReadinessCounts {
            total,
            selected: selected.len() as u32,
            ..Default::default()
        };
        for source in selected {
            match source.state {
                SourceState::Ready => counts.readable += 1,
                SourceState::PartiallyReady => {
                    counts.readable += 1;
                    counts.partial += 1;
                }
                state if state.in_progress() => counts.in_progress += 1,
                _ => counts.blocked += 1,
            }
        }
        counts
    }

    /// Whether asking a question now would have nothing to answer from.
    pub fn nothing_readable(&self) -> bool {
        self.readable == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent_runtime::documents::{DocumentStore, NewExtraction, Sighting};
    use std::collections::BTreeMap;

    const SHA: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const NOTEBOOK_CONVERSATION: &str = "notebook:nb-1";

    fn member(name: &str) -> NotebookDocument {
        NotebookDocument {
            notebook_id: "nb-1".into(),
            document_sha256: SHA.into(),
            document_name: name.into(),
            added_at: "2026-09-12T05:42:11Z".into(),
        }
    }

    fn sighting(conversation: &str) -> Sighting {
        Sighting {
            owner_user_id: "priya".into(),
            conversation_id: conversation.into(),
            message_id: "a-1".into(),
            run_id: "r-1".into(),
            at: "2026-09-12T05:42:11Z".into(),
            ocr_model_id: None,
            ocr_detent: None,
        }
    }

    /// A store with one document recorded against `conversation`.
    ///
    /// Driven through the real `record` so the chunks, the completeness counts
    /// and the sightings are the ones production writes — a hand-built
    /// `ExtractedDocument` would have no chunks and would test nothing.
    fn store_with(
        kind: &str,
        pages: &[(u32, &str)],
        conversation: &str,
    ) -> (tempfile::TempDir, DocumentStore) {
        let dir = tempfile::tempdir().expect("a temporary directory");
        let store = DocumentStore::open(dir.path()).expect("the store opens");
        let page_text: BTreeMap<u32, String> = pages
            .iter()
            .map(|(page, text)| (*page, (*text).to_string()))
            .collect();
        store
            .record(NewExtraction {
                sha256: SHA.into(),
                name: "file".into(),
                kind: kind.into(),
                pages: pages.len() as u32,
                truncated: false,
                page_text,
                sighting: sighting(conversation),
            })
            .expect("the extraction is recorded");
        (dir, store)
    }

    fn presence(store: &DocumentStore) -> SourcePresence {
        store
            .inspect(SHA, "priya", Some(NOTEBOOK_CONVERSATION))
            .expect("the store can be inspected")
    }

    fn caps(vision: bool) -> LocalCapabilities {
        LocalCapabilities { vision }
    }

    /// The bug this module exists for: a membership row with no sighting is not
    /// "missing", it is unassociated, and one click fixes it.
    #[test]
    fn an_unassociated_source_is_named_as_such_and_offers_a_one_click_repair() {
        // Recorded against a chat conversation, never against the notebook —
        // exactly what `notebook.add_source` left behind.
        let (_dir, store) = store_with("pdf-text", &[(1, "the pump is rated at 12 bar")], "c-1");
        let readiness = assess(&member("manual.pdf"), &presence(&store), caps(true));

        assert_eq!(readiness.state, SourceState::Failed);
        assert_eq!(readiness.reasons, vec![ReadinessReason::NotAssociated]);
        assert_eq!(readiness.repair, Some(Repair::Associate));
        assert!(
            !readiness.state.usable_as_evidence(),
            "an unassociated source must never be answered from"
        );
        assert!(
            readiness.passages > 0,
            "the screen can say what repairing it recovers"
        );
    }

    /// And the repair actually works, without re-reading anything.
    #[test]
    fn associating_the_notebook_makes_the_same_source_ready() {
        let (_dir, store) = store_with("pdf-text", &[(1, "the pump is rated at 12 bar")], "c-1");
        assert_eq!(
            assess(&member("manual.pdf"), &presence(&store), caps(true)).state,
            SourceState::Failed
        );

        store
            .associate(SHA, "priya", sighting(NOTEBOOK_CONVERSATION))
            .expect("the notebook may be associated");

        let readiness = assess(&member("manual.pdf"), &presence(&store), caps(true));
        assert_eq!(readiness.state, SourceState::Ready);
        assert!(readiness.reasons.is_empty(), "{:?}", readiness.reasons);
    }

    /// A sighting is an authorisation record, so it cannot be conjured for a
    /// document this person never attached.
    #[test]
    fn another_persons_document_cannot_be_associated_into_a_notebook() {
        let (_dir, store) = store_with("pdf-text", &[(1, "confidential")], "c-1");
        let refusal = store.associate(SHA, "mallory", sighting(NOTEBOOK_CONVERSATION));
        assert!(refusal.is_err(), "a stranger must not gain access");

        // And it does not even read as present for them.
        let unseen = store
            .inspect(SHA, "mallory", Some(NOTEBOOK_CONVERSATION))
            .expect("inspect answers");
        assert!(matches!(unseen, SourcePresence::Absent));
    }

    /// Missing, corrupt and unassociated must not collapse into one message.
    #[test]
    fn the_three_ways_of_having_no_text_are_three_different_answers() {
        let dir = tempfile::tempdir().expect("a temporary directory");
        let store = DocumentStore::open(dir.path()).expect("the store opens");
        let missing = assess(
            &member("a.pdf"),
            &store
                .inspect(SHA, "priya", Some(NOTEBOOK_CONVERSATION))
                .expect("inspect answers"),
            caps(true),
        );

        // A file this build cannot parse, written where the store keeps them.
        let corrupt_dir = tempfile::tempdir().expect("a temporary directory");
        let corrupt_store = DocumentStore::open(corrupt_dir.path()).expect("the store opens");
        std::fs::write(
            corrupt_dir
                .path()
                .join("documents")
                .join("extractions")
                .join(format!("{SHA}.json")),
            b"{ this is not the envelope }",
        )
        .expect("the corrupt file is written");
        let corrupt = assess(
            &member("b.pdf"),
            &corrupt_store
                .inspect(SHA, "priya", Some(NOTEBOOK_CONVERSATION))
                .expect("inspect answers"),
            caps(true),
        );

        let (_unassociated_dir, unassociated_store) = store_with("pdf-text", &[(1, "text")], "c-1");
        let unassociated = assess(&member("c.pdf"), &presence(&unassociated_store), caps(true));

        assert_eq!(missing.reasons[0], ReadinessReason::MissingExtraction);
        assert!(
            matches!(corrupt.reasons[0], ReadinessReason::IncompatibleExtraction { .. }),
            "{:?}",
            corrupt.reasons
        );
        assert_eq!(unassociated.reasons[0], ReadinessReason::NotAssociated);

        assert_eq!(missing.repair, Some(Repair::AddAgain));
        assert_eq!(corrupt.repair, Some(Repair::Reprocess));
        assert_eq!(unassociated.repair, Some(Repair::Associate));
    }

    #[test]
    fn a_complete_read_is_ready_and_carries_no_limitation() {
        let (_dir, store) = store_with(
            "pdf-text",
            &[(1, "the pump is rated at 12 bar"), (2, "and it runs cold")],
            NOTEBOOK_CONVERSATION,
        );
        let readiness = assess(&member("manual.pdf"), &presence(&store), caps(true));

        assert_eq!(readiness.state, SourceState::Ready);
        assert!(readiness.reasons.is_empty(), "{:?}", readiness.reasons);
        assert_eq!(readiness.limitation(), None);
        assert!(readiness.passages > 0);
        assert!(readiness.source_revision.is_some());
    }

    /// Image-only content with no vision model is not "empty".
    #[test]
    fn an_image_only_source_is_not_reported_as_empty() {
        let (_dir, store) = store_with("image", &[], NOTEBOOK_CONVERSATION);

        let with_vision = assess(&member("scan.png"), &presence(&store), caps(true));
        assert_eq!(with_vision.state, SourceState::NeedsVision);
        assert!(with_vision.reasons.contains(&ReadinessReason::RequiresVision));

        let without = assess(&member("scan.png"), &presence(&store), caps(false));
        assert_eq!(without.state, SourceState::Unavailable);
        assert!(without.reasons.contains(&ReadinessReason::VisionUnavailable));
    }

    #[test]
    fn a_text_document_with_no_text_is_failed_rather_than_needing_vision() {
        let (_dir, store) = store_with("docx", &[], NOTEBOOK_CONVERSATION);
        let readiness = assess(&member("empty.docx"), &presence(&store), caps(true));

        assert_eq!(readiness.state, SourceState::Failed);
        assert_eq!(readiness.reasons, vec![ReadinessReason::NoTextExtracted]);
    }

    /// The counter that stops "1 of 1 will be read" appearing above a warning
    /// that the one source cannot be read.
    #[test]
    fn selected_and_readable_are_counted_separately() {
        let (_dir, store) = store_with("pdf-text", &[(1, "text")], NOTEBOOK_CONVERSATION);
        let readable = assess(&member("good.pdf"), &presence(&store), caps(true));

        let empty_dir = tempfile::tempdir().expect("a temporary directory");
        let empty = DocumentStore::open(empty_dir.path()).expect("the store opens");
        let broken = assess(
            &member("bad.pdf"),
            &empty
                .inspect(SHA, "priya", Some(NOTEBOOK_CONVERSATION))
                .expect("inspect answers"),
            caps(true),
        );

        let counts = ReadinessCounts::tally(5, &[readable, broken.clone()]);
        assert_eq!(counts.total, 5);
        assert_eq!(counts.selected, 2);
        assert_eq!(counts.readable, 1);
        assert_eq!(counts.blocked, 1);
        assert!(!counts.nothing_readable());

        let none = ReadinessCounts::tally(5, &[broken]);
        assert!(none.nothing_readable());
        assert_eq!(none.readable, 0);
    }


    /// An I/O failure is not a missing file.
    ///
    /// The regression for `inspect(..).unwrap_or(Absent)`: a permission error
    /// became "Nothing was ever extracted from this file. Add it again", which
    /// sends somebody to re-upload a document whose text is on disk and fine.
    #[test]
    fn a_store_that_could_not_be_read_is_not_reported_as_a_missing_extraction() {
        let unavailable = assess(
            &member("manual.pdf"),
            &SourcePresence::Unavailable {
                problem: "Access is denied. (os error 5)".into(),
            },
            caps(true),
        );
        let missing = assess(&member("manual.pdf"), &SourcePresence::Absent, caps(true));

        assert_eq!(unavailable.state, SourceState::Unavailable);
        assert_eq!(missing.state, SourceState::Failed);
        assert_ne!(unavailable.reasons, missing.reasons);

        assert_eq!(unavailable.repair, Some(Repair::Retry));
        assert_eq!(
            missing.repair,
            Some(Repair::AddAgain),
            "only a genuinely absent extraction asks for the file again"
        );

        let said = unavailable.limitation().expect("it says something");
        assert!(said.contains("os error 5"), "the real fault is carried: {said}");
        assert!(
            !said.contains("Add it again"),
            "and the wrong advice is not: {said}"
        );
    }

    #[test]
    fn no_unsettled_or_failed_state_is_usable_as_evidence() {
        for state in [
            SourceState::Queued,
            SourceState::Reading,
            SourceState::Extracting,
            SourceState::Indexing,
        ] {
            assert!(state.in_progress(), "{state:?}");
            assert!(!state.usable_as_evidence(), "{state:?}");
        }
        for state in [
            SourceState::Failed,
            SourceState::Unavailable,
            SourceState::NeedsVision,
        ] {
            assert!(!state.in_progress(), "{state:?}");
            assert!(!state.usable_as_evidence(), "{state:?}");
        }
        assert!(SourceState::Ready.usable_as_evidence());
        assert!(SourceState::PartiallyReady.usable_as_evidence());
    }
}
