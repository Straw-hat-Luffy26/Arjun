//! Producing an artifact from a content model: validate, repair, render,
//! re-open, regenerate — and accept nothing that did not pass.
//!
//! ## Why this exists beside `production`
//!
//! [`super::production::produce`] drives the *template* path: it asks a
//! [`super::production::ContentSource`] for a map of fields, renders the one
//! Word template, and hands the renderer's objections back. It knows nothing
//! about the typed models, and there is no equivalent for a workbook, a deck or
//! a PDF — those three rendered once, and if the file came out wrong nobody
//! found out.
//!
//! This is that loop for the models. The difference in kind is that a model can
//! be *repaired in memory*, which a map of strings could not be: a slide with
//! twelve bullets is not broken content, it is content that needs two slides,
//! and [`super::doc_model::Deck::repair`] can say so without asking anybody.
//!
//! ## The order, and why it is that order
//!
//! ```text
//! validate the model
//!   |- sound      -> render
//!   `- not sound  -> repair -> validate again
//!                      |- sound     -> render
//!                      `- not sound -> fail, naming what is still wrong
//! render to revision N
//!   `- re-open with the format's own validator
//!        |- sound     -> accept
//!        `- not sound -> regenerate (bounded), then fail if it still will not
//! ```
//!
//! Repair comes before rendering because repairing a *file* is not a thing this
//! can do — a `.docx` with a ragged table has to be written again either way,
//! and the only question is whether the model was corrected first.
//!
//! Regeneration exists for a different failure: the model validated, the writer
//! ran, and the file is wrong. That is a writer or a disk fault rather than a
//! content fault, so retrying is honest but bounded — a second identical
//! failure is not going to become a third success.
//!
//! ## What is never done
//!
//! A failed artifact is never returned as a success, and a file that did not
//! pass its check is never the artifact that stands. Both are the failure this
//! module exists to prevent.

use std::path::{Path, PathBuf};

use super::doc_model::{Deck, Document, Workbook};
use super::docx::DocumentMetadata;
use super::production::Revision;

/// How many times a *file* that came out wrong is written again.
///
/// Two. The model was validated and repaired before the first attempt, so a
/// file that fails its check is a fault in the writing rather than in the
/// content, and a third attempt at the same bytes is not a plan.
pub const MAX_RENDER_ATTEMPTS: usize = 2;

/// What producing an artifact from a model came to.
#[derive(Debug, Clone)]
pub struct ModelOutcome {
    /// The revision that stands. `None` when nothing was accepted.
    pub artifact: Option<PathBuf>,
    /// Every attempt, in order, including the ones that were superseded.
    pub revisions: Vec<Revision>,
    /// What the in-memory repair changed, in words a reader can check.
    pub repairs: Vec<String>,
    /// Set when nothing was produced, in words a person can act on.
    pub failure: Option<String>,
}

impl ModelOutcome {
    pub fn succeeded(&self) -> bool {
        self.artifact.is_some()
    }

    fn failed(failure: String, revisions: Vec<Revision>, repairs: Vec<String>) -> Self {
        Self { artifact: None, revisions, repairs, failure: Some(failure) }
    }
}

/// `report.docx` at revision 2 becomes `report.r2.docx`.
///
/// The same scheme `production::revision_path` uses, and for the same reason: a
/// directory of artifacts should be self-describing to somebody who found it
/// without this application.
fn revision_path(base: &Path, revision: usize) -> PathBuf {
    let stem = base.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
    let name = match base.extension().map(|e| e.to_string_lossy().to_string()) {
        Some(ext) => format!("{stem}.r{revision}.{ext}"),
        None => format!("{stem}.r{revision}"),
    };
    base.with_file_name(name)
}

/// The shared loop. `validate` reports what is wrong with the model, `repair`
/// fixes what it can, `render` writes one file and `check` re-opens it.
/// `check` is given the model as well as the path.
///
/// Not a convenience: the file has to be checked against the model that was
/// *rendered*, which is not the model that was passed in. Repair runs first and
/// can change the shape — splitting an overflowing slide is exactly that — so a
/// count captured before repair describes a deck that was never written.
fn produce<M>(
    base: &Path,
    model: &mut M,
    validate: impl Fn(&M) -> Vec<String>,
    repair: impl Fn(&mut M) -> Vec<String>,
    render: impl Fn(&Path, &M) -> Result<(), String>,
    check: impl Fn(&Path, &M) -> Vec<String>,
    what: &str,
    // How much content there is — sections, rows, slides. Sets the total-time
    // ceiling; see `orchestrator::progress::budget_for`.
    units: u64,
) -> ModelOutcome {
    use crate::orchestrator::progress::{Advance, Phase, Tracker};

    let mut revisions = Vec::new();
    let mut repairs = Vec::new();

    // Phase-aware, so a long render is told apart from a stuck one.
    //
    // `units` is the size of the content, which is known here and nowhere
    // later: it sets a total-time ceiling proportional to the work rather than
    // the fixed one every artifact tool used to share. See
    // `orchestrator::progress` for why a stopwatch was the wrong instrument.
    let mut tracker = Tracker::new(Phase::Validating, units);

    // 1. The model, before anything is written.
    if !validate(model).is_empty() {
        tracker.enter(Phase::Repairing);
        repairs = repair(model);
        tracker.advance(Advance::Checks(repairs.len() as u64));
        let still = validate(model);
        if !still.is_empty() {
            // Repair could not close the gap, and inventing the rest is the one
            // thing this must never do.
            return ModelOutcome::failed(
                format!(
                    "The {what} could not be produced: {}. Nothing was written.",
                    still.join("; ")
                ),
                revisions,
                repairs,
            );
        }
    }

    // 2. Write it, then read it back.
    for attempt in 1..=MAX_RENDER_ATTEMPTS {
        let path = revision_path(base, attempt);

        // The second attempt is a regeneration, not a first render, and its
        // silence means the same thing — so it gets the same window under its
        // own name, which is what the log will say.
        tracker.enter(if attempt == 1 { Phase::Rendering } else { Phase::Regenerating });

        if let Err(why) = render(&path, model) {
            revisions.push(Revision { number: attempt, path: None, superseded_because: Some(why) });
            continue;
        }
        tracker.advance(Advance::Bytes(
            std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0),
        ));

        tracker.enter(Phase::Validating);
        let problems = check(&path, model);
        tracker.advance(Advance::Checks(1));
        log::debug!("[artifact] {what}: {}", tracker.heartbeat());
        if problems.is_empty() {
            revisions.push(Revision {
                number: attempt,
                path: Some(path.clone()),
                superseded_because: None,
            });
            return ModelOutcome { artifact: Some(path), revisions, repairs, failure: None };
        }

        // The file is wrong. It stays on disk under its revision number — "what
        // did it get wrong" is a question a reviewer will ask — but it is never
        // the artifact that stands.
        revisions.push(Revision {
            number: attempt,
            path: Some(path),
            superseded_because: Some(problems.join("; ")),
        });
    }

    let why = revisions
        .last()
        .and_then(|r| r.superseded_because.clone())
        .unwrap_or_else(|| "the file did not pass its own check".to_string());
    ModelOutcome::failed(
        format!(
            "The {what} was written {MAX_RENDER_ATTEMPTS} times and did not pass its own check \
             either time: {why}. Nothing is being presented as finished."
        ),
        revisions,
        repairs,
    )
}

/// A Word document, from a composed model.
pub fn produce_document(
    base: &Path,
    document: &mut Document,
    metadata: &DocumentMetadata,
) -> ModelOutcome {
    let metadata = metadata.clone();
    let document_units = document.sections.len() as u64;
    produce(
        base,
        document,
        Document::problems,
        Document::repair,
        move |path, model| {
            super::docx::write_document_model(path, model, &metadata).map_err(|e| e.message)
        },
        |path, _model| {
            // Re-opened as a package, and asked for the part that carries the
            // body. A writer that produced a ZIP without it is a writer whose
            // output opens as a corrupt file.
            match super::ooxml::read_part(path, "word/document.xml") {
                Ok(body) if body.contains("<w:body>") => Vec::new(),
                Ok(_) => vec!["the document body is missing from the package".to_string()],
                Err(error) => vec![format!("the package does not open: {error}")],
            }
        },
        "document",
        // Sections, which is what rendering a document is linear in.
        document_units,
    )
}

/// A workbook, from a composed model.
pub fn produce_workbook(base: &Path, workbook: &mut Workbook) -> ModelOutcome {
    let workbook_units =
        workbook.sheets.iter().map(|sheet| sheet.rows.len() as u64).sum::<u64>();
    produce(
        base,
        workbook,
        Workbook::problems,
        Workbook::repair,
        |path, model| super::xlsx::write_workbook_model(path, model),
        |path, _model| match super::ooxml::read_part(path, "xl/workbook.xml") {
            Ok(book) if book.contains("<sheet ") => Vec::new(),
            Ok(_) => vec!["the workbook declares no sheets".to_string()],
            Err(error) => vec![format!("the package does not open: {error}")],
        },
        "workbook",
        workbook_units,
    )
}

/// A deck, from a composed model.
pub fn produce_deck(base: &Path, deck: &mut Deck, is_draft: bool) -> ModelOutcome {
    let deck_units = deck.slides.len() as u64;
    // Checked against what this deck claims, not against the briefing
    // template.
    //
    // `pptx::check_deck` asks whether the four `BRIEFING_SECTIONS` are present,
    // which is the right question for `write_deck` and the wrong one here: a
    // composed deck has whatever narrative the model wrote, and demanding
    // "Findings / Recommendation / Assumptions / Evidence" of a shutdown
    // briefing would reject every deck this phase exists to make possible.
    //
    // What is checked instead is what can be checked honestly: the package
    // opens, it holds the slides the model asked for, and every one of them has
    // a heading on it.
    produce(
        base,
        deck,
        Deck::problems,
        Deck::repair,
        move |path, model| {
            super::pptx::write_deck_model(path, model, is_draft).map_err(|e| e.message)
        },
        |path, model: &Deck| {
            let check = super::pptx::check_deck(path);
            if !check.opens {
                return vec!["the presentation does not open".to_string()];
            }
            // The title slide is built by the writer, not supplied.
            let expected = model.slides.len() + 1;
            let mut problems = Vec::new();
            if check.slides != expected {
                problems.push(format!(
                    "the deck holds {} slide(s) and the model asked for {expected}",
                    check.slides
                ));
            }
            if check.headings.iter().any(|heading| heading.trim().is_empty()) {
                problems.push("a slide in the file has no heading".to_string());
            }
            problems
        },
        "deck",
        deck_units,
    )
}

/// A PDF, from the same document model a `.docx` is made from.
pub fn produce_pdf(base: &Path, document: &mut Document) -> ModelOutcome {
    let pdf_units = document.sections.len() as u64;
    produce(
        base,
        document,
        Document::problems,
        Document::repair,
        |path, model| {
            let spec = super::pdf::spec_from_document(model);
            let bytes = super::pdf::render(&spec)?;
            std::fs::write(path, bytes).map_err(|e| format!("the file could not be written: {e}"))
        },
        |path, _model| {
            // The real reader: xref, trailer, page tree, content streams, text.
            // Quality is part of acceptance, not an advisory beside it.
            let check = super::pdf_validate::check_pdf(path);
            let mut problems = check.problems.clone();
            if problems.is_empty() {
                problems.extend(super::pdf_validate::quality::inspect(&check));
            }
            problems
        },
        "PDF",
        pdf_units,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::artifacts::doc_model::{
        Block, Column, ColumnType, Properties, Section, Sheet, SlideModel,
    };

    fn metadata() -> DocumentMetadata {
        DocumentMetadata {
            task_id: "run-f".to_string(),
            created_at: "2026-03-04T09:00:00Z".to_string(),
            model: "Nemotron3-Nano-4B".to_string(),
            classification: "OFFICIAL".to_string(),
            is_draft: false,
        }
    }

    fn document() -> Document {
        Document {
            title: "Shell thickness inspection".to_string(),
            classification: "OFFICIAL".to_string(),
            properties: Properties::default(),
            sections: vec![Section {
                heading: "Findings".to_string(),
                level: 1,
                blocks: vec![Block::Paragraph {
                    text: "Sixteen points were measured and one was below the stated minimum."
                        .to_string(),
                }],
            }],
        }
    }

    #[test]
    fn a_sound_document_is_accepted_at_its_first_revision() {
        let dir = tempfile::tempdir().expect("temp dir");
        let mut model = document();
        let outcome = produce_document(&dir.path().join("report.docx"), &mut model, &metadata());

        assert!(outcome.succeeded(), "{:?}", outcome.failure);
        assert_eq!(
            outcome.artifact.as_ref().and_then(|p| p.file_name()).unwrap(),
            "report.r1.docx"
        );
        assert!(outcome.repairs.is_empty(), "nothing needed repairing: {:?}", outcome.repairs);
    }

    /// Repair, not regeneration: the outline was wrong and could be corrected
    /// without asking anybody for anything.
    #[test]
    fn a_broken_outline_is_repaired_and_the_document_is_produced() {
        let dir = tempfile::tempdir().expect("temp dir");
        let mut model = document();
        model.sections.push(Section {
            heading: "Detail".to_string(),
            level: 4, // skips 2 and 3
            blocks: vec![Block::Paragraph {
                text: "Measured with a calibrated ultrasonic probe on the day.".to_string(),
            }],
        });

        let outcome = produce_document(&dir.path().join("report.docx"), &mut model, &metadata());
        assert!(outcome.succeeded(), "{:?}", outcome.failure);
        assert!(
            outcome.repairs.iter().any(|r| r.contains("does not skip")),
            "the repair must be reported: {:?}",
            outcome.repairs
        );
        assert_eq!(model.sections[1].level, 2, "the model itself was corrected");
    }

    /// The line repair must not cross. A section with no content cannot be
    /// filled in by a machine, so nothing is produced and the failure says why.
    #[test]
    fn content_that_cannot_be_repaired_produces_nothing_and_says_so() {
        let dir = tempfile::tempdir().expect("temp dir");
        let mut model = document();
        model.sections[0].blocks = vec![Block::Paragraph { text: "   ".to_string() }];

        let outcome = produce_document(&dir.path().join("report.docx"), &mut model, &metadata());

        assert!(!outcome.succeeded());
        let failure = outcome.failure.expect("a reason");
        assert!(failure.contains("nothing under it") || failure.contains("could not"), "{failure}");
        assert!(
            std::fs::read_dir(dir.path()).expect("read").next().is_none(),
            "nothing may be written when the model does not validate"
        );
    }

    #[test]
    fn a_workbook_with_an_illegal_sheet_name_is_repaired_and_produced() {
        let dir = tempfile::tempdir().expect("temp dir");
        let mut model = Workbook {
            title: "Readings".to_string(),
            classification: "OFFICIAL".to_string(),
            sheets: vec![Sheet {
                name: "Q1/Q2 readings".to_string(), // `/` is illegal in Excel
                columns: vec![
                    Column { header: "Point".into(), kind: ColumnType::Text, width: None },
                    Column { header: "Measured".into(), kind: ColumnType::Number, width: None },
                ],
                rows: vec![vec!["S-01".into(), "9.4".into()]],
                freeze_header: true,
            }],
        };

        let outcome = produce_workbook(&dir.path().join("readings.xlsx"), &mut model);
        assert!(outcome.succeeded(), "{:?}", outcome.failure);
        assert!(outcome.repairs.iter().any(|r| r.contains("renamed")), "{:?}", outcome.repairs);
        assert!(!model.sheets[0].name.contains('/'));
    }

    /// Prose in a column of numbers is not something a machine may guess at.
    #[test]
    fn a_workbook_whose_types_are_wrong_produces_nothing() {
        let dir = tempfile::tempdir().expect("temp dir");
        let mut model = Workbook {
            title: "Readings".to_string(),
            classification: "OFFICIAL".to_string(),
            sheets: vec![Sheet {
                name: "Readings".to_string(),
                columns: vec![Column {
                    header: "Measured".into(),
                    kind: ColumnType::Number,
                    width: None,
                }],
                rows: vec![vec!["about nine millimetres".into()]],
                freeze_header: false,
            }],
        };

        let outcome = produce_workbook(&dir.path().join("readings.xlsx"), &mut model);
        assert!(!outcome.succeeded());
        assert!(outcome.failure.expect("a reason").contains("declared a number"));
    }

    /// The repair that matters for a deck: the content was fine, there was just
    /// too much of it for one slide.
    #[test]
    fn an_overflowing_deck_is_split_rather_than_refused() {
        let dir = tempfile::tempdir().expect("temp dir");
        let mut model = Deck {
            title: "Outage briefing".to_string(),
            classification: "OFFICIAL".to_string(),
            slides: vec![SlideModel {
                heading: "Findings".to_string(),
                bullets: (1..=11).map(|n| format!("Finding number {n}")).collect(),
                table: None,
                notes: None,
            }],
        };

        let outcome = produce_deck(&dir.path().join("briefing.pptx"), &mut model, false);
        assert!(outcome.succeeded(), "{:?}", outcome.failure);
        assert!(outcome.repairs.iter().any(|r| r.contains("split")), "{:?}", outcome.repairs);
        assert_eq!(model.slides.len(), 2);
        let carried: usize = model.slides.iter().map(|s| s.bullets.len()).sum();
        assert_eq!(carried, 11, "a split must not lose a bullet");
    }

    #[test]
    fn a_pdf_is_produced_and_read_back_by_the_real_validator() {
        let dir = tempfile::tempdir().expect("temp dir");
        let mut model = document();
        let outcome = produce_pdf(&dir.path().join("report.pdf"), &mut model);

        assert!(outcome.succeeded(), "{:?}", outcome.failure);
        let artifact = outcome.artifact.expect("a file");
        let check = crate::artifacts::pdf_validate::check_pdf(&artifact);
        assert!(check.is_sound(), "{:?}", check.problems);
        assert!(check.characters > 0);
    }

    /// Quality is part of acceptance, not an advisory beside it: a PDF that is
    /// structurally perfect and nearly blank must not be handed over.
    #[test]
    fn an_accepted_pdf_has_passed_its_quality_pass_too() {
        let dir = tempfile::tempdir().expect("temp dir");
        let mut model = document();
        model.sections[0].blocks =
            vec![Block::Paragraph { text: "Nothing of note today. OK.".to_string() }];

        let outcome = produce_pdf(&dir.path().join("thin.pdf"), &mut model);
        match outcome.artifact {
            Some(path) => {
                let check = crate::artifacts::pdf_validate::check_pdf(&path);
                let notes = crate::artifacts::pdf_validate::quality::inspect(&check);
                assert!(notes.is_empty(), "an accepted PDF must pass its quality pass: {notes:?}");
            }
            None => {
                assert!(outcome.failure.expect("a reason").contains("sparse"));
            }
        }
    }

    /// Every attempt stays on disk with the reason it was superseded. "What did
    /// it get wrong the first time" is a question a reviewer will ask.
    #[test]
    fn a_superseded_attempt_is_never_the_artifact_that_stands() {
        let dir = tempfile::tempdir().expect("temp dir");
        let mut model = document();
        let outcome = produce_document(&dir.path().join("report.docx"), &mut model, &metadata());

        for revision in &outcome.revisions {
            if revision.superseded_because.is_some() {
                assert_ne!(
                    revision.path, outcome.artifact,
                    "a superseded revision must never be the artifact that stands"
                );
            }
        }
    }
}
