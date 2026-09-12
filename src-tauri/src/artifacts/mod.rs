//! Turning a task's findings into something somebody can hand to their manager.
//!
//! PS 26117: *"Output should be real deliverables, approval notes, PPT/Word/Excel
//! files, working code, calculations with steps shown, not just chat replies."*
//!
//! - [`docx`]: Word documents from templates the model cannot improvise around.
//! - [`xlsx`]: the calculation workbook, where Excel recomputes and can disagree.
//! - [`pptx`]: the briefing deck, the most tightly templated of the three.
//! - [`ooxml`]: escaping and packaging, shared by both.
//! - [`production`]: compose, render, re-open, verify — revising, never
//!   overwriting.
//! - [`verifier`]: the check between a draft and something somebody signs.
//! - [`visible_watermark`]: the printed provenance claim stamped onto every
//!   generated document. See the module docs for the honest scope: this is
//!   traceability, not attribution; the steganographic alternative is
//!   deliberately not implemented.
//! - [`stego_watermark`]: the *refused* counterpart to the visible watermark.
//!   Always returns an error; see its module docs for the written-out
//!   reasoning so the refusal survives the contributor who inherits it.

pub mod chart;
pub mod captured_blocks;
pub mod conversation_store;
pub mod diagram;
pub mod doc_model;
pub mod docx;
pub mod live_source;
pub mod ooxml;
pub mod pdf;
pub mod pdf_validate;
pub mod pptx;
pub mod produce_model;
pub mod production;
pub mod stego_watermark;
pub mod svg_validate;
pub mod visible_watermark;
pub mod xlsx;
pub mod text_formats;
pub mod verifier;

pub use docx::{check_document, write_document, DocumentCheck, DocumentMetadata};
pub use pptx::{check_deck, write_deck, DeckCheck, Slide, BRIEFING_SECTIONS};
pub use production::{produce, ContentSource, ProductionOutcome, Revision};
pub use xlsx::{check_workbook, write_workbook, WorkbookCheck};
pub use pdf_validate::{check_pdf, PdfCheck};
pub use svg_validate::{check_svg, SvgCheck};
pub use verifier::{
    verify, Coverage, Evidence, Grounding, Severity, Standing, VerificationReport,
};

#[cfg(test)]
mod model_round_trip {
    //! Phase E: one content model, four formats, checked by the real
    //! validators.
    //!
    //! Each test renders a composed model and re-opens the file with the
    //! validator that ships for that format — not with an assertion about the
    //! bytes this code just wrote. A writer that satisfies its own test and
    //! produces a file the format's own reader rejects is the failure these
    //! exist to catch.

    use super::doc_model::{
        Block, Column, ColumnType, Deck, Document, Properties, Section, Sheet, SlideModel, Workbook,
    };
    use super::docx::DocumentMetadata;

    fn metadata() -> DocumentMetadata {
        DocumentMetadata {
            task_id: "run-e".to_string(),
            created_at: "2026-03-04T09:00:00Z".to_string(),
            model: "Nemotron3-Nano-4B".to_string(),
            classification: "OFFICIAL".to_string(),
            is_draft: false,
        }
    }

    /// A document that is not an approval note — the thing the template path
    /// could not express at all.
    fn inspection_report() -> Document {
        Document {
            title: "Unit Four shell thickness inspection".to_string(),
            classification: "OFFICIAL".to_string(),
            properties: Properties {
                author: Some("Inspection team".to_string()),
                subject: Some("Shell thickness survey, March outage".to_string()),
                keywords: vec!["inspection".to_string(), "Unit Four".to_string()],
            },
            sections: vec![
                Section {
                    heading: "Scope".to_string(),
                    level: 1,
                    blocks: vec![Block::Paragraph {
                        text: "Ultrasonic thickness readings were taken at sixteen points on the \
                               vessel shell during the March outage."
                            .to_string(),
                    }],
                },
                Section {
                    heading: "Readings".to_string(),
                    level: 2,
                    blocks: vec![Block::Table {
                        header: vec!["Point".into(), "Measured mm".into(), "Minimum mm".into()],
                        rows: vec![
                            vec!["S-01".into(), "9.4".into(), "9.0".into()],
                            vec!["S-02".into(), "8.7".into(), "9.0".into()],
                        ],
                        caption: Some("Thickness against the stated minimum".to_string()),
                    }],
                },
                Section {
                    heading: "Recommendation".to_string(),
                    level: 1,
                    blocks: vec![
                        Block::Paragraph {
                            text: "Point S-02 is below the stated minimum and requires assessment \
                                   before the unit returns to service."
                                .to_string(),
                        },
                        Block::Numbered {
                            items: vec![
                                "Re-measure S-02 to confirm the reading.".to_string(),
                                "Refer to the integrity engineer if confirmed.".to_string(),
                            ],
                        },
                    ],
                },
            ],
        }
    }

    #[test]
    fn a_composed_document_becomes_a_word_file_that_opens() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("inspection.docx");
        let document = inspection_report();

        super::docx::write_document_model(&path, &document, &metadata()).expect("writes");

        // Re-opened through the package reader, not asserted about in memory.
        let parts = super::ooxml::list_parts(&path).expect("the package opens");
        assert!(parts.iter().any(|p| p == "word/document.xml"), "{parts:?}");
        assert!(
            parts.iter().any(|p| p == "docProps/core.xml"),
            "a deliverable must carry its properties: {parts:?}"
        );

        let body = super::ooxml::read_part(&path, "word/document.xml").expect("the body reads");
        // Real outline levels, not everything flattened to Heading1.
        assert!(body.contains("Heading2"), "section headings must carry their level");
        assert!(body.contains("<w:tbl>"), "the table must render as a table");
        assert!(body.contains("<w:tblHeader/>"), "the header row must repeat across pages");
        assert!(body.contains("Point S-02 is below"), "the prose must be in the file");

        let core = super::ooxml::read_part(&path, "docProps/core.xml").expect("properties read");
        assert!(core.contains("Unit Four shell thickness inspection"));
        assert!(core.contains("Inspection team"));
    }

    /// The same model, the other format. Asking for the same content twice
    /// must not produce two different documents.
    #[test]
    fn the_same_document_becomes_a_pdf_the_validator_accepts() {
        let document = inspection_report();
        let spec = super::pdf::spec_from_document(&document);
        let bytes = super::pdf::render(&spec).expect("renders");

        let check = super::pdf_validate::check_pdf_bytes(&bytes);
        assert!(check.is_sound(), "{:?}", check.problems);
        let notes = super::pdf_validate::quality::inspect(&check);
        assert!(notes.is_empty(), "{notes:?}");
        assert!(check.has_metadata, "the PDF must carry its title");
        // The words survived the crossing.
        assert!(check.text.contains("S-02"), "the table must reach the page");
        assert!(check.text.contains("Re-measure"), "the numbered list must reach the page");
    }

    #[test]
    fn a_page_break_starts_a_new_page() {
        let mut document = inspection_report();
        document.sections[1].blocks.insert(0, Block::PageBreak);

        let without = super::pdf::render(&super::pdf::spec_from_document(&inspection_report()))
            .expect("renders");
        let with = super::pdf::render(&super::pdf::spec_from_document(&document)).expect("renders");

        let before = super::pdf_validate::check_pdf_bytes(&without);
        let after = super::pdf_validate::check_pdf_bytes(&with);
        assert!(before.is_sound() && after.is_sound());
        assert!(
            after.pages > before.pages,
            "a page break must add a page: {} then {}",
            before.pages,
            after.pages
        );
    }

    #[test]
    fn a_composed_workbook_becomes_an_excel_file_with_real_types() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("readings.xlsx");
        let workbook = Workbook {
            title: "Shell thickness readings".to_string(),
            classification: "OFFICIAL".to_string(),
            sheets: vec![Sheet {
                name: "Readings".to_string(),
                columns: vec![
                    Column { header: "Point".into(), kind: ColumnType::Text, width: None },
                    Column { header: "Measured".into(), kind: ColumnType::Number, width: None },
                    Column { header: "Inspected".into(), kind: ColumnType::Date, width: None },
                    Column { header: "Margin".into(), kind: ColumnType::Formula, width: None },
                ],
                rows: vec![
                    vec!["S-01".into(), "9.4".into(), "2026-03-04".into(), "=B2-9".into()],
                    vec!["S-02".into(), "8.7".into(), "2026-03-04".into(), "=B3-9".into()],
                ],
                freeze_header: true,
            }],
        };

        super::xlsx::write_workbook_model(&path, &workbook).expect("writes");

        let parts = super::ooxml::list_parts(&path).expect("the package opens");
        assert!(parts.iter().any(|p| p == "xl/workbook.xml"), "{parts:?}");
        assert!(parts.iter().any(|p| p == "xl/worksheets/sheet1.xml"), "{parts:?}");

        let book = super::ooxml::read_part(&path, "xl/workbook.xml").expect("reads");
        assert!(
            book.contains("fullCalcOnLoad"),
            "without this Excel shows empty cells where the formulas are"
        );

        let sheet = super::ooxml::read_part(&path, "xl/worksheets/sheet1.xml").expect("reads");
        // The number is a number, not a string that looks like one. A formula
        // referring to a string cell evaluates to zero, silently.
        assert!(sheet.contains("<v>9.4</v>"), "the measurement must be a numeric cell: {sheet}");
        assert!(sheet.contains("<f>B2-9</f>"), "the formula must be live: {sheet}");
        assert!(sheet.contains("state=\"frozen\""), "the header must stay visible");
        assert!(sheet.contains("customWidth"), "columns must be sized");
    }

    #[test]
    fn a_composed_deck_becomes_a_presentation_with_notes() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("briefing.pptx");
        let deck = Deck {
            title: "Unit Four return to service".to_string(),
            classification: "OFFICIAL".to_string(),
            slides: vec![
                SlideModel {
                    heading: "What we found".to_string(),
                    bullets: vec![
                        "Sixteen points measured on the shell.".to_string(),
                        "One point below the stated minimum.".to_string(),
                    ],
                    table: None,
                    notes: Some(
                        "S-02 is the point in question; the reading was 8.7 mm.".to_string(),
                    ),
                },
                SlideModel {
                    heading: "What we recommend".to_string(),
                    bullets: vec!["Re-measure before the unit returns to service.".to_string()],
                    table: None,
                    notes: None,
                },
            ],
        };

        super::pptx::write_deck_model(&path, &deck, false).expect("writes");

        let parts = super::ooxml::list_parts(&path).expect("the package opens");
        // Title slide plus the two composed ones.
        assert!(parts.iter().any(|p| p == "ppt/slides/slide3.xml"), "{parts:?}");
        assert!(
            parts.iter().any(|p| p == "ppt/notesSlides/notesSlide2.xml"),
            "the speaker notes must be in the package: {parts:?}"
        );

        let check = super::pptx::check_deck(&path);
        assert!(check.opens, "{:?}", check.problems);
        assert_eq!(check.slides, 3);

        let notes =
            super::ooxml::read_part(&path, "ppt/notesSlides/notesSlide2.xml").expect("reads");
        assert!(notes.contains("8.7 mm"), "the note's text must survive");

        // The slide that has no notes must not claim a notes relationship.
        let rels =
            super::ooxml::read_part(&path, "ppt/slides/_rels/slide3.xml.rels").expect("reads");
        assert!(!rels.contains("notesSlide"), "a dangling relationship breaks the package");
    }

    /// Nothing unsound is ever written. Each writer refuses the model rather
    /// than producing a file somebody has to discover is wrong.
    #[test]
    fn an_unsound_model_is_refused_before_a_file_exists() {
        let dir = tempfile::tempdir().expect("temp dir");

        let mut broken = inspection_report();
        broken.sections[0].blocks = vec![Block::Paragraph { text: "   ".to_string() }];
        let path = dir.path().join("broken.docx");
        super::docx::write_document_model(&path, &broken, &metadata())
            .expect_err("an empty section must be refused");
        assert!(!path.exists(), "nothing may be written for a model that does not validate");

        let ragged = Workbook {
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
        let path = dir.path().join("broken.xlsx");
        super::xlsx::write_workbook_model(&path, &ragged)
            .expect_err("prose in a number column must be refused");
        assert!(!path.exists());
    }
}

#[cfg(test)]
mod audit_emit_tests {
    //! Writes one of each artifact to a directory a person can open.
    //!
    //! Not a check in itself: the assertions live in the generators' own tests.
    //! This exists so the files can be validated by something that is not this
    //! codebase - Word's own parser, by way of python-docx and python-pptx -
    //! because a file that satisfies our writer and not Office is a file the
    //! person cannot open, and no test written here would notice.
    use std::collections::BTreeMap;
    use std::path::PathBuf;

    fn out() -> PathBuf {
        let dir = std::env::var("ARJUN_AUDIT_OUT").unwrap_or_default();
        PathBuf::from(dir)
    }

    #[test]
    #[ignore = "writes files for external validation; run with ARJUN_AUDIT_OUT set"]
    fn emit_one_of_each() {
        let dir = out();
        assert!(!dir.as_os_str().is_empty(), "set ARJUN_AUDIT_OUT");
        std::fs::create_dir_all(&dir).expect("out dir");

        let mut fields = BTreeMap::new();
        for (k, v) in [
            ("title", "Approval Note - Pump PV-2201 Replacement"),
            ("recipient", "Head of Maintenance, Unit Four"),
            ("subject", "Replacement of control valve PV-2201"),
            ("findings", "Northern Valve Company attended site during the March outage."),
            ("calculation", "Flow coefficient Cv = 42.0 at 6 bar differential."),
            ("recommendation", "Approve the replacement under the existing supply agreement."),
            ("references", "Maintenance Report Unit Four, page 1."),
            ("assumptions", "The March outage window remains as scheduled."),
        ] {
            fields.insert(k.to_string(), v.to_string());
        }
        let metadata = super::docx::DocumentMetadata {
            task_id: "audit-run".to_string(),
            created_at: "2026-09-08T00:00:00Z".to_string(),
            model: "Nemotron3-Nano-4B".to_string(),
            classification: "OFFICIAL".to_string(),
            is_draft: false,
        };
        super::docx::write_document(&dir.join("approval_note.docx"), "approval_note", &fields, &metadata)
            .expect("docx");

        let slides: Vec<super::pptx::Slide> = super::pptx::BRIEFING_SECTIONS
            .iter()
            .map(|section| super::pptx::Slide {
                heading: section.to_string(),
                bullets: vec![
                    format!("First point for {section}."),
                    format!("Second point for {section}."),
                ],
            })
            .collect();
        super::pptx::write_deck(
            &dir.join("briefing_deck.pptx"),
            "Unit Four Outage Briefing",
            "OFFICIAL",
            &slides,
            false,
        )
        .expect("pptx");

        let records = vec![crate::orchestrator::calculation::evaluate("2 m * 3 m").expect("calc")];
        super::xlsx::write_workbook(&dir.join("calculations.xlsx"), &records, "OFFICIAL")
            .expect("xlsx");

        // Charts, both kinds, so the drawing can be looked at rather than
        // only asserted about.
        for (kind, file) in [
            (super::chart::ChartKind::Bar, "chart_bar.svg"),
            (super::chart::ChartKind::Line, "chart_line.svg"),
        ] {
            let spec = super::chart::ChartSpec {
                kind,
                title: "Throughput by unit".to_string(),
                categories: vec![
                    "Unit One".to_string(),
                    "Unit Four".to_string(),
                    "Unit Seven".to_string(),
                ],
                series: vec![
                    super::chart::Series {
                        name: "Actual".to_string(),
                        values: vec![120.0, 96.0, 143.0],
                    },
                    super::chart::Series {
                        name: "Target".to_string(),
                        values: vec![130.0, 110.0, 130.0],
                    },
                ],
                value_label: "tonnes/day".to_string(),
            };
            let svg = super::chart::render_svg(&spec).expect("chart");
            std::fs::write(dir.join(file), svg).expect("write svg");
        }

        // A PDF and a diagram, for external validation.
        let pdf = super::pdf::render(&super::pdf::PdfSpec {
            title: "Approval Note - PV-2201 Replacement".to_string(),
            classification: "OFFICIAL".to_string(),
            blocks: vec![
                super::pdf::Block::Heading("Findings".to_string()),
                super::pdf::Block::Paragraph(
                    "Northern Valve Company attended site during the March outage and                      replaced control valve PV-2201 within the shift."
                        .to_string(),
                ),
                super::pdf::Block::Bullet("Flow coefficient Cv = 42.0 at 6 bar.".to_string()),
                super::pdf::Block::Fixed("Item        Qty   Cost".to_string()),
                super::pdf::Block::Fixed("PV-2201       1   45000".to_string()),
            ],
        })
        .expect("pdf");
        std::fs::write(dir.join("approval_note.pdf"), pdf).expect("write pdf");

        let diagram = super::diagram::render_svg(&super::diagram::DiagramSpec {
            title: "Unit Four feed line".to_string(),
            direction: super::diagram::Direction::Across,
            nodes: vec![
                super::diagram::Node {
                    id: "inlet".into(),
                    label: "Feed inlet".into(),
                    shape: super::diagram::Shape::Box,
                    tag: None,
                },
                super::diagram::Node {
                    id: "pump".into(),
                    label: "Charge pump".into(),
                    shape: super::diagram::Shape::Box,
                    tag: Some("P-101A".into()),
                },
                super::diagram::Node {
                    id: "cv".into(),
                    label: "Control valve".into(),
                    shape: super::diagram::Shape::Valve,
                    tag: Some("PV-2201".into()),
                },
                super::diagram::Node {
                    id: "drum".into(),
                    label: "Surge drum".into(),
                    shape: super::diagram::Shape::Vessel,
                    tag: Some("TK-101".into()),
                },
                super::diagram::Node {
                    id: "pt".into(),
                    label: "Pressure transmitter".into(),
                    shape: super::diagram::Shape::Instrument,
                    tag: Some("PT-2201".into()),
                },
            ],
            edges: vec![
                super::diagram::Edge { from: "inlet".into(), to: "pump".into(), label: None },
                super::diagram::Edge {
                    from: "pump".into(),
                    to: "cv".into(),
                    label: Some("40 m3/h".into()),
                },
                super::diagram::Edge { from: "cv".into(), to: "drum".into(), label: None },
                super::diagram::Edge { from: "drum".into(), to: "pt".into(), label: None },
            ],
        })
        .expect("diagram");
        std::fs::write(dir.join("feed_line.svg"), diagram).expect("write diagram");

        eprintln!("WROTE>>>{}<<<", dir.display());
    }
}
