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
pub mod diagram;
pub mod docx;
pub mod ooxml;
pub mod pdf;
pub mod pptx;
pub mod production;
pub mod stego_watermark;
pub mod visible_watermark;
pub mod xlsx;
pub mod verifier;

pub use docx::{check_document, write_document, DocumentCheck, DocumentMetadata};
pub use pptx::{check_deck, write_deck, DeckCheck, Slide, BRIEFING_SECTIONS};
pub use production::{produce, ContentSource, ProductionOutcome, Revision};
pub use xlsx::{check_workbook, write_workbook, WorkbookCheck};
pub use verifier::{
    verify, Coverage, Evidence, Grounding, Severity, Standing, VerificationReport,
};

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
