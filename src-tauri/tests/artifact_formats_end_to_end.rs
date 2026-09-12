//! One representative artifact per format, produced and then verified by the
//! validator that actually ships for it.
//!
//! ## What this is for
//!
//! Every format in phases A–I has its own unit tests. This is the pass that
//! puts them together: for each format, compose real content, produce a real
//! file through the production path, and re-open it with the real validator. A
//! format with a generator and a validator that were never run against each
//! other is two halves of a claim.
//!
//! It also refuses each format's characteristic corruption, in the same run, so
//! "the validator accepted it" means something. A validator that cannot fail is
//! worse than no validator, and the only way to know it can is to hand it
//! something broken.
//!
//! ## Inspecting the output
//!
//! Set `ARJUN_ARTIFACT_OUT` to a directory and the files are written there
//! instead of a temporary one, so they can be opened in Word, Excel,
//! PowerPoint, a PDF reader and a browser. Nothing here asserts that they
//! *look* right — no test can — and opening them is how that gets checked.

use std::path::PathBuf;

use sarathi_lib::artifacts::doc_model::{
    Block, Column, ColumnType, Deck, Document, Properties, Section, Sheet, SlideModel, Workbook,
};
use sarathi_lib::artifacts::docx::DocumentMetadata;
use sarathi_lib::artifacts::{pdf, pdf_validate, produce_model, svg_validate, text_formats};

fn out_dir() -> (PathBuf, Option<tempfile::TempDir>) {
    match std::env::var("ARJUN_ARTIFACT_OUT") {
        Ok(dir) if !dir.trim().is_empty() => {
            let path = PathBuf::from(dir);
            std::fs::create_dir_all(&path).expect("the output directory");
            (path, None)
        }
        _ => {
            let dir = tempfile::tempdir().expect("temp dir");
            (dir.path().to_path_buf(), Some(dir))
        }
    }
}

fn metadata() -> DocumentMetadata {
    DocumentMetadata {
        task_id: "audit-run".to_string(),
        created_at: "2026-03-04T09:00:00Z".to_string(),
        model: "Nemotron3-Nano-4B".to_string(),
        classification: "OFFICIAL".to_string(),
        is_draft: false,
    }
}

/// A real deliverable: an inspection report with a table, a list and prose.
fn report() -> Document {
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
                           vessel shell during the March outage, against a stated minimum of \
                           9.0 mm."
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
                        vec!["S-03".into(), "9.1".into(), "9.0".into()],
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

fn readings() -> Workbook {
    Workbook {
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
                vec!["S-03".into(), "9.1".into(), "2026-03-04".into(), "=B4-9".into()],
            ],
            freeze_header: true,
        }],
    }
}

fn briefing() -> Deck {
    Deck {
        title: "Unit Four return to service".to_string(),
        classification: "OFFICIAL".to_string(),
        slides: vec![
            SlideModel {
                heading: "What we inspected".to_string(),
                bullets: vec![
                    "Sixteen points on the vessel shell, during the March outage.".to_string(),
                    "Ultrasonic thickness, against a 9.0 mm minimum.".to_string(),
                ],
                table: None,
                notes: Some("The probe was calibrated on the morning of the survey.".to_string()),
            },
            SlideModel {
                heading: "What we found".to_string(),
                bullets: vec!["One point of sixteen is below the minimum.".to_string()],
                table: Some(Block::Table {
                    header: vec!["Point".into(), "Measured".into()],
                    rows: vec![vec!["S-02".into(), "8.7 mm".into()]],
                    caption: Some("Below the stated minimum".to_string()),
                }),
                notes: None,
            },
            SlideModel {
                heading: "What happens next".to_string(),
                bullets: vec!["Re-measure S-02 before the unit returns to service.".to_string()],
                table: None,
                notes: None,
            },
        ],
    }
}

/// Every format, produced and verified. One test rather than twelve so the
/// output directory holds a complete set after one run.
#[test]
fn every_supported_format_is_produced_and_accepted_by_its_own_validator() {
    let (dir, _keep) = out_dir();
    let mut produced: Vec<(String, u64)> = Vec::new();

    // ── DOCX ────────────────────────────────────────────────────────────
    let mut document = report();
    let outcome =
        produce_model::produce_document(&dir.join("report.docx"), &mut document, &metadata());
    let docx = outcome.artifact.expect("the document is produced");
    let body = sarathi_lib::artifacts::ooxml::read_part(&docx, "word/document.xml")
        .expect("the package opens and holds a body");
    assert!(body.contains("<w:tbl>"), "the table must be a table");
    assert!(body.contains("Heading2"), "the outline must carry real levels");
    assert!(
        sarathi_lib::artifacts::ooxml::read_part(&docx, "docProps/core.xml").is_ok(),
        "a deliverable carries its properties"
    );
    produced.push(("docx".into(), std::fs::metadata(&docx).unwrap().len()));

    // ── PDF, from the same model ────────────────────────────────────────
    let mut same = report();
    let outcome = produce_model::produce_pdf(&dir.join("report.pdf"), &mut same);
    let pdf_path = outcome.artifact.expect("the PDF is produced");
    let check = pdf_validate::check_pdf(&pdf_path);
    assert!(check.is_sound(), "{:?}", check.problems);
    assert!(check.has_metadata, "the PDF carries its title");
    assert!(check.text.contains("S-02"), "the table reached the page");
    assert!(
        pdf_validate::quality::inspect(&check).is_empty(),
        "an accepted PDF passes its quality pass too"
    );
    produced.push(("pdf".into(), std::fs::metadata(&pdf_path).unwrap().len()));

    // ── XLSX ────────────────────────────────────────────────────────────
    let mut workbook = readings();
    let outcome = produce_model::produce_workbook(&dir.join("readings.xlsx"), &mut workbook);
    let xlsx = outcome.artifact.expect("the workbook is produced");
    let sheet = sarathi_lib::artifacts::ooxml::read_part(&xlsx, "xl/worksheets/sheet1.xml")
        .expect("the sheet opens");
    assert!(sheet.contains("<v>9.4</v>"), "a measurement is a number, not a string");
    assert!(sheet.contains("<f>B2-9</f>"), "the formula is live");
    assert!(sheet.contains("state=\"frozen\""), "the header stays visible");
    produced.push(("xlsx".into(), std::fs::metadata(&xlsx).unwrap().len()));

    // ── PPTX ────────────────────────────────────────────────────────────
    let mut deck = briefing();
    let outcome = produce_model::produce_deck(&dir.join("briefing.pptx"), &mut deck, false);
    let pptx = outcome.artifact.expect("the deck is produced");
    let check = sarathi_lib::artifacts::pptx::check_deck(&pptx);
    assert!(check.opens, "{:?}", check.problems);
    assert_eq!(check.slides, 4, "a title slide plus the three composed ones");
    assert!(
        sarathi_lib::artifacts::ooxml::read_part(&pptx, "ppt/notesSlides/notesSlide2.xml").is_ok(),
        "the speaker notes are in the package"
    );
    produced.push(("pptx".into(), std::fs::metadata(&pptx).unwrap().len()));

    // ── SVG: a diagram and a chart ──────────────────────────────────────
    let diagram =
        sarathi_lib::artifacts::diagram::render_svg(&sarathi_lib::artifacts::diagram::DiagramSpec {
            title: "Cooling water loop".to_string(),
            direction: sarathi_lib::artifacts::diagram::Direction::Across,
            nodes: vec![
                sarathi_lib::artifacts::diagram::Node {
                    id: "p1".to_string(),
                    label: "Cooling water pump".to_string(),
                    shape: sarathi_lib::artifacts::diagram::Shape::Box,
                    tag: Some("P-101".to_string()),
                },
                sarathi_lib::artifacts::diagram::Node {
                    id: "v1".to_string(),
                    label: "Control valve".to_string(),
                    shape: sarathi_lib::artifacts::diagram::Shape::Valve,
                    tag: Some("PV-2201".to_string()),
                },
            ],
            edges: vec![sarathi_lib::artifacts::diagram::Edge {
                from: "p1".to_string(),
                to: "v1".to_string(),
                label: Some("CW supply".to_string()),
            }],
        })
        .expect("the diagram renders");
    let path = dir.join("loop.svg");
    std::fs::write(&path, &diagram).expect("written");
    let check = svg_validate::check_svg(&path);
    assert!(check.is_sound(), "{:?}", check.problems);
    assert!(check.shapes > 0 && check.labels > 0);
    produced.push(("svg (diagram)".into(), diagram.len() as u64));

    let chart =
        sarathi_lib::artifacts::chart::render_svg(&sarathi_lib::artifacts::chart::ChartSpec {
            kind: sarathi_lib::artifacts::chart::ChartKind::Bar,
            title: "Thickness by point".to_string(),
            categories: vec!["S-01".to_string(), "S-02".to_string(), "S-03".to_string()],
            series: vec![sarathi_lib::artifacts::chart::Series {
                name: "Measured".to_string(),
                values: vec![9.4, 8.7, 9.1],
            }],
            value_label: "millimetres".to_string(),
        })
        .expect("the chart renders");
    let path = dir.join("thickness.svg");
    std::fs::write(&path, &chart).expect("written");
    assert!(svg_validate::check_svg(&path).is_sound());
    produced.push(("svg (chart)".into(), chart.len() as u64));

    // ── The text formats ────────────────────────────────────────────────
    let markdown = text_formats::to_markdown(&report());
    std::fs::write(dir.join("report.md"), &markdown).expect("written");
    let check = text_formats::check(text_formats::Format::Markdown, &markdown);
    assert!(check.is_sound(), "{:?}", check.problems);
    assert!(text_formats::quality(text_formats::Format::Markdown, &markdown).is_empty());
    produced.push(("md".into(), markdown.len() as u64));

    let html = text_formats::to_html(&report());
    std::fs::write(dir.join("report.html"), &html).expect("written");
    let check = text_formats::check(text_formats::Format::Html, &html);
    assert!(check.is_sound(), "{:?}", check.problems);
    produced.push(("html".into(), html.len() as u64));

    let csv = text_formats::to_csv(
        &["Point".to_string(), "Measured".to_string(), "Note".to_string()],
        &[
            vec!["S-01".into(), "9.4".into(), "Within tolerance".into()],
            vec!["S-02".into(), "8.7".into(), "Below minimum, re-measure".into()],
        ],
    );
    std::fs::write(dir.join("readings.csv"), &csv).expect("written");
    let check = text_formats::check(text_formats::Format::Csv, &csv);
    assert!(check.is_sound(), "{:?}", check.problems);
    // The comma inside a note must not have become a column break.
    let parsed = text_formats::parse_csv(&csv).expect("parses");
    assert_eq!(parsed[2].len(), 3);
    produced.push(("csv".into(), csv.len() as u64));

    let value = serde_json::json!({
        "unit": "Unit Four",
        "minimum_mm": 9.0,
        "readings": [
            {"point": "S-01", "measured_mm": 9.4},
            {"point": "S-02", "measured_mm": 8.7}
        ]
    });

    let json = text_formats::to_json(&value);
    std::fs::write(dir.join("readings.json"), &json).expect("written");
    assert!(text_formats::check(text_formats::Format::Json, &json).is_sound());
    produced.push(("json".into(), json.len() as u64));

    let yaml = text_formats::to_yaml(&value);
    std::fs::write(dir.join("readings.yaml"), &yaml).expect("written");
    let check = text_formats::check(text_formats::Format::Yaml, &yaml);
    assert!(check.is_sound(), "{yaml}\n{:?}", check.problems);
    produced.push(("yaml".into(), yaml.len() as u64));

    let xml = text_formats::to_xml("inspection", &value);
    std::fs::write(dir.join("readings.xml"), &xml).expect("written");
    let check = text_formats::check(text_formats::Format::Xml, &xml);
    assert!(check.is_sound(), "{xml}\n{:?}", check.problems);
    produced.push(("xml".into(), xml.len() as u64));

    println!("\nProduced and verified, in {}:", dir.display());
    for (format, bytes) in &produced {
        println!("  {format:<16} {bytes:>8} bytes");
    }
    assert_eq!(produced.len(), 12, "every format in phases A-I");
}

/// The other half: each validator refuses that format's characteristic
/// corruption. An accepted artifact means nothing if nothing is ever rejected.
#[test]
fn every_validator_rejects_its_formats_characteristic_corruption() {
    // PDF: truncated mid-object-table. The file looks the right shape and will
    // not open.
    let spec = pdf::spec_from_document(&report());
    let bytes = pdf::render(&spec).expect("renders");
    assert!(
        !pdf_validate::check_pdf_bytes(&bytes[..bytes.len() / 2]).is_sound(),
        "a truncated PDF must be refused"
    );

    // SVG: a marker reference nothing defines, so the arrowheads silently
    // vanish.
    let svg = "<svg viewBox=\"0 0 90 90\" aria-label=\"x\">\
               <path marker-end=\"url(#missing)\"/><text x=\"1\" y=\"1\">a</text></svg>";
    assert!(!svg_validate::check_svg_text(svg).is_sound());

    // CSV: a row narrower than its header. Opens, and every column after it is
    // misaligned.
    assert!(!text_formats::check(text_formats::Format::Csv, "A,B\r\n1\r\n").is_sound());

    // JSON: a trailing comma.
    assert!(!text_formats::check(text_formats::Format::Json, "{\"a\": 1,}").is_sound());

    // YAML: a tab, which the specification forbids outright.
    assert!(!text_formats::check(text_formats::Format::Yaml, "a:\n\tb: 1\n").is_sound());

    // XML: an element that never closes.
    assert!(!text_formats::check(text_formats::Format::Xml, "<a><b></a>").is_sound());

    // Markdown: a heading hierarchy that skips.
    assert!(!text_formats::check(
        text_formats::Format::Markdown,
        "# A\n\n### C\n\nText here.\n"
    )
    .is_sound());

    // HTML: no language, so a screen reader cannot choose a voice.
    assert!(!text_formats::check(
        text_formats::Format::Html,
        "<html><head><title>T</title></head><body><h1>H</h1></body></html>"
    )
    .is_sound());

    // The content model itself: a table whose rows disagree with its header.
    let ragged = Document {
        title: "T".to_string(),
        classification: "OFFICIAL".to_string(),
        properties: Properties::default(),
        sections: vec![Section {
            heading: "Readings".to_string(),
            level: 1,
            blocks: vec![Block::Table {
                header: vec!["A".into(), "B".into(), "C".into()],
                rows: vec![vec!["1".into(), "2".into(), "3".into(), "4".into()]],
                caption: None,
            }],
        }],
    };
    assert!(!ragged.is_sound(), "a ragged table must not render");
}

/// A failed artifact is never returned as a success, and never left where the
/// caller asked for it.
#[test]
fn nothing_unsound_is_ever_presented_as_finished() {
    let (dir, _keep) = out_dir();
    let base = dir.join("impossible.docx");

    // A section with a heading and nothing under it: not something repair can
    // invent its way out of.
    let mut document = Document {
        title: "Impossible".to_string(),
        classification: "OFFICIAL".to_string(),
        properties: Properties::default(),
        sections: vec![Section {
            heading: "Findings".to_string(),
            level: 1,
            blocks: vec![Block::Paragraph { text: "   ".to_string() }],
        }],
    };

    let outcome = produce_model::produce_document(&base, &mut document, &metadata());
    assert!(!outcome.succeeded());
    assert!(outcome.artifact.is_none(), "no artifact may be offered");
    assert!(outcome.failure.is_some(), "the failure must say why");
    assert!(!base.exists(), "nothing may be left at the path the caller asked for");
}

/// Phase H, end to end: a task that reports progress is not killed for being
/// slow, and one that goes silent still is.
#[test]
fn a_progressing_task_outlives_the_old_wall_clock_ceilings() {
    use sarathi_lib::orchestrator::progress::{Advance, Phase, Tracker};

    // The old ceiling for `create_pdf` was fifteen seconds and for
    // `create_diagram` ten. A tracker reporting progress passes far more work
    // than that without being stopped.
    let mut tracker = Tracker::new(Phase::Rendering, 5_000);
    for _ in 0..5_000 {
        tracker.advance(Advance::Blocks(1));
        assert!(tracker.should_stop().is_none());
    }
    assert!(tracker.budget().as_secs() > 15, "a large artifact gets more than the old ceiling");

    // And silence is still what stops it.
    let mut quiet = Tracker::new(Phase::Repairing, 1);
    assert!(quiet.should_stop().is_none(), "a phase that just began is not stalled");
    quiet.advance(Advance::Bytes(0));
    assert!(quiet.should_stop().is_none(), "a zero-sized advance is still a heartbeat");
}

/// Phase G, end to end: a real request selects, loads and carries real skills.
#[test]
fn a_request_selects_and_loads_skills_that_survive_being_carried() {
    use sarathi_lib::identity::{Role, Session, User};
    use sarathi_lib::orchestrator::tools::ToolName;
    use sarathi_lib::skills::{selection, SkillContext, SkillRegistry};
    use sarathi_lib::sovereignty::mode::OperatingMode;

    let shipped = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("src-tauri has a parent")
        .join("skills");
    let registry = SkillRegistry::open(shipped);
    let session = Session::open(User::new("priya", "Priya Sharma", vec![Role::Employee]));
    let context =
        SkillContext { session: &session, mode: OperatingMode::Work, run_permits: ToolName::ALL };

    // A request that names both a format and a domain, and reads nothing --
    // `ReadingContext::default()` is a turn with no notebook and no sources, so
    // this still exercises the output-format pass on its own.
    let bound = selection::bind(
        "Write up the hazard and operability study for the new transfer line",
        Some("docx"),
        &selection::ReadingContext::default(),
        &registry,
        &context,
    );

    assert!(!bound.is_empty(), "nothing was loaded: {:?}", bound.refused);
    assert!(
        bound.names().iter().any(|n| n == "docx-authoring"),
        "the format skill must be selected from the output format: {:?}",
        bound.names()
    );
    for skill in &bound.loaded {
        assert!(!skill.body.trim().is_empty(), "{} has an empty body", skill.name);
        assert_eq!(skill.sha256.len(), 64, "a loaded skill carries its trusted hash");
    }

    let text = bound.as_context().expect("something to inject");
    assert!(text.contains("does not grant any tool"), "a skill may not widen the run");

    // Carried across a retry or a model switch: the same guidance, verbatim.
    let carried = bound.clone();
    assert_eq!(carried.as_context(), bound.as_context());
}

/// Existing workflows still work: the approval-note template and the briefing
/// deck's four fixed sections, through the paths that now go via the loop.
#[test]
fn the_templates_this_product_shipped_with_still_produce_their_artifacts() {
    let (dir, _keep) = out_dir();

    // The approval note, through the template writer.
    let mut fields = std::collections::BTreeMap::new();
    for (key, value) in [
        ("title", "Replacement of control valve PV-2201"),
        ("recipient", "Head of Maintenance, Unit Four"),
        ("subject", "Valve replacement under the supply agreement"),
        ("findings", "The valve seat showed measurable wear at the March outage."),
        ("recommendation", "Approve the replacement under the existing agreement."),
        ("references", "Maintenance Report Unit Four, page 1."),
        ("assumptions", "The outage window remains as scheduled."),
    ] {
        fields.insert(key.to_string(), value.to_string());
    }
    let path = dir.join("approval_note.docx");
    sarathi_lib::artifacts::docx::write_document(&path, "approval_note", &fields, &metadata())
        .expect("the approval note still writes");
    assert!(
        sarathi_lib::artifacts::docx::check_document(&path, "approval_note").is_sound(),
        "the shipped template must still pass its own check"
    );

    // The briefing deck's four fixed sections, through the model path.
    let mut deck = Deck {
        title: "Unit Four outage briefing".to_string(),
        classification: "Internal".to_string(),
        slides: sarathi_lib::artifacts::BRIEFING_SECTIONS
            .iter()
            .map(|heading| SlideModel {
                heading: (*heading).to_string(),
                bullets: vec![format!("A point for the {heading} section.")],
                table: None,
                notes: None,
            })
            .collect(),
    };
    let outcome = produce_model::produce_deck(&dir.join("briefing_template.pptx"), &mut deck, true);
    assert!(outcome.succeeded(), "{:?}", outcome.failure);
}
