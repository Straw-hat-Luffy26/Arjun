//! Five separate claims about a produced file, each made only when it holds.
//!
//! ## The ladder
//!
//! | Rung | Claim | Established by |
//! |---|---|---|
//! | `fileCreated` | these exact bytes exist and hash to what was recorded | the store |
//! | `formatReopened` | the bytes *are* the format claimed, the container is safe, and it parses | [`super::package`] and the format's own reader |
//! | `contentChecked` | it says something, carries no placeholder, holds what its template promised, and every citation it makes is bound to evidence | [`super::content`] and the template |
//! | `renderChecked` | a real layout engine laid it out, and no page is blank | [`super::render`] |
//! | `accepted` | every rung above passed **and** everything it rests on is still current | this module |
//!
//! A rung is `passed`, `failed`, `unavailable` (the validator is not on this
//! machine — said, never skipped), `notApplicable` (plain text has no pages)
//! or `notRun` (not asked for, or an earlier rung failed so this one would
//! prove nothing).
//!
//! ## What "accepted" does not mean
//!
//! A person's approval. `accepted` is every machine check this product can
//! make; the DRAFT marking stays on the page until somebody signs, and the
//! independent review of a deliverable is the reviewer role's (plan §5.9).
//!
//! ## Why the extension and the ZIP are not enough
//!
//! The extension is a name somebody chose, and every Office file, every
//! `.jar` and every zip bomb opens as a ZIP. The format is read from the bytes,
//! compared with what the file claims to be, and a disagreement fails the
//! reopen rung — a `.docx` that is really a PDF is a wrong-type file, not a
//! Word document with an odd body.

use std::collections::BTreeSet;
use std::path::Path;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::content::{self, ContentModel};
use super::package::{self, DetectedFormat};
use super::render::{self, Inventory, RenderOutcome, RenderState};

/// One rung's standing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum StageState {
    Passed,
    Failed,
    Unavailable,
    NotApplicable,
    NotRun,
}

impl StageState {
    pub const fn as_str(self) -> &'static str {
        match self {
            StageState::Passed => "passed",
            StageState::Failed => "failed",
            StageState::Unavailable => "unavailable",
            StageState::NotApplicable => "not applicable",
            StageState::NotRun => "not run",
        }
    }
}

/// One rung, with what decided it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Stage {
    pub state: StageState,
    /// Which validator, at which version, made the call.
    pub validator: String,
    pub detail: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub problems: Vec<String>,
}

impl Stage {
    fn new(state: StageState, validator: impl Into<String>, detail: impl Into<String>, problems: Vec<String>) -> Self {
        Stage { state, validator: validator.into(), detail: detail.into(), problems }
    }

    fn not_run(because: &str) -> Self {
        Stage::new(StageState::NotRun, "—", because, Vec::new())
    }

    pub fn passed(&self) -> bool {
        self.state == StageState::Passed
    }
}

/// The whole ladder for one version.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ValidationReport {
    pub sha256: String,
    pub bytes: u64,
    pub detected_format: DetectedFormat,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub claimed_format: Option<DetectedFormat>,
    pub file_created: Stage,
    pub format_reopened: Stage,
    pub content_checked: Stage,
    pub render_checked: Stage,
    pub accepted: Stage,
    /// Units, containers and citations the content reading found.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<ContentSummary>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub render: Option<RenderOutcome>,
    pub checked_at: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ContentSummary {
    pub units: usize,
    pub containers: usize,
    pub outline: Vec<String>,
    pub citations: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub gaps: Vec<String>,
}

impl ValidationReport {
    pub fn is_accepted(&self) -> bool {
        self.accepted.passed()
    }

    /// The five rungs, named, for display.
    pub fn rungs(&self) -> [(&'static str, &Stage); 5] {
        [
            ("file created", &self.file_created),
            ("format reopened", &self.format_reopened),
            ("content checked", &self.content_checked),
            ("render checked", &self.render_checked),
            ("accepted", &self.accepted),
        ]
    }

    /// One line per rung, for a model or a person.
    pub fn summary(&self) -> String {
        let mut out = String::new();
        for (name, stage) in self.rungs() {
            out.push_str(&format!("- {name}: {} — {}", stage.state.as_str(), stage.detail));
            if stage.validator != "—" {
                out.push_str(&format!(" [{}]", stage.validator));
            }
            out.push('\n');
            for problem in stage.problems.iter().take(12) {
                out.push_str(&format!("    · {problem}\n"));
            }
            if stage.problems.len() > 12 {
                out.push_str(&format!("    · … and {} more\n", stage.problems.len() - 12));
            }
        }
        out
    }
}

/// What the registry knows about a version, beyond its bytes.
#[derive(Debug, Clone, Default)]
pub struct Context<'a> {
    /// The hash the store recorded. The bytes must still hash to it.
    pub recorded_sha256: Option<&'a str>,
    pub claimed_mime: &'a str,
    pub filename: Option<&'a str>,
    /// The template id the version was produced from, if any.
    pub template: Option<&'a str>,
    /// Citation markers the registration bound to evidence. `None` when the
    /// version was recorded without an evidence binding, and then an unbound
    /// citation cannot be told from a bound one and is reported as unverified.
    pub bound_markers: Option<&'a BTreeSet<String>>,
    /// Why the version's dependencies are no longer current, if they are not.
    /// `None` when they were not checked.
    pub stale_because: Option<Vec<String>>,
}

/// How to render, when rendering is asked for.
pub struct RenderRequest<'a> {
    pub out_dir: &'a Path,
    pub from: u32,
    pub to: u32,
    pub inventory: &'a Inventory,
}

fn crate_validator(name: &str) -> String {
    format!("{name}@arjun-{}", env!("CARGO_PKG_VERSION"))
}

/// Climbs the ladder for `bytes`.
pub fn validate(bytes: &[u8], context: &Context<'_>, render_request: Option<RenderRequest<'_>>) -> ValidationReport {
    let sha256 = format!("{:x}", Sha256::digest(bytes));
    let detected = package::sniff(bytes);
    let claimed = DetectedFormat::claimed_by(context.claimed_mime, context.filename);

    // ── file created
    let file_created = if bytes.is_empty() {
        Stage::new(StageState::Failed, crate_validator("artifact-store"), "the file is empty", vec!["nothing was written into it".into()])
    } else if context.recorded_sha256.is_some_and(|recorded| recorded != sha256) {
        Stage::new(
            StageState::Failed,
            crate_validator("artifact-store"),
            "the bytes no longer hash to what was recorded",
            vec![format!("recorded {}, now {}", context.recorded_sha256.unwrap_or_default(), sha256)],
        )
    } else {
        Stage::new(StageState::Passed, crate_validator("artifact-store"), format!("{} bytes, sha-256 {}", bytes.len(), &sha256[..12]), Vec::new())
    };

    // ── format reopened
    let mut model: Option<ContentModel> = None;
    let format_reopened = if !file_created.passed() {
        Stage::not_run("the file was not established")
    } else {
        reopen(bytes, detected, claimed, &mut model)
    };

    // ── content checked
    let content_checked = match (&model, format_reopened.state) {
        (Some(model), StageState::Passed) => check_content(bytes, model, context),
        _ => Stage::not_run("the file did not reopen, so its content was not read"),
    };

    // ── render checked
    let (render_checked, rendered) = if !format_reopened.passed() {
        (Stage::not_run("the file did not reopen, so it was not handed to a renderer"), None)
    } else if detected == DetectedFormat::Text {
        (Stage::new(StageState::NotApplicable, "—", "plain text has no pages to lay out", Vec::new()), None)
    } else if let Some(request) = render_request {
        let outcome = render::render(bytes, detected, request.out_dir, request.from, request.to, request.inventory);
        (render_stage(&outcome, model.as_ref()), Some(outcome))
    } else {
        (Stage::not_run("rendering was not asked for"), None)
    };

    // ── accepted
    let mut reasons = Vec::new();
    let mut unavailable = false;
    for (name, stage) in [
        ("file created", &file_created),
        ("format reopened", &format_reopened),
        ("content checked", &content_checked),
        ("render checked", &render_checked),
    ] {
        match stage.state {
            StageState::Passed | StageState::NotApplicable => {}
            StageState::Unavailable => {
                unavailable = true;
                reasons.push(format!("{name} is unavailable here: {}", stage.detail));
            }
            StageState::Failed => reasons.push(format!("{name} failed")),
            StageState::NotRun => reasons.push(format!("{name} was not run")),
        }
    }
    match &context.stale_because {
        Some(stale) if !stale.is_empty() => {
            for why in stale {
                reasons.push(format!("stale: {why}"));
            }
        }
        Some(_) => {}
        None => reasons.push("its dependencies were not rechecked".to_string()),
    }
    let accepted = if reasons.is_empty() {
        Stage::new(
            StageState::Passed,
            crate_validator("acceptance"),
            "every machine check passed and everything it rests on is current; a person's approval is still separate",
            Vec::new(),
        )
    } else if unavailable && reasons.iter().all(|r| r.contains("unavailable")) {
        Stage::new(StageState::Unavailable, crate_validator("acceptance"), "not accepted: a check it needs cannot run on this machine", reasons)
    } else {
        Stage::new(StageState::Failed, crate_validator("acceptance"), "not accepted", reasons)
    };

    let content = model.as_ref().map(|m| ContentSummary {
        units: m.units.len(),
        containers: m.containers,
        outline: m.outline.iter().take(40).cloned().collect(),
        citations: m.citations().into_iter().map(|c| c.marker.clone()).collect(),
        gaps: m.gaps.clone(),
    });

    ValidationReport {
        sha256,
        bytes: bytes.len() as u64,
        detected_format: detected,
        claimed_format: claimed,
        file_created,
        format_reopened,
        content_checked,
        render_checked,
        accepted,
        content,
        render: rendered,
        checked_at: chrono::Utc::now().to_rfc3339(),
    }
}

fn reopen(bytes: &[u8], detected: DetectedFormat, claimed: Option<DetectedFormat>, model: &mut Option<ContentModel>) -> Stage {
    let validator = crate_validator("format-reader");
    // Bytes that begin as a ZIP and do not open as one: a damaged package,
    // reported as that rather than as bytes of no known kind.
    if detected == DetectedFormat::Unknown && bytes.starts_with(b"PK") {
        let report = package::inspect(bytes, &package::LIMITS);
        return Stage::new(
            StageState::Failed,
            validator,
            format!(
                "it does not reopen: it begins as a ZIP package{} and the package is damaged",
                claimed.map(|c| format!(" for a {}", c.label())).unwrap_or_default()
            ),
            report.problems,
        );
    }
    if let Some(claimed) = claimed {
        if claimed != detected {
            return Stage::new(
                StageState::Failed,
                validator,
                format!("it claims to be a {} and its bytes are: {}", claimed.label(), detected.label()),
                vec!["wrong type: the name or media type does not match the content".into()],
            );
        }
    }
    let mut problems = Vec::new();
    match detected {
        DetectedFormat::Docx | DetectedFormat::Pptx | DetectedFormat::Xlsx => {
            let report = package::inspect(bytes, &package::LIMITS);
            problems.extend(report.problems.iter().cloned());
            for part in &report.macros {
                problems.push(format!("it carries macros ({part}); a deliverable must not"));
            }
            let required: &[&str] = match detected {
                DetectedFormat::Docx => &["[Content_Types].xml", "_rels/.rels", "word/document.xml"],
                DetectedFormat::Pptx => &["[Content_Types].xml", "_rels/.rels", "ppt/presentation.xml"],
                _ => &["[Content_Types].xml", "_rels/.rels", "xl/workbook.xml"],
            };
            for part in required {
                if !report.parts.iter().any(|p| p == part) {
                    problems.push(format!("the package is missing {part}"));
                }
            }
            if problems.is_empty() {
                match content::extract(bytes) {
                    Ok(read) => {
                        if read.containers == 0 {
                            problems.push(match detected {
                                DetectedFormat::Pptx => "the presentation lists no slides".to_string(),
                                DetectedFormat::Xlsx => "the workbook lists no sheets".to_string(),
                                _ => "the document body has no paragraphs".to_string(),
                            });
                        }
                        *model = Some(read);
                    }
                    Err(error) => problems.push(error),
                }
            }
        }
        DetectedFormat::Pdf => {
            let check = super::pdf_validate::check_pdf_bytes(bytes);
            if !check.opens {
                // The built-in reader parses this product's own PDFs. Another
                // writer's may be perfectly sound; it is not established here.
                return Stage::new(
                    StageState::Unavailable,
                    validator,
                    "this PDF is outside the subset the built-in reader parses; the rasteriser reopens it during rendering",
                    check.problems,
                );
            }
            problems.extend(check.problems.iter().cloned());
            *model = content::extract(bytes).ok();
        }
        DetectedFormat::Svg => {
            let check = super::svg_validate::check_svg_text(&String::from_utf8_lossy(bytes));
            problems.extend(check.problems.iter().cloned());
            if check.parses {
                *model = content::extract(bytes).ok();
            }
        }
        DetectedFormat::Text => match content::extract(bytes) {
            Ok(read) => *model = Some(read),
            Err(error) => problems.push(error),
        },
        other => problems.push(format!("the bytes are a {}, which this does not reopen", other.label())),
    }
    if problems.is_empty() && model.is_some() {
        Stage::new(StageState::Passed, validator, format!("the bytes are a {} and it parses", detected.label()), Vec::new())
    } else {
        if problems.is_empty() {
            problems.push("it could not be read".into());
        }
        Stage::new(StageState::Failed, validator, format!("it does not reopen as a {}", detected.label()), problems)
    }
}

/// Error values a spreadsheet displays in place of a result.
const CELL_ERRORS: &[&str] = &["#REF!", "#DIV/0!", "#VALUE!", "#NAME?", "#N/A", "#NUM!", "#NULL!"];

fn check_content(bytes: &[u8], model: &ContentModel, context: &Context<'_>) -> Stage {
    let validator = crate_validator("content-checks");
    let mut problems = Vec::new();
    if model.units.is_empty() {
        problems.push("it says nothing: no text was found in it".to_string());
    }
    for unit in &model.units {
        if let Some(marker) = super::doc_model::placeholder_in(&unit.text) {
            problems.push(format!("{} still holds the placeholder {marker:?}", unit.locator));
        }
        if unit.formula.is_some() && CELL_ERRORS.iter().any(|e| unit.text.trim() == *e) {
            problems.push(format!("{} computes to {} (={})", unit.locator, unit.text.trim(), unit.formula.as_deref().unwrap_or_default()));
        }
    }

    // What the template promised.
    if let Some(template) = context.template {
        match template {
            "approval_note" => {
                let headings: Vec<String> = model.outline.iter().map(|h| h.trim().to_string()).collect();
                for field in super::docx::APPROVAL_NOTE.iter().filter(|f| f.required && !f.heading.is_empty()) {
                    if !headings.iter().any(|h| h == field.heading) {
                        problems.push(format!("the template promises a {:?} section and the file has none", field.heading));
                    }
                }
            }
            "briefing_deck" => {
                for section in super::pptx::BRIEFING_SECTIONS {
                    if !model.outline.iter().any(|t| t.eq_ignore_ascii_case(section)) {
                        problems.push(format!("the template promises a {section:?} slide and the deck has none"));
                    }
                }
            }
            "calculation_workbook" => {
                problems.extend(super::xlsx::check_workbook_bytes(bytes).problems);
            }
            _ => {}
        }
    }

    // Every citation it makes must be bound to evidence at registration.
    let citations = model.citations();
    let mut unverified = Vec::new();
    match context.bound_markers {
        Some(bound) => {
            for citation in &citations {
                if !bound.contains(&citation.marker) {
                    problems.push(format!(
                        "it cites {}, which this version's evidence does not bind to any source",
                        citation.marker
                    ));
                }
            }
        }
        None if !citations.is_empty() => unverified.push(format!(
            "{} citation(s) were not checked: this version was recorded without an evidence binding",
            citations.len()
        )),
        None => {}
    }

    if problems.is_empty() && unverified.is_empty() {
        Stage::new(
            StageState::Passed,
            validator,
            format!(
                "{} unit(s) of content, {} citation(s) each bound to evidence{}",
                model.units.len(),
                citations.len(),
                context.template.map(|t| format!(", and everything the {t} template promises")).unwrap_or_default()
            ),
            Vec::new(),
        )
    } else if problems.is_empty() {
        Stage::new(StageState::Failed, validator, "its citations could not be checked", unverified)
    } else {
        problems.extend(unverified);
        Stage::new(StageState::Failed, validator, "its content does not hold up", problems)
    }
}

fn render_stage(outcome: &RenderOutcome, model: Option<&ContentModel>) -> Stage {
    let validator = if outcome.renderers.is_empty() {
        "renderer".to_string()
    } else {
        outcome.renderers.iter().map(|r| format!("{}@{}", r.name, r.version)).collect::<Vec<_>>().join(" + ")
    };
    match outcome.state {
        RenderState::Rendered => {
            let mut problems = outcome.problems.clone();
            if let Some(model) = model.filter(|m| m.format == DetectedFormat::Pptx) {
                if outcome.total_pages as usize != model.containers {
                    problems.push(format!(
                        "the deck has {} slide(s) and laid out to {} page(s)",
                        model.containers, outcome.total_pages
                    ));
                }
            }
            if problems.is_empty() {
                Stage::new(StageState::Passed, validator, outcome.detail.clone(), Vec::new())
            } else {
                Stage::new(StageState::Failed, validator, outcome.detail.clone(), problems)
            }
        }
        RenderState::PdfOnly | RenderState::Unavailable => {
            Stage::new(StageState::Unavailable, validator, outcome.detail.clone(), outcome.problems.clone())
        }
        RenderState::Unsupported => Stage::new(StageState::NotApplicable, validator, outcome.detail.clone(), Vec::new()),
        RenderState::Refused | RenderState::Failed => {
            let mut problems = outcome.problems.clone();
            if problems.is_empty() {
                problems.push(outcome.detail.clone());
            }
            Stage::new(StageState::Failed, validator, outcome.detail.clone(), problems)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::artifacts::content::tests_support::sample_docx;

    const DOCX: &str = "application/vnd.openxmlformats-officedocument.wordprocessingml.document";

    fn bound(markers: &[&str]) -> BTreeSet<String> {
        markers.iter().map(|m| m.to_string()).collect()
    }

    #[test]
    fn a_sound_document_climbs_to_content_and_is_honest_about_the_render_it_was_not_given() {
        let dir = tempfile::tempdir().unwrap();
        let bytes = sample_docx(dir.path());
        let markers = bound(&["[E1]", "[M:mem-1@2]"]);
        let report = validate(
            &bytes,
            &Context {
                claimed_mime: DOCX,
                filename: Some("note.docx"),
                bound_markers: Some(&markers),
                stale_because: Some(Vec::new()),
                ..Default::default()
            },
            None,
        );
        assert!(report.file_created.passed());
        assert!(report.format_reopened.passed(), "{:?}", report.format_reopened);
        assert!(report.content_checked.passed(), "{:?}", report.content_checked);
        assert_eq!(report.render_checked.state, StageState::NotRun);
        assert_eq!(report.accepted.state, StageState::Failed, "unrendered is not accepted");
        assert!(report.accepted.problems.iter().any(|p| p.contains("render checked was not run")));
    }

    #[test]
    fn a_pdf_named_docx_is_a_wrong_type_failure() {
        let pdf = crate::artifacts::pdf::render(&crate::artifacts::pdf::PdfSpec {
            title: "Note".into(),
            classification: "Internal".into(),
            blocks: vec![crate::artifacts::pdf::Block::Paragraph("Enough words to be a real page of text for the checker.".into())],
        })
        .unwrap();
        let report = validate(&pdf, &Context { claimed_mime: DOCX, filename: Some("note.docx"), ..Default::default() }, None);
        assert_eq!(report.detected_format, DetectedFormat::Pdf);
        assert_eq!(report.format_reopened.state, StageState::Failed);
        assert!(report.format_reopened.detail.contains("claims to be a Word document"), "{:?}", report.format_reopened);
        assert_eq!(report.content_checked.state, StageState::NotRun);
        assert!(!report.is_accepted());
    }

    #[test]
    fn a_corrupted_package_fails_to_reopen_and_nothing_above_it_runs() {
        let dir = tempfile::tempdir().unwrap();
        let mut bytes = sample_docx(dir.path());
        let cut = bytes.len() * 2 / 3;
        bytes.truncate(cut);
        let report = validate(&bytes, &Context { claimed_mime: DOCX, filename: Some("note.docx"), ..Default::default() }, None);
        assert_ne!(report.format_reopened.state, StageState::Passed);
        assert_eq!(report.content_checked.state, StageState::NotRun);
        assert!(!report.is_accepted());
    }

    #[test]
    fn bytes_that_no_longer_match_their_recorded_hash_fail_at_the_first_rung() {
        let report = validate(b"changed", &Context { recorded_sha256: Some("0".repeat(64).as_str()), claimed_mime: "text/plain", ..Default::default() }, None);
        assert_eq!(report.file_created.state, StageState::Failed);
        assert_eq!(report.format_reopened.state, StageState::NotRun);
    }

    #[test]
    fn an_unbound_citation_and_a_missing_template_section_fail_the_content_rung() {
        let dir = tempfile::tempdir().unwrap();
        let bytes = sample_docx(dir.path());
        let only_one = bound(&["[E1]"]);
        let report = validate(
            &bytes,
            &Context { claimed_mime: DOCX, template: Some("approval_note"), bound_markers: Some(&only_one), ..Default::default() },
            None,
        );
        assert_eq!(report.content_checked.state, StageState::Failed);
        let problems = report.content_checked.problems.join(" | ");
        assert!(problems.contains("[M:mem-1@2]"), "{problems}");
        assert!(problems.contains("\"Recipient\"") || problems.contains("\"To\""), "{problems}");
    }

    #[test]
    fn stale_dependencies_withhold_acceptance_even_when_every_check_passes() {
        let report = validate(
            b"plain notes that say something",
            &Context {
                claimed_mime: "text/plain",
                filename: Some("notes.txt"),
                bound_markers: Some(&BTreeSet::new()),
                stale_because: Some(vec!["memory item mem-1 moved from revision 2 to 3".into()]),
                ..Default::default()
            },
            None,
        );
        assert!(report.content_checked.passed(), "{:?}", report.content_checked);
        assert_eq!(report.render_checked.state, StageState::NotApplicable);
        assert_eq!(report.accepted.state, StageState::Failed);
        assert!(report.accepted.problems.iter().any(|p| p.contains("mem-1")));

        let current = validate(
            b"plain notes that say something",
            &Context { claimed_mime: "text/plain", bound_markers: Some(&BTreeSet::new()), stale_because: Some(Vec::new()), ..Default::default() },
            None,
        );
        assert!(current.is_accepted(), "{}", current.summary());
    }

    #[test]
    fn an_unavailable_renderer_leaves_acceptance_unavailable_not_passed() {
        let dir = tempfile::tempdir().unwrap();
        let bytes = sample_docx(dir.path());
        let markers = bound(&["[E1]", "[M:mem-1@2]"]);
        let none = Inventory {
            office: serde_json::from_value(serde_json::json!({
                "name": "LibreOffice", "available": false, "qualified": false, "qualifiedSeries": ["24.2"],
                "unavailableBecause": "not on this machine", "licence": "MPL-2.0", "provisioning": "external"
            })).unwrap(),
            rasteriser: serde_json::from_value(serde_json::json!({
                "name": "PyMuPDF", "available": false, "qualified": false, "qualifiedSeries": ["1.28"],
                "unavailableBecause": "not on this machine", "licence": "AGPL-3.0", "provisioning": "bundled"
            })).unwrap(),
        };
        let report = validate(
            &bytes,
            &Context { claimed_mime: DOCX, bound_markers: Some(&markers), stale_because: Some(Vec::new()), ..Default::default() },
            Some(RenderRequest { out_dir: &dir.path().join("r"), from: 1, to: 3, inventory: &none }),
        );
        assert_eq!(report.render_checked.state, StageState::Unavailable, "{:?}", report.render_checked);
        assert_eq!(report.accepted.state, StageState::Unavailable, "{}", report.summary());
    }
}
