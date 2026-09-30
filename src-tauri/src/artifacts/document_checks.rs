//! What a Word document is checked for, from the file and from its pages
//! (P09, and the reviewer's document checks in P10).
//!
//! Two kinds of check, reported apart because they fail apart:
//!
//! - **format checks** read the package: it reopens, the mandatory sections
//!   are there and written, every claim in a section that must cite does,
//!   every marker resolves to something current, every figure is stated by
//!   what its sentence cites, long tables repeat their header, no
//!   placeholder is left and the document still says it is a draft;
//! - **render checks** read the laid-out pages: every page was drawn and is
//!   not blank, every piece of text is on a page and inside it, a long
//!   table's header is on every page it runs onto, and text after a page
//!   break starts on a later page.
//!
//! A render that could not happen makes the render checks *unavailable*,
//! never passed: a document nobody has looked at is not a document that
//! looks right.

use serde::{Deserialize, Serialize};

use super::authoring::{read_sections, AuthoringTemplate, CitationResolver, CitationRule};
use super::content::{citations_in, CitationTarget};
use super::doc_model::placeholder_in;
use super::package::{self, DetectedFormat};
use super::render::{RenderOutcome, RenderState};
use super::xml_events::{self, Event};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum CheckStatus {
    Pass,
    Fail,
    /// The check could not be run; never counted as a pass.
    Unavailable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum CheckSeverity {
    Minor,
    Major,
    Blocking,
}

/// One check, as attempted.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CheckResult {
    pub check: String,
    pub status: CheckStatus,
    pub severity: CheckSeverity,
    /// Where: `section:findings`, `p:12`, `page:3`, `table:1 row:14`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub locations: Vec<String>,
    pub detail: String,
}

impl CheckResult {
    fn pass(check: &str, severity: CheckSeverity, detail: impl Into<String>) -> Self {
        CheckResult { check: check.into(), status: CheckStatus::Pass, severity, locations: Vec::new(), detail: detail.into() }
    }

    fn fail(check: &str, severity: CheckSeverity, locations: Vec<String>, detail: impl Into<String>) -> Self {
        CheckResult { check: check.into(), status: CheckStatus::Fail, severity, locations, detail: detail.into() }
    }

    fn unavailable(check: &str, severity: CheckSeverity, detail: impl Into<String>) -> Self {
        CheckResult { check: check.into(), status: CheckStatus::Unavailable, severity, locations: Vec::new(), detail: detail.into() }
    }

    fn from_problems(check: &str, severity: CheckSeverity, problems: Vec<(String, String)>, passed: impl Into<String>) -> Self {
        if problems.is_empty() {
            return Self::pass(check, severity, passed);
        }
        let mut locations: Vec<String> = Vec::new();
        for (location, _) in &problems {
            if !locations.contains(location) {
                locations.push(location.clone());
            }
        }
        let detail = problems.iter().take(12).map(|(l, d)| format!("{l}: {d}")).collect::<Vec<_>>().join("; ");
        let more = if problems.len() > 12 { format!(" (and {} more)", problems.len() - 12) } else { String::new() };
        Self::fail(check, severity, locations, format!("{detail}{more}"))
    }
}

/// A table with more data rows than this has to repeat its header.
pub const LONG_TABLE_ROWS: usize = 10;

// ── The body, in order ──────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum BodyKind {
    Heading,
    Caption,
    Paragraph,
    Row,
    PageBreak,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BodyUnit {
    pub index: usize,
    pub section: Option<String>,
    pub kind: BodyKind,
    pub text: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub cells: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub table: Option<usize>,
    pub header_row: bool,
}

impl BodyUnit {
    pub fn location(&self) -> String {
        let place = match (self.kind, self.table) {
            (BodyKind::Row, Some(table)) => format!("table:{table} unit:{}", self.index),
            _ => format!("unit:{}", self.index),
        };
        match &self.section {
            Some(section) => format!("section:{section} {place}"),
            None => place,
        }
    }
}

/// Every paragraph, table row and page break of `word/document.xml`, with
/// the section each falls in.
pub fn body_units(document_xml: &str) -> Result<Vec<BodyUnit>, String> {
    let spans = xml_events::scan_spans(document_xml)?;
    let mut units: Vec<BodyUnit> = Vec::new();
    let mut section: Option<(String, String)> = None; // (id, bookmark id)
    let mut table_depth = 0usize;
    let mut tables = 0usize;
    let mut in_text = false;
    let mut paragraph = String::new();
    let mut paragraph_style = String::new();
    let mut cell = String::new();
    let mut cells: Vec<String> = Vec::new();
    let mut header_row = false;
    let mut push = |units: &mut Vec<BodyUnit>, section: &Option<(String, String)>, kind: BodyKind, text: String, cells: Vec<String>, table: Option<usize>, header_row: bool| {
        let index = units.len() + 1;
        units.push(BodyUnit { index, section: section.as_ref().map(|(id, _)| id.clone()), kind, text, cells, table, header_row });
    };
    for spanned in &spans {
        match &spanned.event {
            Event::Start { name, empty, .. } => match xml_events::local(name) {
                "bookmarkStart" => {
                    if let Some(id) = spanned.event.attribute("w:name").and_then(|n| n.strip_prefix(super::authoring::BOOKMARK_PREFIX)) {
                        section = Some((id.to_string(), spanned.event.attribute("w:id").unwrap_or_default().to_string()));
                    }
                }
                "bookmarkEnd" => {
                    if section.as_ref().is_some_and(|(_, bid)| Some(bid.as_str()) == spanned.event.attribute("w:id")) {
                        section = None;
                    }
                }
                "tbl" => {
                    table_depth += 1;
                    if table_depth == 1 {
                        tables += 1;
                    }
                }
                "tr" if table_depth == 1 => {
                    cells.clear();
                    header_row = false;
                }
                "tblHeader" if table_depth == 1 => header_row = true,
                "tc" if table_depth == 1 => cell.clear(),
                "p" if !*empty => {
                    paragraph.clear();
                    paragraph_style.clear();
                }
                "pStyle" => paragraph_style = spanned.event.attribute("w:val").unwrap_or_default().to_string(),
                "t" if !*empty => in_text = true,
                "tab" => paragraph.push(' '),
                "br" if spanned.event.attribute("w:type") == Some("page") => {
                    let before = std::mem::take(&mut paragraph);
                    if !before.trim().is_empty() && table_depth == 0 {
                        push(&mut units, &section, BodyKind::Paragraph, before, Vec::new(), None, false);
                    }
                    push(&mut units, &section, BodyKind::PageBreak, String::new(), Vec::new(), None, false);
                }
                _ => {}
            },
            Event::End { name } => match xml_events::local(name) {
                "t" => in_text = false,
                "p" => {
                    let text = std::mem::take(&mut paragraph);
                    if table_depth > 0 {
                        if !cell.is_empty() && !text.is_empty() {
                            cell.push(' ');
                        }
                        cell.push_str(&text);
                    } else if !text.trim().is_empty() {
                        let kind = if paragraph_style.starts_with("Heading") || paragraph_style == "Title" {
                            BodyKind::Heading
                        } else if paragraph_style == "Caption" {
                            BodyKind::Caption
                        } else {
                            BodyKind::Paragraph
                        };
                        push(&mut units, &section, kind, text, Vec::new(), None, false);
                    }
                    paragraph_style.clear();
                }
                "tc" if table_depth == 1 => cells.push(std::mem::take(&mut cell)),
                "tr" if table_depth == 1 => {
                    let row = std::mem::take(&mut cells);
                    push(&mut units, &section, BodyKind::Row, row.join(" | "), row, Some(tables), header_row);
                }
                "tbl" => table_depth = table_depth.saturating_sub(1),
                _ => {}
            },
            Event::Text(text) => {
                if in_text {
                    paragraph.push_str(text);
                }
            }
        }
    }
    Ok(units)
}

// ── Format checks ───────────────────────────────────────────────────────

/// Everything checked from the file itself.
pub fn format_checks(bytes: &[u8], template: Option<&AuthoringTemplate>, resolver: Option<&dyn CitationResolver>) -> Vec<CheckResult> {
    let mut out = Vec::new();
    let report = package::inspect(bytes, &package::LIMITS);
    if !report.safe_to_open() || report.detected != DetectedFormat::Docx {
        out.push(CheckResult::fail(
            "document.reopen",
            CheckSeverity::Blocking,
            vec!["package".into()],
            if report.safe_to_open() { format!("the file is {}, not a Word document", report.detected.label()) } else { report.problems.join("; ") },
        ));
        return out;
    }
    let xml = match package::read_part(bytes, "word/document.xml") {
        Ok(xml) => xml,
        Err(why) => {
            out.push(CheckResult::fail("document.reopen", CheckSeverity::Blocking, vec!["word/document.xml".into()], why));
            return out;
        }
    };
    let (sections, units) = match (read_sections(&xml), body_units(&xml)) {
        (Ok(sections), Ok(units)) => (sections, units),
        (Err(why), _) | (_, Err(why)) => {
            out.push(CheckResult::fail("document.reopen", CheckSeverity::Blocking, vec!["word/document.xml".into()], format!("the body does not parse: {why}")));
            return out;
        }
    };
    out.push(CheckResult::pass(
        "document.reopen",
        CheckSeverity::Blocking,
        format!("reopened: {} parts, {} body units, {} marked sections", report.parts.len(), units.len(), sections.len()),
    ));

    // Mandatory sections, written.
    let mut problems: Vec<(String, String)> = Vec::new();
    match template {
        Some(template) => {
            for req in template.sections.iter().filter(|s| s.required) {
                match sections.iter().find(|s| s.id == req.id) {
                    None => problems.push((format!("section:{}", req.id), format!("the mandatory section {:?} is missing", req.heading))),
                    Some(found) => {
                        let body: usize = units.iter().filter(|u| u.section.as_deref() == Some(req.id) && u.kind != BodyKind::Heading).map(|u| u.text.trim().chars().count()).sum();
                        if body == 0 {
                            problems.push((format!("section:{}", req.id), "the section has a heading and nothing under it".into()));
                        } else if found.gap {
                            problems.push((format!("section:{}", req.id), "the section states that information is needed and is not written".into()));
                        }
                    }
                }
            }
            out.push(CheckResult::from_problems("document.sections", CheckSeverity::Blocking, problems, format!("all mandatory sections of {} are present and written", template.key())));
        }
        None => out.push(CheckResult::unavailable("document.sections", CheckSeverity::Blocking, "no authoring template is recorded for this version, so which sections are mandatory is not known here")),
    }

    // Stated gaps.
    let gaps: Vec<(String, String)> = units
        .iter()
        .filter(|u| u.text.contains("INFORMATION NEEDED"))
        .map(|u| (u.location(), u.text.chars().take(140).collect()))
        .collect();
    out.push(CheckResult::from_problems("document.gaps", CheckSeverity::Blocking, gaps, "no section states missing information"));

    // Citations present where the template requires them.
    match template {
        Some(template) => {
            let mut problems = Vec::new();
            let mut caption: Vec<super::content::Citation> = Vec::new();
            for unit in &units {
                if unit.kind == BodyKind::Caption {
                    caption = citations_in(&unit.text);
                    continue;
                }
                if unit.kind != BodyKind::Row {
                    caption.clear();
                }
                let Some(section) = unit.section.as_deref() else { continue };
                let rule = template.sections.iter().find(|s| s.id == section).map(|s| s.citations).unwrap_or(CitationRule::Optional);
                if matches!(unit.kind, BodyKind::Heading | BodyKind::PageBreak) || unit.header_row || unit.text.contains("INFORMATION NEEDED") {
                    continue;
                }
                let mut cited = citations_in(&unit.text);
                if unit.kind == BodyKind::Row {
                    cited.extend(caption.iter().cloned());
                }
                let excerpt: String = unit.text.chars().take(90).collect();
                match rule {
                    CitationRule::Required if cited.is_empty() => problems.push((unit.location(), format!("cites nothing: {excerpt:?}"))),
                    CitationRule::Calculation if !cited.iter().any(|c| matches!(c.target, CitationTarget::Calculation { .. })) => {
                        problems.push((unit.location(), format!("states a calculation without its record: {excerpt:?}")))
                    }
                    _ => {}
                }
            }
            out.push(CheckResult::from_problems("document.citations_present", CheckSeverity::Blocking, problems, "every claim in a section that must cite does"));
        }
        None => out.push(CheckResult::unavailable("document.citations_present", CheckSeverity::Blocking, "no authoring template is recorded, so which sections must cite is not known here")),
    }

    // Citations resolve and are current; figures are stated by them.
    match resolver {
        Some(resolver) => {
            let mut unresolved = Vec::new();
            let mut figures = Vec::new();
            for unit in units.iter().filter(|u| !matches!(u.kind, BodyKind::PageBreak) && !u.header_row) {
                if unit.section.as_deref() == Some("references") || unit.section.as_deref() == Some("provenance") {
                    continue;
                }
                let cited = citations_in(&unit.text);
                let mut texts = Vec::new();
                for citation in &cited {
                    match resolver.resolve(&citation.target, &citation.marker) {
                        None => unresolved.push((unit.location(), format!("{} resolves to nothing this run can read", citation.marker))),
                        Some(found) => {
                            if let Some(why) = found.stale_because {
                                unresolved.push((unit.location(), format!("{} no longer stands: {why}", citation.marker)));
                            }
                            texts.push(found.text);
                        }
                    }
                }
                if matches!(unit.kind, BodyKind::Heading | BodyKind::Caption) && cited.is_empty() {
                    continue;
                }
                for (number, unit_text) in super::authoring::figures_in(&unit.text) {
                    if !texts.iter().any(|t| crate::calculation::check::text_states(t, &number, &unit_text)) {
                        figures.push((
                            unit.location(),
                            format!("states {number} {unit_text}, which {}", if cited.is_empty() { "cites nothing" } else { "none of its citations states" }),
                        ));
                    }
                }
            }
            out.push(CheckResult::from_problems("document.citations_resolve", CheckSeverity::Blocking, unresolved, "every citation resolves to something current"));
            out.push(CheckResult::from_problems("document.figures_supported", CheckSeverity::Blocking, figures, "every figure with a unit is stated by what its sentence cites"));
        }
        None => {
            out.push(CheckResult::unavailable("document.citations_resolve", CheckSeverity::Blocking, "no source manifest was given to resolve citations against"));
            out.push(CheckResult::unavailable("document.figures_supported", CheckSeverity::Blocking, "no source manifest was given to hold figures to"));
        }
    }

    // Long tables repeat their header.
    let mut problems = Vec::new();
    let mut table_rows: std::collections::BTreeMap<usize, (usize, bool, String)> = Default::default();
    for unit in units.iter().filter(|u| u.kind == BodyKind::Row) {
        let table = unit.table.unwrap_or(0);
        let entry = table_rows.entry(table).or_insert((0, false, unit.location()));
        if unit.header_row {
            entry.1 = true;
        } else {
            entry.0 += 1;
        }
    }
    for (table, (rows, header, location)) in &table_rows {
        if *rows > LONG_TABLE_ROWS && !header {
            problems.push((format!("table:{table}"), format!("{rows} rows and no header row marked to repeat on each page ({location})")));
        }
    }
    out.push(CheckResult::from_problems("document.long_tables", CheckSeverity::Major, problems, format!("{} table(s); every one longer than {LONG_TABLE_ROWS} rows repeats its header", table_rows.len())));

    // Placeholders.
    let placeholders: Vec<(String, String)> = units.iter().filter_map(|u| placeholder_in(&u.text).map(|m| (u.location(), format!("holds {m:?}")))).collect();
    out.push(CheckResult::from_problems("document.placeholders", CheckSeverity::Blocking, placeholders, "no placeholder text"));

    // The product policy: a generated document is not approved until a
    // person approves it, so it never says it is. The authoring writer also
    // says DRAFT on every page; the legacy writer drops the word once its
    // verifier passes, and still claims no approval.
    let furniture_text = report
        .parts
        .iter()
        .filter(|p| p.starts_with("word/footer") || p.starts_with("word/header"))
        .filter_map(|p| package::read_part(bytes, p).ok())
        .collect::<String>();
    let says_draft = units.iter().take(3).any(|u| u.text.contains("DRAFT")) || furniture_text.contains("DRAFT");
    let claims: Vec<(String, String)> = units
        .iter()
        .filter(|u| {
            let lower = u.text.to_lowercase();
            ["approved by", "status: approved", "this note is approved", "approval granted"].iter().any(|c| lower.contains(c))
                && !lower.contains("not approved")
        })
        .map(|u| (u.location(), format!("claims approval: {:?}", u.text.chars().take(90).collect::<String>())))
        .collect();
    out.push(CheckResult::from_problems(
        "document.approval_status",
        CheckSeverity::Blocking,
        claims,
        if says_draft { "it says it is a draft, not approved" } else { "it claims no approval" },
    ));
    out
}

// ── Render checks ───────────────────────────────────────────────────────

/// Text as compared across a file and its rendering: no whitespace,
/// ligatures expanded, lower case.
fn normalise(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            '\u{FB00}' => out.push_str("ff"),
            '\u{FB01}' => out.push_str("fi"),
            '\u{FB02}' => out.push_str("fl"),
            '\u{FB03}' => out.push_str("ffi"),
            '\u{FB04}' => out.push_str("ffl"),
            '\u{2011}' | '\u{2010}' => out.push('-'),
            '\u{00AD}' => {}
            c if c.is_whitespace() => {}
            c => out.extend(c.to_lowercase()),
        }
    }
    out
}

/// The header and footer lines, with field results (page numbers) taken out,
/// so they can be removed from each page's text before a paragraph that
/// runs across a page is looked for.
fn furniture(bytes: &[u8]) -> Vec<String> {
    let parts = package::inspect(bytes, &package::LIMITS).parts;
    let mut out = Vec::new();
    for part in parts.iter().filter(|p| (p.starts_with("word/header") || p.starts_with("word/footer")) && p.ends_with(".xml")) {
        let Ok(xml) = package::read_part(bytes, part) else { continue };
        let Ok(units) = body_units(&xml) else { continue };
        for unit in units {
            let line: String = normalise(&unit.text).chars().filter(|c| !c.is_ascii_digit()).collect();
            if !line.is_empty() {
                out.push(line);
            }
        }
    }
    out
}

fn page_body(text: &str, furniture: &[String]) -> String {
    text.lines()
        .filter(|line| {
            let digits_out: String = normalise(line).chars().filter(|c| !c.is_ascii_digit()).collect();
            !furniture.iter().any(|f| !digits_out.is_empty() && (f == &digits_out || f.contains(&digits_out) && digits_out.len() > 6))
        })
        .map(normalise)
        .collect()
}

/// Everything checked from the laid-out pages.
pub fn render_checks(bytes: &[u8], render: &RenderOutcome) -> Vec<CheckResult> {
    let names = ["render.pages", "render.text_visible", "render.table_headers", "render.page_breaks"];
    if render.state != RenderState::Rendered {
        let why = format!("the document was not rendered ({:?}): {}", render.state, render.detail);
        return names.iter().map(|n| CheckResult::unavailable(n, CheckSeverity::Blocking, why.clone())).collect();
    }
    let Ok(xml) = package::read_part(bytes, "word/document.xml") else {
        return names.iter().map(|n| CheckResult::unavailable(n, CheckSeverity::Blocking, "the body could not be read")).collect();
    };
    let Ok(units) = body_units(&xml) else {
        return names.iter().map(|n| CheckResult::unavailable(n, CheckSeverity::Blocking, "the body could not be parsed")).collect();
    };
    let mut out = Vec::new();

    // Every page drawn, none blank.
    let mut problems = Vec::new();
    if (render.pages.len() as u32) < render.total_pages {
        problems.push((
            "pages".to_string(),
            format!("{} of {} pages were rasterised; the rest were not looked at", render.pages.len(), render.total_pages),
        ));
    }
    for page in render.pages.iter().filter(|p| p.blank) {
        problems.push((format!("page:{}", page.page), "renders blank".into()));
    }
    out.push(CheckResult::from_problems("render.pages", CheckSeverity::Major, problems, format!("all {} pages rendered, none blank", render.total_pages)));

    // Every unit on a page, in order; nothing drawn off the page.
    let furniture = furniture(bytes);
    let bodies: Vec<String> = render.pages.iter().map(|p| page_body(&p.text, &furniture)).collect();
    let whole: Vec<String> = render.pages.iter().map(|p| normalise(&p.text)).collect();
    let mut page_of: Vec<Option<usize>> = vec![None; units.len()];
    let mut cursor = 0usize;
    let mut missing = Vec::new();
    for (index, unit) in units.iter().enumerate() {
        if unit.kind == BodyKind::PageBreak {
            continue;
        }
        let needles: Vec<String> = if unit.kind == BodyKind::Row { unit.cells.iter().map(|c| normalise(c)).filter(|c| !c.is_empty()).collect() } else { vec![normalise(&unit.text)] };
        if needles.is_empty() {
            continue;
        }
        let on = |page: usize| needles.iter().all(|n| whole[page].contains(n.as_str()) || bodies[page].contains(n.as_str()));
        let spans_to = |page: usize| page + 1 < bodies.len() && needles.iter().all(|n| format!("{}{}", bodies[page], bodies[page + 1]).contains(n.as_str()));
        // On one page, from where the previous unit was; then anywhere; only
        // then run across a page boundary (a paragraph the page split).
        let found = (cursor..whole.len())
            .find(|&p| on(p))
            .or_else(|| (0..whole.len()).find(|&p| on(p)))
            .or_else(|| (cursor..whole.len()).find(|&p| spans_to(p)))
            .or_else(|| (0..whole.len()).find(|&p| spans_to(p)));
        match found {
            Some(page) => {
                page_of[index] = Some(page);
                cursor = cursor.max(page);
            }
            None => missing.push((unit.location(), format!("not on any rendered page: {:?}", unit.text.chars().take(80).collect::<String>()))),
        }
    }
    for page in &render.pages {
        for clipped in &page.clipped {
            missing.push((format!("page:{}", page.page), format!("drawn outside the page: {clipped:?}")));
        }
    }
    out.push(CheckResult::from_problems(
        "render.text_visible",
        CheckSeverity::Blocking,
        missing,
        format!("all {} text units are on a page and inside it", units.iter().filter(|u| u.kind != BodyKind::PageBreak).count()),
    ));

    // A long table's header on every page it runs onto.
    let mut problems = Vec::new();
    let mut tables: std::collections::BTreeMap<usize, (Option<&BodyUnit>, Vec<usize>)> = Default::default();
    for (index, unit) in units.iter().enumerate().filter(|(_, u)| u.kind == BodyKind::Row) {
        let entry = tables.entry(unit.table.unwrap_or(0)).or_insert((None, Vec::new()));
        if unit.header_row && entry.0.is_none() {
            entry.0 = Some(unit);
        } else if let Some(page) = page_of[index] {
            if !entry.1.contains(&page) {
                entry.1.push(page);
            }
        }
    }
    let mut spanning = 0;
    for (table, (header, pages)) in &tables {
        if pages.len() < 2 {
            continue;
        }
        spanning += 1;
        let Some(header) = header else {
            problems.push((format!("table:{table}"), format!("runs over pages {} with no header row", pages.iter().map(|p| (p + 1).to_string()).collect::<Vec<_>>().join(", "))));
            continue;
        };
        for page in pages {
            let shows = header.cells.iter().map(|c| normalise(c)).filter(|c| !c.is_empty()).all(|c| whole[*page].contains(c.as_str()));
            if !shows {
                problems.push((format!("table:{table} page:{}", page + 1), "the header row is not repeated on this page".into()));
            }
        }
    }
    out.push(CheckResult::from_problems("render.table_headers", CheckSeverity::Major, problems, format!("{spanning} table(s) run over pages; each repeats its header")));

    // Page breaks land.
    let mut problems = Vec::new();
    let mut breaks = 0;
    for (index, unit) in units.iter().enumerate().filter(|(_, u)| u.kind == BodyKind::PageBreak) {
        breaks += 1;
        let before = (0..index).rev().find_map(|i| page_of[i]);
        let after = (index + 1..units.len()).find_map(|i| page_of[i]);
        if let (Some(before), Some(after)) = (before, after) {
            if after <= before {
                problems.push((unit.location(), format!("text after the page break is on page {}, the same page as the text before it", after + 1)));
            }
        }
    }
    out.push(CheckResult::from_problems("render.page_breaks", CheckSeverity::Minor, problems, format!("{breaks} page break(s); text after each starts on a later page")));
    out
}

/// Whether a set of results lets a version through: nothing blocking or
/// major failed, and nothing blocking was left unrun.
pub fn accepted(results: &[CheckResult]) -> bool {
    results.iter().all(|r| match r.status {
        CheckStatus::Pass => true,
        CheckStatus::Fail => r.severity == CheckSeverity::Minor,
        CheckStatus::Unavailable => r.severity != CheckSeverity::Blocking,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::artifacts::authoring::{write_docx, AuthoredDocument, AuthoredSection, Stamp, INSPECTION_APPROVAL_NOTE};
    use crate::artifacts::doc_model::Block;

    fn section(id: &str, heading: &str, blocks: Vec<Block>) -> AuthoredSection {
        AuthoredSection { id: id.into(), heading: heading.into(), level: 1, blocks, gap: None }
    }

    pub(crate) fn note(findings: Vec<Block>) -> AuthoredDocument {
        AuthoredDocument {
            template: "inspection_approval_note@1".into(),
            template_sha256: INSPECTION_APPROVAL_NOTE.sha256(),
            title: "V-101 wall loss".into(),
            audience: "Plant manager".into(),
            classification: "Internal".into(),
            fields: vec![("To".into(), "Plant Manager".into())],
            sections: vec![
                section("subject", "Subject", vec![Block::Paragraph { text: "Shell thickness at point C of V-101.".into() }]),
                section("findings", "Findings", findings),
                section("recommendation", "Recommendation", vec![Block::Paragraph { text: "Re-survey point C before restart.".into() }]),
                section("assumptions", "Assumptions", vec![Block::Paragraph { text: "The instrument was in calibration.".into() }]),
                section("references", "Supporting references", vec![Block::Bullets { items: vec!["[E1] UT survey".into()] }]),
            ],
            stamp: Stamp { task_id: "r".into(), model: "fixture".into(), created_at: "2026-09-25T10:00:00Z".into() },
        }
    }

    fn status(results: &[CheckResult], check: &str) -> CheckStatus {
        results.iter().find(|r| r.check == check).unwrap_or_else(|| panic!("{check} missing: {results:?}")).status
    }

    #[test]
    fn a_sound_note_passes_its_format_checks_and_leaves_resolution_unavailable() {
        let bytes = write_docx(&note(vec![Block::Paragraph { text: "Point C reads 8.2 mm [E1].".into() }])).unwrap();
        let results = format_checks(&bytes, Some(&INSPECTION_APPROVAL_NOTE), None);
        for check in ["document.reopen", "document.sections", "document.gaps", "document.citations_present", "document.long_tables", "document.placeholders", "document.approval_status"] {
            assert_eq!(status(&results, check), CheckStatus::Pass, "{check}: {results:?}");
        }
        assert_eq!(status(&results, "document.citations_resolve"), CheckStatus::Unavailable);
        assert!(!accepted(&results), "an unrun blocking check is not acceptance");
    }

    #[test]
    fn an_uncited_claim_a_gap_and_an_unmarked_long_table_are_found() {
        let mut doc = note(vec![Block::Paragraph { text: "The shell is sound.".into() }]);
        doc.sections[2].gap = Some("no decision recorded".into());
        doc.sections[2].blocks.clear();
        let bytes = write_docx(&doc).unwrap();
        let results = format_checks(&bytes, Some(&INSPECTION_APPROVAL_NOTE), None);
        assert_eq!(status(&results, "document.citations_present"), CheckStatus::Fail);
        assert_eq!(status(&results, "document.gaps"), CheckStatus::Fail);
        assert_eq!(status(&results, "document.sections"), CheckStatus::Fail);
        let xml = package::read_part(&bytes, "word/document.xml").unwrap();
        let units = body_units(&xml).unwrap();
        assert!(units.iter().any(|u| u.section.as_deref() == Some("findings") && u.text == "The shell is sound."));

        let rows: Vec<Vec<String>> = (1..=30).map(|i| vec![format!("P{i}"), format!("{}.0 mm [E1]", i)]).collect();
        let long = note(vec![Block::Table { header: vec!["Point".into(), "Reading".into()], rows, caption: None }]);
        let bytes = write_docx(&long).unwrap();
        let xml = package::read_part(&bytes, "word/document.xml").unwrap().replace("<w:tblHeader/>", "");
        let mut archive = zip::ZipArchive::new(std::io::Cursor::new(bytes.as_slice())).unwrap();
        let (stripped, _) = crate::artifacts::edit::rewrite_parts(&mut archive, &[("word/document.xml".into(), xml)]).unwrap();
        assert_eq!(status(&format_checks(&stripped, Some(&INSPECTION_APPROVAL_NOTE), None), "document.long_tables"), CheckStatus::Fail);
        assert_eq!(status(&format_checks(&bytes, Some(&INSPECTION_APPROVAL_NOTE), None), "document.long_tables"), CheckStatus::Pass);
    }

    /// A real layout when this machine has the renderers: a 60-row table
    /// runs over pages and repeats its header, a page break lands, every
    /// unit is on a page. Without them the checks are unavailable, which the
    /// assertion says rather than skipping.
    #[test]
    fn a_long_note_renders_with_its_header_repeated_and_its_break_landing() {
        let rows: Vec<Vec<String>> = (1..=60).map(|i| vec![format!("TML-{i:03}"), format!("{}.{} mm [E1]", 8 + i % 3, i % 10)]).collect();
        let mut doc = note(vec![
            Block::Paragraph { text: "Readings at every monitoring location follow [E1].".into() },
            Block::Table { header: vec!["Location".into(), "Reading".into()], rows, caption: Some("Table 1 - UT readings [E1]".into()) },
            Block::PageBreak,
            Block::Paragraph { text: "After the table, the governing reading is at TML-017 [E1].".into() },
        ]);
        doc.title = "V-101 thickness survey".into();
        let bytes = write_docx(&doc).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let inventory = crate::artifacts::render::inventory();
        let outcome = crate::artifacts::render::render(&bytes, DetectedFormat::Docx, dir.path(), 1, 50, inventory);
        let results = render_checks(&bytes, &outcome);
        eprintln!("P09-RENDER state={:?} pages={} renderers={:?}", outcome.state, outcome.total_pages, outcome.renderers);
        for page in &outcome.pages {
            eprintln!("P09-RENDER page {} fonts={:?} clipped={}", page.page, page.fonts, page.clipped.len());
        }
        for result in &results {
            eprintln!("P09-RENDER {} {:?} {}", result.check, result.status, result.detail);
        }
        if inventory.office.usable() && inventory.rasteriser.usable() {
            assert_eq!(outcome.state, RenderState::Rendered);
            assert!(outcome.total_pages >= 3, "a 60-row table and a break run over pages");
            assert!(results.iter().all(|r| r.status == CheckStatus::Pass), "{results:?}");
            assert!(outcome.pages.iter().all(|p| !p.fonts.is_empty()), "fonts are recorded per page");
            assert!(outcome.font_files.iter().all(|f| f.version.is_some() && !f.file.is_empty()), "{:?}", outcome.font_files);
            eprintln!("P09-RENDER font files {:?}", outcome.font_files);
            // Take the header's repeat away and the check finds it.
            let xml = package::read_part(&bytes, "word/document.xml").unwrap().replace("<w:tblHeader/>", "");
            let mut archive = zip::ZipArchive::new(std::io::Cursor::new(bytes.as_slice())).unwrap();
            let (stripped, _) = crate::artifacts::edit::rewrite_parts(&mut archive, &[("word/document.xml".into(), xml)]).unwrap();
            let dir = tempfile::tempdir().unwrap();
            let outcome = crate::artifacts::render::render(&stripped, DetectedFormat::Docx, dir.path(), 1, 50, inventory);
            let results = render_checks(&stripped, &outcome);
            assert_eq!(status(&results, "render.table_headers"), CheckStatus::Fail, "{results:?}");
        } else {
            assert!(results.iter().all(|r| r.status == CheckStatus::Unavailable), "{results:?}");
        }
    }

    /// The header row is marked, and a page the table runs onto does not
    /// show it: the check reads the pages, not the marking. The page text is
    /// constructed (what a layout engine that ignored the marking would
    /// draw), which is what exercises this branch deterministically.
    #[test]
    fn a_marked_header_missing_from_a_continuation_page_is_found() {
        let rows: Vec<Vec<String>> = (1..=30).map(|i| vec![format!("TML-{i:03}"), format!("{i} readings [E1]")]).collect();
        let bytes = write_docx(&note(vec![Block::Table { header: vec!["Location".into(), "Readings".into()], rows, caption: None }])).unwrap();
        let xml = package::read_part(&bytes, "word/document.xml").unwrap();
        let units = body_units(&xml).unwrap();
        let text_of = |range: std::ops::RangeInclusive<usize>, with_header: bool| {
            let mut out: Vec<String> = Vec::new();
            if with_header {
                out.push("Location\nReadings".into());
            }
            for unit in units.iter().filter(|u| u.kind == BodyKind::Row && !u.header_row) {
                let n: usize = unit.cells[0][4..].parse().unwrap();
                if range.contains(&n) {
                    out.push(unit.cells.join("\n"));
                }
            }
            out.join("\n")
        };
        let everything_else: String = units.iter().filter(|u| u.kind != BodyKind::Row).map(|u| u.text.clone()).collect::<Vec<_>>().join("\n");
        let page = |n: u32, text: String| crate::artifacts::render::PageRender {
            page: n,
            image: format!("page-{n}.png"),
            image_sha256: "0".repeat(64),
            width: 1240,
            height: 1754,
            blank: false,
            text_characters: text.len(),
            text,
            fonts: Vec::new(),
            clipped: Vec::new(),
        };
        let outcome = |repeat: bool| RenderOutcome {
            state: RenderState::Rendered,
            detail: "constructed".into(),
            problems: Vec::new(),
            renderers: Vec::new(),
            total_pages: 2,
            pages: vec![page(1, format!("{everything_else}\n{}", text_of(1..=15, true))), page(2, text_of(16..=30, repeat))],
            pdf_sha256: None,
            font_files: Vec::new(),
        };
        let missing = render_checks(&bytes, &outcome(false));
        assert_eq!(status(&missing, "render.table_headers"), CheckStatus::Fail, "{missing:?}");
        assert!(missing.iter().any(|r| r.locations.iter().any(|l| l == "table:1 page:2")), "{missing:?}");
        let repeated = render_checks(&bytes, &outcome(true));
        assert_eq!(status(&repeated, "render.table_headers"), CheckStatus::Pass, "{repeated:?}");
    }

    #[test]
    fn an_unrendered_document_has_unavailable_render_checks() {
        let bytes = write_docx(&note(vec![Block::Paragraph { text: "Point C reads 8.2 mm [E1].".into() }])).unwrap();
        let render = RenderOutcome { state: RenderState::Unavailable, detail: "no office adapter".into(), problems: Vec::new(), renderers: Vec::new(), total_pages: 0, pages: Vec::new(), pdf_sha256: None, font_files: Vec::new() };
        let results = render_checks(&bytes, &render);
        assert_eq!(results.len(), 4);
        assert!(results.iter().all(|r| r.status == CheckStatus::Unavailable));
        assert!(!accepted(&results));
    }
}
