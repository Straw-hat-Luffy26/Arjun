//! One content and evidence contract, read the same way from every format.
//!
//! ## Why one contract
//!
//! The Word, PowerPoint and Excel writers each have their own model —
//! [`super::doc_model::Document`], [`super::doc_model::Deck`],
//! [`super::doc_model::Workbook`] — and those are the right shapes for
//! *writing*. Reading back, validating, citing, diffing and editing all ask the
//! same questions of any of them: what does the file say, where exactly, and
//! what does each statement rest on. Answering that three times, differently,
//! is how a reviewer ends up able to check a note's citations and not a deck's.
//!
//! So a produced file is read into a [`ContentModel`]: an ordered list of
//! [`ContentUnit`]s, each with a **locator** that says where it is, a **role**
//! that says what it is, its **text**, and the **citations** found in that text.
//! The locator grammar is shared — `p:12`, `slide:3/body:2`, `sheet:Readings!B4`,
//! `page:2`, `line:40` — and so is the region grammar a reader narrows with.
//!
//! ## Evidence
//!
//! A citation is a marker in the text: `[E3]` names passage 3 of the run that
//! wrote it; `[A:art-1@2]` an artifact version; `[M:mem-7@4]` a memory item at a
//! revision; `[S:<sha>@p4]` a source document by hash. Markers are what a model
//! writes. What they *mean* is fixed when a version is registered — see
//! `conversation_store`'s dependency table — because `[E3]` in one run and
//! `[E3]` in the next are different passages.
//!
//! ## What this does not do
//!
//! It reads text and structure. It does not lay anything out, so it cannot say
//! whether a table overflows its page — that is the renderer's question
//! ([`super::render`]), and the two are separate rungs of
//! [`super::validation`].

use std::collections::BTreeMap;
use std::io::Cursor;

use serde::{Deserialize, Serialize};

use super::package::{self, DetectedFormat};
use super::xml_events::{self, Event};

/// What a unit is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum UnitRole {
    Title,
    Heading,
    Paragraph,
    ListItem,
    TableCell,
    SlideTitle,
    SlideBody,
    SlideNotes,
    Cell,
    PageText,
    Line,
}

impl UnitRole {
    pub const fn as_str(self) -> &'static str {
        match self {
            UnitRole::Title => "title",
            UnitRole::Heading => "heading",
            UnitRole::Paragraph => "paragraph",
            UnitRole::ListItem => "listItem",
            UnitRole::TableCell => "tableCell",
            UnitRole::SlideTitle => "slideTitle",
            UnitRole::SlideBody => "slideBody",
            UnitRole::SlideNotes => "slideNotes",
            UnitRole::Cell => "cell",
            UnitRole::PageText => "pageText",
            UnitRole::Line => "line",
        }
    }
}

/// What a citation marker points at.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum CitationTarget {
    /// `[E3]` — a passage in the writing run's evidence table.
    Evidence { number: u32 },
    /// `[A:art-1@2]` — an artifact version.
    Artifact { artifact_id: String, version: Option<u32> },
    /// `[M:mem-7@4]` — a memory item at a revision.
    Memory { item_id: String, revision: Option<u64> },
    /// `[S:ab12…@p4]` — source bytes by hash, with a locator.
    Source { sha256: String, locator: Option<String> },
}

/// One marker, as written.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Citation {
    pub marker: String,
    pub target: CitationTarget,
}

/// One addressable piece of a file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ContentUnit {
    pub locator: String,
    pub role: UnitRole,
    /// Heading level, where the role has one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub level: Option<u8>,
    pub text: String,
    /// A cell's formula, without the leading `=`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub formula: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub citations: Vec<Citation>,
}

/// A file, read into the shared contract.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ContentModel {
    pub format: DetectedFormat,
    pub units: Vec<ContentUnit>,
    /// Section headings (Word), slide titles (PowerPoint), sheet names (Excel).
    pub outline: Vec<String>,
    /// Pages, slides, sheets or lines — whatever the format counts.
    pub containers: usize,
    /// What this reading did not cover, named. Never silently absent.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub gaps: Vec<String>,
}

impl ContentModel {
    /// Every distinct citation, in first-appearance order.
    pub fn citations(&self) -> Vec<&Citation> {
        let mut seen = std::collections::BTreeSet::new();
        self.units
            .iter()
            .flat_map(|unit| unit.citations.iter())
            .filter(|citation| seen.insert(citation.marker.clone()))
            .collect()
    }

    /// The whole text, one unit per line.
    pub fn plain_text(&self) -> String {
        self.units.iter().map(|u| u.text.as_str()).collect::<Vec<_>>().join("\n")
    }
}

/// The most units one reading keeps. A workbook past this is read in part and
/// says so, rather than taking the whole machine's memory.
pub const MAX_UNITS: usize = 200_000;

/// Reads a file's bytes into the shared contract.
///
/// The format comes from the bytes ([`package::sniff`]), and a package that
/// fails [`package::inspect`] is refused before any part is parsed.
pub fn extract(bytes: &[u8]) -> Result<ContentModel, String> {
    let format = package::sniff(bytes);
    match format {
        DetectedFormat::Docx | DetectedFormat::Pptx | DetectedFormat::Xlsx => {
            let report = package::inspect(bytes, &package::LIMITS);
            if !report.safe_to_open() {
                return Err(format!(
                    "the package was not opened: {}",
                    report.problems.join("; ")
                ));
            }
            let mut archive = zip::ZipArchive::new(Cursor::new(bytes))
                .map_err(|error| format!("the package does not open: {error}"))?;
            let mut model = match format {
                DetectedFormat::Docx => docx_units(&mut archive)?,
                DetectedFormat::Pptx => pptx_units(&mut archive)?,
                _ => xlsx_units(&mut archive)?,
            };
            for unit in &mut model.units {
                unit.citations = citations_in(&unit.text);
            }
            Ok(model)
        }
        DetectedFormat::Pdf => Ok(pdf_units(bytes)),
        DetectedFormat::Text | DetectedFormat::Svg => {
            let text = std::str::from_utf8(bytes).map_err(|_| "the text is not UTF-8".to_string())?;
            let mut units = Vec::new();
            for (index, line) in text.lines().enumerate() {
                if units.len() >= MAX_UNITS {
                    break;
                }
                if line.trim().is_empty() {
                    continue;
                }
                units.push(ContentUnit {
                    locator: format!("line:{}", index + 1),
                    role: UnitRole::Line,
                    level: None,
                    text: line.to_string(),
                    formula: None,
                    citations: citations_in(line),
                });
            }
            Ok(ContentModel {
                format,
                containers: text.lines().count(),
                outline: Vec::new(),
                gaps: if units.len() >= MAX_UNITS {
                    vec![format!("only the first {MAX_UNITS} non-empty lines were read")]
                } else {
                    Vec::new()
                },
                units,
            })
        }
        other => Err(format!(
            "the bytes are a {}, which has no content reader here",
            other.label()
        )),
    }
}

fn read_xml<R: std::io::Read + std::io::Seek>(
    archive: &mut zip::ZipArchive<R>,
    part: &str,
) -> Result<Vec<Event>, String> {
    let bytes = package::read_bounded(archive, part, package::LIMITS.max_entry_bytes)
        .ok_or_else(|| format!("the package has no readable {part}"))?;
    let text = String::from_utf8(bytes).map_err(|_| format!("{part} is not UTF-8"))?;
    xml_events::scan(&text).map_err(|error| format!("{part} is not well-formed XML: {error}"))
}

/// Relationship id → resolved part name, for one source part.
fn relationships<R: std::io::Read + std::io::Seek>(
    archive: &mut zip::ZipArchive<R>,
    source: &str,
) -> BTreeMap<String, (String, String)> {
    let (directory, file) = source.rsplit_once('/').unwrap_or(("", source));
    let rels = if directory.is_empty() {
        format!("_rels/{file}.rels")
    } else {
        format!("{directory}/_rels/{file}.rels")
    };
    let Ok(events) = read_xml(archive, &rels) else {
        return BTreeMap::new();
    };
    let mut out = BTreeMap::new();
    for event in &events {
        if event.local_name() != Some("Relationship") || !matches!(event, Event::Start { .. }) {
            continue;
        }
        if event.attribute("TargetMode") == Some("External") {
            continue;
        }
        let (Some(id), Some(target)) = (event.attribute("Id"), event.attribute("Target")) else {
            continue;
        };
        let kind = event
            .attribute("Type")
            .and_then(|t| t.rsplit('/').next())
            .unwrap_or_default()
            .to_string();
        out.insert(id.to_string(), (resolve(directory, target), kind));
    }
    out
}

/// Resolves a relationship target against the directory of its source part.
pub fn resolve(directory: &str, target: &str) -> String {
    let mut parts: Vec<&str> = if let Some(absolute) = target.strip_prefix('/') {
        return absolute.to_string();
    } else if directory.is_empty() {
        Vec::new()
    } else {
        directory.split('/').collect()
    };
    for segment in target.split('/') {
        match segment {
            "" | "." => {}
            ".." => {
                parts.pop();
            }
            other => parts.push(other),
        }
    }
    parts.join("/")
}

fn docx_units<R: std::io::Read + std::io::Seek>(
    archive: &mut zip::ZipArchive<R>,
) -> Result<ContentModel, String> {
    let events = read_xml(archive, "word/document.xml")?;
    let mut units = Vec::new();
    let mut outline = Vec::new();
    let mut paragraph_number = 0usize;
    let mut table_depth = 0usize;
    let mut in_paragraph = false;
    let mut skip_text = 0usize;
    let mut buffer = String::new();
    let mut style: Option<String> = None;
    let mut listed = false;

    let finish = |number: usize,
                      buffer: &mut String,
                      style: &mut Option<String>,
                      listed: &mut bool,
                      table_depth: usize,
                      units: &mut Vec<ContentUnit>,
                      outline: &mut Vec<String>| {
        let text = std::mem::take(buffer);
        let style_name = style.take().unwrap_or_default();
        let is_list = std::mem::replace(listed, false);
        if text.trim().is_empty() || units.len() >= MAX_UNITS {
            return;
        }
        let (role, level) = if style_name.eq_ignore_ascii_case("Title") {
            (UnitRole::Title, Some(0))
        } else if let Some(level) = style_name
            .strip_prefix("Heading")
            .or_else(|| style_name.strip_prefix("heading"))
            .and_then(|n| n.trim().parse::<u8>().ok())
        {
            (UnitRole::Heading, Some(level))
        } else if table_depth > 0 {
            (UnitRole::TableCell, None)
        } else if is_list {
            (UnitRole::ListItem, None)
        } else {
            (UnitRole::Paragraph, None)
        };
        if matches!(role, UnitRole::Heading | UnitRole::Title) {
            outline.push(text.trim().to_string());
        }
        units.push(ContentUnit {
            locator: format!("p:{number}"),
            role,
            level,
            text,
            formula: None,
            citations: Vec::new(),
        });
    };

    for event in &events {
        match event {
            Event::Start { empty, .. } => match event.local_name() {
                Some("tbl") if !empty => table_depth += 1,
                Some("p") => {
                    paragraph_number += 1;
                    if *empty {
                        continue;
                    }
                    in_paragraph = true;
                    buffer.clear();
                    style = None;
                    listed = false;
                }
                Some("pStyle") if in_paragraph => {
                    style = event.attribute("w:val").map(str::to_string);
                }
                Some("numPr") if in_paragraph => listed = true,
                Some("tab") if in_paragraph && *empty => buffer.push('\t'),
                Some("br") | Some("cr") if in_paragraph && *empty => buffer.push('\n'),
                // Field instructions and deleted revisions are not what the
                // document says.
                Some("instrText") | Some("delText") if !empty => skip_text += 1,
                _ => {}
            },
            Event::End { .. } => match event.local_name() {
                Some("tbl") => table_depth = table_depth.saturating_sub(1),
                Some("instrText") | Some("delText") => skip_text = skip_text.saturating_sub(1),
                Some("p") if in_paragraph => {
                    in_paragraph = false;
                    finish(
                        paragraph_number,
                        &mut buffer,
                        &mut style,
                        &mut listed,
                        table_depth,
                        &mut units,
                        &mut outline,
                    );
                }
                _ => {}
            },
            Event::Text(text) if in_paragraph && skip_text == 0 => buffer.push_str(text),
            Event::Text(_) => {}
        }
    }
    let mut gaps = vec!["headers, footers, comments and footnotes are not part of this reading".to_string()];
    if units.len() >= MAX_UNITS {
        gaps.push(format!("only the first {MAX_UNITS} paragraphs were read"));
    }
    Ok(ContentModel {
        format: DetectedFormat::Docx,
        containers: paragraph_number,
        units,
        outline,
        gaps,
    })
}

/// The slide parts of a presentation, in presentation order.
pub fn slide_parts<R: std::io::Read + std::io::Seek>(
    archive: &mut zip::ZipArchive<R>,
) -> Result<Vec<String>, String> {
    let presentation = read_xml(archive, "ppt/presentation.xml")?;
    let rels = relationships(archive, "ppt/presentation.xml");
    let mut slides = Vec::new();
    for event in &presentation {
        if event.local_name() == Some("sldId") {
            if let Some(id) = event.attribute("r:id") {
                if let Some((part, _)) = rels.get(id) {
                    slides.push(part.clone());
                }
            }
        }
    }
    Ok(slides)
}

fn is_drawing_paragraph(event: &Event) -> bool {
    matches!(event, Event::Start { name, .. } if name == "a:p")
}

/// Paragraph texts of one slide part: (is_title, text), every `<a:p>` counted.
fn slide_paragraphs(events: &[Event], skip_placeholders: &[&str]) -> Vec<(bool, String)> {
    let mut out = Vec::new();
    let mut shape_depth = 0usize;
    let mut shape_is_title = false;
    let mut shape_skipped = false;
    let mut in_paragraph = false;
    let mut buffer = String::new();
    for event in events {
        match event {
            Event::Start { empty, .. } => match event.local_name() {
                Some("sp") if !empty => {
                    shape_depth += 1;
                    if shape_depth == 1 {
                        shape_is_title = false;
                        shape_skipped = false;
                    }
                }
                // A shape named "Title …" is PowerPoint's own convention and
                // what this product's writer emits; a title placeholder is the
                // other way a deck says it.
                Some("cNvPr") if shape_depth > 0 => {
                    if event
                        .attribute("name")
                        .is_some_and(|name| name.to_ascii_lowercase().starts_with("title"))
                    {
                        shape_is_title = true;
                    }
                }
                Some("ph") => {
                    let kind = event.attribute("type").unwrap_or_default();
                    if kind == "title" || kind == "ctrTitle" {
                        shape_is_title = true;
                    }
                    if skip_placeholders.contains(&kind) {
                        shape_skipped = true;
                    }
                }
                Some("p") if is_drawing_paragraph(event) && !empty => {
                    in_paragraph = true;
                    buffer.clear();
                }
                Some("p") if is_drawing_paragraph(event) => {
                    if !shape_skipped {
                        out.push((shape_is_title, String::new()));
                    }
                }
                Some("br") if in_paragraph => buffer.push('\n'),
                _ => {}
            },
            Event::End { name } => {
                if name == "p:sp" {
                    shape_depth = shape_depth.saturating_sub(1);
                } else if name == "a:p" && in_paragraph {
                    in_paragraph = false;
                    if !shape_skipped {
                        out.push((shape_is_title, std::mem::take(&mut buffer)));
                    } else {
                        buffer.clear();
                    }
                }
            }
            Event::Text(text) if in_paragraph => buffer.push_str(text),
            Event::Text(_) => {}
        }
    }
    out
}

fn pptx_units<R: std::io::Read + std::io::Seek>(
    archive: &mut zip::ZipArchive<R>,
) -> Result<ContentModel, String> {
    let slides = slide_parts(archive)?;
    let mut units = Vec::new();
    let mut outline = Vec::new();
    for (index, part) in slides.iter().enumerate() {
        let number = index + 1;
        let events = read_xml(archive, part)?;
        let mut title = Vec::new();
        let mut body = 0usize;
        for (is_title, text) in slide_paragraphs(&events, &[]) {
            if is_title {
                if !text.trim().is_empty() {
                    title.push(text);
                }
                continue;
            }
            body += 1;
            if text.trim().is_empty() {
                continue;
            }
            units.push(ContentUnit {
                locator: format!("slide:{number}/body:{body}"),
                role: UnitRole::SlideBody,
                level: None,
                text,
                formula: None,
                citations: Vec::new(),
            });
        }
        if !title.is_empty() {
            let joined = title.join(" ");
            outline.push(joined.clone());
            // The title leads the slide's units, whatever order the shapes
            // are stored in.
            let position = units
                .iter()
                .position(|u| u.locator.starts_with(&format!("slide:{number}/")))
                .unwrap_or(units.len());
            units.insert(
                position,
                ContentUnit {
                    locator: format!("slide:{number}/title"),
                    role: UnitRole::SlideTitle,
                    level: None,
                    text: joined,
                    formula: None,
                    citations: Vec::new(),
                },
            );
        } else {
            outline.push(String::new());
        }
        let rels = relationships(archive, part);
        if let Some((notes, _)) = rels.values().find(|(_, kind)| kind == "notesSlide") {
            if let Ok(events) = read_xml(archive, notes) {
                let text: Vec<String> = slide_paragraphs(&events, &["sldNum", "sldImg", "hdr", "ftr", "dt"])
                    .into_iter()
                    .map(|(_, t)| t)
                    .filter(|t| !t.trim().is_empty())
                    .collect();
                if !text.is_empty() {
                    units.push(ContentUnit {
                        locator: format!("slide:{number}/notes"),
                        role: UnitRole::SlideNotes,
                        level: None,
                        text: text.join("\n"),
                        formula: None,
                        citations: Vec::new(),
                    });
                }
            }
        }
    }
    Ok(ContentModel {
        format: DetectedFormat::Pptx,
        containers: slides.len(),
        units,
        outline,
        gaps: vec!["slide masters, layouts and pictures are not part of this reading".to_string()],
    })
}

/// A1-style reference → zero-based (column, row).
pub fn cell_position(reference: &str) -> Option<(u32, u32)> {
    let reference = reference.trim().trim_start_matches('$');
    let letters: String = reference.chars().take_while(|c| c.is_ascii_alphabetic()).collect();
    let digits = reference[letters.len()..].trim_start_matches('$');
    if letters.is_empty() || digits.is_empty() || !digits.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    let mut column = 0u32;
    for c in letters.chars() {
        column = column.checked_mul(26)? + (c.to_ascii_uppercase() as u32 - 'A' as u32 + 1);
    }
    let row: u32 = digits.parse().ok()?;
    (row >= 1 && column >= 1).then(|| (column - 1, row - 1))
}

/// The worksheets of a workbook: (sheet name, part), in workbook order.
pub fn sheet_parts<R: std::io::Read + std::io::Seek>(
    archive: &mut zip::ZipArchive<R>,
) -> Result<Vec<(String, String)>, String> {
    let workbook = read_xml(archive, "xl/workbook.xml")?;
    let rels = relationships(archive, "xl/workbook.xml");
    let mut sheets = Vec::new();
    for event in &workbook {
        if event.local_name() == Some("sheet") {
            if let (Some(name), Some(id)) = (event.attribute("name"), event.attribute("r:id")) {
                if let Some((part, _)) = rels.get(id) {
                    sheets.push((name.to_string(), part.clone()));
                }
            }
        }
    }
    Ok(sheets)
}

fn shared_strings<R: std::io::Read + std::io::Seek>(archive: &mut zip::ZipArchive<R>) -> Vec<String> {
    let Ok(events) = read_xml(archive, "xl/sharedStrings.xml") else {
        return Vec::new();
    };
    let mut out = Vec::new();
    let mut current: Option<String> = None;
    let mut in_text = false;
    let mut phonetic = 0usize;
    for event in &events {
        match event {
            Event::Start { empty, .. } => match event.local_name() {
                Some("si") if !empty => current = Some(String::new()),
                Some("si") => out.push(String::new()),
                Some("rPh") if !empty => phonetic += 1,
                Some("t") if !empty => in_text = true,
                _ => {}
            },
            Event::End { .. } => match event.local_name() {
                Some("si") => out.push(current.take().unwrap_or_default()),
                Some("rPh") => phonetic = phonetic.saturating_sub(1),
                Some("t") => in_text = false,
                _ => {}
            },
            Event::Text(text) if in_text && phonetic == 0 => {
                if let Some(current) = current.as_mut() {
                    current.push_str(text);
                }
            }
            Event::Text(_) => {}
        }
    }
    out
}

fn xlsx_units<R: std::io::Read + std::io::Seek>(
    archive: &mut zip::ZipArchive<R>,
) -> Result<ContentModel, String> {
    let sheets = sheet_parts(archive)?;
    let strings = shared_strings(archive);
    let mut units = Vec::new();
    let mut gaps = Vec::new();
    for (name, part) in &sheets {
        let events = read_xml(archive, part)?;
        let mut reference = String::new();
        let mut kind = String::new();
        let mut value = String::new();
        let mut formula = String::new();
        let mut field: Option<&'static str> = None;
        let mut in_cell = false;
        for event in &events {
            match event {
                Event::Start { empty, .. } => match event.local_name() {
                    Some("c") => {
                        reference = event.attribute("r").unwrap_or_default().to_string();
                        kind = event.attribute("t").unwrap_or("n").to_string();
                        value.clear();
                        formula.clear();
                        in_cell = !empty;
                    }
                    Some("v") if in_cell && !empty => field = Some("v"),
                    Some("f") if in_cell && !empty => field = Some("f"),
                    Some("t") if in_cell && !empty => field = Some("t"),
                    _ => {}
                },
                Event::End { .. } => match event.local_name() {
                    Some("v") | Some("f") | Some("t") => field = None,
                    Some("c") if in_cell => {
                        in_cell = false;
                        let shown = match kind.as_str() {
                            "s" => value
                                .trim()
                                .parse::<usize>()
                                .ok()
                                .and_then(|i| strings.get(i).cloned())
                                .unwrap_or_default(),
                            "b" => match value.trim() {
                                "1" => "TRUE".to_string(),
                                "0" => "FALSE".to_string(),
                                other => other.to_string(),
                            },
                            _ => value.clone(),
                        };
                        if shown.is_empty() && formula.is_empty() {
                            continue;
                        }
                        if units.len() >= MAX_UNITS {
                            if gaps.is_empty() {
                                gaps.push(format!("only the first {MAX_UNITS} cells were read"));
                            }
                            continue;
                        }
                        units.push(ContentUnit {
                            locator: format!("sheet:{name}!{reference}"),
                            role: UnitRole::Cell,
                            level: None,
                            text: shown,
                            formula: (!formula.is_empty()).then(|| formula.clone()),
                            citations: Vec::new(),
                        });
                    }
                    _ => {}
                },
                Event::Text(text) => match field {
                    Some("v") | Some("t") => value.push_str(text),
                    Some("f") => formula.push_str(text),
                    _ => {}
                },
            }
        }
    }
    gaps.push(
        "a formula's value is the one cached in the file; nothing here recalculates it".to_string(),
    );
    Ok(ContentModel {
        format: DetectedFormat::Xlsx,
        containers: sheets.len(),
        outline: sheets.iter().map(|(name, _)| name.clone()).collect(),
        units,
        gaps,
    })
}

fn pdf_units(bytes: &[u8]) -> ContentModel {
    let check = super::pdf_validate::check_pdf_bytes(bytes);
    if check.opens && !check.page_texts.is_empty() {
        let units = check
            .page_texts
            .iter()
            .enumerate()
            .filter(|(_, text)| !text.trim().is_empty())
            .map(|(index, text)| ContentUnit {
                locator: format!("page:{}", index + 1),
                role: UnitRole::PageText,
                level: None,
                text: text.trim().to_string(),
                formula: None,
                citations: citations_in(text),
            })
            .collect();
        return ContentModel {
            format: DetectedFormat::Pdf,
            containers: check.pages,
            units,
            outline: Vec::new(),
            gaps: Vec::new(),
        };
    }
    // Not this product's own writer's subset. The page text is the
    // renderer's to read (`render::pdf_page_texts`); this says so rather
    // than returning nothing as if the PDF were blank.
    ContentModel {
        format: DetectedFormat::Pdf,
        containers: 0,
        units: Vec::new(),
        outline: Vec::new(),
        gaps: vec![
            "this PDF is not in the subset the built-in reader parses; its page text needs the \
             PDF rasteriser (artifact.render)"
                .to_string(),
        ],
    }
}

/// Every citation marker in `text`, in order.
pub fn citations_in(text: &str) -> Vec<Citation> {
    let mut out = Vec::new();
    let mut rest = text;
    while let Some(open) = rest.find('[') {
        let after = &rest[open + 1..];
        let Some(close) = after.find(']') else { break };
        let inner = &after[..close];
        if let Some(target) = parse_marker(inner) {
            let citation = Citation {
                marker: format!("[{inner}]"),
                target,
            };
            if !out.contains(&citation) {
                out.push(citation);
            }
        }
        rest = &after[close + 1..];
    }
    out
}

fn parse_marker(inner: &str) -> Option<CitationTarget> {
    if inner.contains(char::is_whitespace) || inner.len() > 160 {
        return None;
    }
    if let Some(number) = inner.strip_prefix('E') {
        return number.parse::<u32>().ok().filter(|n| *n > 0).map(|number| CitationTarget::Evidence { number });
    }
    let (kind, value) = inner.split_once(':')?;
    let (id, at) = match value.rsplit_once('@') {
        Some((id, at)) => (id, Some(at)),
        None => (value, None),
    };
    if id.is_empty() {
        return None;
    }
    Some(match kind {
        "A" => CitationTarget::Artifact {
            artifact_id: id.to_string(),
            version: at.and_then(|v| v.parse().ok()),
        },
        "M" => CitationTarget::Memory {
            item_id: id.to_string(),
            revision: at.and_then(|v| v.parse().ok()),
        },
        "S" => CitationTarget::Source {
            sha256: id.to_ascii_lowercase(),
            locator: at.map(str::to_string),
        },
        _ => return None,
    })
}

// ── Regions ──────────────────────────────────────────────────────────────

/// A part of a file a reader asks for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Region {
    Paragraphs { from: u32, to: u32 },
    Section(String),
    Slides { from: u32, to: u32 },
    Sheet { name: String, range: Option<((u32, u32), (u32, u32))> },
    Pages { from: u32, to: u32 },
    Lines { from: u32, to: u32 },
}

/// The region grammar, shared by every format:
///
/// `p:5`, `paragraphs:3-9`, `section:Findings`, `slide:3`, `slides:2-4`,
/// `sheet:Readings`, `sheet:Readings!B4`, `sheet:Readings!A1:C10`, `page:2`,
/// `pages:1-3`, `line:5`, `lines:10-40`.
pub fn parse_region(raw: &str) -> Result<Region, String> {
    let raw = raw.trim();
    let (kind, value) = raw
        .split_once(':')
        .ok_or_else(|| format!("{raw:?} is not a region. {REGION_HELP}"))?;
    let range = |value: &str| -> Result<(u32, u32), String> {
        let (from, to) = match value.split_once('-') {
            Some((a, b)) => (a.trim(), b.trim()),
            None => (value.trim(), value.trim()),
        };
        let from: u32 = from.parse().map_err(|_| format!("{value:?} is not a number or range"))?;
        let to: u32 = to.parse().map_err(|_| format!("{value:?} is not a number or range"))?;
        if from == 0 || to < from {
            return Err(format!("{value:?} is not a range that starts at 1 and runs forward"));
        }
        Ok((from, to))
    };
    Ok(match kind.trim().to_ascii_lowercase().as_str() {
        "p" | "paragraph" | "paragraphs" => {
            let (from, to) = range(value)?;
            Region::Paragraphs { from, to }
        }
        "section" => {
            if value.trim().is_empty() {
                return Err("a section region names its heading".to_string());
            }
            Region::Section(value.trim().to_string())
        }
        "slide" | "slides" => {
            let (from, to) = range(value)?;
            Region::Slides { from, to }
        }
        "page" | "pages" => {
            let (from, to) = range(value)?;
            Region::Pages { from, to }
        }
        "line" | "lines" => {
            let (from, to) = range(value)?;
            Region::Lines { from, to }
        }
        "sheet" => {
            let (name, cells) = match value.split_once('!') {
                Some((name, cells)) => (name, Some(cells)),
                None => (value, None),
            };
            let range = match cells {
                None => None,
                Some(cells) => {
                    let (a, b) = cells.split_once(':').unwrap_or((cells, cells));
                    let start = cell_position(a).ok_or_else(|| format!("{a:?} is not a cell"))?;
                    let end = cell_position(b).ok_or_else(|| format!("{b:?} is not a cell"))?;
                    Some((
                        (start.0.min(end.0), start.1.min(end.1)),
                        (start.0.max(end.0), start.1.max(end.1)),
                    ))
                }
            };
            Region::Sheet { name: name.trim().to_string(), range }
        }
        _ => return Err(format!("{raw:?} is not a region. {REGION_HELP}")),
    })
}

pub const REGION_HELP: &str = "Regions: p:5, paragraphs:3-9, section:<heading>, slide:3, slides:2-4, \
     sheet:<name>, sheet:<name>!A1:C10, page:2, pages:1-3, line:5, lines:10-40.";

fn number_after(locator: &str, prefix: &str) -> Option<u32> {
    let rest = locator.strip_prefix(prefix)?;
    let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
    digits.parse().ok()
}

/// The units a region covers, in document order.
pub fn select<'a>(model: &'a ContentModel, region: &Region) -> Result<Vec<&'a ContentUnit>, String> {
    let wrong = |region: &str| {
        Err(format!(
            "a {region} region does not apply to a {}",
            model.format.label()
        ))
    };
    let picked: Vec<&ContentUnit> = match region {
        Region::Paragraphs { from, to } => {
            if model.format != DetectedFormat::Docx {
                return wrong("paragraph");
            }
            model
                .units
                .iter()
                .filter(|u| number_after(&u.locator, "p:").is_some_and(|n| n >= *from && n <= *to))
                .collect()
        }
        Region::Section(heading) => {
            let Some(start) = model.units.iter().position(|u| {
                matches!(u.role, UnitRole::Heading | UnitRole::Title | UnitRole::SlideTitle)
                    && u.text.trim().eq_ignore_ascii_case(heading.trim())
            }) else {
                return Err(format!(
                    "no heading reads {heading:?}. Headings: {}",
                    model.outline.join("; ")
                ));
            };
            let level = model.units[start].level.unwrap_or(0);
            let slide_prefix = model.units[start]
                .locator
                .split('/')
                .next()
                .map(|s| format!("{s}/"));
            let mut out = vec![&model.units[start]];
            for unit in &model.units[start + 1..] {
                let ends = match model.format {
                    DetectedFormat::Pptx => !slide_prefix
                        .as_deref()
                        .is_some_and(|prefix| unit.locator.starts_with(prefix)),
                    _ => {
                        matches!(unit.role, UnitRole::Heading | UnitRole::Title)
                            && unit.level.unwrap_or(0) <= level
                    }
                };
                if ends {
                    break;
                }
                out.push(unit);
            }
            out
        }
        Region::Slides { from, to } => {
            if model.format != DetectedFormat::Pptx {
                return wrong("slide");
            }
            model
                .units
                .iter()
                .filter(|u| number_after(&u.locator, "slide:").is_some_and(|n| n >= *from && n <= *to))
                .collect()
        }
        Region::Pages { from, to } => {
            if model.format != DetectedFormat::Pdf {
                return wrong("page");
            }
            model
                .units
                .iter()
                .filter(|u| number_after(&u.locator, "page:").is_some_and(|n| n >= *from && n <= *to))
                .collect()
        }
        Region::Lines { from, to } => {
            if !matches!(model.format, DetectedFormat::Text | DetectedFormat::Svg) {
                return wrong("line");
            }
            model
                .units
                .iter()
                .filter(|u| number_after(&u.locator, "line:").is_some_and(|n| n >= *from && n <= *to))
                .collect()
        }
        Region::Sheet { name, range } => {
            if model.format != DetectedFormat::Xlsx {
                return wrong("sheet");
            }
            if !model.outline.iter().any(|sheet| sheet == name) {
                return Err(format!(
                    "the workbook has no sheet {name:?}. Sheets: {}",
                    model.outline.join("; ")
                ));
            }
            let prefix = format!("sheet:{name}!");
            model
                .units
                .iter()
                .filter(|u| {
                    let Some(cell) = u.locator.strip_prefix(&prefix) else {
                        return false;
                    };
                    match (range, cell_position(cell)) {
                        (None, _) => true,
                        (Some(((c0, r0), (c1, r1))), Some((c, r))) => {
                            c >= *c0 && c <= *c1 && r >= *r0 && r <= *r1
                        }
                        (Some(_), None) => false,
                    }
                })
                .collect()
        }
    };
    Ok(picked)
}

/// Renders units for a model to read, within `max_bytes`, saying what was left
/// out. Returns the text and how many units it carried.
pub fn render_units(units: &[&ContentUnit], max_bytes: usize) -> (String, usize) {
    let mut out = String::new();
    let mut shown = 0usize;
    for unit in units {
        let mut line = format!("{} [{}", unit.locator, unit.role.as_str());
        if let Some(level) = unit.level.filter(|l| *l > 0) {
            line.push_str(&format!(" {level}"));
        }
        line.push_str("] ");
        line.push_str(&unit.text.replace('\n', " / "));
        if let Some(formula) = &unit.formula {
            line.push_str(&format!("  (formula ={formula})"));
        }
        line.push('\n');
        if out.len() + line.len() > max_bytes && shown > 0 {
            break;
        }
        out.push_str(&line);
        shown += 1;
    }
    (out, shown)
}

// ── Diff ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Change {
    Added,
    Removed,
    Changed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DiffEntry {
    pub change: Change,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub before: Option<ContentUnit>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub after: Option<ContentUnit>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ContentDiff {
    pub unchanged: usize,
    pub entries: Vec<DiffEntry>,
    /// True when the inputs were too large for an alignment and were compared
    /// position by position instead. Said, because it reports more changes.
    pub positional: bool,
}

/// The largest alignment table built, in cells.
const MAX_ALIGNMENT_CELLS: usize = 4_000_000;

fn key(unit: &ContentUnit) -> (UnitRole, &str, Option<&str>) {
    (unit.role, unit.text.as_str(), unit.formula.as_deref())
}

/// Unit-level differences between two readings.
pub fn diff(before: &ContentModel, after: &ContentModel) -> ContentDiff {
    let a = &before.units;
    let b = &after.units;
    let mut prefix = 0;
    while prefix < a.len() && prefix < b.len() && key(&a[prefix]) == key(&b[prefix]) {
        prefix += 1;
    }
    let mut suffix = 0;
    while suffix < a.len() - prefix
        && suffix < b.len() - prefix
        && key(&a[a.len() - 1 - suffix]) == key(&b[b.len() - 1 - suffix])
    {
        suffix += 1;
    }
    let a_mid = &a[prefix..a.len() - suffix];
    let b_mid = &b[prefix..b.len() - suffix];
    let mut raw: Vec<DiffEntry> = Vec::new();
    let mut unchanged = prefix + suffix;
    let positional = a_mid.len().saturating_mul(b_mid.len()) > MAX_ALIGNMENT_CELLS;

    if positional {
        for index in 0..a_mid.len().max(b_mid.len()) {
            match (a_mid.get(index), b_mid.get(index)) {
                (Some(x), Some(y)) if key(x) == key(y) => unchanged += 1,
                (x, y) => {
                    if let Some(x) = x {
                        raw.push(DiffEntry { change: Change::Removed, before: Some(x.clone()), after: None });
                    }
                    if let Some(y) = y {
                        raw.push(DiffEntry { change: Change::Added, before: None, after: Some(y.clone()) });
                    }
                }
            }
        }
    } else {
        let (n, m) = (a_mid.len(), b_mid.len());
        let mut table = vec![0u32; (n + 1) * (m + 1)];
        for i in (0..n).rev() {
            for j in (0..m).rev() {
                table[i * (m + 1) + j] = if key(&a_mid[i]) == key(&b_mid[j]) {
                    table[(i + 1) * (m + 1) + j + 1] + 1
                } else {
                    table[(i + 1) * (m + 1) + j].max(table[i * (m + 1) + j + 1])
                };
            }
        }
        let (mut i, mut j) = (0, 0);
        while i < n || j < m {
            if i < n && j < m && key(&a_mid[i]) == key(&b_mid[j]) {
                unchanged += 1;
                i += 1;
                j += 1;
            } else if j < m && (i == n || table[i * (m + 1) + j + 1] > table[(i + 1) * (m + 1) + j]) {
                raw.push(DiffEntry { change: Change::Added, before: None, after: Some(b_mid[j].clone()) });
                j += 1;
            } else {
                raw.push(DiffEntry { change: Change::Removed, before: Some(a_mid[i].clone()), after: None });
                i += 1;
            }
        }
    }

    // A removal and an addition at the same locator are one change, whichever
    // order the alignment happened to emit them in.
    let mut entries: Vec<DiffEntry> = Vec::new();
    for entry in raw {
        let (wanted, locator) = match entry.change {
            Change::Added => (Change::Removed, entry.after.as_ref().map(|u| u.locator.clone())),
            Change::Removed => (Change::Added, entry.before.as_ref().map(|u| u.locator.clone())),
            Change::Changed => (Change::Changed, None),
        };
        if locator.is_some() {
            if let Some(slot) = entries.iter_mut().find(|e| {
                e.change == wanted
                    && match wanted {
                        Change::Removed => e.before.as_ref().map(|u| u.locator.clone()) == locator,
                        _ => e.after.as_ref().map(|u| u.locator.clone()) == locator,
                    }
            }) {
                slot.change = Change::Changed;
                if entry.change == Change::Added {
                    slot.after = entry.after;
                } else {
                    slot.before = entry.before;
                }
                continue;
            }
        }
        entries.push(entry);
    }
    ContentDiff { unchanged, entries, positional }
}

/// Fixtures shared by this module's tests and its neighbours'.
#[cfg(test)]
pub(crate) mod tests_support {
    use crate::artifacts::doc_model::{Block, Document, Properties, Section};
    use crate::artifacts::docx::DocumentMetadata;

    pub(crate) fn metadata() -> DocumentMetadata {
        DocumentMetadata {
            task_id: "run-p04".into(),
            created_at: "2026-09-24T00:00:00Z".into(),
            model: "test".into(),
            classification: "Internal".into(),
            is_draft: true,
        }
    }

    /// A three-section Word document with an `[E1]` and an `[M:mem-1@2]` citation.
    pub(crate) fn sample_docx(dir: &std::path::Path) -> Vec<u8> {
        let document = Document {
            title: "Shell thickness".into(),
            classification: "Internal".into(),
            properties: Properties::default(),
            sections: vec![
                Section {
                    heading: "Findings".into(),
                    level: 1,
                    blocks: vec![Block::Paragraph {
                        text: "Point C measured 8.2 mm against a 9.0 mm minimum [E1].".into(),
                    }],
                },
                Section {
                    heading: "Readings".into(),
                    level: 2,
                    blocks: vec![
                        Block::Paragraph {
                            text: "Ultrasonic readings at the three survey points, in millimetres.".into(),
                        },
                        Block::Table {
                            header: vec!["Point".into(), "mm".into()],
                            rows: vec![vec!["C".into(), "8.2".into()], vec!["D".into(), "9.4".into()]],
                            caption: Some("Readings against the stated minimum".into()),
                        },
                    ],
                },
                Section {
                    heading: "Recommendation".into(),
                    level: 1,
                    blocks: vec![Block::Bullets { items: vec!["Replace the shell course [M:mem-1@2].".into()] }],
                },
            ],
        };
        let path = dir.join("note.docx");
        crate::artifacts::docx::write_document_model(&path, &document, &metadata()).expect("writes");
        std::fs::read(path).expect("reads")
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::artifacts::doc_model::{Block, Column, ColumnType, Deck, Document, Properties, Section, Sheet, SlideModel, Workbook};
    use crate::artifacts::docx::DocumentMetadata;

    fn metadata() -> DocumentMetadata {
        DocumentMetadata {
            task_id: "run-p04".into(),
            created_at: "2026-09-24T00:00:00Z".into(),
            model: "test".into(),
            classification: "Internal".into(),
            is_draft: true,
        }
    }

    use super::tests_support::sample_docx;

    #[test]
    fn a_word_document_reads_into_located_cited_units() {
        let dir = tempfile::tempdir().unwrap();
        let model = extract(&sample_docx(dir.path())).expect("extracts");
        assert_eq!(model.format, DetectedFormat::Docx);
        assert!(model.outline.iter().any(|h| h == "Findings"), "{:?}", model.outline);
        let finding = model.units.iter().find(|u| u.text.contains("8.2 mm")).expect("the finding");
        assert!(finding.locator.starts_with("p:"));
        assert_eq!(finding.citations[0].target, CitationTarget::Evidence { number: 1 });
        assert!(model.units.iter().any(|u| u.role == UnitRole::TableCell && u.text == "8.2"));
        let cited: Vec<_> = model.citations().into_iter().map(|c| c.marker.clone()).collect();
        assert_eq!(cited, vec!["[E1]".to_string(), "[M:mem-1@2]".to_string()]);

        let section = select(&model, &parse_region("section:Findings").unwrap()).unwrap();
        assert!(section.iter().any(|u| u.text.contains("8.2 mm")));
        assert!(!section.iter().any(|u| u.text.contains("Replace")), "a section stops at the next heading");
    }

    #[test]
    fn a_deck_reads_slide_titles_bodies_and_notes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("deck.pptx");
        let deck = Deck {
            title: "Unit Four".into(),
            classification: "Internal".into(),
            slides: vec![SlideModel {
                heading: "Findings".into(),
                bullets: vec!["One point below minimum [E2].".into()],
                table: None,
                notes: Some("Point C is the one.".into()),
            }],
        };
        crate::artifacts::pptx::write_deck_model(&path, &deck, false).expect("writes");
        let model = extract(&std::fs::read(&path).unwrap()).expect("extracts");
        assert_eq!(model.format, DetectedFormat::Pptx);
        assert_eq!(model.containers, 2, "title slide plus one");
        assert!(model.units.iter().any(|u| u.locator == "slide:2/title" && u.text == "Findings"), "{:#?}", model.units);
        assert!(model.units.iter().any(|u| u.locator.starts_with("slide:2/body:") && u.text.contains("[E2]")));
        assert!(model.units.iter().any(|u| u.locator == "slide:2/notes" && u.text.contains("Point C")));
        let slide = select(&model, &parse_region("slide:2").unwrap()).unwrap();
        assert!(slide.len() >= 3);
    }

    #[test]
    fn a_workbook_reads_cells_with_their_formulas() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("book.xlsx");
        let workbook = Workbook {
            title: "Readings".into(),
            classification: "Internal".into(),
            sheets: vec![Sheet {
                name: "Readings".into(),
                columns: vec![
                    Column { header: "Point".into(), kind: ColumnType::Text, width: None },
                    Column { header: "Measured".into(), kind: ColumnType::Number, width: None },
                    Column { header: "Margin".into(), kind: ColumnType::Formula, width: None },
                ],
                rows: vec![vec!["C".into(), "8.2".into(), "=B2-9".into()]],
                freeze_header: true,
            }],
        };
        crate::artifacts::xlsx::write_workbook_model(&path, &workbook).expect("writes");
        let model = extract(&std::fs::read(&path).unwrap()).expect("extracts");
        assert_eq!(model.outline, vec!["Readings".to_string()]);
        let measured = model.units.iter().find(|u| u.locator == "sheet:Readings!B2").expect("B2");
        assert_eq!(measured.text, "8.2");
        let margin = model.units.iter().find(|u| u.locator == "sheet:Readings!C2").expect("C2");
        assert_eq!(margin.formula.as_deref(), Some("B2-9"));
        let range = select(&model, &parse_region("sheet:Readings!A2:B2").unwrap()).unwrap();
        assert_eq!(range.len(), 2);
        assert!(select(&model, &parse_region("sheet:Nope").unwrap()).is_err());
        assert!(select(&model, &parse_region("slide:1").unwrap()).is_err(), "a slide region on a workbook");
    }

    #[test]
    fn markers_of_every_kind_are_parsed_and_prose_brackets_are_not() {
        let found = citations_in("A [E3], [A:art-9@2], [M:mem-1@4], [S:ABCDEF12@p4] and [note] [E0].");
        let targets: Vec<_> = found.iter().map(|c| c.target.clone()).collect();
        assert_eq!(targets.len(), 4, "{targets:?}");
        assert!(targets.contains(&CitationTarget::Artifact { artifact_id: "art-9".into(), version: Some(2) }));
        assert!(targets.contains(&CitationTarget::Source { sha256: "abcdef12".into(), locator: Some("p4".into()) }));
    }

    #[test]
    fn a_diff_names_what_changed_where() {
        let unit = |locator: &str, text: &str| ContentUnit {
            locator: locator.into(),
            role: UnitRole::Paragraph,
            level: None,
            text: text.into(),
            formula: None,
            citations: Vec::new(),
        };
        let model = |units: Vec<ContentUnit>| ContentModel {
            format: DetectedFormat::Docx,
            units,
            outline: Vec::new(),
            containers: 0,
            gaps: Vec::new(),
        };
        let before = model(vec![unit("p:1", "a"), unit("p:2", "b"), unit("p:3", "c")]);
        let after = model(vec![unit("p:1", "a"), unit("p:2", "B"), unit("p:3", "c"), unit("p:4", "d")]);
        let d = diff(&before, &after);
        assert_eq!(d.unchanged, 2);
        assert_eq!(d.entries.len(), 2, "{:#?}", d.entries);
        assert_eq!(d.entries[0].change, Change::Changed);
        assert_eq!(d.entries[1].change, Change::Added);
        assert!(diff(&before, &before).entries.is_empty());
    }

    #[test]
    fn regions_parse_and_bad_ones_say_what_is_accepted() {
        assert_eq!(parse_region("pages:2-3").unwrap(), Region::Pages { from: 2, to: 3 });
        assert_eq!(parse_region("p:4").unwrap(), Region::Paragraphs { from: 4, to: 4 });
        assert!(parse_region("pages:3-2").is_err());
        assert!(parse_region("everything").unwrap_err().contains("Regions:"));
        assert_eq!(cell_position("AA10"), Some((26, 9)));
    }

    #[test]
    fn a_corrupted_package_is_refused_not_read_as_empty() {
        let dir = tempfile::tempdir().unwrap();
        let mut bytes = sample_docx(dir.path());
        let middle = bytes.len() / 2;
        bytes.truncate(middle);
        assert!(extract(&bytes).is_err());
        assert!(extract(&[0u8, 159, 146, 150]).is_err(), "random bytes have no reader");
    }
}
