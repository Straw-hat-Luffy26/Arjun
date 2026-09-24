//! Changing one thing in a produced file and nothing else.
//!
//! ## Why not re-render
//!
//! Rendering from the content model rewrites every part of the package. A
//! document that has since picked up a person's comments, an embedded drawing,
//! a custom property or a style this product never writes loses all of it —
//! silently, because the writer does not know those parts exist. For a
//! targeted edit ("change the reading at C to 8.4 mm") that is the wrong
//! trade: the change is one text node, and the other nine hundred parts of the
//! file should come out byte-for-byte as they went in.
//!
//! ## How
//!
//! Every ZIP entry except the one being edited is copied **raw** — its
//! compressed bytes, CRC and header, untouched ([`zip::ZipWriter::raw_copy_file`]).
//! Inside the edited part, the one text node that holds the target text is
//! spliced by byte range ([`super::xml_events::scan_spans`]); every byte
//! around it survives, including namespaces, attributes and markup this code
//! does not understand.
//!
//! ## When it refuses, before anything is written
//!
//! - the package fails [`super::package::inspect`];
//! - the target is not found, or the text to find is not in it exactly once;
//! - the text spans several formatting runs, so no single node holds it;
//! - the target is a formula cell, whose value is computed, or a paragraph
//!   holding a text box;
//! - after the splice, re-reading the file shows any unit other than the
//!   targeted ones changed.
//!
//! The last is the guarantee rather than a hope: the new bytes are read back
//! through [`super::content`] and diffed against the old; anything beyond the
//! requested change refuses the whole edit.

use std::collections::BTreeSet;
use std::io::{Cursor, Write};

use serde::{Deserialize, Serialize};

use super::content::{self, UnitRole};
use super::package::{self, DetectedFormat};
use super::xml_events::{self, Event, Spanned};

/// One requested change.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Edit {
    /// A locator as [`super::content`] reports it: `p:12`,
    /// `slide:3/body:2`, `slide:3/title`, `sheet:Readings!B4`, `line:40`.
    pub locator: String,
    /// The exact text to replace within that unit. Omitted, the unit's whole
    /// text is replaced — allowed only when one node holds all of it.
    #[serde(default)]
    pub find: Option<String>,
    pub replace: String,
}

/// One change as made.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Applied {
    pub locator: String,
    pub before: String,
    pub after: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EditOutcome {
    pub bytes: Vec<u8>,
    pub applied: Vec<Applied>,
    /// Parts rewritten. Everything else was copied raw.
    pub changed_parts: Vec<String>,
    pub untouched_parts: usize,
    pub notes: Vec<String>,
}

/// The most edits one call makes.
pub const MAX_EDITS: usize = 50;

/// Applies `edits` to `bytes`, or refuses without producing anything.
pub fn apply(bytes: &[u8], edits: &[Edit]) -> Result<EditOutcome, String> {
    if edits.is_empty() {
        return Err("no edit was given".into());
    }
    if edits.len() > MAX_EDITS {
        return Err(format!("{} edits were given; at most {MAX_EDITS} are made in one call", edits.len()));
    }
    let format = package::sniff(bytes);
    let before = content::extract(bytes).map_err(|e| format!("the file could not be read: {e}"))?;
    let targets: BTreeSet<&str> = edits.iter().map(|e| e.locator.as_str()).collect();
    if targets.len() != edits.len() {
        return Err("two edits name the same place; combine them into one".into());
    }

    let outcome = match format {
        DetectedFormat::Docx | DetectedFormat::Pptx | DetectedFormat::Xlsx => edit_package(bytes, format, edits)?,
        DetectedFormat::Text | DetectedFormat::Svg => edit_text(bytes, edits)?,
        DetectedFormat::Pdf => {
            return Err("a PDF is produced again from its content, not edited in place".into())
        }
        other => return Err(format!("a {} is not something this edits", other.label())),
    };

    // The guarantee: only the targeted units changed.
    let after = content::extract(&outcome.bytes).map_err(|e| format!("the edited file does not read back: {e}"))?;
    let diff = content::diff(&before, &after);
    for entry in &diff.entries {
        let locator = entry
            .after
            .as_ref()
            .or(entry.before.as_ref())
            .map(|u| u.locator.as_str())
            .unwrap_or_default();
        if !targets.contains(locator) || entry.change != content::Change::Changed {
            return Err(format!(
                "the edit would have changed {locator} as well as what was asked, so nothing was written"
            ));
        }
    }
    for applied in &outcome.applied {
        let unit = after.units.iter().find(|u| u.locator == applied.locator);
        if unit.map(|u| u.text.as_str()) != Some(applied.after.as_str()) {
            return Err(format!("{} does not read back as edited, so nothing was written", applied.locator));
        }
    }
    Ok(outcome)
}

fn replace_in(text: &str, find: Option<&str>, replace: &str, locator: &str) -> Result<String, String> {
    match find {
        None => Ok(replace.to_string()),
        Some(find) if find.is_empty() => Err(format!("the text to find in {locator} is empty")),
        Some(find) => match text.matches(find).count() {
            1 => Ok(text.replacen(find, replace, 1)),
            0 => Err(format!("{locator} does not contain {find:?}")),
            n => Err(format!("{locator} contains {find:?} {n} times; give more of the text so it names one place")),
        },
    }
}

fn edit_text(bytes: &[u8], edits: &[Edit]) -> Result<EditOutcome, String> {
    let text = std::str::from_utf8(bytes).map_err(|_| "the text is not UTF-8".to_string())?;
    let ending = if text.contains("\r\n") { "\r\n" } else { "\n" };
    let mut lines: Vec<String> = text.split(ending).map(str::to_string).collect();
    let mut applied = Vec::new();
    for edit in edits {
        let number: usize = edit
            .locator
            .strip_prefix("line:")
            .and_then(|n| n.parse().ok())
            .filter(|n| *n >= 1)
            .ok_or_else(|| format!("{:?} is not a line locator", edit.locator))?;
        let line = lines.get_mut(number - 1).ok_or_else(|| format!("there is no line {number}"))?;
        if edit.replace.contains('\n') {
            return Err(format!("the replacement for {} holds a line break; edit one line at a time", edit.locator));
        }
        let after = replace_in(line, edit.find.as_deref(), &edit.replace, &edit.locator)?;
        applied.push(Applied { locator: edit.locator.clone(), before: line.clone(), after: after.clone() });
        *line = after;
    }
    Ok(EditOutcome {
        bytes: lines.join(ending).into_bytes(),
        applied,
        changed_parts: vec!["the file".into()],
        untouched_parts: 0,
        notes: Vec::new(),
    })
}

/// A text node that holds part of a unit.
struct Node {
    start: usize,
    end: usize,
    text: String,
}

fn edit_package(bytes: &[u8], format: DetectedFormat, edits: &[Edit]) -> Result<EditOutcome, String> {
    let report = package::inspect(bytes, &package::LIMITS);
    if !report.safe_to_open() {
        return Err(format!("the package was not opened: {}", report.problems.join("; ")));
    }
    let mut archive = zip::ZipArchive::new(Cursor::new(bytes)).map_err(|e| format!("the package does not open: {e}"))?;

    // Group edits by the part they touch.
    let mut by_part: Vec<(String, Vec<&Edit>)> = Vec::new();
    for edit in edits {
        let part = part_for(&mut archive, format, &edit.locator)?;
        match by_part.iter_mut().find(|(p, _)| *p == part) {
            Some((_, list)) => list.push(edit),
            None => by_part.push((part, vec![edit])),
        }
    }

    let mut rewritten: Vec<(String, String)> = Vec::new();
    let mut applied = Vec::new();
    let mut notes = Vec::new();
    for (part, part_edits) in &by_part {
        let raw = package::read_bounded(&mut archive, part, package::LIMITS.max_entry_bytes)
            .ok_or_else(|| format!("the package has no readable {part}"))?;
        let mut xml = String::from_utf8(raw).map_err(|_| format!("{part} is not UTF-8"))?;
        // Splices are applied from the end of the part backwards, so an
        // earlier splice never moves a later one's offsets.
        let mut splices: Vec<(usize, usize, String, Applied)> = Vec::new();
        let spans = xml_events::scan_spans(&xml).map_err(|e| format!("{part} is not well-formed: {e}"))?;
        for edit in part_edits {
            match format {
                DetectedFormat::Xlsx => {
                    let (start, end, replacement, done, note) = cell_splice(&spans, &xml, edit, &mut archive)?;
                    if let Some(note) = note {
                        notes.push(note);
                    }
                    splices.push((start, end, replacement, done));
                }
                _ => {
                    let nodes = match format {
                        DetectedFormat::Docx => word_nodes(&spans, &edit.locator)?,
                        _ => slide_nodes(&spans, &edit.locator)?,
                    };
                    let (node, before_unit) = pick_node(&nodes, edit)?;
                    let after_node = replace_in(&node.text, edit.find.as_deref(), &edit.replace, &edit.locator)?;
                    let unit_after = before_unit.replacen(&node.text, &after_node, 1);
                    splices.push((
                        node.start,
                        node.end,
                        escape_text(&after_node),
                        Applied { locator: edit.locator.clone(), before: before_unit, after: unit_after },
                    ));
                }
            }
        }
        splices.sort_by(|a, b| b.0.cmp(&a.0));
        for window in splices.windows(2) {
            if window[1].1 > window[0].0 {
                return Err("two edits touch the same text; combine them into one".into());
            }
        }
        for (start, end, replacement, done) in splices {
            xml.replace_range(start..end, &replacement);
            applied.push(done);
        }
        rewritten.push((part.clone(), xml));
    }

    // Everything but the rewritten parts is copied raw.
    let mut out = Cursor::new(Vec::new());
    let mut untouched = 0usize;
    {
        let mut writer = zip::ZipWriter::new(&mut out);
        let options: zip::write::FileOptions<'_, ()> =
            zip::write::FileOptions::default().compression_method(zip::CompressionMethod::Deflated);
        for index in 0..archive.len() {
            let name = archive
                .by_index_raw(index)
                .map_err(|e| format!("entry {index} could not be read: {e}"))?
                .name()
                .to_string();
            if let Some((_, body)) = rewritten.iter().find(|(part, _)| *part == name) {
                writer.start_file(name.as_str(), options).map_err(|e| format!("{name} could not be written: {e}"))?;
                writer.write_all(body.as_bytes()).map_err(|e| format!("{name} could not be written: {e}"))?;
            } else {
                let entry = archive.by_index_raw(index).map_err(|e| format!("entry {index} could not be read: {e}"))?;
                writer.raw_copy_file(entry).map_err(|e| format!("{name} could not be copied: {e}"))?;
                untouched += 1;
            }
        }
        writer.finish().map_err(|e| format!("the package could not be finished: {e}"))?;
    }
    applied.sort_by(|a, b| a.locator.cmp(&b.locator));
    Ok(EditOutcome {
        bytes: out.into_inner(),
        applied,
        changed_parts: rewritten.into_iter().map(|(part, _)| part).collect(),
        untouched_parts: untouched,
        notes,
    })
}

fn escape_text(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            c if (c as u32) < 0x20 && c != '\t' && c != '\n' => {}
            c => out.push(c),
        }
    }
    out
}

/// The node an edit applies to, and the unit's whole text before it.
fn pick_node<'a>(nodes: &'a [Node], edit: &Edit) -> Result<(&'a Node, String), String> {
    let whole: String = nodes.iter().map(|n| n.text.as_str()).collect();
    if nodes.is_empty() {
        return Err(format!("{} holds no text to edit", edit.locator));
    }
    match edit.find.as_deref() {
        None => {
            if nodes.len() == 1 {
                Ok((&nodes[0], whole))
            } else {
                Err(format!(
                    "{}'s text is split across {} formatting runs; give the exact text to change \
                     within one of them as `find`",
                    edit.locator,
                    nodes.len()
                ))
            }
        }
        Some(find) => {
            let holding: Vec<&Node> = nodes.iter().filter(|n| n.text.contains(find)).collect();
            match holding.len() {
                1 => Ok((holding[0], whole)),
                0 if whole.contains(find) => Err(format!(
                    "{find:?} spans more than one formatting run in {}; give a shorter piece of text \
                     that sits within one",
                    edit.locator
                )),
                0 => Err(format!("{} does not contain {find:?}", edit.locator)),
                _ => Err(format!("{find:?} appears in more than one run of {}; give more of the text", edit.locator)),
            }
        }
    }
}

fn part_for<R: std::io::Read + std::io::Seek>(
    archive: &mut zip::ZipArchive<R>,
    format: DetectedFormat,
    locator: &str,
) -> Result<String, String> {
    match format {
        DetectedFormat::Docx => {
            if locator.strip_prefix("p:").and_then(|n| n.parse::<usize>().ok()).is_none() {
                return Err(format!("{locator:?} is not a paragraph locator (p:N)"));
            }
            Ok("word/document.xml".into())
        }
        DetectedFormat::Pptx => {
            let number: usize = locator
                .strip_prefix("slide:")
                .and_then(|rest| rest.split('/').next())
                .and_then(|n| n.parse().ok())
                .ok_or_else(|| format!("{locator:?} is not a slide locator (slide:N/title or slide:N/body:K)"))?;
            if locator.ends_with("/notes") {
                return Err("speaker notes are not edited in place yet; this edits slide text".into());
            }
            let slides = content::slide_parts(archive)?;
            slides
                .get(number.wrapping_sub(1))
                .cloned()
                .ok_or_else(|| format!("the deck has no slide {number}"))
        }
        DetectedFormat::Xlsx => {
            let (sheet, _) = locator
                .strip_prefix("sheet:")
                .and_then(|rest| rest.split_once('!'))
                .ok_or_else(|| format!("{locator:?} is not a cell locator (sheet:Name!B4)"))?;
            content::sheet_parts(archive)?
                .into_iter()
                .find(|(name, _)| name == sheet)
                .map(|(_, part)| part)
                .ok_or_else(|| format!("the workbook has no sheet {sheet:?}"))
        }
        _ => Err("not a package".into()),
    }
}

/// The text nodes of the `N`th Word paragraph, counted as the content reader
/// counts them.
fn word_nodes(spans: &[Spanned], locator: &str) -> Result<Vec<Node>, String> {
    let wanted: usize = locator.strip_prefix("p:").and_then(|n| n.parse().ok()).unwrap_or(0);
    let mut number = 0usize;
    let mut index = 0usize;
    while index < spans.len() {
        if let Event::Start { empty, .. } = &spans[index].event {
            if spans[index].event.local_name() == Some("p") {
                number += 1;
                if number == wanted {
                    if *empty {
                        return Ok(Vec::new());
                    }
                    return paragraph_nodes(&spans[index + 1..], "t", &["instrText", "delText"]);
                }
            }
        }
        index += 1;
    }
    Err(format!("the document has no paragraph {wanted}"))
}

/// Text nodes of `text_element` inside one paragraph, up to its end. A nested
/// paragraph (a text box) is refused rather than walked.
fn paragraph_nodes(after_start: &[Spanned], text_element: &str, skipped: &[&str]) -> Result<Vec<Node>, String> {
    let mut nodes = Vec::new();
    let mut inside_text = false;
    let mut skip = 0usize;
    for spanned in after_start {
        match &spanned.event {
            Event::Start { empty, .. } => match spanned.event.local_name() {
                Some("p") => return Err("the paragraph holds a text box, which is not edited in place".into()),
                Some(name) if name == text_element && !empty => inside_text = true,
                Some(name) if skipped.contains(&name) && !empty => skip += 1,
                _ => {}
            },
            Event::End { .. } => match spanned.event.local_name() {
                Some("p") => return Ok(nodes),
                Some(name) if name == text_element => inside_text = false,
                Some(name) if skipped.contains(&name) => skip = skip.saturating_sub(1),
                _ => {}
            },
            Event::Text(text) if inside_text && skip == 0 => nodes.push(Node {
                start: spanned.start,
                end: spanned.end,
                text: text.clone(),
            }),
            Event::Text(_) => {}
        }
    }
    Err("the paragraph is never closed".into())
}

/// The text nodes behind `slide:N/title` or `slide:N/body:K`, walked exactly as
/// the content reader walks a slide.
fn slide_nodes(spans: &[Spanned], locator: &str) -> Result<Vec<Node>, String> {
    let target = locator.split_once('/').map(|(_, t)| t).unwrap_or_default();
    let wanted_body: Option<usize> = target.strip_prefix("body:").and_then(|k| k.parse().ok());
    if target != "title" && wanted_body.is_none() {
        return Err(format!("{locator:?} names neither the title nor a body paragraph"));
    }
    let mut shape_depth = 0usize;
    let mut shape_is_title = false;
    let mut body = 0usize;
    let mut title_nodes = Vec::new();
    let mut index = 0usize;
    while index < spans.len() {
        let spanned = &spans[index];
        match &spanned.event {
            Event::Start { name, empty, .. } => {
                if name == "p:sp" && !empty {
                    shape_depth += 1;
                    if shape_depth == 1 {
                        shape_is_title = false;
                    }
                } else if spanned.event.local_name() == Some("cNvPr") && shape_depth > 0 {
                    if spanned
                        .event
                        .attribute("name")
                        .is_some_and(|n| n.to_ascii_lowercase().starts_with("title"))
                    {
                        shape_is_title = true;
                    }
                } else if spanned.event.local_name() == Some("ph") {
                    let kind = spanned.event.attribute("type").unwrap_or_default();
                    if kind == "title" || kind == "ctrTitle" {
                        shape_is_title = true;
                    }
                } else if name == "a:p" {
                    let nodes = if *empty { Vec::new() } else { paragraph_nodes(&spans[index + 1..], "t", &[])? };
                    if shape_is_title {
                        title_nodes.extend(nodes);
                    } else {
                        body += 1;
                        if wanted_body == Some(body) {
                            return Ok(nodes);
                        }
                    }
                }
            }
            Event::End { name } if name == "p:sp" => shape_depth = shape_depth.saturating_sub(1),
            _ => {}
        }
        index += 1;
    }
    if target == "title" {
        return Ok(title_nodes);
    }
    Err(format!("the slide has no body paragraph {}", wanted_body.unwrap_or(0)))
}

type CellSplice = (usize, usize, String, Applied, Option<String>);

/// The whole `<c>` element for a cell, rewritten.
fn cell_splice<R: std::io::Read + std::io::Seek>(
    spans: &[Spanned],
    xml: &str,
    edit: &Edit,
    archive: &mut zip::ZipArchive<R>,
) -> Result<CellSplice, String> {
    let (_, cell) = edit.locator.split_once('!').ok_or("not a cell locator")?;
    let start_index = spans
        .iter()
        .position(|s| s.event.local_name() == Some("c") && s.event.attribute("r") == Some(cell))
        .ok_or_else(|| format!("{} holds no value; adding a cell is not a targeted edit", edit.locator))?;
    let opening = &spans[start_index];
    let (end_index, empty) = match &opening.event {
        Event::Start { empty: true, .. } => (start_index, true),
        _ => (
            spans[start_index..]
                .iter()
                .position(|s| matches!(&s.event, Event::End { .. }) && s.event.local_name() == Some("c"))
                .map(|offset| start_index + offset)
                .ok_or("the cell is never closed")?,
            false,
        ),
    };
    let inner = &spans[start_index..=end_index];
    if inner.iter().any(|s| s.event.local_name() == Some("f")) {
        return Err(format!(
            "{} holds a formula; its value is computed, so edit the cells it reads instead",
            edit.locator
        ));
    }
    let kind = opening.event.attribute("t").unwrap_or("n").to_string();
    let mut raw_value = String::new();
    let mut field = false;
    for s in inner {
        match &s.event {
            Event::Start { .. } if matches!(s.event.local_name(), Some("v") | Some("t")) => field = true,
            Event::End { .. } if matches!(s.event.local_name(), Some("v") | Some("t")) => field = false,
            Event::Text(text) if field => raw_value.push_str(text),
            _ => {}
        }
    }
    let shown = if kind == "s" {
        let index: usize = raw_value.trim().parse().map_err(|_| "a shared-string cell without an index")?;
        shared_string(archive, index)?
    } else {
        raw_value
    };
    if empty && shown.is_empty() && edit.find.is_some() {
        return Err(format!("{} is empty, so there is nothing to find in it", edit.locator));
    }
    let after = replace_in(&shown, edit.find.as_deref(), &edit.replace, &edit.locator)?;

    // Attributes kept as written, except the type, which follows the value.
    let attributes: String = match &opening.event {
        Event::Start { attributes, .. } => attributes
            .iter()
            .filter(|(name, _)| name != "t")
            .map(|(name, value)| format!(" {name}=\"{}\"", super::ooxml::escape(value)))
            .collect(),
        _ => String::new(),
    };
    let numeric = matches!(kind.as_str(), "n") && after.trim().parse::<f64>().is_ok_and(f64::is_finite);
    let replacement = if numeric {
        format!("<c{attributes}><v>{}</v></c>", after.trim())
    } else {
        format!("<c{attributes} t=\"inlineStr\"><is><t xml:space=\"preserve\">{}</t></is></c>", escape_text(&after))
    };
    let note = (kind == "s").then(|| {
        format!(
            "{} was a shared string; it now holds its own text, so other cells showing the same words are unchanged",
            edit.locator
        )
    });
    let _ = xml;
    Ok((
        opening.start,
        spans[end_index].end,
        replacement,
        Applied { locator: edit.locator.clone(), before: shown, after: if numeric { after.trim().to_string() } else { after } },
        note,
    ))
}

fn shared_string<R: std::io::Read + std::io::Seek>(archive: &mut zip::ZipArchive<R>, wanted: usize) -> Result<String, String> {
    let raw = package::read_bounded(archive, "xl/sharedStrings.xml", package::LIMITS.max_entry_bytes)
        .ok_or("the workbook has no shared strings")?;
    let text = String::from_utf8(raw).map_err(|_| "the shared strings are not UTF-8")?;
    let events = xml_events::scan(&text)?;
    let mut index = 0usize;
    let mut current: Option<String> = None;
    let mut in_text = false;
    for event in &events {
        match event {
            Event::Start { empty, .. } if event.local_name() == Some("si") => {
                if *empty {
                    if index == wanted {
                        return Ok(String::new());
                    }
                    index += 1;
                } else {
                    current = Some(String::new());
                }
            }
            Event::End { .. } if event.local_name() == Some("si") => {
                if index == wanted {
                    return Ok(current.take().unwrap_or_default());
                }
                index += 1;
                current = None;
            }
            Event::Start { empty: false, .. } if event.local_name() == Some("t") => in_text = true,
            Event::End { .. } if event.local_name() == Some("t") => in_text = false,
            Event::Text(t) if in_text => {
                if let Some(current) = current.as_mut() {
                    current.push_str(t);
                }
            }
            _ => {}
        }
    }
    Err(format!("shared string {wanted} does not exist"))
}

/// Whether the role at a locator is one this edits.
pub fn editable(role: UnitRole) -> bool {
    !matches!(role, UnitRole::SlideNotes | UnitRole::PageText)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::artifacts::content::tests_support::sample_docx;
    use std::io::Read;

    /// Every entry of a package, name → (crc, raw compressed bytes).
    fn raw_entries(bytes: &[u8]) -> std::collections::BTreeMap<String, (u32, Vec<u8>)> {
        let mut archive = zip::ZipArchive::new(Cursor::new(bytes)).unwrap();
        let mut out = std::collections::BTreeMap::new();
        for index in 0..archive.len() {
            let mut entry = archive.by_index_raw(index).unwrap();
            let mut raw = Vec::new();
            entry.read_to_end(&mut raw).unwrap();
            out.insert(entry.name().to_string(), (entry.crc32(), raw));
        }
        out
    }

    /// Adds a part this product never writes, the way a person's edit in Word
    /// would, so preservation of an *unsupported* part is what is tested.
    fn with_foreign_part(bytes: &[u8]) -> Vec<u8> {
        let mut archive = zip::ZipArchive::new(Cursor::new(bytes)).unwrap();
        let mut out = Cursor::new(Vec::new());
        {
            let mut writer = zip::ZipWriter::new(&mut out);
            for index in 0..archive.len() {
                writer.raw_copy_file(archive.by_index_raw(index).unwrap()).unwrap();
            }
            let options: zip::write::FileOptions<'_, ()> = zip::write::FileOptions::default();
            writer.start_file("customXml/item1.xml", options).unwrap();
            writer.write_all(b"<reviewer-notes>kept by a person, unknown to ARJUN</reviewer-notes>").unwrap();
            writer.finish().unwrap();
        }
        out.into_inner()
    }

    #[test]
    fn a_targeted_word_edit_changes_one_node_and_copies_every_other_part_raw() {
        let dir = tempfile::tempdir().unwrap();
        let original = with_foreign_part(&sample_docx(dir.path()));
        let model = content::extract(&original).unwrap();
        let finding = model.units.iter().find(|u| u.text.contains("8.2 mm")).unwrap().locator.clone();

        let outcome = apply(
            &original,
            &[Edit { locator: finding.clone(), find: Some("8.2 mm".into()), replace: "8.4 mm".into() }],
        )
        .expect("edits");
        assert_eq!(outcome.changed_parts, vec!["word/document.xml".to_string()]);

        let before = raw_entries(&original);
        let after = raw_entries(&outcome.bytes);
        assert_eq!(before.keys().collect::<Vec<_>>(), after.keys().collect::<Vec<_>>(), "no part added or lost");
        for (name, entry) in &before {
            if name != "word/document.xml" {
                assert_eq!(entry, &after[name], "{name} must be byte-identical");
            }
        }
        assert!(after.contains_key("customXml/item1.xml"), "the unsupported part survived");

        let reread = content::extract(&outcome.bytes).unwrap();
        let edited = reread.units.iter().find(|u| u.locator == finding).unwrap();
        assert!(edited.text.contains("8.4 mm") && !edited.text.contains("8.2 mm"));
        let d = content::diff(&model, &reread);
        assert_eq!(d.entries.len(), 1, "exactly one unit changed: {:#?}", d.entries);
    }

    #[test]
    fn an_ambiguous_or_missing_target_is_refused_and_nothing_is_produced() {
        let dir = tempfile::tempdir().unwrap();
        let original = sample_docx(dir.path());
        assert!(apply(&original, &[Edit { locator: "p:999".into(), find: None, replace: "x".into() }]).is_err());
        let model = content::extract(&original).unwrap();
        let finding = model.units.iter().find(|u| u.text.contains("8.2 mm")).unwrap().locator.clone();
        let error = apply(&original, &[Edit { locator: finding.clone(), find: Some("nine".into()), replace: "x".into() }]).unwrap_err();
        assert!(error.contains("does not contain"), "{error}");
        let error = apply(&original, &[Edit { locator: finding, find: Some("m".into()), replace: "x".into() }]).unwrap_err();
        assert!(error.contains("times"), "{error}");
        assert!(apply(&original, &[]).is_err());
    }

    #[test]
    fn a_workbook_cell_is_edited_and_a_formula_cell_is_refused() {
        use crate::artifacts::doc_model::{Column, ColumnType, Sheet, Workbook};
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("book.xlsx");
        crate::artifacts::xlsx::write_workbook_model(
            &path,
            &Workbook {
                title: "Readings".into(),
                classification: "Internal".into(),
                sheets: vec![Sheet {
                    name: "Readings".into(),
                    columns: vec![
                        Column { header: "Point".into(), kind: ColumnType::Text, width: None },
                        Column { header: "Measured".into(), kind: ColumnType::Number, width: None },
                        Column { header: "Margin".into(), kind: ColumnType::Formula, width: None },
                    ],
                    rows: vec![vec!["C".into(), "8.2".into(), "=B2-9".into()], vec!["D".into(), "9.4".into(), "=B3-9".into()]],
                    freeze_header: true,
                }],
            },
        )
        .unwrap();
        let original = std::fs::read(&path).unwrap();
        let outcome = apply(&original, &[Edit { locator: "sheet:Readings!B2".into(), find: None, replace: "8.4".into() }]).expect("edits");
        let reread = content::extract(&outcome.bytes).unwrap();
        assert_eq!(reread.units.iter().find(|u| u.locator == "sheet:Readings!B2").unwrap().text, "8.4");
        assert_eq!(reread.units.iter().find(|u| u.locator == "sheet:Readings!B3").unwrap().text, "9.4");
        assert!(outcome.bytes.len() > 100);

        let error = apply(&original, &[Edit { locator: "sheet:Readings!C2".into(), find: None, replace: "1".into() }]).unwrap_err();
        assert!(error.contains("formula"), "{error}");
    }

    #[test]
    fn a_slide_body_line_is_edited_in_place() {
        use crate::artifacts::doc_model::{Deck, SlideModel};
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("deck.pptx");
        crate::artifacts::pptx::write_deck_model(
            &path,
            &Deck {
                title: "Unit Four".into(),
                classification: "Internal".into(),
                slides: vec![SlideModel {
                    heading: "Findings".into(),
                    bullets: vec!["Point C measured 8.2 mm.".into(), "Point D measured 9.4 mm.".into()],
                    table: None,
                    notes: Some("Speaker notes stay as they are.".into()),
                }],
            },
            false,
        )
        .unwrap();
        let original = std::fs::read(&path).unwrap();
        let model = content::extract(&original).unwrap();
        let target = model.units.iter().find(|u| u.text.contains("Point C")).unwrap().locator.clone();
        let outcome = apply(&original, &[Edit { locator: target.clone(), find: Some("8.2".into()), replace: "8.4".into() }]).expect("edits");
        let reread = content::extract(&outcome.bytes).unwrap();
        assert!(reread.units.iter().any(|u| u.locator == target && u.text.contains("8.4 mm")));
        assert!(reread.units.iter().any(|u| u.text.contains("Speaker notes stay")));
        assert_eq!(outcome.changed_parts.len(), 1);
        assert!(crate::artifacts::pptx::check_deck(&{
            let p = dir.path().join("edited.pptx");
            std::fs::write(&p, &outcome.bytes).unwrap();
            p
        })
        .opens);
    }

    #[test]
    fn a_pdf_is_not_edited_in_place() {
        let error = apply(b"%PDF-1.7 nothing", &[Edit { locator: "page:1".into(), find: None, replace: "x".into() }]).unwrap_err();
        assert!(!error.is_empty());
    }
}
