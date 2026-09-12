//! Producing a Word document that opens, from a template nobody can improvise.
//!
//! ARJUN design rule 29: *"The artifact service should use deterministic templates wherever
//! possible. The language model supplies content, but the template engine
//! controls document structure, formatting, calculations, and required fields."*
//!
//! That division is the whole design. A model asked to produce an approval note
//! writes a *plausible* approval note — usually with the right headings, and
//! occasionally without the approval block, or with the classification label
//! quietly dropped because the content did not seem to need one. A template
//! cannot do that. It has a fixed set of sections, and a missing required field
//! is an error before a file exists rather than a gap somebody notices at
//! signing.
//!
//! ## Written as XML rather than through a document library
//!
//! A `.docx` is a ZIP of XML parts, and the parts are written here directly.
//! That keeps the document's structure legible in this repository: a reviewer
//! asking "does the approval note always carry a classification label?" can read
//! the answer rather than trusting a library's behaviour. It also keeps the
//! dependency small, which matters for a build that has to work offline.
//!
//! Escaping and packaging live in [`super::ooxml`], shared with the workbook and
//! deck writers so there is one implementation of each to get right.

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use super::ooxml::{escape, list_parts, placeholders_in, read_part, write_parts};

/// A section the template requires, and what it is for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FieldSpec {
    pub key: &'static str,
    /// Heading printed above it. Empty for the title.
    pub heading: &'static str,
    pub required: bool,
}

/// The approval note — the deliverable the problem statement's own example asks
/// for, and the one a refinery signs most often.
///
/// The order here is the order in the document. It follows how an approval note
/// is actually read: what it concerns, what was found, what follows from that,
/// what it rests on, and only then who signs.
pub const APPROVAL_NOTE: &[FieldSpec] = &[
    FieldSpec { key: "title", heading: "", required: true },
    FieldSpec { key: "recipient", heading: "To", required: true },
    FieldSpec { key: "subject", heading: "Subject", required: true },
    FieldSpec { key: "findings", heading: "Findings", required: true },
    FieldSpec { key: "calculation", heading: "Calculation", required: false },
    FieldSpec { key: "recommendation", heading: "Recommendation", required: true },
    FieldSpec { key: "references", heading: "Supporting references", required: true },
    // Required even when empty, because "we assumed nothing" is itself a claim a
    // reviewer should see stated rather than infer from an absent section.
    FieldSpec { key: "assumptions", heading: "Assumptions", required: true },
];

/// Templates this service can produce.
pub fn template_for(name: &str) -> Option<&'static [FieldSpec]> {
    match name {
        "approval_note" => Some(APPROVAL_NOTE),
        _ => None,
    }
}

/// Everything stamped onto a document regardless of its template.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DocumentMetadata {
    pub task_id: String,
    pub created_at: String,
    /// Model that supplied the content, so a reader knows what produced it.
    pub model: String,
    /// The document's sensitivity, printed on the page rather than only stored.
    pub classification: String,
    /// Whether the verifier passed. A draft says so on its face.
    pub is_draft: bool,
}

fn paragraph(text: &str, style: Option<&str>) -> String {
    let properties = match style {
        Some(s) => format!("<w:pPr><w:pStyle w:val=\"{s}\"/></w:pPr>"),
        None => String::new(),
    };
    format!(
        "<w:p>{properties}<w:r><w:t xml:space=\"preserve\">{}</w:t></w:r></w:p>",
        escape(text)
    )
}

fn heading(text: &str) -> String {
    format!(
        "<w:p><w:pPr><w:pStyle w:val=\"Heading1\"/></w:pPr>\
         <w:r><w:rPr><w:b/></w:rPr><w:t xml:space=\"preserve\">{}</w:t></w:r></w:p>",
        escape(text)
    )
}

/// Builds `word/document.xml` from the template and the supplied content.
fn document_xml(
    template: &[FieldSpec],
    content: &BTreeMap<String, String>,
    metadata: &DocumentMetadata,
) -> String {
    let mut body = String::new();

    // The draft banner comes first, before anything a reader might act on.
    // A document that is not ready has to say so where it cannot be missed.
    if metadata.is_draft {
        body.push_str(&paragraph(
            "DRAFT — not verified. Do not act on this document until it has been reviewed.",
            Some("Heading1"),
        ));
    }

    body.push_str(&paragraph(
        &format!("Classification: {}", metadata.classification),
        None,
    ));

    for field in template {
        let Some(value) = content.get(field.key) else {
            continue;
        };

        if field.heading.is_empty() {
            body.push_str(&heading(value));
        } else {
            body.push_str(&heading(field.heading));
            for line in value.lines() {
                body.push_str(&paragraph(line, None));
            }
        }
    }

    // Provenance, at the foot where it belongs — present on every document so a
    // reader can always trace one back to the task that produced it.
    body.push_str(&heading("How this was produced"));
    body.push_str(&paragraph(
        &format!(
            "Task {} · generated {} · content by {} · figures computed by ARJUN's calculation \
             engine, not by the model.",
            metadata.task_id, metadata.created_at, metadata.model
        ),
        None,
    ));

    // The visible watermark (see `artifacts::visible_watermark`) is stamped
    // onto the body itself rather than into a header/footer relationship
    // part, so it survives copy-paste and shows up in any reader, not just
    // ones that render page margins. It carries the same fields a reviewer
    // would look for in the audit log, so a watermark that disagrees with
    // the audit entry is itself a discrepancy worth investigating.
    let stamp = super::visible_watermark::stamp_from_metadata(
        &metadata.task_id,
        &metadata.model,
        &metadata.created_at,
        &metadata.classification,
        metadata.is_draft,
    );
    body = super::visible_watermark::apply_to_docx_body(&body, &stamp);

    format!(
        r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:body>{body}<w:sectPr><w:pgSz w:w="11906" w:h="16838"/></w:sectPr></w:body>
</w:document>"#
    )
}

const CONTENT_TYPES: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>"#;

const ROOT_RELS: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"#;

/// What went wrong before a file existed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TemplateError {
    pub message: String,
    /// Fields the template requires that the content did not supply.
    pub missing: Vec<String>,
}

/// Writes an approval note, or explains why it could not.
///
/// Required fields are checked *before* anything is written, so a failure never
/// leaves a half-formed document on disk for somebody to find and use.
pub fn write_document(
    path: &Path,
    template_name: &str,
    content: &BTreeMap<String, String>,
    metadata: &DocumentMetadata,
) -> Result<(), TemplateError> {
    let Some(template) = template_for(template_name) else {
        return Err(TemplateError {
            message: format!(
                "There is no {template_name:?} template. Available templates: approval_note."
            ),
            missing: Vec::new(),
        });
    };

    let missing: Vec<String> = template
        .iter()
        .filter(|field| field.required)
        .filter(|field| {
            content
                .get(field.key)
                .map(|v| v.trim().is_empty())
                .unwrap_or(true)
        })
        .map(|field| field.key.to_string())
        .collect();

    if !missing.is_empty() {
        return Err(TemplateError {
            message: format!(
                "The approval note template requires {} that were not supplied: {}. Nothing was \
                 written.",
                if missing.len() == 1 { "a field" } else { "fields" },
                missing.join(", ")
            ),
            missing,
        });
    }

    let parts = [
        ("[Content_Types].xml", CONTENT_TYPES.to_string()),
        ("_rels/.rels", ROOT_RELS.to_string()),
        ("word/document.xml", document_xml(template, content, metadata)),
    ];

    write_parts(path, &parts).map_err(|e| TemplateError {
        message: format!("The document could not be written: {e}"),
        missing: Vec::new(),
    })
}

/// Renders a [`crate::artifacts::doc_model::Document`] into a `.docx`.
///
/// ## Why this exists beside `write_document`
///
/// `write_document` renders a *template*: a fixed list of fields, of which
/// exactly one exists (`approval_note`). It is how every Word deliverable this
/// product has produced, and asking it for a maintenance procedure, an
/// inspection report or a set of minutes gets "Available templates:
/// approval_note."
///
/// This renders a document the model composed: real heading levels, lists,
/// tables, page breaks and document properties. The template path is untouched
/// and still works.
///
/// The model is validated before a byte is written. A ragged table or a heading
/// hierarchy that skips a level is caught in memory, where it can be repaired,
/// rather than found by re-opening the file afterwards.
pub fn write_document_model(
    path: &Path,
    document: &crate::artifacts::doc_model::Document,
    metadata: &DocumentMetadata,
) -> Result<(), TemplateError> {
    let problems = document.problems();
    if !problems.is_empty() {
        return Err(TemplateError {
            message: format!(
                "The document is not fit to render: {}. Nothing was written.",
                problems.join("; ")
            ),
            missing: Vec::new(),
        });
    }

    let parts = [
        ("[Content_Types].xml", MODEL_CONTENT_TYPES.to_string()),
        ("_rels/.rels", MODEL_ROOT_RELS.to_string()),
        ("docProps/core.xml", core_properties(document, metadata)),
        ("word/document.xml", model_document_xml(document, metadata)),
    ];

    write_parts(path, &parts).map_err(|e| TemplateError {
        message: format!("The document could not be written: {e}"),
        missing: Vec::new(),
    })
}

/// `docProps/core.xml` - what Word's Properties panel shows.
///
/// A deliverable that opens as untitled with no author is one somebody has to
/// fill in by hand before they can file it.
fn core_properties(
    document: &crate::artifacts::doc_model::Document,
    metadata: &DocumentMetadata,
) -> String {
    let author = document
        .properties
        .author
        .clone()
        .unwrap_or_else(|| "ARJUN".to_string());
    let subject = document.properties.subject.clone().unwrap_or_default();
    let keywords = document.properties.keywords.join(", ");
    format!(
        r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
<dc:title>{}</dc:title><dc:creator>{}</dc:creator><cp:lastModifiedBy>{}</cp:lastModifiedBy>
<dc:subject>{}</dc:subject><cp:keywords>{}</cp:keywords>
<dcterms:created xsi:type="dcterms:W3CDTF">{}</dcterms:created>
</cp:coreProperties>"#,
        escape(&document.title),
        escape(&author),
        escape(&author),
        escape(&subject),
        escape(&keywords),
        escape(&metadata.created_at),
    )
}

/// One heading at its real outline level.
///
/// `Heading1`..`Heading6`, so Word's navigation pane, a generated table of
/// contents and a screen reader all see the structure the author intended. The
/// template path emits `Heading1` for everything, which is why a templated
/// document has a flat outline.
fn heading_at(text: &str, level: u8) -> String {
    let level = level.clamp(1, 6);
    format!(
        "<w:p><w:pPr><w:pStyle w:val=\"Heading{level}\"/><w:outlineLvl w:val=\"{}\"/></w:pPr>\
         <w:r><w:rPr><w:b/></w:rPr><w:t xml:space=\"preserve\">{}</w:t></w:r></w:p>",
        level - 1,
        escape(text)
    )
}

/// One list item, indented with its marker.
///
/// Rendered as a real indented paragraph rather than through a numbering
/// definition part: a numbering part a reader does not resolve shows as an
/// unmarked paragraph, and this shows as a list in every reader, which is the
/// property a deliverable needs.
fn list_paragraph(text: &str, numbered: bool, index: usize) -> String {
    let marker = if numbered {
        format!("{}.", index + 1)
    } else {
        "\u{2022}".to_string()
    };
    format!(
        "<w:p><w:pPr><w:ind w:left=\"360\" w:hanging=\"360\"/></w:pPr>\
         <w:r><w:t xml:space=\"preserve\">{} {}</w:t></w:r></w:p>",
        marker,
        escape(text)
    )
}

fn table_xml(header: &[String], rows: &[Vec<String>], caption: Option<&str>) -> String {
    let mut out = String::new();
    if let Some(caption) = caption.filter(|c| !c.trim().is_empty()) {
        out.push_str(&paragraph(caption, None));
    }
    out.push_str(
        "<w:tbl><w:tblPr><w:tblStyle w:val=\"TableGrid\"/>\
         <w:tblW w:w=\"5000\" w:type=\"pct\"/>\
         <w:tblBorders>\
         <w:top w:val=\"single\" w:sz=\"4\" w:color=\"auto\"/>\
         <w:left w:val=\"single\" w:sz=\"4\" w:color=\"auto\"/>\
         <w:bottom w:val=\"single\" w:sz=\"4\" w:color=\"auto\"/>\
         <w:right w:val=\"single\" w:sz=\"4\" w:color=\"auto\"/>\
         <w:insideH w:val=\"single\" w:sz=\"4\" w:color=\"auto\"/>\
         <w:insideV w:val=\"single\" w:sz=\"4\" w:color=\"auto\"/>\
         </w:tblBorders></w:tblPr>",
    );

    // The header row is marked `tblHeader`, so it repeats when the table breaks
    // across a page. A printed table whose headings appear only on the first
    // page is one a reader has to hold in their head.
    out.push_str("<w:tr><w:trPr><w:tblHeader/></w:trPr>");
    for cell in header {
        out.push_str(&format!(
            "<w:tc><w:tcPr><w:tcW w:w=\"0\" w:type=\"auto\"/></w:tcPr>\
             <w:p><w:r><w:rPr><w:b/></w:rPr><w:t xml:space=\"preserve\">{}</w:t></w:r></w:p></w:tc>",
            escape(cell)
        ));
    }
    out.push_str("</w:tr>");

    for row in rows {
        out.push_str("<w:tr>");
        for cell in row {
            out.push_str(&format!(
                "<w:tc><w:tcPr><w:tcW w:w=\"0\" w:type=\"auto\"/></w:tcPr>\
                 <w:p><w:r><w:t xml:space=\"preserve\">{}</w:t></w:r></w:p></w:tc>",
                escape(cell)
            ));
        }
        out.push_str("</w:tr>");
    }
    out.push_str("</w:tbl>");
    // A table butted against the next paragraph is a layout fault in Word.
    out.push_str("<w:p/>");
    out
}

fn model_document_xml(
    document: &crate::artifacts::doc_model::Document,
    metadata: &DocumentMetadata,
) -> String {
    use crate::artifacts::doc_model::Block;

    let mut body = String::new();

    if metadata.is_draft {
        body.push_str(&paragraph(
            "DRAFT - not verified. Do not act on this document until it has been reviewed.",
            Some("Heading1"),
        ));
    }
    body.push_str(&heading_at(&document.title, 1));
    body.push_str(&paragraph(
        &format!("Classification: {}", document.classification),
        None,
    ));

    for section in &document.sections {
        // Section headings sit one level below the document title, which is
        // level 1. A section the model called level 1 is therefore a level 2
        // heading in the file, and the outline reads correctly from the top.
        body.push_str(&heading_at(&section.heading, section.level.saturating_add(1)));
        for block in &section.blocks {
            match block {
                Block::Paragraph { text } => {
                    for line in text.lines() {
                        body.push_str(&paragraph(line, None));
                    }
                }
                Block::Bullets { items } => {
                    for (index, item) in items.iter().enumerate() {
                        body.push_str(&list_paragraph(item, false, index));
                    }
                }
                Block::Numbered { items } => {
                    for (index, item) in items.iter().enumerate() {
                        body.push_str(&list_paragraph(item, true, index));
                    }
                }
                Block::Table { header, rows, caption } => {
                    body.push_str(&table_xml(header, rows, caption.as_deref()));
                }
                Block::PageBreak => {
                    body.push_str("<w:p><w:r><w:br w:type=\"page\"/></w:r></w:p>");
                }
            }
        }
    }

    body.push_str(&heading_at("How this was produced", 2));
    body.push_str(&paragraph(
        &format!(
            "Task {} - generated {} - content by {} - figures computed by ARJUN's calculation \
             engine, not by the model.",
            metadata.task_id, metadata.created_at, metadata.model
        ),
        None,
    ));

    let stamp = super::visible_watermark::stamp_from_metadata(
        &metadata.task_id,
        &metadata.model,
        &metadata.created_at,
        &metadata.classification,
        metadata.is_draft,
    );
    body = super::visible_watermark::apply_to_docx_body(&body, &stamp);

    format!(
        r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:body>{body}<w:sectPr><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1134" w:right="1134" w:bottom="1134" w:left="1134"/></w:sectPr></w:body>
</w:document>"#
    )
}

/// Content types for the model path, which carries `docProps/core.xml`.
const MODEL_CONTENT_TYPES: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
</Types>"#;

const MODEL_ROOT_RELS: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
</Relationships>"#;

/// What re-opening a produced document found.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DocumentCheck {
    pub opens: bool,
    /// Headings found inside. Compared against what the template promised.
    pub sections: Vec<String>,
    pub problems: Vec<String>,
}

impl DocumentCheck {
    pub fn is_sound(&self) -> bool {
        self.opens && self.problems.is_empty()
    }
}

/// Re-opens a produced document and checks it is what it claims to be.
///
/// ARJUN design rule 30 asks that the application open the generated file locally and
/// confirm it is not corrupt and that required sections exist. Checking the
/// file rather than the code that wrote it is the point: a bug between the
/// template and the ZIP would pass every test of the template and still produce
/// something Word refuses to open.
pub fn check_document(path: &Path, template_name: &str) -> DocumentCheck {
    let body = match read_part(path, "word/document.xml") {
        Ok(body) => body,
        Err(error) => {
            return DocumentCheck {
                opens: false,
                sections: Vec::new(),
                problems: vec![format!("{path_display}: {error}", path_display = path.display())],
            }
        }
    };

    let mut problems = Vec::new();
    let present = list_parts(path).unwrap_or_default();
    for required in ["[Content_Types].xml", "_rels/.rels", "word/document.xml"] {
        if !present.iter().any(|p| p == required) {
            problems.push(format!("the document is missing {required}"));
        }
    }

    // Headings are what the template promised, so they are what is checked.
    let sections: Vec<String> = body
        .split("Heading1")
        .skip(1)
        .filter_map(|fragment| {
            let start = fragment.find("<w:t xml:space=\"preserve\">")?;
            let rest = &fragment[start + 26..];
            let end = rest.find("</w:t>")?;
            Some(rest[..end].to_string())
        })
        .collect();

    if let Some(template) = template_for(template_name) {
        for field in template.iter().filter(|f| f.required && !f.heading.is_empty()) {
            if !sections.iter().any(|s| s == field.heading) {
                problems.push(format!("the {:?} section is missing", field.heading));
            }
        }
    }

    for marker in placeholders_in(&body) {
        problems.push(format!(
            "the document still contains the placeholder {marker:?} — a field was never filled in"
        ));
    }

    if body.len() < 200 {
        problems.push("the document has almost no content in it".to_string());
    }

    DocumentCheck { opens: true, sections, problems }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metadata() -> DocumentMetadata {
        DocumentMetadata {
            task_id: "task-42".into(),
            created_at: "2026-08-26T10:00:00Z".into(),
            model: "Qwen2.5-7B-Instruct".into(),
            classification: "P&ID / process diagram".into(),
            is_draft: false,
        }
    }

    fn complete_note() -> BTreeMap<String, String> {
        let mut content = BTreeMap::new();
        content.insert("title".into(), "Approval note — PV-2201 wall thickness".into());
        content.insert("recipient".into(), "M. Rao, Maintenance".into());
        content.insert("subject".into(), "Thickness below minimum on PV-2201".into());
        content.insert(
            "findings".into(),
            "Measured 8.2 mm against a minimum of 9.0 mm [E1].".into(),
        );
        content.insert("recommendation".into(), "Replace within 90 days.".into());
        content.insert("references".into(), "Maintenance SOP rev C, page 4.".into());
        content.insert("assumptions".into(), "None.".into());
        content
    }

    fn temp() -> tempfile::TempDir {
        tempfile::tempdir().unwrap()
    }

    #[test]
    fn a_complete_note_is_written_and_reopens_soundly() {
        let dir = temp();
        let path = dir.path().join("note.docx");

        write_document(&path, "approval_note", &complete_note(), &metadata()).unwrap();

        let check = check_document(&path, "approval_note");
        assert!(check.is_sound(), "{:?}", check.problems);
        assert!(check.sections.contains(&"Findings".to_string()));
        assert!(check.sections.contains(&"Recommendation".to_string()));
    }

    /// A missing required field is an error before a file exists, not a gap
    /// somebody notices at signing.
    #[test]
    fn a_missing_required_field_refuses_and_writes_nothing() {
        let dir = temp();
        let path = dir.path().join("note.docx");

        let mut content = complete_note();
        content.remove("recommendation");

        let error = write_document(&path, "approval_note", &content, &metadata()).unwrap_err();

        assert_eq!(error.missing, vec!["recommendation"]);
        assert!(error.message.contains("Nothing was written"));
        assert!(!path.exists(), "no half-formed file should be left behind");
    }

    #[test]
    fn every_missing_field_is_named_at_once() {
        let dir = temp();
        let mut content = complete_note();
        content.remove("findings");
        content.remove("references");

        let error =
            write_document(&dir.path().join("n.docx"), "approval_note", &content, &metadata())
                .unwrap_err();

        assert_eq!(error.missing, vec!["findings", "references"]);
    }

    /// "We assumed nothing" is a claim a reviewer should see stated.
    #[test]
    fn an_empty_assumptions_section_is_still_required() {
        let dir = temp();
        let mut content = complete_note();
        content.insert("assumptions".into(), "   ".into());

        let error =
            write_document(&dir.path().join("n.docx"), "approval_note", &content, &metadata())
                .unwrap_err();
        assert_eq!(error.missing, vec!["assumptions"]);
    }

    #[test]
    fn an_unknown_template_names_the_ones_that_exist() {
        let dir = temp();
        let error = write_document(
            &dir.path().join("n.docx"),
            "invoice",
            &complete_note(),
            &metadata(),
        )
        .unwrap_err();

        assert!(error.message.contains("no \"invoice\" template"));
        assert!(error.message.contains("approval_note"));
    }

    // ── Escaping ─────────────────────────────────────────────────────────

    /// A `<` in a finding is enough to produce a file Word will not open.
    #[test]
    fn model_content_containing_xml_still_produces_a_readable_document() {
        let dir = temp();
        let path = dir.path().join("note.docx");

        let mut content = complete_note();
        content.insert(
            "findings".into(),
            "Thickness < 9.0 mm & flagged as \"severe\" <w:p>injected</w:p>".into(),
        );

        write_document(&path, "approval_note", &content, &metadata()).unwrap();
        let check = check_document(&path, "approval_note");
        assert!(check.is_sound(), "{:?}", check.problems);
    }

    // ── Draft standing ───────────────────────────────────────────────────

    /// A document that is not verified has to say so where it cannot be missed.
    #[test]
    fn a_draft_carries_its_warning_before_anything_else() {
        let dir = temp();
        let path = dir.path().join("draft.docx");

        let mut meta = metadata();
        meta.is_draft = true;
        write_document(&path, "approval_note", &complete_note(), &meta).unwrap();

        let check = check_document(&path, "approval_note");
        assert_eq!(
            check.sections[0],
            "DRAFT — not verified. Do not act on this document until it has been reviewed."
        );
    }

    #[test]
    fn a_verified_document_carries_no_draft_warning() {
        let dir = temp();
        let path = dir.path().join("final.docx");
        write_document(&path, "approval_note", &complete_note(), &metadata()).unwrap();

        let check = check_document(&path, "approval_note");
        assert!(!check.sections.iter().any(|s| s.contains("DRAFT")));
    }

    #[test]
    fn every_document_records_how_it_was_produced() {
        let dir = temp();
        let path = dir.path().join("note.docx");
        write_document(&path, "approval_note", &complete_note(), &metadata()).unwrap();

        let check = check_document(&path, "approval_note");
        assert!(check.sections.iter().any(|s| s == "How this was produced"));
    }

    // ── Checking the file, not the code that wrote it ────────────────────

    /// A document that reaches a reviewer still saying `TBD` has failed in the
    /// way that matters most: it looks finished.
    #[test]
    fn a_placeholder_left_in_the_content_fails_the_post_render_check() {
        let dir = temp();
        let path = dir.path().join("note.docx");

        let mut content = complete_note();
        content.insert("recommendation".into(), "TBD".into());

        write_document(&path, "approval_note", &content, &metadata()).unwrap();

        let check = check_document(&path, "approval_note");
        assert!(!check.is_sound());
        assert!(check.problems.iter().any(|p| p.contains("placeholder \"TBD\"")));
    }

    #[test]
    fn a_file_that_is_not_a_document_is_reported_as_corrupt() {
        let dir = temp();
        let path = dir.path().join("not-really.docx");
        std::fs::write(&path, "this is not a zip archive").unwrap();

        let check = check_document(&path, "approval_note");
        assert!(!check.is_sound());
        assert!(!check.opens);
    }

    #[test]
    fn a_missing_file_is_reported_rather_than_panicking() {
        let dir = temp();
        let check = check_document(&dir.path().join("absent.docx"), "approval_note");
        assert!(!check.is_sound());
        assert!(!check.opens);
    }
}
