//! Replacing one marked section of a Word document, and nothing else (P09).
//!
//! The patch is a splice: the bytes of `word/document.xml` before the
//! section's opening marker and after its closing marker are kept exactly,
//! every other part of the package is copied without being decompressed, and
//! the result is checked for all of that before it is returned. Styles,
//! headers, footers, numbering, page settings, relationships and every other
//! section therefore survive by construction, and the check proves it.
//!
//! What the writer cannot carry through a rewrite of the section itself —
//! pictures, embedded objects, comments, tracked changes, content controls,
//! fields, notes, linked hyperlinks, section breaks, other bookmarks — is a
//! refusal before anything is written, naming what is there. So is a
//! document that records revisions: a silent rewrite of a tracked document
//! is an untracked change.

use serde::Serialize;

use super::authoring::{read_sections, section_inner_xml, AuthoredSection, Dialect, SectionRange, BULLET_NUM_ID};
use super::package::{self, DetectedFormat};
use super::xml_events::{self, Event};

/// Elements a section rewrite would drop, and what each is called.
const UNPRESERVABLE: &[(&str, &str)] = &[
    ("drawing", "a picture or chart"),
    ("pict", "a legacy picture"),
    ("object", "an embedded object"),
    ("commentRangeStart", "a comment"),
    ("commentReference", "a comment"),
    ("ins", "a tracked insertion"),
    ("del", "a tracked deletion"),
    ("moveFrom", "a tracked move"),
    ("moveTo", "a tracked move"),
    ("sdt", "a content control"),
    ("fldSimple", "a field"),
    ("fldChar", "a field"),
    ("instrText", "a field"),
    ("footnoteReference", "a footnote"),
    ("endnoteReference", "an endnote"),
    ("sectPr", "a section break carrying page settings"),
    ("bookmarkStart", "another bookmark"),
    ("altChunk", "an imported part"),
    ("AlternateContent", "alternate content"),
    ("oMath", "an equation"),
    ("oMathPara", "an equation"),
    ("customXml", "custom XML markup"),
    ("smartTag", "a smart tag"),
];

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PatchOutcome {
    #[serde(skip)]
    pub bytes: Vec<u8>,
    pub section_id: String,
    pub before: String,
    pub after: String,
    /// Every part other than `word/document.xml`, compared after the write.
    pub parts_identical: Vec<String>,
    /// Bytes of `word/document.xml` kept verbatim before and after the section.
    pub prefix_bytes: usize,
    pub suffix_bytes: usize,
    /// Every other section, compared after the write.
    pub sections_identical: Vec<String>,
    /// How lists were written into this document.
    pub list_dialect: &'static str,
}

/// What the document offers the replacement: its own bullet numbering and
/// styles when it has the ones the authoring writer makes.
fn dialect_of(bytes: &[u8]) -> Dialect {
    let numbering = package::read_part(bytes, "word/numbering.xml").unwrap_or_default();
    let styles = package::read_part(bytes, "word/styles.xml").unwrap_or_default();
    Dialect {
        bullet_num_id: numbering.contains(&format!("w:numId=\"{BULLET_NUM_ID}\"")).then_some(BULLET_NUM_ID),
        styled: styles.contains("w:styleId=\"GapNote\"") && styles.contains("w:styleId=\"Caption\""),
    }
}

/// The heading level the section is written at now.
fn level_of(inner: &str) -> u8 {
    for level in 1..=6u8 {
        if let Some(position) = inner.find(&format!("w:val=\"Heading{level}\"")) {
            if position < inner.find("</w:p>").unwrap_or(usize::MAX) {
                return level;
            }
        }
    }
    1
}

/// What a rewrite of `inner` would lose, named.
fn unpreservable_in(inner: &str) -> Result<Vec<String>, String> {
    let events = xml_events::scan(&format!("<x xmlns:w=\"w\" xmlns:r=\"r\">{inner}</x>"))
        .map_err(|e| format!("the section's XML could not be read: {e}"))?;
    let mut found: Vec<String> = Vec::new();
    for event in &events {
        let Event::Start { name, .. } = event else { continue };
        let local = xml_events::local(name);
        let named = UNPRESERVABLE.iter().find(|(element, _)| *element == local).map(|(_, what)| (*what).to_string()).or_else(|| {
            (local == "hyperlink" && event.attribute("r:id").is_some()).then(|| "a hyperlink to an outside target".to_string())
        });
        if let Some(what) = named {
            if !found.contains(&what) {
                found.push(what);
            }
        }
    }
    Ok(found)
}

/// Replaces section `section_id` of the Word document `base` with
/// `replacement` (whose heading and level default to the section's own).
pub fn patch_section(base: &[u8], section_id: &str, replacement: &AuthoredSection) -> Result<PatchOutcome, String> {
    let report = package::inspect(base, &package::LIMITS);
    if !report.safe_to_open() {
        return Err(format!("the base version is not a safe package: {}", report.problems.join("; ")));
    }
    if report.detected != DetectedFormat::Docx {
        return Err(format!("document.patch_section edits Word documents; this version is {}", report.detected.label()));
    }
    if package::read_part(base, "word/settings.xml").is_ok_and(|s| s.contains("<w:trackRevisions")) {
        return Err("the document records tracked changes; a section rewrite would be an untracked change inside it. \
                    Accept or reject the tracked changes and turn tracking off first, or edit it in Word"
            .into());
    }
    let xml = package::read_part(base, "word/document.xml")?;
    let sections = read_sections(&xml)?;
    let matching: Vec<&SectionRange> = sections.iter().filter(|s| s.id == section_id).collect();
    let section = match matching.as_slice() {
        [one] => *one,
        [] => {
            return Err(format!(
                "the document has no section {section_id:?}; its sections are {}",
                if sections.is_empty() { "none (it carries no section markers)".to_string() } else { sections.iter().map(|s| s.id.as_str()).collect::<Vec<_>>().join(", ") }
            ))
        }
        _ => return Err(format!("the document marks {section_id:?} more than once; which one is meant cannot be told")),
    };
    if !section.body_level {
        return Err(format!("the markers of {section_id:?} sit inside a paragraph or table, not around whole paragraphs; a section rewrite would split them"));
    }
    if section.id == "provenance" {
        return Err("the provenance section records how the document was produced and is not edited".into());
    }
    let inner = &xml[section.inner_start..section.inner_end];
    let lost = unpreservable_in(inner)?;
    if !lost.is_empty() {
        return Err(format!(
            "section {section_id:?} holds {} which this editor cannot carry through a rewrite; nothing was written. Edit it in Word, or ask for a new document",
            lost.join(", ")
        ));
    }

    let dialect = dialect_of(base);
    let mut replacement = replacement.clone();
    replacement.id = section_id.to_string();
    if replacement.heading.trim().is_empty() {
        replacement.heading = section.heading.clone();
    }
    if replacement.level == 0 {
        replacement.level = level_of(inner);
    }
    let new_inner = section_inner_xml(&replacement, &dialect);
    let new_xml = format!("{}{}{}", &xml[..section.inner_start], new_inner, &xml[section.inner_end..]);
    xml_events::scan_spans(&new_xml).map_err(|e| format!("the rewritten section is not well-formed ({e}); nothing was written"))?;

    let mut archive = zip::ZipArchive::new(std::io::Cursor::new(base)).map_err(|e| e.to_string())?;
    let (bytes, _) = super::edit::rewrite_parts(&mut archive, &[("word/document.xml".to_string(), new_xml.clone())])?;

    // Prove what was promised, from the written bytes.
    let after = package::inspect(&bytes, &package::LIMITS);
    if !after.safe_to_open() || after.detected != DetectedFormat::Docx || after.parts != report.parts {
        return Err("the patched package did not reopen with the same parts; nothing was kept".into());
    }
    let mut parts_identical = Vec::new();
    for part in report.parts.iter().filter(|p| p.as_str() != "word/document.xml") {
        let before_part = raw_part(base, part)?;
        let after_part = raw_part(&bytes, part)?;
        if before_part != after_part {
            return Err(format!("{part} changed during the patch; nothing was kept"));
        }
        parts_identical.push(part.clone());
    }
    let written = package::read_part(&bytes, "word/document.xml")?;
    let suffix = &xml[section.inner_end..];
    if !written.starts_with(&xml[..section.inner_start]) || !written.ends_with(suffix) {
        return Err("text outside the section changed during the patch; nothing was kept".into());
    }
    let after_sections = read_sections(&written)?;
    let ids = |list: &[SectionRange]| list.iter().map(|s| s.id.clone()).collect::<Vec<_>>();
    if ids(&after_sections) != ids(&sections) {
        return Err("the document's sections changed during the patch; nothing was kept".into());
    }
    let mut sections_identical = Vec::new();
    for (old, new) in sections.iter().zip(&after_sections) {
        if old.id == section_id {
            continue;
        }
        if old.text != new.text {
            return Err(format!("section {:?} changed during the patch; nothing was kept", old.id));
        }
        sections_identical.push(old.id.clone());
    }
    let after_text = after_sections.iter().find(|s| s.id == section_id).map(|s| s.text.clone()).unwrap_or_default();
    Ok(PatchOutcome {
        bytes,
        section_id: section_id.to_string(),
        before: section.text.clone(),
        after: after_text,
        parts_identical,
        prefix_bytes: section.inner_start,
        suffix_bytes: suffix.len(),
        sections_identical,
        list_dialect: if dialect.bullet_num_id.is_some() { "the document's own bullet numbering" } else { "typed markers (the document has no bullet definition this editor knows)" },
    })
}

/// A part's bytes, uncompressed.
fn raw_part(bytes: &[u8], name: &str) -> Result<Vec<u8>, String> {
    use std::io::Read;
    let mut archive = zip::ZipArchive::new(std::io::Cursor::new(bytes)).map_err(|e| e.to_string())?;
    let mut file = archive.by_name(name).map_err(|e| format!("{name}: {e}"))?;
    let mut out = Vec::new();
    file.read_to_end(&mut out).map_err(|e| format!("{name}: {e}"))?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::artifacts::authoring::{write_docx, AuthoredDocument, Stamp};
    use crate::artifacts::doc_model::Block;

    fn document() -> AuthoredDocument {
        let section = |id: &str, heading: &str, blocks: Vec<Block>| AuthoredSection { id: id.into(), heading: heading.into(), level: 1, blocks, gap: None };
        AuthoredDocument {
            template: "authored_document@1".into(),
            template_sha256: "0".repeat(64),
            title: "Patch fixture".into(),
            audience: "a reviewer".into(),
            classification: "Internal".into(),
            fields: vec![("To".into(), "Plant Manager".into())],
            sections: vec![
                section("scope", "Scope", vec![Block::Paragraph { text: "Shell of V-101 only; nozzles are out of scope.".into() }]),
                section(
                    "readings",
                    "Readings",
                    vec![Block::Table {
                        header: vec!["Point".into(), "Thickness".into()],
                        rows: vec![vec!["A".into(), "9.6 mm".into()], vec!["C".into(), "8.2 mm".into()]],
                        caption: Some("Table 1 - UT readings".into()),
                    }],
                ),
                section("actions", "Actions", vec![Block::Bullets { items: vec!["Re-survey point C within 30 days.".into()] }]),
            ],
            stamp: Stamp { task_id: "r".into(), model: "fixture".into(), created_at: "2026-09-25T10:00:00Z".into() },
        }
    }

    fn replacement(text: &str) -> AuthoredSection {
        AuthoredSection { id: String::new(), heading: String::new(), level: 0, blocks: vec![Block::Bullets { items: vec![text.into()] }], gap: None }
    }

    #[test]
    fn one_section_changes_and_everything_else_is_byte_identical() {
        let base = write_docx(&document()).unwrap();
        let outcome = patch_section(&base, "actions", &replacement("Re-survey point C within 14 days, per the corrected reading.")).unwrap();
        assert!(outcome.after.contains("14 days") && !outcome.after.contains("30 days"));
        assert!(outcome.after.starts_with("Actions"), "the heading is kept: {}", outcome.after);
        for part in ["word/styles.xml", "word/numbering.xml", "word/header1.xml", "word/footer1.xml", "word/settings.xml", "word/_rels/document.xml.rels", "[Content_Types].xml"] {
            assert!(outcome.parts_identical.iter().any(|p| p == part), "{part}");
        }
        assert_eq!(outcome.sections_identical, vec!["scope", "readings", "provenance"]);
        let written = package::read_part(&outcome.bytes, "word/document.xml").unwrap();
        assert!(written.contains(&format!("<w:numId w:val=\"{BULLET_NUM_ID}\"/>")), "own bullet numbering is used");
        assert!(written.contains("<w:tblHeader/>") && written.contains("<w:sectPr>"));
    }

    #[test]
    fn what_the_editor_cannot_carry_is_refused_before_writing() {
        let base = write_docx(&document()).unwrap();
        let xml = package::read_part(&base, "word/document.xml").unwrap();
        let with_picture = xml.replacen("Re-survey point C within 30 days.</w:t></w:r>", "Re-survey point C within 30 days.</w:t></w:r><w:r><w:drawing/></w:r>", 1);
        let mut archive = zip::ZipArchive::new(std::io::Cursor::new(base.as_slice())).unwrap();
        let (tampered, _) = crate::artifacts::edit::rewrite_parts(&mut archive, &[("word/document.xml".into(), with_picture)]).unwrap();
        let refused = patch_section(&tampered, "actions", &replacement("x")).unwrap_err();
        assert!(refused.contains("a picture or chart"), "{refused}");
        // The unrelated section is still patchable.
        assert!(patch_section(&tampered, "scope", &replacement("Shell and nozzles of V-101.")).is_ok());
    }

    #[test]
    fn a_tracked_document_an_unknown_section_and_provenance_are_refused() {
        let base = write_docx(&document()).unwrap();
        let settings = package::read_part(&base, "word/settings.xml").unwrap().replace("<w:zoom", "<w:trackRevisions/><w:zoom");
        let mut archive = zip::ZipArchive::new(std::io::Cursor::new(base.as_slice())).unwrap();
        let (tracked, _) = crate::artifacts::edit::rewrite_parts(&mut archive, &[("word/settings.xml".into(), settings)]).unwrap();
        assert!(patch_section(&tracked, "actions", &replacement("x")).unwrap_err().contains("tracked changes"));
        let unknown = patch_section(&base, "summary", &replacement("x")).unwrap_err();
        assert!(unknown.contains("scope, readings, actions, provenance"), "{unknown}");
        assert!(patch_section(&base, "provenance", &replacement("x")).is_err());
    }

    #[test]
    fn a_legacy_approval_note_is_patchable_by_field_key() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("note.docx");
        let content: std::collections::BTreeMap<String, String> = [
            ("title", "Approval note"),
            ("recipient", "Plant Manager"),
            ("subject", "Wall loss at point C"),
            ("findings", "Point C reads 8.2 mm [E1]."),
            ("recommendation", "De-rate pending repair."),
            ("references", "[E1] UT survey"),
            ("assumptions", "None beyond the survey."),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
        let metadata = crate::artifacts::DocumentMetadata {
            task_id: "r".into(),
            created_at: "2026-09-25".into(),
            model: "fixture".into(),
            classification: "Internal".into(),
            is_draft: true,
        };
        crate::artifacts::write_document(&path, "approval_note", &content, &metadata).unwrap();
        let base = std::fs::read(&path).unwrap();
        let outcome = patch_section(&base, "recommendation", &replacement("Repair before restart.")).unwrap();
        assert!(outcome.after.contains("Repair before restart") && outcome.after.starts_with("Recommendation"));
        assert!(outcome.list_dialect.starts_with("typed markers"));
        assert!(outcome.sections_identical.contains(&"findings".to_string()));
    }
}
