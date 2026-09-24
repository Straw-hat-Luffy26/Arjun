//! The templates and composition structures a run can produce from, each with
//! an exact version.
//!
//! ## Why a template has a hash
//!
//! A document is checked against the template it was produced from: the
//! approval note's required sections, the briefing deck's four slides. When a
//! template changes, a document produced from the old one is not wrong — it is
//! *of a different template*, and checking it against the new one reports
//! missing sections that were never promised. So a registered version records
//! the template's id, version and the SHA-256 of its definition, and a later
//! check compares hashes rather than names. A definition edited without its
//! version being bumped is still caught, because the hash moves.
//!
//! The hash is of the canonical JSON of the definition below — the same bytes
//! on every machine and every run, so it is reproducible from source.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::package::DetectedFormat;

/// One template or composition structure.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TemplateInfo {
    pub id: String,
    pub version: String,
    pub format: DetectedFormat,
    /// The tool that produces from it.
    pub tool: String,
    /// `template`: fixed sections the file must hold. `structure`: a
    /// composition model the caller fills freely within its rules.
    pub kind: String,
    pub description: String,
    /// Fields or sections a produced file must carry.
    pub required: Vec<String>,
    pub optional: Vec<String>,
    /// SHA-256 of the canonical definition (every field above, as JSON).
    pub sha256: String,
}

fn definition(
    id: &str,
    version: &str,
    format: DetectedFormat,
    tool: &str,
    kind: &str,
    description: &str,
    required: Vec<String>,
    optional: Vec<String>,
) -> TemplateInfo {
    let canonical = serde_json::json!({
        "id": id, "version": version, "format": format, "tool": tool, "kind": kind,
        "description": description, "required": required, "optional": optional,
    });
    let sha256 = format!("{:x}", Sha256::digest(canonical.to_string().as_bytes()));
    TemplateInfo {
        id: id.to_string(),
        version: version.to_string(),
        format,
        tool: tool.to_string(),
        kind: kind.to_string(),
        description: description.to_string(),
        required,
        optional,
        sha256,
    }
}

/// Everything this build can produce from.
pub fn catalogue() -> Vec<TemplateInfo> {
    let note = super::docx::APPROVAL_NOTE;
    vec![
        definition(
            "approval_note",
            "1",
            DetectedFormat::Docx,
            "artifact.create_approval_note",
            "template",
            "An approval note: title, recipient, subject, findings, recommendation, supporting \
             references and assumptions, stamped DRAFT until a person approves it.",
            note.iter().filter(|f| f.required).map(|f| f.key.to_string()).collect(),
            note.iter().filter(|f| !f.required).map(|f| f.key.to_string()).collect(),
        ),
        definition(
            "document_model",
            "1",
            DetectedFormat::Docx,
            "artifact.create_approval_note",
            "structure",
            "A composed Word document: a title and sections, each with a heading, a level and \
             blocks of kind paragraph, bullets, numbered, table or pageBreak.",
            vec!["title".into(), "sections".into()],
            vec!["classification".into()],
        ),
        definition(
            "briefing_deck",
            "1",
            DetectedFormat::Pptx,
            "artifact.create_briefing_deck",
            "template",
            "A briefing deck: a title slide, then one slide per section in this order, each with \
             at least one bullet.",
            std::iter::once("title".to_string())
                .chain(super::pptx::BRIEFING_SECTIONS.iter().map(|s| s.to_ascii_lowercase()))
                .collect(),
            Vec::new(),
        ),
        definition(
            "calculation_workbook",
            "1",
            DetectedFormat::Xlsx,
            "artifact.create_calculation_workbook",
            "template",
            "The calculations this task ran, with the working shown and live formulas Excel \
             recomputes. Written from the calculation engine's records, not from text.",
            vec!["calculations run in this task".into()],
            Vec::new(),
        ),
        definition(
            "workbook_model",
            "1",
            DetectedFormat::Xlsx,
            "artifact.create_calculation_workbook",
            "structure",
            "A composed workbook: sheets, each with typed columns (text, number, currency, \
             percent, date, formula) and rows.",
            vec!["sheets".into()],
            vec!["title".into(), "classification".into()],
        ),
    ]
}

/// One template by id.
pub fn find(id: &str) -> Option<TemplateInfo> {
    catalogue().into_iter().find(|t| t.id == id)
}

/// The template a produce call used, from the tool and its arguments.
pub fn used_by(tool: crate::orchestrator::tools::ToolName, arguments: &serde_json::Value) -> Option<TemplateInfo> {
    use crate::orchestrator::tools::ToolName;
    let id = match tool {
        ToolName::CreateDocx if arguments.get("sections").is_some() => "document_model",
        ToolName::CreateDocx => arguments.get("template").and_then(|t| t.as_str()).unwrap_or("approval_note"),
        ToolName::CreatePptx => "briefing_deck",
        ToolName::CreateXlsx if arguments.get("sheets").is_some() => "workbook_model",
        ToolName::CreateXlsx => "calculation_workbook",
        _ => return None,
    };
    find(id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_template_has_a_stable_hash_of_its_definition() {
        let once = catalogue();
        let twice = catalogue();
        assert_eq!(once, twice, "the same build must hash the same definition the same way");
        for template in &once {
            assert_eq!(template.sha256.len(), 64);
        }
        let mut changed = find("approval_note").unwrap();
        let rehashed = definition(
            &changed.id, &changed.version, changed.format, &changed.tool, &changed.kind,
            &changed.description, vec!["title".into()], changed.optional.clone(),
        );
        assert_ne!(rehashed.sha256, changed.sha256, "dropping a required field moves the hash");
        changed.required.clear();
    }

    #[test]
    fn the_approval_note_template_requires_what_the_writer_requires() {
        let note = find("approval_note").unwrap();
        for key in ["title", "recipient", "subject", "findings", "recommendation", "references", "assumptions"] {
            assert!(note.required.iter().any(|r| r == key), "{key} missing from {:?}", note.required);
        }
        assert_eq!(note.optional, vec!["calculation".to_string()]);
    }

    #[test]
    fn a_produce_call_names_the_template_it_used() {
        use crate::orchestrator::tools::ToolName;
        let sections = serde_json::json!({"sections": []});
        assert_eq!(used_by(ToolName::CreateDocx, &sections).unwrap().id, "document_model");
        assert_eq!(used_by(ToolName::CreateDocx, &serde_json::json!({})).unwrap().id, "approval_note");
        assert_eq!(used_by(ToolName::CreatePptx, &serde_json::json!({})).unwrap().id, "briefing_deck");
        assert!(used_by(ToolName::CreatePdf, &serde_json::json!({})).is_none());
    }
}
