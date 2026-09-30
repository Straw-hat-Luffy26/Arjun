//! Authoring a Word document from a structured specification (P09).
//!
//! The model supplies content; this module supplies everything else — the
//! package, styles, numbering, header, footer, page settings, the section
//! markers a later patch finds its way by, and the checks a document has to
//! pass before a byte is written:
//!
//! - every mandatory field and section is there, or declared a **gap** — a
//!   stated need for information nobody supplied, rendered visibly and
//!   reported as a clarification request, never filled in;
//! - every unit of a section that makes claims cites something the run can
//!   resolve, and a calculation section cites calculation records;
//! - every figure with a unit is stated by what its sentence cites;
//! - nothing cites an item that has been corrected since.
//!
//! ## Stable section ids
//!
//! Each section is wrapped in a body-level bookmark named `_sec_<id>`. Word
//! treats a leading underscore as hidden, so the marker does not clutter the
//! person's bookmark list, and it survives an ordinary save. It is how
//! `document.patch_section` finds exactly one section of an existing version
//! and nothing else.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::content::{citations_in, CitationTarget};
use super::doc_model::{placeholder_in, Block};
use super::ooxml::escape;
use super::xml_events::{self, Event};

// ── Templates ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum CitationRule {
    /// Every paragraph, list item and table row cites something.
    Required,
    /// Every unit cites a calculation record (`[C:calc-…]`).
    Calculation,
    Optional,
}

#[derive(Debug, Clone, Copy)]
pub struct FieldReq {
    pub key: &'static str,
    pub label: &'static str,
    pub required: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct SectionReq {
    pub id: &'static str,
    pub heading: &'static str,
    pub required: bool,
    pub citations: CitationRule,
    /// Written by the composer from the document's own citations; a spec may
    /// not supply it.
    pub generated: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct AuthoringTemplate {
    pub id: &'static str,
    pub version: &'static str,
    pub description: &'static str,
    pub fields: &'static [FieldReq],
    /// Fixed sections, in order. Empty for a free structure.
    pub sections: &'static [SectionReq],
    /// Whether sections beyond `sections` are accepted (any id, optional
    /// citations).
    pub free_sections: bool,
}

/// The approval note the problem statement's own example asks for, with
/// stable section ids and the rules each section is held to.
pub const INSPECTION_APPROVAL_NOTE: AuthoringTemplate = AuthoringTemplate {
    id: "inspection_approval_note",
    version: "1",
    description: "An approval note drawn from inspection evidence: subject, cited findings, the \
                  calculation records it relies on, a recommendation, generated supporting \
                  references and stated assumptions. Stamped DRAFT until a person approves it.",
    fields: &[
        FieldReq { key: "recipient", label: "To", required: true },
        FieldReq { key: "from", label: "From", required: false },
        FieldReq { key: "reference", label: "Reference", required: false },
    ],
    sections: &[
        SectionReq { id: "subject", heading: "Subject", required: true, citations: CitationRule::Optional, generated: false },
        SectionReq { id: "findings", heading: "Findings", required: true, citations: CitationRule::Required, generated: false },
        SectionReq { id: "calculation", heading: "Calculation", required: false, citations: CitationRule::Calculation, generated: false },
        SectionReq { id: "recommendation", heading: "Recommendation", required: true, citations: CitationRule::Optional, generated: false },
        SectionReq { id: "assumptions", heading: "Assumptions", required: true, citations: CitationRule::Optional, generated: false },
        SectionReq { id: "references", heading: "Supporting references", required: true, citations: CitationRule::Optional, generated: true },
    ],
    free_sections: false,
};

/// A document composed freely, section by section, with the same checks.
pub const AUTHORED_DOCUMENT: AuthoringTemplate = AuthoringTemplate {
    id: "authored_document",
    version: "1",
    description: "A composed Word document: any sections with stable ids, each of paragraphs, \
                  lists, tables and page breaks. Figures must be cited; generated references.",
    fields: &[
        FieldReq { key: "recipient", label: "To", required: false },
        FieldReq { key: "reference", label: "Reference", required: false },
    ],
    sections: &[SectionReq {
        id: "references",
        heading: "Supporting references",
        required: false,
        citations: CitationRule::Optional,
        generated: true,
    }],
    free_sections: true,
};

pub const AUTHORING_TEMPLATES: &[AuthoringTemplate] = &[INSPECTION_APPROVAL_NOTE, AUTHORED_DOCUMENT];

/// The legacy `approval_note@1` (written by `artifact.create_approval_note`),
/// described in the same terms so its versions can be checked and patched by
/// field key. Not offered to `document.compose` as an authoring template: a
/// compatible request goes through the legacy writer itself.
pub const LEGACY_APPROVAL_NOTE: AuthoringTemplate = AuthoringTemplate {
    id: "approval_note",
    version: "1",
    description: "The approval note artifact.create_approval_note writes, section ids by field key.",
    fields: &[],
    sections: &[
        SectionReq { id: "title", heading: "", required: false, citations: CitationRule::Optional, generated: false },
        SectionReq { id: "recipient", heading: "To", required: true, citations: CitationRule::Optional, generated: false },
        SectionReq { id: "subject", heading: "Subject", required: true, citations: CitationRule::Optional, generated: false },
        SectionReq { id: "findings", heading: "Findings", required: true, citations: CitationRule::Optional, generated: false },
        SectionReq { id: "calculation", heading: "Calculation", required: false, citations: CitationRule::Optional, generated: false },
        SectionReq { id: "recommendation", heading: "Recommendation", required: true, citations: CitationRule::Optional, generated: false },
        SectionReq { id: "references", heading: "Supporting references", required: true, citations: CitationRule::Optional, generated: false },
        SectionReq { id: "assumptions", heading: "Assumptions", required: true, citations: CitationRule::Optional, generated: false },
    ],
    free_sections: false,
};

/// The rules a stored version is checked and patched under, by its recorded
/// template id.
pub fn checking_template(id: &str) -> Option<&'static AuthoringTemplate> {
    AUTHORING_TEMPLATES.iter().find(|t| t.id == id).or((id == LEGACY_APPROVAL_NOTE.id).then_some(&LEGACY_APPROVAL_NOTE))
}

impl AuthoringTemplate {
    pub fn key(&self) -> String {
        format!("{}@{}", self.id, self.version)
    }

    fn section(&self, id: &str) -> Option<&SectionReq> {
        self.sections.iter().find(|s| s.id == id)
    }

    /// The definition as canonical JSON, and its SHA-256: what a version
    /// records so a later check knows which rules it was written under.
    pub fn canonical(&self) -> serde_json::Value {
        serde_json::json!({
            "id": self.id,
            "version": self.version,
            "description": self.description,
            "fields": self.fields.iter().map(|f| serde_json::json!({"key": f.key, "label": f.label, "required": f.required})).collect::<Vec<_>>(),
            "sections": self.sections.iter().map(|s| serde_json::json!({"id": s.id, "heading": s.heading, "required": s.required, "citations": s.citations, "generated": s.generated})).collect::<Vec<_>>(),
            "freeSections": self.free_sections,
        })
    }

    /// The definition's hash as the template catalogue records it, which
    /// covers the fields, the sections and each section's citation rule.
    pub fn sha256(&self) -> String {
        super::templates::find(self.id)
            .map(|t| t.sha256)
            .unwrap_or_else(|| format!("{:x}", Sha256::digest(self.canonical().to_string().as_bytes())))
    }
}

/// `id@version`, or `id` for the current version.
pub fn authoring_template(named: &str) -> Result<&'static AuthoringTemplate, String> {
    let (id, version) = match named.trim().split_once('@') {
        Some((id, version)) => (id, Some(version)),
        None => (named.trim(), None),
    };
    let found = AUTHORING_TEMPLATES.iter().find(|t| t.id == id).ok_or_else(|| {
        format!(
            "there is no authoring template {id:?}; document.template_list shows {}",
            AUTHORING_TEMPLATES.iter().map(|t| t.key()).collect::<Vec<_>>().join(", ")
        )
    })?;
    match version {
        Some(v) if v != found.version => Err(format!("{id} has no version {v}; the current version is {}", found.key())),
        _ => Ok(found),
    }
}

// ── The specification ────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SpecSection {
    pub id: String,
    #[serde(default)]
    pub heading: Option<String>,
    #[serde(default)]
    pub level: Option<u8>,
    #[serde(default)]
    pub blocks: Vec<Block>,
    /// What information is missing, stated instead of content.
    #[serde(default)]
    pub gap: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DocumentSpec {
    /// `inspection_approval_note@1`, `authored_document@1`.
    pub template: String,
    pub title: String,
    /// Who it is written for.
    pub audience: String,
    /// The file name it is kept under (`v101-approval-note.docx`).
    pub output: String,
    /// Mandatory and optional header fields. `?` states that a field's value
    /// is not known: a gap, not a guess.
    #[serde(default)]
    pub fields: BTreeMap<String, String>,
    pub sections: Vec<SpecSection>,
}

/// What a citation marker resolves to, as the run can see it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ResolvedCitation {
    pub marker: String,
    /// How it reads in the references list.
    pub label: String,
    /// The text a figure in the citing sentence is held to.
    #[serde(skip)]
    pub text: String,
    pub calculation: bool,
    /// Why it no longer stands (corrected, withdrawn, stale), if it does not.
    pub stale_because: Option<String>,
}

/// Resolves markers against what the run actually holds.
pub trait CitationResolver {
    fn resolve(&self, target: &CitationTarget, marker: &str) -> Option<ResolvedCitation>;
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Problem {
    pub code: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub section: Option<String>,
    pub detail: String,
}

impl Problem {
    fn at(code: &'static str, section: Option<&str>, detail: impl Into<String>) -> Self {
        Problem { code, section: section.map(str::to_string), detail: detail.into() }
    }

    pub fn describe(&self) -> String {
        match &self.section {
            Some(section) => format!("{} [{section}]: {}", self.code, self.detail),
            None => format!("{}: {}", self.code, self.detail),
        }
    }
}

/// A stated need for information, carried into the document and the result.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Gap {
    /// A section id, or `field:<key>`.
    pub at: String,
    pub heading: String,
    pub need: String,
}

/// Which markers each section rests on.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SectionLineage {
    pub section_id: String,
    pub heading: String,
    pub markers: Vec<String>,
    #[serde(default)]
    pub calculations: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gap: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AuthoredSection {
    pub id: String,
    pub heading: String,
    pub level: u8,
    pub blocks: Vec<Block>,
    #[serde(default)]
    pub gap: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Stamp {
    pub task_id: String,
    pub model: String,
    pub created_at: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AuthoredDocument {
    pub template: String,
    pub template_sha256: String,
    pub title: String,
    pub audience: String,
    pub classification: String,
    /// Label → value, in template order.
    pub fields: Vec<(String, String)>,
    pub sections: Vec<AuthoredSection>,
    pub stamp: Stamp,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Composition {
    pub document: AuthoredDocument,
    pub gaps: Vec<Gap>,
    pub citations: Vec<ResolvedCitation>,
    pub lineage: Vec<SectionLineage>,
}

pub const MAX_SECTIONS: usize = 30;
pub const MAX_BLOCKS: usize = 300;
pub const MAX_TABLE_ROWS: usize = 500;
pub const MAX_TABLE_COLUMNS: usize = 20;
pub const MAX_TEXT_CHARS: usize = 400_000;

fn valid_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 32
        && id.chars().next().is_some_and(|c| c.is_ascii_lowercase())
        && id.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
}

fn valid_output(name: &str) -> bool {
    name.len() <= 90
        && name.to_ascii_lowercase().ends_with(".docx")
        && name.chars().next().is_some_and(|c| c.is_ascii_alphanumeric())
        && name.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-' | ' '))
        && !name.contains("..")
}

/// The units a citation rule applies to: each paragraph, each list item,
/// each table row (header excluded). A caption rides with its table.
fn claim_units(block: &Block) -> Vec<String> {
    match block {
        Block::Paragraph { text } => vec![text.clone()],
        Block::Bullets { items } | Block::Numbered { items } => items.clone(),
        Block::Table { rows, .. } => rows.iter().map(|r| r.join(" | ")).collect(),
        Block::PageBreak => Vec::new(),
    }
}

/// Figures in a unit: numbers with a unit or a percent sign. Plain numbers
/// (counts, years, clause numbers) and dates are not held to a source, and
/// nor is anything inside a citation marker (`[M:mi-82a…@1]` is a name, not
/// 82 years).
pub fn figures_in(text: &str) -> Vec<(String, String)> {
    crate::calculation::check::quantities_in(&without_markers(text))
        .into_iter()
        .filter(|(_, unit)| !unit.is_empty())
        .collect()
}

/// `text` with every citation marker blanked out.
pub fn without_markers(text: &str) -> String {
    let mut out = text.to_string();
    for citation in citations_in(text) {
        out = out.replace(&citation.marker, " ");
    }
    out
}

/// Checks `spec` against its template and the run's evidence, and composes
/// it — or says everything that is wrong, with nothing written.
pub fn compose(
    spec: &DocumentSpec,
    classification: &str,
    stamp: &Stamp,
    resolver: &dyn CitationResolver,
) -> Result<Composition, Vec<Problem>> {
    let mut problems: Vec<Problem> = Vec::new();
    let template = match authoring_template(&spec.template) {
        Ok(t) => t,
        Err(why) => return Err(vec![Problem::at("template", None, why)]),
    };
    if spec.title.trim().is_empty() || spec.title.chars().count() > 200 {
        problems.push(Problem::at("title", None, "a document needs a title of at most 200 characters"));
    }
    if spec.audience.trim().is_empty() {
        problems.push(Problem::at("audience", None, "say who the document is written for (audience)"));
    }
    if !valid_output(&spec.output) {
        problems.push(Problem::at(
            "output",
            None,
            format!("{:?} is not a usable file name: letters, digits, '.', '_', '-' and spaces, ending .docx, no folders", spec.output),
        ));
    }
    if spec.sections.len() > MAX_SECTIONS {
        return Err(vec![Problem::at("bounds", None, format!("at most {MAX_SECTIONS} sections"))]);
    }
    let blocks: usize = spec.sections.iter().map(|s| s.blocks.len()).sum();
    let characters: usize = spec.sections.iter().flat_map(|s| s.blocks.iter()).map(|b| b.text().chars().count()).sum();
    if blocks > MAX_BLOCKS || characters > MAX_TEXT_CHARS {
        return Err(vec![Problem::at("bounds", None, format!("at most {MAX_BLOCKS} blocks and {MAX_TEXT_CHARS} characters"))]);
    }

    let mut gaps: Vec<Gap> = Vec::new();

    // Fields.
    let mut fields: Vec<(String, String)> = Vec::new();
    for key in spec.fields.keys() {
        if !template.fields.iter().any(|f| f.key == key) {
            problems.push(Problem::at(
                "field_unknown",
                None,
                format!("{key:?} is not a field of {}; its fields are {}", template.key(), template.fields.iter().map(|f| f.key).collect::<Vec<_>>().join(", ")),
            ));
        }
    }
    for field in template.fields {
        match spec.fields.get(field.key).map(|v| v.trim()) {
            Some("?") => gaps.push(Gap { at: format!("field:{}", field.key), heading: field.label.into(), need: format!("the {} is not known", field.label.to_lowercase()) }),
            Some(value) if !value.is_empty() => {
                if let Some(marker) = placeholder_in(value) {
                    problems.push(Problem::at("placeholder", None, format!("the {} field holds the placeholder {marker:?}", field.key)));
                }
                fields.push((field.label.to_string(), value.to_string()));
            }
            _ if field.required => problems.push(Problem::at(
                "field_missing",
                None,
                format!("the mandatory field {:?} ({}) was not supplied; give it, or write \"?\" to report it as unknown", field.key, field.label),
            )),
            _ => {}
        }
    }

    // Sections: ids, order, presence.
    let mut seen: Vec<&str> = Vec::new();
    let mut last_fixed: Option<usize> = None;
    for section in &spec.sections {
        if !valid_id(&section.id) {
            problems.push(Problem::at("section_id", Some(&section.id), "a section id is 1-32 lowercase letters, digits or _, starting with a letter"));
            continue;
        }
        if seen.contains(&section.id.as_str()) {
            problems.push(Problem::at("section_duplicate", Some(&section.id), "two sections share this id"));
        }
        seen.push(&section.id);
        match template.section(&section.id) {
            Some(req) if req.generated => problems.push(Problem::at(
                "section_generated",
                Some(&section.id),
                "this section is written from the document's citations; do not supply it",
            )),
            Some(_) => {
                let position = template.sections.iter().position(|s| s.id == section.id);
                if let (Some(position), Some(last)) = (position, last_fixed) {
                    if position < last {
                        problems.push(Problem::at("section_order", Some(&section.id), format!("{} puts its sections in the order {}", template.key(), template.sections.iter().map(|s| s.id).collect::<Vec<_>>().join(", "))));
                    }
                }
                last_fixed = position.or(last_fixed);
            }
            None if !template.free_sections => problems.push(Problem::at(
                "section_unknown",
                Some(&section.id),
                format!("{} has sections {}", template.key(), template.sections.iter().filter(|s| !s.generated).map(|s| s.id).collect::<Vec<_>>().join(", ")),
            )),
            None => {}
        }
    }
    for req in template.sections.iter().filter(|s| s.required && !s.generated) {
        if !spec.sections.iter().any(|s| s.id == req.id) {
            problems.push(Problem::at(
                "section_missing",
                Some(req.id),
                format!("the mandatory section {:?} ({}) is missing; write it, or include it with a gap saying what is needed", req.id, req.heading),
            ));
        }
    }

    // Structure, through the document model's own checks.
    let model = super::doc_model::Document {
        title: spec.title.clone(),
        classification: classification.to_string(),
        sections: spec
            .sections
            .iter()
            .filter(|s| s.gap.is_none())
            .map(|s| super::doc_model::Section {
                heading: s.heading.clone().or_else(|| template.section(&s.id).map(|r| r.heading.to_string())).unwrap_or_else(|| s.id.clone()),
                level: s.level.unwrap_or(1),
                blocks: s.blocks.clone(),
            })
            .collect(),
        properties: Default::default(),
    };
    if !model.sections.is_empty() {
        // Placeholders are reported per unit below, with where they are.
        for problem in model.problems().into_iter().filter(|p| !p.contains("the placeholder")) {
            problems.push(Problem::at("structure", None, problem));
        }
    }
    // Citations, figures, currency and content, section by section.
    let mut resolved: BTreeMap<String, ResolvedCitation> = BTreeMap::new();
    let mut lineage: Vec<SectionLineage> = Vec::new();
    for section in &spec.sections {
        let (found, line) = check_section(template, section, resolver, &mut resolved);
        problems.extend(found);
        if let Some(gap) = &section.gap {
            gaps.push(Gap { at: section.id.clone(), heading: line.heading.clone(), need: gap.trim().to_string() });
        }
        lineage.push(line);
    }

    if !problems.is_empty() {
        return Err(problems);
    }

    // The order the template fixes; free sections where they were written.
    let mut sections: Vec<AuthoredSection> = Vec::new();
    for section in &spec.sections {
        let req = template.section(&section.id);
        sections.push(AuthoredSection {
            id: section.id.clone(),
            heading: section.heading.clone().or_else(|| req.map(|r| r.heading.to_string())).unwrap_or_else(|| section.id.clone()),
            level: section.level.unwrap_or(1).clamp(1, 3),
            blocks: section.blocks.clone(),
            gap: section.gap.clone().map(|g| g.trim().to_string()),
        });
    }
    // Generated references, from what was cited.
    if let Some(req) = template.sections.iter().find(|s| s.generated) {
        let items: Vec<String> = resolved.values().map(|c| format!("{} {}", c.marker, c.label)).collect();
        if !items.is_empty() || req.required {
            sections.push(AuthoredSection {
                id: req.id.to_string(),
                heading: req.heading.to_string(),
                level: 1,
                blocks: if items.is_empty() {
                    vec![Block::Paragraph { text: "This document cites no source.".into() }]
                } else {
                    vec![Block::Bullets { items }]
                },
                gap: None,
            });
            lineage.push(SectionLineage {
                section_id: req.id.to_string(),
                heading: req.heading.to_string(),
                markers: resolved.keys().cloned().collect(),
                calculations: resolved.values().filter(|c| c.calculation).map(|c| c.marker.clone()).collect(),
                gap: None,
            });
        }
    }

    Ok(Composition {
        document: AuthoredDocument {
            template: template.key(),
            template_sha256: template.sha256(),
            title: spec.title.trim().to_string(),
            audience: spec.audience.trim().to_string(),
            classification: classification.to_string(),
            fields,
            sections,
            stamp: stamp.clone(),
        },
        gaps,
        citations: resolved.into_values().collect(),
        lineage,
    })
}

/// One section's content held to its template's rules: written or a gap,
/// every claim cited where the section requires it, every marker resolving
/// to something current, every figure stated by what its sentence cites.
/// Used for a whole composition and for a patch of one section alike.
pub fn check_section(
    template: &AuthoringTemplate,
    section: &SpecSection,
    resolver: &dyn CitationResolver,
    resolved: &mut BTreeMap<String, ResolvedCitation>,
) -> (Vec<Problem>, SectionLineage) {
    let mut problems: Vec<Problem> = Vec::new();
    let has_content = section.blocks.iter().any(|b| b.characters() > 0);
    match (&section.gap, has_content) {
        (Some(_), true) => problems.push(Problem::at("gap_and_content", Some(&section.id), "a section is either written or declared a gap, not both")),
        (Some(gap), false) if gap.trim().is_empty() => problems.push(Problem::at("gap", Some(&section.id), "a gap says what information is needed")),
        (None, false) => problems.push(Problem::at(
            "section_empty",
            Some(&section.id),
            "the section has no content; write it from the evidence, or declare a gap saying what is missing",
        )),
        _ => {}
    }
    for block in &section.blocks {
        if let Block::Table { header, rows, .. } = block {
            if rows.len() > MAX_TABLE_ROWS || header.len() > MAX_TABLE_COLUMNS {
                problems.push(Problem::at("bounds", Some(&section.id), format!("a table is at most {MAX_TABLE_ROWS} rows by {MAX_TABLE_COLUMNS} columns")));
            }
            if rows.iter().any(|r| r.len() > header.len()) {
                problems.push(Problem::at("structure", Some(&section.id), "a table row has more cells than its header"));
            }
        }
    }
    let rule = template.section(&section.id).map(|r| r.citations).unwrap_or(CitationRule::Optional);
    let mut markers: Vec<String> = Vec::new();
    let mut calculations: Vec<String> = Vec::new();
    for (index, block) in section.blocks.iter().enumerate() {
        let caption_citations = match block {
            Block::Table { caption: Some(caption), .. } => citations_in(caption),
            _ => Vec::new(),
        };
        for (unit_index, unit) in claim_units(block).iter().enumerate() {
            let mut cited = citations_in(unit);
            cited.extend(caption_citations.iter().cloned());
            let place = format!("block {} unit {}", index + 1, unit_index + 1);
            let excerpt: String = unit.chars().take(90).collect();
            if cited.is_empty() && rule == CitationRule::Required {
                problems.push(Problem::at(
                    "unsupported_claim",
                    Some(&section.id),
                    format!("{place} cites nothing: {excerpt:?}. Cite the passage, memory item or record it rests on, or remove it"),
                ));
            }
            if rule == CitationRule::Calculation && !cited.iter().any(|c| matches!(c.target, CitationTarget::Calculation { .. })) {
                problems.push(Problem::at(
                    "calculation_citation",
                    Some(&section.id),
                    format!("{place} states a calculation without citing its record ([C:calc-…]): {excerpt:?}"),
                ));
            }
            let mut texts: Vec<String> = Vec::new();
            for citation in &cited {
                let found = resolved
                    .get(&citation.marker)
                    .cloned()
                    .or_else(|| resolver.resolve(&citation.target, &citation.marker));
                match found {
                    None => problems.push(Problem::at(
                        "unbound_citation",
                        Some(&section.id),
                        format!("{} does not resolve to anything this run can read", citation.marker),
                    )),
                    Some(found) => {
                        if let Some(why) = &found.stale_because {
                            problems.push(Problem::at(
                                "stale_citation",
                                Some(&section.id),
                                format!("{} no longer stands: {why}. Cite what replaced it", citation.marker),
                            ));
                        }
                        if found.calculation && !calculations.contains(&citation.marker) {
                            calculations.push(citation.marker.clone());
                        }
                        texts.push(found.text.clone());
                        resolved.insert(citation.marker.clone(), found);
                    }
                }
                if !markers.contains(&citation.marker) {
                    markers.push(citation.marker.clone());
                }
            }
            for (number, unit_text) in figures_in(unit) {
                let supported = texts.iter().any(|t| crate::calculation::check::text_states(t, &number, &unit_text));
                if !supported {
                    problems.push(Problem::at(
                        "unsupported_figure",
                        Some(&section.id),
                        format!(
                            "{place} states {number} {unit_text}, which {} — a figure has to come from what its sentence cites",
                            if cited.is_empty() { "cites nothing" } else { "none of its citations states" }
                        ),
                    ));
                }
            }
            if let Some(marker) = placeholder_in(unit) {
                problems.push(Problem::at("placeholder", Some(&section.id), format!("{place} holds the placeholder {marker:?}")));
            }
        }
    }
    let heading = section.heading.clone().or_else(|| template.section(&section.id).map(|r| r.heading.to_string())).unwrap_or_else(|| section.id.clone());
    (problems, SectionLineage { section_id: section.id.clone(), heading, markers, calculations, gap: section.gap.clone() })
}

// ── Writing the package ──────────────────────────────────────────────────

/// How lists are written: with the document's own bullet numbering when it
/// has the definition this writer makes, else with typed markers — which is
/// what a patch into a document from elsewhere falls back to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dialect {
    pub bullet_num_id: Option<u32>,
    pub styled: bool,
}

pub const BULLET_NUM_ID: u32 = 9001;
pub const BOOKMARK_PREFIX: &str = "_sec_";

fn run(text: &str) -> String {
    format!("<w:r><w:t xml:space=\"preserve\">{}</w:t></w:r>", escape(text))
}

fn styled_paragraph(style: &str, text: &str) -> String {
    format!("<w:p><w:pPr><w:pStyle w:val=\"{style}\"/></w:pPr>{}</w:p>", run(text))
}

fn plain_paragraph(text: &str) -> String {
    format!("<w:p>{}</w:p>", run(text))
}

fn heading_xml(text: &str, level: u8) -> String {
    let level = level.clamp(1, 6);
    format!(
        "<w:p><w:pPr><w:pStyle w:val=\"Heading{level}\"/><w:outlineLvl w:val=\"{}\"/></w:pPr>{}</w:p>",
        level - 1,
        run(text)
    )
}

fn table_xml(header: &[String], rows: &[Vec<String>], caption: Option<&str>, dialect: &Dialect) -> String {
    let mut out = String::new();
    if let Some(caption) = caption.filter(|c| !c.trim().is_empty()) {
        out.push_str(&if dialect.styled { styled_paragraph("Caption", caption) } else { plain_paragraph(caption) });
    }
    out.push_str(
        "<w:tbl><w:tblPr><w:tblStyle w:val=\"TableGrid\"/><w:tblW w:w=\"5000\" w:type=\"pct\"/>\
         <w:tblBorders><w:top w:val=\"single\" w:sz=\"4\" w:color=\"auto\"/><w:left w:val=\"single\" w:sz=\"4\" w:color=\"auto\"/>\
         <w:bottom w:val=\"single\" w:sz=\"4\" w:color=\"auto\"/><w:right w:val=\"single\" w:sz=\"4\" w:color=\"auto\"/>\
         <w:insideH w:val=\"single\" w:sz=\"4\" w:color=\"auto\"/><w:insideV w:val=\"single\" w:sz=\"4\" w:color=\"auto\"/>\
         </w:tblBorders></w:tblPr>",
    );
    // The header row repeats on every page a long table runs onto, and no
    // row splits across a page.
    out.push_str("<w:tr><w:trPr><w:cantSplit/><w:tblHeader/></w:trPr>");
    for cell in header {
        out.push_str(&format!("<w:tc><w:p><w:r><w:rPr><w:b/></w:rPr><w:t xml:space=\"preserve\">{}</w:t></w:r></w:p></w:tc>", escape(cell)));
    }
    out.push_str("</w:tr>");
    for row in rows {
        out.push_str("<w:tr><w:trPr><w:cantSplit/></w:trPr>");
        for index in 0..header.len().max(row.len()) {
            let cell = row.get(index).map(String::as_str).unwrap_or("");
            out.push_str(&format!("<w:tc><w:p>{}</w:p></w:tc>", if cell.is_empty() { String::new() } else { run(cell) }));
        }
        out.push_str("</w:tr>");
    }
    out.push_str("</w:tbl><w:p/>");
    out
}

/// A section's content — heading, blocks, gap — without its markers.
pub fn section_inner_xml(section: &AuthoredSection, dialect: &Dialect) -> String {
    let mut out = heading_xml(&section.heading, section.level);
    if let Some(gap) = &section.gap {
        let text = format!("INFORMATION NEEDED: {gap}. This section is incomplete; nothing has been written in its place.");
        out.push_str(&if dialect.styled { styled_paragraph("GapNote", &text) } else { format!("<w:p><w:r><w:rPr><w:b/><w:color w:val=\"C00000\"/></w:rPr><w:t xml:space=\"preserve\">{}</w:t></w:r></w:p>", escape(&text)) });
        return out;
    }
    for block in &section.blocks {
        match block {
            Block::Paragraph { text } => {
                for line in text.lines().filter(|l| !l.trim().is_empty()) {
                    out.push_str(&plain_paragraph(line));
                }
            }
            Block::Bullets { items } => {
                for item in items {
                    out.push_str(&match dialect.bullet_num_id {
                        Some(num) => format!(
                            "<w:p><w:pPr><w:pStyle w:val=\"ListParagraph\"/><w:numPr><w:ilvl w:val=\"0\"/><w:numId w:val=\"{num}\"/></w:numPr></w:pPr>{}</w:p>",
                            run(item)
                        ),
                        None => format!("<w:p><w:pPr><w:ind w:left=\"720\" w:hanging=\"360\"/></w:pPr>{}</w:p>", run(&format!("\u{2022} {item}"))),
                    });
                }
            }
            Block::Numbered { items } => {
                // Typed markers: a list restarts at 1 in every section without
                // a numbering definition per list.
                for (index, item) in items.iter().enumerate() {
                    out.push_str(&format!("<w:p><w:pPr><w:ind w:left=\"720\" w:hanging=\"360\"/></w:pPr>{}</w:p>", run(&format!("{}. {item}", index + 1))));
                }
            }
            Block::Table { header, rows, caption } => out.push_str(&table_xml(header, rows, caption.as_deref(), dialect)),
            Block::PageBreak => out.push_str("<w:p><w:r><w:br w:type=\"page\"/></w:r></w:p>"),
        }
    }
    out
}

fn bookmark_start(id: u32, section_id: &str) -> String {
    format!("<w:bookmarkStart w:id=\"{id}\" w:name=\"{BOOKMARK_PREFIX}{section_id}\"/>")
}

fn bookmark_end(id: u32) -> String {
    format!("<w:bookmarkEnd w:id=\"{id}\"/>")
}

fn document_xml(doc: &AuthoredDocument) -> String {
    let dialect = Dialect { bullet_num_id: Some(BULLET_NUM_ID), styled: true };
    let mut body = String::new();
    body.push_str(&styled_paragraph(
        "DraftBanner",
        "DRAFT - not approved. Do not act on this document until a person has reviewed and approved it.",
    ));
    body.push_str(&styled_paragraph("Title", &doc.title));
    body.push_str(&plain_paragraph(&format!("Classification: {}", doc.classification)));
    let mut rows: Vec<(String, String)> = doc.fields.clone();
    rows.push(("Audience".into(), doc.audience.clone()));
    rows.push(("Date".into(), doc.stamp.created_at.chars().take(10).collect()));
    for (label, value) in &rows {
        body.push_str(&format!(
            "<w:p><w:r><w:rPr><w:b/></w:rPr><w:t xml:space=\"preserve\">{}: </w:t></w:r>{}</w:p>",
            escape(label),
            run(value)
        ));
    }
    for (index, section) in doc.sections.iter().enumerate() {
        let id = index as u32 + 1;
        body.push_str(&bookmark_start(id, &section.id));
        body.push_str(&section_inner_xml(section, &dialect));
        body.push_str(&bookmark_end(id));
    }
    let stamp = super::visible_watermark::stamp_from_metadata(&doc.stamp.task_id, &doc.stamp.model, &doc.stamp.created_at, &doc.classification, true);
    let provenance = AuthoredSection {
        id: "provenance".into(),
        heading: "How this was produced".into(),
        level: 1,
        blocks: vec![Block::Paragraph {
            text: format!(
                "Task {} · template {} (definition sha-256 {}) · content by {} · figures from ARJUN's calculation engine, cited by record · {}\n{}",
                doc.stamp.task_id,
                doc.template,
                &doc.template_sha256[..12],
                doc.stamp.model,
                stamp.header_line(),
                stamp.footer_sentence()
            ),
        }],
        gap: None,
    };
    let id = doc.sections.len() as u32 + 1;
    body.push_str(&bookmark_start(id, &provenance.id));
    body.push_str(&section_inner_xml(&provenance, &dialect));
    body.push_str(&bookmark_end(id));
    format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>\n<w:document xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\" xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\"><w:body>{body}\
         <w:sectPr><w:headerReference w:type=\"default\" r:id=\"rId4\"/><w:footerReference w:type=\"default\" r:id=\"rId5\"/>\
         <w:pgSz w:w=\"11906\" w:h=\"16838\"/><w:pgMar w:top=\"1134\" w:right=\"1134\" w:bottom=\"1134\" w:left=\"1134\" w:header=\"567\" w:footer=\"567\" w:gutter=\"0\"/></w:sectPr>\
         </w:body></w:document>"
    )
}

fn styles_xml() -> String {
    let heading = |n: u8, size: u8| {
        format!(
            "<w:style w:type=\"paragraph\" w:styleId=\"Heading{n}\"><w:name w:val=\"heading {n}\"/><w:basedOn w:val=\"Normal\"/><w:next w:val=\"Normal\"/>\
             <w:qFormat/><w:pPr><w:keepNext/><w:spacing w:before=\"240\" w:after=\"80\"/><w:outlineLvl w:val=\"{}\"/></w:pPr><w:rPr><w:b/><w:sz w:val=\"{size}\"/></w:rPr></w:style>",
            n - 1
        )
    };
    format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>\n<w:styles xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\">\
         <w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii=\"Arial\" w:hAnsi=\"Arial\" w:cs=\"Arial\" w:eastAsia=\"Arial\"/><w:sz w:val=\"21\"/><w:lang w:val=\"en-IN\"/></w:rPr></w:rPrDefault>\
         <w:pPrDefault><w:pPr><w:spacing w:after=\"100\" w:line=\"264\" w:lineRule=\"auto\"/></w:pPr></w:pPrDefault></w:docDefaults>\
         <w:style w:type=\"paragraph\" w:default=\"1\" w:styleId=\"Normal\"><w:name w:val=\"Normal\"/><w:qFormat/></w:style>\
         <w:style w:type=\"paragraph\" w:styleId=\"Title\"><w:name w:val=\"Title\"/><w:basedOn w:val=\"Normal\"/><w:next w:val=\"Normal\"/><w:qFormat/><w:pPr><w:spacing w:after=\"160\"/></w:pPr><w:rPr><w:b/><w:sz w:val=\"34\"/></w:rPr></w:style>\
         {}{}{}\
         <w:style w:type=\"paragraph\" w:styleId=\"Caption\"><w:name w:val=\"caption\"/><w:basedOn w:val=\"Normal\"/><w:pPr><w:keepNext/></w:pPr><w:rPr><w:i/><w:sz w:val=\"19\"/></w:rPr></w:style>\
         <w:style w:type=\"paragraph\" w:styleId=\"ListParagraph\"><w:name w:val=\"List Paragraph\"/><w:basedOn w:val=\"Normal\"/><w:pPr><w:ind w:left=\"720\"/></w:pPr></w:style>\
         <w:style w:type=\"paragraph\" w:styleId=\"GapNote\"><w:name w:val=\"Gap Note\"/><w:basedOn w:val=\"Normal\"/><w:pPr><w:shd w:val=\"clear\" w:color=\"auto\" w:fill=\"FFF2CC\"/></w:pPr><w:rPr><w:b/><w:color w:val=\"C00000\"/></w:rPr></w:style>\
         <w:style w:type=\"paragraph\" w:styleId=\"DraftBanner\"><w:name w:val=\"Draft Banner\"/><w:basedOn w:val=\"Normal\"/><w:pPr><w:jc w:val=\"center\"/></w:pPr><w:rPr><w:b/><w:color w:val=\"C00000\"/></w:rPr></w:style>\
         <w:style w:type=\"paragraph\" w:styleId=\"Header\"><w:name w:val=\"header\"/><w:basedOn w:val=\"Normal\"/><w:rPr><w:sz w:val=\"17\"/></w:rPr></w:style>\
         <w:style w:type=\"paragraph\" w:styleId=\"Footer\"><w:name w:val=\"footer\"/><w:basedOn w:val=\"Normal\"/><w:rPr><w:sz w:val=\"17\"/></w:rPr></w:style>\
         <w:style w:type=\"table\" w:styleId=\"TableGrid\"><w:name w:val=\"Table Grid\"/><w:tblPr><w:tblCellMar><w:left w:w=\"80\" w:type=\"dxa\"/><w:right w:w=\"80\" w:type=\"dxa\"/></w:tblCellMar></w:tblPr></w:style>\
         </w:styles>",
        heading(1, 28),
        heading(2, 24),
        heading(3, 22)
    )
}

fn numbering_xml() -> String {
    format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>\n<w:numbering xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\">\
         <w:abstractNum w:abstractNumId=\"{BULLET_NUM_ID}\"><w:multiLevelType w:val=\"singleLevel\"/>\
         <w:lvl w:ilvl=\"0\"><w:start w:val=\"1\"/><w:numFmt w:val=\"bullet\"/><w:lvlText w:val=\"\u{2022}\"/><w:lvlJc w:val=\"left\"/>\
         <w:pPr><w:ind w:left=\"720\" w:hanging=\"360\"/></w:pPr></w:lvl></w:abstractNum>\
         <w:num w:numId=\"{BULLET_NUM_ID}\"><w:abstractNumId w:val=\"{BULLET_NUM_ID}\"/></w:num></w:numbering>"
    )
}

fn header_xml(doc: &AuthoredDocument) -> String {
    format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>\n<w:hdr xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\"><w:p><w:pPr><w:pStyle w:val=\"Header\"/></w:pPr>{}</w:p></w:hdr>",
        run(&format!("{} · {}", doc.classification, doc.title))
    )
}

fn footer_xml() -> String {
    format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>\n<w:ftr xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\"><w:p><w:pPr><w:pStyle w:val=\"Footer\"/></w:pPr>\
         {}<w:fldSimple w:instr=\" PAGE \"><w:r><w:t>1</w:t></w:r></w:fldSimple>{}<w:fldSimple w:instr=\" NUMPAGES \"><w:r><w:t>1</w:t></w:r></w:fldSimple></w:p></w:ftr>",
        run("DRAFT - not approved · Page "),
        run(" of ")
    )
}

const CONTENT_TYPES: &str = "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>\n<Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\">\
<Default Extension=\"rels\" ContentType=\"application/vnd.openxmlformats-package.relationships+xml\"/><Default Extension=\"xml\" ContentType=\"application/xml\"/>\
<Override PartName=\"/word/document.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml\"/>\
<Override PartName=\"/word/styles.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml\"/>\
<Override PartName=\"/word/settings.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.wordprocessingml.settings+xml\"/>\
<Override PartName=\"/word/numbering.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.wordprocessingml.numbering+xml\"/>\
<Override PartName=\"/word/header1.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.wordprocessingml.header+xml\"/>\
<Override PartName=\"/word/footer1.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.wordprocessingml.footer+xml\"/>\
<Override PartName=\"/docProps/core.xml\" ContentType=\"application/vnd.openxmlformats-package.core-properties+xml\"/>\
<Override PartName=\"/docProps/app.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.extended-properties+xml\"/></Types>";

const ROOT_RELS: &str = "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>\n<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">\
<Relationship Id=\"rId1\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument\" Target=\"word/document.xml\"/>\
<Relationship Id=\"rId2\" Type=\"http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties\" Target=\"docProps/core.xml\"/>\
<Relationship Id=\"rId3\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties\" Target=\"docProps/app.xml\"/></Relationships>";

const DOCUMENT_RELS: &str = "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>\n<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">\
<Relationship Id=\"rId1\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles\" Target=\"styles.xml\"/>\
<Relationship Id=\"rId2\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/settings\" Target=\"settings.xml\"/>\
<Relationship Id=\"rId3\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/numbering\" Target=\"numbering.xml\"/>\
<Relationship Id=\"rId4\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/header\" Target=\"header1.xml\"/>\
<Relationship Id=\"rId5\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/footer\" Target=\"footer1.xml\"/></Relationships>";

fn settings_xml() -> String {
    format!("<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>\n<w:settings xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\"><w:zoom w:percent=\"100\"/><w:defaultTabStop w:val=\"720\"/><w:characterSpacingControl w:val=\"doNotCompress\"/></w:settings>")
}

fn core_xml(doc: &AuthoredDocument) -> String {
    format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>\n<cp:coreProperties xmlns:cp=\"http://schemas.openxmlformats.org/package/2006/metadata/core-properties\" xmlns:dc=\"http://purl.org/dc/elements/1.1/\" xmlns:dcterms=\"http://purl.org/dc/terms/\" xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\">\
         <dc:title>{}</dc:title><dc:creator>ARJUN</dc:creator><cp:lastModifiedBy>ARJUN</cp:lastModifiedBy><dc:subject>{}</dc:subject><cp:keywords>{}</cp:keywords><cp:contentStatus>Draft</cp:contentStatus>\
         <dcterms:created xsi:type=\"dcterms:W3CDTF\">{}</dcterms:created></cp:coreProperties>",
        escape(&doc.title),
        escape(&doc.audience),
        escape(&doc.template),
        escape(&doc.stamp.created_at)
    )
}

const APP_XML: &str = "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>\n<Properties xmlns=\"http://schemas.openxmlformats.org/officeDocument/2006/extended-properties\"><Application>ARJUN</Application></Properties>";

/// The whole package: document, styles, settings, numbering, header,
/// footer, properties.
pub fn write_docx(doc: &AuthoredDocument) -> Result<Vec<u8>, String> {
    let parts = [
        ("[Content_Types].xml", CONTENT_TYPES.to_string()),
        ("_rels/.rels", ROOT_RELS.to_string()),
        ("word/document.xml", document_xml(doc)),
        ("word/_rels/document.xml.rels", DOCUMENT_RELS.to_string()),
        ("word/styles.xml", styles_xml()),
        ("word/settings.xml", settings_xml()),
        ("word/numbering.xml", numbering_xml()),
        ("word/header1.xml", header_xml(doc)),
        ("word/footer1.xml", footer_xml()),
        ("docProps/core.xml", core_xml(doc)),
        ("docProps/app.xml", APP_XML.to_string()),
    ];
    super::ooxml::package_bytes(&parts).map_err(|e| format!("the document could not be packaged: {e}"))
}

// ── Reading sections back ────────────────────────────────────────────────

/// One section of an existing version, found by its marker.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SectionRange {
    pub id: String,
    pub bookmark_id: String,
    pub heading: String,
    /// Byte offsets in `word/document.xml` of the content between the
    /// markers (the markers themselves are outside).
    #[serde(skip)]
    pub inner_start: usize,
    #[serde(skip)]
    pub inner_end: usize,
    /// Whether the markers sit at body level, where this writer puts them.
    pub body_level: bool,
    pub text: String,
    pub gap: bool,
}

/// The sections a document marks, in order.
pub fn read_sections(document_xml: &str) -> Result<Vec<SectionRange>, String> {
    let spans = xml_events::scan_spans(document_xml)?;
    let mut depth_stack: Vec<String> = Vec::new();
    let mut starts: Vec<(String, String, usize, bool)> = Vec::new(); // (id, bookmark id, inner_start, body_level)
    let mut out = Vec::new();
    for spanned in &spans {
        match &spanned.event {
            Event::Start { name, empty, .. } => {
                let local = xml_events::local(name);
                if local == "bookmarkStart" {
                    if let Some(section) = spanned.event.attribute("w:name").and_then(|n| n.strip_prefix(BOOKMARK_PREFIX)) {
                        let body_level = depth_stack.last().map(|p| xml_events::local(p) == "body").unwrap_or(false);
                        starts.push((section.to_string(), spanned.event.attribute("w:id").unwrap_or_default().to_string(), spanned.end, body_level));
                    }
                } else if local == "bookmarkEnd" {
                    let id = spanned.event.attribute("w:id").unwrap_or_default();
                    if let Some(position) = starts.iter().position(|(_, bid, _, _)| bid == id) {
                        let (section, bookmark_id, inner_start, start_body) = starts.remove(position);
                        let end_body = depth_stack.last().map(|p| xml_events::local(p) == "body").unwrap_or(false);
                        let inner_end = spanned.start;
                        let inner = &document_xml[inner_start.min(inner_end)..inner_end];
                        let (heading, text) = section_text(inner);
                        out.push(SectionRange {
                            gap: text.contains("INFORMATION NEEDED:"),
                            id: section,
                            bookmark_id,
                            heading,
                            inner_start,
                            inner_end,
                            body_level: start_body && end_body,
                            text,
                        });
                    }
                }
                if !empty {
                    depth_stack.push(name.clone());
                }
            }
            Event::End { .. } => {
                depth_stack.pop();
            }
            Event::Text(_) => {}
        }
    }
    out.sort_by_key(|s| s.inner_start);
    Ok(out)
}

/// The first paragraph's text (the heading) and all the text, of a range.
fn section_text(inner: &str) -> (String, String) {
    let Ok(events) = xml_events::scan(&format!("<x>{inner}</x>")) else {
        return (String::new(), String::new());
    };
    let mut heading = String::new();
    let mut all = String::new();
    let mut paragraphs = 0usize;
    for event in events {
        match &event {
            Event::Start { .. } if event.local_name() == Some("p") => {
                paragraphs += 1;
                if !all.is_empty() {
                    all.push('\n');
                }
            }
            Event::Text(text) => {
                if paragraphs == 1 {
                    heading.push_str(text);
                }
                all.push_str(text);
            }
            _ => {}
        }
    }
    (heading.trim().to_string(), all)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Evidence;
    impl CitationResolver for Evidence {
        fn resolve(&self, target: &CitationTarget, marker: &str) -> Option<ResolvedCitation> {
            match target {
                CitationTarget::Evidence { number: 1 } => Some(ResolvedCitation {
                    marker: marker.into(),
                    label: "UT survey, point C, page 1".into(),
                    text: "Governing reading at point C: 8.2 mm; pitting noted.".into(),
                    calculation: false,
                    stale_because: None,
                }),
                CitationTarget::Evidence { number: 2 } => Some(ResolvedCitation {
                    marker: marker.into(),
                    label: "SOP-114 rev D §3.1".into(),
                    text: "Where pitting is recorded the minimum allowable thickness is 9.0 mm.".into(),
                    calculation: false,
                    stale_because: None,
                }),
                CitationTarget::Memory { item_id, .. } if item_id == "mi-old" => Some(ResolvedCitation {
                    marker: marker.into(),
                    label: "a corrected reading".into(),
                    text: "8.2 mm".into(),
                    calculation: false,
                    stale_because: Some("corrected by mi-new".into()),
                }),
                CitationTarget::Calculation { calculation_id } if calculation_id == "calc-0123456789abcdef" => Some(ResolvedCitation {
                    marker: marker.into(),
                    label: "calc-0123456789abcdef wall loss = 8.889 %".into(),
                    text: "wall_loss = (t_min - t_meas) / t_min = 8.889 % with t_min = 9.0 mm, t_meas = 8.2 mm".into(),
                    calculation: true,
                    stale_because: None,
                }),
                _ => None,
            }
        }
    }

    fn stamp() -> Stamp {
        Stamp { task_id: "r".into(), model: "fixture".into(), created_at: "2026-09-25T10:00:00Z".into() }
    }

    fn paragraph(text: &str) -> Block {
        Block::Paragraph { text: text.into() }
    }

    pub(crate) fn note_spec() -> DocumentSpec {
        DocumentSpec {
            template: "inspection_approval_note@1".into(),
            title: "V-101 shell wall loss at point C".into(),
            audience: "Unit 4 plant manager".into(),
            output: "v101-note.docx".into(),
            fields: [("recipient".to_string(), "Plant Manager, Unit 4".to_string())].into_iter().collect(),
            sections: vec![
                SpecSection { id: "subject".into(), heading: None, level: None, blocks: vec![paragraph("Shell thickness at point C of V-101 after the August survey.")], gap: None },
                SpecSection {
                    id: "findings".into(),
                    heading: None,
                    level: None,
                    blocks: vec![
                        paragraph("The governing reading at point C is 8.2 mm, with pitting recorded [E1]."),
                        paragraph("With pitting recorded the minimum allowable thickness is 9.0 mm [E2]."),
                    ],
                    gap: None,
                },
                SpecSection { id: "calculation".into(), heading: None, level: None, blocks: vec![paragraph("Wall loss against the minimum is 8.889 % [C:calc-0123456789abcdef].")], gap: None },
                SpecSection { id: "recommendation".into(), heading: None, level: None, blocks: Vec::new(), gap: Some("no decision on repair or de-rating has been recorded".into()) },
                SpecSection { id: "assumptions".into(), heading: None, level: None, blocks: vec![paragraph("The survey instrument was within its calibration interval.")], gap: None },
            ],
        }
    }

    #[test]
    fn a_sound_spec_composes_with_generated_references_and_a_reported_gap() {
        let composed = compose(&note_spec(), "Internal", &stamp(), &Evidence).expect("composes");
        let ids: Vec<&str> = composed.document.sections.iter().map(|s| s.id.as_str()).collect();
        assert_eq!(ids, vec!["subject", "findings", "calculation", "recommendation", "assumptions", "references"]);
        assert_eq!(composed.gaps.len(), 1);
        assert_eq!(composed.gaps[0].at, "recommendation");
        assert_eq!(composed.citations.len(), 3);
        let findings = composed.lineage.iter().find(|l| l.section_id == "findings").unwrap();
        assert_eq!(findings.markers, vec!["[E1]", "[E2]"]);
        let calculation = composed.lineage.iter().find(|l| l.section_id == "calculation").unwrap();
        assert_eq!(calculation.calculations, vec!["[C:calc-0123456789abcdef]"]);
    }

    #[test]
    fn a_missing_mandatory_field_or_section_is_refused_by_name() {
        let mut spec = note_spec();
        spec.fields.clear();
        spec.sections.retain(|s| s.id != "findings");
        let problems = compose(&spec, "Internal", &stamp(), &Evidence).unwrap_err();
        let codes: Vec<&str> = problems.iter().map(|p| p.code).collect();
        assert!(codes.contains(&"field_missing") && codes.contains(&"section_missing"), "{problems:?}");
        let mut unknown = note_spec();
        unknown.fields.insert("recipient".into(), "?".into());
        let composed = compose(&unknown, "Internal", &stamp(), &Evidence).expect("an unknown field is a gap, not a refusal");
        assert!(composed.gaps.iter().any(|g| g.at == "field:recipient"));
    }

    #[test]
    fn an_unsupported_claim_or_figure_is_refused() {
        let mut spec = note_spec();
        spec.sections[1].blocks.push(paragraph("The shell is otherwise sound."));
        spec.sections[1].blocks.push(paragraph("Point D reads 9.4 mm [E1]."));
        let problems = compose(&spec, "Internal", &stamp(), &Evidence).unwrap_err();
        assert!(problems.iter().any(|p| p.code == "unsupported_claim" && p.detail.contains("otherwise sound")), "{problems:?}");
        assert!(problems.iter().any(|p| p.code == "unsupported_figure" && p.detail.contains("9.4 mm")), "{problems:?}");
        let mut uncited = note_spec();
        uncited.sections[2].blocks = vec![paragraph("Wall loss is 8.889 % of the minimum [E1].")];
        let problems = compose(&uncited, "Internal", &stamp(), &Evidence).unwrap_err();
        assert!(problems.iter().any(|p| p.code == "calculation_citation"), "{problems:?}");
    }

    #[test]
    fn a_corrected_source_and_an_unbound_marker_are_refused() {
        let mut spec = note_spec();
        spec.sections[1].blocks.push(paragraph("An earlier reading was 8.2 mm [M:mi-old@1]."));
        spec.sections[1].blocks.push(paragraph("See also [E9]."));
        let problems = compose(&spec, "Internal", &stamp(), &Evidence).unwrap_err();
        assert!(problems.iter().any(|p| p.code == "stale_citation"), "{problems:?}");
        assert!(problems.iter().any(|p| p.code == "unbound_citation" && p.detail.contains("[E9]")), "{problems:?}");
    }

    #[test]
    fn the_package_is_a_complete_word_document_with_section_markers() {
        let composed = compose(&note_spec(), "Internal", &stamp(), &Evidence).unwrap();
        let bytes = write_docx(&composed.document).unwrap();
        let report = crate::artifacts::package::inspect(&bytes, &crate::artifacts::package::LIMITS);
        assert!(report.safe_to_open(), "{report:?}");
        assert_eq!(report.detected, crate::artifacts::package::DetectedFormat::Docx);
        for part in ["word/styles.xml", "word/numbering.xml", "word/header1.xml", "word/footer1.xml", "word/settings.xml"] {
            assert!(report.parts.iter().any(|p| p == part), "{part} missing");
        }
        let xml = crate::artifacts::package::read_part(&bytes, "word/document.xml").unwrap();
        let sections = read_sections(&xml).unwrap();
        let ids: Vec<&str> = sections.iter().map(|s| s.id.as_str()).collect();
        assert_eq!(ids, vec!["subject", "findings", "calculation", "recommendation", "assumptions", "references", "provenance"]);
        assert!(sections.iter().all(|s| s.body_level));
        assert!(sections.iter().find(|s| s.id == "recommendation").unwrap().gap);
        assert_eq!(sections[1].heading, "Findings");
        let model = crate::artifacts::content::extract(&bytes).unwrap();
        assert!(model.units.iter().any(|u| u.text.contains("8.2 mm")));
    }
}
