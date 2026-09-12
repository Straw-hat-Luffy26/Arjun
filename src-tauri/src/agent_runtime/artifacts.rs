//! Producing the two deliverables a run can hand to somebody.
//!
//! PS 26117: *"Output should be real deliverables, approval notes, PPT/Word/Excel
//! files, working code, calculations with steps shown, not just chat replies."*
//! [`crate::artifacts`] already renders those files. What was missing was a way
//! for a run to reach it, and `runner.rs` said so plainly rather than pretending
//! otherwise.
//!
//! ## Why the model supplies the content directly
//!
//! [`crate::artifacts::production::produce`] runs its own correction loop: ask a
//! model for the fields, render, and if the renderer objects, ask again with the
//! objection. That was the right design when the model was reachable from Rust.
//!
//! It is not any more. The model lives in the Node runtime, and a Rust-side
//! correction loop would have to call back into it — Rust asking the runtime to
//! ask the model, while the runtime waits on Rust. The agent loop already *is* a
//! correction loop: a failed render comes back as a tool error, the model reads
//! the objection and calls again. So the tool takes the fields as arguments and
//! `produce`'s loop is simply not the one used here.
//!
//! That is why `ToolSpec` for these tools already declares a `content` object
//! argument. The contract anticipated this shape.
//!
//! ## Why the workbook is not composed by the model at all
//!
//! A calculation workbook is written from what the calculation engine actually
//! computed during the run, not from what the model says it computed. The engine
//! exists precisely because a model is usually about right, and "usually" is not
//! a property a pump specification can have.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};

use crate::artifacts::{
    check_deck, check_document, check_workbook, write_deck, write_document, write_workbook,
    DocumentMetadata, Slide, BRIEFING_SECTIONS,
};
use crate::identity::Session;
use crate::orchestrator::calculation::CalculationRecord;
use crate::orchestrator::tools::ToolCall;

use super::CallParams;

/// What kind of file a run produced, and therefore how it is checked.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Kind {
    Document,
    Workbook,
    /// A briefing deck.
    Deck,
    /// A PDF report or note. Not a `Document`: that means a `.docx`, and the
    /// check for one asks whether the template's sections are present, which a
    /// PDF has no answer to.
    Pdf,
    /// A diagram, written as SVG.
    Diagram,
    /// A note or draft. Checked for being present and non-empty, no more —
    /// there is no structure to check it against.
    Text,
}

impl Kind {
    pub fn label(self) -> &'static str {
        match self {
            Kind::Document => "Word document",
            Kind::Workbook => "Workbook",
            Kind::Deck => "Briefing deck",
            Kind::Pdf => "PDF",
            Kind::Diagram => "Diagram",
            Kind::Text => "Text file",
        }
    }
}

/// A file this run produced, remembered so it can be re-opened afterwards.
///
/// The template is kept because a document cannot really be checked without it:
/// the check asks whether the sections the template promised are in the file,
/// and a checker that does not know which template was used can only ask
/// whether the ZIP opens.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Produced {
    /// What to call it. Relative to the run's workspace, which is where it is.
    pub name: String,
    pub path: String,
    pub kind: Kind,
    pub template: Option<String>,
    /// RFC 3339, UTC.
    pub produced_at: String,
}

/// Files produced so far, keyed by run id — the same shape as the calculation
/// and evidence tables, and per run for the same reason.
pub type RunArtifacts = Arc<Mutex<HashMap<String, Vec<Produced>>>>;

/// Records a produced file against its run.
///
/// A path written twice replaces the earlier entry rather than appearing twice:
/// the model correcting a document it just produced is ordinary, and a list
/// showing one file as two deliverables would misreport what the run made.
pub fn remember(table: &RunArtifacts, run_id: &str, produced: Produced) {
    if let Ok(mut table) = table.lock() {
        let entries = table.entry(run_id.to_string()).or_default();
        if let Some(existing) = entries.iter_mut().find(|kept| kept.path == produced.path) {
            *existing = produced;
        } else {
            entries.push(produced);
        }
    }
}

/// Everything this run produced, in the order it was produced.
pub fn for_run(table: &RunArtifacts, run_id: &str) -> Vec<Produced> {
    table
        .lock()
        .ok()
        .and_then(|table| table.get(run_id).cloned())
        .unwrap_or_default()
}

/// Drops a finished run's list once its report has been written.
pub fn forget(table: &RunArtifacts, run_id: &str) {
    if let Ok(mut table) = table.lock() {
        table.remove(run_id);
    }
}

/// Builds the record of a produced file.
///
/// The name is relative to the workspace, so the name the model wrote is the
/// name that comes back — an absolute path with a UUID in it tells the person
/// reading the task nothing they wanted to know.
pub fn produced_from(
    path: &Path,
    root: Option<&Path>,
    kind: Kind,
    template: Option<String>,
) -> Produced {
    let name = root
        .and_then(|root| path.strip_prefix(root).ok())
        .map(|relative| relative.display().to_string())
        .unwrap_or_else(|| {
            Path::new(path.file_name().unwrap_or_default())
                .display()
                .to_string()
        });
    Produced {
        name,
        path: path.display().to_string(),
        kind,
        template,
        produced_at: chrono::Utc::now().to_rfc3339(),
    }
}

/// What re-opening a produced file found.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ArtifactReport {
    pub name: String,
    pub path: String,
    pub kind: Kind,
    /// Carried through so a later re-check asks the same question this one did.
    /// A document checked against a different template than it was rendered
    /// from reports missing sections that were never promised.
    #[serde(default)]
    pub template: Option<String>,
    pub bytes: u64,
    /// False when the file is missing, empty, will not open, or is missing
    /// something the template promised.
    pub sound: bool,
    /// One line, in the words somebody reading the task would use.
    pub detail: String,
    pub problems: Vec<String>,
    pub produced_at: String,
}

/// Re-opens a produced file and reports what is actually in it.
///
/// ARJUN design rule 30 asks that the application open the generated file locally and
/// confirm it is not corrupt and that required sections exist. Checking the
/// file rather than the code that wrote it is the whole point: a bug between
/// the template and the ZIP passes every test of the template and still
/// produces a document that opens to a page of placeholders.
pub fn check(produced: &Produced) -> ArtifactReport {
    let path = PathBuf::from(&produced.path);
    let report = |sound: bool, detail: String, problems: Vec<String>, bytes: u64| ArtifactReport {
        name: produced.name.clone(),
        path: produced.path.clone(),
        kind: produced.kind,
        template: produced.template.clone(),
        bytes,
        sound,
        detail,
        problems,
        produced_at: produced.produced_at.clone(),
    };

    let Ok(metadata) = std::fs::metadata(&path) else {
        return report(
            false,
            "The file is not where the task said it wrote it.".to_string(),
            vec!["the file does not exist".to_string()],
            0,
        );
    };
    let bytes = metadata.len();
    if bytes == 0 {
        return report(
            false,
            "The file was created but nothing was written into it.".to_string(),
            vec!["the file is empty".to_string()],
            0,
        );
    }

    match produced.kind {
        Kind::Document => {
            let template = produced.template.as_deref().unwrap_or("approval_note");
            let check = check_document(&path, template);
            let detail = if check.is_sound() {
                format!(
                    "Opens, and holds the {} section(s) the {template} template promises.",
                    check.sections.len()
                )
            } else if check.opens {
                "Opens, but does not hold everything the template promises.".to_string()
            } else {
                "Does not open as a Word document.".to_string()
            };
            report(check.is_sound(), detail, check.problems, bytes)
        }
        Kind::Workbook => {
            let check = check_workbook(&path);
            let detail = if check.is_sound() {
                format!(
                    "Opens, with {} calculation(s), {} of them live formulas Excel recomputes.",
                    check.calculations, check.live_formulas
                )
            } else if check.opens {
                "Opens, but the working in it is not sound.".to_string()
            } else {
                "Does not open as a workbook.".to_string()
            };
            report(check.is_sound(), detail, check.problems, bytes)
        }
        // Nothing to check it against beyond being there and having content,
        // and claiming more than that would be inventing a standard.
        Kind::Deck => {
            let check = check_deck(&path);
            let detail = if check.is_sound() {
                format!(
                    "Opens, with {} slide(s): {}.",
                    check.slides,
                    check.headings.join("; ")
                )
            } else if check.opens {
                "Opens, but the deck is not sound.".to_string()
            } else {
                "Does not open as a presentation.".to_string()
            };
            report(check.is_sound(), detail, check.problems, bytes)
        }
        // A PDF is read back: header, cross-reference table, trailer, page
        // tree, content streams and the text they paint. See
        // `artifacts::pdf_validate` for what that does and does not claim.
        //
        // This used to be "Present, N byte(s)." alongside SVG and text, with a
        // comment arguing that verifying a PDF "would be inventing a standard
        // this has no way to hold anything to". That was true while nothing
        // could read the file. It stopped being true once the writer emitted a
        // real object graph, and in the meantime a PDF truncated halfway
        // through its object table passed as a finished deliverable.
        Kind::Pdf => {
            let check = crate::artifacts::pdf_validate::check_pdf(&path);
            let mut problems = check.problems.clone();
            // Quality is a separate claim from validity, and both have to hold
            // before a person is handed the file. A structurally perfect PDF of
            // near-blank pages is not a deliverable.
            if problems.is_empty() {
                problems.extend(crate::artifacts::pdf_validate::quality::inspect(&check));
            }
            let sound = problems.is_empty();
            let detail = if sound {
                format!(
                    "Opens, {} page(s), {} characters of text, titled.",
                    check.pages, check.characters
                )
            } else if check.opens {
                "Opens, but is not sound.".to_string()
            } else {
                "Does not open as a PDF.".to_string()
            };
            report(sound, detail, problems, bytes)
        }
        // An SVG is parsed: root element, viewBox, the shapes and labels it
        // declares, and that every internal reference resolves.
        Kind::Diagram => {
            let check = crate::artifacts::svg_validate::check_svg(&path);
            let mut problems = check.problems.clone();
            if problems.is_empty() {
                problems.extend(crate::artifacts::svg_validate::quality::inspect(&check));
            }
            let sound = problems.is_empty();
            let detail = if sound {
                format!(
                    "Parses, {} shape(s) and {} label(s) inside the viewBox.",
                    check.shapes, check.labels
                )
            } else if check.parses {
                "Parses, but is not sound.".to_string()
            } else {
                "Does not parse as SVG.".to_string()
            };
            report(sound, detail, problems, bytes)
        }
        // A note has no structure to check it against, and claiming otherwise
        // would be inventing a standard. Format-aware text files are checked by
        // their own validators before they are written — see
        // `artifacts::text_formats` — so what reaches here is genuinely
        // unstructured.
        Kind::Text => report(true, format!("Present, {bytes} byte(s)."), Vec::new(), bytes),
    }
}

/// Re-opens everything a run produced.
pub fn report_for_run(table: &RunArtifacts, run_id: &str) -> Vec<ArtifactReport> {
    for_run(table, run_id).iter().map(check).collect()
}

/// Renders a Word document from fields the model supplied.
pub fn create_docx(
    call: &CallParams,
    resolved_path: Option<&Path>,
    session: &Session,
    tool_call: &ToolCall,
) -> Result<String, String> {
    create_docx_with_evidence(call, resolved_path, session, tool_call, &[], &[])
}

/// Produces a Word document through the correction loop.
///
/// ## What changed, and why it matters
///
/// This used to render once, check once, and on failure return a sentence
/// asking the model to try again — leaving the broken file on disk under the
/// name the person had been told to expect. `artifacts::production::produce`
/// has always implemented the loop this needs (compose, verify, render,
/// re-open, feed the renderer's own objections back, revise rather than
/// overwrite) and had no production caller at all; it was reachable only from
/// its own tests.
///
/// It is now the live path. See [`crate::artifacts::live_source`] for the part
/// that could not simply be plugged in: `produce` expects to be able to *ask*
/// the model, and a tool handler cannot — it is already inside the model's
/// call. So the inner loop makes the repairs that need no new information, and
/// the outer loop is the model's own next turn, which now receives the
/// renderer's objections by field name.
///
/// `passages` and `calculations` are what the run actually retrieved and
/// computed. They decide the draft's standing: a document whose claims the run
/// cannot support is stamped DRAFT rather than presented as finished, and that
/// is settled before the file is written rather than after.
pub fn create_docx_with_evidence(
    call: &CallParams,
    resolved_path: Option<&Path>,
    _session: &Session,
    tool_call: &ToolCall,
    passages: &[crate::knowledge::SearchResult],
    calculations: &[crate::orchestrator::calculation::CalculationRecord],
) -> Result<String, String> {
    let path = resolved_path.ok_or_else(|| {
        "No path was resolved for the document, so nothing was written.".to_string()
    })?;

    // The general path, when the caller composed a document rather than filling
    // in the one template.
    //
    // `sections` and `template` are alternatives. A run that supplies sections
    // is writing a procedure, a report, a specification or a set of minutes —
    // none of which the approval-note template can express — and it goes
    // through the same validate / repair / render / re-open loop.
    if tool_call.arguments.get("sections").is_some() {
        return create_docx_from_sections(call, path, tool_call);
    }

    let template = tool_call.text("template").ok_or_else(|| {
        "The document needs either a \"template\" or a \"sections\" list. Available templates: \
         approval_note."
            .to_string()
    })?;

    // Checked here rather than left to `produce`, whose refusal names the
    // template that was asked for but not the ones that exist. A model told
    // only "there is no invoice template" guesses again; one told what is
    // available picks.
    if crate::artifacts::docx::template_for(&template).is_none() {
        return Err(format!(
            "There is no {template:?} template. Available templates: approval_note."
        ));
    }

    let content = fields_from(tool_call)?;
    let supplied = content.len();

    let metadata = DocumentMetadata {
        task_id: call.run_id.clone(),
        created_at: chrono::Utc::now().to_rfc3339(),
        // The model that produced the content, so a reader knows what wrote it.
        // Recorded per run by the caller; unknown here rather than guessed.
        model: call.model.clone().unwrap_or_else(|| "unrecorded".to_string()),
        classification: "Internal".to_string(),
        // Overwritten per attempt by `produce`, which settles the standing from
        // the verifier before it stamps the page.
        is_draft: true,
    };

    let evidence = crate::artifacts::verifier::Evidence {
        // A document assembled from a run's own findings is a claim about the
        // organisation's record, and has to rest on what the run retrieved.
        grounding: crate::artifacts::verifier::Grounding::OrganisationRecord,
        passages,
        calculations,
        unread_pages: &[],
    };

    let mut source = crate::artifacts::live_source::ModelSupplied::new(content);
    let outcome = crate::artifacts::production::produce(
        path,
        &template,
        &mut source,
        &metadata,
        &evidence,
    );

    let Some(artifact) = outcome.artifact.clone() else {
        // Honest failure. Every attempt is named with what was wrong with it,
        // so the model's next turn has the field names rather than a summary.
        let attempts: Vec<String> = outcome
            .revisions
            .iter()
            .filter_map(|revision| {
                revision
                    .superseded_because
                    .as_ref()
                    .map(|why| format!("attempt {}: {why}", revision.number))
            })
            .collect();
        let failure = outcome
            .failure
            .unwrap_or_else(|| "the document could not be produced".to_string());
        return Err(if attempts.is_empty() {
            failure
        } else {
            format!("{failure} {}", attempts.join("; "))
        });
    };

    // The revision that stands is also placed at the path the caller asked
    // for.
    //
    // `produce` writes every attempt under its own number and never touches
    // the bare name, which is right for the evidence trail: the attempt that
    // was corrected stays on disk with the reason. But the caller asked for
    // `note.docx`, the artifact table records `note.docx`, the preview opens
    // `note.docx` and the person was told `note.docx` — so if only
    // `note.r2.docx` exists, every one of those points at nothing.
    //
    // Copied, not renamed: the numbered revision is the record and must
    // survive. What the person opens and what the run was judged on are then
    // byte-identical, which is the property that matters.
    if artifact != path {
        std::fs::copy(&artifact, path).map_err(|error| {
            format!(
                "{} was produced but could not be placed at {}: {error}",
                artifact.display(),
                path.display()
            )
        })?;
    }

    let superseded = outcome
        .revisions
        .iter()
        .filter(|revision| revision.superseded_because.is_some())
        .count();

    let standing = if outcome.is_ready() {
        "It passed its checks and its claims are supported by the run's evidence."
    } else {
        "It is marked DRAFT: a person has to look before it is relied on."
    };

    let corrected = if superseded == 0 {
        String::new()
    } else {
        format!(
            " {superseded} earlier attempt(s) were corrected and kept beside it as revisions."
        )
    };

    Ok(format!(
        "Wrote {} from the {template} template ({supplied} field(s) supplied), kept as {}.          {standing}{corrected}",
        path.display(),
        artifact
            .file_name()
            .map(|name| name.to_string_lossy().to_string())
            .unwrap_or_default()
    ))
}

/// Produces a workbook the model composed, sheet by sheet.
///
/// The shape accepted, which is [`crate::artifacts::doc_model::Sheet`]'s own
/// serialisation:
///
/// ```json
/// {
///   "sheets": [{
///     "name": "Readings",
///     "columns": [
///       {"header": "Point", "type": "text"},
///       {"header": "Measured", "type": "number"},
///       {"header": "Margin", "type": "formula"}
///     ],
///     "rows": [["S-01", "9.4", "=B2-9"]],
///     "freezeHeader": true
///   }]
/// }
/// ```
///
/// The column type is not decoration. A number written as text looks identical
/// on screen and every formula referring to it evaluates to zero, silently;
/// declaring the type is what lets the writer emit a numeric cell and the
/// validator catch prose in a column of numbers before anything is written.
fn create_xlsx_from_sheets(
    path: &Path,
    sheets: serde_json::Value,
    tool_call: Option<&ToolCall>,
) -> Result<String, String> {
    use crate::artifacts::doc_model::{Sheet, Workbook};

    let sheets: Vec<Sheet> = serde_json::from_value(sheets).map_err(|error| {
        format!(
            "The workbook's \"sheets\" could not be read: {error}. Each sheet is              {{\"name\": \"...\", \"columns\": [{{\"header\": \"...\", \"type\":              \"number\"}}], \"rows\": [[\"...\"]]}}. Column types: text, number, currency,              percent, date, formula."
        )
    })?;

    let mut workbook = Workbook {
        title: tool_call
            .and_then(|call| call.text("title"))
            .map(str::to_string)
            .filter(|t| !t.trim().is_empty())
            .unwrap_or_else(|| "Workbook".to_string()),
        classification: tool_call
            .and_then(|call| call.text("classification"))
            .map(str::to_string)
            .filter(|c| !c.trim().is_empty())
            .unwrap_or_else(|| "Internal".to_string()),
        sheets,
    };

    let outcome = crate::artifacts::produce_model::produce_workbook(path, &mut workbook);
    let Some(artifact) = outcome.artifact.clone() else {
        return Err(outcome
            .failure
            .unwrap_or_else(|| "the workbook could not be produced".to_string()));
    };
    if artifact != path {
        std::fs::copy(&artifact, path).map_err(|error| {
            format!(
                "{} was produced but could not be placed at {}: {error}",
                artifact.display(),
                path.display()
            )
        })?;
    }

    let rows: usize = workbook.sheets.iter().map(|sheet| sheet.rows.len()).sum();
    let corrected = if outcome.repairs.is_empty() {
        String::new()
    } else {
        format!(" Corrected before writing: {}.", outcome.repairs.join("; "))
    };

    Ok(format!(
        "Wrote {} with {} sheet(s) and {rows} row(s), kept as {}.{corrected} Formulas are live:          Excel recomputes them when it opens the file.",
        path.display(),
        workbook.sheets.len(),
        artifact.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default()
    ))
}

/// Produces a document the model composed, section by section.
///
/// The shape accepted, which is [`crate::artifacts::doc_model::Document`]'s own
/// serialisation:
///
/// ```json
/// {
///   "title": "Unit Four shell thickness inspection",
///   "classification": "OFFICIAL",
///   "sections": [
///     { "heading": "Scope", "level": 1,
///       "blocks": [ { "kind": "paragraph", "text": "..." } ] },
///     { "heading": "Readings", "level": 2,
///       "blocks": [ { "kind": "table", "header": ["Point", "mm"],
///                     "rows": [["S-01", "9.4"]], "caption": "..." } ] }
///   ]
/// }
/// ```
///
/// Nothing here is written until the model validates: a ragged table, a heading
/// hierarchy that skips a level, an empty section and a placeholder are all
/// caught in memory, and the repairable ones are repaired before rendering.
fn create_docx_from_sections(
    call: &CallParams,
    path: &Path,
    tool_call: &ToolCall,
) -> Result<String, String> {
    use crate::artifacts::doc_model::{Document, Properties, Section};

    let sections: Vec<Section> = serde_json::from_value(
        tool_call.arguments.get("sections").cloned().unwrap_or(serde_json::Value::Null),
    )
    .map_err(|error| {
        format!(
            "The document's \"sections\" could not be read: {error}. Each section is \
             {{\"heading\": \"...\", \"level\": 1, \"blocks\": [{{\"kind\": \"paragraph\", \
             \"text\": \"...\"}}]}}. Block kinds: paragraph, bullets, numbered, table, pageBreak."
        )
    })?;

    let title = tool_call
        .text("title")
        .map(str::to_string)
        .filter(|t| !t.trim().is_empty())
        .ok_or_else(|| "A composed document needs a \"title\".".to_string())?;

    let mut document = Document {
        title,
        classification: tool_call
            .text("classification")
            .map(str::to_string)
            .filter(|c| !c.trim().is_empty())
            .unwrap_or_else(|| "Internal".to_string()),
        sections,
        properties: Properties::default(),
    };

    let metadata = DocumentMetadata {
        task_id: call.run_id.clone(),
        created_at: chrono::Utc::now().to_rfc3339(),
        model: call.model.clone().unwrap_or_else(|| "unrecorded".to_string()),
        classification: document.classification.clone(),
        // A composed document is a draft until a person signs it, exactly as a
        // templated one is.
        is_draft: true,
    };

    let outcome = crate::artifacts::produce_model::produce_document(path, &mut document, &metadata);
    let Some(artifact) = outcome.artifact.clone() else {
        return Err(outcome
            .failure
            .unwrap_or_else(|| "the document could not be produced".to_string()));
    };
    if artifact != path {
        std::fs::copy(&artifact, path).map_err(|error| {
            format!(
                "{} was produced but could not be placed at {}: {error}",
                artifact.display(),
                path.display()
            )
        })?;
    }

    let corrected = if outcome.repairs.is_empty() {
        String::new()
    } else {
        format!(" Corrected before writing: {}.", outcome.repairs.join("; "))
    };

    Ok(format!(
        "Wrote {} with {} section(s), kept as {}.{corrected} It is marked DRAFT until somebody \
         approves it.",
        path.display(),
        document.sections.len(),
        artifact.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default()
    ))
}

/// Reads the `content` object into template fields.
///
/// Values are required to be strings. A model that supplies a number or a
/// nested object for a document field has misunderstood the template, and
/// silently stringifying it would put `{"value":9}` into an approval note.
fn fields_from(tool_call: &ToolCall) -> Result<BTreeMap<String, String>, String> {
    let object = tool_call
        .arguments
        .get("content")
        .and_then(|value| value.as_object())
        .ok_or_else(|| {
            "The document's content must be an object of field name to text, for example \
             {\"title\": \"...\", \"recommendation\": \"...\"}."
                .to_string()
        })?;

    let mut fields = BTreeMap::new();
    for (key, value) in object {
        match value.as_str() {
            Some(text) => {
                fields.insert(key.clone(), text.to_string());
            }
            None => {
                return Err(format!(
                    "The field {key:?} must be text, but was {}. Supply each field as a string.",
                    match value {
                        serde_json::Value::Null => "null",
                        serde_json::Value::Bool(_) => "a boolean",
                        serde_json::Value::Number(_) => "a number",
                        serde_json::Value::Array(_) => "a list",
                        _ => "an object",
                    }
                ))
            }
        }
    }
    Ok(fields)
}

/// Writes a briefing deck.
///
/// PS 26117 lists the deliverables as *"approval notes, PPT/Word/Excel files,
/// working code, calculations with steps shown"*. `artifacts::pptx` has been
/// able to write a deck since it was added; until this function it had no tool,
/// no command and no caller, so a run could not ask for one. The renderer was
/// finished work behind a missing doorway.
///
/// ## Why the sections are fixed
///
/// A deck is a template, exactly as an approval note is: the four headings in
/// [`BRIEFING_SECTIONS`] in that order, each with at least one bullet. The model
/// supplies the bullets, not the structure. A model free to invent headings
/// produces a deck whose shape depends on the prompt, and `check_deck` could not
/// then say whether what came back was what was asked for.
///
/// `Evidence` is required for the same reason a citation is required in a note:
/// a briefing whose findings have no source is the failure this whole codebase
/// is arranged to prevent.
pub fn create_pptx(
    call: &CallParams,
    resolved_path: Option<&Path>,
    tool_call: &ToolCall,
) -> Result<String, String> {
    let path = resolved_path
        .ok_or_else(|| "No path was resolved for the deck, so nothing was written.".to_string())?;

    let content = tool_call
        .arguments
        .get("content")
        .and_then(|value| value.as_object())
        .ok_or_else(|| {
            format!(
                "The deck's content must be an object with a \"title\" and one list of bullets per                  section. Sections: {}.",
                BRIEFING_SECTIONS.join(", ")
            )
        })?;

    let title = content
        .get("title")
        .and_then(|value| value.as_str())
        .filter(|text| !text.trim().is_empty())
        .ok_or_else(|| "The deck needs a \"title\", as non-empty text.".to_string())?;

    let mut sections = Vec::with_capacity(BRIEFING_SECTIONS.len());
    for heading in BRIEFING_SECTIONS {
        let key = heading.to_ascii_lowercase();
        let value = content.get(&key).ok_or_else(|| {
            format!(
                "The deck is missing the {heading:?} section. Supply {key:?} as a list of bullet                  strings. Sections: {}. Nothing was written.",
                BRIEFING_SECTIONS.join(", ")
            )
        })?;

        // One string is accepted as a single bullet: a section that is one
        // sentence is common, and refusing it would teach the model to wrap
        // every sentence in a list for no reason.
        let raw = match value {
            serde_json::Value::String(text) => vec![text.clone()],
            serde_json::Value::Array(items) => {
                let mut bullets = Vec::with_capacity(items.len());
                for (n, item) in items.iter().enumerate() {
                    let text = item.as_str().ok_or_else(|| {
                        format!(
                            "Bullet {} of {heading:?} must be text. Supply each bullet as a                              string.",
                            n + 1
                        )
                    })?;
                    bullets.push(text.to_string());
                }
                bullets
            }
            _ => {
                return Err(format!(
                    "{heading:?} must be a list of bullet strings, or one string."
                ))
            }
        };

        let bullets: Vec<String> = raw
            .into_iter()
            .filter(|bullet| !bullet.trim().is_empty())
            .collect();

        if bullets.is_empty() {
            // Refused rather than written empty. A slide with a heading and
            // nothing under it reads as a section that found nothing to report,
            // which is a different and worse claim than one never assembled.
            return Err(format!(
                "The {heading:?} section has no bullets, so the deck was not written. A section                  with a heading and nothing under it reads as a finding of nothing, which is not                  the same as having nothing to say."
            ));
        }

        sections.push(Slide {
            heading: (*heading).to_string(),
            bullets,
        });
    }

    // Through the content model and its production loop.
    //
    // The briefing template's contract is unchanged above: the same four
    // sections are still required and the same refusals still happen before
    // anything is written. What changes is what happens after — the sections
    // become a `doc_model::Deck`, which is validated and *repaired* in memory
    // before rendering, written as a numbered revision, and re-opened.
    //
    // The repair matters here more than anywhere else. `write_deck` silently
    // truncated any section past `BULLETS_PER_SLIDE` and reported the overflow
    // as a count on the slide; the model splits it across slides instead, so a
    // ten-bullet findings section becomes two slides rather than seven bullets
    // and a footnote saying three were dropped.
    let mut deck = crate::artifacts::doc_model::Deck {
        title: title.to_string(),
        classification: "Internal".to_string(),
        slides: sections
            .iter()
            .map(|slide| crate::artifacts::doc_model::SlideModel {
                heading: slide.heading.clone(),
                bullets: slide.bullets.clone(),
                table: None,
                notes: None,
            })
            .collect(),
    };

    // Every deck a run produces is a draft until a person signs it, for the same
    // reason every document is: the word goes on the slide, not only into a
    // field, so a file that escapes into an inbox still says what it is.
    let outcome = crate::artifacts::produce_model::produce_deck(path, &mut deck, true);
    let Some(artifact) = outcome.artifact.clone() else {
        return Err(outcome
            .failure
            .unwrap_or_else(|| "the deck could not be produced".to_string()));
    };

    // The accepted revision is also placed where the caller asked for it: the
    // artifact table, the preview and the person were all told this path.
    if artifact != path {
        std::fs::copy(&artifact, path).map_err(|error| {
            format!(
                "{} was produced but could not be placed at {}: {error}",
                artifact.display(),
                path.display()
            )
        })?;
    }

    let check = check_deck(path);
    let corrected = if outcome.repairs.is_empty() {
        String::new()
    } else {
        format!(" Corrected before writing: {}.", outcome.repairs.join("; "))
    };

    let _ = call;
    Ok(format!(
        "Wrote {} with {} slide(s): {}.{corrected} It is marked DRAFT until somebody approves it.",
        path.display(),
        check.slides,
        check.headings.join("; ")
    ))
}

/// Writes the run's calculations into a workbook Excel can recompute.
pub fn create_xlsx(
    resolved_path: Option<&Path>,
    calculations: &Arc<Mutex<HashMap<String, Vec<CalculationRecord>>>>,
    run_id: &str,
    // `None` from callers that only ever want the calculation workbook.
    tool_call: Option<&ToolCall>,
) -> Result<String, String> {
    let path = resolved_path.ok_or_else(|| {
        "No path was resolved for the workbook, so nothing was written.".to_string()
    })?;

    // The general path, when the caller composed a workbook rather than asking
    // for the run's calculations.
    //
    // Absent `sheets`, this is the calculation workbook it has always been:
    // the run's own working, which is the thing a reviewer can check. With
    // `sheets`, it is a workbook of readings, a schedule or a bill of
    // quantities — none of which the calculation workbook can express.
    if let Some(sheets) = tool_call.and_then(|call| call.arguments.get("sheets")).cloned() {
        return create_xlsx_from_sheets(path, sheets, tool_call);
    }

    let records = calculations
        .lock()
        .map_err(|_| "the calculation record is unavailable".to_string())?
        .get(run_id)
        .cloned()
        .unwrap_or_default();

    if records.is_empty() {
        // Said as a refusal rather than an empty file: a workbook with no rows
        // looks like a calculation that produced nothing, which is a different
        // and worse claim than one that was never run.
        return Err(
            "No calculations have been run in this task, so there is nothing to put in a \
             workbook. Use run_calculation first; the workbook shows the working from those \
             calls, not figures written out again."
                .to_string(),
        );
    }

    write_workbook(path, &records, "Internal")?;

    let check = check_workbook(path);
    if !check.problems.is_empty() {
        return Err(format!(
            "{} was written but did not pass its own check: {}.",
            path.display(),
            check.problems.join("; ")
        ));
    }

    Ok(format!(
        "Wrote {} with {} calculation(s), {} of them as live formulas Excel recomputes. \
         If Excel disagrees with a figure, Excel is right and the note needs correcting.",
        path.display(),
        check.calculations,
        check.live_formulas
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::{Role, User};
    use crate::orchestrator::calculation::evaluate;
    use serde_json::json;

    fn call_params(run_id: &str) -> CallParams {
        CallParams {
            run_id: run_id.to_string(),
            tool_call_id: "tc-1".into(),
            tool: "create_docx".into(),
            args: json!({}),
            model: Some("qwen2.5-coder-7b".into()),
        }
    }

    fn author() -> Session {
        Session::open(User::new("priya", "Priya Sharma", vec![Role::Employee]))
    }

    /// Phase E, live: the tools accept a composed document and workbook, not
    /// only the one template and the run's calculation records.
    mod the_general_path {
        use super::*;

        #[test]
        fn a_composed_document_is_produced_through_the_tool() {
            let dir = tempfile::tempdir().expect("temp dir");
            let path = dir.path().join("procedure.docx");
            let tool_call = ToolCall::new(
                "create_docx",
                json!({
                    "path": "procedure.docx",
                    "title": "Isolating the cooling water line",
                    "classification": "OFFICIAL",
                    "sections": [
                        {
                            "heading": "Before you start",
                            "level": 1,
                            "blocks": [
                                {"kind": "paragraph",
                                 "text": "Confirm the permit is signed and the line is drained."},
                                {"kind": "numbered",
                                 "items": ["Close PV-2201.", "Lock and tag the valve.",
                                           "Verify zero pressure at the gauge."]}
                            ]
                        },
                        {
                            "heading": "Checks",
                            "level": 2,
                            "blocks": [
                                {"kind": "table",
                                 "header": ["Step", "Checked by"],
                                 "rows": [["Valve closed", "-"], ["Tag fitted", "-"]],
                                 "caption": "To be completed on the day"}
                            ]
                        }
                    ]
                }),
            );

            let message = create_docx(&call_params("run-e"), Some(&path), &author(), &tool_call)
                .expect("a composed document is produced");

            assert!(message.contains("2 section(s)"), "{message}");
            assert!(path.exists(), "the file is at the path the caller asked for");
            let body = crate::artifacts::ooxml::read_part(&path, "word/document.xml")
                .expect("the package opens");
            assert!(body.contains("<w:tbl>"), "the table must render as a table");
            assert!(body.contains("Lock and tag the valve."), "the procedure must be in the file");
        }

        /// The template path is untouched: a call with `template` and `content`
        /// still produces the approval note it always did.
        #[test]
        fn the_template_path_still_works_alongside_it() {
            let dir = tempfile::tempdir().expect("temp dir");
            let path = dir.path().join("note.docx");
            let tool_call = ToolCall::new(
                "create_docx",
                json!({
                    "path": "note.docx",
                    "template": "approval_note",
                    "content": {
                        "title": "Replacement of control valve PV-2201",
                        "recipient": "Head of Maintenance",
                        "subject": "Valve replacement",
                        "findings": "The seat showed measurable wear at the March outage.",
                        "recommendation": "Approve the replacement.",
                        "references": "Maintenance Report Unit Four, page 1.",
                        "assumptions": "The outage window is unchanged."
                    }
                }),
            );

            create_docx(&call_params("run-e"), Some(&path), &author(), &tool_call)
                .expect("the template still produces its document");
            assert!(path.exists());
        }

        #[test]
        fn a_call_with_neither_says_which_is_missing() {
            let dir = tempfile::tempdir().expect("temp dir");
            let tool_call = ToolCall::new("create_docx", json!({ "path": "x.docx" }));
            let error = create_docx(
                &call_params("run-e"),
                Some(&dir.path().join("x.docx")),
                &author(),
                &tool_call,
            )
            .expect_err("must refuse");
            assert!(error.contains("sections"), "{error}");
            assert!(error.contains("approval_note"), "{error}");
        }

        #[test]
        fn a_composed_workbook_is_produced_with_live_formulas() {
            let dir = tempfile::tempdir().expect("temp dir");
            let path = dir.path().join("readings.xlsx");
            let table: Arc<Mutex<HashMap<String, Vec<CalculationRecord>>>> = Arc::default();
            let tool_call = ToolCall::new(
                "create_xlsx",
                json!({
                    "path": "readings.xlsx",
                    "title": "Shell thickness",
                    "sheets": [{
                        "name": "Readings",
                        "columns": [
                            {"header": "Point", "type": "text"},
                            {"header": "Measured", "type": "number"},
                            {"header": "Margin", "type": "formula"}
                        ],
                        "rows": [["S-01", "9.4", "=B2-9"], ["S-02", "8.7", "=B3-9"]],
                        "freezeHeader": true
                    }]
                }),
            );

            let message = create_xlsx(Some(&path), &table, "run-e", Some(&tool_call))
                .expect("a composed workbook is produced");
            assert!(message.contains("2 row(s)"), "{message}");

            let sheet = crate::artifacts::ooxml::read_part(&path, "xl/worksheets/sheet1.xml")
                .expect("the sheet opens");
            assert!(sheet.contains("<v>9.4</v>"), "a measurement must be a numeric cell");
            assert!(sheet.contains("<f>B2-9</f>"), "the formula must be live");
        }

        /// Prose in a column of numbers is refused before anything is written.
        #[test]
        fn a_composed_workbook_whose_types_are_wrong_is_refused() {
            let dir = tempfile::tempdir().expect("temp dir");
            let path = dir.path().join("bad.xlsx");
            let table: Arc<Mutex<HashMap<String, Vec<CalculationRecord>>>> = Arc::default();
            let tool_call = ToolCall::new(
                "create_xlsx",
                json!({
                    "path": "bad.xlsx",
                    "sheets": [{
                        "name": "Readings",
                        "columns": [{"header": "Measured", "type": "number"}],
                        "rows": [["about nine millimetres"]]
                    }]
                }),
            );

            let error = create_xlsx(Some(&path), &table, "run-e", Some(&tool_call))
                .expect_err("must refuse");
            assert!(error.contains("declared a number"), "{error}");
            assert!(!path.exists(), "nothing may be written");
        }

        /// Without `sheets` it is still the calculation workbook.
        #[test]
        fn the_calculation_workbook_still_works_alongside_it() {
            let dir = tempfile::tempdir().expect("temp dir");
            let path = dir.path().join("working.xlsx");
            let table: Arc<Mutex<HashMap<String, Vec<CalculationRecord>>>> = Arc::default();
            table
                .lock()
                .unwrap()
                .insert("run-e".to_string(), vec![evaluate("2 m * 3 m").expect("calculates")]);
            let tool_call = ToolCall::new("create_xlsx", json!({ "path": "working.xlsx" }));

            create_xlsx(Some(&path), &table, "run-e", Some(&tool_call))
                .expect("the calculation workbook still writes");
            assert!(path.exists());
        }
    }

    /// Phase D: the live document path goes through the correction loop.
    ///
    /// These are about the *production* call site, not `produce` itself, which
    /// has its own tests. What is being proved is that the live path reaches
    /// it at all — it did not, for the entire life of the module.
    mod through_the_repair_loop {
        use super::*;

        fn complete() -> serde_json::Value {
            json!({
                "path": "note.docx",
                "template": "approval_note",
                "content": {
                    "title": "Replacement of control valve PV-2201",
                    "recipient": "Head of Maintenance, Unit Four",
                    "subject": "Valve replacement under the supply agreement",
                    "findings": "The valve was inspected during the March outage and the seat \
                                 showed measurable wear.",
                    "recommendation": "Approve the replacement under the existing agreement.",
                    "references": "Maintenance Report Unit Four, page 1.",
                    "assumptions": "The March outage window remains as scheduled."
                }
            })
        }

        fn write(dir: &std::path::Path, args: serde_json::Value) -> Result<String, String> {
            let path = dir.join("note.docx");
            let tool_call = ToolCall {
                tool: "create_docx".into(),
                arguments: args.clone(),
            };
            create_docx_with_evidence(
                &call_params("run-d"),
                Some(&path),
                &author(),
                &tool_call,
                &[],
                &[],
            )
        }

        #[test]
        fn a_complete_document_is_written_as_a_revision() {
            let dir = tempfile::tempdir().expect("temp dir");
            let message = write(dir.path(), complete()).expect("produces");

            // `produce` numbers every attempt. The bare name is never written,
            // which is what stops a retry overwriting the evidence of the
            // attempt before it.
            assert!(
                message.contains("note.r1.docx"),
                "the file that stands must be a numbered revision: {message}"
            );
            assert!(dir.path().join("note.r1.docx").is_file(), "the revision is on disk");
        }

        /// The transcription error the inner loop exists for. The model names
        /// the box differently; nothing is missing.
        #[test]
        fn a_field_under_the_wrong_name_is_repaired_without_asking_the_model_again() {
            let dir = tempfile::tempdir().expect("temp dir");
            let mut args = complete();
            let content = args["content"].as_object_mut().expect("content");
            let value = content.remove("recommendation").expect("present");
            content.insert("Recommendations".to_string(), value);

            let message = write(dir.path(), args).expect("the repair must recover this");
            assert!(
                message.contains("corrected and kept beside it"),
                "the message must say a correction happened: {message}"
            );
            // Attempt 1 failed and is kept; attempt 2 stands. Both on disk.
            assert!(dir.path().join("note.r2.docx").is_file(), "the corrected revision");
        }

        /// The line the loop must never cross.
        #[test]
        fn content_nobody_supplied_is_never_invented() {
            let dir = tempfile::tempdir().expect("temp dir");
            let mut args = complete();
            args["content"]
                .as_object_mut()
                .expect("content")
                .remove("recommendation");

            let error = write(dir.path(), args).expect_err("must not produce a document");
            assert!(
                error.to_lowercase().contains("recommendation"),
                "the failure must name the field that is missing: {error}"
            );
        }

        /// Bounded. A model that has been told the same thing three times is
        /// missing the information, not the instruction.
        #[test]
        fn attempts_are_bounded_and_the_failure_is_honest() {
            let dir = tempfile::tempdir().expect("temp dir");
            let args = json!({
                "path": "note.docx",
                "template": "approval_note",
                "content": { "findings": "Something was found." }
            });
            let error = write(dir.path(), args).expect_err("must fail");

            // Never a success message, and never a path to a file that is not
            // sound. That is the whole point of honest failure.
            assert!(!error.contains("Wrote "), "{error}");
            assert!(
                error.contains("attempt")
                    || error.to_lowercase().contains("required")
                    || error.to_lowercase().contains("missing"),
                "the failure must say what went wrong: {error}"
            );
        }

        #[test]
        fn an_unknown_template_is_refused_before_anything_is_written() {
            let dir = tempfile::tempdir().expect("temp dir");
            let mut args = complete();
            args["template"] = json!("maintenance_procedure");
            let error = write(dir.path(), args).expect_err("must fail");
            assert!(error.contains("maintenance_procedure"), "{error}");
            assert!(
                std::fs::read_dir(dir.path()).expect("read").next().is_none(),
                "nothing may be written for a template that does not exist"
            );
        }
    }

    fn deck_content() -> serde_json::Value {
        json!({
            "path": "briefing.pptx",
            "content": {
                "title": "Q3 seal inspection",
                "findings": ["Measured 8.2 mm against a 9.0 mm minimum [E1]", "No leak observed"],
                "recommendation": ["Replace the seal at the next shutdown"],
                "assumptions": "None.",
                "evidence": ["Maintenance SOP rev C, page 4"]
            }
        })
    }

    #[test]
    fn a_deck_is_written_re_opened_and_says_it_is_a_draft() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("briefing.pptx");
        let tool_call = ToolCall::new("create_pptx", deck_content());

        let message = create_pptx(&call_params("run-deck"), Some(&path), &tool_call)
            .expect("the deck is written");

        assert!(path.exists(), "no file was written");
        // The point of the check: the file re-opens as a presentation, and the
        // headings that come back are the ones the template promises.
        let check = check_deck(&path);
        assert!(check.is_sound(), "deck did not re-open: {:?}", check.problems);
        for heading in BRIEFING_SECTIONS {
            assert!(
                check.headings.iter().any(|h| h.eq_ignore_ascii_case(heading)),
                "{heading} missing from {:?}",
                check.headings
            );
        }
        assert!(message.contains("DRAFT"), "{message}");
    }

    /// The whole path, from a person's words to a file that re-opens.
    ///
    /// Every link in this chain was already built and tested except one, and
    /// that one made the others worthless: no plan permitted `create_pptx`, so
    /// the catalogue never offered the tool and the gateway refused it. The
    /// renderer's own tests passed throughout, because the renderer was never
    /// the problem.
    ///
    /// So this test starts where a user starts — a sentence — and refuses to
    /// stop at "the writer works". It asserts the request reaches the tool, the
    /// tool writes a file, the file re-opens as a presentation, and the plan
    /// then closes on that evidence rather than on the model's say-so.
    #[test]
    fn a_request_for_slides_plans_the_deck_produces_it_and_re_opens_it() {
        use crate::agent_runtime::planning::{self, Satisfies};
        use crate::agent_runtime::tasks::PlanRecord;
        use crate::orchestrator::tools::ToolName;

        let prompt = "put together a briefing deck on the Q3 seal inspection";

        // 1. The request can reach the tool. This is the link that was missing.
        let derived = planning::derive(prompt);
        assert!(
            derived.budget.permits(ToolName::CreatePptx),
            "the plan does not permit create_pptx, so the catalogue would not offer it"
        );
        let deck_step = derived
            .steps
            .iter()
            .position(|step| step.satisfied_by == Satisfies::Tool(ToolName::CreatePptx))
            .expect("the plan does not expect a deck");

        // The catalogue is filtered against this list, so a deck tool absent
        // from it is one the model is never told about.
        let plan = planning::plan_for("run-e2e", prompt);
        let mut record = PlanRecord::of(&plan);
        assert!(
            record
                .permitted_tools
                .iter()
                .any(|tool| tool == ToolName::CreatePptx.as_str()),
            "the deck tool is not in the plan the catalogue is filtered against"
        );

        // 2. The tool runs and writes a real file.
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("briefing.pptx");
        let tool_call = ToolCall::new("create_pptx", deck_content());
        let message = create_pptx(&call_params("run-e2e"), Some(&path), &tool_call)
            .expect("the deck is written");
        assert!(path.exists(), "no file was written");

        // 3. It re-opens as a presentation, carrying the template's headings.
        //    PowerPoint is stricter than Word about what it will open, so a
        //    deck that wrote without error is not yet a deck that opens.
        let check = check_deck(&path);
        assert!(check.is_sound(), "the deck did not re-open: {:?}", check.problems);
        for heading in BRIEFING_SECTIONS {
            assert!(
                check.headings.iter().any(|h| h.eq_ignore_ascii_case(heading)),
                "{heading} missing from {:?}",
                check.headings
            );
        }
        assert!(message.contains("DRAFT"), "an unsigned deck must say so: {message}");

        // 4. The plan closes on the evidence. `settle` matches the namespaced
        //    name the runtime records, which is why the step and the recorded
        //    call have to agree on spelling — if they drifted, a run could
        //    produce the deck and still be reported as never having made one.
        record.settle(
            &derived.steps,
            &[ToolName::CreatePptx.as_str().to_string()],
            true,
            true,
        );
        assert!(
            record.steps[deck_step].done,
            "the deck was produced and re-opened, but the plan step did not settle"
        );
    }

    #[test]
    fn a_missing_section_is_named_and_nothing_is_written() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("partial.pptx");
        let tool_call = ToolCall::new(
            "create_pptx",
            json!({
                "path": "partial.pptx",
                "content": {
                    "title": "Q3 seal inspection",
                    "findings": ["Worn beyond limit"],
                    "recommendation": ["Replace"],
                    "assumptions": ["None"]
                }
            }),
        );

        let refusal = create_pptx(&call_params("run-deck"), Some(&path), &tool_call)
            .expect_err("a deck without Evidence must be refused");

        assert!(refusal.contains("Evidence"), "{refusal}");
        assert!(refusal.contains("Nothing was written"), "{refusal}");
        assert!(!path.exists(), "a refused deck must not leave a file behind");
    }

    #[test]
    fn an_empty_section_is_refused_rather_than_written_as_a_blank_slide() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("blank.pptx");
        let mut content = deck_content();
        content["content"]["assumptions"] = json!([]);
        let tool_call = ToolCall::new("create_pptx", content);

        let refusal = create_pptx(&call_params("run-deck"), Some(&path), &tool_call)
            .expect_err("an empty section must be refused");

        assert!(refusal.contains("Assumptions"), "{refusal}");
        assert!(!path.exists());
    }

    #[test]
    fn a_bullet_that_is_not_text_is_refused_by_position() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("deck.pptx");
        let mut content = deck_content();
        content["content"]["findings"] = json!(["fine", 9.0]);
        let tool_call = ToolCall::new("create_pptx", content);

        let refusal = create_pptx(&call_params("run-deck"), Some(&path), &tool_call)
            .expect_err("a numeric bullet must be refused");

        // Named by position, so the model can find the one it got wrong.
        assert!(refusal.contains("Bullet 2"), "{refusal}");
        assert!(refusal.contains("Findings"), "{refusal}");
    }

    #[test]
    fn one_string_is_accepted_as_a_single_bullet() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("single.pptx");
        let tool_call = ToolCall::new("create_pptx", deck_content());

        create_pptx(&call_params("run-deck"), Some(&path), &tool_call)
            .expect("assumptions supplied as one string is legitimate");
        assert!(check_deck(&path).is_sound());
    }

    #[test]
    fn a_document_is_written_and_says_it_is_a_draft() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("note.docx");
        let tool_call = ToolCall::new(
            "create_docx",
            json!({
                "path": "note.docx",
                "template": "approval_note",
                "content": {
                    "title": "Pump seal replacement",
                    "recipient": "Maintenance Manager",
                    "subject": "P-101 mechanical seal",
                    "findings": "The seal is worn beyond the 9.0 mm limit.",
                    "recommendation": "Replace the seal at the next shutdown.",
                    "references": "Maintenance SOP p.4.",
                    "assumptions": "None."
                }
            }),
        );

        let message = create_docx(&call_params("run-1"), Some(&path), &author(), &tool_call)
            .expect("the document is written");

        assert!(path.exists());
        assert!(message.contains("DRAFT"), "{message}");
    }

    #[test]
    fn a_field_that_is_not_text_is_refused_by_name() {
        let dir = tempfile::tempdir().expect("temp dir");
        let tool_call = ToolCall::new(
            "create_docx",
            json!({ "template": "approval_note", "content": { "title": 9 } }),
        );

        let error = create_docx(
            &call_params("run-1"),
            Some(&dir.path().join("note.docx")),
            &author(),
            &tool_call,
        )
        .unwrap_err();

        assert!(error.contains("\"title\""), "{error}");
        assert!(error.contains("a number"), "{error}");
    }

    #[test]
    fn content_that_is_not_an_object_says_what_shape_is_wanted() {
        let dir = tempfile::tempdir().expect("temp dir");
        let tool_call = ToolCall::new(
            "create_docx",
            json!({ "template": "approval_note", "content": "just some text" }),
        );

        let error = create_docx(
            &call_params("run-1"),
            Some(&dir.path().join("note.docx")),
            &author(),
            &tool_call,
        )
        .unwrap_err();

        assert!(error.contains("field name to text"), "{error}");
    }

    #[test]
    fn an_unknown_template_names_the_ones_that_exist() {
        let dir = tempfile::tempdir().expect("temp dir");
        let tool_call = ToolCall::new(
            "create_docx",
            json!({ "template": "invoice", "content": { "title": "x" } }),
        );

        let error = create_docx(
            &call_params("run-1"),
            Some(&dir.path().join("note.docx")),
            &author(),
            &tool_call,
        )
        .unwrap_err();

        assert!(error.contains("approval_note"), "{error}");
    }

    #[test]
    fn a_workbook_holds_the_calculations_the_engine_actually_ran() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("working.xlsx");
        let table: Arc<Mutex<HashMap<String, Vec<CalculationRecord>>>> = Arc::default();
        table.lock().unwrap().insert(
            "run-1".into(),
            vec![
                evaluate("2 m * 3 m").expect("evaluates"),
                evaluate("10 kg / 2 s").expect("evaluates"),
            ],
        );

        let message = create_xlsx(Some(&path), &table, "run-1", None).expect("the workbook is written");

        assert!(path.exists());
        assert!(message.contains("2 calculation(s)"), "{message}");
        // The point of the workbook: Excel recomputes and may disagree.
        assert!(message.contains("Excel is right"), "{message}");
    }

    #[test]
    fn a_workbook_with_nothing_to_show_is_refused_rather_than_written_empty() {
        let dir = tempfile::tempdir().expect("temp dir");
        let table: Arc<Mutex<HashMap<String, Vec<CalculationRecord>>>> = Arc::default();

        let error = create_xlsx(Some(&dir.path().join("working.xlsx")), &table, "run-1", None).unwrap_err();

        assert!(error.contains("run_calculation first"), "{error}");
        assert!(!dir.path().join("working.xlsx").exists());
    }

    #[test]
    fn one_runs_calculations_do_not_appear_in_anothers_workbook() {
        let dir = tempfile::tempdir().expect("temp dir");
        let table: Arc<Mutex<HashMap<String, Vec<CalculationRecord>>>> = Arc::default();
        table
            .lock()
            .unwrap()
            .insert("run-1".into(), vec![evaluate("2 m * 3 m").expect("evaluates")]);

        let error = create_xlsx(Some(&dir.path().join("other.xlsx")), &table, "run-2", None).unwrap_err();
        assert!(error.contains("No calculations have been run in this task"), "{error}");
    }
}
