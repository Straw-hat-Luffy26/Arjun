//! The Document & Vision Analyst's tools, answered on the agent path (P06).
//!
//! | Tool | What it returns |
//! |---|---|
//! | `document.layout_map` | page sizes, text-layer blocks and embedded tables as evidence regions |
//! | `document.render_regions` | preserved crop images of page regions, by id and hash |
//! | `document.ocr_regions` | local Unlimited-OCR transcription of bounded pages or crops |
//! | `document.extract_tables` | tables from the text layer or from OCR, cell by cell |
//! | `media.extract_findings` | requested fields, each with the region it was read in |
//!
//! Each authorises the document through [`crate::extraction::service`] — owner
//! and conversation, bytes still hashing to the id — and answers with region
//! ids, boxes in a named coordinate space, and the method that read them. A
//! transcription and an interpretation are never presented the same way: the
//! second is always headed as a proposal.

use std::fmt::Write as _;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::agent_runtime::cancellation::CancelToken;
use crate::ai_engine::ocr_profile::OcrDetent;
use crate::extraction::fields::FieldState;
use crate::extraction::regions::{BBox, EvidenceRegion, Method, RegionStatus};
use crate::extraction::service::{
    AuthorisedDocument, FindingsReport, LayoutReport, OcrOutcome, PageCoverage, TablesReport,
    MIN_TEXT_LAYER_CHARS,
};
use crate::identity::Session;
use crate::orchestrator::executor::ToolRunner;
use crate::orchestrator::runner::LocalToolRunner;
use crate::orchestrator::tools::{ToolCall, ToolName};

use super::{CallParams, RuntimeDeps};

/// The detent the analyst reads at: the accuracy tier (Q6_K) with the decode
/// cap of the detailed stop, which is the inventoried configuration's default.
pub const ANALYST_DETENT: OcrDetent = OcrDetent::Detailed;

/// Kept under the contract's 120-second ceiling, with room to render the
/// answer after the last unit.
const CALL_BUDGET: Duration = Duration::from_secs(110);

/// Regions listed per page before the listing says how many more there are.
const REGIONS_PER_PAGE: usize = 40;
const EXCERPT_CHARS: usize = 160;

fn excerpt(text: &str) -> String {
    let flat = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if flat.chars().count() > EXCERPT_CHARS {
        format!("{}…", flat.chars().take(EXCERPT_CHARS).collect::<String>())
    } else {
        flat
    }
}

fn document_arg(tool_call: &ToolCall) -> Result<String, String> {
    let sha = tool_call.text("documentSha256").unwrap_or_default().trim().to_string();
    if sha.is_empty() {
        return Err(
            "documentSha256 is required. It is the id on the <attachment> tag of the document."
                .to_string(),
        );
    }
    Ok(sha)
}

fn authorise(
    deps: &Arc<RuntimeDeps>,
    call: &CallParams,
    session: &Session,
    tool_call: &ToolCall,
) -> Result<AuthorisedDocument, String> {
    let sha = document_arg(tool_call)?;
    let conversation = conversation_of(deps, call)?;
    deps.extraction
        .authorise(&deps.documents, &sha, &session.user.id, &conversation)
}

fn conversation_of(deps: &Arc<RuntimeDeps>, call: &CallParams) -> Result<String, String> {
    deps.run_to_conversation.lookup(&call.run_id).ok_or_else(|| {
        "This run is not attached to a conversation, so it cannot read a conversation's documents."
            .to_string()
    })
}

/// A list argument's strings, whatever the model wrapped them in.
fn strings(tool_call: &ToolCall, key: &str) -> Vec<String> {
    match tool_call.arguments.get(key) {
        Some(serde_json::Value::Array(items)) => items
            .iter()
            .filter_map(|item| match item {
                serde_json::Value::String(s) => Some(s.trim().to_string()),
                serde_json::Value::Number(n) => Some(n.to_string()),
                _ => None,
            })
            .filter(|s| !s.is_empty())
            .collect(),
        Some(serde_json::Value::String(s)) => s
            .split(['\n', ';'])
            .map(|part| part.trim().to_string())
            .filter(|part| !part.is_empty())
            .collect(),
        _ => Vec::new(),
    }
}

/// What `regions` names: region ids (`rg-…`) and boxes, each box on `page`.
///
/// A box is `"x0,y0,x1,y1"`, `[x0,y0,x1,y1]` or `{x0,y0,x1,y1}`. A region id is
/// resolved against the document's region store, so it carries its own page.
fn region_targets(
    deps: &Arc<RuntimeDeps>,
    doc: &AuthorisedDocument,
    tool_call: &ToolCall,
    page: Option<u32>,
) -> Result<Vec<(u32, Option<BBox>)>, String> {
    let Some(value) = tool_call.arguments.get("regions") else {
        return Ok(Vec::new());
    };
    let items = value
        .as_array()
        .ok_or_else(|| "`regions` is a list of region ids or boxes".to_string())?;
    let mut ids = Vec::new();
    let mut targets = Vec::new();
    for item in items {
        if let Some(id) = item.as_str().map(str::trim).filter(|s| s.starts_with("rg-")) {
            ids.push(id.to_string());
            continue;
        }
        let numbers: Vec<f64> = match item {
            serde_json::Value::String(s) => s
                .split(',')
                .map(|part| part.trim().parse::<f64>())
                .collect::<Result<_, _>>()
                .map_err(|_| format!("{s:?} is neither a region id nor a box x0,y0,x1,y1"))?,
            serde_json::Value::Array(values) => values.iter().filter_map(|v| v.as_f64()).collect(),
            serde_json::Value::Object(map) => ["x0", "y0", "x1", "y1"]
                .iter()
                .filter_map(|k| map.get(*k).and_then(|v| v.as_f64()))
                .collect(),
            _ => Vec::new(),
        };
        let bbox = BBox::from_slice(&numbers).ok_or_else(|| {
            format!("{item} is not a box: it needs four numbers x0,y0,x1,y1 with x1 > x0 and y1 > y0")
        })?;
        let page = page.ok_or("a box in `regions` is on a `page`; name the page".to_string())?;
        targets.push((page, Some(bbox)));
    }
    targets.extend(deps.extraction.region_targets(doc, &ids)?);
    Ok(targets)
}

/// A token that trips when the run's durable record says it ended.
///
/// The agent path has no per-run token of its own: a stop lands in the event
/// log. Watching that is what lets a Stop pressed during a page close the
/// socket instead of waiting minutes for the page to finish.
fn run_stop(deps: &Arc<RuntimeDeps>, run_id: &str) -> (CancelToken, tokio::task::JoinHandle<()>) {
    let token = CancelToken::never();
    let (watched, events, run) = (token.clone(), Arc::clone(&deps.events), run_id.to_string());
    let watcher = tokio::spawn(async move {
        loop {
            if events.ending(&run).is_some() {
                watched.cancel();
                return;
            }
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
    });
    (token, watcher)
}

// -- document.layout_map ----------------------------------------------------

pub(super) fn layout_map(
    deps: &Arc<RuntimeDeps>,
    call: &CallParams,
    session: &Session,
    tool_call: &ToolCall,
) -> Result<String, String> {
    let doc = authorise(deps, call, session, tool_call)?;
    let from = tool_call.integer("fromPage").unwrap_or(1);
    let to = tool_call.integer("toPage").unwrap_or(from);
    let report = deps.extraction.layout_map(&doc, from, to)?;
    Ok(render_layout(&report, doc.is_image))
}

pub fn render_layout(report: &LayoutReport, is_image: bool) -> String {
    let mut out = String::new();
    let space = report
        .page_records
        .first()
        .map(|p| p.coord_space.label())
        .unwrap_or("page");
    let _ = writeln!(
        out,
        "{} (document {}…), pages {}-{} of {}. Boxes are x0,y0,x1,y1 in {space}, origin top-left. \
         Every region below is transcription evidence from the file's own text layer unless it \
         says otherwise; cite it by its rg- id.",
        report.name,
        &report.document_sha256[..12],
        report.from_page,
        report.to_page,
        report.pages
    );
    for record in &report.page_records {
        let route = if is_image || !record.text_layer_adequate(MIN_TEXT_LAYER_CHARS) {
            "no adequate text layer — a scan or picture: read it with document.ocr_regions"
                .to_string()
        } else {
            "the text layer is adequate; no OCR is needed".to_string()
        };
        let _ = writeln!(
            out,
            "\nPage {} — {:.1} x {:.1} {}, rotation {}, text layer {} characters: {route}.{}",
            record.page,
            record.width,
            record.height,
            record.coord_space.label(),
            record.rotation,
            record.text_layer_chars,
            record
                .skew_degrees
                .map(|d| format!(" Measured skew {d:.2} degrees."))
                .unwrap_or_default()
        );
        let on_page: Vec<&EvidenceRegion> =
            report.regions.iter().filter(|r| r.page == record.page).collect();
        for region in on_page.iter().take(REGIONS_PER_PAGE) {
            out.push_str(&region_line(region));
        }
        if on_page.len() > REGIONS_PER_PAGE {
            let _ = writeln!(
                out,
                "  … {} more region(s) on this page not listed. Ask for a narrower range or use \
                 document.render_regions on the area you need.",
                on_page.len() - REGIONS_PER_PAGE
            );
        }
        if on_page.is_empty() {
            out.push_str("  No blocks on this page.\n");
        }
    }
    out.push_str(&render_coverage(&report.coverage));
    out
}

fn region_line(region: &EvidenceRegion) -> String {
    let mut line = format!(
        "  {} {} {} [{}]",
        region.region_id,
        region.label,
        region.bbox.describe(),
        region.method.label()
    );
    match region.status {
        RegionStatus::Read if !region.cells.is_empty() => {
            let rows = region.cells.iter().map(|c| c.row + 1).max().unwrap_or(0);
            let cols = region.cells.iter().map(|c| c.col + 1).max().unwrap_or(0);
            let _ = write!(line, " table {rows}x{cols}: {}", excerpt(&region.text));
        }
        RegionStatus::Read => {
            let _ = write!(line, " \"{}\"", excerpt(&region.text));
        }
        status => {
            let _ = write!(
                line,
                " {}: {}",
                status.label().to_uppercase(),
                region.notes.first().cloned().unwrap_or_default()
            );
            if !region.text.is_empty() {
                let _ = write!(line, " (kept before the problem: \"{}\")", excerpt(&region.text));
            }
        }
    }
    line.push('\n');
    line
}

fn render_coverage(coverage: &[PageCoverage]) -> String {
    let mut out = String::from("\nCoverage:\n");
    for page in coverage {
        let methods = page
            .methods
            .iter()
            .map(|m| m.label())
            .collect::<Vec<_>>()
            .join(", ");
        let mut flags = Vec::new();
        if page.unreadable > 0 {
            flags.push(format!("{} unreadable region(s)", page.unreadable));
        }
        if page.malformed > 0 {
            flags.push(format!("{} malformed", page.malformed));
        }
        if page.looped {
            flags.push("OCR looped and was cut".to_string());
        }
        if page.truncated {
            flags.push("OCR stopped at its decode cap".to_string());
        }
        let _ = writeln!(
            out,
            "  page {}: {}{}{}{}",
            page.page,
            match page.route.as_str() {
                "embedded-text" => "text layer",
                "ocr" => "local OCR",
                _ => "NOT READ",
            },
            if methods.is_empty() { String::new() } else { format!(" ({methods})") },
            if flags.is_empty() { String::new() } else { format!("; {}", flags.join("; ")) },
            page.reason
                .as_ref()
                .map(|r| format!(" — {r}"))
                .unwrap_or_default()
        );
    }
    if coverage.iter().any(|p| p.route == "unread") {
        out.push_str(
            "A page marked NOT READ holds content nobody has transcribed. Do not describe it, and \
             do not treat a field missing from it as absent from the document.\n",
        );
    }
    out
}

// -- document.render_regions ------------------------------------------------

pub(super) fn render_regions(
    deps: &Arc<RuntimeDeps>,
    call: &CallParams,
    session: &Session,
    tool_call: &ToolCall,
) -> Result<String, String> {
    let doc = authorise(deps, call, session, tool_call)?;
    let page = tool_call.integer("page").unwrap_or(1);
    let dpi = tool_call.integer("dpi");
    let mut targets = region_targets(deps, &doc, tool_call, Some(page))?;
    if targets.is_empty() {
        targets.push((page, None));
    }
    let crops = deps.extraction.render_regions(&doc, &targets, dpi)?;
    let mut out = format!(
        "Rendered {} crop(s) of {}. Each is preserved under its id and hash; the original \
         document is untouched.\n",
        crops.len(),
        doc.name
    );
    for crop in &crops {
        let _ = writeln!(
            out,
            "  {} page {} {} {} → {}x{} px ({:.3} px per unit{}), sha256 {}",
            crop.crop_id,
            crop.page,
            crop.bbox.describe(),
            crop.coord_space.label(),
            crop.width,
            crop.height,
            crop.pixels_per_unit,
            crop.dpi.map(|d| format!(", {d} dpi")).unwrap_or_else(|| ", stored pixels".into()),
            &crop.image_sha256[..16]
        );
    }
    out.push_str(
        "A crop is pixels, not text. Read it with document.ocr_regions, or ask about it with \
         media.extract_findings and a question.\n",
    );
    Ok(out)
}

// -- document.ocr_regions ---------------------------------------------------

pub(super) async fn ocr_regions(
    deps: &Arc<RuntimeDeps>,
    call: &CallParams,
    session: &Session,
    tool_call: &ToolCall,
) -> Result<String, String> {
    let deadline = Instant::now() + CALL_BUDGET;
    let doc = {
        let (deps, call, session, tool_call) = (deps.clone(), call.clone(), session.clone(), tool_call.clone());
        tokio::task::spawn_blocking(move || authorise(&deps, &call, &session, &tool_call))
            .await
            .map_err(|error| format!("authorising the document did not finish: {error}"))??
    };
    let page = tool_call.integer("page");
    let mut targets: Vec<(u32, Option<BBox>)> = Vec::new();
    for raw in strings(tool_call, "pages") {
        let number: u32 = raw
            .parse()
            .map_err(|_| format!("{raw:?} in `pages` is not a page number"))?;
        targets.push((number, None));
    }
    targets.extend(region_targets(deps, &doc, tool_call, page)?);
    if targets.is_empty() {
        targets.push((page.unwrap_or(1), None));
    }
    let (stop, watcher) = run_stop(deps, &call.run_id);
    let outcome = deps
        .extraction
        .ocr(&doc, targets, None, ANALYST_DETENT, None, &stop, deadline)
        .await;
    watcher.abort();
    Ok(render_ocr(&outcome?, &doc))
}

pub fn render_ocr(outcome: &OcrOutcome, doc: &AuthorisedDocument) -> String {
    let batch = &outcome.batch;
    let mut out = String::new();
    match &batch.identity {
        Some(identity) => {
            let _ = writeln!(
                out,
                "Local OCR of {}: {} ({} detent, weights {}, projector {}), via {}{}. This is \
                 transcription of pixels, not interpretation; cite each region by its rg- id.",
                doc.name,
                identity.model_id,
                identity.detent,
                identity
                    .weights_sha256
                    .as_deref()
                    .map(|s| format!("sha256 {}…", &s[..12.min(s.len())]))
                    .unwrap_or_else(|| "not hash-pinned".into()),
                identity.projector_file.as_deref().unwrap_or("none"),
                batch.transport,
                batch
                    .residency
                    .as_ref()
                    .map(|r| format!("; {r}"))
                    .unwrap_or_default()
            );
        }
        None => {
            let _ = writeln!(out, "Local OCR of {}: no OCR model is usable here.", doc.name);
        }
    }
    for unit in &batch.read {
        let _ = writeln!(
            out,
            "\nPage {} crop {} — {}, {} region(s){}",
            unit.page,
            unit.crop_id,
            if unit.cached {
                "from the cache (same bytes, crop, model and settings)".to_string()
            } else {
                format!("read in {:.1}s, {} tokens", unit.elapsed_ms as f64 / 1000.0, unit.tokens)
            },
            unit.regions.len(),
            if unit.looped || unit.truncated { " — INCOMPLETE" } else { "" }
        );
        for region in unit.regions.iter().take(REGIONS_PER_PAGE) {
            out.push_str(&region_line(region));
        }
        if unit.regions.len() > REGIONS_PER_PAGE {
            let _ = writeln!(out, "  … {} more region(s) not listed.", unit.regions.len() - REGIONS_PER_PAGE);
        }
        for note in &unit.notes {
            let _ = writeln!(out, "  Note: {note}");
        }
    }
    for (page, reason) in &outcome.skipped {
        let _ = writeln!(out, "\nNot sent to OCR: {reason} (page {page})");
    }
    if !batch.unread.is_empty() {
        out.push_str("\nNot read:\n");
        for unread in &batch.unread {
            let _ = writeln!(out, "  page {}: {}", unread.page, unread.reason);
        }
        out.push_str(
            "Anything on a page or region not read is unknown, not absent. Do not describe it.\n",
        );
    }
    if !batch.check.clean() {
        let _ = writeln!(
            out,
            "\nBatch check failed: missing {:?}, duplicated {:?}, out of order {}. Treat this \
             result as incomplete.",
            batch.check.missing, batch.check.duplicated, batch.check.out_of_order
        );
    }
    out
}

// -- document.extract_tables ------------------------------------------------

pub(super) fn extract_tables(
    deps: &Arc<RuntimeDeps>,
    call: &CallParams,
    session: &Session,
    tool_call: &ToolCall,
) -> Result<String, String> {
    let doc = authorise(deps, call, session, tool_call)?;
    let from = tool_call.integer("fromPage").unwrap_or(1);
    let to = tool_call.integer("toPage").unwrap_or(from);
    Ok(render_tables(&deps.extraction.tables(&doc, from, to)?))
}

/// Rows shown per table before the rest is summarised.
const ROWS_SHOWN: u32 = 40;

pub fn render_tables(report: &TablesReport) -> String {
    let mut out = format!(
        "Tables on pages {}-{} of {}: {} found.\n",
        report.from_page,
        report.to_page,
        report.name,
        report.tables.len()
    );
    for table in &report.tables {
        let _ = writeln!(
            out,
            "\n{} page {} {} {} — {} x {} cells, {}, {}",
            table.region_id,
            table.page,
            table.bbox.describe(),
            table.coord_space.label(),
            table.rows,
            table.cols,
            table.method.label(),
            table.status.label()
        );
        if table.cells.is_empty() {
            out.push_str("  No cell boundaries were delimited, so no cell is reported.\n");
        }
        for row in 0..table.rows.min(ROWS_SHOWN) {
            let mut cells: Vec<_> = table.cells.iter().filter(|c| c.row == row).collect();
            cells.sort_by_key(|c| c.col);
            let line = cells.iter().map(|c| c.text.as_str()).collect::<Vec<_>>().join(" | ");
            let _ = writeln!(out, "  r{row}: {line}");
        }
        if table.rows > ROWS_SHOWN {
            let _ = writeln!(out, "  … {} more row(s) not shown.", table.rows - ROWS_SHOWN);
        }
        if table.method == Method::EmbeddedTable {
            out.push_str("  Cell boxes are recorded for every cell of an embedded table.\n");
        } else {
            out.push_str("  OCR locates the table, not its cells: cite the table's box.\n");
        }
        for note in &table.notes {
            let _ = writeln!(out, "  Note: {note}");
        }
    }
    if !report.not_read.is_empty() {
        let _ = writeln!(
            out,
            "\nScanned page(s) {} have not been OCR-read, so any tables on them are unknown. \
             Read them with document.ocr_regions first.",
            report.not_read.iter().map(u32::to_string).collect::<Vec<_>>().join(", ")
        );
    }
    out
}

// -- media.extract_findings -------------------------------------------------

pub(super) async fn extract_findings(
    deps: &Arc<RuntimeDeps>,
    call: &CallParams,
    session: &Session,
    tool_call: &ToolCall,
) -> Result<String, String> {
    let deadline = Instant::now() + CALL_BUDGET;
    let sha = document_arg(tool_call)?;
    // An attached document is read here. Anything else is a knowledge-base
    // document, answered by the shelf reader it always was.
    let attached = match conversation_of(deps, call) {
        Ok(conversation) => deps
            .documents
            .get(&sha, &session.user.id, Some(&conversation))
            .ok()
            .flatten()
            .is_some(),
        Err(_) => false,
    };
    if !attached {
        return LocalToolRunner::new(deps.index.as_ref(), session)
            .run(ToolName::MediaExtractFindings, tool_call, None)
            .await;
    }
    let doc = {
        let (deps, call, session, tool_call) = (deps.clone(), call.clone(), session.clone(), tool_call.clone());
        tokio::task::spawn_blocking(move || authorise(&deps, &call, &session, &tool_call))
            .await
            .map_err(|error| format!("authorising the document did not finish: {error}"))??
    };
    let from = tool_call.integer("fromPage").unwrap_or(1);
    let to = tool_call.integer("toPage").unwrap_or(from);
    let wanted = strings(tool_call, "fields");
    let question = tool_call.text("question").map(str::to_string);
    let regions = strings(tool_call, "regionIds");
    let (stop, watcher) = run_stop(deps, &call.run_id);
    let report = deps
        .extraction
        .findings(
            &doc,
            from,
            to,
            &wanted,
            question.as_deref(),
            &regions,
            ANALYST_DETENT,
            None,
            &stop,
            deadline,
        )
        .await;
    watcher.abort();
    Ok(render_findings(&report?))
}

pub fn render_findings(report: &FindingsReport) -> String {
    let mut out = format!(
        "Findings for pages {}-{} of {} ({} page(s) in all). A value below is a checked \
         observation: its text occurs in the cited region, which was transcribed by the method \
         named. No confidence number is given because none was measured.\n",
        report.from_page, report.to_page, report.name, report.pages
    );
    if report.fields.is_empty() {
        out.push_str("\nNo fields were asked for.\n");
    }
    for field in &report.fields {
        let state = match field.state {
            FieldState::Found => "FOUND",
            FieldState::Conflicting => "CONFLICTING VALUES — report all, choose none",
            FieldState::Uncertain => "UNCERTAIN — only in a read that was cut or malformed",
            FieldState::NotFound => "NOT FOUND on the pages that were read",
        };
        let _ = writeln!(out, "\nField \"{}\": {state}", field.field);
        for value in &field.values {
            let _ = writeln!(
                out,
                "  \"{}\" — {} p.{} {} {} ({}; {}; matched as {})",
                value.value,
                value.region_id,
                value.page,
                value.bbox.describe(),
                value.coord_space.label(),
                value.method.label(),
                value.status.label(),
                value.matched_as
            );
        }
    }
    if !report.unreadable.is_empty() {
        out.push_str("\nRegions that could not be quoted:\n");
        for region in report.unreadable.iter().take(REGIONS_PER_PAGE) {
            out.push_str(&region_line(region));
        }
        out.push_str(
            "A label in an unreadable region is unknown. Say it could not be read; never supply \
             a likely value.\n",
        );
    }
    if let Some(ocr) = &report.ocr {
        let read: Vec<String> = ocr.batch.read.iter().map(|u| u.page.to_string()).collect();
        if !read.is_empty() {
            let _ = writeln!(
                out,
                "\nScanned page(s) {} were read by local OCR for this call ({}).",
                read.join(", "),
                ocr.batch
                    .identity
                    .as_ref()
                    .map(|i| format!("{} {}", i.model_id, i.detent))
                    .unwrap_or_default()
            );
        }
    }
    out.push_str(&render_coverage(&report.coverage));
    if let Some(interpretation) = &report.interpretation {
        match (&interpretation.model_id, interpretation.proposals.is_empty()) {
            (Some(model), false) => {
                let _ = writeln!(
                    out,
                    "\nPROPOSAL — vision-model inference by {model} ({}). Not a transcription and \
                     not verified against the page; present it as a possibility and do not merge \
                     it with the findings above:",
                    interpretation.transport
                );
                for proposal in &interpretation.proposals {
                    let _ = writeln!(
                        out,
                        "  {} p.{} {} (crop {}): {}",
                        proposal.region_id,
                        proposal.page,
                        proposal.bbox.describe(),
                        proposal.crop_id.as_deref().unwrap_or("-"),
                        excerpt(&proposal.text)
                    );
                }
            }
            _ => {}
        }
        if let Some(why) = &interpretation.refused {
            let _ = writeln!(
                out,
                "\nInterpretation was not {}: {why}",
                if interpretation.proposals.is_empty() { "attempted" } else { "completed" }
            );
        }
    }
    let observed: Vec<String> = report
        .observations
        .iter()
        .map(|(method, count)| format!("{count} by {method}"))
        .collect();
    if !observed.is_empty() {
        let _ = writeln!(out, "\nRegions read cleanly on these pages: {}.", observed.join(", "));
    }
    out
}

// -- document.read_pages / document.search, completed ----------------------

/// How each returned page was read, and what the store says about coverage.
///
/// Appended to `document.read_pages`. A page's method comes from the region
/// store's layout when it has one — a text layer of at least
/// [`MIN_TEXT_LAYER_CHARS`] means the words are the file's own — and otherwise
/// from what the attachment read recorded, which for a mixed PDF is one detent
/// for the whole file and is said to be so.
pub(super) fn page_methods(
    deps: &Arc<RuntimeDeps>,
    read: &super::documents::PageRead,
    owner: &str,
    conversation: &str,
) -> String {
    let Ok(Some(stored)) = deps.documents.get(&read.sha256, owner, Some(conversation)) else {
        return String::new();
    };
    let regions = deps.extraction.regions.load(&read.sha256).ok();
    let detent_label = |quality: u8| {
        OcrDetent::ALL
            .iter()
            .find(|d| d.quality() == quality)
            .map(|d| d.label())
            .unwrap_or("an unrecorded")
    };
    let mut out = String::from("\nHow each page was read:\n");
    for page in read.from_page..=read.to_page {
        let layout = regions.as_ref().and_then(|held| held.pages.get(&page));
        let ocr_regions = regions
            .as_ref()
            .map(|held| held.regions.values().any(|r| r.page == page && r.method == Method::Ocr))
            .unwrap_or(false);
        let quality = stored.page_quality.get(&page).copied();
        let has_text = stored.page_text.iter().any(|p| p.page == page);
        let method = match (layout, quality) {
            (Some(record), _) if record.text_layer_adequate(MIN_TEXT_LAYER_CHARS) => {
                "the file's own text layer".to_string()
            }
            (Some(_), Some(q)) if q != u8::MAX && has_text => {
                format!("local OCR at the {} detent when it was attached", detent_label(q))
            }
            (Some(_), _) if ocr_regions => "local OCR (document.ocr_regions)".to_string(),
            (None, Some(u8::MAX)) if has_text => "the file's own text (no model was used)".to_string(),
            (None, Some(q)) if has_text => format!(
                "its text layer or local OCR at the {} detent — the attachment read recorded one \
                 detent for the whole file, not per page; document.layout_map settles which",
                detent_label(q)
            ),
            _ if has_text => "not recorded".to_string(),
            _ => "nothing: no text is stored for this page".to_string(),
        };
        let _ = writeln!(out, "  page {page}: {method}");
    }
    let completeness = &stored.completeness;
    let _ = writeln!(
        out,
        "Coverage: {} of {} page(s) of this document hold text{}.",
        completeness.pages_extracted,
        completeness.pages_total.max(stored.pages),
        if completeness.pages_failed.is_empty() {
            String::new()
        } else {
            format!(
                "; no text from page(s) {}",
                completeness
                    .pages_failed
                    .iter()
                    .map(u32::to_string)
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        }
    );
    out
}

/// Appended to `document.search`: which searched documents have pages with no
/// text, so a miss is not read as absence.
pub(super) fn search_coverage(deps: &Arc<RuntimeDeps>, owner: &str, conversation: &str) -> String {
    let Ok(documents) = deps.documents.for_conversation(owner, conversation) else {
        return String::new();
    };
    let gaps: Vec<String> = documents
        .iter()
        .filter(|d| !d.completeness.pages_failed.is_empty() || d.truncated)
        .map(|d| {
            let mut line = format!("{} (id {}…)", d.name, &d.sha256[..12]);
            if !d.completeness.pages_failed.is_empty() {
                let _ = write!(
                    line,
                    ": no text from page(s) {}",
                    d.completeness
                        .pages_failed
                        .iter()
                        .map(u32::to_string)
                        .collect::<Vec<_>>()
                        .join(", ")
                );
            }
            if d.truncated {
                line.push_str(": cut short when it was first read");
            }
            line
        })
        .collect();
    if gaps.is_empty() {
        return String::new();
    }
    format!(
        "\nSearch cannot find what was never transcribed: {}. A passage missing from these results \
         may be on those pages; read them with document.ocr_regions before saying it is absent.\n",
        gaps.join("; ")
    )
}
