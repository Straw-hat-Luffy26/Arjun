//! The tools, actually doing things.
//!
//! Everything here runs only after [`super::gateway`] has permitted it, so this
//! module contains no permission checks — a second, weaker copy of a rule
//! enforced properly elsewhere is how the two drift apart and the weaker one
//! becomes the real policy.
//!
//! What it does contain is the checks the gateway *cannot* make, because they
//! depend on what is on disk at the moment of the call rather than on what the
//! model asked for: whether a file exists, whether it is text, whether it is
//! bigger than it claimed to be.
//!
//! ## Output is written for the model to read
//!
//! Every tool returns a string that goes straight back into the conversation as
//! that call's result. So results say what happened in words the model can act
//! on — "no passages matched" rather than an empty list, and "the file is
//! 40 MB, above the limit" rather than a truncated read. A result the model
//! cannot interpret costs a step and teaches it nothing.

use std::path::Path;

use super::calculation;
use super::executor::ToolRunner;
use super::sandbox::{assess, SandboxPolicy, SandboxTier};
use super::sandbox_exec;
use super::tools::{spec_for, ToolCall, ToolName};
use crate::identity::Session;
use crate::knowledge::{
    ImageRegion, KnowledgeIndex, MultimodalIndex, SearchResult, TableChunk,
};
use crate::subagents::{InheritedPolicy, SubagentManager};

/// How many passages a search returns to the model.
///
/// Enough to answer a question, few enough to leave room for the task itself.
/// A model handed forty passages spends its attention on them rather than on
/// what it was asked to do.
const SEARCH_LIMIT: usize = 6;

/// Most pages one `load_more_evidence` call may span.
///
/// Not a performance guard. A model that asks for a hundred pages has stopped
/// asking for a region and started asking for the document, and serving that is
/// how a run's context overflows and the inference server refuses the prompt.
const REGION_PAGE_LIMIT: u32 = 10;

/// Most passages one region read returns, however many pages it spans.
///
/// A dense page can hold a dozen chunks. This is the ceiling that actually
/// bounds what reaches the window; `REGION_PAGE_LIMIT` bounds what is asked for.
const REGION_CHUNK_LIMIT: usize = 24;

/// How many outbound attempts `sovereignty.get_evidence` lists in one answer.
///
/// The count is always exact; only the listing is capped. A machine that
/// refused four hundred attempts should say four hundred and show the recent
/// ones, not shrink the number to what fits.
const SOVEREIGNTY_EVENT_LIMIT: usize = 20;

/// The first eight characters of a digest, for naming a document in prose.
fn short_sha(sha256: &str) -> String {
    sha256.chars().take(8).collect()
}

/// A file's name, without the directory it happens to live in.
///
/// The model wrote a relative name and can only use a relative name; echoing
/// the resolved path back would put the operator's home directory and the run's
/// internal id into a sentence the model may repeat into a document.
fn name_of(path: &Path) -> String {
    path.file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| "that file".to_string())
}

/// Most lines returned by one windowed read.
///
/// Chosen against the same pressure as `READ_CHARS`: enough to hold a section
/// of a draft, few enough that a window cannot fill the context on its own.
const READ_LINES: usize = 400;

/// Longest file content handed back in one read.
///
/// Below the gateway's own ceiling on purpose: the gateway stops a file from
/// exhausting memory, this stops one from exhausting the context window and
/// pushing the task's own instructions out of it.
const READ_CHARS: usize = 24_000;

/// Writes retrieved passages the way the model is asked to cite them.
///
/// Each passage carries the marker it is to be cited by rather than its
/// position in this result, so a run that searches several times numbers its
/// evidence once across the whole task. That matters more than it looks:
/// [`crate::artifacts::verifier`] resolves each `[En]` in the draft against the
/// run's accumulated passages, and per-call numbering would make `[E1]` mean a
/// different passage depending on when in the run it was written.
pub fn render_passages(query: &str, marked: &[(usize, &SearchResult)]) -> String {
    if marked.is_empty() {
        // Said explicitly. An empty result the model has to infer from silence
        // is how a summary ends up citing something that was never found — PS
        // Part C asks for exactly this behaviour.
        return format!(
            "No passages matched {query:?}. Nothing in the connected collections says this, \
             so do not assert it. Try different wording, or state that no source was \
             found. If the question was not about this organisation's documents \
             in the first place, this search was the wrong move: answer it directly \
             instead of reporting a missing source."
        );
    }

    let mut out = format!("{} passage(s) found.\n\n", marked.len());
    for (marker, hit) in marked {
        out.push_str(&format!("[E{marker}] {}\n{}\n\n", hit.citation(), hit.text));
    }
    out.push_str(
        "Cite these by their marker — write [E1] after a claim that came from that passage.\n",
    );
    out
}

pub struct LocalToolRunner<'a> {
    pub index: &'a KnowledgeIndex,
    /// Multimodal index, for image regions and table rows. Optional because
    /// the runner is constructed during early boot when the multimodal
    /// database may not yet be open. A `None` here turns the multimodal
    /// tool into a refusal that names the missing piece, rather than a
    /// silent skip.
    pub multimodal: Option<&'a MultimodalIndex>,
    pub session: &'a Session,
    pub sandbox_tier: SandboxTier,
    pub sandbox_policy: SandboxPolicy,
    /// The subagent manager, for delegating to read-only workers. `None`
    /// turns `agent.delegate_readonly` into a refusal that names the
    /// missing piece — the alternative (silently doing nothing) is how a
    /// model ends up quoting a result that was never produced.
    pub subagents: Option<&'a SubagentManager>,
    /// The policy that a delegated worker would inherit. Built from the
    /// session and the run's permitted tools, so the worker's effective
    /// policy is *narrower* than the parent's, never wider.
    ///
    /// `None` when no run context is in reach — the runner is then the
    /// one built during early boot, before any run started.
    pub inherited: Option<&'a InheritedPolicy>,
    /// Where the run is writing. Workers inherit this as their workspace
    /// root, so a child cannot reach files outside the run.
    pub run_workspace: Option<&'a std::path::Path>,
}

impl<'a> LocalToolRunner<'a> {
    pub fn new(index: &'a KnowledgeIndex, session: &'a Session) -> Self {
        Self {
            index,
            multimodal: None,
            session,
            sandbox_tier: super::sandbox::detect_tier(),
            sandbox_policy: SandboxPolicy::default(),
            subagents: None,
            inherited: None,
            run_workspace: None,
        }
    }

    /// Build a runner that has both the prose and the multimodal index in
    /// reach. Used by the agent path once the database is open.
    pub fn with_multimodal(
        index: &'a KnowledgeIndex,
        multimodal: &'a MultimodalIndex,
        session: &'a Session,
    ) -> Self {
        Self {
            index,
            multimodal: Some(multimodal),
            session,
            sandbox_tier: super::sandbox::detect_tier(),
            sandbox_policy: SandboxPolicy::default(),
            subagents: None,
            inherited: None,
            run_workspace: None,
        }
    }

    /// Build a runner that has the subagent manager and the run's inherited
    /// policy in reach. Used once a run has been routed and a sub-task may
    /// legitimately be delegated.
    pub fn with_subagents(
        index: &'a KnowledgeIndex,
        session: &'a Session,
        subagents: &'a SubagentManager,
        inherited: &'a InheritedPolicy,
        run_workspace: &'a std::path::Path,
    ) -> Self {
        Self {
            index,
            multimodal: None,
            session,
            sandbox_tier: super::sandbox::detect_tier(),
            sandbox_policy: SandboxPolicy::default(),
            subagents: Some(subagents),
            inherited: Some(inherited),
            run_workspace: Some(run_workspace),
        }
    }

    /// Runs the search and hands back the passages themselves.
    ///
    /// Separate from [`Self::search`] because a caller that accumulates a whole
    /// run's evidence needs the passages, not the prose about them — and the
    /// two must not be able to disagree about what was retrieved.
    pub fn search_hits(&self, call: &ToolCall) -> Result<(String, Vec<SearchResult>), String> {
        let query = call.text("query").unwrap_or_default().to_string();
        // Clamped, never refused. A model asking for forty passages has
        // misjudged how much it needs, which is a different thing from a model
        // asking for a document it may not read — the first deserves the six it
        // can have, the second deserves a refusal. Spending a turn on a
        // quarrel about a count would teach it nothing.
        let wanted = call
            .integer("maxResults")
            .map(|n| (n as usize).clamp(1, SEARCH_LIMIT))
            .unwrap_or(SEARCH_LIMIT);
        let hits = self
            .index
            .search(self.session, &query, wanted)
            .map_err(|e| format!("the knowledge base could not be searched: {e}"))?;
        Ok((query, hits))
    }

    /// Runs a page-range read and hands back the passages themselves.
    ///
    /// The same split as [`Self::search_hits`], for the same reason: the caller
    /// that accumulates the run's evidence needs the passages, not the prose
    /// about them, and the two must not be able to disagree.
    pub fn region_hits(
        &self,
        call: &ToolCall,
    ) -> Result<(String, u32, u32, Vec<SearchResult>), String> {
        let document = call
            .text("documentSha256")
            .ok_or("load_more_evidence needs documentSha256, which is on every passage you have already retrieved")?
            .to_string();
        let from_page = call.integer("fromPage").ok_or("load_more_evidence needs fromPage")?;
        // A caller naming only a start page means that page. Defaulting to the
        // end of the document would put the whole thing back in the window,
        // which is the outcome this tool exists to avoid.
        let to_page = call.integer("toPage").unwrap_or(from_page);

        // Bounded here rather than trusted. A model that asks for pages 1 to
        // 10,000 is not asking for a region, it is asking for the document, and
        // serving that request is how the window overflows.
        if to_page.saturating_sub(from_page) >= REGION_PAGE_LIMIT {
            return Err(format!(
                "That is {} pages. Ask for at most {REGION_PAGE_LIMIT} pages at a time, and cite                  the passages you already hold for anything outside that range.",
                to_page.saturating_sub(from_page) + 1
            ));
        }

        let hits = self
            .index
            .region(self.session, &document, from_page, to_page, REGION_CHUNK_LIMIT)
            .map_err(|e| format!("that page range could not be read: {e}"))?;
        Ok((document, from_page, to_page, hits))
    }

    fn load_more_evidence(&self, call: &ToolCall) -> Result<String, String> {
        let (_, from_page, to_page, hits) = self.region_hits(call)?;
        let name = hits
            .first()
            .map(|hit| hit.document_name.clone())
            .unwrap_or_else(|| "that document".to_string());
        let marked: Vec<(usize, &SearchResult)> =
            hits.iter().enumerate().map(|(i, hit)| (i + 1, hit)).collect();
        let described = if from_page == to_page {
            format!("page {from_page} of {name}")
        } else {
            format!("pages {from_page} to {to_page} of {name}")
        };
        let rendered = render_passages(&described, &marked);
        if hits.is_empty() {
            return Ok(rendered);
        }
        // Which pages actually came back, not which were asked for. A page that
        // holds nothing indexable returns nothing, and a model that assumes it
        // received the range it named will cite a page it never read.
        Ok(format!("Read {described}.

{rendered}"))
    }

    fn search(&self, call: &ToolCall) -> Result<String, String> {
        let (query, hits) = self.search_hits(call)?;
        let marked: Vec<(usize, &SearchResult)> =
            hits.iter().enumerate().map(|(i, hit)| (i + 1, hit)).collect();

        // The cheap reading, for a model deciding *which* passages it wants
        // before spending window on their text. Without it the only way to find
        // out whether a search was useful is to read all six passages in full,
        // and a run that searches four times has then paid for twenty-four
        // passages to use three.
        if call.text("detail") == Some("citations") {
            if marked.is_empty() {
                return Ok(render_passages(&query, &marked));
            }
            let mut out = format!("{} passage(s) found, citations only.\n\n", marked.len());
            for (marker, hit) in &marked {
                out.push_str(&format!("[E{marker}] {}\n", hit.citation()));
            }
            out.push_str(
                "\nThese are citations, not the passage text — you have not read these passages \
                 and must not quote or paraphrase them yet. Search again with detail \"passages\", \
                 or use knowledge.load_evidence_region for the pages you want.\n",
            );
            return Ok(out);
        }

        Ok(render_passages(&query, &marked))
    }

    /// Multimodal retrieval: text passages, image regions, and table rows in
    /// one ranked result set.
    ///
    /// Three subsearches (prose, image regions, tables) are run, and the
    /// passages and regions are emitted side by side, each with its own
    /// marker ([E1], [I1], [T1]). The model sees one document and one set of
    /// citations, not three parallel ones it has to merge.
    ///
    /// The optional ``documentType`` argument lets a caller narrow by the
    /// auto-detected type. ``P&ID``-shaped queries are the obvious case:
    /// a model that asks about "PT-2201" wants the *symbol* region, and a
    /// query for "design pressure 14 bar" wants the datasheet *table*. A
    /// type filter skips the irrelevant subsearches and keeps the result
    /// set coherent.
    ///
    /// ``documentSha256`` narrows to a single document, the way the prose
    /// region's `LoadMoreEvidence` does. A model that has a passage and
    /// wants the visual evidence on the same page uses this rather than
    /// running a full multimodal search.
    fn multimodal_retrieve(&self, call: &ToolCall) -> Result<String, String> {
        let query = call.text("query").unwrap_or_default().to_string();
        if query.trim().is_empty() {
            return Err("knowledge.multimodal_retrieve needs a non-empty query.".into());
        }

        let Some(multimodal) = self.multimodal else {
            return Err(
                "The multimodal index is not available on this machine. Re-run a sync so the \
                 image and table index is populated, then try again."
                    .into(),
            );
        };

        // The cap on each subsearch. The combined result is the sum, so the
        // model gets up to 18 citations in one call — six of each kind.
        // Larger than the prose-only search because multimodal retrieval is
        // the only way to find regions and tables, and the cost of one
        // truncated call is another turn of the loop.
        let per_kind = call
            .integer("maxResults")
            .map(|n| (n as usize).clamp(1, 6))
            .unwrap_or(4);

        let document_filter: Option<&str> = call.text("documentSha256");
        let document_type_filter: Option<&str> = call.text("documentType");

        // Prose — the same search the text tool runs. We deliberately use
        // the same index, the same clearance, and the same ranking so a
        // passage that ranks well here is the same one the prose search
        // would have returned, and a citation [E1] points to the same
        // thing in either tool.
        let mut hits = self
            .index
            .search(self.session, &query, per_kind)
            .map_err(|e| format!("text search failed: {e}"))?;
        if let Some(sha) = document_filter {
            hits.retain(|hit| hit.document_sha256 == sha);
        }
        let passages: Vec<SearchResult> = hits;

        // Image regions. FTS over captions, with the same sanitisation.
        let mut regions: Vec<ImageRegion> = multimodal
            .search_regions(self.session, &query, per_kind)
            .map_err(|e| format!("image region search failed: {e}"))?;
        if let Some(sha) = document_filter {
            regions.retain(|r| r.document_sha256 == sha);
        }
        if let Some(doc_type) = document_type_filter {
            // The region FTS does not index the document type directly.
            // We use the per-document metadata to filter: a region whose
            // document does not match the requested type is dropped.
            regions.retain(|r| {
                multimodal
                    .document_meta(&r.document_sha256)
                    .ok()
                    .flatten()
                    .map(|m| m.document_type == doc_type)
                    .unwrap_or(false)
            });
        }

        // Tables. Same sanitisation; flat_text is the indexed form.
        let mut tables: Vec<TableChunk> = multimodal
            .search_tables(self.session, &query, per_kind)
            .map_err(|e| format!("table search failed: {e}"))?;
        if let Some(sha) = document_filter {
            tables.retain(|t| t.document_sha256 == sha);
        }
        if let Some(doc_type) = document_type_filter {
            tables.retain(|t| {
                multimodal
                    .document_meta(&t.document_sha256)
                    .ok()
                    .flatten()
                    .map(|m| m.document_type == doc_type)
                    .unwrap_or(false)
            });
        }

        if passages.is_empty() && regions.is_empty() && tables.is_empty() {
            return Ok(format!(
                "No text, image region, or table matched {query:?}. The connected collections \
                 do not contain this, so do not assert it. Either try different wording, or \
                 specify a documentSha256 to search within a known document."
            ));
        }

        // Render. Markers are [E1..] for prose, [I1..] for image regions,
        // [T1..] for tables. Distinct prefixes so a model citing one cannot
        // accidentally cite another of the same number from a different
        // kind.
        let mut out = format!("Multimodal search: {query:?}.\n\n");
        if !passages.is_empty() {
            out.push_str(&format!("{} text passage(s):\n", passages.len()));
            for (i, hit) in passages.iter().enumerate() {
                out.push_str(&format!("[E{}] {}\n{}\n\n", i + 1, hit.citation(), hit.text));
            }
        }
        if !regions.is_empty() {
            out.push_str(&format!("{} image region(s):\n", regions.len()));
            for (i, region) in regions.iter().enumerate() {
                out.push_str(&format!(
                    "[I{i}] {caption} ({kind:?}, page {page}, box [{l:.2},{t:.2}]-[{r:.2},{b:.2}], \
                     confidence {conf:.2})\n  document: {doc}\n\n",
                    i = i + 1,
                    caption = region.caption,
                    kind = region.kind,
                    page = region.page,
                    l = region.bbox.left,
                    t = region.bbox.top,
                    r = region.bbox.right,
                    b = region.bbox.bottom,
                    conf = region.box_confidence,
                    doc = region.document_name,
                ));
            }
        }
        if !tables.is_empty() {
            out.push_str(&format!("{} table(s):\n", tables.len()));
            for (i, table) in tables.iter().enumerate() {
                out.push_str(&format!(
                    "[T{i}] {citation}\n{flat}\n\n",
                    i = i + 1,
                    citation = table.citation(),
                    flat = table.flat_text,
                ));
            }
        }
        out.push_str(
            "Cite text by [E#], image regions by [I#] (which gives the page and box, not the \
             image itself), and tables by [T#]. A citation with no marker is not a citation.\n",
        );
        Ok(out)
    }

    /// Reports what a page range does and does not yield, rather than asserting.
    ///
    /// ## Why this is a findings tool and not an OCR tool
    ///
    /// A scanned inspection report reaches the index as pages, and the pages
    /// that were pictures rather than text yield no chunks at all. From the
    /// model's side that is indistinguishable from a page that was blank — and
    /// the two lead to opposite conclusions. A model that reads "nothing on
    /// page 5" concludes the clause it was looking for is not there; the truth
    /// is that nobody has read page 5 yet.
    ///
    /// So this names the difference. Pages with extracted text come back as
    /// citable passages; pages without come back as *unread*, said in the words
    /// that stop a model asserting anything about them. It never guesses at
    /// content, and when no OCR or vision engine is installed it says so rather
    /// than returning an empty result that reads like an answer.
    fn extract_findings(&self, call: &ToolCall) -> Result<String, String> {
        let (document, from_page, to_page, hits) = self.region_hits(call)?;

        let name = hits
            .first()
            .map(|hit| hit.document_name.clone())
            .unwrap_or_else(|| format!("document {}", short_sha(&document)));

        // Which pages in the asked-for range actually produced text. A page
        // absent from this set was not read, whatever the reason.
        let read_pages: std::collections::BTreeSet<u32> =
            hits.iter().map(|hit| hit.page).collect();
        let unread: Vec<u32> = (from_page..=to_page)
            .filter(|page| !read_pages.contains(page))
            .collect();

        let described = if from_page == to_page {
            format!("page {from_page} of {name}")
        } else {
            format!("pages {from_page} to {to_page} of {name}")
        };

        let mut out = format!("Findings for {described}.\n\n");

        if read_pages.is_empty() {
            out.push_str(
                "No page in this range holds extracted text. These pages are images that no \
                 installed engine has read: this deployment has no OCR or document vision model \
                 available, so their contents are unknown. Do not describe or quote them. Say \
                 that the pages could not be read and that a person needs to look at them.\n",
            );
            return Ok(out);
        }

        let marked: Vec<(usize, &SearchResult)> =
            hits.iter().enumerate().map(|(i, hit)| (i + 1, hit)).collect();
        out.push_str(&render_passages(&described, &marked));

        if !unread.is_empty() {
            // The load-bearing sentence. Without it a partial read looks whole.
            out.push_str(&format!(
                "\nUnread in this range: page(s) {}. They hold no extracted text — they are \
                 images, and no OCR or document vision model is installed to read them. Anything \
                 on those pages is unknown, not absent: do not conclude from this result that a \
                 clause or figure is missing from the document.\n",
                unread
                    .iter()
                    .map(u32::to_string)
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }

        Ok(out)
    }

    /// This machine's own account of what it sent and what it refused to send.
    ///
    /// Read from the broker rather than from a log the run wrote, because the
    /// question it answers — "did this task leak anything?" — is not one the
    /// task is a credible witness to. The broker is the thing every outbound
    /// attempt passes through, so its record is the one worth quoting.
    fn sovereignty_evidence(&self) -> Result<String, String> {
        let broker = crate::sovereignty::global_broker();
        let mode = broker.mode();
        let events = broker.recent_events();

        let mut out = format!(
            "Operating mode: {}. In this mode, {}.\n",
            mode.label(),
            if mode.permits_network() {
                "outbound calls are permitted to allow-listed hosts only, and confidential \
                 material may not be opened"
            } else {
                "no outbound call is permitted at all, and confidential material may be opened"
            }
        );

        if events.is_empty() {
            out.push_str(
                "\nNo outbound call has been attempted since this machine started. \
                 That is the whole record, not a summary of it.\n",
            );
            return Ok(out);
        }

        let refused = events.iter().filter(|event| !event.permitted).count();
        out.push_str(&format!(
            "\n{} outbound attempt(s) recorded since start, {refused} of them refused.\n\n",
            events.len()
        ));
        for event in events.iter().take(SOVEREIGNTY_EVENT_LIMIT) {
            out.push_str(&format!(
                "  {} {}{} — {}\n",
                if event.permitted { "sent   " } else { "refused" },
                event.host,
                // Named, because a canary is the app testing its own controls
                // rather than the app trying to reach somewhere. A reader who
                // could not tell the two apart would read a healthy self-test
                // as an attempted leak.
                if event.canary { " (self-test)" } else { "" },
                event.reason
            ));
        }
        if events.len() > SOVEREIGNTY_EVENT_LIMIT {
            out.push_str(&format!(
                "\n[{} older attempt(s) not shown. The full record is in the audit log, which a \
                 person can read.]\n",
                events.len() - SOVEREIGNTY_EVENT_LIMIT
            ));
        }
        Ok(out)
    }

    fn read(&self, call: &ToolCall, path: Option<&Path>) -> Result<String, String> {
        let path = path.ok_or("no path was resolved for this read")?;

        if !path.exists() {
            return Err(format!(
                "{} does not exist. List what is in the workspace before reading from it.",
                path.display()
            ));
        }

        let bytes = std::fs::read(path).map_err(|e| format!("{} could not be read: {e}", path.display()))?;

        // Checked here rather than at the gateway because the size on disk is
        // not knowable from the call the model wrote.
        let limit = spec_for(ToolName::ReadScopedFile)
            .max_bytes
            .unwrap_or(u64::MAX);
        if bytes.len() as u64 > limit {
            return Err(format!(
                "{} is {} MB, above the {} MB read limit.",
                path.display(),
                bytes.len() / 1024 / 1024,
                limit / 1024 / 1024
            ));
        }

        let text = String::from_utf8(bytes).map_err(|_| {
            format!(
                "{} is not text. Use the document tools to read a PDF or an image.",
                path.display()
            )
        })?;

        // A named window over the file, for the model that already knows where
        // it is going. The alternative — read the head, be told it was cut,
        // read again and be told the same thing — is how a run spends four
        // steps to reach line 900 of a thousand-line draft.
        let lines: Vec<&str> = text.lines().collect();
        let total = lines.len();
        let from = call
            .integer("fromLine")
            .map(|n| (n as usize).max(1))
            .unwrap_or(1);
        let asked_for_a_window = call.integer("fromLine").is_some() || call.integer("maxLines").is_some();

        if asked_for_a_window {
            if from > total {
                // Said as a fact about the file rather than as an empty result.
                // A model handed nothing concludes the file is empty; a model
                // told the file has 40 lines asks for line 1.
                return Err(format!(
                    "{} has {total} line(s), so line {from} is past the end of it. \
                     Ask for a line between 1 and {total}.",
                    name_of(path)
                ));
            }
            let span = call
                .integer("maxLines")
                .map(|n| (n as usize).clamp(1, READ_LINES))
                .unwrap_or(READ_LINES);
            let end = (from - 1 + span).min(total);
            let window = lines[from - 1..end].join("\n");
            let mut out = format!("{} lines {from}–{end} of {total}.\n\n{window}", name_of(path));
            if end < total {
                out.push_str(&format!(
                    "\n\n[{} more line(s) follow. This is a window on the file, not the whole \
                     of it — ask for fromLine {} to read on.]",
                    total - end,
                    end + 1
                ));
            }
            return Ok(out);
        }

        if text.chars().count() > READ_CHARS {
            let kept: String = text.chars().take(READ_CHARS).collect();
            // Truncation is stated, never silent: a model that believes it has
            // the whole file will confidently answer from the half it got.
            return Ok(format!(
                "{kept}\n\n[This file was longer than {READ_CHARS} characters and was cut off \
                 here. What you have above is the beginning of it, not the whole thing. It has \
                 {total} line(s) — ask again with fromLine to read a named window instead.]"
            ));
        }

        Ok(text)
    }

    fn write(&self, call: &ToolCall, path: Option<&Path>) -> Result<String, String> {
        let path = path.ok_or("no path was resolved for this write")?;
        let content = call.text("content").unwrap_or_default();

        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("could not prepare {}: {e}", parent.display()))?;
        }

        std::fs::write(path, content)
            .map_err(|e| format!("{} could not be written: {e}", path.display()))?;

        Ok(format!(
            "Wrote {} byte(s) to {}.",
            content.len(),
            path.display()
        ))
    }

    fn calculate(&self, call: &ToolCall) -> Result<String, String> {
        let expression = call.text("expression").unwrap_or_default();

        match calculation::evaluate(expression) {
            Ok(record) => {
                let mut out = format!("{} = {}\n", record.expression, record.formatted);
                out.push_str("Working:\n");
                for step in &record.steps {
                    out.push_str(&format!("  {} = {}\n", step.description, step.result));
                }
                out.push_str(&format!(
                    "Rounded to {}. Use this figure exactly as written; do not recompute it.",
                    record.rounding
                ));
                Ok(out)
            }
            // Returned as an error so the model sees it as a failed call and
            // corrects the expression, rather than as a result it might quote.
            Err(problem) => Err(problem.message),
        }
    }

    /// Runs a program the model wrote, in a container, and reports what it did.
    ///
    /// Two gates before anything runs, and they answer different questions.
    /// [`assess`] asks whether this *machine* may run untrusted code at all;
    /// [`sandbox_exec::run_in_container`] asks whether the isolation it would
    /// actually get keeps the promise. A tier an administrator has accepted the
    /// risk on passes the first and is refused by the second, because accepting
    /// a risk is a statement about policy, not a container this code knows how
    /// to drive.
    ///
    /// The result is written for a model to read, and the rule it exists to
    /// enforce is in the first line of every failure: **nothing ran, so there is
    /// no output to describe**. A model told that a program failed will
    /// otherwise reach for what the program would have printed.
    fn execute_code(&self, call: &ToolCall) -> Result<String, String> {
        let assessment = assess(self.sandbox_tier, &self.sandbox_policy);
        if let super::sandbox::SandboxAssessment::Refused { reason } = assessment {
            return Err(format!(
                "Code was not run: {reason} Nothing was executed, so no result exists — do not \
                 describe what the code would have produced."
            ));
        }

        let language = call.text("language").ok_or_else(|| {
            format!(
                "sandbox.run_code needs a language. Supported: {}.",
                sandbox_exec::SUPPORTED_LANGUAGES
            )
        })?;
        let source = call
            .text("source")
            .ok_or("sandbox.run_code needs the program to run, as `source`.")?;

        let workspace = self.run_workspace.ok_or(
            "this run has no workspace, so there is nowhere to put the program or its output. \
             Nothing was executed.",
        )?;

        let execution = sandbox_exec::run_in_container(
            self.sandbox_tier,
            &self.sandbox_policy,
            workspace,
            language,
            source,
        )?;

        // The tier goes in the result, not only in the audit record: a reader
        // deciding whether to trust an output needs to know what contained it.
        let mut out = format!(
            "Ran {language} in a {} sandbox ({:.1}s).\n",
            self.sandbox_tier.label(),
            execution.duration.as_secs_f64()
        );

        if execution.timed_out {
            out.push_str(&format!(
                "\nThe program did not finish within {}s and was stopped. Any output below is \
                 partial, and the work it was doing did not complete.\n",
                self.sandbox_policy.timeout.as_secs()
            ));
        } else {
            match execution.exit_code {
                Some(0) => out.push_str("Exit status: 0 (success).\n"),
                Some(code) => out.push_str(&format!(
                    "Exit status: {code}. The program ran and failed; the error is below, and it \
                     is the program's, not the sandbox's.\n"
                )),
                None => out.push_str("The program was terminated by a signal.\n"),
            }
        }

        if execution.stdout.trim().is_empty() {
            out.push_str("\nstdout: (empty)\n");
        } else {
            out.push_str(&format!("\nstdout:\n{}\n", execution.stdout));
        }

        if !execution.stderr.trim().is_empty() {
            out.push_str(&format!("\nstderr:\n{}\n", execution.stderr));
        }

        if execution.truncated_bytes > 0 {
            out.push_str(&format!(
                "\n{} byte(s) of output were dropped at the limit. What is shown above is the \
                 beginning of the output, not all of it.\n",
                execution.truncated_bytes
            ));
        }

        Ok(out)
    }

    fn validate(&self, path: Option<&Path>) -> Result<String, String> {
        let path = path.ok_or("no path was resolved for this check")?;

        if !path.exists() {
            return Err(format!("{} does not exist, so there is nothing to check.", path.display()));
        }

        let size = std::fs::metadata(path)
            .map_err(|e| format!("{} could not be inspected: {e}", path.display()))?
            .len();

        if size == 0 {
            return Err(format!(
                "{} exists but is empty, so it is not a usable file.",
                path.display()
            ));
        }

        Ok(format!("{} exists and holds {size} byte(s).", path.display()))
    }

    /// Hands a bounded, read-only sub-task to a worker.
    ///
    /// This is the model-facing surface of the subagent manager. The model
    /// sees a profile name and a one-sentence objective; the runner turns
    /// those into a packet, hands it to the manager, and renders the result
    /// back in the form a model is meant to read and cite.
    ///
    /// The runner refuses clearly when anything required is missing. The
    /// alternative — a silent no-op — is how a model ends up claiming a
    /// worker answered a question nobody asked.
    async fn delegate_to_subagent(&self, call: &ToolCall) -> Result<String, String> {
        let profile = call
            .text("profile")
            .ok_or("agent.delegate_readonly needs a profile name; one of: knowledge-retriever, document-extractor, calculation-checker, artifact-reviewer.")?;
        let task = call
            .text("task")
            .ok_or("agent.delegate_readonly needs a one-sentence task describing what to find out.")?;

        let subagents = self.subagents.ok_or(
            "Subagents are not available on this machine. The parent run was started without a \
             subagent manager; check the deployment configuration.",
        )?;
        let inherited = self.inherited.ok_or(
            "Subagent delegation is not wired into this run. The inherited policy was not \
             provided; this is a build configuration error, not something to retry.",
        )?;

        // The model decides to delegate; Rust decides whether to let it.
        // Capability is carried by the inherited policy, which the
        // subagent manager narrows; no second check is needed here.
        let profile_owned = profile.to_string();
        let task_owned = task.to_string();
        let inherited_clone = inherited.clone();
        let inputs = subagents.handoff_inputs(profile, inherited)?;
        // These registered specialists use local deterministic services.
        let decision = crate::subagents::certification::Decision {
            model_id: String::from("deterministic-local-services-v1"),
            role: crate::registry::ModelRole::Reasoning,
            cheaper_than_parent: false,
            reason: "bounded local service worker; no inference model or comparative cost measurement".to_string(),
            tier: None,
            score: None,
        };
        let result = subagents
            .spawn(&profile_owned, &inherited_clone, &task_owned, inputs, decision)
            .await
            .map_err(|refusal| refusal.explain())?;

        let child = result.result();
        if !child.is_complete() { return Err(child.describe()); }
        // The shape the model sees: status, findings (or refusal text),
        // and a one-line citation so a model citing the result has
        // something concrete to point at.
        let mut out = format!(
            "Subagent {profile} ran as child {child_id} (status: {status}).\n\n",
            profile = child.profile,
            child_id = child.child_id,
            status = child.status.as_str(),
        );
        if !child.findings.is_empty() {
            out.push_str("Findings:\n");
            for finding in &child.findings {
                out.push_str(&format!("- {}", finding.statement));
                for evidence in &finding.evidence {
                    if let Some(marker) = evidence.marker { out.push_str(&format!(" [E{marker}]")); }
                    out.push_str(&format!(" ({})", evidence.citation));
                }
                out.push('\n');
            }
        }
        if !child.uncertainty.is_empty() {
            out.push_str("\nUncertainty:\n");
            for note in &child.uncertainty {
                out.push_str(&format!("- {}\n", note));
            }
        }
        if let Some(detail) = &child.detail {
            out.push_str(&format!("\nDetail:\n{detail}\n"));
        }
        if result.is_reused() {
            out.push_str(
                "\nNote: this result was reused from an earlier call with the same idempotency key. \
                 No second child was started.\n",
            );
        }
        Ok(out)
    }
}

#[async_trait::async_trait]
impl ToolRunner for LocalToolRunner<'_> {
    async fn run(
        &self,
        tool: ToolName,
        call: &ToolCall,
        resolved_path: Option<&Path>,
    ) -> Result<String, String> {
        match tool {
            ToolName::SearchDocuments => self.search(call),
            ToolName::LoadMoreEvidence => self.load_more_evidence(call),
            ToolName::MediaExtractFindings => self.extract_findings(call),
            ToolName::KnowledgeMultimodalRetrieve => self.multimodal_retrieve(call),
            ToolName::ReadScopedFile => self.read(call, resolved_path),
            ToolName::WriteScopedFile => self.write(call, resolved_path),
            ToolName::RunCalculation => self.calculate(call),
            ToolName::ExecuteCode => self.execute_code(call),
            ToolName::ValidateArtifact => self.validate(resolved_path),
            ToolName::SovereigntyGetEvidence => self.sovereignty_evidence(),
            // Handled on the agent path, where the run's session and the memory
            // store are both in reach. The runner is built fresh per call and
            // holds neither, so serving these here would mean answering a
            // question about entitlement with no knowledge of who is asking.
            ToolName::MemoryRecallAuthorized | ToolName::MemoryPromoteApproved => Err(
                "Memory is served on the agent path, not by this runner.".to_string(),
            ),
            // Also agent-path: one needs the skill registry and the run's own
            // permitted-tool list, the other needs the subagent manager and the
            // parent's inherited policy. Both are properties of the run rather
            // than of this machine, and this runner knows only the machine.
            ToolName::CapabilitySearch => Err(format!(
                "{} is served on the agent path, not by this runner.",
                tool.as_str()
            )),
            // The delegation tool IS the subagent manager's surface. Wired here
            // rather than on the agent path so the executor's single-step
            // model is preserved: a parent that delegates is still doing one
            // step of its own, just one that returns a subagent's findings.
            ToolName::AgentDelegateReadonly => self.delegate_to_subagent(call).await,
            // Phase 6. Said plainly so a model does not describe a document it
            // has not produced.
            // Artifact production needs the run's accumulated state — its
            // calculations, its evidence, the files it has already produced —
            // which this runner is rebuilt too often to hold. The agent path
            // handles all three in `agent_runtime::artifacts`. Said plainly so
            // a model on the orchestrator path does not describe a document it
            // has not produced.
            ToolName::CreateDocx | ToolName::CreateXlsx | ToolName::CreatePptx => Err(format!(
                "Producing a {} is not available on this path, so no file was created and none exists.",
                match tool {
                    ToolName::CreateDocx => "Word document",
                    ToolName::CreateXlsx => "spreadsheet",
                    _ => "briefing deck",
                }
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::{Role, User};
    use crate::knowledge::{Chunk, ChunkKind};
    use crate::policy::Classification;
    use serde_json::json;
    use std::path::PathBuf;

    struct Fixture {
        _dir: tempfile::TempDir,
        root: PathBuf,
        index: KnowledgeIndex,
        session: Session,
    }

    fn fixture() -> Fixture {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().to_path_buf();
        let index = KnowledgeIndex::open(&root).unwrap();
        index
            .index_document(
                "Maintenance SOP",
                Classification::Internal,
                &[Chunk {
                    id: "c1".into(),
                    document_sha256: "sop".into(),
                    ordinal: 0,
                    text: "Minimum acceptable wall thickness is 9.0 mm.".into(),
                    page: 4,
                    section_path: vec!["4.2 Wall Thickness".into()],
                    kind: ChunkKind::Prose,
                    char_count: 44,
                }],
            )
            .unwrap();

        Fixture {
            _dir: dir,
            root,
            index,
            session: Session::open(User::new("kiran", "Kiran", vec![Role::Employee])),
        }
    }

    fn runner(f: &Fixture) -> LocalToolRunner<'_> {
        LocalToolRunner {
            index: &f.index,
            multimodal: None,
            session: &f.session,
            sandbox_tier: SandboxTier::JobObject,
            sandbox_policy: SandboxPolicy::default(),
            subagents: None,
            inherited: None,
            run_workspace: None,
        }
    }

    // ── Search ───────────────────────────────────────────────────────────

    #[tokio::test]
    async fn a_search_returns_passages_with_their_citations() {
        let f = fixture();
        let out = runner(&f)
            .run(
                ToolName::SearchDocuments,
                &ToolCall::new("search_documents", json!({ "query": "wall thickness" })),
                None,
            ).await.unwrap();

        assert!(out.contains("1 passage(s) found"));
        assert!(out.contains("Maintenance SOP"));
        assert!(out.contains("9.0 mm"));
    }

    /// PS Part C: no source found must be said, not left as silence for the
    /// model to fill in.
    #[tokio::test]
    async fn finding_nothing_says_so_and_tells_the_model_not_to_assert_it() {
        let f = fixture();
        let out = runner(&f)
            .run(
                ToolName::SearchDocuments,
                &ToolCall::new("search_documents", json!({ "query": "sasquatch" })),
                None,
            ).await.unwrap();

        assert!(out.contains("No passages matched"));
        assert!(out.contains("do not assert it"));
    }

    // ── Read ─────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn reading_a_file_returns_its_text() {
        let f = fixture();
        let path = f.root.join("note.txt");
        std::fs::write(&path, "Wall thickness measured at 8.2 mm.").unwrap();

        let out = runner(&f)
            .run(ToolName::ReadScopedFile, &ToolCall::new("read_scoped_file", json!({})), Some(&path)).await.unwrap();

        assert_eq!(out, "Wall thickness measured at 8.2 mm.");
    }

    #[tokio::test]
    async fn reading_a_missing_file_says_what_to_do_instead() {
        let f = fixture();
        let missing = f.root.join("absent.txt");

        let error = runner(&f)
            .run(ToolName::ReadScopedFile, &ToolCall::new("read_scoped_file", json!({})), Some(&missing)).await.unwrap_err();

        assert!(error.contains("does not exist"));
        assert!(error.contains("List what is in the workspace"));
    }

    /// A model that believes it has the whole file will confidently answer from
    /// the half it got.
    #[tokio::test]
    async fn a_long_file_is_truncated_and_says_so() {
        let f = fixture();
        let path = f.root.join("long.txt");
        std::fs::write(&path, "x".repeat(READ_CHARS + 5_000)).unwrap();

        let out = runner(&f)
            .run(ToolName::ReadScopedFile, &ToolCall::new("read_scoped_file", json!({})), Some(&path)).await.unwrap();

        assert!(out.contains("was cut off here"));
        assert!(out.contains("not the whole thing"));
    }

    #[tokio::test]
    async fn reading_a_binary_file_suggests_the_document_tools() {
        let f = fixture();
        let path = f.root.join("image.bin");
        std::fs::write(&path, [0xff, 0xfe, 0x00, 0x80]).unwrap();

        let error = runner(&f)
            .run(ToolName::ReadScopedFile, &ToolCall::new("read_scoped_file", json!({})), Some(&path)).await.unwrap_err();

        assert!(error.contains("is not text"));
        assert!(error.contains("document tools"));
    }

    // ── Write ────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn writing_creates_the_file_and_any_missing_directories() {
        let f = fixture();
        let path = f.root.join("out/deep/note.txt");

        let out = runner(&f)
            .run(
                ToolName::WriteScopedFile,
                &ToolCall::new("write_scoped_file", json!({ "content": "hello" })),
                Some(&path),
            ).await.unwrap();

        assert!(out.contains("Wrote 5 byte(s)"));
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "hello");
    }

    // ── Calculation ──────────────────────────────────────────────────────

    #[tokio::test]
    async fn a_calculation_returns_the_figure_and_its_working() {
        let f = fixture();
        let out = runner(&f)
            .run(
                ToolName::RunCalculation,
                &ToolCall::new("run_calculation", json!({ "expression": "(8.2 mm - 9.0 mm) / 9.0 mm * 100" })),
                None,
            ).await.unwrap();

        assert!(out.contains("-8.889"));
        assert!(out.contains("Working:"));
        // The model must quote this figure, not produce its own.
        assert!(out.contains("do not recompute it"));
    }

    /// A bad expression comes back as a failure so the model fixes it, rather
    /// than as a result it might quote.
    #[tokio::test]
    async fn an_impossible_calculation_is_an_error_not_a_result() {
        let f = fixture();
        let error = runner(&f)
            .run(
                ToolName::RunCalculation,
                &ToolCall::new("run_calculation", json!({ "expression": "8.2 mm + 9.0 kg" })),
                None,
            ).await.unwrap_err();

        assert!(error.contains("units do not match"));
    }

    // ── Things that are not built, said plainly ──────────────────────────

    /// The machine cannot isolate code, so nothing runs — and the model is told
    /// not to describe output that does not exist.
    #[tokio::test]
    async fn running_code_on_a_weak_sandbox_refuses_and_forbids_inventing_output() {
        let f = fixture();
        let error = runner(&f)
            .run(
                ToolName::ExecuteCode,
                &ToolCall::new("execute_code", json!({ "language": "python", "source": "print(1)" })),
                None,
            ).await.unwrap_err();

        assert!(error.contains("Code was not run"));
        assert!(error.contains("do not describe what the code would have produced"));
    }

    /// Artifact production is not on this path, and says so without ambiguity.
    ///
    /// All three artifact tools are handled by `agent_runtime::artifacts`,
    /// which holds the run's accumulated calculations and evidence. This runner
    /// is rebuilt per call and holds none of it, so it refuses.
    ///
    /// The assertion that matters is the second one. A model told only that
    /// something "is not available" may still write a sentence describing the
    /// document; one told plainly that no file exists has nothing to describe.
    #[tokio::test]
    async fn an_artifact_tool_on_this_path_says_no_file_exists() {
        for (tool, wire, name, label) in [
            (ToolName::CreateDocx, "create_docx", "note.docx", "Word document"),
            (ToolName::CreateXlsx, "create_xlsx", "working.xlsx", "spreadsheet"),
            (ToolName::CreatePptx, "create_pptx", "deck.pptx", "briefing deck"),
        ] {
            let f = fixture();
            let error = runner(&f)
                .run(tool, &ToolCall::new(wire, json!({})), Some(&f.root.join(name)))
                .await
                .unwrap_err();

            assert!(error.contains(label), "{wire}: refusal should name what it is: {error}");
            assert!(
                error.contains("no file was created and none exists"),
                "{wire}: refusal must foreclose describing a document that does not exist: {error}"
            );
        }
    }

    // ── Validation ───────────────────────────────────────────────────────

    #[tokio::test]
    async fn validating_reports_a_real_file() {
        let f = fixture();
        let path = f.root.join("note.txt");
        std::fs::write(&path, "content").unwrap();

        let out = runner(&f)
            .run(ToolName::ValidateArtifact, &ToolCall::new("validate_artifact", json!({})), Some(&path)).await.unwrap();

        assert!(out.contains("7 byte(s)"));
    }

    /// An empty file exists but is not a usable artifact, and saying "it exists"
    /// would let a task report success on nothing.
    #[tokio::test]
    async fn an_empty_file_fails_validation() {
        let f = fixture();
        let path = f.root.join("empty.docx");
        std::fs::write(&path, "").unwrap();

        let error = runner(&f)
            .run(ToolName::ValidateArtifact, &ToolCall::new("validate_artifact", json!({})), Some(&path)).await.unwrap_err();

        assert!(error.contains("empty"));
    }

    // ── Multimodal retrieval ─────────────────────────────────────────────

    use crate::knowledge::multimodal::{
        BBox, DocumentMeta, MultimodalIndex, NewRegion, NewTable,
    };

    /// Fixture that includes a multimodal index with one image region and
    /// one table on the same document the prose index holds a passage on.
    struct MultimodalFixture {
        _dir: tempfile::TempDir,
        root: PathBuf,
        index: KnowledgeIndex,
        multimodal: MultimodalIndex,
        session: Session,
    }

    fn multimodal_fixture() -> MultimodalFixture {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().to_path_buf();
        let index = KnowledgeIndex::open(&root).unwrap();
        index
            .index_document(
                "Maintenance SOP",
                Classification::Internal,
                &[Chunk {
                    id: "c1".into(),
                    document_sha256: "sop".into(),
                    ordinal: 0,
                    text: "Minimum acceptable wall thickness is 9.0 mm.".into(),
                    page: 4,
                    section_path: vec!["4.2 Wall Thickness".into()],
                    kind: ChunkKind::Prose,
                    char_count: 44,
                }],
            )
            .unwrap();
        let multimodal = MultimodalIndex::open(&root).unwrap();
        let meta = DocumentMeta {
            sha256: "sop".into(),
            name: "Maintenance SOP".into(),
            document_type: "sop".into(),
            type_confidence: 0.92,
            type_abstained: false,
            extraction_engine: "docling".into(),
            classification: Classification::Internal,
            page_count: 1,
        };
        let regions = vec![NewRegion {
            id: "r1",
            page: 4,
            kind_str: "image",
            bbox: BBox { left: 0.1, top: 0.2, right: 0.6, bottom: 0.8 },
            caption: "schematic of wall thickness gauge",
            label: Some("figure"),
            box_confidence: 0.8,
        }];
        let headers = vec!["Parameter".into(), "Value".into()];
        let rows = vec![
            vec!["Wall thickness".into(), "9.0 mm".into()],
            vec!["Material".into(), "SS 316".into()],
        ];
        let flat = "Parameter: Wall thickness | Value: 9.0 mm\nParameter: Material | Value: SS 316";
        let tables = vec![NewTable {
            id: "t1",
            page: 4,
            headers: &headers,
            rows: &rows,
            flat_text: flat,
            bbox: BBox { left: 0.0, top: 0.0, right: 1.0, bottom: 1.0 },
        }];
        multimodal.index_document(&meta, &regions, &tables).unwrap();

        MultimodalFixture {
            _dir: dir,
            root,
            index,
            multimodal,
            session: Session::open(User::new("kiran", "Kiran", vec![Role::Employee])),
        }
    }

    fn multimodal_runner(f: &MultimodalFixture) -> LocalToolRunner<'_> {
        LocalToolRunner::with_multimodal(&f.index, &f.multimodal, &f.session)
    }

    #[tokio::test]
    async fn multimodal_search_finds_text_regions_and_tables() {
        let f = multimodal_fixture();
        let out = multimodal_runner(&f)
            .run(
                ToolName::KnowledgeMultimodalRetrieve,
                &ToolCall::new(
                    "knowledge.multimodal_retrieve",
                    json!({ "query": "wall thickness" }),
                ),
                None,
            ).await.unwrap();
        // All three modalities should be present.
        assert!(out.contains("[E1]"), "expected a text passage marker, got {out}");
        assert!(out.contains("[I1]"), "expected an image region marker, got {out}");
        assert!(out.contains("[T1]"), "expected a table marker, got {out}");
    }

    #[tokio::test]
    async fn multimodal_search_with_no_match_says_so() {
        let f = multimodal_fixture();
        let out = multimodal_runner(&f)
            .run(
                ToolName::KnowledgeMultimodalRetrieve,
                &ToolCall::new(
                    "knowledge.multimodal_retrieve",
                    json!({ "query": "no-such-thing-anywhere" }),
                ),
                None,
            ).await.unwrap();
        assert!(out.contains("No text, image region, or table matched"));
    }

    #[tokio::test]
    async fn multimodal_search_refuses_empty_query() {
        let f = multimodal_fixture();
        let error = multimodal_runner(&f)
            .run(
                ToolName::KnowledgeMultimodalRetrieve,
                &ToolCall::new(
                    "knowledge.multimodal_retrieve",
                    json!({ "query": "" }),
                ),
                None,
            ).await.unwrap_err();
        assert!(error.contains("non-empty"));
    }

    #[tokio::test]
    async fn multimodal_search_filters_by_document() {
        let f = multimodal_fixture();
        let out = multimodal_runner(&f)
            .run(
                ToolName::KnowledgeMultimodalRetrieve,
                &ToolCall::new(
                    "knowledge.multimodal_retrieve",
                    json!({
                        "query": "wall thickness",
                        "documentSha256": "sop"
                    }),
                ),
                None,
            ).await.unwrap();
        assert!(out.contains("Maintenance SOP"));
    }

    #[tokio::test]
    async fn multimodal_search_without_index_refuses() {
        let f = fixture();
        // The base fixture has no multimodal index, so the runner is built
        // with `multimodal: None`.
        let error = runner(&f)
            .run(
                ToolName::KnowledgeMultimodalRetrieve,
                &ToolCall::new(
                    "knowledge.multimodal_retrieve",
                    json!({ "query": "wall thickness" }),
                ),
                None,
            ).await.unwrap_err();
        assert!(error.contains("multimodal index is not available"));
    }
}
