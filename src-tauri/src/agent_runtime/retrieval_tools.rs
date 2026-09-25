//! The Knowledge Retriever's tools (P07), on the agent path.
//!
//! | Tool | What it does |
//! |---|---|
//! | `knowledge.hybrid_search` | keyword + semantic search inside the run's pinned scope, recorded as `[En]` evidence |
//! | `knowledge.rerank` | a bounded local reorder of passages this run already retrieved |
//! | `knowledge.source_version` | the version history of one indexed source |
//! | `memory.neighbours` | a bounded walk of the task's memory graph |
//!
//! All four are here rather than in `LocalToolRunner` because each needs the
//! run: its pinned scope, its evidence table, its task's memory. All four
//! authorise inside the index or the graph before anything is scored, walked
//! or counted — the handlers only render what came back.

use std::fmt::Write as _;
use std::sync::Arc;

use crate::identity::Session;
use crate::knowledge::graph::runtime_memory::{EdgeKind, MemoryScope};
use crate::knowledge::hybrid::{rerank as rerank_hits, EvidenceHit, HybridRequest, HybridResponse, LexicalProximityReranker, Reranker, RERANK_LIMIT};
use crate::orchestrator::tools::ToolCall;

use super::{retrieval, CallParams, RuntimeDeps};

/// Default number of passages when the model does not ask for a count.
const DEFAULT_RESULTS: usize = 6;

/// Most items `memory.neighbours` returns.
const NEIGHBOURHOOD_LIMIT: usize = 24;

/// A list argument's strings.
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
            .split([',', ';', '\n'])
            .map(|part| part.trim().to_string())
            .filter(|part| !part.is_empty())
            .collect(),
        _ => Vec::new(),
    }
}

/// Renders a hybrid response with the markers the run's table gave it.
///
/// Excerpts, not whole passages: the model reads more with
/// `knowledge.load_evidence_region`. Every score is named for what it is, and
/// the coverage footer says what the search could not do.
pub fn render(response: &HybridResponse, markers: &[usize]) -> String {
    let mut out = String::new();
    if response.hits.is_empty() {
        let _ = writeln!(
            out,
            "No passage you may read answers {:?}. This is a finding: do not assert what no \
             source says. Try other wording, or say that no source was found.",
            response.query
        );
    } else {
        let _ = writeln!(out, "{} passage(s) for {:?}.\n", response.hits.len(), response.query);
    }
    for (hit, marker) in response.hits.iter().zip(markers) {
        let version = match &hit.source {
            Some(source) if source.superseded_since_pin => {
                format!(" · version {} (superseded since this work was queued)", source.version)
            }
            Some(source) => format!(" · version {} ({})", source.version, source.status),
            None if hit.handle.starts_with("nb:") => " · notebook source".to_string(),
            None => String::new(),
        };
        let _ = writeln!(
            out,
            "[E{marker}] {}{version} · found by {} · {}",
            hit.passage.citation(),
            hit.method.label(),
            hit.handle
        );
        let _ = writeln!(out, "{}", hit.excerpt);
        let _ = writeln!(out, "  scores: {}", hit.scores.describe());
        if !hit.duplicates.is_empty() {
            let _ = writeln!(out, "  the same text is also in: {}", hit.duplicates.join("; "));
        }
        let _ = writeln!(out, "  documentSha256: {}\n", hit.passage.document_sha256);
    }
    let coverage = &response.coverage;
    let _ = write!(out, "Retrieval: {}", coverage.mode);
    if let Some(space) = &coverage.space_key {
        let _ = write!(out, " (semantic space {space})");
    }
    if let Some(because) = &coverage.degraded_because {
        let _ = write!(out, ". Keyword only because {}", because.trim_end_matches('.'));
    }
    out.push_str(".\n");
    for line in &coverage.partial {
        let _ = writeln!(out, "Partial: {}.", line.trim_end_matches('.'));
    }
    if !response.hits.is_empty() {
        out.push_str(
            "Cite by marker. Excerpts only: read the page with knowledge.load_evidence_region \
             before quoting anything the excerpt cuts off.\n",
        );
    }
    out
}

/// `knowledge.hybrid_search`.
///
/// Returns the rendering and the chunk ids it recorded, for the receipt.
pub async fn hybrid_search(
    deps: &Arc<RuntimeDeps>,
    call: &CallParams,
    session: &Session,
    tool_call: &ToolCall,
) -> (Result<String, String>, Vec<String>) {
    let query = tool_call.text("query").unwrap_or_default().trim().to_string();
    if query.is_empty() {
        return (Err("knowledge.hybrid_search needs a non-empty query.".into()), Vec::new());
    }
    let limit = tool_call
        .integer("maxResults")
        .map(|n| (n as usize).clamp(1, crate::knowledge::hybrid::MAX_RESULTS))
        .unwrap_or(DEFAULT_RESULTS);
    let documents = strings(tool_call, "documentSha256s");
    let mut request = HybridRequest {
        query,
        limit,
        ..Default::default()
    };
    if !documents.is_empty() {
        request.scope.documents = Some(documents);
    }
    let scope = deps.retrieval.scope_for(&call.run_id);
    let response = match deps.retrieval.search_scoped(session, &request, &scope).await {
        Ok(response) => response,
        Err(error) => return (Err(format!("the knowledge base could not be searched: {error}")), Vec::new()),
    };
    let passages: Vec<_> = response.hits.iter().map(|hit| hit.passage.clone()).collect();
    let Some(markers) = retrieval::record_markers(&deps.passages, &call.run_id, &passages) else {
        return (
            Err("the passages were found but could not be recorded as this task's evidence, so \
                 nothing retrieved now can be cited"
                .into()),
            Vec::new(),
        );
    };
    let chunks = passages.iter().map(|passage| passage.chunk_id.clone()).collect();
    (Ok(render(&response, &markers)), chunks)
}

/// `knowledge.rerank`.
pub fn rerank(deps: &Arc<RuntimeDeps>, call: &CallParams, tool_call: &ToolCall) -> Result<String, String> {
    let query = tool_call.text("query").unwrap_or_default().trim().to_string();
    if query.is_empty() {
        return Err("knowledge.rerank needs the question to rank against.".into());
    }
    let held = retrieval::for_run(&deps.passages, &call.run_id);
    if held.is_empty() {
        return Err("This task has retrieved nothing yet, so there is nothing to reorder. Search first.".into());
    }
    let wanted: Vec<usize> = strings(tool_call, "markers")
        .iter()
        .filter_map(|marker| marker.trim_start_matches(['E', 'e']).parse::<usize>().ok())
        .collect();
    let chosen: Vec<usize> = if wanted.is_empty() {
        (1..=held.len()).collect()
    } else {
        wanted
    };
    let unknown: Vec<String> = chosen
        .iter()
        .filter(|marker| **marker == 0 || **marker > held.len())
        .map(|marker| format!("E{marker}"))
        .collect();
    if !unknown.is_empty() {
        return Err(format!(
            "{} are not markers this task holds; it holds E1 to E{}.",
            unknown.join(", "),
            held.len()
        ));
    }
    let bounded: Vec<usize> = chosen.into_iter().take(RERANK_LIMIT).collect();
    let mut response = HybridResponse {
        query: query.clone(),
        hits: bounded
            .iter()
            .map(|marker| {
                let passage = held[marker - 1].clone();
                EvidenceHit {
                    handle: EvidenceHit::handle_for(&passage.chunk_id),
                    excerpt: String::new(),
                    passage,
                    method: crate::knowledge::hybrid::HitMethod::Lexical,
                    scores: Default::default(),
                    source: None,
                    duplicates: Vec::new(),
                }
            })
            .collect(),
        ..Default::default()
    };
    let reranker = LexicalProximityReranker;
    rerank_hits(&mut response, &reranker);
    let mut out = format!("Reordered {} passage(s) by {}.\n", response.hits.len(), reranker.describe());
    for hit in &response.hits {
        let marker = held
            .iter()
            .position(|kept| kept.chunk_id == hit.passage.chunk_id)
            .map(|index| index + 1)
            .unwrap_or(0);
        let _ = writeln!(
            out,
            "[E{marker}] {} — rerank {:.2} (ordering only)",
            hit.passage.citation(),
            hit.scores.rerank.unwrap_or(0.0)
        );
    }
    Ok(out)
}

/// `knowledge.source_version`.
pub fn source_version(deps: &Arc<RuntimeDeps>, session: &Session, tool_call: &ToolCall) -> Result<String, String> {
    let sha = tool_call.text("documentSha256").unwrap_or_default().trim().to_string();
    if sha.is_empty() {
        return Err("knowledge.source_version needs documentSha256, which every passage carries.".into());
    }
    let history = deps
        .retrieval
        .index
        .source_versions(session, &sha)
        .map_err(|error| format!("the version history could not be read: {error}"))?;
    let Some(history) = history else {
        // The same sentence for "never indexed" and "not yours to see".
        return Ok(format!("No indexed document with hash {sha} is visible to you."));
    };
    if history.is_empty() {
        return Ok(format!(
            "{sha} is indexed but was not read from a collection, so it has no version history. \
             Treat it as the only version known."
        ));
    }
    let mut out = String::new();
    for version in history.iter().filter(|version| version.document_sha256 == sha) {
        let _ = writeln!(
            out,
            "{} in {}: version {} is {}.",
            version.relative_path, version.collection_id, version.version, version.status
        );
    }
    out.push_str("History:\n");
    for version in &history {
        let replaced = version
            .superseded_by
            .as_ref()
            .map(|by| format!(", replaced by {by}"))
            .unwrap_or_default();
        let retired = version
            .retired_at
            .as_ref()
            .map(|at| format!(", retired {at}"))
            .unwrap_or_default();
        let _ = writeln!(
            out,
            "- {} v{}: {} (indexed {}{retired}{replaced}) sha {}",
            version.relative_path,
            version.version,
            version.status,
            version.indexed_at,
            version.document_sha256
        );
    }
    let current = history.iter().find(|version| version.status == "current");
    match history.iter().find(|version| version.document_sha256 == sha) {
        Some(asked) if asked.status == "current" => out.push_str("This is the current version.\n"),
        Some(_) => match current {
            Some(current) => {
                let _ = writeln!(
                    out,
                    "This is not the current version. The current one is {} (v{}): cite that, \
                     or say the passage you hold is from an older version.",
                    current.document_sha256, current.version
                );
            }
            None => out.push_str(
                "No current version exists: the source was withdrawn or revoked. Do not rely on \
                 it as current guidance.\n",
            ),
        },
        None => {}
    }
    Ok(out)
}

/// `memory.neighbours`.
pub fn neighbours(deps: &Arc<RuntimeDeps>, call: &CallParams, session: &Session, tool_call: &ToolCall) -> Result<String, String> {
    let Some(graph) = deps.memory_graph.as_ref() else {
        return Err(
            "This deployment has no runtime memory graph, so there are no links to follow.".into(),
        );
    };
    let item_id = tool_call.text("itemId").unwrap_or_default().trim().to_string();
    if item_id.is_empty() {
        return Err("memory.neighbours needs itemId, an mi- id from this task's memory.".into());
    }
    let depth = tool_call.integer("depth").unwrap_or(1).clamp(1, 2) as u32;
    let kinds: Vec<EdgeKind> = strings(tool_call, "edgeKinds")
        .iter()
        .filter_map(|kind| EdgeKind::parse(kind))
        .collect();
    let task_id = deps
        .plans
        .lock()
        .ok()
        .and_then(|plans| plans.get(&call.run_id).map(|plan| plan.task_id.clone()))
        .filter(|task| !task.trim().is_empty())
        .unwrap_or_else(|| call.run_id.clone());
    // The authorised set first; the walk never leaves it.
    let authorised = graph
        .snapshot(session, &MemoryScope::Task { task_id }, None)
        .map_err(|error| error.explain())?;
    let (items, edges) = graph
        .neighbourhood(
            &item_id,
            &authorised,
            depth,
            NEIGHBOURHOOD_LIMIT,
            (!kinds.is_empty()).then_some(kinds.as_slice()),
        )
        .map_err(|error| error.explain())?;
    if items.is_empty() {
        return Ok(format!("No memory item {item_id} is visible to you in this task."));
    }
    let mut out = String::new();
    for item in &items {
        let content: String = item.content.chars().take(200).collect();
        let _ = writeln!(
            out,
            "{} r{} [{} · {}] {}",
            item.item_id,
            item.revision,
            item.kind.as_str(),
            item.status.as_str(),
            content
        );
    }
    if edges.is_empty() {
        out.push_str(
            "No visible links. That is not evidence the item is true; it is only unconnected.\n",
        );
    } else {
        out.push_str("Links:\n");
        for edge in &edges {
            let _ = writeln!(out, "- {} {} {}", edge.from_item, edge.kind.as_str(), edge.to_item);
        }
    }
    if items.len() >= NEIGHBOURHOOD_LIMIT {
        let _ = writeln!(out, "Stopped at {NEIGHBOURHOOD_LIMIT} items; walk from one of these for more.");
    }
    Ok(out)
}
