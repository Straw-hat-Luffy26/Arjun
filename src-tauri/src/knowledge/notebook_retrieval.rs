//! Turning "ask my notebook this" into passages a run can cite.
//!
//! ## What the frontend is allowed to say
//!
//! Three lists of ids: a notebook, some of its sources, and — when the question
//! came off the graph — some nodes and claims. Nothing else. It does not send
//! document text, it does not send rendered Markdown, and it does not send
//! citation labels, because every one of those would be evidence the backend
//! never read and cannot check.
//!
//! That is a change from how the graph reached a chat before this module. The
//! Notebooks screen rendered a subgraph to Markdown and attached it as a
//! document, so the "evidence" behind an answer was a summary of a graph, with
//! no way back to the passages underneath. It read well and it could not be
//! checked.
//!
//! Here the ids are resolved against the store, for the signed-in owner, and the
//! answer is built from the original source passages.
//!
//! ## Authorisation happens before anything is loaded
//!
//! [`resolve`] establishes ownership of the notebook first, then validates every
//! source id against that notebook's membership, then resolves graph ids within
//! it. A source id that belongs to a different notebook — or to nobody — is
//! refused by name rather than quietly dropped, because a question answered from
//! four of the five sources somebody selected is a wrong answer that looks
//! right.
//!
//! ## The retrieval mode is reported, never implied
//!
//! [`RetrievalMode::Keyword`] is what runs when no embedding model is loaded,
//! and it says so. This module will not start a download, will not fall back to
//! a zero vector, and will not describe a term-frequency scan as semantic
//! search. See [`crate::knowledge::embedding`], which has the same rule.

use std::collections::{BTreeMap, BTreeSet};

use crate::agent_runtime::doc_pipeline;
use crate::agent_runtime::documents::{DocumentStore, ExtractedDocument};
use crate::knowledge::chunking::Chunk;
use crate::knowledge::graph::research::{
    EvidenceEntry, EvidenceManifest, ResearchScope, RetrievalMode,
};
use crate::knowledge::graph::{
    Assertion, AssertionStatus, Notebook, NotebookDocument, NotebookStore,
};
use crate::knowledge::index::{Retrieval, SearchResult};
use crate::policy::Classification;

/// Most passages one research turn puts in front of the model.
///
/// Bounded because the window is, and because an answer built on forty passages
/// cites none of them usefully. The count is reported on the manifest, so
/// "sixteen passages from four of your six sources" is a statement the screen
/// can make.
pub const MAX_PASSAGES: usize = 16;

/// Most characters of passage text one turn may inject.
///
/// The ceiling that makes the count above safe on a small window. When it bites,
/// the turn says so — see [`NotebookRetrieval::limitations`] — rather than
/// quietly answering from half the evidence.
pub const MAX_PASSAGE_CHARS: usize = 24_000;

/// How far above its keyword score a passage rises for naming a focused term.
///
/// Multiplicative rather than additive so it re-orders without swamping: a
/// passage that mentions the selected entity *and* answers the question still
/// beats one that only mentions the entity.
const FOCUS_BOOST: f64 = 2.5;

/// The boost for a passage a selected claim was actually read out of.
///
/// Higher than [`FOCUS_BOOST`], because this is not "related to the thing you
/// selected" — it is the sentence the claim came from.
const CLAIM_EVIDENCE_BOOST: f64 = 4.0;

/// A scope that has been checked, with everything it names resolved.
#[derive(Debug, Clone)]
pub struct ResolvedScope {
    pub notebook: Notebook,
    /// The sources this question may use, in the notebook's own order.
    pub sources: Vec<NotebookDocument>,
    /// Display labels of the selected graph nodes, for the prompt and the UI.
    pub focus_labels: Vec<String>,
    /// Normalised terms behind those nodes, for ranking.
    pub focus_terms: Vec<String>,
    /// The selected claims, with their direction and review status intact.
    pub focus_assertions: Vec<Assertion>,
    /// Chunks the selected claims were read out of.
    pub focus_chunk_ids: BTreeSet<String>,
    /// What could not be resolved, in sentences meant for a person.
    pub limitations: Vec<String>,
}

impl ResolvedScope {
    /// Whether the graph narrowed this question.
    pub fn is_graph_focused(&self) -> bool {
        !self.focus_terms.is_empty() || !self.focus_assertions.is_empty()
    }
}

/// Checks a scope and resolves everything it names.
///
/// Returns an error — not an empty result — when the notebook is not this
/// person's or a named source is not in it. A silent narrowing here is the
/// failure mode worth refusing loudly.
pub fn resolve(
    store: &NotebookStore,
    owner_user_id: &str,
    scope: &ResearchScope,
) -> Result<ResolvedScope, String> {
    // 1. The notebook, and whose it is. Before anything else is read.
    let notebook = store
        .get(&scope.notebook_id, owner_user_id)
        .map_err(|error| format!("the notebook could not be read: {error}"))?
        .ok_or_else(|| "That notebook does not exist.".to_string())?;

    let members = store
        .documents(&notebook.id, owner_user_id)
        .map_err(|error| format!("the notebook's documents could not be read: {error}"))?;

    // 2. Every named source must be in this notebook. An id from another
    //    notebook is refused rather than dropped.
    let known: BTreeMap<&str, &NotebookDocument> = members
        .iter()
        .map(|member| (member.document_sha256.as_str(), member))
        .collect();
    let sources: Vec<NotebookDocument> = if scope.source_sha256s.is_empty() {
        members.clone()
    } else {
        let mut chosen = Vec::with_capacity(scope.source_sha256s.len());
        for sha in &scope.source_sha256s {
            match known.get(sha.as_str()) {
                Some(member) => chosen.push((*member).clone()),
                None => {
                    return Err(format!(
                        "One of the selected sources is not in \"{}\". Reopen the notebook and \
                         choose the sources again.",
                        notebook.name
                    ))
                }
            }
        }
        chosen
    };

    if sources.is_empty() {
        return Err(format!(
            "\"{}\" has no sources yet. Add documents to it before asking questions of it.",
            notebook.name
        ));
    }

    let mut limitations = Vec::new();

    // 3. Graph ids, inside this notebook. A node id is a hash of the notebook
    //    and the term, so one from another notebook cannot match — and the view
    //    is read scoped to this notebook anyway, so there is no path on which
    //    it could.
    let mut focus_labels = Vec::new();
    let mut focus_terms = Vec::new();
    if !scope.node_ids.is_empty() {
        let view = store
            .graph(&notebook.id, owner_user_id, None, None, 1, 1, true)
            .map_err(|error| format!("the graph could not be read: {error}"))?;
        let wanted: BTreeSet<&String> = scope.node_ids.iter().collect();
        for node in &view.nodes {
            if !wanted.contains(&node.id) {
                continue;
            }
            focus_labels.push(node.label.clone());
            // A document node focuses the question on that file rather than on
            // a term, and it has no term to rank by.
            if node.document_sha256.is_none() {
                focus_terms.push(node.label.to_lowercase());
            }
        }
        let missing = scope.node_ids.len().saturating_sub(focus_labels.len());
        if missing > 0 {
            limitations.push(format!(
                "{missing} of the selected graph items are no longer in this notebook's graph, so \
                 they did not narrow this question."
            ));
        }
    }

    // 4. Claims, with their direction preserved.
    let mut focus_assertions = Vec::new();
    let mut focus_chunk_ids = BTreeSet::new();
    for id in &scope.assertion_ids {
        match store.assertion(&notebook.id, owner_user_id, id) {
            Ok(Some(assertion)) => {
                if !assertion.status.usable_as_evidence() {
                    limitations.push(format!(
                        "The relationship \"{} {} {}\" was rejected on review, so it was not used.",
                        assertion.subject_label, assertion.predicate, assertion.object_label
                    ));
                    continue;
                }
                for evidence in &assertion.evidence {
                    focus_chunk_ids.insert(evidence.chunk_id.clone());
                }
                if !focus_terms.contains(&assertion.subject) {
                    focus_terms.push(assertion.subject.clone());
                }
                if !focus_terms.contains(&assertion.object) {
                    focus_terms.push(assertion.object.clone());
                }
                focus_assertions.push(assertion);
            }
            Ok(None) => limitations.push(
                "One of the selected relationships is no longer in this notebook's graph."
                    .to_string(),
            ),
            Err(error) => {
                limitations.push(format!("A selected relationship could not be read: {error}"))
            }
        }
    }

    // Terms are matched against lowercase passage text.
    for term in &mut focus_terms {
        *term = term.to_lowercase();
    }
    focus_terms.retain(|term| !term.trim().is_empty());
    focus_terms.sort();
    focus_terms.dedup();

    Ok(ResolvedScope {
        notebook,
        sources,
        focus_labels,
        focus_terms,
        focus_assertions,
        focus_chunk_ids,
        limitations,
    })
}

/// What a retrieval produced, and what it could not.
#[derive(Debug, Clone)]
pub struct NotebookRetrieval {
    /// The passages, best first, in the shape the run's evidence table and the
    /// citation verifier already speak.
    pub passages: Vec<SearchResult>,
    pub mode: RetrievalMode,
    pub limitations: Vec<String>,
    /// Sources in scope that contributed nothing, by name.
    pub sources_unused: Vec<String>,
    /// The extraction revision each source was read at.
    pub revisions: BTreeMap<String, String>,
}

/// Finds the passages that answer a question, within a checked scope.
///
/// ## Ranking
///
/// [`doc_pipeline::rank_chunks`] per source, then merged — the same scorer the
/// composer uses to decide which passages of an attached document go into a
/// turn, so a passage the turn would have included is the passage a search for
/// the same words returns. Ranking per document and merging afterwards matters:
/// term rarity is a property of a document, and pooling six unrelated documents
/// makes a word that is common in one look rare because the others never use it.
///
/// A focused term or a claim's own passage multiplies the score rather than
/// replacing it, so selecting an entity narrows the question without turning the
/// answer into a list of every mention of that entity.
pub fn retrieve(
    documents: &DocumentStore,
    owner_user_id: &str,
    scope: &ResolvedScope,
    question: &str,
) -> NotebookRetrieval {
    let conversation = format!("notebook:{}", scope.notebook.id);
    let mut limitations = Vec::new();
    let mut revisions = BTreeMap::new();
    let mut scored: Vec<(f64, SearchResult)> = Vec::new();

    for source in &scope.sources {
        // Asked for under the notebook's own synthetic conversation id, which
        // is how a notebook's documents were recorded when they were added.
        // This is the authorisation boundary, not a lookup convenience: a
        // document this owner never put in this notebook is not visible here.
        let document =
            match documents.get(&source.document_sha256, owner_user_id, Some(&conversation)) {
                Ok(Some(document)) => document,
                Ok(None) => {
                    limitations.push(format!(
                        "{}: its extracted text is missing, so nothing from it could be used.",
                        source.document_name
                    ));
                    continue;
                }
                Err(error) => {
                    limitations.push(format!("{}: {error}", source.document_name));
                    continue;
                }
            };

        revisions.insert(source.document_sha256.clone(), revision_of(&document));

        if document.chunks.is_empty() {
            limitations.push(format!(
                "{}: no readable text was extracted from it, so it could not be searched.",
                source.document_name
            ));
            continue;
        }
        if document.truncated {
            limitations.push(format!(
                "{}: the reader stopped early when this was first read, so its later pages are \
                 not in the index at all.",
                source.document_name
            ));
        }

        for (score, chunk) in doc_pipeline::rank_chunks(question, &document.chunks) {
            let boosted = score * boost_for(chunk, scope);
            scored.push((boosted, to_result(&document, chunk, boosted)));
        }

        // A claim's own passage is evidence for the question even when it
        // shares no words with it — "what connects these two?" rarely repeats
        // the sentence that connects them.
        if !scope.focus_chunk_ids.is_empty() {
            for chunk in &document.chunks {
                if !scope.focus_chunk_ids.contains(&chunk.id) {
                    continue;
                }
                if scored.iter().any(|(_, hit)| hit.chunk_id == chunk.id) {
                    continue;
                }
                scored.push((
                    CLAIM_EVIDENCE_BOOST,
                    to_result(&document, chunk, CLAIM_EVIDENCE_BOOST),
                ));
            }
        }
    }

    scored.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.1.document_name.cmp(&b.1.document_name))
            .then_with(|| a.1.page.cmp(&b.1.page))
    });

    let mut passages: Vec<SearchResult> = Vec::new();
    let mut characters = 0usize;
    let mut budget_bit = false;
    for (_, hit) in scored {
        if passages.len() >= MAX_PASSAGES {
            budget_bit = true;
            break;
        }
        if characters + hit.text.len() > MAX_PASSAGE_CHARS {
            budget_bit = true;
            continue;
        }
        characters += hit.text.len();
        passages.push(hit);
    }

    if budget_bit {
        limitations.push(format!(
            "More passages matched than fit in one turn. The best {} were used; ask a narrower \
             question, or deselect sources, to reach the rest.",
            passages.len()
        ));
    }

    let used: BTreeSet<&str> = passages
        .iter()
        .map(|hit| hit.document_sha256.as_str())
        .collect();
    let sources_unused: Vec<String> = scope
        .sources
        .iter()
        .filter(|source| !used.contains(source.document_sha256.as_str()))
        .map(|source| source.document_name.clone())
        .collect();

    NotebookRetrieval {
        passages,
        // Honest by construction: this function ranks by shared words. When an
        // embedding model is wired in, the caller that wires it is the one that
        // changes this, and it will be changing a real second ranking rather
        // than a label.
        mode: RetrievalMode::Keyword,
        limitations,
        sources_unused,
        revisions,
    }
}

/// Builds the record of what a turn used.
///
/// The markers are 1-based and match the order the passages were handed to the
/// run's evidence table, which is what makes `[E2]` in the answer resolve to
/// entry 2 here a year later.
pub fn manifest(
    run_id: &str,
    conversation_id: &str,
    message_id: &str,
    scope: &ResearchScope,
    resolved: &ResolvedScope,
    retrieval: &NotebookRetrieval,
    graph_revision: Option<String>,
) -> EvidenceManifest {
    let entries = retrieval
        .passages
        .iter()
        .enumerate()
        .map(|(index, hit)| EvidenceEntry {
            marker: index as u32 + 1,
            chunk_id: hit.chunk_id.clone(),
            document_sha256: hit.document_sha256.clone(),
            document_name: hit.document_name.clone(),
            page: hit.page,
            section_path: hit.section_path.clone(),
            source_revision: retrieval
                .revisions
                .get(&hit.document_sha256)
                .cloned()
                .unwrap_or_default(),
            quote: hit.text.clone(),
        })
        .collect();

    let mut limitations = resolved.limitations.clone();
    limitations.extend(retrieval.limitations.iter().cloned());

    EvidenceManifest {
        run_id: run_id.to_string(),
        notebook_id: resolved.notebook.id.clone(),
        conversation_id: conversation_id.to_string(),
        message_id: message_id.to_string(),
        created_at: chrono::Utc::now().to_rfc3339(),
        scope: scope.clone(),
        retrieval_mode: retrieval.mode,
        entries,
        limitations,
        graph_revision,
        sources_unused: retrieval.sources_unused.clone(),
    }
}

/// The sentences a turn is told about its own scope.
///
/// Facts only, and every one of them measured: which notebook, how many of its
/// sources are in play, what the graph selection was, how the passages were
/// found, and what could not be done. A model told "you have the whole
/// collection" when it has four of nine documents will answer as though the
/// other five said nothing.
pub fn describe_scope(resolved: &ResolvedScope, retrieval: &NotebookRetrieval) -> String {
    let mut lines = Vec::new();
    lines.push(format!(
        "This question is scoped to the notebook \"{}\": {} of its {} source{} {} selected.",
        resolved.notebook.name,
        resolved.sources.len(),
        resolved.notebook.document_count,
        if resolved.notebook.document_count == 1 { "" } else { "s" },
        if resolved.sources.len() == 1 { "is" } else { "are" },
    ));

    if !resolved.focus_labels.is_empty() {
        lines.push(format!(
            "The person narrowed it from the knowledge graph to: {}.",
            resolved.focus_labels.join(", ")
        ));
    }
    for assertion in &resolved.focus_assertions {
        let standing = match (assertion.status, assertion.direction_certain) {
            (AssertionStatus::Accepted, _) => "confirmed by the person who owns this notebook",
            (_, false) => {
                "proposed by an extraction model, and its direction could not be recovered from \
                 older storage — treat which way round it runs as unverified"
            }
            _ => "proposed by an extraction model and not yet reviewed",
        };
        lines.push(format!(
            "Selected relationship: {} — {} → {}. This is {}. It is a claim about the sources, \
             not a source itself; support it from the passages below or say it is unsupported.",
            assertion.subject_label, assertion.predicate, assertion.object_label, standing
        ));
    }

    lines.push(format!(
        "The passages below were found by {}.",
        retrieval.mode.describe()
    ));

    if !retrieval.sources_unused.is_empty() {
        lines.push(format!(
            "These selected sources contributed no passage and were therefore not read for this \
             question: {}. Do not describe them as saying anything.",
            retrieval.sources_unused.join(", ")
        ));
    }

    let mut limitations = resolved.limitations.clone();
    limitations.extend(retrieval.limitations.iter().cloned());
    for limitation in &limitations {
        lines.push(format!("Limitation: {limitation}"));
    }

    if retrieval.passages.is_empty() {
        lines.push(
            "Nothing in the selected sources matched this question. Say so plainly. Do not answer \
             from general knowledge, and do not describe the sources as silent on the subject \
             unless you have read them — you have not."
                .to_string(),
        );
    }

    lines.join("\n")
}

/// The extraction revision of a document.
///
/// `extracted_at` is written whenever the page text changes, so it identifies
/// the text a passage was read from. Paired with the content address it is a
/// complete answer to "which reading of which bytes".
fn revision_of(document: &ExtractedDocument) -> String {
    document.extracted_at.clone()
}

/// How much a focused selection lifts one passage.
fn boost_for(chunk: &Chunk, scope: &ResolvedScope) -> f64 {
    if scope.focus_chunk_ids.contains(&chunk.id) {
        return CLAIM_EVIDENCE_BOOST;
    }
    if scope.focus_terms.is_empty() {
        return 1.0;
    }
    let lowered = chunk.text.to_lowercase();
    if scope.focus_terms.iter().any(|term| lowered.contains(term)) {
        FOCUS_BOOST
    } else {
        1.0
    }
}

/// A notebook passage in the shape the rest of the run already speaks.
///
/// [`SearchResult`] is what [`crate::agent_runtime::retrieval::record`] numbers,
/// what [`crate::orchestrator::runner::render_passages`] renders, and what
/// [`crate::artifacts::verifier`] resolves citations against. Producing it here
/// is what connects notebook retrieval to the existing citation machinery
/// instead of building a second one beside it.
fn to_result(document: &ExtractedDocument, chunk: &Chunk, score: f64) -> SearchResult {
    SearchResult {
        chunk_id: chunk.id.clone(),
        document_sha256: document.sha256.clone(),
        document_name: document.name.clone(),
        text: chunk.text.clone(),
        page: chunk.page,
        section_path: chunk.section_path.clone(),
        // A notebook's documents are the person's own files, read on this
        // machine. They carry no organisational classification of their own.
        classification: Classification::Internal,
        score,
        retrieval: Retrieval::Keyword,
    }
}
