//! The commands behind the Notebooks screen.
//!
//! A notebook is a named library of documents that outlives the conversation it
//! was assembled in. These commands create one, list them, and put documents in.
//!
//! ## Documents enter through the pipeline that already exists
//!
//! There is no second reader here. `notebook_add_documents` takes the same
//! [`ChatAttachment`] the composer sends — bytes over the boundary, never a path
//! — and hands it to [`crate::commands::ocr::read_attachment`], which routes it
//! to the local extractor or to vision OCR exactly as a chat attachment is
//! routed. The result is recorded in the same
//! [`crate::agent_runtime::documents::DocumentStore`], content-addressed, and
//! the notebook stores only the sha256.
//!
//! That matters for a reason beyond saving code: a document added to a notebook
//! and the same document attached to a chat are *the same record*, cut into the
//! same passages. A graph built over a notebook cites chunk ids a chat turn can
//! resolve.
//!
//! ## Where a notebook's documents live in the isolation model
//!
//! [`crate::agent_runtime::documents::ExtractedDocument::visible_to`] answers
//! "has this owner ever attached this document, and if a conversation is named,
//! in that conversation?". A notebook is not a conversation, so its sightings
//! are recorded under the synthetic id `notebook:<notebook_id>`.
//!
//! Two consequences, both wanted. A notebook's documents are visible to their
//! owner when asked without a conversation, which is how the notebook reads
//! them. And a chat run, which always asks with its own real conversation id,
//! does not silently gain access to them — importing a notebook into a chat
//! stays an explicit act rather than an ambient one.

use std::collections::BTreeMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, State};

use crate::agent_runtime::cancellation::CancelToken;
use crate::agent_runtime::documents::{NewExtraction, Sighting};
use crate::agent_runtime::stages::StageTag;
use crate::ai_engine::ocr_profile::OcrDetent;
use crate::commands::agent::DocumentsState;
use crate::commands::governance::{require_session, CurrentSession};
use crate::commands::ocr::ChatAttachment;
use crate::knowledge::graph::relations::{
    self, model_directory, RelationSidecar, RELATION_VERSION,
};
use crate::knowledge::graph::typing::{
    build_prompt, request_typing, verify, TYPING_VERSION,
};
use crate::knowledge::graph::{
    extract, EvidenceRow, GraphView, Notebook, NotebookDocument, NotebookStore, EXTRACTOR_VERSION,
};
use crate::registry::ModelRegistry;
use crate::serving::ModelServers;

/// Most files one call may add.
///
/// The reads are sequential and a scanned page can take seconds, so a hundred
/// files chosen by accident would look like the application had hung. Refusing
/// the batch is honest; a progress bar over an unbounded queue is not.
const MAX_FILES_PER_CALL: usize = 25;

/// What became of one file the caller asked to add.
///
/// A file that could not be read is reported with the reason and `added: false`
/// rather than being dropped from the response. A caller that sent five names
/// gets five rows back, so "why is my document not in the list?" is answerable
/// on the screen instead of in a log.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AddedDocument {
    pub name: String,
    /// The content address, once the file has been read. `None` when it was not.
    pub sha256: Option<String>,
    pub pages: u32,
    /// True when this call put the document into the notebook.
    ///
    /// False for a document the notebook already had — which is not a failure,
    /// and is reported separately from one so the screen can say "already here"
    /// rather than "added" or "failed".
    pub added: bool,
    /// Why this file is not in the notebook, in a sentence meant for a person.
    pub problem: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AddDocumentsRequest {
    pub notebook_id: String,
    pub attachments: Vec<ChatAttachment>,
    /// How hard to work at reading scans. Defaults to the same detent a chat
    /// turn defaults to, so a document added here reads as well as one attached
    /// to a message.
    pub ocr_detent: Option<OcrDetent>,
}

/// Every notebook the signed-in person owns.
#[tauri::command]
pub async fn notebook_list(
    store: State<'_, Arc<NotebookStore>>,
    session: State<'_, CurrentSession>,
) -> Result<Vec<Notebook>, String> {
    let signed_in = require_session(&session)?;
    store
        .list(&signed_in.user.id)
        .map_err(|error| format!("the notebooks could not be read: {error}"))
}

/// Creates a notebook owned by the signed-in person.
#[tauri::command]
pub async fn notebook_create(
    name: String,
    store: State<'_, Arc<NotebookStore>>,
    session: State<'_, CurrentSession>,
) -> Result<Notebook, String> {
    let signed_in = require_session(&session)?;
    store
        .create(&signed_in.user.id, &name)
        .map_err(|error| format!("the notebook could not be created: {error}"))
}

/// Renames a notebook the signed-in person owns.
#[tauri::command]
pub async fn notebook_rename(
    notebook_id: String,
    name: String,
    store: State<'_, Arc<NotebookStore>>,
    session: State<'_, CurrentSession>,
) -> Result<Notebook, String> {
    let signed_in = require_session(&session)?;
    store
        .rename(&notebook_id, &signed_in.user.id, &name)
        .map_err(|error| format!("the notebook could not be renamed: {error}"))
}

/// Deletes a notebook and the graph built over it.
///
/// The documents stay where they are. A notebook groups files; deleting the
/// grouping must not delete somebody's source material.
#[tauri::command]
pub async fn notebook_delete(
    notebook_id: String,
    store: State<'_, Arc<NotebookStore>>,
    session: State<'_, CurrentSession>,
) -> Result<(), String> {
    let signed_in = require_session(&session)?;
    store
        .delete(&notebook_id, &signed_in.user.id)
        .map_err(|error| format!("the notebook could not be deleted: {error}"))
}

/// Takes one document out of a notebook, with the evidence that came from it.
#[tauri::command]
pub async fn notebook_remove_document(
    notebook_id: String,
    document_sha256: String,
    store: State<'_, Arc<NotebookStore>>,
    session: State<'_, CurrentSession>,
) -> Result<(), String> {
    let signed_in = require_session(&session)?;
    store
        .remove_document(&notebook_id, &signed_in.user.id, &document_sha256)
        .map_err(|error| format!("the document could not be removed: {error}"))
}

/// The documents in one notebook.
///
/// An empty list for a notebook belonging to somebody else, because the store
/// scopes the query rather than filtering its results.
#[tauri::command]
pub async fn notebook_documents(
    notebook_id: String,
    store: State<'_, Arc<NotebookStore>>,
    session: State<'_, CurrentSession>,
) -> Result<Vec<NotebookDocument>, String> {
    let signed_in = require_session(&session)?;
    store
        .documents(&notebook_id, &signed_in.user.id)
        .map_err(|error| format!("the notebook's documents could not be read: {error}"))
}

/// Reads files and puts them in a notebook.
///
/// Sequential on purpose: the reader loads an OCR model, and two files racing
/// for it would contend for the same GPU rather than finish sooner.
#[tauri::command]
pub async fn notebook_add_documents(
    app: AppHandle,
    request: AddDocumentsRequest,
    registry: State<'_, Arc<ModelRegistry>>,
    servers: State<'_, Arc<ModelServers>>,
    documents: State<'_, DocumentsState>,
    store: State<'_, Arc<NotebookStore>>,
    session: State<'_, CurrentSession>,
) -> Result<Vec<AddedDocument>, String> {
    let signed_in = require_session(&session)?;
    let owner = &signed_in.user.id;

    if request.attachments.is_empty() {
        return Ok(Vec::new());
    }
    if request.attachments.len() > MAX_FILES_PER_CALL {
        return Err(format!(
            "That is {} files. Add up to {MAX_FILES_PER_CALL} at a time.",
            request.attachments.len()
        ));
    }

    // Ownership is established once, before any file is read. Reading first and
    // discovering afterwards that the notebook is not this person's would burn
    // minutes of OCR to produce a refusal.
    let notebook = store
        .get(&request.notebook_id, owner)
        .map_err(|error| format!("the notebook could not be read: {error}"))?
        .ok_or_else(|| "That notebook does not exist.".to_string())?;

    let detent = request.ocr_detent.unwrap_or(OcrDetent::Detailed);

    // Nothing here can be cancelled from a Stop button, because there is no run
    // to stop — the notebook screen is not a turn. `never()` is the named form
    // of that, rather than an unreachable flag that reads as working
    // cancellation.
    let cancel = CancelToken::never();
    let tag = StageTag::new(None, None, Some(format!("notebook:{}", notebook.id)));

    let mut outcomes = Vec::with_capacity(request.attachments.len());
    for attachment in &request.attachments {
        let read = match crate::commands::ocr::read_attachment(
            &app,
            registry.inner(),
            servers.inner(),
            attachment,
            detent,
            &tag,
            &cancel,
        )
        .await
        {
            Ok(read) => read,
            Err(problem) => {
                outcomes.push(AddedDocument {
                    name: attachment.name.clone(),
                    sha256: None,
                    pages: 0,
                    added: false,
                    problem: Some(problem),
                });
                continue;
            }
        };

        // Stored before the membership row. A document whose extraction is on
        // disk but whose notebook row failed is recoverable by adding it again;
        // a membership row pointing at an extraction that was never written is
        // a document the notebook lists and cannot open.
        let page_text: BTreeMap<u32, String> = read.page_text.clone();
        let recorded = documents.0.record(NewExtraction {
            sha256: read.sha256.clone(),
            name: read.name.clone(),
            kind: read.kind.clone(),
            pages: read.pages,
            truncated: read.truncated,
            page_text,
            sighting: Sighting {
                owner_user_id: owner.clone(),
                conversation_id: format!("notebook:{}", notebook.id),
                message_id: format!("notebook-add:{}", read.sha256),
                run_id: format!("notebook:{}", notebook.id),
                at: chrono::Utc::now().to_rfc3339(),
                ocr_model_id: read.ocr_model_id.clone(),
                ocr_detent: read.ocr_detent,
            },
        });

        if let Err(error) = recorded {
            outcomes.push(AddedDocument {
                name: read.name.clone(),
                sha256: Some(read.sha256.clone()),
                pages: read.pages,
                added: false,
                problem: Some(format!("the document could not be stored: {error}")),
            });
            continue;
        }

        match store.add_document(&notebook.id, owner, &read.sha256, &read.name) {
            Ok(added) => outcomes.push(AddedDocument {
                name: read.name.clone(),
                sha256: Some(read.sha256.clone()),
                pages: read.pages,
                added,
                problem: (!added).then(|| "This notebook already has that document.".to_string()),
            }),
            Err(error) => outcomes.push(AddedDocument {
                name: read.name.clone(),
                sha256: Some(read.sha256.clone()),
                pages: read.pages,
                added: false,
                problem: Some(format!("the notebook could not be updated: {error}")),
            }),
        }
    }

    Ok(outcomes)
}

/// What one build of the statistical pass actually did.
///
/// Every number is counted from work performed. `documents_skipped` is the
/// resumability made visible: a second build over an unchanged notebook does
/// nothing and says so, rather than looking like it worked and produced the
/// same graph by coincidence.
#[derive(Debug, Clone, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BuildOutcome {
    pub documents_total: u32,
    pub documents_built: u32,
    pub documents_skipped: u32,
    /// Documents in the notebook whose extraction could not be read back.
    pub documents_unreadable: u32,
    pub chunks: u32,
    pub candidates_found: u32,
    pub nodes_kept: u32,
    pub dropped_rare: u32,
    pub dropped_generic: u32,
    /// Named files that could not be read, so the screen can say which.
    pub problems: Vec<String>,
}

/// Builds the statistical graph over a notebook's documents.
///
/// Resumable: documents already processed at this extractor version are skipped,
/// so an interrupted build is finished by running it again. Bumping
/// `EXTRACTOR_VERSION` invalidates every unit and forces a full rebuild, which
/// is what makes a change to the algorithm safe to ship.
///
/// No model is involved. This is the cheap pass; see
/// [`crate::knowledge::graph::statistical`] for why it exists.
#[tauri::command]
pub async fn notebook_build_graph(
    notebook_id: String,
    rebuild: Option<bool>,
    documents: State<'_, DocumentsState>,
    store: State<'_, Arc<NotebookStore>>,
    session: State<'_, CurrentSession>,
) -> Result<BuildOutcome, String> {
    let signed_in = require_session(&session)?;
    let owner = &signed_in.user.id;

    let notebook = store
        .get(&notebook_id, owner)
        .map_err(|error| format!("the notebook could not be read: {error}"))?
        .ok_or_else(|| "That notebook does not exist.".to_string())?;

    let members = store
        .documents(&notebook.id, owner)
        .map_err(|error| format!("the notebook's documents could not be read: {error}"))?;

    let already = if rebuild.unwrap_or(false) {
        Default::default()
    } else {
        store
            .completed_units(&notebook.id, "statistical", EXTRACTOR_VERSION)
            .map_err(|error| format!("the build state could not be read: {error}"))?
    };

    let mut outcome = BuildOutcome {
        documents_total: members.len() as u32,
        ..Default::default()
    };

    for member in &members {
        if already.contains(&member.document_sha256) {
            outcome.documents_skipped += 1;
            continue;
        }

        // Asked for without a conversation: a notebook's documents were sighted
        // under the notebook's own synthetic id, and `visible_to` with `None`
        // answers "has this owner ever attached it", which is the right
        // question here.
        let document = match documents.0.get(&member.document_sha256, owner, None) {
            Ok(Some(document)) => document,
            Ok(None) => {
                outcome.documents_unreadable += 1;
                outcome.problems.push(format!(
                    "{}: the extraction is missing, so it contributed nothing",
                    member.document_name
                ));
                continue;
            }
            Err(error) => {
                outcome.documents_unreadable += 1;
                outcome
                    .problems
                    .push(format!("{}: {error}", member.document_name));
                continue;
            }
        };

        // The chunker already recorded each passage's page. Passing a lookup
        // rather than re-deriving it keeps the citation the graph stores
        // identical to the one a chat turn would resolve.
        let pages: std::collections::BTreeMap<String, u32> = document
            .chunks
            .iter()
            .map(|chunk| (chunk.id.clone(), chunk.page))
            .collect();
        let page_of = |chunk_id: &str| pages.get(chunk_id).copied().unwrap_or(0);

        let draft = extract(&document.chunks);
        outcome.chunks += draft.stats.chunks;
        outcome.candidates_found += draft.stats.candidates_found;
        outcome.nodes_kept += draft.stats.kept;
        outcome.dropped_rare += draft.stats.dropped_rare;
        outcome.dropped_generic += draft.stats.dropped_generic;

        match store.store_document_graph(
            &notebook.id,
            owner,
            &member.document_sha256,
            &page_of,
            &draft,
            EXTRACTOR_VERSION,
        ) {
            Ok(()) => outcome.documents_built += 1,
            Err(error) => {
                outcome.documents_unreadable += 1;
                outcome
                    .problems
                    .push(format!("{}: {error}", member.document_name));
            }
        }
    }

    Ok(outcome)
}

/// A view of the notebook's graph.
///
/// `focus` gives the local view — one node and its neighbours out to `depth`.
/// That is the default the interface should use: a global graph past a few
/// hundred nodes is an unreadable tangle, and the local view stays legible at
/// any size.
///
/// `with_documents` adds the notebook's files as nodes, joined to the terms
/// found in them. Defaults to on, because the relation a person opens a
/// *notebook* graph to see is which file said what; passing `false` gives the
/// term-only graph for when the files are in the way.
///
/// `documentSha256` narrows the whole view to one file, counts included. That
/// is a different question from focusing on the file's node: a focus walks out
/// from it across the notebook and reaches terms other files contributed, which
/// is right when the question is "what does this connect to". Scoping asks
/// "what is in this file", and the answer must not include anything else.
#[tauri::command]
pub async fn notebook_graph(
    notebook_id: String,
    document_sha256: Option<String>,
    focus: Option<String>,
    depth: Option<u32>,
    min_weight: Option<u32>,
    with_documents: Option<bool>,
    store: State<'_, Arc<NotebookStore>>,
    session: State<'_, CurrentSession>,
) -> Result<GraphView, String> {
    let signed_in = require_session(&session)?;
    store
        .graph(
            &notebook_id,
            &signed_in.user.id,
            document_sha256.as_deref(),
            focus.as_deref(),
            depth.unwrap_or(1).clamp(1, 3),
            min_weight.unwrap_or(1).max(1),
            with_documents.unwrap_or(true),
        )
        .map_err(|error| format!("the graph could not be read: {error}"))
}

/// The passages behind one node — the answer to "why do you think that?".
#[tauri::command]
pub async fn notebook_node_evidence(
    notebook_id: String,
    node_id: String,
    store: State<'_, Arc<NotebookStore>>,
    session: State<'_, CurrentSession>,
) -> Result<Vec<EvidenceRow>, String> {
    let signed_in = require_session(&session)?;
    store
        .node_evidence(&notebook_id, &signed_in.user.id, &node_id)
        .map_err(|error| format!("the evidence could not be read: {error}"))
}

/// Most passages sent in one typing request.
///
/// A whole document would overflow the window of the 7B-class model these
/// machines run. Bounded here rather than hoping: the request that would not fit
/// is the one that returns nothing after four minutes.
const MAX_CHUNKS_PER_TYPING_REQUEST: usize = 12;

/// Most terms offered for classification in one request.
const MAX_TERMS_PER_TYPING_REQUEST: usize = 60;

/// What one run of the typing pass did.
///
/// The drop counts are reported, not hidden. A pass that kept nine proposals out
/// of two hundred is telling you the model is wrong for the job or the document
/// is scanned noise, and a screen showing only the nine would read as a small
/// graph rather than a failing extraction.
#[derive(Debug, Clone, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TypingOutcome {
    pub documents_total: u32,
    pub documents_typed: u32,
    pub documents_skipped: u32,
    pub documents_failed: u32,
    pub proposed_nodes: u32,
    pub proposed_edges: u32,
    pub kept_nodes: u32,
    pub kept_edges: u32,
    pub dropped_unknown_type: u32,
    pub dropped_unknown_term: u32,
    pub dropped_uncited: u32,
    /// Claims whose quote was not in the passage they cited. Fabrication,
    /// caught — the number worth watching.
    pub dropped_misquoted: u32,
    /// Distinct terms now carrying a type, and the total. Both, always.
    pub typed_terms: u32,
    pub total_terms: u32,
    pub problems: Vec<String>,
}

/// Types a notebook's graph using whichever model is already loaded.
///
/// ## It uses a warm server rather than starting one
///
/// Typing is background work over an index, not a turn somebody is waiting on.
/// Admitting a model to VRAM can evict the one the user is chatting with and
/// then read gigabytes off disk, so this pass takes an endpoint that is already
/// running and refuses plainly when there is none. "Open a chat first" is a
/// better outcome than silently evicting their model.
///
/// ## Resumable, and honest when it fails
///
/// One document is one unit. A document typed at this `TYPING_VERSION` is
/// skipped, so an interrupted pass is finished by running it again. A document
/// the model fails on is counted and named; the rest of the pass continues, and
/// the graph keeps whatever the statistical pass gave it.
#[tauri::command]
pub async fn notebook_type_graph(
    app: AppHandle,
    notebook_id: String,
    documents: State<'_, DocumentsState>,
    servers: State<'_, Arc<ModelServers>>,
    store: State<'_, Arc<NotebookStore>>,
    session: State<'_, CurrentSession>,
) -> Result<TypingOutcome, String> {
    let signed_in = require_session(&session)?;
    let owner = &signed_in.user.id;

    let notebook = store
        .get(&notebook_id, owner)
        .map_err(|error| format!("the notebook could not be read: {error}"))?
        .ok_or_else(|| "That notebook does not exist.".to_string())?;

    let endpoint = servers.running_endpoints().into_iter().next().ok_or_else(|| {
        "No model is loaded, so nothing can be typed. Ask a question in a chat first — that \
         loads a model — then run this again."
            .to_string()
    })?;

    let members = store
        .documents(&notebook.id, owner)
        .map_err(|error| format!("the notebook's documents could not be read: {error}"))?;
    let already = store
        .completed_units(&notebook.id, "typing", TYPING_VERSION)
        .map_err(|error| format!("the build state could not be read: {error}"))?;

    let mut outcome = TypingOutcome {
        documents_total: members.len() as u32,
        ..Default::default()
    };

    for (index, member) in members.iter().enumerate() {
        if already.contains(&member.document_sha256) {
            outcome.documents_skipped += 1;
            continue;
        }

        let terms = store
            .document_terms(&notebook.id, owner, &member.document_sha256)
            .unwrap_or_default();
        if terms.is_empty() {
            // Nothing for the model to classify. Not a failure: the statistical
            // pass may simply not have run for this document yet.
            outcome.documents_skipped += 1;
            continue;
        }

        let Ok(Some(document)) = documents.0.get(&member.document_sha256, owner, None) else {
            outcome.documents_failed += 1;
            outcome
                .problems
                .push(format!("{}: the extraction could not be read", member.document_name));
            continue;
        };

        // Only the passages the chosen terms actually came from, so the window
        // is spent on text that can support a claim.
        let wanted: std::collections::BTreeSet<String> =
            terms.iter().take(MAX_TERMS_PER_TYPING_REQUEST).map(|(n, _)| n.clone()).collect();
        let chunk_texts: Vec<(String, String)> = document
            .chunks
            .iter()
            .filter(|chunk| {
                let lowered = chunk.text.to_lowercase();
                wanted.iter().any(|term| lowered.contains(term))
            })
            .take(MAX_CHUNKS_PER_TYPING_REQUEST)
            .map(|chunk| (chunk.id.clone(), chunk.text.clone()))
            .collect();
        if chunk_texts.is_empty() {
            outcome.documents_skipped += 1;
            continue;
        }

        // Emitted before the request, because the request is the slow part and a
        // stage that appears only afterwards tells nobody anything. The counts
        // are measured, never interpolated — see `agent_runtime::stages`.
        let _ = app.emit(
            "notebook:graph_progress",
            serde_json::json!({
                "notebookId": notebook.id,
                "documentsDone": index as u32,
                "documentsTotal": outcome.documents_total,
                "document": member.document_name,
                "chunks": chunk_texts.len() as u32,
                "terms": wanted.len() as u32,
            }),
        );

        let labels: Vec<String> = terms
            .iter()
            .take(MAX_TERMS_PER_TYPING_REQUEST)
            .map(|(_, label)| label.clone())
            .collect();
        let prompt = build_prompt(&member.document_name, &labels, &chunk_texts);

        let proposal = match request_typing(
            &endpoint.base_url,
            &endpoint.served_model_id,
            &prompt,
        )
        .await
        {
            Ok(proposal) => proposal,
            Err(problem) => {
                outcome.documents_failed += 1;
                outcome
                    .problems
                    .push(format!("{}: {problem}", member.document_name));
                continue;
            }
        };

        let known: BTreeMap<String, String> = terms.into_iter().collect();
        let chunk_map: BTreeMap<String, String> = chunk_texts.into_iter().collect();
        let verdict = verify(&proposal, &known, &chunk_map);

        outcome.proposed_nodes += verdict.stats.proposed_nodes;
        outcome.proposed_edges += verdict.stats.proposed_edges;
        outcome.kept_nodes += verdict.stats.kept_nodes;
        outcome.kept_edges += verdict.stats.kept_edges;
        outcome.dropped_unknown_type += verdict.stats.dropped_unknown_type;
        outcome.dropped_unknown_term += verdict.stats.dropped_unknown_term;
        outcome.dropped_uncited += verdict.stats.dropped_uncited;
        outcome.dropped_misquoted += verdict.stats.dropped_misquoted;

        match store.store_document_typing(
            &notebook.id,
            owner,
            &member.document_sha256,
            &verdict,
            TYPING_VERSION,
        ) {
            Ok(()) => outcome.documents_typed += 1,
            Err(error) => {
                outcome.documents_failed += 1;
                outcome
                    .problems
                    .push(format!("{}: {error}", member.document_name));
            }
        }
    }

    if let Ok((typed, total)) = store.typing_coverage(&notebook.id, owner) {
        outcome.typed_terms = typed;
        outcome.total_terms = total;
    }

    Ok(outcome)
}

/// Most passages sent to the relation extractor in one request.
///
/// Larger than the typing budget because REBEL reads one passage at a time
/// rather than fitting them all into a shared context window - there is no
/// prompt to overflow. The cap is about how long a user waits for one document,
/// not about what fits.
const MAX_CHUNKS_PER_RELATION_REQUEST: usize = 40;

/// What one run of the relation pass did.
///
/// Every drop counted separately, because the counts say different things.
/// `dropped_offdomain` is high on a healthy run over industrial documents -
/// REBEL is proposing Wikipedia relations and the allowlist is doing its job.
/// `dropped_misquoted` being anything but low is the number that means trouble.
#[derive(Debug, Clone, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RelationOutcome {
    pub documents_total: u32,
    pub documents_named: u32,
    pub documents_skipped: u32,
    pub documents_failed: u32,
    pub proposed: u32,
    pub kept: u32,
    pub dropped_uncited: u32,
    pub dropped_misquoted: u32,
    pub dropped_unknown_term: u32,
    pub dropped_unknown_edge: u32,
    pub dropped_offdomain: u32,
    pub problems: Vec<String>,
}

/// Names a notebook's edges with Babelscape/rebel-large.
///
/// ## It starts its own model, and that is the point
///
/// Unlike [`notebook_type_graph`], this pass does not wait for a warm
/// llama-server. REBEL is 400M parameters on CPU, so starting it costs seconds
/// and no VRAM, and nothing the user is chatting with gets evicted. The sidecar
/// lives for the length of the pass and is killed when it ends.
///
/// ## Resumable, and honest when it fails
///
/// One document is one unit, recorded at [`RELATION_VERSION`]. A document
/// already done is skipped, so an interrupted pass is finished by running it
/// again. A document the extractor fails on is counted and named; the rest of
/// the pass continues, and the graph keeps whatever the earlier passes gave it.
#[tauri::command]
pub async fn notebook_extract_relations(
    app: AppHandle,
    notebook_id: String,
    documents: State<'_, DocumentsState>,
    registry: State<'_, Arc<ModelRegistry>>,
    store: State<'_, Arc<NotebookStore>>,
    session: State<'_, CurrentSession>,
) -> Result<RelationOutcome, String> {
    let signed_in = require_session(&session)?;
    let owner = &signed_in.user.id;

    let notebook = store
        .get(&notebook_id, owner)
        .map_err(|error| format!("the notebook could not be read: {error}"))?
        .ok_or_else(|| "That notebook does not exist.".to_string())?;

    let members = store
        .documents(&notebook.id, owner)
        .map_err(|error| format!("the notebook documents could not be read: {error}"))?;
    let already = store
        .completed_units(&notebook.id, "relations", RELATION_VERSION)
        .map_err(|error| format!("the build state could not be read: {error}"))?;

    let mut outcome = RelationOutcome {
        documents_total: members.len() as u32,
        ..Default::default()
    };

    // Nothing left to do. Checked before the sidecar starts, so a second run
    // over a finished notebook does not pay 1.6 GB of model load to discover it
    // has no work.
    if members
        .iter()
        .all(|member| already.contains(&member.document_sha256))
    {
        outcome.documents_skipped = outcome.documents_total;
        return Ok(outcome);
    }

    let model_dir = model_directory(registry.models_dir());
    let mut sidecar = RelationSidecar::spawn(&model_dir).map_err(|error| {
        format!(
            "The relation extractor could not start. {error} Install \
             Babelscape/rebel-large into the model library, then run this again."
        )
    })?;

    for (index, member) in members.iter().enumerate() {
        if already.contains(&member.document_sha256) {
            outcome.documents_skipped += 1;
            continue;
        }

        let terms: BTreeMap<String, String> = store
            .document_terms(&notebook.id, owner, &member.document_sha256)
            .unwrap_or_default()
            .into_iter()
            .collect();
        let edges = store
            .document_edges(&notebook.id, owner, &member.document_sha256)
            .unwrap_or_default();
        if terms.is_empty() || edges.is_empty() {
            // No edges means nothing to name. Not a failure: the statistical
            // pass may simply not have run for this document yet.
            outcome.documents_skipped += 1;
            continue;
        }

        let Ok(Some(document)) = documents.0.get(&member.document_sha256, owner, None) else {
            outcome.documents_failed += 1;
            outcome.problems.push(format!(
                "{}: the extraction could not be read",
                member.document_name
            ));
            continue;
        };

        // Only passages mentioning at least two known terms. A passage naming
        // one thing cannot support a relation between two, so sending it spends
        // seconds to receive nothing.
        let chunk_texts: Vec<(String, String)> = document
            .chunks
            .iter()
            .filter(|chunk| {
                let lowered = chunk.text.to_lowercase();
                terms
                    .keys()
                    .filter(|term| lowered.contains(*term))
                    .take(2)
                    .count()
                    == 2
            })
            .take(MAX_CHUNKS_PER_RELATION_REQUEST)
            .map(|chunk| (chunk.id.clone(), chunk.text.clone()))
            .collect();
        if chunk_texts.is_empty() {
            outcome.documents_skipped += 1;
            continue;
        }

        // Emitted before the request, because the request is the slow part and
        // a stage that appears only afterwards tells nobody anything. Counts are
        // measured, never interpolated - see `agent_runtime::stages`.
        let _ = app.emit(
            "notebook:graph_progress",
            serde_json::json!({
                "notebookId": notebook.id,
                "documentsDone": index as u32,
                "documentsTotal": outcome.documents_total,
                "document": member.document_name,
                "chunks": chunk_texts.len() as u32,
                "pass": "relations",
            }),
        );

        let triplets = match sidecar.extract(&chunk_texts) {
            Ok(triplets) => triplets,
            Err(problem) => {
                outcome.documents_failed += 1;
                outcome
                    .problems
                    .push(format!("{}: {problem}", member.document_name));
                continue;
            }
        };

        let chunk_map: BTreeMap<String, String> = chunk_texts.into_iter().collect();
        let verdict = relations::verify(&triplets, &terms, &chunk_map, &edges);

        outcome.proposed += verdict.stats.proposed;
        outcome.kept += verdict.stats.kept;
        outcome.dropped_uncited += verdict.stats.dropped_uncited;
        outcome.dropped_misquoted += verdict.stats.dropped_misquoted;
        outcome.dropped_unknown_term += verdict.stats.dropped_unknown_term;
        outcome.dropped_unknown_edge += verdict.stats.dropped_unknown_edge;
        outcome.dropped_offdomain += verdict.stats.dropped_offdomain;

        match store.store_document_relations(
            &notebook.id,
            owner,
            &member.document_sha256,
            &verdict,
            RELATION_VERSION,
        ) {
            Ok(()) => outcome.documents_named += 1,
            Err(error) => {
                outcome.documents_failed += 1;
                outcome
                    .problems
                    .push(format!("{}: {error}", member.document_name));
            }
        }
    }

    Ok(outcome)
}

/// A subgraph rendered for a chat turn.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RenderedSubgraph {
    /// The filename it will arrive under, so the context meter row is
    /// recognisable as "the thing I imported from my notebook".
    pub name: String,
    pub markdown: String,
    pub node_count: u32,
    pub edge_count: u32,
}

/// Renders a selection of terms into text that can be attached to a chat turn.
///
/// The caller attaches the result as an ordinary document, which is the whole
/// trick: the existing attachment path already gives it a content address,
/// chunking, a token budget, a row in the context meter, and pinning by that
/// address. None of that had to be built for the graph.
#[tauri::command]
pub async fn notebook_render_subgraph(
    notebook_id: String,
    node_ids: Vec<String>,
    store: State<'_, Arc<NotebookStore>>,
    session: State<'_, CurrentSession>,
) -> Result<RenderedSubgraph, String> {
    let signed_in = require_session(&session)?;
    let owner = &signed_in.user.id;

    if node_ids.is_empty() {
        return Err("Select at least one term first.".to_string());
    }

    let notebook = store
        .get(&notebook_id, owner)
        .map_err(|error| format!("the notebook could not be read: {error}"))?
        .ok_or_else(|| "That notebook does not exist.".to_string())?;

    let whole = store
        .graph(&notebook_id, owner, None, None, 1, 1, false)
        .map_err(|error| format!("the graph could not be read: {error}"))?;

    let wanted: std::collections::BTreeSet<&String> = node_ids.iter().collect();
    let chosen: Vec<_> = whole
        .nodes
        .iter()
        .filter(|node| wanted.contains(&node.id))
        .collect();
    if chosen.is_empty() {
        return Err("Those terms are not in this notebook's graph.".to_string());
    }

    let label_of = |id: &str| {
        chosen
            .iter()
            .find(|node| node.id == id)
            .map(|node| node.label.clone())
            .unwrap_or_else(|| id.to_string())
    };

    // Only links whose *both* ends were chosen. An edge reaching out to a term
    // the person did not select would name something the text never introduces.
    let edges: Vec<crate::knowledge::graph::render::RenderEdge> = whole
        .edges
        .iter()
        .filter(|edge| wanted.contains(&edge.source) && wanted.contains(&edge.target))
        .map(|edge| {
            (
                label_of(&edge.source),
                label_of(&edge.target),
                edge.weight,
                edge.relation.clone(),
            )
        })
        .collect();

    let names: BTreeMap<String, String> = store
        .documents(&notebook_id, owner)
        .map_err(|error| format!("the notebook's documents could not be read: {error}"))?
        .into_iter()
        .map(|document| (document.document_sha256, document.document_name))
        .collect();

    let mut citations: Vec<crate::knowledge::graph::render::RenderCitation> = Vec::new();
    for node in &chosen {
        let rows = store
            .node_evidence(&notebook_id, owner, &node.id)
            .unwrap_or_default();
        citations.push((
            node.label.clone(),
            rows.into_iter()
                .map(|row| {
                    (
                        names
                            .get(&row.document_sha256)
                            .cloned()
                            .unwrap_or_else(|| "an unnamed document".to_string()),
                        row.page,
                    )
                })
                .collect(),
        ));
    }

    let nodes: Vec<crate::knowledge::graph::render::RenderNode> = chosen
        .iter()
        .map(|node| (node.label.clone(), node.node_type.clone(), node.occurrences))
        .collect();

    Ok(RenderedSubgraph {
        name: format!("{} — selected terms.md", notebook.name),
        markdown: crate::knowledge::graph::render::render_markdown(
            &notebook.name,
            &nodes,
            &edges,
            &citations,
        ),
        node_count: nodes.len() as u32,
        edge_count: edges.len() as u32,
    })
}
