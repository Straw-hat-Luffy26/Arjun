//! The commands behind the notebook research workspace.
//!
//! Reading sources, saving notes, opening a citation at the passage it points
//! at, and reviewing what the extraction passes claimed.
//!
//! ## Why these are separate from `commands::notebook`
//!
//! That file is about getting documents *into* a notebook and building a graph
//! over them. This one is about working with what is there. They share a store
//! and nothing else, and keeping them in one file would make a thousand-line
//! module two thousand.
//!
//! ## The rule every command here follows
//!
//! `require_session`, then a store method that takes the owner id and puts it in
//! the SQL. Never a check followed by an unscoped query — see
//! [`crate::knowledge::graph`] for the ten deleted commands that made that
//! mistake and why it is invisible in review.

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tauri::State;

use crate::agent_runtime::conversations::MessageRole;
use crate::commands::agent::DocumentsState;
use crate::commands::conversations::ConversationsState;
use crate::commands::governance::{require_session, CurrentSession};
use crate::knowledge::graph::notes::{Note, NoteKind};
use crate::knowledge::graph::research::EvidenceManifest;
use crate::knowledge::graph::{Assertion, AssertionStatus};
use crate::knowledge::NotebookStore;

// ── Notebook-scoped conversations ────────────────────────────────────────

/// Records that a conversation belongs to a notebook.
///
/// Called when a notebook's chat starts a thread. Without it the thread is
/// indistinguishable from any other, and reopening the notebook tomorrow could
/// not restore what was said in it.
#[tauri::command]
pub async fn notebook_bind_conversation(
    notebook_id: String,
    conversation_id: String,
    store: State<'_, Arc<NotebookStore>>,
    session: State<'_, CurrentSession>,
) -> Result<(), String> {
    let signed_in = require_session(&session)?;
    store
        .bind_conversation(&notebook_id, &signed_in.user.id, &conversation_id)
        .map_err(|error| format!("the conversation could not be linked to the notebook: {error}"))
}

/// A notebook's threads, newest first.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NotebookThread {
    pub conversation_id: String,
    pub title: String,
    pub last_activity_at: String,
    pub message_count: u32,
}

/// The conversations started from one notebook.
///
/// Bindings whose conversation has since been deleted are skipped rather than
/// returned as empty rows: a thread that cannot be opened is not a thread.
#[tauri::command]
pub async fn notebook_threads(
    notebook_id: String,
    store: State<'_, Arc<NotebookStore>>,
    conversations: State<'_, ConversationsState>,
    session: State<'_, CurrentSession>,
) -> Result<Vec<NotebookThread>, String> {
    let signed_in = require_session(&session)?;
    let ids = store
        .notebook_conversations(&notebook_id, &signed_in.user.id)
        .map_err(|error| format!("the notebook's conversations could not be read: {error}"))?;

    let mut threads = Vec::new();
    for id in ids {
        let Ok(Some(conversation)) = conversations.0.get(&id, Some(&signed_in.user.id)) else {
            continue;
        };
        threads.push(NotebookThread {
            conversation_id: conversation.id,
            title: conversation.title,
            last_activity_at: conversation.last_activity_at,
            // System messages are scaffolding, not turns somebody had.
            message_count: conversation
                .messages
                .iter()
                .filter(|message| !matches!(message.role, MessageRole::System))
                .count() as u32,
        });
    }
    Ok(threads)
}

// ── Evidence, citations and the source reader ────────────────────────────

/// The manifest for one answer, if that turn recorded one.
///
/// Returns `None` rather than an error for a turn that was not notebook-scoped,
/// which is the ordinary case for general chat.
#[tauri::command]
pub async fn notebook_turn_evidence(
    conversation_id: String,
    message_id: String,
    store: State<'_, Arc<NotebookStore>>,
    session: State<'_, CurrentSession>,
) -> Result<Option<EvidenceManifest>, String> {
    let signed_in = require_session(&session)?;
    store
        .research_turn(&signed_in.user.id, &conversation_id, &message_id)
        .map_err(|error| format!("the evidence for this answer could not be read: {error}"))
}

/// Why a citation cannot be opened, when it cannot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum CitationState {
    /// The passage is there and the text still matches.
    Available,
    /// The source has been taken out of the notebook since the answer was
    /// written. The quote is still shown, from the manifest.
    SourceRemoved,
    /// The source is still here but has been re-read since, so the passage this
    /// citation names may no longer be the text that was cited.
    SourceChanged,
    /// The source is in the notebook and its extracted text cannot be read.
    Unavailable,
}

/// One citation, resolved for the reader.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ResolvedCitation {
    pub marker: u32,
    pub state: CitationState,
    pub document_sha256: String,
    pub document_name: String,
    pub page: u32,
    pub section_path: Vec<String>,
    /// The passage as the answer used it. Always present — it comes from the
    /// manifest, which does not change when the sources do.
    pub quote: String,
    /// The page as it reads now, when it can still be read. `None` when the
    /// source is gone.
    pub page_text: Option<String>,
    /// Where in `page_text` the quote begins, when it is still there verbatim.
    ///
    /// `None` when the text has moved or changed — which is a real finding and
    /// is shown as one, rather than highlighting a nearby paragraph and
    /// implying it is the same sentence.
    pub highlight_start: Option<u32>,
    pub highlight_length: Option<u32>,
    /// A sentence for the reader when the state is not `Available`.
    pub problem: Option<String>,
}

/// Opens one citation of one answer at the passage it points at.
///
/// ## What "opening" means when the source has moved
///
/// The manifest is authoritative about what the answer used, and it is not
/// rewritten when a notebook changes. So this resolves in two steps: the quote
/// always comes back, because the manifest holds it; and the *current* page is
/// fetched separately and may be missing, changed, or unreadable. The two are
/// reported as different states, because "this citation was never real" and
/// "this citation was real and the document has since been removed" call for
/// completely different reactions from a person.
#[tauri::command]
pub async fn notebook_open_citation(
    conversation_id: String,
    message_id: String,
    marker: u32,
    store: State<'_, Arc<NotebookStore>>,
    documents: State<'_, DocumentsState>,
    session: State<'_, CurrentSession>,
) -> Result<ResolvedCitation, String> {
    let signed_in = require_session(&session)?;
    let owner = &signed_in.user.id;

    let manifest = store
        .research_turn(owner, &conversation_id, &message_id)
        .map_err(|error| format!("the evidence for this answer could not be read: {error}"))?
        .ok_or_else(|| {
            "This answer has no recorded evidence, so its citations cannot be opened. It was \
             written before evidence was kept, or it was not a notebook question."
                .to_string()
        })?;

    let entry = manifest
        .entries
        .iter()
        .find(|entry| entry.marker == marker)
        .ok_or_else(|| {
            format!(
                "This answer cited [E{marker}], and only {} passages were retrieved for it. The \
                 citation does not point at anything.",
                manifest.entries.len()
            )
        })?;

    // Still in the notebook?
    let members = store
        .documents(&manifest.notebook_id, owner)
        .map_err(|error| format!("the notebook's documents could not be read: {error}"))?;
    let still_a_member = members
        .iter()
        .any(|member| member.document_sha256 == entry.document_sha256);

    if !still_a_member {
        return Ok(ResolvedCitation {
            marker,
            state: CitationState::SourceRemoved,
            document_sha256: entry.document_sha256.clone(),
            document_name: entry.document_name.clone(),
            page: entry.page,
            section_path: entry.section_path.clone(),
            quote: entry.quote.clone(),
            page_text: None,
            highlight_start: None,
            highlight_length: None,
            problem: Some(format!(
                "\"{}\" has been taken out of this notebook since this answer was written. The \
                 passage it cited is shown as it was used.",
                entry.document_name
            )),
        });
    }

    let conversation = format!("notebook:{}", manifest.notebook_id);
    let document = match documents
        .0
        .get(&entry.document_sha256, owner, Some(&conversation))
    {
        Ok(Some(document)) => document,
        _ => {
            return Ok(ResolvedCitation {
                marker,
                state: CitationState::Unavailable,
                document_sha256: entry.document_sha256.clone(),
                document_name: entry.document_name.clone(),
                page: entry.page,
                section_path: entry.section_path.clone(),
                quote: entry.quote.clone(),
                page_text: None,
                highlight_start: None,
                highlight_length: None,
                problem: Some(format!(
                    "The extracted text of \"{}\" could not be read, so the page cannot be \
                     shown. The passage it cited is shown as it was used.",
                    entry.document_name
                )),
            })
        }
    };

    let page_text = document
        .page_text
        .iter()
        .find(|page| page.page == entry.page)
        .map(|page| page.text.clone());

    // The quote's position on the page as it reads now. An exact `find`: a
    // fuzzy match would highlight something that is not what was cited.
    let (start, length) = match &page_text {
        Some(text) => match text.find(entry.quote.trim()) {
            Some(index) => (
                Some(text[..index].chars().count() as u32),
                Some(entry.quote.trim().chars().count() as u32),
            ),
            None => (None, None),
        },
        None => (None, None),
    };

    let changed = document.extracted_at != entry.source_revision;
    let state = if changed {
        CitationState::SourceChanged
    } else {
        CitationState::Available
    };
    let problem = if changed {
        Some(format!(
            "\"{}\" has been read again since this answer was written, so the page below is not \
             necessarily the text that was cited. The passage as used is shown above it.",
            entry.document_name
        ))
    } else if start.is_none() && page_text.is_some() {
        Some(
            "The cited passage was not found verbatim on this page, so nothing is highlighted."
                .to_string(),
        )
    } else {
        None
    };

    Ok(ResolvedCitation {
        marker,
        state,
        document_sha256: entry.document_sha256.clone(),
        document_name: entry.document_name.clone(),
        page: entry.page,
        section_path: entry.section_path.clone(),
        quote: entry.quote.clone(),
        page_text,
        highlight_start: start,
        highlight_length: length,
        problem,
    })
}

/// One page of a source, for the reader.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SourcePage {
    pub document_sha256: String,
    pub document_name: String,
    pub page: u32,
    pub total_pages: u32,
    pub text: String,
    /// True when the reader stopped early when this file was first read, so
    /// pages past this point may not exist in the store at all.
    pub source_truncated: bool,
    /// How the text on this page was produced — `pdf-text`, `pdf-scan`,
    /// `image`, `docx`. Shown so a reader knows whether they are looking at
    /// extracted text or at OCR output.
    pub extraction_kind: String,
    pub extracted_at: String,
}

/// Reads one page of a notebook source.
///
/// Owner- *and* notebook-scoped: the document is fetched under the notebook's
/// own synthetic conversation id, so a content address belonging to a document
/// this person has in a different notebook is not readable through this
/// notebook.
#[tauri::command]
pub async fn notebook_source_page(
    notebook_id: String,
    document_sha256: String,
    page: u32,
    store: State<'_, Arc<NotebookStore>>,
    documents: State<'_, DocumentsState>,
    session: State<'_, CurrentSession>,
) -> Result<SourcePage, String> {
    let signed_in = require_session(&session)?;
    let owner = &signed_in.user.id;

    let members = store
        .documents(&notebook_id, owner)
        .map_err(|error| format!("the notebook's documents could not be read: {error}"))?;
    let member = members
        .iter()
        .find(|member| member.document_sha256 == document_sha256)
        .ok_or_else(|| "That source is not in this notebook.".to_string())?;

    let conversation = format!("notebook:{notebook_id}");
    let document = documents
        .0
        .get(&document_sha256, owner, Some(&conversation))
        .map_err(|error| format!("the source could not be read: {error}"))?
        .ok_or_else(|| {
            format!(
                "The extracted text of \"{}\" is missing. It was added to this notebook but its \
                 extraction is not on disk — remove it and add it again.",
                member.document_name
            )
        })?;

    let page = page.max(1);
    let text = document
        .page_text
        .iter()
        .find(|held| held.page == page)
        .map(|held| held.text.clone())
        .unwrap_or_default();

    Ok(SourcePage {
        document_sha256: document.sha256.clone(),
        document_name: document.name.clone(),
        page,
        total_pages: document.pages.max(document.last_page_with_text()),
        text,
        source_truncated: document.truncated,
        extraction_kind: document.kind.clone(),
        extracted_at: document.extracted_at.clone(),
    })
}

// ── Notes and reports ────────────────────────────────────────────────────

#[tauri::command]
pub async fn notebook_notes(
    notebook_id: String,
    store: State<'_, Arc<NotebookStore>>,
    session: State<'_, CurrentSession>,
) -> Result<Vec<Note>, String> {
    let signed_in = require_session(&session)?;
    store
        .notes(&notebook_id, &signed_in.user.id)
        .map_err(|error| format!("the notebook's notes could not be read: {error}"))
}

#[tauri::command]
pub async fn notebook_create_note(
    notebook_id: String,
    title: String,
    body: Option<String>,
    store: State<'_, Arc<NotebookStore>>,
    session: State<'_, CurrentSession>,
) -> Result<Note, String> {
    let signed_in = require_session(&session)?;
    store
        .create_note(
            &notebook_id,
            &signed_in.user.id,
            &title,
            body.as_deref().unwrap_or(""),
            NoteKind::Written,
            None,
            None,
            None,
        )
        .map_err(|error| format!("the note could not be created: {error}"))
}

#[tauri::command]
pub async fn notebook_update_note(
    notebook_id: String,
    note_id: String,
    title: Option<String>,
    body: Option<String>,
    store: State<'_, Arc<NotebookStore>>,
    session: State<'_, CurrentSession>,
) -> Result<Note, String> {
    let signed_in = require_session(&session)?;
    store
        .update_note(
            &notebook_id,
            &signed_in.user.id,
            &note_id,
            title.as_deref(),
            body.as_deref(),
        )
        .map_err(|error| format!("the note could not be saved: {error}"))
}

#[tauri::command]
pub async fn notebook_delete_note(
    notebook_id: String,
    note_id: String,
    store: State<'_, Arc<NotebookStore>>,
    session: State<'_, CurrentSession>,
) -> Result<(), String> {
    let signed_in = require_session(&session)?;
    store
        .delete_note(&notebook_id, &signed_in.user.id, &note_id)
        .map_err(|error| format!("the note could not be deleted: {error}"))
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SaveAnswerRequest {
    pub notebook_id: String,
    pub conversation_id: String,
    pub message_id: String,
    pub title: String,
    /// `answer`, `summary` or `comparison`. Anything else is stored as a
    /// written note, which is the conservative reading.
    pub kind: Option<String>,
}

/// Saves an assistant answer into the notebook, with the evidence it used.
///
/// ## The text comes from the transcript, not from the caller
///
/// The frontend sends three ids and a title. The body is read out of the
/// conversation store and the manifest out of the research store, both
/// owner-scoped. A caller that could supply the body could save *anything* as a
/// verified, cited answer — which is the one thing a saved answer is supposed to
/// mean.
#[tauri::command]
pub async fn notebook_save_answer(
    request: SaveAnswerRequest,
    store: State<'_, Arc<NotebookStore>>,
    conversations: State<'_, ConversationsState>,
    session: State<'_, CurrentSession>,
) -> Result<Note, String> {
    let signed_in = require_session(&session)?;
    let owner = &signed_in.user.id;

    let conversation = conversations
        .0
        .get(&request.conversation_id, Some(owner))
        .map_err(|error| format!("the conversation could not be read: {error}"))?
        .ok_or_else(|| "That conversation does not exist.".to_string())?;

    let message = conversation
        .messages
        .iter()
        .find(|message| message.id == request.message_id)
        .ok_or_else(|| "That answer is not in this conversation.".to_string())?;

    if message.content.trim().is_empty() {
        return Err(
            "That answer is empty, so there is nothing to save. Wait for it to finish, or ask \
             again."
                .to_string(),
        );
    }

    let manifest = store
        .research_turn(owner, &request.conversation_id, &request.message_id)
        .map_err(|error| format!("the evidence for this answer could not be read: {error}"))?;
    let evidence_json = manifest
        .as_ref()
        .and_then(|manifest| serde_json::to_string(manifest).ok());

    let kind = match request.kind.as_deref() {
        Some("summary") => NoteKind::Summary,
        Some("comparison") => NoteKind::Comparison,
        _ => NoteKind::Answer,
    };

    store
        .create_note(
            &request.notebook_id,
            owner,
            &request.title,
            &message.content,
            kind,
            Some(&request.conversation_id),
            Some(&request.message_id),
            evidence_json.as_deref(),
        )
        .map_err(|error| format!("the note could not be saved: {error}"))
}

// ── Relationship review ──────────────────────────────────────────────────

#[tauri::command]
pub async fn notebook_assertions(
    notebook_id: String,
    document_sha256: Option<String>,
    include_rejected: Option<bool>,
    store: State<'_, Arc<NotebookStore>>,
    session: State<'_, CurrentSession>,
) -> Result<Vec<Assertion>, String> {
    let signed_in = require_session(&session)?;
    store
        .assertions(
            &notebook_id,
            &signed_in.user.id,
            document_sha256.as_deref(),
            include_rejected.unwrap_or(false),
        )
        .map_err(|error| format!("the relationships could not be read: {error}"))
}

/// Accepts or rejects one extracted relationship.
#[tauri::command]
pub async fn notebook_review_assertion(
    notebook_id: String,
    assertion_id: String,
    status: String,
    note: Option<String>,
    store: State<'_, Arc<NotebookStore>>,
    session: State<'_, CurrentSession>,
) -> Result<Assertion, String> {
    let signed_in = require_session(&session)?;
    let owner = &signed_in.user.id;

    let status = match status.as_str() {
        "accepted" => AssertionStatus::Accepted,
        "rejected" => AssertionStatus::Rejected,
        "proposed" => AssertionStatus::Proposed,
        other => {
            return Err(format!(
                "\"{other}\" is not a review decision. Use accepted, rejected or proposed."
            ))
        }
    };

    store
        .review_assertion(&notebook_id, owner, &assertion_id, status, note.as_deref())
        .map_err(|error| format!("the review could not be recorded: {error}"))?;
    store
        .assertion(&notebook_id, owner, &assertion_id)
        .map_err(|error| format!("the relationship could not be read back: {error}"))?
        .ok_or_else(|| "that relationship is not in this notebook".to_string())
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CorrectAssertionRequest {
    pub notebook_id: String,
    pub assertion_id: String,
    /// Normalised subject term. The direction the person says it runs.
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub note: Option<String>,
}

/// Corrects a relationship's wording or which way round it runs.
#[tauri::command]
pub async fn notebook_correct_assertion(
    request: CorrectAssertionRequest,
    store: State<'_, Arc<NotebookStore>>,
    session: State<'_, CurrentSession>,
) -> Result<Assertion, String> {
    let signed_in = require_session(&session)?;
    let owner = &signed_in.user.id;
    let id = store
        .correct_assertion(
            &request.notebook_id,
            owner,
            &request.assertion_id,
            &request.subject,
            &request.predicate,
            &request.object,
            request.note.as_deref(),
        )
        .map_err(|error| format!("the correction could not be saved: {error}"))?;
    store
        .assertion(&request.notebook_id, owner, &id)
        .map_err(|error| format!("the relationship could not be read back: {error}"))?
        .ok_or_else(|| "the corrected relationship could not be read back".to_string())
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NewAssertionRequest {
    pub notebook_id: String,
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub note: Option<String>,
}

/// Records a relationship a person is asserting themselves.
///
/// Stored with no evidence, because there is none: nothing in the sources was
/// read to produce it. Every surface that shows it says so, and a retrieval that
/// uses it tells the model the same thing.
#[tauri::command]
pub async fn notebook_create_assertion(
    request: NewAssertionRequest,
    store: State<'_, Arc<NotebookStore>>,
    session: State<'_, CurrentSession>,
) -> Result<Assertion, String> {
    let signed_in = require_session(&session)?;
    let owner = &signed_in.user.id;
    let id = store
        .assert_by_hand(
            &request.notebook_id,
            owner,
            &request.subject,
            &request.predicate,
            &request.object,
            request.note.as_deref(),
        )
        .map_err(|error| format!("the relationship could not be added: {error}"))?;
    store
        .assertion(&request.notebook_id, owner, &id)
        .map_err(|error| format!("the relationship could not be read back: {error}"))?
        .ok_or_else(|| "the new relationship could not be read back".to_string())
}

/// Deletes a relationship a person added by hand.
#[tauri::command]
pub async fn notebook_delete_assertion(
    notebook_id: String,
    assertion_id: String,
    store: State<'_, Arc<NotebookStore>>,
    session: State<'_, CurrentSession>,
) -> Result<(), String> {
    let signed_in = require_session(&session)?;
    store
        .delete_hand_assertion(&notebook_id, &signed_in.user.id, &assertion_id)
        .map_err(|error| format!("the relationship could not be deleted: {error}"))
}

// ── What a question is about to use ──────────────────────────────────────

/// What the next question in this notebook would retrieve from, as a summary.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ScopePreview {
    pub notebook_name: String,
    pub sources_selected: u32,
    pub sources_total: u32,
    /// Selected sources whose extracted text cannot be read, by name. These
    /// will contribute nothing and the interface should say so before the
    /// question is asked rather than after it is answered.
    pub unreadable: Vec<String>,
    pub focus_labels: Vec<String>,
    pub focus_relationships: Vec<String>,
    /// How passages will be found, in the words the answer will use.
    pub retrieval_mode: String,
    pub retrieval_explanation: String,
    /// True when no graph has been built, so a graph selection is impossible.
    pub graph_built: bool,
}

/// Describes the scope of the next question without asking it.
///
/// Exists so the composer can say "8 sources · focused on PV-2201 · keyword
/// search" before somebody presses send, rather than the person discovering
/// after a two-minute answer that three of their sources were unreadable.
#[tauri::command]
pub async fn notebook_scope_preview(
    scope: crate::knowledge::ResearchScope,
    store: State<'_, Arc<NotebookStore>>,
    documents: State<'_, DocumentsState>,
    session: State<'_, CurrentSession>,
) -> Result<ScopePreview, String> {
    let signed_in = require_session(&session)?;
    let owner = &signed_in.user.id;

    let resolved = crate::knowledge::notebook_retrieval::resolve(store.inner(), owner, &scope)?;
    let conversation = format!("notebook:{}", resolved.notebook.id);

    let unreadable: Vec<String> = resolved
        .sources
        .iter()
        .filter(|source| {
            !matches!(
                documents
                    .0
                    .get(&source.document_sha256, owner, Some(&conversation)),
                Ok(Some(_))
            )
        })
        .map(|source| source.document_name.clone())
        .collect();

    let view = store
        .graph(&resolved.notebook.id, owner, None, None, 1, 1, false)
        .map_err(|error| format!("the graph could not be read: {error}"))?;

    // Keyword until an embedding model is actually wired into this path. Named
    // from the same enum the manifest records, so the preview and the answer
    // can never disagree about how the search ran.
    let mode = crate::knowledge::RetrievalMode::Keyword;

    Ok(ScopePreview {
        notebook_name: resolved.notebook.name.clone(),
        sources_selected: resolved.sources.len() as u32,
        sources_total: resolved.notebook.document_count,
        unreadable,
        focus_labels: resolved.focus_labels.clone(),
        focus_relationships: resolved
            .focus_assertions
            .iter()
            .map(|assertion| {
                format!(
                    "{} — {} → {}",
                    assertion.subject_label, assertion.predicate, assertion.object_label
                )
            })
            .collect(),
        retrieval_mode: mode.as_str().to_string(),
        retrieval_explanation: mode.describe().to_string(),
        graph_built: view.total_terms > 0,
    })
}
