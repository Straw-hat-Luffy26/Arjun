//! Tauri commands for chat conversations.
//!
//! These commands are the *front* of the conversation layer: the chat
//! surface calls `agent_create_conversation` once on first open, then
//! `agent_append_turn` for every user message, and `agent_get_conversation` /
//! `agent_list_conversations` to render the sidebar.
//!
//! The relationship between a conversation and the existing run machinery is
//! this: `agent_start_run` continues to start a run, and from now on it
//! returns the new `runId` paired with a freshly-created `conversationId`.
//! The chat surface treats that first turn the same as any later one — the
//! commands below are the only difference between "first message" and
//! "follow-up", and the chat surface does not need to care which it is.

use std::sync::Arc;

use tauri::State;

use crate::agent_runtime::conversations::{
    Conversation, ConversationHealth, ConversationState, ConversationStore, Message,
    MessageCompletion, MessageStatus, RunToConversation,
};
use crate::commands::governance::{require_session, CurrentSession};

/// Tauri-managed wrapper around the conversation store.
pub struct ConversationsState(pub Arc<ConversationStore>);

/// Tauri-managed wrapper around the run→conversation index.
pub struct RunToConversationState(pub Arc<RunToConversation>);

/// Whether this session's chats will still be here tomorrow.
///
/// Consulted before a *new* conversation is created. Reading what is already
/// there is always allowed — it is the only way a person finds out what is
/// wrong. See [`crate::agent_runtime::conversations::ConversationHealth`].
pub struct ConversationHealthState(pub Arc<ConversationHealth>);

/// What the surface is told about conversation storage.
///
/// A separate command rather than a field on every response: the answer is the
/// same for the whole session, and a banner that has to wait for a conversation
/// to load is one a person sees after they have already typed.
#[tauri::command]
pub fn agent_conversation_health(
    health: State<'_, ConversationHealthState>,
) -> ConversationState {
    health.0.state().clone()
}

/// Create a new conversation with one system welcome message.
#[tauri::command]
pub fn agent_create_conversation(
    title: String,
    welcome: Option<String>,
    state: State<'_, ConversationsState>,
    health: State<'_, ConversationHealthState>,
    session: State<'_, CurrentSession>,
) -> Result<Conversation, String> {
    let session = require_session(&session)?;
    // A new thread the person will lose is worse than no new thread. Reading
    // what is already there stays allowed, which is how they find out why.
    if let Some(refusal) = health.0.refusal() {
        return Err(refusal);
    }
    let welcome = welcome.unwrap_or_else(|| {
        "Arjun is ready. Ask anything; nothing leaves this machine.".to_string()
    });
    state
        .0
        .create(title, welcome, &session.user.id)
        .map_err(|error| format!("the conversation could not be created: {error}"))
}

/// Read one conversation by id.
///
/// Returns `Ok(None)` for a conversation that does not exist, which the
/// front-end treats as a fresh empty conversation rather than an error.
#[tauri::command]
pub fn agent_get_conversation(
    id: String,
    state: State<'_, ConversationsState>,
    session: State<'_, CurrentSession>,
) -> Result<Option<Conversation>, String> {
    let session = require_session(&session)?;
    // Per-user isolation: a different user gets `None`, not the
    // contents. The front-end treats `None` as "no such
    // conversation", which is the honest answer for any caller
    // who is not the owner.
    state
        .0
        .get(&id, Some(&session.user.id))
        .map_err(|e| e.to_string())
}

/// All conversations, newest first by `lastActivityAt`.
#[tauri::command]
pub fn agent_list_conversations(
    state: State<'_, ConversationsState>,
    session: State<'_, CurrentSession>,
) -> Result<Vec<Conversation>, String> {
    let session = require_session(&session)?;
    // Per-user isolation: a user sees only their own
    // conversations. Administrators have no special visibility
    // here either — the audit log is the place to see
    // cross-account activity, not the chat sidebar.
    state
        .0
        .list(Some(&session.user.id))
        .map_err(|e| e.to_string())
}

/// Delete a conversation by id. Idempotent: a delete of a missing id
/// returns `Ok(false)` rather than an error, so the chat surface can
/// retry without surfacing a misleading "not found" to the user.
///
/// The deletion removes the on-disk JSON file for the conversation. The
/// in-memory `RunToConversation` index is left alone — the index is
/// rebuilt lazily on the next `agent_append_turn` for a run, and a run
/// whose conversation has been deleted will simply resolve to `None`
/// when the front-end asks the back-end "which conversation is this
/// run in?".
#[tauri::command]
pub fn agent_delete_conversation(
    id: String,
    state: State<'_, ConversationsState>,
    session: State<'_, CurrentSession>,
) -> Result<bool, String> {
    let session = require_session(&session)?;
    state
        .0
        .delete(&id, &session.user.id)
        .map_err(|e| e.to_string())
}

/// The user has sent a new message in an existing conversation.
///
/// This command:
///  1. Persists a new user `Message` and a fresh streaming assistant
///     `Message` in the conversation, both stamped with the supplied
///     `runId`.
///  2. Records the `(runId → conversationId)` mapping in the in-memory
///     index so subsequent `agent://event` messages for the run can be
///     routed back to the right conversation's assistant cell.
///
/// The actual `agent_start_run` is *not* triggered here: the front-end
/// already has the user's prompt and calls `agent_start_run` itself so the
/// existing event-subscription machinery is unchanged. This command only
/// persists the user message and reserves the assistant cell.
/// Reserve the user message and the streaming assistant cell for a new turn.
///
/// The assistant `message_id` is supplied by the front-end (which
/// already chose a stable id it can correlate `message_end` with) and
/// is named `message_id` here so the wire form is `messageId` — the
/// same name the rest of the conversation API uses. The role
/// (user or assistant) is decided by the store, not by the argument
/// name.
#[tauri::command]
pub fn agent_append_turn(
    conversation_id: String,
    run_id: String,
    message_id: String,
    user_prompt: String,
    conversations: State<'_, ConversationsState>,
    run_to_conversation: State<'_, RunToConversationState>,
    session: State<'_, CurrentSession>,
) -> Result<Option<Conversation>, String> {
    let session = require_session(&session)?;
    let updated = conversations
        .0
        .append_user_turn(
            &conversation_id,
            &user_prompt,
            &message_id,
            &run_id,
            &session.user.id,
        )
        .map_err(|e| e.to_string())?;
    if updated.is_some() {
        run_to_conversation.0.bind(&run_id, &conversation_id);
    }
    Ok(updated)
}

/// Streaming-content update for an in-flight assistant message.
///
/// Called by the front-end as `message_update` events arrive. The front-end
/// keeps the live `content` in component state and writes the snapshot
/// through this command so a remount can pick up the latest in-progress
/// text from disk rather than from the (best-effort) event channel.
#[tauri::command]
pub fn agent_update_streaming_content(
    conversation_id: String,
    message_id: String,
    content: String,
    state: State<'_, ConversationsState>,
    session: State<'_, CurrentSession>,
) -> Result<Option<Conversation>, String> {
    let session = require_session(&session)?;
    state
        .0
        .update_streaming_content(
            &conversation_id,
            &message_id,
            &content,
            &session.user.id,
        )
        .map_err(|e| e.to_string())
}

/// Mark an assistant message as finished (clean or failed).
///
/// Called by the front-end on `message_end` or run completion. The
/// `(runId → conversationId)` mapping is dropped on success: the run is
/// over and the index entry would just leak.
#[tauri::command]
pub fn agent_complete_message(
    conversation_id: String,
    message_id: String,
    run_id: String,
    final_content: Option<String>,
    elapsed_ms: Option<u64>,
    model_name: Option<String>,
    model_role: Option<String>,
    used_fallback: Option<bool>,
    error: Option<String>,
    // `outcome` is how the run ended, as `RunOutcome::kind` spells it. Optional
    // because the front-end learns it only when `agent_start_run` resolves; a
    // `message_end` writer sends `None` and leaves whatever the run recorded.
    outcome: Option<String>,
    // `verification` is what the verifier concluded: `ready` or `needsReview`.
    // Sent only by the run; a `message_end` writer omits it.
    verification: Option<String>,
    failed: bool,
    tokens_in: Option<u64>,
    tokens_out: Option<u64>,
    conversations: State<'_, ConversationsState>,
    run_to_conversation: State<'_, RunToConversationState>,
    session: State<'_, CurrentSession>,
) -> Result<Option<Conversation>, String> {
    let session = require_session(&session)?;
    let updated = conversations
        .0
        .record_message_completion(
            &conversation_id,
            &message_id,
            &run_id,
            MessageCompletion {
                final_content: final_content.as_deref(),
                elapsed_ms,
                model_name: model_name.as_deref(),
                model_role: model_role.as_deref(),
                used_fallback,
                error: error.as_deref(),
                outcome: outcome.as_deref(),
                verification: verification.as_deref(),
                // Not written from this side. The tool summary is built where
                // the tool calls are known — at the end of the run in
                // `commands::agent` — and `None` here means "I do not know
                // this", never "clear it".
                tool_summary: None,
                failed,
                tokens_in,
                tokens_out,
            },
            &session.user.id,
        )
        .map_err(|e| e.to_string())?;
    run_to_conversation.0.unbind(&run_id);
    Ok(updated)
}

/// Reverse-lookup: which conversation does this run belong to?
///
/// Used by the front-end when an event arrives on `agent://event` to figure
/// out which `Conversation` (and therefore which `Message`) to update. The
/// index is in-memory; on a remount the index is rebuilt lazily as
/// `agent_append_turn` is called again.
#[tauri::command]
pub fn agent_run_conversation(
    run_id: String,
    run_to_conversation: State<'_, RunToConversationState>,
    session: State<'_, CurrentSession>,
) -> Option<String> {
    // require_session returns an error string; for an Option-returning command
    // we return None for an unauthenticated caller so the front-end treats
    // it as "no conversation known for this run", which is the same shape
    // it already handles for a run whose conversation has been deleted.
    if require_session(&session).is_err() {
        return None;
    }
    run_to_conversation.0.lookup(&run_id)
}

/// Re-export for callers that need the types.
pub use crate::agent_runtime::conversations::{MessageRole, RunMeta};

/// Helper to construct a fresh assistant message id from the front-end.
pub fn new_assistant_message_id() -> String {
    format!("a-{}", uuid::Uuid::new_v4())
}

/// Helper to read the final status of a message without having to plumb a
/// full Conversation through every callsite that needs only the status.
pub fn is_terminal(status: MessageStatus) -> bool {
    matches!(status, MessageStatus::Done | MessageStatus::Failed)
}

/// Read-only access to a single message by id.
///
/// Used by the front-end on remount: the durable event channel has the
/// `runId` of an in-flight run, the in-memory index can be empty, and
/// reading from disk is the honest answer.
#[tauri::command]
pub fn agent_get_message(
    conversation_id: String,
    message_id: String,
    state: State<'_, ConversationsState>,
    session: State<'_, CurrentSession>,
) -> Result<Option<Message>, String> {
    let session = require_session(&session)?;
    let conv = state
        .0
        .get(&conversation_id, Some(&session.user.id))
        .map_err(|e| e.to_string())?;
    Ok(conv.and_then(|c| c.messages.into_iter().find(|m| m.id == message_id)))
}
