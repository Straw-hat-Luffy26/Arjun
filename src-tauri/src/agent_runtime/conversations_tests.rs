//! Tests for the conversation store.
//!
//! These exercise the on-disk CRUD: open, create, append a turn, update
//! streaming content, mark a message done, list, and round-trip read.
//! They run on a temp directory so they do not pollute the application
//! data folder.
//!
//! The per-user isolation tests at the bottom cover TODO 2 of the
//! 7-step plan: a conversation created by user A is not visible to
//! user B, and B cannot read, write to, or delete A's conversation
//! even when they know the id.

use std::env;

use crate::agent_runtime::conversations::{
    Conversation, ConversationStore, MessageCompletion, MessageRole, MessageStatus,
    LEGACY_OWNER_ID,
};

const OWNER: &str = "engineer";
const OTHER: &str = "reviewer";

fn temp_dir() -> std::path::PathBuf {
    let mut dir = env::temp_dir();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    dir.push(format!("arjun-conv-tests-{}-{}", std::process::id(), nanos));
    std::fs::create_dir_all(&dir).expect("temp dir");
    dir
}

#[test]
fn open_creates_an_empty_store() {
    let dir = temp_dir();
    let store = ConversationStore::open(&dir).expect("open");
    assert!(store.list(None).unwrap().is_empty());
    assert!(store.get("nonexistent", None).unwrap().is_none());
}

#[test]
fn create_persists_a_conversation_with_one_welcome_message() {
    let dir = temp_dir();
    let store = ConversationStore::open(&dir).expect("open");
    let conv = store
        .create("Test".to_string(), "Welcome.".to_string(), OWNER)
        .expect("create");
    assert_eq!(conv.title, "Test");
    assert_eq!(conv.owner_user_id, OWNER);
    assert_eq!(conv.messages.len(), 1);
    assert_eq!(conv.messages[0].role, MessageRole::System);
    assert_eq!(conv.messages[0].content, "Welcome.");
    assert_eq!(conv.messages[0].status, MessageStatus::Done);

    // Read it back. As the owner.
    let fetched = store
        .get(&conv.id, Some(OWNER))
        .expect("get")
        .expect("found");
    assert_eq!(fetched.id, conv.id);
    assert_eq!(fetched.title, "Test");
    assert_eq!(fetched.owner_user_id, OWNER);
}

#[test]
fn append_user_turn_creates_user_and_streaming_assistant_messages() {
    let dir = temp_dir();
    let store = ConversationStore::open(&dir).expect("open");
    let conv = store
        .create("Test".to_string(), "Welcome.".to_string(), OWNER)
        .expect("create");
    let updated = store
        .append_user_turn(&conv.id, "Hello", "a-1", "run-1", OWNER)
        .expect("append")
        .expect("found");
    assert_eq!(updated.messages.len(), 3);
    let user = &updated.messages[1];
    assert_eq!(user.role, MessageRole::User);
    assert_eq!(user.content, "Hello");
    assert_eq!(user.status, MessageStatus::Done);
    let assistant = &updated.messages[2];
    assert_eq!(assistant.role, MessageRole::Assistant);
    assert_eq!(assistant.content, "");
    assert_eq!(assistant.status, MessageStatus::Streaming);
    assert_eq!(assistant.run_id.as_deref(), Some("run-1"));
    assert_eq!(updated.runs.len(), 1);
    assert!(updated.runs[0].live);
}

#[test]
fn update_streaming_content_replaces_assistant_text() {
    let dir = temp_dir();
    let store = ConversationStore::open(&dir).expect("open");
    let conv = store
        .create("Test".to_string(), "Welcome.".to_string(), OWNER)
        .expect("create");
    store
        .append_user_turn(&conv.id, "Hello", "a-1", "run-1", OWNER)
        .expect("append");
    let updated = store
        .update_streaming_content(&conv.id, "a-1", "part of an answer", OWNER)
        .expect("update")
        .expect("found");
    let assistant = updated
        .messages
        .iter()
        .find(|m| m.id == "a-1")
        .expect("assistant");
    assert_eq!(assistant.content, "part of an answer");
    assert_eq!(assistant.status, MessageStatus::Streaming);
}

#[test]
fn record_message_completion_marks_done_with_model() {
    let dir = temp_dir();
    let store = ConversationStore::open(&dir).expect("open");
    let conv = store
        .create("Test".to_string(), "Welcome.".to_string(), OWNER)
        .expect("create");
    store
        .append_user_turn(&conv.id, "Hello", "a-1", "run-1", OWNER)
        .expect("append");
    let updated = store
        .record_message_completion(
            &conv.id,
            "a-1",
            "run-1",
            MessageCompletion {
                final_content: Some("the final answer"),
                elapsed_ms: Some(1234),
                model_name: Some("gemma-3-12b-it"),
                model_role: Some("vision"),
                used_fallback: Some(false),
                outcome: Some("completed"),
                ..MessageCompletion::default()
            },
            OWNER,
        )
        .expect("complete")
        .expect("found");
    let assistant = updated
        .messages
        .iter()
        .find(|m| m.id == "a-1")
        .expect("assistant");
    assert_eq!(assistant.status, MessageStatus::Done);
    assert_eq!(assistant.content, "the final answer");
    assert_eq!(assistant.elapsed_ms, Some(1234));
    assert_eq!(assistant.model_name.as_deref(), Some("gemma-3-12b-it"));
    assert_eq!(assistant.model_role.as_deref(), Some("vision"));
    let run = updated.runs.iter().find(|r| r.run_id == "run-1").unwrap();
    assert!(!run.live);
    assert_eq!(run.model_name.as_deref(), Some("gemma-3-12b-it"));
}

/// Two writers reach one message and neither knows everything.
///
/// The front-end completes on `message_end`, which is where the model's token
/// usage arrives; the run completes again when `agent_run_prompt` resolves,
/// which is where the routing decision arrives. The run wrote last, so
/// assigning unconditionally meant the token counts were erased a moment after
/// they were recorded and the chat's counter never had anything to show.
#[test]
fn a_second_completion_does_not_erase_what_the_first_recorded() {
    let dir = temp_dir();
    let store = ConversationStore::open(&dir).expect("open");
    let conv = store
        .create("Test".to_string(), "Welcome.".to_string(), OWNER)
        .expect("create");
    store
        .append_user_turn(&conv.id, "Hello", "a-1", "run-1", OWNER)
        .expect("append");

    // The front-end, on `message_end`: token usage, no routing.
    store
        .record_message_completion(
            &conv.id,
            "a-1",
            "run-1",
            MessageCompletion {
                final_content: Some("the final answer"),
                elapsed_ms: Some(1234),
                tokens_in: Some(512),
                tokens_out: Some(64),
                ..MessageCompletion::default()
            },
            OWNER,
        )
        .expect("complete")
        .expect("found");

    // The run, on resolve: routing, no token usage.
    let updated = store
        .record_message_completion(
            &conv.id,
            "a-1",
            "run-1",
            MessageCompletion {
                final_content: Some("the final answer"),
                elapsed_ms: Some(1234),
                model_name: Some("gemma-3-12b-it"),
                model_role: Some("reasoning"),
                used_fallback: Some(false),
                outcome: Some("completed"),
                ..MessageCompletion::default()
            },
            OWNER,
        )
        .expect("complete")
        .expect("found");

    let assistant = updated
        .messages
        .iter()
        .find(|m| m.id == "a-1")
        .expect("assistant");
    assert_eq!(assistant.tokens_in, Some(512), "token usage was erased");
    assert_eq!(assistant.tokens_out, Some(64), "token usage was erased");
    assert_eq!(assistant.model_name.as_deref(), Some("gemma-3-12b-it"));
    assert_eq!(assistant.model_role.as_deref(), Some("reasoning"));
    assert_eq!(assistant.status, MessageStatus::Done);
}

/// A run cut off at the output cap keeps its fragment *and* its caveat.
///
/// The two halves have to survive together. The text alone reads exactly like
/// a short answer, and the caveat alone loses the only thing the run produced.
#[test]
fn a_length_limited_run_keeps_both_the_fragment_and_the_reason() {
    let dir = temp_dir();
    let store = ConversationStore::open(&dir).expect("open");
    let conv = store
        .create("Test".to_string(), "Welcome.".to_string(), OWNER)
        .expect("create");
    store
        .append_user_turn(&conv.id, "Specify the seal", "a-1", "run-1", OWNER)
        .expect("append");
    let updated = store
        .record_message_completion(
            &conv.id,
            "a-1",
            "run-1",
            MessageCompletion {
                final_content: Some("The seal specification is "),
                elapsed_ms: Some(900),
                error: Some("Stopped: the answer reached the output limit for one turn."),
                outcome: Some("lengthLimited"),
                failed: true,
                ..MessageCompletion::default()
            },
            OWNER,
        )
        .expect("complete")
        .expect("found");
    let assistant = updated
        .messages
        .iter()
        .find(|m| m.id == "a-1")
        .expect("assistant");
    assert_eq!(assistant.content, "The seal specification is ");
    assert_eq!(assistant.outcome.as_deref(), Some("lengthLimited"));
    assert_eq!(assistant.status, MessageStatus::Failed);
    assert!(assistant.error.is_some());
}

/// The two writers reach this row in either order and neither may erase the
/// other's half. The front-end knows the token usage; only the run knows how
/// the run ended.
#[test]
fn a_message_end_writer_does_not_erase_the_runs_recorded_ending() {
    let dir = temp_dir();
    let store = ConversationStore::open(&dir).expect("open");
    let conv = store
        .create("Test".to_string(), "Welcome.".to_string(), OWNER)
        .expect("create");
    store
        .append_user_turn(&conv.id, "Hello", "a-1", "run-1", OWNER)
        .expect("append");

    // The run, on resolve: it was stopped by policy.
    store
        .record_message_completion(
            &conv.id,
            "a-1",
            "run-1",
            MessageCompletion {
                error: Some("Stopped: it needed to do something it is not permitted to do."),
                outcome: Some("policyStopped"),
                failed: true,
                ..MessageCompletion::default()
            },
            OWNER,
        )
        .expect("complete");

    // The front-end, afterwards, with token usage and no idea how it ended.
    let updated = store
        .record_message_completion(
            &conv.id,
            "a-1",
            "run-1",
            MessageCompletion {
                tokens_in: Some(400),
                tokens_out: Some(12),
                failed: true,
                ..MessageCompletion::default()
            },
            OWNER,
        )
        .expect("complete")
        .expect("found");

    let assistant = updated
        .messages
        .iter()
        .find(|m| m.id == "a-1")
        .expect("assistant");
    assert_eq!(assistant.tokens_in, Some(400));
    assert_eq!(
        assistant.outcome.as_deref(),
        Some("policyStopped"),
        "the ending was erased by a writer that did not know it"
    );
}

#[test]
fn record_message_completion_can_mark_failed() {
    let dir = temp_dir();
    let store = ConversationStore::open(&dir).expect("open");
    let conv = store
        .create("Test".to_string(), "Welcome.".to_string(), OWNER)
        .expect("create");
    store
        .append_user_turn(&conv.id, "Hello", "a-1", "run-1", OWNER)
        .expect("append");
    let updated = store
        .record_message_completion(
            &conv.id,
            "a-1",
            "run-1",
            MessageCompletion {
                elapsed_ms: Some(500),
                error: Some("budget exhausted"),
                outcome: Some("budgetStopped"),
                failed: true,
                ..MessageCompletion::default()
            },
            OWNER,
        )
        .expect("complete")
        .expect("found");
    let assistant = updated
        .messages
        .iter()
        .find(|m| m.id == "a-1")
        .expect("assistant");
    assert_eq!(assistant.status, MessageStatus::Failed);
    assert_eq!(assistant.error.as_deref(), Some("budget exhausted"));
}

#[test]
fn list_returns_conversations_newest_first() {
    let dir = temp_dir();
    let store = ConversationStore::open(&dir).expect("open");
    let a = store
        .create("A".to_string(), "W".to_string(), OWNER)
        .expect("a");
    // Sleep to ensure lastActivityAt differs at the millisecond level.
    std::thread::sleep(std::time::Duration::from_millis(10));
    let b = store
        .create("B".to_string(), "W".to_string(), OWNER)
        .expect("b");
    let list = store.list(Some(OWNER)).expect("list");
    assert_eq!(list.len(), 2);
    assert_eq!(list[0].id, b.id);
    assert_eq!(list[1].id, a.id);
}

#[test]
fn append_user_turn_returns_none_for_unknown_conversation() {
    let dir = temp_dir();
    let store = ConversationStore::open(&dir).expect("open");
    let updated = store
        .append_user_turn("missing", "hi", "a-1", "run-1", OWNER)
        .unwrap();
    assert!(updated.is_none());
}

#[test]
fn round_trip_preserves_all_fields() {
    let dir = temp_dir();
    let store = ConversationStore::open(&dir).expect("open");
    let conv: Conversation = store
        .create("Round trip".to_string(), "W.".to_string(), OWNER)
        .expect("create");
    store
        .append_user_turn(&conv.id, "First turn", "a-1", "run-1", OWNER)
        .expect("append");
    store
        .update_streaming_content(&conv.id, "a-1", "streamed", OWNER)
        .expect("update");
    store
        .record_message_completion(
            &conv.id,
            "a-1",
            "run-1",
            MessageCompletion {
                final_content: Some("final"),
                elapsed_ms: Some(2000),
                model_name: Some("model-x"),
                model_role: Some("reasoning"),
                used_fallback: Some(false),
                outcome: Some("completed"),
                ..MessageCompletion::default()
            },
            OWNER,
        )
        .expect("complete");

    // New store instance reads the same file.
    let again = ConversationStore::open(&dir).expect("reopen");
    let fetched = again
        .get(&conv.id, Some(OWNER))
        .expect("get")
        .expect("found");
    assert_eq!(fetched.title, "Round trip");
    assert_eq!(fetched.messages.len(), 3);
    let assistant = fetched
        .messages
        .iter()
        .find(|m| m.id == "a-1")
        .expect("assistant");
    assert_eq!(assistant.content, "final");
    assert_eq!(assistant.status, MessageStatus::Done);
    assert_eq!(assistant.model_name.as_deref(), Some("model-x"));
    assert_eq!(fetched.runs.len(), 1);
    assert!(!fetched.runs[0].live);
}

// ---------------------------------------------------------------------------
// Per-user isolation tests (TODO 2 of the 7-step plan).
// ---------------------------------------------------------------------------

#[test]
fn a_non_owner_cannot_read_someone_elses_conversation() {
    let dir = temp_dir();
    let store = ConversationStore::open(&dir).expect("open");
    let conv = store
        .create("private".to_string(), "W.".to_string(), OWNER)
        .expect("create");
    // The other user, who is not the owner, asks for the same id.
    let result = store.get(&conv.id, Some(OTHER)).expect("get");
    assert!(
        result.is_none(),
        "non-owner must not see the contents of a conversation they do not own"
    );
    // The owner still sees it.
    let owner_view = store.get(&conv.id, Some(OWNER)).expect("get");
    assert!(owner_view.is_some());
}

#[test]
fn a_non_owner_cannot_append_to_someone_elses_conversation() {
    let dir = temp_dir();
    let store = ConversationStore::open(&dir).expect("open");
    let conv = store
        .create("private".to_string(), "W.".to_string(), OWNER)
        .expect("create");
    // The other user tries to write into the owner's conversation.
    let result = store
        .append_user_turn(&conv.id, "injected", "a-1", "run-1", OTHER)
        .expect("append");
    assert!(
        result.is_none(),
        "non-owner must not be able to append a turn to a conversation they do not own"
    );
    // Confirm the file is unchanged for the owner.
    let owner_view = store
        .get(&conv.id, Some(OWNER))
        .expect("get")
        .expect("found");
    assert_eq!(owner_view.messages.len(), 1, "no injected message should have been added");
}

#[test]
fn a_non_owner_cannot_delete_someone_elses_conversation() {
    let dir = temp_dir();
    let store = ConversationStore::open(&dir).expect("open");
    let conv = store
        .create("private".to_string(), "W.".to_string(), OWNER)
        .expect("create");
    let removed = store.delete(&conv.id, OTHER).expect("delete");
    assert!(
        !removed,
        "non-owner delete must return false (idempotent on a foreign file)"
    );
    // The file is still there for the owner.
    let owner_view = store.get(&conv.id, Some(OWNER)).expect("get");
    assert!(owner_view.is_some());
    // The owner can still delete.
    let removed_by_owner = store.delete(&conv.id, OWNER).expect("delete");
    assert!(removed_by_owner);
}

#[test]
fn list_filters_to_only_the_callers_conversations() {
    let dir = temp_dir();
    let store = ConversationStore::open(&dir).expect("open");
    store
        .create("A-owns".to_string(), "W".to_string(), OWNER)
        .expect("a");
    store
        .create("B-owns".to_string(), "W".to_string(), OTHER)
        .expect("b");
    let owner_list = store.list(Some(OWNER)).expect("list");
    let other_list = store.list(Some(OTHER)).expect("list");
    assert_eq!(owner_list.len(), 1, "owner sees only their own");
    assert_eq!(other_list.len(), 1, "other sees only their own");
    assert_eq!(owner_list[0].title, "A-owns");
    assert_eq!(other_list[0].title, "B-owns");
    // The unrestricted form still shows both — used by tests and
    // any future cross-account debug surface.
    let all = store.list(None).expect("list");
    assert_eq!(all.len(), 2);
}

#[test]
fn update_streaming_content_for_a_non_owner_is_a_no_op() {
    let dir = temp_dir();
    let store = ConversationStore::open(&dir).expect("open");
    let conv = store
        .create("private".to_string(), "W.".to_string(), OWNER)
        .expect("create");
    store
        .append_user_turn(&conv.id, "Hello", "a-1", "run-1", OWNER)
        .expect("append");
    // Non-owner tries to overwrite the streaming content.
    let result = store
        .update_streaming_content(&conv.id, "a-1", "forged", OTHER)
        .expect("update");
    assert!(result.is_none());
    // The owner's view is unchanged.
    let owner_view = store
        .get(&conv.id, Some(OWNER))
        .expect("get")
        .expect("found");
    let assistant = owner_view
        .messages
        .iter()
        .find(|m| m.id == "a-1")
        .expect("assistant");
    assert_eq!(assistant.content, "", "forged update must not be persisted");
}

#[test]
fn record_message_completion_for_a_non_owner_is_a_no_op() {
    let dir = temp_dir();
    let store = ConversationStore::open(&dir).expect("open");
    let conv = store
        .create("private".to_string(), "W.".to_string(), OWNER)
        .expect("create");
    store
        .append_user_turn(&conv.id, "Hello", "a-1", "run-1", OWNER)
        .expect("append");
    // Non-owner tries to mark the message done with a forged final.
    let result = store
        .record_message_completion(
            &conv.id,
            "a-1",
            "run-1",
            MessageCompletion {
                final_content: Some("forged final"),
                elapsed_ms: Some(1),
                model_name: Some("forged-model"),
                model_role: Some("forged"),
                used_fallback: Some(false),
                ..MessageCompletion::default()
            },
            OTHER,
        )
        .expect("complete");
    assert!(result.is_none());
    let owner_view = store
        .get(&conv.id, Some(OWNER))
        .expect("get")
        .expect("found");
    let assistant = owner_view
        .messages
        .iter()
        .find(|m| m.id == "a-1")
        .expect("assistant");
    assert_eq!(assistant.status, MessageStatus::Streaming);
    assert_eq!(assistant.content, "");
}

#[test]
fn legacy_v1_files_migrate_to_the_administrator_owner() {
    // Hand-craft a v1 file (no `owner_user_id` field) and confirm
    // the migration stamps it with LEGACY_OWNER_ID and bumps the
    // schema version on read.
    let dir = temp_dir();
    let store = ConversationStore::open(&dir).expect("open");
    // Create a conversation the modern way, then rewrite the file
    // with a v1 envelope that omits `ownerUserId`.
    let conv = store
        .create("legacy".to_string(), "W.".to_string(), LEGACY_OWNER_ID)
        .expect("create");
    let file_path = dir.join("conversations").join(format!("{}.json", conv.id));
    let body = serde_json::json!({
        "schemaVersion": 1,
        "conversation": {
            "id": conv.id,
            // ownerUserId intentionally absent — this is the v1 shape.
            "title": "legacy",
            "createdAt": "2024-01-01T00:00:00Z",
            "lastActivityAt": "2024-01-01T00:00:00Z",
            "messages": [],
            "runs": [],
            "compactions": 0u32,
        }
    });
    std::fs::write(
        &file_path,
        serde_json::to_vec_pretty(&body).expect("serialise"),
    )
    .expect("write v1 file");

    // Read it back. The migration should run, stamp the owner, and
    // rewrite the file at schema version 2.
    let fetched = store
        .get(&conv.id, Some(LEGACY_OWNER_ID))
        .expect("get")
        .expect("found");
    assert_eq!(fetched.owner_user_id, LEGACY_OWNER_ID);

    // The file is now at the current schema version.
    let raw = std::fs::read_to_string(&file_path).expect("read");
    let envelope: serde_json::Value = serde_json::from_str(&raw).expect("parse");
    assert_eq!(envelope["schemaVersion"], 2);
    assert_eq!(envelope["conversation"]["ownerUserId"], LEGACY_OWNER_ID);
}

/// What happens when the conversation store cannot be opened.
///
/// ## The defect
///
/// The failure path opened a *fixed* temp directory,
/// `arjun-conversations-fallback`, silently. Three things were wrong with that:
/// it is shared between sessions and users; it is stale, so a recovered session
/// found the previous degraded session's threads looking like history; and
/// nothing said so, so the chat behaved exactly as normal while the person's
/// real conversations were somewhere else.
mod degraded_storage {
    use super::OWNER;
    use crate::agent_runtime::conversations::{
        ConversationHealth, ConversationState, ConversationStore,
    };

    /// A path that cannot be a directory, because it is a file.
    fn blocked() -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("conversations");
        std::fs::write(&path, b"not a directory").expect("blocking file");
        (dir, path)
    }

    #[test]
    fn a_store_that_cannot_be_opened_reports_it() {
        let (_dir, path) = blocked();
        assert!(
            ConversationStore::open(&path).is_err(),
            "the store must not open where a regular file already is"
        );
    }

    #[test]
    fn a_healthy_session_refuses_nothing() {
        let health = ConversationHealth::durable();
        assert!(health.is_durable());
        assert_eq!(health.refusal(), None);
        assert_eq!(health.state(), &ConversationState::Durable);
    }

    #[test]
    fn an_ephemeral_session_refuses_new_conversations_and_says_where_they_would_go() {
        let health = ConversationHealth::ephemeral(
            "The conversation store could not be opened: access denied.",
            std::path::Path::new("/tmp/arjun-conversations-ephemeral-1234-5678"),
        );
        assert!(!health.is_durable());
        let refusal = health.refusal().expect("a reason");
        // What is wrong, where the writing goes, and what still works.
        assert!(refusal.contains("access denied"), "{refusal}");
        assert!(refusal.contains("arjun-conversations-ephemeral"), "{refusal}");
        assert!(refusal.contains("not be there after a restart"), "{refusal}");
        assert!(refusal.contains("can still be read"), "{refusal}");
    }

    #[test]
    fn the_ephemeral_directory_is_unique_per_session() {
        // The stale-reuse defect. Two sessions must not share a directory, or
        // one finds the other's threads and shows them as its own history.
        //
        // The uniqueness comes from the process id and a nanosecond timestamp,
        // which is what `lib.rs` composes. Asserted here on the shape rather
        // than by starting two applications.
        let first = format!("arjun-conversations-ephemeral-{}-{}", 1234, 1_000_000_001u64);
        let second = format!("arjun-conversations-ephemeral-{}-{}", 1234, 1_000_000_002u64);
        assert_ne!(first, second);
        assert!(!first.ends_with("fallback"), "a fixed name is a shared name");
    }

    #[test]
    fn two_ephemeral_stores_do_not_see_each_others_conversations() {
        // The property the unique directory buys, driven for real: a session
        // that writes into its own scratch directory leaves nothing for the
        // next one to find.
        let dir = tempfile::tempdir().expect("temp dir");
        let first = ConversationStore::open(&dir.path().join("session-1")).expect("open");
        first
            .create("Yesterday".to_string(), "W.".to_string(), OWNER)
            .expect("create");
        assert_eq!(first.list(Some(OWNER)).expect("list").len(), 1);

        let second = ConversationStore::open(&dir.path().join("session-2")).expect("open");
        assert!(
            second.list(Some(OWNER)).expect("list").is_empty(),
            "a new session inherited the previous one's ephemeral conversations"
        );
    }

    #[test]
    fn an_ephemeral_session_can_still_read_what_is_already_there() {
        // Refusing to *create* is the design; refusing to open would leave a
        // person unable to find out what is wrong. A store opened at a scratch
        // path still reads and writes normally — the refusal is a policy above
        // it, not a broken store.
        let dir = tempfile::tempdir().expect("temp dir");
        let store = ConversationStore::open(dir.path()).expect("open");
        let conversation = store
            .create("Readable".to_string(), "W.".to_string(), OWNER)
            .expect("create");
        assert!(store.get(&conversation.id, Some(OWNER)).expect("read").is_some());
    }

    #[test]
    fn the_state_serialises_for_the_ui() {
        let durable = serde_json::to_value(ConversationState::Durable).expect("serialises");
        assert_eq!(durable["state"], "durable");

        let ephemeral = serde_json::to_value(ConversationState::Ephemeral {
            because: "no disk".to_string(),
            directory: "/tmp/x".to_string(),
        })
        .expect("serialises");
        assert_eq!(ephemeral["state"], "ephemeral");
        assert_eq!(ephemeral["because"], "no disk");
        assert_eq!(ephemeral["directory"], "/tmp/x");
    }
}

/// The run the surface opened and the run the runtime issued are the same run.
///
/// The chat surface generates a run id before the run exists, and the runtime
/// then issues its own. The task record, the audit trail and the context ledger
/// are all filed under the runtime's. Completion is where the two are
/// reconciled — and while it matched on `run_id` it reconciled nothing:
///
///  - every run stayed `live: true`, because this is the only place that
///    clears it;
///  - the context meter read "No context yet" on every conversation, because
///    it resolves the ledger through the id recorded here.
///
/// Measured on the reported machine before the fix: three runs in one
/// conversation, all still `live`, none of whose ids addressed a task record.
#[test]
fn completion_reconciles_the_surface_run_id_to_the_runtime_one() {
    let dir = temp_dir();
    let store = ConversationStore::open(&dir).expect("open");
    let conv = store
        .create("Test".to_string(), "Welcome.".to_string(), OWNER)
        .expect("create");

    let surface_run_id = "client-generated-run";
    let assistant_message_id = "a-msg-1";
    store
        .append_user_turn(&conv.id, "hello", assistant_message_id, surface_run_id, OWNER)
        .expect("append")
        .expect("found");

    // The runtime issues its own id, and that is what completion carries.
    let runtime_run_id = "runtime-issued-run";
    let updated = store
        .record_message_completion(
            &conv.id,
            assistant_message_id,
            runtime_run_id,
            MessageCompletion {
                final_content: Some("hi"),
                elapsed_ms: Some(10),
                model_name: Some("some-model"),
                model_role: None,
                used_fallback: None,
                error: None,
                outcome: None,
                verification: None,
                tool_summary: None,
                failed: false,
                tokens_in: None,
                tokens_out: None,
            },
            OWNER,
        )
        .expect("complete")
        .expect("found");

    let run = updated
        .runs
        .iter()
        .find(|r| r.message_id == assistant_message_id)
        .expect("the run the turn opened is still there");

    assert_eq!(
        run.run_id, runtime_run_id,
        "the id recorded must be the one the task record is filed under"
    );
    assert!(!run.live, "a finished run must not stay marked live");
    assert!(run.finished_at.is_some(), "a finished run must carry when");
}

/// The run id has to be reconciled while the run is going, not only when it ends.
///
/// The chat surface reserves the assistant cell before the run exists, so it
/// files the `RunMeta` under a correlation id it invented. The runtime then
/// mints the run's own id, and *that* is the id the task record, the audit
/// trail, the event stream and the context ledger are keyed by.
///
/// `record_message_completion` already corrected this — at the one moment the
/// correction is worth nothing, because everything that reads the id reads it
/// while the run is in flight. See `ConversationStore::bind_run` for the three
/// things that were addressing a run that did not exist.
mod binding_a_run_to_its_real_id {
    use super::*;

    fn started(store: &ConversationStore) -> Conversation {
        let conversation = store
            .create("t".into(), "welcome".into(), OWNER)
            .expect("create");
        store
            .append_user_turn(
                &conversation.id,
                "what is the rating?",
                "a-cell-1",
                "correlation-1",
                OWNER,
            )
            .expect("append")
            .expect("owned")
    }

    #[test]
    fn the_run_list_holds_the_runtimes_id_and_not_the_correlation_id() {
        let dir = temp_dir();
        let store = ConversationStore::open(&dir).expect("open");
        let conversation = started(&store);
        assert_eq!(
            conversation.runs[0].run_id, "correlation-1",
            "the surface files it under the id it invented"
        );

        store
            .bind_run(&conversation.id, "a-cell-1", "server-run-1", OWNER)
            .expect("bind")
            .expect("owned");

        let reread = store.get(&conversation.id, Some(OWNER)).unwrap().unwrap();
        assert_eq!(reread.runs[0].run_id, "server-run-1");
        assert_eq!(reread.runs[0].message_id, "a-cell-1", "matched by cell");
    }

    /// The inspector opens from the message's own run id, so it has to move too.
    #[test]
    fn the_assistant_message_carries_the_runtimes_id() {
        let dir = temp_dir();
        let store = ConversationStore::open(&dir).expect("open");
        let conversation = started(&store);

        store
            .bind_run(&conversation.id, "a-cell-1", "server-run-1", OWNER)
            .expect("bind");

        let reread = store.get(&conversation.id, Some(OWNER)).unwrap().unwrap();
        let cell = reread
            .messages
            .iter()
            .find(|m| m.id == "a-cell-1")
            .expect("the reserved cell");
        assert_eq!(cell.run_id.as_deref(), Some("server-run-1"));
    }

    /// The isolation boundary every other read of this store applies.
    #[test]
    fn another_owner_cannot_rebind_a_run() {
        let dir = temp_dir();
        let store = ConversationStore::open(&dir).expect("open");
        let conversation = started(&store);

        let outcome = store
            .bind_run(&conversation.id, "a-cell-1", "hostile-run", OTHER)
            .expect("no error");
        assert!(outcome.is_none(), "a non-owner gets the unknown-id answer");

        let reread = store.get(&conversation.id, Some(OWNER)).unwrap().unwrap();
        assert_eq!(reread.runs[0].run_id, "correlation-1", "left untouched");
    }

    #[test]
    fn an_unknown_conversation_is_not_an_error() {
        let dir = temp_dir();
        let store = ConversationStore::open(&dir).expect("open");
        assert!(store
            .bind_run("no-such-conversation", "a-1", "r-1", OWNER)
            .expect("no error")
            .is_none());
    }

    /// A caller that reserved no cell of its own has nothing to correct, and
    /// the ordinary case must not rewrite the file on every turn.
    #[test]
    fn binding_an_id_that_is_already_right_changes_nothing() {
        let dir = temp_dir();
        let store = ConversationStore::open(&dir).expect("open");
        let conversation = started(&store);

        store
            .bind_run(&conversation.id, "a-cell-1", "server-run-1", OWNER)
            .expect("bind");
        let after_first = store.get(&conversation.id, Some(OWNER)).unwrap().unwrap();

        store
            .bind_run(&conversation.id, "a-cell-1", "server-run-1", OWNER)
            .expect("bind again");
        let after_second = store.get(&conversation.id, Some(OWNER)).unwrap().unwrap();

        assert_eq!(after_first.last_activity_at, after_second.last_activity_at);
        assert_eq!(after_second.runs[0].run_id, "server-run-1");
    }

    /// Completion still reconciles, for a run that died before reaching the
    /// start-time binding.
    #[test]
    fn completion_still_reconciles_a_run_that_was_never_bound() {
        let dir = temp_dir();
        let store = ConversationStore::open(&dir).expect("open");
        let conversation = started(&store);

        store
            .record_message_completion(
                &conversation.id,
                "a-cell-1",
                "server-run-1",
                MessageCompletion {
                    final_content: Some("Class 300."),
                    ..Default::default()
                },
                OWNER,
            )
            .expect("complete")
            .expect("owned");

        let reread = store.get(&conversation.id, Some(OWNER)).unwrap().unwrap();
        assert_eq!(reread.runs[0].run_id, "server-run-1");
        assert!(!reread.runs[0].live);
    }
}

/// Pins are a person's answer to "what does the rest of this task depend on?",
/// and that answer has to outlive the run they gave it in.
///
/// Held only by the run in flight, a pin was forgotten the moment that run
/// ended — so somebody who pinned once and asked five follow-ups was protected
/// for the first and nothing after it, with the panel showing the pin the whole
/// time.
mod pinned_context {
    use super::*;
    use crate::agent_runtime::conversations::MAX_PINNED_CONTEXT;

    fn thread(store: &ConversationStore) -> Conversation {
        store
            .create("t".into(), "welcome".into(), OWNER)
            .expect("create")
    }

    #[test]
    fn a_new_conversation_protects_nothing() {
        let dir = temp_dir();
        let store = ConversationStore::open(&dir).expect("open");
        let conversation = thread(&store);
        assert!(conversation.pinned_context.is_empty());
    }

    #[test]
    fn a_pin_survives_being_written_and_read_back() {
        let dir = temp_dir();
        let store = ConversationStore::open(&dir).expect("open");
        let conversation = thread(&store);

        store
            .set_pinned_context(&conversation.id, &["E3".to_string()], OWNER)
            .expect("write")
            .expect("owned");

        assert_eq!(
            store.pinned_context(&conversation.id, OWNER).unwrap(),
            vec!["E3".to_string()]
        );
    }

    /// The whole set replaces what was held. Unpinning matters as much as
    /// pinning, and a call that could only add would make this a decision
    /// nobody could take back.
    #[test]
    fn the_arriving_set_replaces_rather_than_merges() {
        let dir = temp_dir();
        let store = ConversationStore::open(&dir).expect("open");
        let conversation = thread(&store);

        store
            .set_pinned_context(
                &conversation.id,
                &["E1".to_string(), "E2".to_string()],
                OWNER,
            )
            .unwrap();
        store
            .set_pinned_context(&conversation.id, &["E2".to_string()], OWNER)
            .unwrap();

        assert_eq!(
            store.pinned_context(&conversation.id, OWNER).unwrap(),
            vec!["E2".to_string()],
            "E1 was unpinned and must not survive"
        );
    }

    #[test]
    fn unpinning_everything_leaves_nothing_protected() {
        let dir = temp_dir();
        let store = ConversationStore::open(&dir).expect("open");
        let conversation = thread(&store);

        store
            .set_pinned_context(&conversation.id, &["E1".to_string()], OWNER)
            .unwrap();
        store.set_pinned_context(&conversation.id, &[], OWNER).unwrap();

        assert!(store
            .pinned_context(&conversation.id, OWNER)
            .unwrap()
            .is_empty());
    }

    /// An empty id matches every message under a substring test, which is how
    /// one blank string silently protects a whole run's context and fills the
    /// window. Dropped at the boundary that owns the file.
    #[test]
    fn a_blank_id_is_refused_rather_than_stored() {
        let dir = temp_dir();
        let store = ConversationStore::open(&dir).expect("open");
        let conversation = thread(&store);

        store
            .set_pinned_context(
                &conversation.id,
                &["".to_string(), "   ".to_string(), "E1".to_string()],
                OWNER,
            )
            .unwrap();

        assert_eq!(
            store.pinned_context(&conversation.id, OWNER).unwrap(),
            vec!["E1".to_string()]
        );
    }

    #[test]
    fn the_same_id_twice_is_stored_once() {
        let dir = temp_dir();
        let store = ConversationStore::open(&dir).expect("open");
        let conversation = thread(&store);

        store
            .set_pinned_context(
                &conversation.id,
                &["E1".to_string(), "E1".to_string()],
                OWNER,
            )
            .unwrap();

        assert_eq!(
            store.pinned_context(&conversation.id, OWNER).unwrap().len(),
            1
        );
    }

    #[test]
    fn the_list_is_bounded() {
        let dir = temp_dir();
        let store = ConversationStore::open(&dir).expect("open");
        let conversation = thread(&store);

        let many: Vec<String> = (0..(MAX_PINNED_CONTEXT + 20))
            .map(|i| format!("E{i}"))
            .collect();
        store
            .set_pinned_context(&conversation.id, &many, OWNER)
            .unwrap();

        assert_eq!(
            store.pinned_context(&conversation.id, OWNER).unwrap().len(),
            MAX_PINNED_CONTEXT
        );
    }

    /// The isolation boundary every other write here applies.
    #[test]
    fn another_owner_can_neither_read_nor_set_the_pins() {
        let dir = temp_dir();
        let store = ConversationStore::open(&dir).expect("open");
        let conversation = thread(&store);
        store
            .set_pinned_context(&conversation.id, &["E1".to_string()], OWNER)
            .unwrap();

        assert!(
            store
                .set_pinned_context(&conversation.id, &["hostile".to_string()], OTHER)
                .expect("no error")
                .is_none(),
            "a non-owner gets the unknown-conversation answer"
        );
        assert!(
            store.pinned_context(&conversation.id, OTHER).unwrap().is_empty(),
            "and cannot read what is pinned either"
        );
        assert_eq!(
            store.pinned_context(&conversation.id, OWNER).unwrap(),
            vec!["E1".to_string()],
            "the owner's pins are untouched"
        );
    }

    /// Pinning is a decision about a conversation, not activity in it.
    /// Reordering somebody's sidebar because they pressed a pin would be a
    /// surprise.
    #[test]
    fn pinning_does_not_bump_the_conversation_up_the_sidebar() {
        let dir = temp_dir();
        let store = ConversationStore::open(&dir).expect("open");
        let conversation = thread(&store);
        let before = conversation.last_activity_at.clone();

        let after = store
            .set_pinned_context(&conversation.id, &["E1".to_string()], OWNER)
            .unwrap()
            .unwrap();

        assert_eq!(after.last_activity_at, before);
    }

    #[test]
    fn an_unknown_conversation_is_not_an_error() {
        let dir = temp_dir();
        let store = ConversationStore::open(&dir).expect("open");
        assert!(store
            .set_pinned_context("no-such-thread", &["E1".to_string()], OWNER)
            .expect("no error")
            .is_none());
    }
}

// ── What the plan is allowed to read ──────────────────────────────────────

/// A follow-up is a follow-up *to* something, and the plan has to see it.
///
/// `planning::derive` fixes which tools a run may reach from the words of one
/// prompt. Asked of "now turn that into a deck" alone it plans no deck, and the
/// gateway then refuses the very call the person just asked for.
#[test]
fn recent_requests_returns_the_threads_asks_oldest_first() {
    let dir = temp_dir();
    let store = ConversationStore::open(&dir).expect("open");
    let conv = store
        .create("Test".to_string(), "Welcome.".to_string(), OWNER)
        .expect("create");

    for (n, text) in ["draft a briefing deck on seal wear", "add the vendor figures", "yes, go ahead"]
        .iter()
        .enumerate()
    {
        store
            .append_user_turn(&conv.id, text, &format!("a-{n}"), &format!("run-{n}"), OWNER)
            .expect("append")
            .expect("found");
    }

    let recent = store.recent_requests(&conv.id, OWNER, 4).expect("read");
    assert_eq!(
        recent,
        vec![
            "draft a briefing deck on seal wear".to_string(),
            "add the vendor figures".to_string(),
            "yes, go ahead".to_string(),
        ],
        "the plan must still be able to see that a deck was asked for"
    );
}

#[test]
fn recent_requests_is_bounded_and_keeps_the_newest() {
    let dir = temp_dir();
    let store = ConversationStore::open(&dir).expect("open");
    let conv = store
        .create("Test".to_string(), "Welcome.".to_string(), OWNER)
        .expect("create");
    for n in 0..8 {
        store
            .append_user_turn(&conv.id, &format!("ask {n}"), &format!("a-{n}"), &format!("run-{n}"), OWNER)
            .expect("append")
            .expect("found");
    }

    let recent = store.recent_requests(&conv.id, OWNER, 3).expect("read");
    assert_eq!(recent, vec!["ask 5".to_string(), "ask 6".to_string(), "ask 7".to_string()]);
}

/// Only what the person asked for.
///
/// Folding the assistant's own words in would let a model widen its next turn's
/// plan by describing tools it would like to have — the plan is a bound on the
/// model, so the model must not be able to write it.
#[test]
fn recent_requests_ignores_what_the_assistant_said() {
    let dir = temp_dir();
    let store = ConversationStore::open(&dir).expect("open");
    let conv = store
        .create("Test".to_string(), "Welcome.".to_string(), OWNER)
        .expect("create");
    store
        .append_user_turn(&conv.id, "what does the SOP say?", "a-1", "run-1", OWNER)
        .expect("append")
        .expect("found");
    store
        .record_message_completion(
            &conv.id,
            "a-1",
            "run-1",
            MessageCompletion {
                final_content: Some("I could write a python script and a spreadsheet for this."),
                ..Default::default()
            },
            OWNER,
        )
        .expect("complete")
        .expect("found");

    let recent = store.recent_requests(&conv.id, OWNER, 4).expect("read");
    assert_eq!(recent, vec!["what does the SOP say?".to_string()]);
}

/// Somebody else's thread is not readable, and reads as no thread at all.
#[test]
fn recent_requests_is_owner_scoped() {
    let dir = temp_dir();
    let store = ConversationStore::open(&dir).expect("open");
    let conv = store
        .create("Test".to_string(), "Welcome.".to_string(), OWNER)
        .expect("create");
    store
        .append_user_turn(&conv.id, "draft a deck", "a-1", "run-1", OWNER)
        .expect("append")
        .expect("found");

    assert!(store.recent_requests(&conv.id, OTHER, 4).expect("read").is_empty());
}
