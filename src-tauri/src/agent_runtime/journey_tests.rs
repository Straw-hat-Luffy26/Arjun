//! The OCR-to-thinking journey, driven through the production paths.
//!
//! ## What these prove that the unit tests do not
//!
//! The unit tests around this work assert that each part behaves: the store
//! keeps pages, `turn_context` fits history, the gateway authorises tools. None
//! of them answers the question the whole change exists for:
//!
//! > A document is read. The model writes an answer that does not mention some
//! > detail on page 31. Two turns later somebody asks about that detail, and
//! > does not re-attach the file. Can the run still get it?
//!
//! Replaying the conversation cannot answer it — the assistant's visible
//! answers contain what the model chose to write, which by construction
//! excludes what it did not. So these drive the *retrieval* path end to end,
//! through `authorize` and `execute` exactly as a model's tool call does, and
//! assert on the text the gateway actually returns.
//!
//! That is the evidence: an authorised retrieval result carrying a fact that
//! appears in no message of the conversation.

use std::collections::BTreeMap;
use std::sync::Arc;

use serde_json::{json, Value};

use super::documents::{NewExtraction, Sighting};
use super::*;
use crate::identity::{Role, Session, User};

/// A detail that exists only on a later page of the document.
///
/// Deliberately specific and unguessable: a model that has not read page 31
/// cannot produce this, so finding it in a result proves it came from the store
/// rather than from the conversation or the model's weights.
const FACT_ON_PAGE_31: &str = "flange gasket torque 47 Nm, revision C, sheet 31 of 40";

const OWNER_ID: &str = "priya";
const OTHER_ID: &str = "reviewer";
const CONVERSATION: &str = "c-journey-1";
const RUN: &str = "run-journey-1";

fn session_for(id: &str) -> Arc<std::sync::RwLock<Option<Session>>> {
    Arc::new(std::sync::RwLock::new(Some(Session::open(User::new(
        id,
        "A Person",
        vec![Role::Employee],
    )))))
}

fn sha() -> String {
    "ab".repeat(32)
}

/// A forty-page scan whose interesting page is 31, as `read_attachment` leaves
/// one: page-keyed text, a content hash, and provenance naming the turn.
fn a_read_document(owner: &str, conversation: &str) -> NewExtraction {
    let mut page_text = BTreeMap::new();
    page_text.insert(1, "General arrangement. Nothing notable.".to_string());
    page_text.insert(2, "Notes and revisions.".to_string());
    page_text.insert(31, FACT_ON_PAGE_31.to_string());
    NewExtraction {
        sha256: sha(),
        name: "pump-skid-rev-c.pdf".to_string(),
        kind: "pdf-scan".to_string(),
        pages: 40,
        truncated: false,
        page_text,
        sighting: Sighting {
            owner_user_id: owner.to_string(),
            conversation_id: conversation.to_string(),
            message_id: "a-cell-1".to_string(),
            run_id: "run-that-read-it".to_string(),
            at: chrono::Utc::now().to_rfc3339(),
            ocr_model_id: Some("unlimited-ocr-q6-k".to_string()),
            ocr_detent: Some(crate::ai_engine::ocr_profile::OcrDetent::Detailed),
        },
    }
}

/// The tool call a model makes, in the shape the gateway receives it.
fn read_pages(from: u32, to: u32) -> Value {
    json!({
        "runId": RUN,
        "toolCallId": "tc-read-pages",
        "tool": "document.read_pages",
        "args": { "documentSha256": sha(), "fromPage": from, "toPage": to },
    })
}

/// Deps with the document stored and the run bound to its conversation, which
/// is the state `agent_start_run` leaves behind before the loop is handed
/// anything.
fn journey(
    owner: &str,
    stored_for_owner: &str,
    stored_in_conversation: &str,
) -> (Arc<RuntimeDeps>, tempfile::TempDir) {
    let (deps, dir) = super::tests::deps_with(session_for(owner));
    deps.documents
        .record(a_read_document(stored_for_owner, stored_in_conversation))
        .expect("the document is stored, as the run stores it");
    deps.run_to_conversation.bind(RUN, CONVERSATION);
    // The plan the run is held to, derived the way `agent_start_run` derives
    // it. Without one the gateway refuses every call outright — which is the
    // correct behaviour and is tested elsewhere, but it would make these tests
    // assert that refusal instead of the retrieval they are about.
    deps.plans.lock().expect("fresh lock").insert(
        RUN.to_string(),
        planning::plan_for(RUN, "what does the drawing say about the gasket?"),
    );
    (deps, dir)
}

/// Runs a tool call the way a model's does: authorise, then spend the grant.
async fn through_the_gateway(call: Value, deps: &Arc<RuntimeDeps>) -> Result<String, String> {
    let allow = authorize(call.clone(), deps)
        .await
        .map_err(|error| format!("authorise: {}", error.message))?;
    let grant = allow
        .get("grant")
        .and_then(Value::as_str)
        .ok_or_else(|| format!("the gateway refused: {allow}"))?
        .to_string();
    let mut spent = call;
    spent["grant"] = json!(grant);
    let result = execute(spent, deps)
        .await
        .map_err(|error| format!("execute: {}", error.message))?;
    Ok(result["text"].as_str().unwrap_or_default().to_string())
}

/// The other tool call a model makes: find a passage without knowing its page.
fn search(query: &str) -> Value {
    json!({
        "runId": RUN,
        "toolCallId": "tc-document-search",
        "tool": "document.search",
        "args": { "query": query },
    })
}

/// The headline. A fact that reached no answer is still retrievable, by a later
/// turn, with nothing re-attached.
#[tokio::test]
async fn a_later_turn_recalls_a_page_the_answer_never_mentioned() {
    let (deps, _dir) = journey(OWNER_ID, OWNER_ID, CONVERSATION);

    // What the conversation actually holds: an answer about the drawing that
    // says nothing about page 31. This is what a replay would give a later
    // turn, and it is why a replay is not enough.
    let visible_answer =
        "The pump skid drawing is revision C. The general arrangement is on sheet 1.";
    assert!(
        !visible_answer.contains("47 Nm"),
        "the fixture must not leak the fact into the visible answer"
    );

    let retrieved = through_the_gateway(read_pages(31, 31), &deps)
        .await
        .expect("a later turn may read a page of its own conversation's document");

    // The evidence: an authorised retrieval result carrying a fact that appears
    // in no message of the conversation.
    assert!(
        retrieved.contains(FACT_ON_PAGE_31),
        "page 31 was read minutes ago and is not retrievable: {retrieved}"
    );
    assert!(
        retrieved.contains("31"),
        "the page must be cited by its own number: {retrieved}"
    );
}

/// The later-page case specifically: a large document whose interesting page is
/// far past anything the context budget would have admitted.
#[tokio::test]
async fn a_page_far_beyond_what_the_budget_admitted_is_still_reachable() {
    let (deps, _dir) = journey(OWNER_ID, OWNER_ID, CONVERSATION);

    // Pages 1-2 are what a chunked injection would have carried. Page 31 is
    // what it dropped.
    let early = through_the_gateway(read_pages(1, 2), &deps).await.unwrap();
    assert!(early.contains("General arrangement"));
    assert!(
        !early.contains(FACT_ON_PAGE_31),
        "the fixture is not shaped right"
    );

    let late = through_the_gateway(read_pages(31, 31), &deps).await.unwrap();
    assert!(late.contains(FACT_ON_PAGE_31));
}

/// A page with no stored text is reported as unread rather than omitted. "This
/// page is blank" and "nobody read this page" lead to opposite conclusions.
#[tokio::test]
async fn a_range_naming_pages_with_no_text_says_which_it_did_not_return() {
    let (deps, _dir) = journey(OWNER_ID, OWNER_ID, CONVERSATION);
    let text = through_the_gateway(read_pages(1, 3), &deps).await.unwrap();

    assert!(text.contains("General arrangement"), "page 1 came back");
    assert!(text.contains("Not returned"), "page 3 must be named: {text}");
    assert!(text.contains('3'));
}

/// Owner isolation, at the gateway rather than at the store.
#[tokio::test]
async fn another_person_cannot_read_the_document_through_the_tool() {
    // Stored by `priya`; the caller is somebody else entirely.
    let (deps, _dir) = journey(OTHER_ID, OWNER_ID, CONVERSATION);

    let outcome = through_the_gateway(read_pages(31, 31), &deps).await;
    match outcome {
        Ok(text) => {
            assert!(
                !text.contains(FACT_ON_PAGE_31),
                "another person read the document: {text}"
            );
            assert!(
                text.contains("No document with that id"),
                "the refusal must not confirm the document exists: {text}"
            );
        }
        Err(error) => assert!(
            !error.contains(FACT_ON_PAGE_31),
            "the refusal leaked the content: {error}"
        ),
    }
}

/// Conversation isolation. The same person, a document they attached to a
/// *different* thread.
#[tokio::test]
async fn a_document_from_another_conversation_is_not_in_this_one() {
    let (deps, _dir) = journey(OWNER_ID, OWNER_ID, "c-some-other-thread");

    let outcome = through_the_gateway(read_pages(31, 31), &deps).await;
    match outcome {
        Ok(text) => assert!(
            !text.contains(FACT_ON_PAGE_31),
            "a document from another thread was readable here: {text}"
        ),
        Err(error) => assert!(!error.contains(FACT_ON_PAGE_31)),
    }
}

/// A run with no conversation reads nothing: without a thread there is no scope
/// to check a document against, and an unscoped read of a content-addressed
/// store is a read of every document on the machine.
#[tokio::test]
async fn a_run_bound_to_no_conversation_reads_nothing() {
    let (deps, dir) = super::tests::deps_with(session_for(OWNER_ID));
    deps.documents
        .record(a_read_document(OWNER_ID, CONVERSATION))
        .expect("stored");
    // Deliberately not bound.

    let outcome = through_the_gateway(read_pages(31, 31), &deps).await;
    match outcome {
        Ok(text) => assert!(!text.contains(FACT_ON_PAGE_31), "{text}"),
        Err(error) => assert!(!error.contains(FACT_ON_PAGE_31)),
    }
    drop(dir);
}

/// The bound is enforced through the gateway too, and refuses rather than
/// quietly trimming — so the model always knows what it did and did not get.
#[tokio::test]
async fn a_range_wider_than_the_cap_is_refused_through_the_tool() {
    let (deps, _dir) = journey(OWNER_ID, OWNER_ID, CONVERSATION);

    let outcome = through_the_gateway(read_pages(1, 40), &deps).await;
    let message = match outcome {
        Ok(text) => text,
        Err(error) => error,
    };
    assert!(
        message.contains("at most") || message.contains("40 pages"),
        "a 40-page range must be refused, not trimmed: {message}"
    );
    assert!(!message.contains(FACT_ON_PAGE_31));
}

/// A model-supplied id is not a filename.
#[tokio::test]
async fn a_hostile_document_id_reaches_no_file() {
    let (deps, _dir) = journey(OWNER_ID, OWNER_ID, CONVERSATION);

    let mut call = read_pages(1, 1);
    call["args"]["documentSha256"] = json!("../../conversations/secret");
    let outcome = through_the_gateway(call, &deps).await;
    let message = match outcome {
        Ok(text) => text,
        Err(error) => error,
    };
    assert!(
        message.contains("No document with that id") || message.contains("refused"),
        "a path-shaped id must be refused: {message}"
    );
}

/// The conversation's own record must not carry the fact, or the test above
/// proves nothing: retrieval would be recovering something a replay could also
/// have found.
#[test]
fn the_fact_is_absent_from_everything_a_replay_could_reach() {
    use super::conversations::{Conversation, Message, MessageRole, MessageStatus};

    let message = |id: &str, role: MessageRole, content: &str| Message {
        id: id.to_string(),
        conversation_id: CONVERSATION.to_string(),
        role,
        content: content.to_string(),
        status: MessageStatus::Done,
        run_id: None,
        created_at: "2026-01-01T00:00:00Z".to_string(),
        completed_at: None,
        elapsed_ms: None,
        error: None,
        model_name: None,
        model_role: None,
        used_fallback: None,
        tokens_in: None,
        tokens_out: None,
        outcome: None,
        verification: None,
        tool_summary: None,
    };
    let conversation = Conversation {
        id: CONVERSATION.to_string(),
        owner_user_id: OWNER_ID.to_string(),
        title: "t".to_string(),
        created_at: "2026-01-01T00:00:00Z".to_string(),
        last_activity_at: "2026-01-01T00:00:00Z".to_string(),
        messages: vec![
            message("u1", MessageRole::User, "What is this drawing?"),
            message(
                "a1",
                MessageRole::Assistant,
                "The pump skid drawing is revision C. The general arrangement is on sheet 1.",
            ),
            message("u2", MessageRole::User, "What is the gasket torque?"),
            message("a2", MessageRole::Assistant, ""),
        ],
        runs: Vec::new(),
        compactions: 0,
        pinned_context: Vec::new(),
        routed_role: None,
        routed_model_id: None,
    };

    let fitted = super::turn_context::fit(&conversation, "a2", 10_000, &[]);
    let replayable: String = fitted
        .turns
        .iter()
        .map(|turn| turn.content.clone())
        .collect::<Vec<_>>()
        .join("\n");

    assert!(
        !replayable.contains("47 Nm"),
        "the history carries the fact, so retrieval is not what recovered it"
    );
    assert!(
        !replayable.contains(FACT_ON_PAGE_31),
        "the history carries the fact verbatim"
    );
    // …and the history is genuinely non-empty, or the assertion above is
    // trivially true.
    assert!(replayable.contains("revision C"));
}

/// The case a page range cannot serve: the model does not know the page.
///
/// This is the normal state after a turn that could only afford part of a
/// document. The prompt says which pages are not shown; it does not say which
/// of them holds the answer, and `read_pages` can only walk them ten at a time.
/// Searching answers the question the model actually has.
#[tokio::test]
async fn a_passage_is_found_by_what_it_says_without_knowing_its_page() {
    let (deps, _dir) = journey(OWNER_ID, OWNER_ID, CONVERSATION);

    let found = through_the_gateway(search("flange gasket torque"), &deps)
        .await
        .expect("a run may search its own conversation's documents");

    assert!(
        found.contains(FACT_ON_PAGE_31),
        "the passage was not found by its content: {found}"
    );
    // Citable, or it is not evidence.
    assert!(
        found.contains("page 31"),
        "the passage arrived without the page it came from: {found}"
    );
    assert!(
        found.contains("pump-skid-rev-c.pdf"),
        "the passage arrived without the document it came from: {found}"
    );
}

/// Finding nothing is reported as finding nothing, with the size of what was
/// looked at — so a model can tell "not in this document" from "no documents".
#[tokio::test]
async fn a_search_that_matches_nothing_says_how_much_it_looked_at() {
    let (deps, _dir) = journey(OWNER_ID, OWNER_ID, CONVERSATION);

    let found = through_the_gateway(search("zirconium centrifuge calibration"), &deps)
        .await
        .expect("a search that matches nothing is still a successful call");

    assert!(found.contains("No passage matched"), "{found}");
    assert!(
        found.contains("section(s)"),
        "the model was not told how much was searched: {found}"
    );
    assert!(
        !found.contains(FACT_ON_PAGE_31),
        "a non-matching search returned the document anyway: {found}"
    );
}

/// The isolation boundary, on the new door as well as the old one.
///
/// Content-addressed storage means the same bytes may be somebody else's
/// document. A query is a fine way to find out what one says, so search is
/// scoped exactly as `read_pages` is: this owner, this conversation.
#[tokio::test]
async fn searching_reaches_no_document_this_person_did_not_attach() {
    // Stored by somebody else, in the same conversation id.
    let (deps, _dir) = journey(OWNER_ID, OTHER_ID, CONVERSATION);

    let found = through_the_gateway(search("flange gasket torque"), &deps)
        .await
        .expect("the call is answered rather than erroring");

    assert!(
        !found.contains(FACT_ON_PAGE_31),
        "another person's document was searchable: {found}"
    );
    assert!(
        found.contains("0 document(s)"),
        "the count leaked that a document exists: {found}"
    );
}

/// A document the same person attached to a *different* thread is not part of
/// this one, and searching must not be a way around that.
#[tokio::test]
async fn searching_reaches_no_document_from_another_conversation() {
    let (deps, _dir) = journey(OWNER_ID, OWNER_ID, "c-somewhere-else");

    let found = through_the_gateway(search("flange gasket torque"), &deps)
        .await
        .expect("the call is answered rather than erroring");

    assert!(
        !found.contains(FACT_ON_PAGE_31),
        "a document from another conversation was searchable: {found}"
    );
}

/// An empty query is refused with a sentence that says what to send instead,
/// rather than returning everything or nothing.
#[tokio::test]
async fn an_empty_query_is_refused_with_something_actionable() {
    let (deps, _dir) = journey(OWNER_ID, OWNER_ID, CONVERSATION);

    let refusal = through_the_gateway(search("   "), &deps)
        .await
        .expect_err("an empty query is refused");

    assert!(
        refusal.contains("query is required"),
        "the refusal did not say what was missing: {refusal}"
    );
}
