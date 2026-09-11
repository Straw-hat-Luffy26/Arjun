//! The conversation a fresh turn starts from.
//!
//! ## The failure this exists to remove
//!
//! Every message the chat surface sent created a new run, and a run was handed
//! exactly one thing: the prompt the person had just typed. The conversation was
//! on screen and in a file on disk, and none of it entered the model's request.
//! So the second turn of every conversation began with a model that had never
//! seen the first — "what was the pressure rating you just quoted?" was answered
//! by a model with no quote, no question, and no way to say it had lost either.
//!
//! Switching models made the same failure louder rather than different: nothing
//! was carried across because nothing was carried at all.
//!
//! ## Why this is built here and not in the runtime
//!
//! Three reasons, and each of them is a boundary:
//!
//! - **Ownership.** [`Conversation`] carries `owner_user_id`, and every read of
//!   the store goes through the owner filter. This side is the side that has the
//!   signed-in user. The runtime has a JSON-RPC channel and no idea who is at
//!   the keyboard, so a runtime that assembled its own history would be
//!   assembling it without the check that keeps one person's thread out of
//!   another's.
//! - **Lifecycle.** This is derived from the conversation, not from a run. A run
//!   is one turn; the thread outlives every run in it, and a run that never
//!   started, crashed, or was stopped leaves the thread intact. Deriving the
//!   history from run state is what tied it to a lifecycle that ends every time
//!   somebody presses Enter.
//! - **Budget.** The destination model — and therefore its window — is chosen on
//!   this side, after routing. The runtime learns the window; it does not pick
//!   it, and it cannot fit history to a model that has not been chosen yet.
//!
//! ## Why the new question is not in here
//!
//! It is the one message this module refuses to return, and that refusal is the
//! contract. The runtime seeds these turns as the transcript a run *starts*
//! from, and then submits the new question through the ordinary prompt path. If
//! the question were also in the history, the runtime would be holding a
//! transcript whose last message is the user's — which is a transcript that
//! looks already-asked, and the two failure modes are the ones the audit named:
//! a run that skips generation because the last seeded message was an assistant
//! answer, and a run that continues without ever putting the new question to the
//! model.
//!
//! Excluding it here means the count is structural. The question is appended
//! exactly once because there is exactly one place that appends it, and it is
//! not this one.

use serde::Serialize;

use super::conversations::{Conversation, Message, MessageRole, MessageStatus};
use crate::ai_engine::ocr_budget::estimate_tokens;

/// One prior turn, in the shape the runtime seeds a transcript from.
///
/// Deliberately narrow. A [`Message`] carries token counts, a run id, a verdict
/// from the verifier and an outcome, none of which the model should be reading
/// as though the person had said it. What crosses the wire is a role and words.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ContextTurn {
    /// `user` or `assistant`. Never `system`: see [`is_eligible`].
    pub role: &'static str,
    pub content: String,
}

/// The history that fitted, and an honest account of what did not.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FittedContext {
    /// Oldest first, so the runtime can seed them in order.
    pub turns: Vec<ContextTurn>,
    /// Eligible messages the budget could not hold, dropped oldest-first.
    ///
    /// Reported rather than swallowed. A run whose history was trimmed is a run
    /// that may be missing the sentence the question refers to, and the only
    /// thing worse than trimming it is trimming it silently.
    pub dropped: u32,
    /// Estimated tokens the kept turns occupy. An estimate, and named as one.
    pub tokens: u32,
}

impl FittedContext {
    /// Nothing carried, because there was nothing to carry.
    pub fn empty() -> Self {
        FittedContext {
            turns: Vec::new(),
            dropped: 0,
            tokens: 0,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.turns.is_empty()
    }
}

/// Whether a stored message belongs in a later turn's context.
///
/// Four rules, each removing a specific way a transcript lies:
///
/// - **System messages are excluded.** The only ones written are the surface's
///   own greeting ("Arjun is ready…"), seeded by `ConversationStore::create`.
///   Feeding a model its own product's welcome banner as conversation is noise,
///   and the real system prompt is composed separately and sent every turn.
/// - **A streaming message is excluded.** Its content is whatever had arrived
///   when the file was last written, which is a sentence that stops mid-word.
///   That is a fragment presented as a finished answer.
/// - **A failed message is excluded.** It holds no answer — the run that owned
///   it did not produce one — and its `content` is either empty or a partial
///   stream. What went wrong is on the screen for the person; the model has no
///   use for a turn where nothing was said.
/// - **An empty message is excluded.** A blank turn in a transcript teaches a
///   model that blank turns are acceptable output.
pub fn is_eligible(message: &Message) -> bool {
    if message.role == MessageRole::System {
        return false;
    }
    if message.status != MessageStatus::Done {
        return false;
    }
    !message.content.trim().is_empty()
}

/// Rewrites this-run-only evidence markers so a later turn cannot reuse them.
///
/// ## The hazard
///
/// `[E1]…[En]` are handed out by `retrieval::record`, numbered from one, into a
/// table keyed by **run id** — and every chat turn is its own run. So the `[E2]`
/// in an assistant message from turn 3 does not merely dangle in turn 4: turn 4
/// does its own searching, is handed its own `[E1]`, `[E2]`, and the verifier
/// resolves markers against *that* table. A model that copies the earlier
/// marker forward, which is exactly what a model reading its own last answer
/// does, produces a claim the verifier then confirms against a completely
/// different passage. A wrong citation that passes checking is worse than a
/// missing one.
///
/// ## Why this is not solved by scoping the table to the conversation
///
/// That is the fuller fix and it is a bigger change than it looks: the
/// numbering would have to continue across turns rather than restart, the table
/// would have to be bounded so a long thread does not accumulate every passage
/// it ever retrieved, and the verifier would have to agree with both. It is
/// worth doing and it is not this. What is done here removes the hazard
/// outright: a marker that cannot be resolved in this turn is not offered to
/// the model as though it could be.
///
/// The fact that the claim *was* grounded survives — `[cited earlier]` says so
/// — and this turn can still search for itself.
fn neutralise_evidence_markers(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;

    while let Some(open) = rest.find("[E") {
        let after = &rest[open + 2..];
        let digits: String = after.chars().take_while(char::is_ascii_digit).collect();
        let closes = !digits.is_empty() && after[digits.len()..].starts_with(']');

        out.push_str(&rest[..open]);
        if closes {
            out.push_str("[cited earlier]");
            rest = &after[digits.len() + 1..];
        } else {
            // Not a marker — `[Every`, `[E]`, a stray bracket. Left alone.
            out.push_str("[E");
            rest = after;
        }
    }
    out.push_str(rest);
    out
}

/// One turn's text, with a line saying what its tools did.
///
/// ## Why the tools have to be in the text
///
/// [`ContextTurn`] carries a role and words, and that is all that crosses to
/// the model. Nothing writes tool calls or tool results into the transcript, so
/// a turn that searched three documents and produced an approval note reached
/// the *next* turn as whatever sentence it happened to end with. A person then
/// asking "add the vendor figures to it" was asking about a file the model had
/// no record of existing.
///
/// The summary is one line — `[this turn used: artifact.create_approval_note,
/// knowledge.search_authorized ×3]` — written where the calls were known, at
/// the end of the run. Appending it here rather than storing it inside
/// `content` keeps the message the person sees exactly what the model wrote.
///
/// A message with no summary is returned unchanged, which is every user turn,
/// every turn that called no tool, and every message written before the field
/// existed.
fn with_tool_summary(message: &Message) -> String {
    // Markers first: they belong to the run that issued them, and this message
    // is from an earlier one. See `neutralise_evidence_markers`.
    let content = neutralise_evidence_markers(message.content.trim());
    match message.tool_summary.as_deref().map(str::trim) {
        Some(summary) if !summary.is_empty() => format!("{content}
{summary}"),
        _ => content,
    }
}

/// The wire role for a message that passed [`is_eligible`].
fn role_of(message: &Message) -> Option<&'static str> {
    match message.role {
        MessageRole::User => Some("user"),
        MessageRole::Assistant => Some("assistant"),
        MessageRole::System => None,
    }
}

/// The messages that are history for the turn streaming into `cell_message_id`.
///
/// Everything from that assistant cell onward is this turn and not history. So
/// is the user message immediately before it, which is this turn's question —
/// `ConversationStore::append_user_turn` writes the pair together, so the cell's
/// position is what identifies the question, not a string comparison against the
/// prompt.
///
/// Comparing prompt text would be wrong in a way that is easy to miss: by the
/// time a run is composed, the prompt has attachment text folded into it, so it
/// no longer equals the message that was stored. A position cannot drift.
///
/// A cell that is not in this conversation returns everything, which is the safe
/// direction: a caller that reserved no cell has no turn in the transcript to
/// exclude.
fn history_slice<'a>(conversation: &'a Conversation, cell_message_id: &str) -> &'a [Message] {
    let Some(cell) = conversation
        .messages
        .iter()
        .position(|message| message.id == cell_message_id)
    else {
        return &conversation.messages;
    };
    // The question that goes with this cell, when there is one.
    // `append_user_turn` pushes user then assistant, so it is directly before.
    let end = match cell.checked_sub(1) {
        Some(before) if conversation.messages[before].role == MessageRole::User => before,
        _ => cell,
    };
    &conversation.messages[..end]
}

/// Builds the history for one turn, fitted to what the destination model affords.
///
/// `budget_tokens` is what is genuinely free for history *after* the system
/// prompt, this turn's question and any attached documents have been charged —
/// the caller owns that arithmetic because the caller is what composed them.
///
/// Newest first, then reversed. Dropping from the old end is what everyone
/// expects of a conversation, and it is also the only end that can be dropped
/// without breaking the thing history is for: the message the question refers to
/// is nearly always the most recent one.
///
/// A budget of zero yields nothing and reports every eligible message as
/// dropped, rather than squeezing one message in past a budget that said there
/// was no room.
pub fn fit(
    conversation: &Conversation,
    cell_message_id: &str,
    budget_tokens: u32,
    pinned: &[String],
) -> FittedContext {
    let history = history_slice(conversation, cell_message_id);
    let eligible: Vec<&Message> = history.iter().filter(|m| is_eligible(m)).collect();

    let mut kept: Vec<ContextTurn> = Vec::new();
    let mut spent: u32 = 0;
    let mut dropped: u32 = 0;
    // Messages the budget could not hold but a person asked to keep, oldest
    // first. Collected while walking backwards and prepended at the end, so
    // they arrive in the order they were said rather than the order they were
    // rescued in.
    let mut rescued: Vec<ContextTurn> = Vec::new();
    let mut trimming = false;

    for message in eligible.iter().rev() {
        let Some(role) = role_of(message) else {
            continue;
        };
        let content = with_tool_summary(message);
        let content = content.as_str();
        let cost = estimate_tokens(content);
        let held = is_pinned(message, content, pinned);

        // Once the budget is gone, only pinned messages are still collected.
        // Everything else is counted as dropped.
        if trimming {
            if held {
                spent = spent.saturating_add(cost);
                rescued.push(ContextTurn {
                    role,
                    content: content.to_string(),
                });
            } else {
                dropped = dropped.saturating_add(1);
            }
            continue;
        }

        // `>` rather than `>=`, so a message that exactly fills the remaining
        // budget is kept. Saturating, because a caller that passes a budget
        // smaller than one message must get zero rather than a wrap-around.
        if spent.saturating_add(cost) > budget_tokens {
            // Everything older is dropped too: stopping at the first message
            // that does not fit, rather than skipping it and trying the next,
            // keeps the kept set a contiguous tail. A history with a hole in the
            // middle reads to a model as a conversation where somebody's reply
            // vanished, which is worse than a shorter one.
            //
            // A *pinned* message is the one exception, and it is a deliberate
            // hole: somebody said the task depends on it, and carrying it out of
            // order is better than dropping the thing they asked to keep. The
            // marker the runtime prepends says the history was shortened, so the
            // model is not told this is a continuous transcript.
            trimming = true;
            if held {
                spent = spent.saturating_add(cost);
                rescued.push(ContextTurn {
                    role,
                    content: content.to_string(),
                });
            } else {
                dropped = dropped.saturating_add(1);
            }
            continue;
        }
        spent = spent.saturating_add(cost);
        kept.push(ContextTurn {
            role,
            content: content.to_string(),
        });
    }

    kept.reverse();
    rescued.reverse();

    // Pinned survivors go ahead of the contiguous tail, which is where they were
    // said: everything rescued is older than everything kept.
    //
    // And the join between them is marked, because it is a hole. The kept tail
    // is contiguous by construction, so a model reading this sees one
    // conversation — but between a rescued pin from turn 1 and the tail
    // starting at turn 9 there are seven turns that were dropped. The runtime
    // prepends a note saying *how many* messages did not fit; nothing said
    // *where*, so the ordering the model was shown is one that never happened.
    //
    // A `user` turn for the same reason the runtime's own marker is one: a
    // system-role message can be reordered away from the thing it describes,
    // and this has to stay exactly between the two stretches it separates.
    if !rescued.is_empty() && !kept.is_empty() {
        rescued.push(ContextTurn {
            role: "user",
            content: "[The message(s) above were kept because they were pinned. What follows                       is a later, continuous part of the conversation — the turns in between                       did not fit and are not shown.]"
                .to_string(),
        });
    }

    rescued.extend(kept);
    FittedContext {
        turns: rescued,
        dropped,
        tokens: spent,
    }
}

/// Whether a person asked for this message to be kept.
///
/// Matched two ways, because the context meter draws two kinds of row and a pin
/// has to mean the same thing whichever one it was pressed on:
///
/// - **By message id.** A row for a turn.
/// - **By something the message names.** A document's content hash appears in
///   the text of the turn that attached it, so pinning the drawing keeps the
///   turn that carries it.
///
/// Case-insensitive, matching `pruneStaleToolResults` in the runtime, so a pin
/// cannot be honoured by one side and dropped by the other over how an id
/// happened to be spelled.
fn is_pinned(message: &Message, content: &str, pinned: &[String]) -> bool {
    if pinned.is_empty() {
        return false;
    }
    let upper = content.to_uppercase();
    pinned.iter().any(|id| {
        let id = id.trim();
        // An empty id matches everything under `contains`, which would pin the
        // entire history from one blank string. The store drops these on the
        // way in; this is the second guard, because the cost of being wrong
        // here is a window that fills and a turn that fails.
        !id.is_empty()
            && (message.id.eq_ignore_ascii_case(id) || upper.contains(&id.to_uppercase()))
    })
}

/// The share of a model's window that history may occupy.
///
/// A third, for the same reason [`crate::ai_engine::ocr_budget`] gives a
/// document a half of what is free: a history permitted to fill everything that
/// is left leaves no room for the answer, and the run compacts on its first turn
/// — throwing away the history it was just given, having spent the window
/// carrying it.
///
/// A third rather than a half because history is charged *after* documents, and
/// a turn that attached a drawing has already spent the larger share on it.
pub const HISTORY_SHARE: f64 = 1.0 / 3.0;

/// How many tokens history may spend, given what the turn has already committed.
///
/// A `window` of zero means nobody told this process the model's context size.
/// The honest answer there is no history rather than unbounded history: an
/// unknown window is not evidence of room, and the failure of guessing wrong is
/// a request the inference server refuses outright.
pub fn budget_for(window: u32, committed: u32) -> u32 {
    if window == 0 {
        return 0;
    }
    let free = window.saturating_sub(committed);
    (f64::from(free) * HISTORY_SHARE) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn message(id: &str, role: MessageRole, content: &str, status: MessageStatus) -> Message {
        Message {
            id: id.to_string(),
            conversation_id: "c1".to_string(),
            role,
            content: content.to_string(),
            status,
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
        }
    }

    fn conversation(messages: Vec<Message>) -> Conversation {
        Conversation {
            id: "c1".to_string(),
            owner_user_id: "owner-1".to_string(),
            title: "t".to_string(),
            created_at: "2026-01-01T00:00:00Z".to_string(),
            last_activity_at: "2026-01-01T00:00:00Z".to_string(),
            messages,
            runs: Vec::new(),
            compactions: 0,
            pinned_context: Vec::new(),
            routed_role: None,
            routed_model_id: None,
        }
    }

    /// The whole point of the module: a second turn can see the first.
    #[test]
    fn a_prior_exchange_becomes_history() {
        let convo = conversation(vec![
            message(
                "u1",
                MessageRole::User,
                "What is the rating?",
                MessageStatus::Done,
            ),
            message(
                "a1",
                MessageRole::Assistant,
                "Class 300.",
                MessageStatus::Done,
            ),
            message("u2", MessageRole::User, "And the flange?", MessageStatus::Done),
            message("a2", MessageRole::Assistant, "", MessageStatus::Streaming),
        ]);
        let fitted = fit(&convo, "a2", 10_000, &[]);
        assert_eq!(
            fitted.turns,
            vec![
                ContextTurn {
                    role: "user",
                    content: "What is the rating?".into()
                },
                ContextTurn {
                    role: "assistant",
                    content: "Class 300.".into()
                },
            ]
        );
        assert_eq!(fitted.dropped, 0);
    }

    /// The invariant the audit named. The question this turn is asking must not
    /// be in the history, because the runtime submits it separately — and a
    /// question that appears in both is asked twice.
    #[test]
    fn this_turns_question_is_never_in_the_history() {
        let convo = conversation(vec![
            message("u1", MessageRole::User, "First.", MessageStatus::Done),
            message("a1", MessageRole::Assistant, "Answered.", MessageStatus::Done),
            message(
                "u2",
                MessageRole::User,
                "The new question.",
                MessageStatus::Done,
            ),
            message("a2", MessageRole::Assistant, "", MessageStatus::Streaming),
        ]);
        let fitted = fit(&convo, "a2", 10_000, &[]);
        assert!(
            !fitted
                .turns
                .iter()
                .any(|turn| turn.content == "The new question."),
            "the live question leaked into the history: {:?}",
            fitted.turns
        );
    }

    /// The history stops at the previous answer. The live question follows it
    /// through the prompt path, exactly once.
    #[test]
    fn the_history_stops_before_the_turn_being_asked() {
        let convo = conversation(vec![
            message("u1", MessageRole::User, "First.", MessageStatus::Done),
            message("a1", MessageRole::Assistant, "Answered.", MessageStatus::Done),
            message("u2", MessageRole::User, "Second.", MessageStatus::Done),
            message("a2", MessageRole::Assistant, "", MessageStatus::Streaming),
        ]);
        let fitted = fit(&convo, "a2", 10_000, &[]);
        assert_eq!(fitted.turns.last().unwrap().role, "assistant");
        assert_eq!(fitted.turns.len(), 2);
    }

    #[test]
    fn a_streaming_message_is_not_history() {
        let convo = conversation(vec![
            message("u1", MessageRole::User, "Q", MessageStatus::Done),
            message(
                "a1",
                MessageRole::Assistant,
                "half a sen",
                MessageStatus::Streaming,
            ),
            message("u2", MessageRole::User, "Q2", MessageStatus::Done),
            message("a2", MessageRole::Assistant, "", MessageStatus::Streaming),
        ]);
        let fitted = fit(&convo, "a2", 10_000, &[]);
        assert_eq!(
            fitted.turns,
            vec![ContextTurn {
                role: "user",
                content: "Q".into()
            }]
        );
    }

    #[test]
    fn a_failed_turn_is_not_history() {
        let convo = conversation(vec![
            message("u1", MessageRole::User, "Q", MessageStatus::Done),
            message("a1", MessageRole::Assistant, "", MessageStatus::Failed),
            message("u2", MessageRole::User, "Q2", MessageStatus::Done),
            message("a2", MessageRole::Assistant, "", MessageStatus::Streaming),
        ]);
        let fitted = fit(&convo, "a2", 10_000, &[]);
        assert_eq!(fitted.turns.len(), 1);
        assert_eq!(fitted.turns[0].role, "user");
    }

    #[test]
    fn the_welcome_banner_is_not_conversation() {
        let convo = conversation(vec![
            message(
                "s1",
                MessageRole::System,
                "Arjun is ready.",
                MessageStatus::Done,
            ),
            message("u1", MessageRole::User, "Q", MessageStatus::Done),
            message("a1", MessageRole::Assistant, "A", MessageStatus::Done),
            message("u2", MessageRole::User, "Q2", MessageStatus::Done),
            message("a2", MessageRole::Assistant, "", MessageStatus::Streaming),
        ]);
        let fitted = fit(&convo, "a2", 10_000, &[]);
        assert!(fitted.turns.iter().all(|t| t.role != "system"));
        assert_eq!(fitted.turns.len(), 2);
    }

    /// Trimming happens from the old end, and says how much it took.
    #[test]
    fn a_tight_budget_keeps_the_newest_and_reports_the_rest() {
        let long = "x".repeat(4_000); // ~1000 tokens each
        let convo = conversation(vec![
            message("u1", MessageRole::User, &long, MessageStatus::Done),
            message("a1", MessageRole::Assistant, &long, MessageStatus::Done),
            message(
                "u2",
                MessageRole::User,
                "recent question",
                MessageStatus::Done,
            ),
            message(
                "a2",
                MessageRole::Assistant,
                "recent answer",
                MessageStatus::Done,
            ),
            message("u3", MessageRole::User, "live", MessageStatus::Done),
            message("a3", MessageRole::Assistant, "", MessageStatus::Streaming),
        ]);
        let fitted = fit(&convo, "a3", 100, &[]);
        assert_eq!(
            fitted.turns,
            vec![
                ContextTurn {
                    role: "user",
                    content: "recent question".into()
                },
                ContextTurn {
                    role: "assistant",
                    content: "recent answer".into()
                },
            ]
        );
        assert_eq!(fitted.dropped, 2, "the two long messages were dropped");
    }

    #[test]
    fn a_budget_of_zero_carries_nothing_and_says_how_much_it_dropped() {
        let convo = conversation(vec![
            message("u1", MessageRole::User, "Q", MessageStatus::Done),
            message("a1", MessageRole::Assistant, "A", MessageStatus::Done),
            message("u2", MessageRole::User, "live", MessageStatus::Done),
            message("a2", MessageRole::Assistant, "", MessageStatus::Streaming),
        ]);
        let fitted = fit(&convo, "a2", 0, &[]);
        assert!(fitted.turns.is_empty());
        assert_eq!(fitted.dropped, 2);
    }

    /// The kept set is a contiguous tail. A hole in the middle would read to the
    /// model as a conversation where somebody's reply disappeared.
    #[test]
    fn trimming_never_leaves_a_hole_in_the_middle() {
        let long = "y".repeat(8_000);
        let convo = conversation(vec![
            message("u1", MessageRole::User, "old and short", MessageStatus::Done),
            message("a1", MessageRole::Assistant, &long, MessageStatus::Done),
            message("u2", MessageRole::User, "new and short", MessageStatus::Done),
            message(
                "a2",
                MessageRole::Assistant,
                "also short",
                MessageStatus::Done,
            ),
            message("u3", MessageRole::User, "the live question", MessageStatus::Done),
            message("a3", MessageRole::Assistant, "", MessageStatus::Streaming),
        ]);
        let fitted = fit(&convo, "a3", 50, &[]);
        // "old and short" fits the budget on its own, but the long message
        // between it and the tail does not — so it is dropped with everything
        // older, rather than being reattached across the gap.
        assert_eq!(
            fitted.turns,
            vec![
                ContextTurn {
                    role: "user",
                    content: "new and short".into()
                },
                ContextTurn {
                    role: "assistant",
                    content: "also short".into()
                },
            ]
        );
        assert_eq!(fitted.dropped, 2);
    }

    /// A pinned message is carried even when the budget has run out.
    ///
    /// This is the destination-context half of the pin. Protecting a message
    /// from the compactor while the *history budget* had already dropped it on
    /// the way in would be the same control failing one step earlier — and the
    /// panel would show it protected throughout.
    #[test]
    fn a_pinned_message_survives_a_budget_that_would_have_dropped_it() {
        let long = "x".repeat(4_000); // ~1000 tokens
        let convo = conversation(vec![
            message("u1", MessageRole::User, "the pinned question", MessageStatus::Done),
            message("a1", MessageRole::Assistant, &long, MessageStatus::Done),
            message("u2", MessageRole::User, "recent", MessageStatus::Done),
            message("a2", MessageRole::Assistant, "answer", MessageStatus::Done),
            message("u3", MessageRole::User, "live", MessageStatus::Done),
            message("a3", MessageRole::Assistant, "", MessageStatus::Streaming),
        ]);

        let without = fit(&convo, "a3", 100, &[]);
        assert!(
            !without.turns.iter().any(|t| t.content == "the pinned question"),
            "the fixture must actually drop it when nothing is pinned"
        );

        let held = fit(&convo, "a3", 100, &["u1".to_string()]);
        assert!(
            held.turns.iter().any(|t| t.content == "the pinned question"),
            "a pinned message was dropped: {:?}",
            held.turns
        );
    }

    /// Pinned by something the message *names* — a document's content hash
    /// appears in the turn that attached it, so pinning the drawing keeps the
    /// turn carrying it. The meter draws rows of both kinds and a pin has to
    /// mean the same thing whichever one it was pressed on.
    #[test]
    fn a_message_is_pinned_by_a_document_id_it_mentions() {
        let sha = "ab".repeat(32);
        let long = "y".repeat(4_000);
        let convo = conversation(vec![
            message(
                "u1",
                MessageRole::User,
                &format!("here is the drawing {sha}"),
                MessageStatus::Done,
            ),
            message("a1", MessageRole::Assistant, &long, MessageStatus::Done),
            message("u2", MessageRole::User, "recent", MessageStatus::Done),
            message("a2", MessageRole::Assistant, "answer", MessageStatus::Done),
            message("u3", MessageRole::User, "live", MessageStatus::Done),
            message("a3", MessageRole::Assistant, "", MessageStatus::Streaming),
        ]);

        let fitted = fit(&convo, "a3", 100, &[sha.clone()]);
        assert!(fitted.turns.iter().any(|t| t.content.contains(&sha)));
    }

    /// The rescued messages stay in the order they were said, ahead of the
    /// contiguous tail — everything rescued is older than everything kept.
    #[test]
    fn a_rescued_message_is_placed_where_it_was_said() {
        let long = "z".repeat(4_000);
        let convo = conversation(vec![
            message("u1", MessageRole::User, "oldest and pinned", MessageStatus::Done),
            message("a1", MessageRole::Assistant, &long, MessageStatus::Done),
            message("u2", MessageRole::User, "recent", MessageStatus::Done),
            message("a2", MessageRole::Assistant, "answer", MessageStatus::Done),
            message("u3", MessageRole::User, "live", MessageStatus::Done),
            message("a3", MessageRole::Assistant, "", MessageStatus::Streaming),
        ]);

        let fitted = fit(&convo, "a3", 100, &["u1".to_string()]);
        assert_eq!(fitted.turns.first().unwrap().content, "oldest and pinned");
        assert_eq!(fitted.turns.last().unwrap().content, "answer");
    }

    /// A blank id would match every message under a substring test and pin the
    /// entire history from one empty string, filling the window.
    #[test]
    fn a_blank_pin_protects_nothing() {
        let long = "w".repeat(8_000);
        let convo = conversation(vec![
            message("u1", MessageRole::User, &long, MessageStatus::Done),
            message("u2", MessageRole::User, "recent", MessageStatus::Done),
            message("a2", MessageRole::Assistant, "answer", MessageStatus::Done),
            message("u3", MessageRole::User, "live", MessageStatus::Done),
            message("a3", MessageRole::Assistant, "", MessageStatus::Streaming),
        ]);

        let fitted = fit(&convo, "a3", 100, &["".to_string(), "  ".to_string()]);
        assert!(!fitted.turns.iter().any(|t| t.content.len() > 1_000));
        assert_eq!(fitted.dropped, 1);
    }

    #[test]
    fn an_unknown_window_affords_no_history() {
        assert_eq!(budget_for(0, 0), 0);
    }

    #[test]
    fn an_over_committed_turn_affords_no_history() {
        assert_eq!(budget_for(8_000, 12_000), 0, "saturating, not wrapping");
    }

    #[test]
    fn history_never_takes_more_than_a_third_of_what_is_free() {
        for committed in [0u32, 1_000, 8_000, 30_000] {
            let free = 32_000u32.saturating_sub(committed);
            assert!(budget_for(32_000, committed) <= free / 3 + 1);
        }
    }

    /// A caller that reserved no cell of its own still gets a coherent history
    /// rather than a panic or an empty one.
    #[test]
    fn an_unknown_cell_yields_the_whole_transcript() {
        let convo = conversation(vec![
            message("u1", MessageRole::User, "Q", MessageStatus::Done),
            message("a1", MessageRole::Assistant, "A", MessageStatus::Done),
        ]);
        let fitted = fit(&convo, "not-in-this-conversation", 10_000, &[]);
        assert_eq!(fitted.turns.len(), 2);
    }

    #[test]
    fn a_first_turn_has_no_history() {
        let convo = conversation(vec![
            message(
                "u1",
                MessageRole::User,
                "the very first question",
                MessageStatus::Done,
            ),
            message("a1", MessageRole::Assistant, "", MessageStatus::Streaming),
        ]);
        let fitted = fit(&convo, "a1", 10_000, &[]);
        assert!(fitted.is_empty());
        assert_eq!(fitted.dropped, 0);
    }
}

#[cfg(test)]
mod tool_summary_tests {
    use super::*;

    fn assistant(content: &str, summary: Option<&str>) -> Message {
        Message {
            id: "a-1".to_string(),
            conversation_id: "c1".to_string(),
            role: MessageRole::Assistant,
            content: content.to_string(),
            status: MessageStatus::Done,
            run_id: Some("run-1".to_string()),
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
            tool_summary: summary.map(str::to_string),
        }
    }

    /// The defect this closes.
    ///
    /// History crossing to the model is `{role, content}`. A turn that wrote an
    /// approval note reached the next turn as the sentence it ended with, so
    /// "add the vendor figures to it" was a question about a file the model had
    /// no record of.
    #[test]
    fn a_turns_tools_travel_with_its_words() {
        let carried = with_tool_summary(&assistant(
            "I've prepared the note.",
            Some("[this turn used: artifact.create_approval_note]"),
        ));

        assert!(carried.contains("I've prepared the note."), "{carried}");
        assert!(
            carried.contains("artifact.create_approval_note"),
            "the next turn cannot tell a file was written: {carried}"
        );
    }

    /// A marker from an earlier turn must not read as one this turn can use.
    ///
    /// `[E1]…[En]` are numbered from one, per run, and every chat turn is its
    /// own run. So turn 3's `[E2]` does not merely dangle in turn 4 — turn 4
    /// has its own `[E2]`, pointing at a different passage, and the verifier
    /// resolves against that one. A model copying its own last answer forward
    /// would produce a citation that passes checking and is wrong.
    #[test]
    fn an_earlier_turns_citations_cannot_be_reused_by_this_one() {
        let carried = with_tool_summary(&assistant(
            "The seal is rated to 65 mm [E2], and the SOP requires annual inspection [E11].",
            None,
        ));

        assert!(!carried.contains("[E2]"), "{carried}");
        assert!(!carried.contains("[E11]"), "{carried}");
        // The claim was grounded, and saying so is not the same as offering a
        // number this turn can resolve.
        assert_eq!(carried.matches("[cited earlier]").count(), 2, "{carried}");
        assert!(carried.contains("rated to 65 mm"), "{carried}");
    }

    /// Only actual markers. A bracket is not a citation.
    #[test]
    fn text_that_merely_looks_like_a_marker_is_left_alone() {
        for text in ["[Every] valve", "[E] alone", "[E2x] not a marker", "an [Edge] case"] {
            let carried = with_tool_summary(&assistant(text, None));
            assert_eq!(carried, text, "rewrote {text:?}");
        }
    }

    /// A turn that only talked adds nothing.
    ///
    /// A line saying "no tools were used" would cost window on every turn of
    /// every conversation to say nothing, which is how a signal becomes noise.
    #[test]
    fn a_turn_that_used_no_tool_is_unchanged() {
        assert_eq!(
            with_tool_summary(&assistant("Yes, that is right.", None)),
            "Yes, that is right."
        );
        assert_eq!(
            with_tool_summary(&assistant("Yes, that is right.", Some("   "))),
            "Yes, that is right."
        );
    }

    /// And the person's own view of the message is untouched.
    #[test]
    fn the_summary_is_added_for_the_model_not_stored_in_the_message() {
        let message = assistant("Done.", Some("[this turn used: sandbox.run_code]"));
        assert_eq!(message.content, "Done.", "the transcript was rewritten");
        assert!(with_tool_summary(&message).contains("sandbox.run_code"));
    }
}
