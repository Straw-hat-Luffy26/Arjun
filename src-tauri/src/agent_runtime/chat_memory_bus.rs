//! Model-agnostic chat memory bus.
//!
//! ## The problem this solves
//!
//! Every prior turn of context was fitted against a single model's physical
//! window, and anything that did not fit was permanently dropped from the turn.
//! A conversation that ran for fifty messages on a 32k model carried at most
//! 24k tokens of history — and when the orchestrator switched to a model with a
//! smaller window, most of that history was silently abandoned.
//!
//! ## The architecture
//!
//! The chat memory bus separates two concerns:
//!
//! 1. **Retention** — how much history is *kept*. This belongs to the
//!    conversation, not the model, and the limit is 1,000,000 tokens (the
//!    full extent of what fits on disk).
//! 2. **Projection** — how much history is *sent to a model* on any given
//!    turn. This is bounded by the model's physical context window and is
//!    computed freshly for every turn.
//!
//! Nothing is ever deleted from retention. Projection is non-destructive: it
//! selects a subset, and the full record remains available for the next turn,
//! the next model, or the next session.

use super::conversations::{Conversation, Message, MessageRole, MessageStatus};
use crate::ai_engine::ocr_budget::estimate_tokens;

/// The maximum number of tokens retained in the chat memory bus.
///
/// This is the *chat* context limit, distinct from any model's physical window.
/// 1,000,000 tokens of UTF-8 text is roughly 4 MB of storage — a file smaller
/// than most of the PDFs this product routinely ingests.
pub const CHAT_RETENTION_LIMIT: u32 = 1_000_000;

/// How a projected turn is classified for the memory bus.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurnPriority {
    /// Recent turns: always included first (recency anchor).
    Recent,
    /// Turns the user pinned: never dropped.
    Pinned,
    /// Turns selected by keyword relevance to the current question.
    Relevant,
    /// Earlier turns included if budget remains.
    Background,
}

/// One turn ready for projection into a model's window.
#[derive(Debug, Clone)]
pub struct ProjectedTurn {
    /// Index into the full conversation message list.
    pub message_index: usize,
    /// `user` or `assistant`.
    pub role: &'static str,
    /// The text content.
    pub content: String,
    /// Estimated tokens this turn occupies.
    pub tokens: u32,
    /// Why this turn was selected.
    pub priority: TurnPriority,
}

/// The result of projecting the chat memory bus into a model's window.
#[derive(Debug, Clone)]
pub struct Projection {
    /// Turns to send, oldest first.
    pub turns: Vec<ProjectedTurn>,
    /// Total estimated tokens in the projection.
    pub projected_tokens: u32,
    /// How many eligible turns were retained but not projected (still safe in
    /// the conversation store).
    pub retained_not_projected: u32,
    /// Total tokens across the entire conversation history (the full retention).
    pub total_retained_tokens: u32,
}

/// Whether a stored message is eligible for the memory bus.
///
/// The rules are identical to `turn_context::is_eligible`: system messages,
/// streaming messages, failed messages and empty messages are excluded.
fn is_eligible(message: &Message) -> bool {
    if message.role == MessageRole::System {
        return false;
    }
    if message.status != MessageStatus::Done {
        return false;
    }
    !message.content.trim().is_empty()
}

/// The wire role for an eligible message.
fn role_of(message: &Message) -> Option<&'static str> {
    match message.role {
        MessageRole::User => Some("user"),
        MessageRole::Assistant => Some("assistant"),
        MessageRole::System => None,
    }
}

/// Simple keyword overlap score between a question and a past message.
///
/// Returns a value between 0.0 and 1.0. This is deliberately simple —
/// a full embedding-based retrieval is a future enhancement. Even naive
/// keyword overlap rescues turns that mention the same entities as the
/// current question, which is the single most common reason a person
/// asks a follow-up.
fn relevance_score(question_words: &[String], message: &str) -> f64 {
    if question_words.is_empty() {
        return 0.0;
    }
    let lower = message.to_lowercase();
    let hits = question_words
        .iter()
        .filter(|word| lower.contains(word.as_str()))
        .count();
    hits as f64 / question_words.len() as f64
}

/// Extract meaningful keywords from a question (words with 4+ chars).
fn extract_keywords(question: &str) -> Vec<String> {
    question
        .to_lowercase()
        .split_whitespace()
        .filter(|w| w.len() >= 4)
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
        .filter(|w| !w.is_empty())
        .collect()
}

/// Projects the full conversation history into a model's physical window.
///
/// `budget_tokens` is the number of tokens available for history after the
/// system prompt, documents, tools, and reply reserve have been charged.
///
/// `cell_message_id` identifies the assistant cell for this turn — everything
/// from that cell onward is this turn and is excluded from history.
///
/// `pinned` is the set of message IDs the user has explicitly protected.
///
/// `question` is the current turn's prompt, used for relevance scoring.
///
/// ## Strategy
///
/// 1. **Pinned turns** are included first (user-protected, never dropped).
/// 2. **Recent turns** are included next, newest first, up to 60% of budget.
/// 3. **Relevant turns** from earlier history are scored by keyword overlap
///    with the question and included in descending relevance order.
/// 4. **Background turns** fill any remaining budget, oldest to newest.
///
/// The result is always sorted oldest-first for the model's benefit.
pub fn project(
    conversation: &Conversation,
    cell_message_id: &str,
    budget_tokens: u32,
    pinned: &[String],
    question: &str,
) -> Projection {
    if budget_tokens == 0 {
        let total = total_retained_tokens(conversation);
        let eligible_count = eligible_before_cell(conversation, cell_message_id).len() as u32;
        return Projection {
            turns: Vec::new(),
            projected_tokens: 0,
            retained_not_projected: eligible_count,
            total_retained_tokens: total,
        };
    }

    let history = eligible_before_cell(conversation, cell_message_id);
    let total = total_retained_tokens(conversation);
    let keywords = extract_keywords(question);

    // Phase 1: Classify every eligible turn.
    let mut classified: Vec<(usize, &Message, TurnPriority, u32, f64)> = history
        .iter()
        .map(|(idx, msg)| {
            let tokens = estimate_tokens(&msg.content);
            let relevance = relevance_score(&keywords, &msg.content);
            let priority = if pinned.iter().any(|p| p == &msg.id) {
                TurnPriority::Pinned
            } else {
                TurnPriority::Background // classified further below
            };
            (*idx, *msg, priority, tokens, relevance)
        })
        .collect();

    // Phase 2: Mark recent turns (last 60% of budget or last N turns).
    let recency_budget = (budget_tokens as f64 * 0.6) as u32;
    let mut recency_spent: u32 = 0;
    for item in classified.iter_mut().rev() {
        if item.2 == TurnPriority::Pinned {
            continue; // already classified
        }
        if recency_spent + item.3 <= recency_budget {
            item.2 = TurnPriority::Recent;
            recency_spent += item.3;
        }
    }

    // Phase 3: Mark relevant turns (relevance > 0.3, not already recent/pinned).
    for item in classified.iter_mut() {
        if item.2 == TurnPriority::Background && item.4 > 0.3 {
            item.2 = TurnPriority::Relevant;
        }
    }

    // Phase 4: Select turns in priority order until budget is exhausted.
    let priority_order = [
        TurnPriority::Pinned,
        TurnPriority::Recent,
        TurnPriority::Relevant,
        TurnPriority::Background,
    ];

    let mut selected: Vec<(usize, &Message, TurnPriority, u32)> = Vec::new();
    let mut remaining = budget_tokens;

    for priority in &priority_order {
        // For relevant turns, sort by relevance score descending.
        let mut candidates: Vec<&(usize, &Message, TurnPriority, u32, f64)> = classified
            .iter()
            .filter(|item| item.2 == *priority)
            .collect();

        if *priority == TurnPriority::Relevant {
            candidates.sort_by(|a, b| b.4.partial_cmp(&a.4).unwrap_or(std::cmp::Ordering::Equal));
        } else if *priority == TurnPriority::Background {
            // Fill background newest-first rather than oldest-first so a contiguous
            // recent window is formed rather than jumping back to the dawn of time.
            candidates.reverse();
        }

        for candidate in candidates {
            if candidate.3 > remaining {
                continue; // skip turns that don't fit
            }
            selected.push((candidate.0, candidate.1, candidate.2, candidate.3));
            remaining = remaining.saturating_sub(candidate.3);
        }
    }

    // Sort by original position (oldest first) for the model.
    selected.sort_by_key(|item| item.0);

    let projected_tokens: u32 = selected.iter().map(|s| s.3).sum();
    let retained_not_projected =
        (classified.len() as u32).saturating_sub(selected.len() as u32);

    let turns = selected
        .into_iter()
        .filter_map(|(idx, msg, priority, tokens)| {
            let role = role_of(msg)?;
            Some(ProjectedTurn {
                message_index: idx,
                role,
                content: msg.content.trim().to_string(),
                tokens,
                priority,
            })
        })
        .collect();

    Projection {
        turns,
        projected_tokens,
        retained_not_projected,
        total_retained_tokens: total,
    }
}

/// Eligible messages before the current turn's cell.
fn eligible_before_cell<'a>(
    conversation: &'a Conversation,
    cell_message_id: &str,
) -> Vec<(usize, &'a Message)> {
    let cell_pos = conversation
        .messages
        .iter()
        .position(|m| m.id == cell_message_id);

    let end = match cell_pos {
        Some(pos) => {
            // Exclude the user question paired with this cell too.
            match pos.checked_sub(1) {
                Some(before) if conversation.messages[before].role == MessageRole::User => before,
                _ => pos,
            }
        }
        None => conversation.messages.len(),
    };

    conversation.messages[..end]
        .iter()
        .enumerate()
        .filter(|(_, m)| is_eligible(m))
        .collect()
}

/// Total estimated tokens across all eligible messages in the conversation.
fn total_retained_tokens(conversation: &Conversation) -> u32 {
    conversation
        .messages
        .iter()
        .filter(|m| is_eligible(m))
        .map(|m| estimate_tokens(&m.content))
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent_runtime::conversations::{
        Conversation, Message, MessageRole, MessageStatus,
    };

    fn make_message(id: &str, role: MessageRole, content: &str) -> Message {
        Message {
            id: id.to_string(),
            conversation_id: "conv-1".to_string(),
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
        }
    }

    fn make_conversation(messages: Vec<Message>) -> Conversation {
        Conversation {
            id: "conv-1".to_string(),
            title: "Test Conversation".to_string(),
            messages,
            runs: vec![],
            owner_user_id: "user-1".to_string(),
            created_at: "2026-01-01T00:00:00Z".to_string(),
            last_activity_at: "2026-01-01T00:00:00Z".to_string(),
            compactions: 0,
            pinned_context: vec![],
            routed_role: None,
            routed_model_id: None,
        }
    }

    #[test]
    fn empty_conversation_produces_empty_projection() {
        let conv = make_conversation(vec![]);
        let proj = project(&conv, "cell-1", 4096, &[], "hello");
        assert!(proj.turns.is_empty());
        assert_eq!(proj.projected_tokens, 0);
        assert_eq!(proj.retained_not_projected, 0);
    }

    #[test]
    fn zero_budget_retains_but_projects_nothing() {
        let conv = make_conversation(vec![
            make_message("u1", MessageRole::User, "What is the pressure rating?"),
            make_message("a1", MessageRole::Assistant, "The pressure rating is 150 PSI."),
            make_message("u2", MessageRole::User, "Can you explain more?"),
            make_message("cell", MessageRole::Assistant, ""),
        ]);
        let proj = project(&conv, "cell", 0, &[], "explain more");
        assert!(proj.turns.is_empty());
        // The cell and u2 are this turn; u1 and a1 are eligible history.
        assert_eq!(proj.retained_not_projected, 2);
        assert!(proj.total_retained_tokens > 0);
    }

    #[test]
    fn pinned_turns_always_included() {
        let conv = make_conversation(vec![
            make_message("u1", MessageRole::User, "Important instruction: always use metric units"),
            make_message("a1", MessageRole::Assistant, "Understood, I will use metric units."),
            make_message("u2", MessageRole::User, "What is the temperature?"),
            make_message("a2", MessageRole::Assistant, "The temperature is 25°C."),
            make_message("u3", MessageRole::User, "New question"),
            make_message("cell", MessageRole::Assistant, ""),
        ]);
        // Pin the first user message.
        let proj = project(&conv, "cell", 100, &["u1".to_string()], "new question");
        assert!(proj.turns.iter().any(|t| t.content.contains("metric units")));
    }

    #[test]
    fn recent_turns_preferred_over_old() {
        let mut messages = vec![];
        for i in 0..20 {
            messages.push(make_message(
                &format!("u{i}"),
                MessageRole::User,
                &format!("User message number {i} with some padding text to consume tokens"),
            ));
            messages.push(make_message(
                &format!("a{i}"),
                MessageRole::Assistant,
                &format!("Assistant response number {i} with enough text to be meaningful"),
            ));
        }
        messages.push(make_message("uN", MessageRole::User, "final question"));
        messages.push(make_message("cell", MessageRole::Assistant, ""));

        let conv = make_conversation(messages);
        // Small budget: only recent turns should fit.
        let proj = project(&conv, "cell", 200, &[], "final question");
        // The most recent turns should be present.
        assert!(proj.turns.iter().any(|t| t.content.contains("number 19")
            || t.content.contains("number 18")));
        // Very old turns should NOT be present (budget too small).
        assert!(!proj.turns.iter().any(|t| t.content.contains("number 0")));
    }
}
