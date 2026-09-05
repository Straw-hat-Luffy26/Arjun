//! Waiting for a person.
//!
//! Three of the eight tools leave a trace outside the task — a file, a
//! document, an execution — and the gateway marks those `needs_approval`. Until
//! now the runtime turned that verdict into a refusal, because there was
//! nothing to wait on. That is the wrong shape: the model asked whether it may
//! act, and "a person has not looked yet" is not the same answer as "no".
//!
//! ## Why the waiting happens here and not in the runtime
//!
//! The obvious alternative is to answer `needsApproval` immediately and let the
//! Node runtime poll. That would put the pending state in two places and give
//! the runtime a reason to know about approvals at all — and the runtime is the
//! side that must not be trusted with policy.
//!
//! So `tool.authorize` simply takes longer when a person is involved. From the
//! loop's point of view a slow authorisation is indistinguishable from a slow
//! anything else; from the operator's, the request appears on the approvals
//! screen and the run continues when they decide. Neither side has to
//! understand the other's model of waiting.
//!
//! ## Why it polls
//!
//! [`ApprovalQueue`] is a plain mutex-guarded list, deliberately: it is read by
//! Tauri commands, the health panel and this module, and a notification channel
//! would make its lock ordering something to reason about. A quarter-second
//! poll costs nothing next to the minutes a human takes, and the queue stays
//! the simple thing that many callers can read safely.

use std::sync::Arc;
use std::time::Duration;

use crate::identity::Session;
use crate::orchestrator::approvals::{ApprovalQueue, ApprovalRequest};
use crate::orchestrator::tools::ToolName;

/// How often the queue is checked.
const POLL: Duration = Duration::from_millis(250);

/// How long a run waits before giving up on a person.
///
/// Long enough that an approver can finish what they were doing and come back;
/// short enough that a run started before lunch does not hold a model server
/// resident all afternoon. On expiry the model is told plainly that nobody
/// answered, which is something it can report rather than something it has to
/// infer from a hang.
const WAIT_LIMIT: Duration = Duration::from_secs(15 * 60);

/// What waiting came to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ApprovalOutcome {
    Approved { by: String },
    Rejected { by: String, because: String },
    /// Nobody decided within the limit.
    TimedOut,
    /// Missing or corrupt authority never becomes permission.
    Unavailable { detail: String },
}

impl ApprovalOutcome {
    /// The sentence handed back to the model when the answer is not yes.
    ///
    /// Written for a model to act on: it says what happened, and what the model
    /// should do about it. "Rejected" alone invites the same action again.
    pub fn refusal(&self) -> String {
        match self {
            ApprovalOutcome::Approved { .. } => String::new(),
            ApprovalOutcome::Unavailable { detail } => format!("Approval could not be verified: {detail}. The action did not happen and requires review."),
            ApprovalOutcome::Rejected { by, because } => format!(
                "{by} did not approve this action, because: {because}. It did not happen. Do not \
                 propose the same action again — address the objection first, or explain to the \
                 user why you cannot."
            ),
            ApprovalOutcome::TimedOut =>
                "Nobody responded to the approval request in time, so the action did not happen. \
                 Say so plainly rather than describing what it would have produced."
                    .to_string(),
        }
    }
}

/// Puts a proposed action in front of a person and waits for their answer.
#[allow(clippy::too_many_arguments)]
pub async fn await_decision(
    queue: &Arc<ApprovalQueue>,
    events: &Arc<crate::agent_runtime::events::TaskEventLog>,
    session: &Session,
    run_id: &str,
    tool: ToolName,
    summary: String,
    target: String,
    tool_call_id: &str,
    raw_arguments: &serde_json::Value,
) -> ApprovalOutcome {
    use crate::agent_runtime::events::{ApprovalStatus, DurableApproval};
    let id = approval_id(run_id, tool_call_id);
    let asked_at = chrono::Utc::now();
    let binding = serde_json::json!({ "tool": tool.as_str(), "args": raw_arguments });
    let proposed = DurableApproval::requested(id.clone(), run_id, tool.as_str(), target.clone(),
        &binding, summary.clone(), asked_at, Some(asked_at + chrono::Duration::seconds(WAIT_LIMIT.as_secs() as i64)));
    let durable = match events.approval(&id) {
        Ok(Some(existing)) if existing.run_id == run_id && existing.tool == tool.as_str()
            && existing.args_fingerprint == proposed.args_fingerprint => existing,
        Ok(Some(_)) => return ApprovalOutcome::Unavailable { detail: "The saved request describes different arguments".into() },
        Ok(None) => {
            if let Err(error) = events.record_approval(&proposed) {
                return ApprovalOutcome::Unavailable { detail: error };
            }
            proposed
        }
        Err(error) => return ApprovalOutcome::Unavailable { detail: error },
    };
    // A retry attaches to the original request and deadline. It does not reset
    // consent, generate a new question, or extend the approval window.
    queue.restore(vec![ApprovalRequest {
        id: id.clone(),
        task_id: run_id.to_string(),
        tool: tool.as_str().to_string(),
        target,
        arguments: vec![raw_arguments.to_string()],
        // Populated in a later phase, when a run carries the passages it relied
        // on. Empty is honest; inventing evidence to fill the field would make
        // the approval screen less trustworthy, not more.
        evidence: Vec::new(),
        expected_output: summary,
        consequences: tool.describe().to_string(),
        requested_by: session.user.id.clone(),
        requested_at: chrono::DateTime::parse_from_rfc3339(&durable.created_at).map(|at| at.with_timezone(&chrono::Utc)).unwrap_or(asked_at),
    }]);

    loop {
        let item = match events.approval(&id) {
            Ok(Some(item)) => item,
            _ => return ApprovalOutcome::Unavailable { detail: "The durable decision cannot be read".into() },
        };
        let now = chrono::Utc::now();
        if item.expires_at.as_ref().and_then(|at| chrono::DateTime::parse_from_rfc3339(at).ok()).is_none_or(|at| now >= at) {
            return ApprovalOutcome::TimedOut;
        }
        match item.status {
            ApprovalStatus::Approved => match item.authorises(&binding, now) {
                Ok(()) => return ApprovalOutcome::Approved { by: item.resolved_by.unwrap_or_default() },
                Err(error) => return ApprovalOutcome::Unavailable { detail: error.explain() },
            },
            ApprovalStatus::Rejected => return ApprovalOutcome::Rejected { by: item.resolved_by.unwrap_or_default(), because: item.resolution.unwrap_or_default() },
            ApprovalStatus::Pending => {}
        }
        tokio::time::sleep(POLL).await;
    }
}

pub(super) fn approval_id(run_id: &str, tool_call_id: &str) -> String {
    crate::agent_runtime::events::derive_key(run_id, "approval.v1", &serde_json::json!({ "toolCallId": tool_call_id }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::{Role, User};

    /// An in-memory event log for the tests. The durable write is not what
    /// these cases are about, and a real file would make them touch the disk.
    fn log() -> Arc<crate::agent_runtime::events::TaskEventLog> {
        Arc::new(
            crate::agent_runtime::events::TaskEventLog::in_memory()
                .expect("an in-memory task event log"),
        )
    }

    fn approver() -> Session {
        // In the 2-role model, an Employee holds `ApproveOutput`. The
        // approver and the author are deliberately different sessions,
        // because an actor may not approve a task they themselves own.
        Session::open(User::new("ravi", "Ravi Menon", vec![Role::Employee]))
    }

    fn author() -> Session {
        Session::open(User::new("priya", "Priya Sharma", vec![Role::Employee]))
    }

    #[tokio::test]
    async fn an_approved_action_names_who_approved_it() {
        let queue = Arc::new(ApprovalQueue::new());
        let events = log();
        let waiting = {
            let queue = queue.clone();
            let events = events.clone();
            tokio::spawn(async move {
                await_decision(
                    &queue,
                    &events,
                    &author(),
                    "run-1",
                    ToolName::WriteScopedFile,
                    "Write 5 bytes to note.txt".into(),
                    "note.txt".into(),
                    "call-1", &serde_json::json!({ "path": "note.txt", "content": "hello" }),
                )
                .await
            })
        };

        // Let the request reach the queue before deciding it.
        let id = loop {
            if let Some(item) = queue.pending().first() {
                break item.request.id.clone();
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        };
        queue.decide(&approver(), &id, true, None).expect("approved");
        events.resolve_approval(&id, crate::agent_runtime::events::ApprovalStatus::Approved, "ravi", None, chrono::Utc::now()).unwrap();

        assert_eq!(
            waiting.await.expect("task finished"),
            ApprovalOutcome::Approved { by: "ravi".into() }
        );
    }

    #[tokio::test]
    async fn a_rejection_carries_the_reason_back_to_the_model() {
        let queue = Arc::new(ApprovalQueue::new());
        let events = log();
        let waiting = {
            let queue = queue.clone();
            let events = events.clone();
            tokio::spawn(async move {
                await_decision(
                    &queue,
                    &events,
                    &author(),
                    "run-1",
                    ToolName::WriteScopedFile,
                    "Write 5 bytes to note.txt".into(),
                    "note.txt".into(),
                    "call-1", &serde_json::json!({ "path": "note.txt", "content": "hello" }),
                )
                .await
            })
        };

        let id = loop {
            if let Some(item) = queue.pending().first() {
                break item.request.id.clone();
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        };
        queue
            .decide(&approver(), &id, false, Some("the seal figure is unsourced"))
            .expect("rejected");
        events.resolve_approval(&id, crate::agent_runtime::events::ApprovalStatus::Rejected, "ravi", Some("the seal figure is unsourced"), chrono::Utc::now()).unwrap();

        let outcome = waiting.await.expect("task finished");
        let refusal = outcome.refusal();
        assert!(refusal.contains("the seal figure is unsourced"), "{refusal}");
        // A model told only "rejected" proposes the same thing again.
        assert!(refusal.contains("Do not propose the same action again"), "{refusal}");
    }

    #[test]
    fn a_timeout_tells_the_model_to_say_so_rather_than_invent_a_result() {
        let refusal = ApprovalOutcome::TimedOut.refusal();
        assert!(refusal.contains("did not happen"));
        assert!(refusal.contains("rather than describing what it would have produced"));
    }

    #[tokio::test]
    async fn the_request_reaches_the_queue_with_what_an_approver_needs_to_judge_it() {
        let queue = Arc::new(ApprovalQueue::new());
        let handle = {
            let queue = queue.clone();
            tokio::spawn(async move {
                await_decision(
                    &queue,
                    &log(),
                    &author(),
                    "run-42",
                    ToolName::CreateDocx,
                    "Produce an approval note at note.docx".into(),
                    "note.docx".into(),
                    "call-42", &serde_json::json!({ "template": "approval-note" }),
                )
                .await
            })
        };

        let item = loop {
            if let Some(item) = queue.pending().first().cloned() {
                break item;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        };

        assert_eq!(item.request.task_id, "run-42");
        assert_eq!(item.request.tool, "artifact.create_approval_note");
        assert_eq!(item.request.target, "note.docx");
        assert_eq!(item.request.requested_by, "priya");
        // The consequence, not just the name — an approver reading "create_docx"
        // learns nothing they did not already know.
        assert_eq!(item.request.consequences, "produce a Word document");

        handle.abort();
    }
}
