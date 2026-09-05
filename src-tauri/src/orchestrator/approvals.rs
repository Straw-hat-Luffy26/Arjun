//! The queue of actions waiting on a person.
//!
//! ARJUN design rule 26: *"Before any risky action the application pauses and shows the
//! user the target, the arguments, the supporting evidence, the expected output
//! and the consequences."*
//!
//! The list is the easy part. Three things about it are not, and they are what
//! this module is for.
//!
//! **What the approver saw is what gets recorded.** The prompt is captured when
//! the request is raised and never regenerated. A summary written after the fact
//! — from the tool call, or from the model's account of it — is a different
//! document from the one somebody actually read before signing, and only the
//! first is evidence.
//!
//! **Only an approver may decide.** The check is against
//! [`Permission::ApproveOutput`]. In the 2-role product both `Administrator`
//! and `Employee` hold this permission; an actor who is one cannot
//! approve a task they themselves own — the policy gateway's
//! separation-of-duties check handles that case. Whatever a second
//! person with the permission can sign off, the actor of a task cannot.
//!
//! **A decision is final.** Approving twice, or reversing an approval, is
//! refused rather than silently overwritten. An approval record that could be
//! edited afterwards proves nothing.

use std::sync::Mutex;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::identity::{Permission, Session};

/// What a person is being asked to allow.
///
/// Every field is filled in at the moment the request is raised. Nothing here is
/// derived later.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ApprovalRequest {
    pub id: String,
    pub task_id: String,
    /// The tool that wants to run.
    pub tool: String,
    /// What it would act on — a path, a document, a recipient.
    pub target: String,
    /// The arguments, rendered as the approver will read them.
    pub arguments: Vec<String>,
    /// Passages and calculations the proposed action rests on, cited.
    pub evidence: Vec<String>,
    /// What the action is expected to produce.
    pub expected_output: String,
    /// What it would change, and what could not be undone.
    pub consequences: String,
    /// Who asked.
    pub requested_by: String,
    pub requested_at: DateTime<Utc>,
}

/// How a request was settled.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", tag = "decision")]
pub enum Decision {
    Approved { by: String, at: DateTime<Utc> },
    Rejected { by: String, at: DateTime<Utc>, because: String },
}

impl Decision {
    pub fn approved(&self) -> bool {
        matches!(self, Decision::Approved { .. })
    }

    pub fn decided_by(&self) -> &str {
        match self {
            Decision::Approved { by, .. } | Decision::Rejected { by, .. } => by,
        }
    }
}

/// A request and, once settled, its decision.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ApprovalItem {
    pub request: ApprovalRequest,
    pub decision: Option<Decision>,
}

impl ApprovalItem {
    pub fn is_pending(&self) -> bool {
        self.decision.is_none()
    }
}

/// Why a decision was refused.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ApprovalError {
    pub message: String,
}

impl ApprovalError {
    fn new(message: impl Into<String>) -> Self {
        Self { message: message.into() }
    }
}

/// The queue. Session-scoped: an approval does not outlive the run it belongs
/// to, and a pending request that survived a restart would be a decision made
/// about a task nobody can still see.
#[derive(Default)]
pub struct ApprovalQueue {
    items: Mutex<Vec<ApprovalItem>>,
}

impl ApprovalQueue {
    pub fn new() -> Self {
        Self::default()
    }

    /// Puts requests back that were raised before this process started.
    ///
    /// Called once at startup from the durable ledger. Anything already in the
    /// queue with the same id is left alone: the live entry is the one being
    /// waited on, and replacing it would detach the waiter from its answer.
    ///
    /// A restored request has no waiter — the loop that asked for it died with
    /// the last process. It is put back so a person can still see and decide
    /// it, and so the run that resumes can read the decision rather than
    /// asking again.
    pub fn restore(&self, requests: Vec<ApprovalRequest>) -> usize {
        let mut items = self
            .items
            .lock()
            .expect("the approval queue lock is never poisoned");
        let mut restored = 0;
        for request in requests {
            if items.iter().any(|held| held.request.id == request.id) {
                continue;
            }
            items.push(ApprovalItem { request, decision: None });
            restored += 1;
        }
        restored
    }

    /// Raises a request and returns its id.
    pub fn request(&self, request: ApprovalRequest) -> String {
        let id = request.id.clone();
        let mut items = self.items.lock().expect("the approval queue lock is never poisoned");
        items.push(ApprovalItem { request, decision: None });
        id
    }

    /// Everything raised this session, newest first.
    pub fn all(&self) -> Vec<ApprovalItem> {
        let items = self.items.lock().expect("the approval queue lock is never poisoned");
        let mut all = items.clone();
        all.reverse();
        all
    }

    pub fn pending(&self) -> Vec<ApprovalItem> {
        self.all().into_iter().filter(ApprovalItem::is_pending).collect()
    }

    pub fn pending_count(&self) -> usize {
        let items = self.items.lock().expect("the approval queue lock is never poisoned");
        items.iter().filter(|i| i.is_pending()).count()
    }

    pub fn find(&self, id: &str) -> Option<ApprovalItem> {
        let items = self.items.lock().expect("the approval queue lock is never poisoned");
        items.iter().find(|i| i.request.id == id).cloned()
    }

    /// Settles a request.
    ///
    /// `because` is required for a rejection and ignored for an approval: a
    /// person refusing an action owes the task an explanation, and the model
    /// needs it to do anything other than propose the same thing again.
    pub fn decide(
        &self,
        session: &Session,
        id: &str,
        approve: bool,
        because: Option<&str>,
    ) -> Result<Decision, ApprovalError> {
        self.decide_durable(session, id, approve, because, |_| Ok(()))
    }

    /// Validate under the queue lock, commit durably, then wake live readers.
    /// If the commit fails no in-memory decision is exposed to the waiter.
    pub fn decide_durable(
        &self,
        session: &Session,
        id: &str,
        approve: bool,
        because: Option<&str>,
        commit: impl FnOnce(&Decision) -> Result<(), String>,
    ) -> Result<Decision, ApprovalError> {
        // The actor must hold ApproveOutput. In the 2-role model that is
        // both Administrator and Employee; the policy gateway separately
        // refuses an actor from approving a task they themselves own.
        if !session.user.holds(Permission::ApproveOutput) {
            return Err(ApprovalError::new(format!(
                "{} is not permitted to approve or reject an action.",
                session.user.display_name
            )));
        }

        if !approve {
            let reason = because.map(str::trim).unwrap_or("");
            if reason.is_empty() {
                return Err(ApprovalError::new(
                    "A rejection needs a reason. The task cannot do anything but propose the same \
                     action again without one.",
                ));
            }
        }

        let mut items = self.items.lock().expect("the approval queue lock is never poisoned");
        let Some(item) = items.iter_mut().find(|i| i.request.id == id) else {
            return Err(ApprovalError::new(format!("There is no approval request {id:?}.")));
        };

        if item.request.requested_by == session.user.id {
            return Err(ApprovalError::new("You cannot approve or reject your own action."));
        }

        // A decision is final. Reversing one silently would make every approval
        // record unprovable.
        if let Some(existing) = &item.decision {
            return Err(ApprovalError::new(format!(
                "This action was already {} by {}. A decision cannot be changed — raise a new \
                 request instead.",
                if existing.approved() { "approved" } else { "rejected" },
                existing.decided_by()
            )));
        }

        let decision = if approve {
            Decision::Approved { by: session.user.id.clone(), at: Utc::now() }
        } else {
            Decision::Rejected {
                by: session.user.id.clone(),
                at: Utc::now(),
                because: because.unwrap_or_default().trim().to_string(),
            }
        };

        commit(&decision).map_err(ApprovalError::new)?;
        item.decision = Some(decision.clone());
        Ok(decision)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::UserDirectory;

    #[test]
    fn a_failed_durable_commit_never_releases_a_waiter() {
        let queue = ApprovalQueue::new();
        let actor = crate::identity::Session::open(crate::identity::User::new("reviewer", "Reviewer", vec![crate::identity::Role::Employee]));
        queue.request(request("a"));
        assert!(queue.decide_durable(&actor, "a", true, None, |_| Err("disk full".into())).is_err());
        assert!(queue.find("a").unwrap().is_pending());
    }

    fn session(user_id: &str) -> Session {
        let directory = UserDirectory::seeded();
        Session::open(directory.find(user_id).expect("seeded user").clone())
    }

    fn request(id: &str) -> ApprovalRequest {
        ApprovalRequest {
            id: id.into(),
            task_id: "task-42".into(),
            tool: "write_file".into(),
            target: r"D:\tasks\task-42\approval-note.docx".into(),
            arguments: vec!["template: approval_note".into(), "revision: 1".into()],
            evidence: vec![
                "[E1] PV-2201 inspection, page 2 — governing measurement 8.2 mm".into(),
                "[E2] Maintenance SOP rev C, section 3 — minimum allowable 9.0 mm".into(),
            ],
            expected_output: "A Word approval note recommending replacement within 90 days."
                .into(),
            consequences: "Writes a new file into the task folder. Nothing existing is \
                           overwritten; the note is a draft until a reviewer signs it."
                .into(),
            requested_by: "engineer".into(),
            requested_at: Utc::now(),
        }
    }

    #[test]
    fn a_raised_request_is_pending_and_carries_everything_the_approver_needs() {
        let queue = ApprovalQueue::new();
        queue.request(request("a1"));

        let pending = queue.pending();
        assert_eq!(pending.len(), 1);
        assert_eq!(queue.pending_count(), 1);

        // ARJUN design rule 26 names five things. All five have to be there.
        let r = &pending[0].request;
        assert!(!r.target.is_empty());
        assert!(!r.arguments.is_empty());
        assert!(!r.evidence.is_empty());
        assert!(!r.expected_output.is_empty());
        assert!(!r.consequences.is_empty());
    }

    #[test]
    fn an_administrator_may_approve() {
        let queue = ApprovalQueue::new();
        queue.request(request("a1"));

        let decision = queue.decide(&session("admin"), "a1", true, None).unwrap();
        assert!(decision.approved());
        assert_eq!(queue.pending_count(), 0);
    }

    /// An Employee holds ApproveOutput in the 2-role model, so an Employee
    /// can also approve a task — provided they are not the task owner.
    #[test]
    fn an_employee_may_approve_someone_elses_task() {
        let queue = ApprovalQueue::new();
        let mut action = request("a1");
        action.requested_by = "another-employee".into();
        queue.request(action);

        let decision = queue.decide(&session("engineer"), "a1", true, None).unwrap();
        assert!(decision.approved());
        assert_eq!(queue.pending_count(), 0);
    }

    #[test]
    fn the_requester_cannot_approve_their_own_action() {
        let queue = ApprovalQueue::new();
        queue.request(request("a1"));
        assert!(queue.decide(&session("engineer"), "a1", true, None).is_err());
        assert!(queue.find("a1").unwrap().is_pending());
    }

    /// A legacy role (kept on the enum for test compat) grants nothing in
    /// the active product, so an attempt to approve under one is refused.
    /// Pinned here so a regression that re-enables a legacy role is caught.
    #[test]
    fn a_legacy_role_cannot_approve() {
        // Build a session directly with a legacy role, since the seeded
        // directory no longer offers it.
        let legacy = crate::identity::User::new("legacy", "Legacy", vec![crate::identity::Role::Reviewer]);
        let session = Session::open(legacy);
        let queue = ApprovalQueue::new();
        queue.request(request("a1"));

        let error = queue.decide(&session, "a1", true, None).unwrap_err();
        assert!(error.message.contains("not permitted"));
    }

    /// Without a reason the task can do nothing but propose the same thing.
    #[test]
    fn a_rejection_needs_a_reason() {
        let queue = ApprovalQueue::new();
        queue.request(request("a1"));

        let error = queue.decide(&session("admin"), "a1", false, None).unwrap_err();
        assert!(error.message.contains("needs a reason"));

        let blank = queue.decide(&session("admin"), "a1", false, Some("   ")).unwrap_err();
        assert!(blank.message.contains("needs a reason"));

        assert_eq!(queue.pending_count(), 1);
    }

    #[test]
    fn a_rejection_with_a_reason_records_it() {
        let queue = ApprovalQueue::new();
        queue.request(request("a1"));

        let decision = queue
            .decide(&session("admin"), "a1", false, Some("  the 90-day window is wrong  "))
            .unwrap();

        match decision {
            Decision::Rejected { because, .. } => assert_eq!(because, "the 90-day window is wrong"),
            Decision::Approved { .. } => panic!("expected a rejection"),
        }
    }

    /// An approval record that could be edited afterwards proves nothing.
    #[test]
    fn a_decision_cannot_be_reversed_or_repeated() {
        let queue = ApprovalQueue::new();
        queue.request(request("a1"));
        queue.decide(&session("admin"), "a1", true, None).unwrap();

        let again = queue.decide(&session("admin"), "a1", true, None).unwrap_err();
        assert!(again.message.contains("already approved"));

        let reversed = queue
            .decide(&session("admin"), "a1", false, Some("changed my mind"))
            .unwrap_err();
        assert!(reversed.message.contains("cannot be changed"));
        assert!(reversed.message.contains("raise a new request"));
    }

    #[test]
    fn deciding_something_that_was_never_raised_says_so() {
        let queue = ApprovalQueue::new();
        let error = queue.decide(&session("admin"), "nope", true, None).unwrap_err();
        assert!(error.message.contains("no approval request"));
    }

    /// The prompt the approver read is the one that gets recorded — it is
    /// captured once and never regenerated.
    #[test]
    fn a_settled_request_keeps_what_the_approver_was_shown() {
        let queue = ApprovalQueue::new();
        queue.request(request("a1"));
        queue.decide(&session("admin"), "a1", true, None).unwrap();

        let item = queue.find("a1").unwrap();
        assert!(!item.is_pending());
        assert_eq!(item.request.evidence.len(), 2);
        assert!(item.request.consequences.contains("Nothing existing is overwritten"));
        assert_eq!(item.decision.unwrap().decided_by(), "admin");
    }

    #[test]
    fn the_queue_shows_the_newest_request_first() {
        let queue = ApprovalQueue::new();
        queue.request(request("a1"));
        queue.request(request("a2"));

        let all = queue.all();
        assert_eq!(all[0].request.id, "a2");
        assert_eq!(all[1].request.id, "a1");
    }

    #[test]
    fn settled_requests_stay_visible_but_leave_the_pending_list() {
        let queue = ApprovalQueue::new();
        queue.request(request("a1"));
        queue.request(request("a2"));
        queue.decide(&session("admin"), "a1", true, None).unwrap();

        assert_eq!(queue.pending().len(), 1);
        assert_eq!(queue.all().len(), 2, "a decision is history, not a deletion");
    }
}
