//! An approval that outlives the process that asked for it.
//!
//! ## The failure this exists for
//!
//! [`crate::orchestrator::approvals::ApprovalQueue`] is a
//! `Mutex<Vec<ApprovalItem>>`, and [`crate::agent_runtime::approval`] waits on
//! it by making `tool.authorize` take longer — the run is held inside a live
//! call while a poll waits for somebody to decide.
//!
//! That is an elegant arrangement while the process is alive and loses
//! everything when it is not. The question and the answer are both in memory,
//! so a crash while a person was deciding takes the request with it. The
//! operator comes back to a run that was stopped for a reason nothing recorded,
//! and the only safe thing left to do is start again.
//!
//! So the request goes on disk when it is raised, and the decision goes on disk
//! when it is made. The in-memory queue stays exactly as it is and becomes a
//! cache over this, in the same relationship [`super::projection`] has with the
//! event log.
//!
//! ## Why the arguments are fingerprinted and not just stored
//!
//! An approval authorises *a call*, not a tool. A person who allowed
//! `create_docx` writing `approval-note.docx` has not thereby allowed
//! `create_docx` writing anything else, and a resumption that reuses the
//! decision for different arguments has manufactured consent nobody gave.
//!
//! The fingerprint is [`super::idempotency::args_fingerprint`], reused rather
//! than reimplemented, so "the same arguments" means one thing across the
//! approval ledger and the effect ledger. Two subsystems with two definitions
//! of sameness is how a resumption ends up satisfying one and violating the
//! other.
//!
//! ## Why expiry is checked at use and not swept
//!
//! A background sweep that marks approvals expired is a second writer racing
//! the thing that reads them, and it is wrong for a fraction of a second every
//! time regardless. Expiry is therefore a question asked when the approval is
//! *used* — [`DurableApproval::authorises`] — against the clock at that moment.
//! The stored status is what a person decided; whether it still holds is
//! derived.

use chrono::{DateTime, Utc};
use rusqlite::{Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::idempotency::args_fingerprint;

/// Where an approval stands, as a person left it.
///
/// Deliberately does not include "expired": expiry is a fact about the clock
/// rather than a decision anybody made, and storing it would mean writing to
/// the row at a moment nobody asked for anything. See the module note.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ApprovalStatus {
    /// Raised, nobody has decided.
    Pending,
    /// A person allowed it.
    Approved,
    /// A person refused it.
    Rejected,
}

impl ApprovalStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            ApprovalStatus::Pending => "pending",
            ApprovalStatus::Approved => "approved",
            ApprovalStatus::Rejected => "rejected",
        }
    }

    pub fn from_str(raw: &str) -> Option<Self> {
        match raw {
            "pending" => Some(ApprovalStatus::Pending),
            "approved" => Some(ApprovalStatus::Approved),
            "rejected" => Some(ApprovalStatus::Rejected),
            _ => None,
        }
    }
}

/// Why an approval that exists does not authorise the call in hand.
///
/// Each variant is a sentence an operator can act on, because every one of
/// these stops a run and the person reading it has to decide what to do next.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ApprovalInvalid {
    /// Nobody has decided yet.
    StillPending,
    /// A person refused it.
    Rejected { because: String },
    /// A person allowed it, and the window they allowed it for has closed.
    Expired { at: String },
    /// A person allowed a different call.
    ArgumentsChanged,
}

impl ApprovalInvalid {
    pub fn explain(&self) -> String {
        match self {
            ApprovalInvalid::StillPending => {
                "This is waiting for somebody to approve it.".to_string()
            }
            ApprovalInvalid::Rejected { because } => {
                if because.trim().is_empty() {
                    "Somebody refused this.".to_string()
                } else {
                    format!("Somebody refused this: {because}")
                }
            }
            ApprovalInvalid::Expired { at } => format!(
                "The approval for this expired at {at}. Approving it again is a decision \
                 somebody has to take now rather than one taken earlier."
            ),
            ApprovalInvalid::ArgumentsChanged => {
                "The approval was given for a different call. What was approved and what is \
                 about to run are not the same thing, so the earlier decision does not cover it."
                    .to_string()
            }
        }
    }
}

/// One approval, as stored.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DurableApproval {
    pub approval_id: String,
    pub run_id: String,
    pub tool: String,
    /// What the call acts on — a path, a document name. Kept for the screen.
    pub target: String,
    /// [`args_fingerprint`] of the arguments as approved.
    pub args_fingerprint: String,
    /// The arguments as approved, serialised. Shown to the person deciding.
    pub arguments: String,
    pub reason: String,
    pub status: ApprovalStatus,
    /// What a person is allowed to answer. Empty means the ordinary two.
    pub allowed_decisions: Vec<String>,
    pub created_at: String,
    pub expires_at: Option<String>,
    pub resolved_at: Option<String>,
    pub resolved_by: Option<String>,
    pub resolution: Option<String>,
}

impl DurableApproval {
    /// Show exact new call arguments, retaining support for older display-line rows.
    pub fn display_arguments(&self) -> Vec<String> {
        let Ok(value) = serde_json::from_str::<Value>(&self.arguments) else {
            return vec![self.arguments.clone()];
        };
        if let Some(args) = value.get("args") {
            return vec![args.to_string()];
        }
        if let Some(lines) = value.get("arguments").and_then(Value::as_array) {
            return lines.iter().map(|item| item.as_str().map(str::to_string)
                .unwrap_or_else(|| item.to_string())).collect();
        }
        vec![value.to_string()]
    }

    /// A fresh, undecided request.
    #[allow(clippy::too_many_arguments)]
    pub fn requested(
        approval_id: impl Into<String>,
        run_id: impl Into<String>,
        tool: impl Into<String>,
        target: impl Into<String>,
        arguments: &Value,
        reason: impl Into<String>,
        at: DateTime<Utc>,
        expires_at: Option<DateTime<Utc>>,
    ) -> Self {
        Self {
            approval_id: approval_id.into(),
            run_id: run_id.into(),
            tool: tool.into(),
            target: target.into(),
            args_fingerprint: args_fingerprint(arguments),
            arguments: arguments.to_string(),
            reason: reason.into(),
            status: ApprovalStatus::Pending,
            allowed_decisions: Vec::new(),
            created_at: at.to_rfc3339(),
            expires_at: expires_at.map(|at| at.to_rfc3339()),
            resolved_at: None,
            resolved_by: None,
            resolution: None,
        }
    }

    /// Whether this approval authorises `arguments` at `now`.
    ///
    /// Every check that can refuse is made here, in one place, so a caller
    /// cannot satisfy some of them and forget another. Order matters only for
    /// which sentence the operator sees first.
    pub fn authorises(&self, arguments: &Value, now: DateTime<Utc>) -> Result<(), ApprovalInvalid> {
        match self.status {
            ApprovalStatus::Pending => return Err(ApprovalInvalid::StillPending),
            ApprovalStatus::Rejected => {
                return Err(ApprovalInvalid::Rejected {
                    because: self.resolution.clone().unwrap_or_default(),
                })
            }
            ApprovalStatus::Approved => {}
        }

        // Checked before expiry on purpose. "You were approved for a different
        // call" is the more useful thing to be told, and an expired approval
        // for the wrong call is still the wrong call.
        if self.args_fingerprint != args_fingerprint(arguments) {
            return Err(ApprovalInvalid::ArgumentsChanged);
        }

        if let Some(expires_at) = &self.expires_at {
            // An unparseable expiry is treated as expired rather than ignored.
            // The alternative is that a corrupt timestamp silently widens an
            // approval, which is the one direction this must never fail in.
            let expired = DateTime::parse_from_rfc3339(expires_at)
                .map(|deadline| now >= deadline.with_timezone(&Utc))
                .unwrap_or(true);
            if expired {
                return Err(ApprovalInvalid::Expired {
                    at: expires_at.clone(),
                });
            }
        }

        Ok(())
    }
}

/// Writes a request down before anybody is asked.
pub(super) fn record(conn: &Connection, approval: &DurableApproval) -> rusqlite::Result<()> {
    conn.execute(
        "INSERT INTO run_approvals
            (approval_id, run_id, tool, target, args_fingerprint, arguments, reason,
             status, allowed_decisions, created_at, expires_at, resolved_at,
             resolved_by, resolution)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)",
        rusqlite::params![
            approval.approval_id,
            approval.run_id,
            approval.tool,
            approval.target,
            approval.args_fingerprint,
            approval.arguments,
            approval.reason,
            approval.status.as_str(),
            serde_json::to_string(&approval.allowed_decisions).unwrap_or_else(|_| "[]".into()),
            approval.created_at,
            approval.expires_at,
            approval.resolved_at,
            approval.resolved_by,
            approval.resolution,
        ],
    )?;
    Ok(())
}

/// Records what a person decided.
///
/// Refuses to move an approval that is already decided. A second decision is
/// either a double-click or a race, and in both cases the first answer is the
/// one a person gave with the run stopped in front of them.
pub(super) fn resolve(
    conn: &Connection,
    approval_id: &str,
    status: ApprovalStatus,
    by: &str,
    resolution: Option<&str>,
    at: DateTime<Utc>,
) -> rusqlite::Result<bool> {
    if status == ApprovalStatus::Pending { return Ok(false); }
    let changed = conn.execute(
        "UPDATE run_approvals
            SET status = ?2, resolved_by = ?3, resolution = ?4, resolved_at = ?5
          WHERE approval_id = ?1 AND status = 'pending'
            AND (expires_at IS NULL OR julianday(expires_at) > julianday(?5))",
        rusqlite::params![
            approval_id,
            status.as_str(),
            by,
            resolution,
            at.to_rfc3339()
        ],
    )?;
    Ok(changed > 0)
}

/// One approval by id.
pub(super) fn get(
    conn: &Connection,
    approval_id: &str,
) -> rusqlite::Result<Option<DurableApproval>> {
    conn.query_row(
        "SELECT approval_id, run_id, tool, target, args_fingerprint, arguments, reason,
                status, allowed_decisions, created_at, expires_at, resolved_at,
                resolved_by, resolution
           FROM run_approvals WHERE approval_id = ?1",
        [approval_id],
        row_to_approval,
    )
    .optional()
}

/// Everything still undecided, oldest first — what a restart has to put back in
/// front of somebody.
pub(super) fn pending(conn: &Connection) -> rusqlite::Result<Vec<DurableApproval>> {
    let mut statement = conn.prepare(
        "SELECT approval_id, run_id, tool, target, args_fingerprint, arguments, reason,
                status, allowed_decisions, created_at, expires_at, resolved_at,
                resolved_by, resolution
           FROM run_approvals WHERE status = 'pending' ORDER BY created_at, approval_id",
    )?;
    let rows = statement.query_map([], row_to_approval)?;
    rows.collect()
}

/// Every approval raised for one run, oldest first.
pub(super) fn for_run(conn: &Connection, run_id: &str) -> rusqlite::Result<Vec<DurableApproval>> {
    let mut statement = conn.prepare(
        "SELECT approval_id, run_id, tool, target, args_fingerprint, arguments, reason,
                status, allowed_decisions, created_at, expires_at, resolved_at,
                resolved_by, resolution
           FROM run_approvals WHERE run_id = ?1 ORDER BY created_at, approval_id",
    )?;
    let rows = statement.query_map([run_id], row_to_approval)?;
    rows.collect()
}

fn row_to_approval(row: &rusqlite::Row<'_>) -> rusqlite::Result<DurableApproval> {
    let status: String = row.get(7)?;
    let allowed: String = row.get(8)?;
    Ok(DurableApproval {
        approval_id: row.get(0)?,
        run_id: row.get(1)?,
        tool: row.get(2)?,
        target: row.get(3)?,
        args_fingerprint: row.get(4)?,
        arguments: row.get(5)?,
        reason: row.get(6)?,
        // An unreadable status is treated as pending: it stops the run and asks
        // a person, which is the safe direction for a value nothing can parse.
        status: ApprovalStatus::from_str(&status).unwrap_or(ApprovalStatus::Pending),
        allowed_decisions: serde_json::from_str(&allowed).unwrap_or_default(),
        created_at: row.get(9)?,
        expires_at: row.get(10)?,
        resolved_at: row.get(11)?,
        resolved_by: row.get(12)?,
        resolution: row.get(13)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;
    use serde_json::json;

    fn database() -> Connection {
        let conn = Connection::open_in_memory().expect("an in-memory database");
        super::super::migrations::apply(&conn).expect("the schema migrates");
        conn
    }

    #[test]
    fn recording_the_same_id_cannot_reset_a_decision() {
        let conn = database();
        let now = Utc::now();
        let request = approval(now, None);
        record(&conn, &request).unwrap();
        resolve(&conn, &request.approval_id, ApprovalStatus::Approved, "reviewer", None, now).unwrap();
        assert!(record(&conn, &request).is_err());
        assert_eq!(get(&conn, &request.approval_id).unwrap().unwrap().status, ApprovalStatus::Approved);
    }

    #[test]
    fn an_expired_request_cannot_be_approved() {
        let conn = database();
        let now = Utc::now();
        record(&conn, &approval(now, Some(Duration::seconds(1)))).unwrap();
        assert!(!resolve(&conn, "approval-1", ApprovalStatus::Approved, "reviewer", None, now + Duration::seconds(2)).unwrap());
    }

    fn approval(at: DateTime<Utc>, expires_in: Option<Duration>) -> DurableApproval {
        DurableApproval::requested(
            "approval-1",
            "run-1",
            "create_docx",
            "approval-note.docx",
            &json!({ "path": "approval-note.docx" }),
            "writes a document",
            at,
            expires_in.map(|window| at + window),
        )
    }

    #[test]
    fn a_request_survives_being_written_and_read_back() {
        let conn = database();
        let now = Utc::now();
        record(&conn, &approval(now, None)).expect("the request is recorded");

        let read = get(&conn, "approval-1")
            .expect("the read succeeds")
            .expect("the approval is there");
        assert_eq!(read.tool, "create_docx");
        assert_eq!(read.status, ApprovalStatus::Pending);
        assert_eq!(read.run_id, "run-1");
    }

    /// The restart case: a request raised and never decided is what a new
    /// process has to put back in front of somebody.
    #[test]
    fn a_pending_request_is_still_pending_after_a_restart() {
        let conn = database();
        record(&conn, &approval(Utc::now(), None)).expect("recorded");

        let waiting = pending(&conn).expect("the pending list reads");
        assert_eq!(waiting.len(), 1);
        assert_eq!(waiting[0].approval_id, "approval-1");
    }

    #[test]
    fn a_decision_is_recorded_and_the_request_leaves_the_pending_list() {
        let conn = database();
        let now = Utc::now();
        record(&conn, &approval(now, None)).expect("recorded");

        assert!(resolve(
            &conn,
            "approval-1",
            ApprovalStatus::Approved,
            "operator",
            None,
            now
        )
        .expect("the decision is recorded"));
        assert!(pending(&conn).expect("reads").is_empty());

        let read = get(&conn, "approval-1").expect("reads").expect("is there");
        assert_eq!(read.status, ApprovalStatus::Approved);
        assert_eq!(read.resolved_by.as_deref(), Some("operator"));
    }

    /// A second decision does not overwrite the first.
    #[test]
    fn an_already_decided_approval_is_not_decided_again() {
        let conn = database();
        let now = Utc::now();
        record(&conn, &approval(now, None)).expect("recorded");
        resolve(
            &conn,
            "approval-1",
            ApprovalStatus::Approved,
            "first",
            None,
            now,
        )
        .expect("first");

        let changed = resolve(
            &conn,
            "approval-1",
            ApprovalStatus::Rejected,
            "second",
            Some("changed my mind"),
            now,
        )
        .expect("the second attempt is answered");
        assert!(!changed, "the second decision must not take");

        let read = get(&conn, "approval-1").expect("reads").expect("is there");
        assert_eq!(read.status, ApprovalStatus::Approved);
        assert_eq!(read.resolved_by.as_deref(), Some("first"));
    }

    #[test]
    fn an_approved_request_authorises_the_call_it_was_approved_for() {
        let now = Utc::now();
        let mut item = approval(now, None);
        item.status = ApprovalStatus::Approved;
        assert!(item
            .authorises(&json!({ "path": "approval-note.docx" }), now)
            .is_ok());
    }

    /// The rule the whole fingerprint exists for.
    #[test]
    fn an_approval_does_not_carry_over_to_different_arguments() {
        let now = Utc::now();
        let mut item = approval(now, None);
        item.status = ApprovalStatus::Approved;
        assert_eq!(
            item.authorises(&json!({ "path": "something-else.docx" }), now),
            Err(ApprovalInvalid::ArgumentsChanged)
        );
    }

    #[test]
    fn an_expired_approval_does_not_authorise_anything() {
        let now = Utc::now();
        let mut item = approval(now, Some(Duration::minutes(5)));
        item.status = ApprovalStatus::Approved;

        let args = json!({ "path": "approval-note.docx" });
        assert!(item.authorises(&args, now + Duration::minutes(1)).is_ok());
        assert!(matches!(
            item.authorises(&args, now + Duration::minutes(6)),
            Err(ApprovalInvalid::Expired { .. })
        ));
    }

    /// An expiry nothing can parse must fail closed, never open.
    #[test]
    fn an_unreadable_expiry_is_treated_as_expired() {
        let now = Utc::now();
        let mut item = approval(now, None);
        item.status = ApprovalStatus::Approved;
        item.expires_at = Some("not a timestamp".into());
        assert!(matches!(
            item.authorises(&json!({ "path": "approval-note.docx" }), now),
            Err(ApprovalInvalid::Expired { .. })
        ));
    }

    #[test]
    fn a_pending_or_rejected_request_authorises_nothing() {
        let now = Utc::now();
        let args = json!({ "path": "approval-note.docx" });

        let waiting = approval(now, None);
        assert_eq!(
            waiting.authorises(&args, now),
            Err(ApprovalInvalid::StillPending)
        );

        let mut refused = approval(now, None);
        refused.status = ApprovalStatus::Rejected;
        refused.resolution = Some("not this one".into());
        assert!(matches!(
            refused.authorises(&args, now),
            Err(ApprovalInvalid::Rejected { .. })
        ));
    }

    #[test]
    fn approvals_are_listed_per_run() {
        let conn = database();
        let now = Utc::now();
        record(&conn, &approval(now, None)).expect("recorded");

        let mut other = approval(now, None);
        other.approval_id = "approval-2".into();
        other.run_id = "run-2".into();
        record(&conn, &other).expect("recorded");

        assert_eq!(for_run(&conn, "run-1").expect("reads").len(), 1);
        assert_eq!(for_run(&conn, "run-2").expect("reads").len(), 1);
    }

    #[test]
    fn restored_approval_displays_exact_arguments_and_legacy_lines() {
        let mut item = approval(Utc::now(), None);
        let args = json!({"path":"final.txt","content":"PUMP-A17\nexact text"});
        item.arguments = json!({"tool":"workspace.write_text","args":args}).to_string();
        assert_eq!(item.display_arguments(), vec![args.to_string()]);
        item.arguments = json!({"arguments":["path: legacy.txt", "content: legacy"]}).to_string();
        assert_eq!(item.display_arguments(), vec!["path: legacy.txt", "content: legacy"]);
        item.arguments = "unreadable record".into();
        assert_eq!(item.display_arguments(), vec!["unreadable record"]);
    }
}
