//! Commands behind the Approvals surface.
//!
//! Thin on purpose: every rule that matters — who may decide, that a rejection
//! carries a reason, that a decision is final — lives in
//! [`crate::orchestrator::approvals`], where it is tested without a UI. A rule
//! enforced only in a command is a rule that stops applying the moment anything
//! else calls the same code.

use std::sync::Arc;

use tauri::State;

use crate::audit::{AuditKind, AuditService};
use crate::commands::governance::{require_permission, require_session, CurrentSession};
use crate::identity::Permission;
use crate::orchestrator::approvals::{ApprovalItem, ApprovalQueue, Decision};

/// Everything raised this session, newest first, settled ones included.
#[tauri::command]
pub async fn list_approvals(
    queue: State<'_, Arc<ApprovalQueue>>,
    session: State<'_, CurrentSession>,
) -> Result<Vec<ApprovalItem>, String> {
    // The approvals queue is reviewer's work. The matrix puts the
    // ability to see it under `ApproveOutput`. A `User` or `Auditor`
    // does not get to see what they are not allowed to decide.
    require_permission(&session, Permission::ApproveOutput)?;
    Ok(queue.all())
}

/// Approves or rejects one request.
#[tauri::command]
pub async fn decide_approval(
    queue: State<'_, Arc<ApprovalQueue>>,
    session: State<'_, CurrentSession>,
    audit: State<'_, Arc<AuditService>>,
    events: State<'_, crate::commands::agent::TaskEvents>,
    id: String,
    approve: bool,
    because: Option<String>,
) -> Result<Decision, String> {
    let signed_in = require_session(&session)?;

    // Restored display requests are caches, not authority. Resolve the actor
    // from the durable run and refuse missing, expired or terminal requests.
    let stored = events.approval(&id)?.ok_or("This approval is not durable.")?;
    let run = events.snapshot(&stored.run_id)?.ok_or("The approval's run is missing.")?;
    if run.actor == signed_in.user.id || run.state.is_terminal() {
        return Err("This run cannot be approved by the active operator.".into());
    }

    let decision = queue
        .decide_durable(&signed_in, &id, approve, because.as_deref(), |decision| {
            let status = if decision.approved() {
                crate::agent_runtime::events::ApprovalStatus::Approved
            } else { crate::agent_runtime::events::ApprovalStatus::Rejected };
            if !events.resolve_approval(&id, status, decision.decided_by(), because.as_deref(), chrono::Utc::now())? {
                return Err("This request expired or was already decided. No new decision was applied.".into());
            }
            Ok(())
        })
        .map_err(|e| {
            // A refused decision is recorded too. "Who tried to approve their
            // own work" is exactly the question an auditor asks later.
            let _ = audit.record(
                &signed_in.user.id,
                AuditKind::PolicyDecision,
                format!("Approval decision refused: {}", e.message),
                Some(serde_json::json!({ "approvalId": id, "allowed": false })),
            );
            e.message
        })?;

    let item = queue.find(&id);
    let _ = audit.record(
        &signed_in.user.id,
        AuditKind::Approval,
        format!(
            "{} {} for {}",
            if approve { "Approved" } else { "Rejected" },
            item.as_ref().map(|i| i.request.tool.as_str()).unwrap_or("an action"),
            item.as_ref().map(|i| i.request.target.as_str()).unwrap_or("an unknown target"),
        ),
        Some(serde_json::json!({
            "approvalId": id,
            "taskId": item.as_ref().map(|i| i.request.task_id.clone()),
            "approved": approve,
        })),
    );

    Ok(decision)
}
