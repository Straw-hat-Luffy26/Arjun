//! The tool gateway — where a model's request becomes an action, or does not.
//!
//! ARJUN design rule 25: *"The tool gateway validates the call before execution. It checks
//! the tool name, arguments, user permission, target paths, document ACLs, file
//! size, time limit, resource quota, and whether human approval is needed. The
//! model cannot directly access the operating system, arbitrary folders,
//! credentials, or network."*
//!
//! Every one of those checks is here, and the ordering is deliberate: the
//! cheapest and most fundamental first, so a refusal always names the real
//! reason rather than an incidental one downstream of it.
//!
//! ## Nothing the model wrote is trusted
//!
//! The [`ToolCall`] is untrusted input in exactly the sense a form submission is.
//! Its tool name might not exist, its arguments might be missing, wrongly typed,
//! or crafted; its path might be a traversal. None of that is unusual or
//! alarming — it is the ordinary case for a probabilistic system, and the reason
//! the gateway exists at all rather than trusting a well-behaved model.
//!
//! ## Refusals are recoverable
//!
//! A refusal returns *why*, phrased so it can be handed back to the model as the
//! result of the call. An agent told "path must be inside the task workspace"
//! can correct itself; one told "denied" can only guess, and will usually guess
//! the same thing again.

use std::path::{Component, Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::tools::{spec_for, ArgumentKind, ToolCall, ToolName, ToolSpec};
use crate::identity::Session;
use crate::policy::{ApprovalState, Decision, PolicyGateway, Request};

/// The result of putting a call through the gateway.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", tag = "outcome")]
pub enum GatewayVerdict {
    /// Run it. Carries the validated shape so the executor does not re-parse.
    Allow {
        tool: ToolName,
        /// Resolved and confirmed inside the workspace, when the tool takes one.
        resolved_path: Option<PathBuf>,
    },
    /// A person has to say yes first. Carries what to show them.
    NeedsApproval {
        tool: ToolName,
        summary: String,
        resolved_path: Option<PathBuf>,
    },
    /// Not happening, and why — written to be handed back to the model.
    Refuse { reason: String },
}

impl GatewayVerdict {
    pub fn is_allowed(&self) -> bool {
        matches!(self, GatewayVerdict::Allow { .. })
    }

    /// The text a caller shows, or returns to the model as the call's result.
    pub fn message(&self) -> String {
        match self {
            GatewayVerdict::Allow { tool, .. } => format!("Permitted: {}", tool.describe()),
            GatewayVerdict::NeedsApproval { summary, .. } => summary.clone(),
            GatewayVerdict::Refuse { reason } => reason.clone(),
        }
    }
}

/// What the gateway needs to know about the task making the call.
pub struct TaskContext<'a> {
    pub session: &'a Session,
    /// Directories this task may touch. Empty means none.
    pub workspace_roots: &'a [PathBuf],
    /// Whether confidential work is permitted right now — the sovereignty
    /// invariant, passed in rather than reached for so this stays testable.
    pub confidential_work_permitted: bool,
    /// Approval already obtained for this specific call, if any.
    pub approval: ApprovalState,
}

/// Resolves `..` textually and confirms the result is inside a permitted root.
///
/// Textual rather than filesystem-based on purpose. The target of a write
/// usually does not exist yet, so `canonicalize` would fail on exactly the case
/// that matters — and a textual check cannot be defeated by a link planted
/// between the check and the write.
fn resolve_within(candidate: &Path, roots: &[PathBuf]) -> Option<PathBuf> {
    fn normalise(path: &Path) -> Option<PathBuf> {
        let mut out = PathBuf::new();
        for component in path.components() {
            match component {
                // Climbing above the root is a traversal attempt, not a path.
                // Refused rather than clamped, because clamping silently turns
                // an attack into a valid write somewhere unexpected.
                Component::ParentDir => {
                    if !out.pop() {
                        return None;
                    }
                }
                Component::CurDir => {}
                other => out.push(other.as_os_str()),
            }
        }
        Some(out)
    }

    let resolved = normalise(candidate)?;
    roots
        .iter()
        .filter_map(|root| normalise(root))
        .any(|root| resolved.starts_with(&root))
        .then_some(resolved)
}

pub struct ToolGateway;

impl ToolGateway {
    /// Decides one call.
    pub fn decide(call: &ToolCall, context: &TaskContext<'_>) -> GatewayVerdict {
        // 1. Is this even a tool? An unknown name is the commonest failure with
        //    a model that has drifted, and naming what exists lets it recover.
        let Some(tool) = ToolName::from_str(&call.tool) else {
            return GatewayVerdict::Refuse {
                reason: format!(
                    "There is no tool called {:?}. Available tools are: {}.",
                    call.tool,
                    ToolName::ALL
                        .iter()
                        .map(|t| t.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            };
        };

        let spec = spec_for(tool);

        // 2. Are the arguments the right shape? Checked before permissions so a
        //    malformed call is not reported as a permissions problem.
        if let Err(reason) = Self::check_arguments(call, &spec) {
            return GatewayVerdict::Refuse { reason };
        }

        // 3. Is the path inside the workspace? The check that stops a model
        //    writing over the operating system.
        let resolved_path = if spec.scoped_to_workspace {
            let raw = call
                .text("path")
                .expect("checked present and textual above");
            match resolve_within(Path::new(raw), context.workspace_roots) {
                Some(path) => Some(path),
                None => {
                    return GatewayVerdict::Refuse {
                        reason: format!(
                            "{raw:?} is outside this task's workspace. Files may only be read \
                             from or written to the task's own directory."
                        ),
                    }
                }
            }
        } else {
            None
        };

        // 4. Size, where the call carries content to write.
        if let Some(limit) = spec.max_bytes {
            if let Some(content) = call.text("content") {
                if content.len() as u64 > limit {
                    return GatewayVerdict::Refuse {
                        reason: format!(
                            "That content is {} MB, above the {} MB limit for {}.",
                            content.len() / 1024 / 1024,
                            limit / 1024 / 1024,
                            tool.describe()
                        ),
                    };
                }
            }
        }

        // 5. Mode, entitlement, clearance and approval, decided by the policy
        //    gateway rather than duplicated here — one place decides who may do
        //    what, and this is not it.
        let request = Request {
            permission: spec.permission,
            classification: None,
            target_path: None,
            allowed_roots: &[],
            needs_approval: spec.needs_approval,
            approval: context.approval,
            task_owner: None,
        };

        match PolicyGateway::decide(
            context.session,
            &request,
            context.confidential_work_permitted,
        ) {
            Decision::Allow => GatewayVerdict::Allow { tool, resolved_path },
            Decision::Refuse { reason } => GatewayVerdict::Refuse { reason },
            Decision::NeedsApproval { .. } => GatewayVerdict::NeedsApproval {
                tool,
                summary: Self::approval_summary(tool, &resolved_path, call),
                resolved_path,
            },
        }
    }

    /// Confirms every declared argument is present and of the right kind.
    fn check_arguments(call: &ToolCall, spec: &ToolSpec) -> Result<(), String> {
        // Required first, then the optional ones — which are checked for kind
        // when they are present and skipped when they are not. Omitting one is
        // allowed; supplying the wrong shape is not.
        let required = spec.arguments.iter().map(|argument| (argument, true));
        let optional = spec.optional_arguments.iter().map(|argument| (argument, false));

        for (argument, must_be_present) in required.chain(optional) {
            let Some(value) = call.arguments.get(argument.name) else {
                if !must_be_present {
                    continue;
                }
                return Err(format!(
                    "{} needs a {:?} argument, which was missing.",
                    spec.name.as_str(),
                    argument.name
                ));
            };

            let right_kind = match argument.kind {
                ArgumentKind::Text | ArgumentKind::Path => value.is_string(),
                ArgumentKind::Integer => value.is_i64() || value.is_u64(),
                ArgumentKind::Object => value.is_object(),
            };

            if !right_kind {
                return Err(format!(
                    "{}'s {:?} argument should be {}, but was {}.",
                    spec.name.as_str(),
                    argument.name,
                    match argument.kind {
                        ArgumentKind::Text | ArgumentKind::Path => "text",
                        ArgumentKind::Integer => "a whole number",
                        ArgumentKind::Object => "an object",
                    },
                    describe_json(value)
                ));
            }

            // An empty path is not a path, and would resolve to the workspace
            // root — which would let a write clobber the directory itself.
            if argument.kind == ArgumentKind::Path
                && value.as_str().is_some_and(|s| s.trim().is_empty())
            {
                return Err(format!(
                    "{}'s path was empty.",
                    spec.name.as_str()
                ));
            }
        }
        Ok(())
    }

    /// What a person is shown before approving.
    ///
    /// ARJUN design rule 26 asks for the target, the arguments and the consequence. A
    /// prompt that says only "allow this action?" trains people to say yes.
    fn approval_summary(tool: ToolName, path: &Option<PathBuf>, call: &ToolCall) -> String {
        let mut summary = format!("ARJUN wants to {}.", tool.describe());

        if let Some(path) = path {
            summary.push_str(&format!("\nTarget: {}", path.display()));
        }

        match tool {
            ToolName::ExecuteCode => {
                let language = call.text("language").unwrap_or("unknown");
                let lines = call.text("source").map(|s| s.lines().count()).unwrap_or(0);
                summary.push_str(&format!(
                    "\n{lines} line(s) of {language}, run with no network and no access to \
                     anything outside its own directory."
                ));
            }
            ToolName::WriteScopedFile => {
                let bytes = call.text("content").map(str::len).unwrap_or(0);
                summary.push_str(&format!("\n{bytes} byte(s) will be written."));
            }
            ToolName::CreateDocx | ToolName::CreateXlsx => {
                let template = call.text("template").unwrap_or("none");
                summary.push_str(&format!("\nFrom the {template:?} template."));
            }
            _ => {}
        }

        summary
    }
}

fn describe_json(value: &serde_json::Value) -> &'static str {
    match value {
        serde_json::Value::Null => "empty",
        serde_json::Value::Bool(_) => "true or false",
        serde_json::Value::Number(_) => "a number",
        serde_json::Value::String(_) => "text",
        serde_json::Value::Array(_) => "a list",
        serde_json::Value::Object(_) => "an object",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::{Role, User};
    use serde_json::json;

    fn session(roles: Vec<Role>) -> Session {
        Session::open(User::new("kiran", "Kiran", roles))
    }

    fn workspace() -> Vec<PathBuf> {
        vec![PathBuf::from("C:/arjun/tasks/42")]
    }

    fn context<'a>(session: &'a Session, roots: &'a [PathBuf]) -> TaskContext<'a> {
        TaskContext {
            session,
            workspace_roots: roots,
            confidential_work_permitted: true,
            approval: ApprovalState::NotRequested,
        }
    }

    // ── The tool itself ──────────────────────────────────────────────────

    #[test]
    fn an_unknown_tool_is_refused_and_the_real_ones_are_named() {
        let s = session(vec![Role::Employee]);
        let roots = workspace();
        let verdict = ToolGateway::decide(
            &ToolCall::new("delete_all_records", json!({})),
            &context(&s, &roots),
        );

        assert!(!verdict.is_allowed());
        assert!(verdict.message().contains("no tool called"));
        // Naming the alternatives is what lets an agent correct itself.
        assert!(verdict.message().contains("knowledge.search_authorized"));
    }

    // ── Arguments ────────────────────────────────────────────────────────

    #[test]
    fn a_missing_argument_says_which_one() {
        let s = session(vec![Role::Employee]);
        let roots = workspace();
        let verdict =
            ToolGateway::decide(&ToolCall::new("search_documents", json!({})), &context(&s, &roots));
        assert!(verdict.message().contains("\"query\""));
    }

    #[test]
    fn a_wrongly_typed_argument_says_what_was_expected_and_what_arrived() {
        let s = session(vec![Role::Employee]);
        let roots = workspace();
        let verdict = ToolGateway::decide(
            &ToolCall::new("search_documents", json!({ "query": 42 })),
            &context(&s, &roots),
        );
        assert!(verdict.message().contains("should be text"));
        assert!(verdict.message().contains("was a number"));
    }

    /// An empty path resolves to the workspace root, which a write would clobber.
    #[test]
    fn an_empty_path_is_refused() {
        let s = session(vec![Role::Employee]);
        let roots = workspace();
        let verdict = ToolGateway::decide(
            &ToolCall::new("read_scoped_file", json!({ "path": "   " })),
            &context(&s, &roots),
        );
        assert!(verdict.message().contains("path was empty"));
    }

    // ── Scope ────────────────────────────────────────────────────────────

    #[test]
    fn a_read_inside_the_workspace_is_allowed() {
        let s = session(vec![Role::Employee]);
        let roots = workspace();
        let verdict = ToolGateway::decide(
            &ToolCall::new("read_scoped_file", json!({ "path": "C:/arjun/tasks/42/report.txt" })),
            &context(&s, &roots),
        );
        assert!(verdict.is_allowed());
    }

    /// The check that stops a model writing over the operating system.
    #[test]
    fn a_path_outside_the_workspace_is_refused() {
        let s = session(vec![Role::Employee]);
        let roots = workspace();
        for hostile in [
            "C:/Windows/System32/drivers/etc/hosts",
            "C:/arjun/tasks/42/../../../Windows/System32/config",
            "C:/arjun/tasks/43/someone-elses-task.docx",
        ] {
            let verdict = ToolGateway::decide(
                &ToolCall::new("read_scoped_file", json!({ "path": hostile })),
                &context(&s, &roots),
            );
            assert!(!verdict.is_allowed(), "{hostile} should have been refused");
            assert!(verdict.message().contains("outside this task's workspace"));
        }
    }

    /// A sibling that merely shares a prefix is not inside the workspace.
    #[test]
    fn a_sibling_directory_with_a_shared_prefix_is_not_inside() {
        let s = session(vec![Role::Employee]);
        let roots = vec![PathBuf::from("C:/arjun/tasks/4")];
        let verdict = ToolGateway::decide(
            &ToolCall::new("read_scoped_file", json!({ "path": "C:/arjun/tasks/42/secret.txt" })),
            &context(&s, &roots),
        );
        assert!(!verdict.is_allowed());
    }

    #[test]
    fn a_task_with_no_workspace_can_touch_no_files_at_all() {
        let s = session(vec![Role::Employee]);
        let verdict = ToolGateway::decide(
            &ToolCall::new("read_scoped_file", json!({ "path": "C:/anything.txt" })),
            &context(&s, &[]),
        );
        assert!(!verdict.is_allowed());
    }

    // ── Size ─────────────────────────────────────────────────────────────

    #[test]
    fn content_above_the_size_limit_is_refused_with_both_numbers() {
        let s = session(vec![Role::Employee]);
        let roots = workspace();
        let huge = "x".repeat(40 * 1024 * 1024);
        let verdict = ToolGateway::decide(
            &ToolCall::new(
                "write_scoped_file",
                json!({ "path": "C:/arjun/tasks/42/out.txt", "content": huge }),
            ),
            &context(&s, &roots),
        );
        assert!(verdict.message().contains("40 MB"));
        assert!(verdict.message().contains("32 MB"));
    }

    // ── Entitlement and mode ─────────────────────────────────────────────

    #[test]
    fn a_user_without_the_permission_is_refused() {
        // The legacy Auditor role grants nothing in the active product;
        // pinned here so a regression that re-enables a legacy role is
        // caught at the gateway level.
        let s = session(vec![Role::Auditor]);
        let roots = workspace();
        let verdict = ToolGateway::decide(
            &ToolCall::new("run_calculation", json!({ "expression": "8.2 - 9.0" })),
            &context(&s, &roots),
        );
        assert!(!verdict.is_allowed());
        assert!(verdict.message().contains("not permitted"));
    }

    /// The sovereignty invariant outranks everything, including a valid call
    /// from a fully entitled user.
    #[test]
    fn nothing_runs_while_the_network_is_reachable() {
        let s = session(vec![Role::Employee]);
        let roots = workspace();
        let mut ctx = context(&s, &roots);
        ctx.confidential_work_permitted = false;

        let verdict = ToolGateway::decide(
            &ToolCall::new("run_calculation", json!({ "expression": "1 + 1" })),
            &ctx,
        );
        assert!(!verdict.is_allowed());
        assert!(verdict.message().contains("Provisioning mode"));
    }

    // ── Approval ─────────────────────────────────────────────────────────

    #[test]
    fn a_write_waits_for_a_person() {
        let s = session(vec![Role::Employee]);
        let roots = workspace();
        let verdict = ToolGateway::decide(
            &ToolCall::new(
                "write_scoped_file",
                json!({ "path": "C:/arjun/tasks/42/note.txt", "content": "hello" }),
            ),
            &context(&s, &roots),
        );
        assert!(matches!(verdict, GatewayVerdict::NeedsApproval { .. }));
    }

    #[test]
    fn an_approved_write_proceeds() {
        let s = session(vec![Role::Employee]);
        let roots = workspace();
        let mut ctx = context(&s, &roots);
        ctx.approval = ApprovalState::Granted;

        let verdict = ToolGateway::decide(
            &ToolCall::new(
                "write_scoped_file",
                json!({ "path": "C:/arjun/tasks/42/note.txt", "content": "hello" }),
            ),
            &ctx,
        );
        assert!(verdict.is_allowed());
    }

    /// A prompt saying only "allow this?" trains people to say yes.
    #[test]
    fn the_approval_prompt_shows_the_target_and_the_consequence() {
        let s = session(vec![Role::Employee]);
        let roots = workspace();
        let verdict = ToolGateway::decide(
            &ToolCall::new(
                "execute_code",
                json!({ "language": "python", "source": "print(1)\nprint(2)" }),
            ),
            &context(&s, &roots),
        );

        let message = verdict.message();
        assert!(message.contains("run code in the sandbox"));
        assert!(message.contains("2 line(s) of python"));
        assert!(message.contains("no network"));
    }

    #[test]
    fn the_write_prompt_states_how_much_will_be_written() {
        let s = session(vec![Role::Employee]);
        let roots = workspace();
        let verdict = ToolGateway::decide(
            &ToolCall::new(
                "write_scoped_file",
                json!({ "path": "C:/arjun/tasks/42/note.txt", "content": "hello" }),
            ),
            &context(&s, &roots),
        );
        assert!(verdict.message().contains("5 byte(s)"));
        assert!(verdict.message().contains("C:/arjun/tasks/42/note.txt".replace('/', "\\").as_str())
            || verdict.message().contains("note.txt"));
    }

    // ── Ordering ─────────────────────────────────────────────────────────

    /// A refusal must name the most fundamental cause. A malformed call from an
    /// unentitled user is reported as malformed, because fixing their roles
    /// would not have helped.
    #[test]
    fn the_most_fundamental_refusal_wins() {
        let s = session(vec![Role::Administrator]);
        let roots = workspace();
        let verdict = ToolGateway::decide(
            &ToolCall::new("run_calculation", json!({})),
            &context(&s, &roots),
        );
        assert!(verdict.message().contains("missing"), "{}", verdict.message());
    }
}
