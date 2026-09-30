//! The runtime's side of a worker's [`ToolPort`] (P09).
//!
//! A worker's call goes through exactly the functions a model loop's call
//! goes through: `authorize` against a plan narrowed to the worker's granted
//! tools, then `execute`, which records the durable receipt and registers
//! whatever the tool produced. Nothing here grants anything the plan does not.

use std::path::Path;
use std::sync::{Arc, Weak};
use std::time::Duration;

use async_trait::async_trait;
use serde_json::{json, Value};

use crate::orchestrator::plan::{Budget, PlanRun};
use crate::orchestrator::tools::ToolName;
use crate::subagents::tool_port::{PortAnswer, ToolPort};

use super::RuntimeDeps;

/// Held weakly: the runtime's dependencies own the workers (through the
/// subagent manager), and a strong handle back would be a cycle.
pub struct RuntimeToolPort {
    deps: Weak<RuntimeDeps>,
}

impl RuntimeToolPort {
    pub fn new(deps: &Arc<RuntimeDeps>) -> Self {
        Self { deps: Arc::downgrade(deps) }
    }

    fn deps(&self) -> Result<Arc<RuntimeDeps>, String> {
        self.deps.upgrade().ok_or_else(|| "the runtime has stopped, so no tool call can be made".to_string())
    }
}

#[async_trait]
impl ToolPort for RuntimeToolPort {
    fn open(
        &self,
        child_run: &str,
        parent_run: &str,
        workspace_root: &Path,
        tools: &[ToolName],
        max_steps: u32,
        max_duration: Duration,
    ) -> Result<(), String> {
        let deps = self.deps()?;
        let mut budget = Budget::standard(tools.to_vec());
        budget.max_steps = max_steps.max(1);
        budget.max_duration = max_duration;
        deps.plans
            .lock()
            .map_err(|_| "the plan table is poisoned".to_string())?
            .insert(child_run.to_string(), PlanRun::new(child_run, vec![format!("work for {parent_run}")], budget));
        if let Ok(mut workspaces) = deps.workspaces.lock() {
            workspaces.insert(child_run.to_string(), super::workspace::Workspace::at(workspace_root.to_path_buf()));
        }
        // The parent's conversation, so a version the child registers is one
        // the parent and the person can see -- and one a reviewer can reach.
        if let Some(conversation) = deps.run_to_conversation.lookup(parent_run) {
            deps.run_to_conversation.bind(child_run, &conversation);
        }
        crate::subagents::tool_port::adopt(child_run, parent_run);
        Ok(())
    }

    async fn call(&self, child_run: &str, tool: ToolName, args: Value) -> Result<PortAnswer, String> {
        let deps = self.deps()?;
        let tool_call_id = format!("{child_run}:{}:{}", tool.as_str(), uuid::Uuid::new_v4().simple());
        let request = json!({ "runId": child_run, "toolCallId": tool_call_id, "tool": tool.as_str(), "args": args });
        let allow = super::authorize(request.clone(), &deps).await.map_err(|e| e.message)?;
        let Some(grant) = allow.get("grant").and_then(Value::as_str) else {
            return Err(format!(
                "the gateway refused {}: {}",
                tool.as_str(),
                allow.get("reason").and_then(Value::as_str).unwrap_or("no reason given")
            ));
        };
        let mut spent = request;
        spent["grant"] = json!(grant);
        let result = super::execute(spent, &deps).await.map_err(|e| e.message)?;
        let text = result.get("text").and_then(Value::as_str).unwrap_or_default().to_string();
        let (event_seq, output_sha256) = deps
            .calls
            .lock()
            .ok()
            .and_then(|table| {
                table.get(child_run).and_then(|calls| {
                    calls
                        .iter()
                        .rev()
                        .find(|c| c.tool_call_id.as_deref() == Some(tool_call_id.as_str()))
                        .map(|c| (c.event_seq, c.output_sha256.clone()))
                })
            })
            .unwrap_or((None, None));
        Ok(PortAnswer { text, event_seq, output_sha256 })
    }

    fn close(&self, child_run: &str) {
        let Ok(deps) = self.deps() else { return };
        if let Ok(mut plans) = deps.plans.lock() {
            plans.remove(child_run);
        }
        if let Ok(mut workspaces) = deps.workspaces.lock() {
            workspaces.remove(child_run);
        }
        deps.run_to_conversation.unbind(child_run);
        crate::subagents::tool_port::release(child_run);
    }
}
