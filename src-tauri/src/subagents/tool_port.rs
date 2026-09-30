//! A worker's way through the runtime's own tool gateway (P09).
//!
//! A worker that writes or checks a deliverable has to do it the way a model
//! loop would: through `authorize` and `execute`, under a plan narrowed to its
//! granted tools, with every call a durable receipt and every version
//! registered by the same code that registers a parent's. It must not reach
//! the conversation store or the document writer directly -- that would be a
//! second path to the same effects, with none of the gateway's refusals.
//!
//! The runtime's dependencies are built when the runtime first starts, after
//! the workers exist, so the port is late-bound into a slot the runtime fills
//! (the same arrangement as the child loop's `AgentRuntimeHandle`). An empty
//! slot is reported by name: the work is then blocked, not faked.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use serde_json::Value;

use crate::orchestrator::tools::ToolName;

/// What one successful call returned, and the event it was recorded as.
#[derive(Debug, Clone)]
pub struct PortAnswer {
    pub text: String,
    pub event_seq: Option<i64>,
    pub output_sha256: Option<String>,
}

#[async_trait]
pub trait ToolPort: Send + Sync {
    /// Makes `child_run` a run the gateway knows: a plan narrowed to `tools`,
    /// the parent's workspace, and the parent's conversation.
    fn open(
        &self,
        child_run: &str,
        parent_run: &str,
        workspace_root: &Path,
        tools: &[ToolName],
        max_steps: u32,
        max_duration: Duration,
    ) -> Result<(), String>;

    /// One call through `authorize` and `execute`. A refusal or a failed tool
    /// is an `Err` carrying the gateway's or the tool's own words.
    async fn call(&self, child_run: &str, tool: ToolName, args: Value) -> Result<PortAnswer, String>;

    /// Removes what `open` registered.
    fn close(&self, child_run: &str);
}

/// The task a child run works for, by the child's run id.
///
/// A version or an open question a child registers belongs to its parent's
/// task -- that is where the parent, its siblings and the reviewer read the
/// task's memory -- not to a scope named after one attempt. Registered when a
/// child run is opened and removed when it closes; a run that is not a child
/// is its own task.
static CHILD_TASKS: std::sync::LazyLock<std::sync::Mutex<std::collections::HashMap<String, String>>> =
    std::sync::LazyLock::new(Default::default);

/// Records that `child_run` works for `parent_run`'s task.
pub fn adopt(child_run: &str, parent_run: &str) {
    let task = task_of(parent_run);
    if let Ok(mut table) = CHILD_TASKS.lock() {
        table.insert(child_run.to_string(), task);
    }
}

/// Forgets a child run.
pub fn release(child_run: &str) {
    if let Ok(mut table) = CHILD_TASKS.lock() {
        table.remove(child_run);
    }
}

/// The task `run_id` works for: its parent's, for a child; its own otherwise.
pub fn task_of(run_id: &str) -> String {
    CHILD_TASKS.lock().ok().and_then(|table| table.get(run_id).cloned()).unwrap_or_else(|| run_id.to_string())
}

/// Filled by the runtime when it starts.
pub type ToolPortSlot = Arc<std::sync::RwLock<Option<Arc<dyn ToolPort>>>>;

/// The port in a slot, or why there is none.
pub fn port_in(slot: &ToolPortSlot) -> Result<Arc<dyn ToolPort>, String> {
    slot.read()
        .map_err(|_| "the tool port lock is poisoned".to_string())?
        .clone()
        .ok_or_else(|| {
            "the runtime's tool gateway has not started on this machine, so a worker cannot write or check a \
             deliverable through it. Nothing was done."
                .to_string()
        })
}

/// Closes a port's child run however the work ends.
pub struct OpenRun {
    port: Arc<dyn ToolPort>,
    child_run: String,
}

impl OpenRun {
    pub fn new(port: Arc<dyn ToolPort>, child_run: &str) -> Self {
        Self { port, child_run: child_run.to_string() }
    }

    pub async fn call(&self, tool: ToolName, args: Value) -> Result<PortAnswer, String> {
        self.port.call(&self.child_run, tool, args).await
    }
}

impl Drop for OpenRun {
    fn drop(&mut self) {
        self.port.close(&self.child_run);
    }
}
