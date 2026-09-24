//! The orchestrator's tools, answered inside the main run.
//!
//! ## One coordinator, not a nested one
//!
//! The coordinating model (Spark X2.5 4B Q8 in the target deployment) is the
//! model of the *main run*. It plans with `task.plan_update`, hands scoped jobs
//! to specialists with `agent.delegate`, watches them with `agent.status`,
//! stops them with `agent.cancel` and gates deliverables with
//! `task.request_review` — each an ordinary tool call through the same gateway,
//! plan budget, approval queue and event log as every other call. There is no
//! second orchestrator process and no child that can delegate: a job's child is
//! narrowed by `SubagentManager` exactly as `agent.delegate_readonly`'s is, and
//! no profile holds these tools.
//!
//! ## What each tool is allowed to decide
//!
//! The model proposes; Rust decides and records:
//!
//! - The plan changes only by validated patches ([`super::task_plan`]); step
//!   status and receipts are written only here, from settled jobs, verified
//!   tool receipts and reviews.
//! - A job is dispatched only for a pending delegate step whose prerequisites
//!   are complete, whose memory requirements already hold, whose repair budget
//!   has an attempt left, and while fewer than [`MAX_RUNNING_JOBS_PER_RUN`] of
//!   the run's jobs are running.
//! - A role that writes is escalated to a person before the call runs (see
//!   [`approval_needed`], consulted by the runtime's `decide`), and the handler
//!   refuses a writer job whose call carries no recorded approval. The
//!   read-only alias never reaches this path.
//! - The definition is resolved from the registry at dispatch and recorded
//!   with its version; the manager resolves it again and pins it into the
//!   packet, so an edit saved mid-job reaches the next job, not this one.
//!
//! ## Model switching and the card
//!
//! P05 is meant to *use* P03's parent-lease suspension: release the
//! coordinator's model while a capacity-dependent child runs on another, and
//! rebind it afterwards. P03 is not built in this checkout. So each job records
//! the lease decision it actually got ([`lease_decision`]): a child on the
//! coordinator's own model shares it; a child on another model is serialised by
//! `subagents::scheduling` and the coordinator's model is **not** suspended.
//! The record says so rather than implying a suspension that did not happen.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use tokio::sync::watch;

use super::cancellation::CancelToken;
use super::events::plans::{JobRecord, JobStatus, PlanWriteError};
use super::events::{TaskEventLog, TaskEventType};
use super::task_plan::{self, JobSettlement, StepKind, StepRef, StepStatus, TaskPlan};
use super::{CallParams, RuntimeDeps};
use crate::identity::Session;
use crate::knowledge::graph::runtime_memory::{
    Authority, ItemStatus, MemoryItem, MemoryKind, MemoryScope, Provenance,
};
use crate::orchestrator::tools::{ToolCall, ToolName};
use crate::subagents::manager::{DelegationMode, Dispatch, SpawnRefusal, Spawned};
use crate::subagents::profile::Isolation;
use crate::subagents::{ChildResult, InputRef};

/// Jobs one run may have running at once. Two: a coordinator on a 4B model
/// keeps track of two concurrent specialists reliably, and the card behind them
/// holds one heavy generation at a time anyway (`subagents::scheduling`).
pub const MAX_RUNNING_JOBS_PER_RUN: usize = 2;
/// Longest `agent.delegate` waits before answering "still running". Below the
/// tool's 120-second ceiling, so the answer always arrives before the loop
/// stops listening; a job that outlives the wait keeps running and is reported
/// as running.
pub const MAX_WAIT_SECONDS: u64 = 100;
/// What `agent.delegate` waits when the call does not say.
pub const DEFAULT_WAIT_SECONDS: u64 = 60;
/// Longest `agent.status` waits.
pub const MAX_STATUS_WAIT_SECONDS: u64 = 100;
/// The reviewer child's deadline inside `task.request_review`, which must
/// answer within its own 120-second ceiling after the validation ran.
const REVIEWER_DEADLINE_SECONDS: u64 = 40;
/// A job's deadline when the call does not give one.
pub const DEFAULT_DEADLINE_SECONDS: u64 = 180;
/// The longest deadline a job may ask for. The definition's own limit applies
/// too, and the shorter wins.
pub const MAX_DEADLINE_SECONDS: u64 = 600;
/// Longest objective a job may carry.
const MAX_OBJECTIVE: usize = 800;
/// A backend plan write that loses a race with another process retries this
/// many times before it is logged and dropped.
const PLAN_WRITE_RETRIES: usize = 8;
/// Findings kept in a job's durable summary, each bounded.
const KEPT_FINDINGS: usize = 8;

/// The jobs this process is running, and the lock plan writes take.
#[derive(Default)]
pub struct JobBoard {
    live: Mutex<HashMap<String, LiveJob>>,
    /// Serialises a plan's read-modify-write inside this process, so the
    /// coordinator's patch and a job settling in the background never race to
    /// the same version. Across processes, `append_plan`'s version check does.
    plan_writes: Mutex<()>,
}

#[derive(Clone)]
struct LiveJob {
    run_id: String,
    cancel: CancelToken,
    /// Flips to `true` once the job has settled and its plan version is written.
    done: watch::Receiver<bool>,
    /// Set when a correction stopped it, so the attempt is not charged.
    superseded: Arc<AtomicBool>,
}

impl JobBoard {
    fn running_for_run(&self, run_id: &str) -> usize {
        self.live
            .lock()
            .map(|live| live.values().filter(|job| job.run_id == run_id).count())
            .unwrap_or(0)
    }

    fn get(&self, job_id: &str) -> Option<LiveJob> {
        self.live.lock().ok()?.get(job_id).cloned()
    }

    /// Stops one job. `false` when this process is not running it.
    pub fn cancel(&self, job_id: &str, superseded: bool) -> bool {
        match self.get(job_id) {
            Some(job) => {
                if superseded {
                    job.superseded.store(true, Ordering::SeqCst);
                }
                job.cancel.cancel();
                true
            }
            None => false,
        }
    }

    /// Stops every job of a run. Returns how many were told to stop.
    pub fn cancel_run(&self, run_id: &str) -> usize {
        let jobs: Vec<LiveJob> = self
            .live
            .lock()
            .map(|live| live.values().filter(|job| job.run_id == run_id).cloned().collect())
            .unwrap_or_default();
        for job in &jobs {
            job.cancel.cancel();
        }
        jobs.len()
    }

    /// Job ids of a run that this process is running.
    pub fn live_jobs(&self, run_id: &str) -> Vec<String> {
        self.live
            .lock()
            .map(|live| {
                live.iter()
                    .filter(|(_, job)| job.run_id == run_id)
                    .map(|(id, _)| id.clone())
                    .collect()
            })
            .unwrap_or_default()
    }
}

fn now() -> String {
    chrono::Utc::now().to_rfc3339()
}

fn sha256_hex(text: &str) -> String {
    format!("{:x}", Sha256::digest(text.as_bytes()))
}

// ─── The plan ──────────────────────────────────────────────────────────────

/// Checks a claimed tool receipt against the run's durable events.
struct EventReceipts<'a>(&'a TaskEventLog);

impl task_plan::ReceiptSource for EventReceipts<'_> {
    fn tool_succeeded(&self, run_id: &str, event_seq: i64) -> Option<task_plan::ToolReceipt> {
        let event = self.0.event_at(run_id, event_seq).ok()??;
        if event.event_type != TaskEventType::ToolSucceeded {
            return None;
        }
        let tool = event.payload.get("tool").and_then(Value::as_str)?.to_string();
        // An orchestrator call is not work a step can be settled by.
        if matches!(
            ToolName::from_str(&tool),
            Some(ToolName::TaskPlanUpdate | ToolName::AgentStatus | ToolName::AgentCancel)
        ) {
            return None;
        }
        Some(task_plan::ToolReceipt {
            tool,
            output_sha256: event
                .payload
                .get("detail")
                .and_then(|detail| detail.get("sha256"))
                .and_then(Value::as_str)
                .map(str::to_string),
        })
    }
}

/// Applies `change` to the run's newest plan and writes the result, as the
/// backend. `None` when there is no plan, the change does not apply, or the
/// write failed (logged).
fn update_plan(
    deps: &Arc<RuntimeDeps>,
    run_id: &str,
    change: impl Fn(&TaskPlan) -> Option<TaskPlan>,
) -> Option<TaskPlan> {
    update_plan_in(&deps.events, &deps.jobs, run_id, change)
}

fn update_plan_in(
    events: &TaskEventLog,
    board: &JobBoard,
    run_id: &str,
    change: impl Fn(&TaskPlan) -> Option<TaskPlan>,
) -> Option<TaskPlan> {
    let _serial = board.plan_writes.lock().ok();
    for _ in 0..PLAN_WRITE_RETRIES {
        let current = match events.latest_plan(run_id) {
            Ok(Some(plan)) => plan,
            Ok(None) => return None,
            Err(error) => {
                log::warn!("[orchestrator] run {run_id}: the plan could not be read: {error}");
                return None;
            }
        };
        let next = change(&current)?;
        match events.append_plan(&next) {
            Ok(()) => return Some(next),
            Err(PlanWriteError::Conflict { .. }) => continue,
            Err(PlanWriteError::Storage(detail)) => {
                log::warn!("[orchestrator] run {run_id}: a plan version was not written: {detail}");
                return None;
            }
        }
    }
    log::warn!("[orchestrator] run {run_id}: a plan version lost {PLAN_WRITE_RETRIES} races and was dropped");
    None
}

/// `task.plan_update`.
pub(super) fn plan_update(
    deps: &Arc<RuntimeDeps>,
    call: &CallParams,
    _session: &Session,
    tool_call: &ToolCall,
) -> Result<String, String> {
    let patch = task_plan::parse_patch(&tool_call.arguments).map_err(|reason| {
        format!("{reason} Nothing was changed.")
    })?;
    let _serial = deps.jobs.plan_writes.lock().ok();
    let current = deps.events.latest_plan(&call.run_id)?;
    let next = match task_plan::apply_patch(
        &call.run_id,
        current.as_ref(),
        &patch,
        &EventReceipts(&deps.events),
        &now(),
    ) {
        Ok(next) => next,
        Err(refusal) => {
            let mut out = refusal.explain();
            if let Some(plan) = &current {
                out.push_str("\n\n");
                out.push_str(&task_plan::project(plan));
            }
            return Err(out);
        }
    };
    match deps.events.append_plan(&next) {
        Ok(()) => {}
        Err(PlanWriteError::Conflict { current }) => {
            return Err(format!(
                "The plan moved on to version {current} while this patch was being applied. \
                 Nothing was changed; read it with agent.status and patch that version."
            ))
        }
        Err(PlanWriteError::Storage(detail)) => {
            return Err(format!("The plan could not be written: {detail}. Nothing was changed."))
        }
    }
    Ok(format!(
        "Plan version {} written ({}).\n\n{}",
        next.version,
        next.reason,
        task_plan::project(&next)
    ))
}

// ─── Capability ────────────────────────────────────────────────────────────

/// The roles this deployment can delegate to, as `capability.search` lists
/// them: resolved from the registry now, with whether a worker exists, whether
/// the role writes, and the definition version a job would pin.
pub(super) fn roles(deps: &Arc<RuntimeDeps>) -> Vec<Value> {
    let mut names: Vec<String> = deps.subagents.profiles().map(|profile| profile.name.clone()).collect();
    names.sort();
    names
        .into_iter()
        .map(|name| match deps.subagents.resolve(&name) {
            Ok(resolved) => {
                let performable = deps.subagents.has_worker(&resolved.capability)
                    || deps.subagents.has_worker(&resolved.profile.name);
                let writes = resolved.profile.isolation != Isolation::ReadOnly;
                json!({
                    "role": name,
                    "agentId": resolved.agent_id,
                    "definitionVersion": resolved.definition_version,
                    "origin": resolved.origin.as_str(),
                    "capability": resolved.capability,
                    "status": if performable { "ready" } else { "no worker in this build" },
                    "writes": writes,
                    "approval": if writes { "a person approves each job" } else { "none" },
                    "output": resolved.profile.required_schema.as_str(),
                    "modelRole": resolved.profile.model_role.label(),
                    "maxSeconds": resolved.profile.limits.max_duration_seconds,
                    "tools": resolved.profile.allowed_tools.iter().map(|t| t.as_str()).collect::<Vec<_>>(),
                })
            }
            Err(refusal) => json!({
                "role": name,
                "status": "unavailable",
                "detail": refusal.explain(),
            }),
        })
        .collect()
}

/// The role lines `capability.search` appends for the model.
pub(super) fn render_roles(roles: &[Value]) -> String {
    if roles.is_empty() {
        return String::new();
    }
    let mut out = String::from("\nSpecialist roles agent.delegate can hand a job to:\n");
    for role in roles {
        let name = role.get("role").and_then(Value::as_str).unwrap_or("?");
        let status = role.get("status").and_then(Value::as_str).unwrap_or("?");
        if status == "unavailable" {
            out.push_str(&format!("- {name}: unavailable — {}\n", role.get("detail").and_then(Value::as_str).unwrap_or("")));
            continue;
        }
        out.push_str(&format!(
            "- {name} ({}, {} v{}): {status}; returns {}; {}{}\n",
            role.get("capability").and_then(Value::as_str).unwrap_or("?"),
            role.get("origin").and_then(Value::as_str).unwrap_or("?"),
            role.get("definitionVersion").and_then(Value::as_u64).map(|v| v.to_string()).unwrap_or_else(|| "-".into()),
            role.get("output").and_then(Value::as_str).unwrap_or("?"),
            if role.get("writes").and_then(Value::as_bool) == Some(true) {
                "writes, so a person approves each job"
            } else {
                "read-only"
            },
            match role.get("maxSeconds").and_then(Value::as_u64) {
                Some(seconds) => format!("; at most {seconds}s"),
                None => String::new(),
            },
        ));
    }
    out
}

// ─── Approval for writer jobs ──────────────────────────────────────────────

/// Why a person must approve this `agent.delegate` call, or `None` when the
/// job only reads.
///
/// Consulted by the runtime's `decide` before the call is authorised, so a
/// writer job goes through the approval queue a direct write does. A call this
/// cannot read (no role, an unknown role) is not escalated here; the handler
/// refuses it with the reason.
pub(super) fn approval_needed(deps: &Arc<RuntimeDeps>, call: &CallParams) -> Option<String> {
    let role = call.args.get("role").and_then(Value::as_str)?;
    let step = call.args.get("step").and_then(Value::as_str).unwrap_or("?");
    let resolved = deps.subagents.resolve(role).ok()?;
    let requested = requested_tools(&call.args).ok()?;
    // A request the handler will refuse -- a tool the role is not given -- is
    // not put to a person: nobody should be asked to approve a job that cannot
    // be started, and the refusal names what is wrong.
    if requested.iter().any(|tool| !resolved.profile.allowed_tools.contains(tool)) {
        return None;
    }
    let writes = writer_job(&resolved.profile.isolation, &requested, &resolved.profile.allowed_tools);
    writes.then(|| {
        let tools: Vec<&str> = if requested.is_empty() {
            resolved.profile.allowed_tools.iter().filter(|t| !t.is_read_only()).map(|t| t.as_str()).collect()
        } else {
            requested.iter().filter(|t| !t.is_read_only()).map(|t| t.as_str()).collect()
        };
        format!(
            "Start a {role} job for plan step {step} that can write: it may use {}. Each write \
             it makes is still checked and approved on its own.",
            if tools.is_empty() { "its writing tools".to_string() } else { tools.join(", ") }
        )
    })
}

/// Whether a job may write: its role is declared as writing, or the call asked
/// for a tool that writes. A read-only role asked for nothing extra is run
/// read-only, and the manager withholds any write tool an edited definition
/// gave it.
fn writer_job(isolation: &Isolation, requested: &[ToolName], _declared: &[ToolName]) -> bool {
    *isolation != Isolation::ReadOnly || requested.iter().any(|tool| !tool.is_read_only())
}

fn requested_tools(arguments: &Value) -> Result<Vec<ToolName>, String> {
    let Some(items) = arguments.get("allowed_tools") else {
        return Ok(Vec::new());
    };
    let items = items.as_array().ok_or("`allowed_tools` must be a list of tool names.")?;
    items
        .iter()
        .map(|item| {
            let name = item.as_str().ok_or("each entry in `allowed_tools` is a tool name")?;
            ToolName::from_str(name).ok_or_else(|| format!("{name:?} is not a tool this runtime knows."))
        })
        .collect()
}

/// Whether a person approved this exact call, by the durable record.
fn approved(deps: &Arc<RuntimeDeps>, call: &CallParams) -> bool {
    deps.events
        .events_since(&call.run_id, 0)
        .map(|page| {
            page.events.iter().any(|event| {
                event.event_type == TaskEventType::ApprovalDecided
                    && event.payload.get("toolCallId").and_then(Value::as_str) == Some(call.tool_call_id.as_str())
                    && event.payload.get("approved").and_then(Value::as_bool) == Some(true)
            })
        })
        .unwrap_or(false)
}

// ─── Delegation ────────────────────────────────────────────────────────────

/// The model this run is on and the attempt it is in, off its checkpoint seed.
fn parent_model_and_attempt(deps: &Arc<RuntimeDeps>, run_id: &str) -> (Option<String>, Option<String>) {
    deps.checkpoints
        .lock()
        .ok()
        .and_then(|seeds| seeds.get(run_id).map(|seed| (Some(seed.model_id.clone()), Some(seed.attempt_id.clone()))))
        .unwrap_or((None, None))
}

/// What happens to the card for this job, as a line for the record.
pub fn lease_decision(parent_model: &str, child_model: &str) -> String {
    if child_model.trim().is_empty() {
        "no model: the role runs without one".to_string()
    } else if child_model == parent_model {
        format!("shares the coordinator's model {parent_model}; no second model is loaded")
    } else {
        format!(
            "routed to {child_model}; the coordinator's model {parent_model} is not suspended \
             (P03 parent-lease suspension is not built), so the child is serialised on the card \
             by subagents::scheduling"
        )
    }
}

/// `agent.delegate`.
pub(super) async fn delegate(
    deps: &Arc<RuntimeDeps>,
    call: &CallParams,
    session: &Session,
    tool_call: &ToolCall,
) -> Result<String, String> {
    let step_id = tool_call.text("step").ok_or("`step` names the plan step this job is for.")?.trim().to_string();
    let role = tool_call.text("role").ok_or("`role` names the specialist.")?.trim().to_string();
    let objective = tool_call.text("objective").ok_or("`objective` says what the job must establish.")?.trim().to_string();
    if objective.is_empty() || objective.chars().count() > MAX_OBJECTIVE {
        return Err(format!("The objective must be between 1 and {MAX_OBJECTIVE} characters. Nothing was started."));
    }
    let wait = Duration::from_secs(
        tool_call
            .integer("wait_seconds")
            .map(|seconds| seconds.max(0) as u64)
            .unwrap_or(DEFAULT_WAIT_SECONDS)
            .min(MAX_WAIT_SECONDS),
    );
    let deadline_seconds = tool_call
        .integer("deadline_seconds")
        .map(|seconds| seconds.max(1) as u64)
        .unwrap_or(DEFAULT_DEADLINE_SECONDS);
    if deadline_seconds > MAX_DEADLINE_SECONDS {
        return Err(format!("A job's deadline is at most {MAX_DEADLINE_SECONDS} seconds. Nothing was started."));
    }

    let plan = deps
        .events
        .latest_plan(&call.run_id)?
        .ok_or("This run has no plan yet. Create one with task.plan_update, with a delegate step for this job, before delegating. Nothing was started.")?;
    let step = plan
        .step(&step_id)
        .ok_or_else(|| format!("The plan has no step {step_id:?}. Nothing was started.\n\n{}", task_plan::project(&plan)))?
        .clone();
    if step.kind != StepKind::Delegate {
        return Err(format!(
            "Step {step_id} is a {} step; only a delegate step is handed to a specialist. Nothing was started.",
            step.kind.as_str()
        ));
    }
    if step.role.as_deref() != Some(role.as_str()) {
        return Err(format!(
            "Step {step_id} is for the {} role, not {role}. Nothing was started.",
            step.role.as_deref().unwrap_or("?")
        ));
    }

    // A second call for the same step is answered from what already happened
    // rather than starting a second job.
    match step.status {
        StepStatus::Running => {
            let job_id = step.job_id.clone().unwrap_or_default();
            let mut out = format!(
                "Step {step_id} already has job {job_id} running; no second job was started.\n\n"
            );
            out.push_str(&status_text(deps, &call.run_id, Some(&job_id), wait).await?);
            return Ok(out);
        }
        StepStatus::Completed | StepStatus::AwaitingReview => {
            let last = step.receipts.iter().rev().find(|receipt| receipt.kind == task_plan::ReceiptKind::Job);
            return Ok(format!(
                "Step {step_id} is already {} ({}); no second job was started.\n\n{}",
                step.status.as_str(),
                last.map(|receipt| format!("job {} — {}", receipt.reference, receipt.summary)).unwrap_or_default(),
                task_plan::project(&plan)
            ));
        }
        StepStatus::Pending => {}
        other => {
            return Err(format!(
                "Step {step_id} is {} ({}). Reopen it with task.plan_update if it should be tried \
                 again (attempts {}/{}). Nothing was started.",
                other.as_str(),
                step.note.as_deref().unwrap_or("no note"),
                step.repair.attempts,
                step.repair.max_attempts
            ))
        }
    }

    let unmet = plan.unmet_dependencies(&step);
    if !unmet.is_empty() {
        return Err(format!(
            "Step {step_id} waits on {}. It was not started and no attempt was used; delegate it \
             once those are complete.",
            unmet
                .iter()
                .map(|(id, status)| format!("{id} ({})", status.as_str()))
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    if step.repair.remaining() == 0 {
        return Err(format!("Step {step_id} has used all {} of its attempts. Nothing was started.", step.repair.max_attempts));
    }
    let running = deps.jobs.running_for_run(&call.run_id);
    if running >= MAX_RUNNING_JOBS_PER_RUN {
        return Err(format!(
            "{running} jobs are already running for this task, which is the limit. Wait for one \
             with agent.status and delegate this afterwards. Nothing was started."
        ));
    }

    // The definition, from the registry, now.
    let resolved = deps.subagents.resolve(&role).map_err(|refusal| format!("{} Nothing was started.", refusal.explain()))?;
    let requested = requested_tools(&tool_call.arguments).map_err(|reason| format!("{reason} Nothing was started."))?;
    let permitted = super::registered_plan_tools(deps, &call.run_id).unwrap_or_default();
    for tool in &requested {
        if !resolved.profile.allowed_tools.contains(tool) {
            return Err(format!(
                "The {role} role is not given {}; its tools are {}. Nothing was started.",
                tool.as_str(),
                resolved.profile.allowed_tools.iter().map(|t| t.as_str()).collect::<Vec<_>>().join(", ")
            ));
        }
        if !permitted.contains(tool) {
            return Err(format!(
                "This task may not use {}, so no job it starts may either. Nothing was started.",
                tool.as_str()
            ));
        }
    }
    let writer = writer_job(&resolved.profile.isolation, &requested, &resolved.profile.allowed_tools);
    if writer && !approved(deps, call) {
        // The runtime asks a person before a writer job's call runs; reaching
        // here without a recorded yes is a call that bypassed that, and it is
        // refused rather than trusted.
        return Err(format!(
            "A {role} job can write, and this call carries no person's approval. Nothing was \
             started."
        ));
    }

    // Memory the job needs, present before it starts.
    let mut after_revision = tool_call.integer("after_revision").map(i64::from).filter(|revision| *revision > 0);
    let memory_items = match tool_call.arguments.get("memory_items") {
        None => Vec::new(),
        Some(items) => items
            .as_array()
            .ok_or("`memory_items` is a list of mi-…#rN references. Nothing was started.")?
            .iter()
            .map(|item| item.as_str().unwrap_or_default().to_string())
            .collect(),
    };
    let mut required: Vec<(String, u64)> = Vec::new();
    for raw in &memory_items {
        match raw.split_once("#r").and_then(|(id, rev)| Some((id.trim().to_string(), rev.parse::<u64>().ok()?))) {
            Some(pair) => required.push(pair),
            None => return Err(format!("{raw:?} is not an mi-…#rN reference. Nothing was started.")),
        }
    }
    for input in &step.inputs {
        if let StepRef::Memory { item_id, revision } = input {
            required.push((item_id.clone(), *revision));
        }
    }
    if let Some(problem) = memory_not_ready(deps, session, &required) {
        return Err(format!("{problem} The job was not started and no attempt was used."));
    }

    // Inputs: the call's own references and the plan step's.
    let mut inputs = crate::orchestrator::runner::delegation_inputs(tool_call)
        .map_err(|reason| reason.to_string())?;
    for input in &step.inputs {
        match input {
            StepRef::Artifact { artifact_id, version } => {
                inputs.push(artifact_input(deps, call, session, artifact_id, *version)?);
            }
            StepRef::Document { sha256 } => inputs.push(InputRef::Document { sha256: sha256.clone(), page: None }),
            StepRef::Step { step_id: source } => {
                let producer = plan.step(source).expect("validated when the plan was written");
                for output in &producer.outputs {
                    if let StepRef::Artifact { artifact_id, version } = output {
                        inputs.push(artifact_input(deps, call, session, artifact_id, *version)?);
                    }
                }
            }
            StepRef::Memory { .. } => {}
        }
    }
    // A sibling's published memory: wait for the revision it landed at.
    for dependency in &step.depends_on {
        for job in deps.events.jobs_for_run(&call.run_id).unwrap_or_default() {
            if job.step_id == *dependency && job.status == JobStatus::Completed {
                if let Some(revision) = job.result.as_ref().and_then(|r| r.get("graphRevision")).and_then(Value::as_i64) {
                    after_revision = Some(after_revision.map_or(revision, |held| held.max(revision)));
                }
            }
        }
    }
    if inputs.is_empty() && resolved.capability != "knowledge-retriever" {
        return Err(format!(
            "The {role} role is pointed at things rather than asked a question, and neither the \
             call nor step {step_id} names any. Pass documents, files, expressions or artifacts. \
             Nothing was started."
        ));
    }

    // The model the child will use, chosen rather than asserted.
    let (parent_model, attempt_id) = parent_model_and_attempt(deps, &call.run_id);
    let parent_model = parent_model.ok_or("This run has no model recorded, so a child cannot be routed to one. Nothing was started.")?;
    let registry = deps
        .registry
        .as_deref()
        .ok_or("The model registry is not in reach, so a child's model cannot be chosen. Nothing was started.")?;
    let model_role = resolved.profile.model_role;
    let candidates: Vec<_> = registry
        .all()
        .iter()
        .filter(|entry| entry.enabled && entry.serves(model_role))
        .map(|entry| (entry, None))
        .collect();
    let decision = crate::subagents::certification::choose(model_role, &parent_model, &candidates);

    // The policy the child inherits: the run's grant, narrowed to what the
    // call asked for when it asked.
    let workspace = deps.root_for(&call.run_id).ok_or("This run has no workspace, so no child can be confined to one. Nothing was started.")?;
    let narrowed: Vec<ToolName> = if requested.is_empty() {
        permitted.clone()
    } else {
        permitted.iter().copied().filter(|tool| requested.contains(tool)).collect()
    };
    let inherited = crate::subagents::InheritedPolicy::of_run(
        session,
        crate::policy::Classification::Internal,
        &workspace,
        &narrowed,
    );

    // The record, before anything starts.
    let attempt = step.repair.attempts + 1;
    let job_id = format!("job-{}", &uuid::Uuid::new_v4().simple().to_string()[..12]);
    let child_id = uuid::Uuid::new_v4().to_string();
    let deadline = Duration::from_secs(deadline_seconds.min(resolved.profile.limits.max_duration_seconds.max(1)));
    let record = JobRecord {
        job_id: job_id.clone(),
        run_id: call.run_id.clone(),
        step_id: step_id.clone(),
        role: role.clone(),
        mode: if writer { DelegationMode::Writer } else { DelegationMode::ReadOnly }.as_str().to_string(),
        attempt,
        child_id: child_id.clone(),
        status: JobStatus::Queued,
        definition_id: resolved.agent_id.clone(),
        definition_version: resolved.definition_version,
        definition_origin: resolved.origin.as_str().to_string(),
        model_id: Some(decision.model_id.clone()).filter(|id| !id.is_empty()),
        parent_model_id: Some(parent_model.clone()),
        lease: lease_decision(&parent_model, &decision.model_id),
        objective_sha256: sha256_hex(&objective),
        deadline_at: (chrono::Utc::now() + chrono::Duration::from_std(deadline).unwrap_or_default()).to_rfc3339(),
        created_at: now(),
        settled_at: None,
        result: None,
        result_hash: None,
    };
    deps.events.record_job(&record).map_err(|error| format!("The job could not be recorded, so it was not started: {error}"))?;
    // `job_started` refuses a step that is no longer pending, so of two calls
    // racing for one step exactly one starts a job.
    if update_plan(deps, &call.run_id, |plan| task_plan::job_started(plan, &step_id, &job_id, &now())).is_none() {
        let _ = deps.events.settle_job(&job_id, JobStatus::Refused, &json!({"status": "refused", "detail": "the step was no longer pending, or the plan could not record the start"}), "", &now());
        return Err(format!(
            "Step {step_id} was no longer pending when this job was about to start (another call \
             started it), or the plan could not record the start. Nothing was started; read \
             agent.status."
        ));
    }

    // A role this build cannot perform is settled at once as blocked, by
    // name, rather than dispatched to fail or -- worse -- to be reported as
    // though something ran.
    let performable = deps.subagents.has_worker(&resolved.capability) || deps.subagents.has_worker(&resolved.profile.name);

    let (done_tx, done_rx) = watch::channel(false);
    let cancel = CancelToken::never();
    let superseded = Arc::new(AtomicBool::new(false));
    if let Ok(mut live) = deps.jobs.live.lock() {
        live.insert(
            job_id.clone(),
            LiveJob {
                run_id: call.run_id.clone(),
                cancel: cancel.clone(),
                done: done_rx,
                superseded: Arc::clone(&superseded),
            },
        );
    }

    let mut dispatch = Dispatch::for_task(&role, call.run_id.clone())
        .delivering(tool_call.text("deliverable").unwrap_or("findings with a citation for each"))
        .as_child(child_id.clone())
        .scoped_to(format!("step {step_id} attempt {attempt}"))
        .stoppable(cancel.clone())
        .within(deadline);
    if writer {
        dispatch = dispatch.writing();
    }
    if let Some(revision) = after_revision {
        dispatch = dispatch.after(revision);
    }
    if let Some(attempt_id) = attempt_id {
        dispatch = dispatch.in_attempt(attempt_id);
    }

    let task_deps = Arc::clone(deps);
    let task_session = session.clone();
    let task_record = record.clone();
    let task_role = role.clone();
    tokio::spawn(async move {
        let _ = task_deps.events.start_job(&task_record.job_id);
        let outcome = if performable {
            task_deps
                .subagents
                .spawn(&task_role, &inherited, &objective, inputs, decision, &dispatch)
                .await
        } else {
            Err(SpawnRefusal::NoWorker { profile: task_role.clone() })
        };
        settle(&task_deps, &task_session, &task_record, outcome, superseded.load(Ordering::SeqCst));
        if let Ok(mut live) = task_deps.jobs.live.lock() {
            live.remove(&task_record.job_id);
        }
        let _ = done_tx.send(true);
    });

    let mut out = format!(
        "Job {job_id} started for step {step_id}: {role} ({} v{}, {}), attempt {attempt}/{}, {} \
         mode, deadline {}s. Lease: {}.\n\n",
        resolved.agent_id,
        resolved.definition_version.map(|v| v.to_string()).unwrap_or_else(|| "-".into()),
        resolved.origin.as_str(),
        step.repair.max_attempts,
        if writer { "writer" } else { "read-only" },
        deadline.as_secs(),
        record.lease,
    );
    out.push_str(&status_text(deps, &call.run_id, Some(&job_id), wait).await?);
    Ok(out)
}

/// Why the memory a job needs is not there yet, or `None` when it is.
fn memory_not_ready(deps: &Arc<RuntimeDeps>, session: &Session, required: &[(String, u64)]) -> Option<String> {
    if required.is_empty() {
        return None;
    }
    let Some(graph) = deps.memory_graph.as_ref() else {
        return Some("This deployment has no shared memory graph, so the memory items the job needs cannot be checked.".into());
    };
    for (item_id, revision) in required {
        let versions = graph.versions_of(session, item_id, None).unwrap_or_default();
        let Some(latest) = versions.last() else {
            return Some(format!("The memory item {item_id} does not exist or is not readable yet."));
        };
        if latest.revision < *revision {
            return Some(format!(
                "The memory item {item_id} is at revision {} and the job needs revision {revision}.",
                latest.revision
            ));
        }
        if !latest.item.status.usable_as_evidence() {
            return Some(format!(
                "The memory item {item_id} is {} and cannot be relied on.",
                latest.item.status.as_str()
            ));
        }
    }
    None
}

/// An exact artifact version the job reads, resolved as the P04 tools resolve
/// one: owned by the signed-in person *and* produced in this run's
/// conversation, with the same answer for "absent" and "somebody else's".
fn artifact_input(
    deps: &Arc<RuntimeDeps>,
    call: &CallParams,
    session: &Session,
    artifact_id: &str,
    version: u32,
) -> Result<InputRef, String> {
    match super::artifact_tools::resolve(deps, call, session, artifact_id, Some(version)) {
        Ok(record) => Ok(InputRef::Artifact {
            artifact_id: artifact_id.to_string(),
            revision: version,
            sha256: record.sha256,
        }),
        Err(_) => Err(format!(
            "The artifact {artifact_id}@{version} does not exist in this conversation or is not \
             readable here. Nothing was started."
        )),
    }
}

/// Records a job's ending, durably, and settles its step.
fn settle(
    deps: &Arc<RuntimeDeps>,
    session: &Session,
    record: &JobRecord,
    outcome: Result<Spawned, SpawnRefusal>,
    superseded: bool,
) {
    let at = now();
    let (status, summary, settlement) = match &outcome {
        Ok(spawned) => {
            let child: &ChildResult = spawned.result();
            // A replay from the durable ledger carries the key as its child id
            // and no findings: the work was done by an earlier process. It is
            // reported for what it is rather than as this job's result.
            let replayed = spawned.is_reused() && child.child_id != record.child_id;
            let evidenced = child.findings.iter().filter(|finding| !finding.evidence.is_empty()).count();
            let artifacts: Vec<StepRef> = child
                .artifacts
                .iter()
                .map(|artifact| StepRef::Artifact { artifact_id: artifact.artifact_id.clone(), version: artifact.version })
                .collect();
            let artifacts_accepted = (!artifacts.is_empty()).then(|| {
                child.artifacts.iter().all(|artifact| {
                    deps.conversation_artifacts
                        .latest_validation(
                            &session.user.id,
                            &crate::artifacts::conversation_store::ArtifactRef {
                                artifact_id: artifact.artifact_id.clone(),
                                version: artifact.version,
                            },
                        )
                        .ok()
                        .flatten()
                        .is_some_and(|validation| validation.accepted)
                })
            });
            // What the child published, at the revision it landed, and how many
            // of those the graph admitted -- which it does only after resolving
            // the item's tool receipt against the event log. An admitted item is
            // a receipt-backed result even when the finding cites no passage (a
            // figure the calculation engine computed, a file re-opened).
            let mut admitted = 0usize;
            let published: Vec<StepRef> = child
                .published
                .iter()
                .filter_map(|id| {
                    let graph = deps.memory_graph.as_ref()?;
                    let latest = graph.versions_of(session, id, None).ok()?.pop()?;
                    if latest.item.status == ItemStatus::Admitted {
                        admitted += 1;
                    }
                    Some(StepRef::Memory { item_id: id.clone(), revision: latest.revision })
                })
                .collect();
            let status = if replayed { "failed".to_string() } else { child.status.as_str().to_string() };
            // Built from the counts rather than `ChildResult::describe`, which
            // calls a completed child's findings "partial" whenever it carries
            // a detail note -- and a completed retrieval always does (where it
            // published). The detail is kept only for a child that did not
            // finish, where it is the reason.
            let summary = if replayed {
                "this exact work was already run under the same key, and its findings are not held by this job".to_string()
            } else {
                let head = format!(
                    "{} {} — {} finding(s), {} citing a source, {} published",
                    child.profile,
                    child.status.describe(),
                    child.findings.len(),
                    evidenced,
                    child.published.len()
                );
                match child.detail.as_deref().and_then(|detail| detail.lines().find(|line| !line.trim().is_empty())) {
                    Some(reason) if !child.status.is_complete() => format!("{head}: {reason}"),
                    _ => head,
                }
            };
            let mut missing = child.missing.clone();
            if replayed {
                missing.push("findings from the earlier run of this work".into());
            }
            (
                status.clone(),
                summary.clone(),
                JobSettlement {
                    job_id: record.job_id.clone(),
                    status,
                    findings: if replayed { 0 } else { child.findings.len() },
                    evidenced: if replayed { 0 } else { evidenced },
                    receipts: if replayed { 0 } else { child.receipts.len() + admitted },
                    published,
                    artifacts,
                    artifacts_accepted,
                    result_hash: child.result_hash.clone(),
                    event_seq: stopped_event_seq(&deps.events, &record.run_id, &record.child_id),
                    summary,
                    missing,
                    superseded: superseded && child.status == crate::subagents::ChildStatus::Cancelled,
                },
            )
        }
        Err(refusal) => {
            let status = match refusal {
                SpawnRefusal::NoWorker { .. } => "blocked",
                _ => "refused",
            };
            let summary = refusal.explain();
            (
                status.to_string(),
                summary.clone(),
                JobSettlement {
                    job_id: record.job_id.clone(),
                    status: status.to_string(),
                    findings: 0,
                    evidenced: 0,
                    receipts: 0,
                    published: Vec::new(),
                    artifacts: Vec::new(),
                    artifacts_accepted: None,
                    result_hash: sha256_hex(&summary),
                    event_seq: None,
                    summary,
                    missing: match refusal {
                        SpawnRefusal::NoWorker { profile } => vec![format!("a worker for the {profile} role in this build")],
                        _ => Vec::new(),
                    },
                    superseded: false,
                },
            )
        }
    };

    let graph_revision = deps.memory_graph.as_ref().and_then(|graph| graph.graph_revision().ok());
    let findings: Vec<Value> = match &outcome {
        Ok(spawned) if settlement.findings > 0 => spawned
            .result()
            .findings
            .iter()
            .take(KEPT_FINDINGS)
            .map(|finding| {
                json!({
                    "statement": finding.statement.chars().take(300).collect::<String>(),
                    "evidence": finding.evidence.iter().map(|e| e.citation.clone()).collect::<Vec<_>>(),
                })
            })
            .collect(),
        _ => Vec::new(),
    };
    let result = json!({
        "status": status,
        "summary": summary.chars().take(400).collect::<String>(),
        "findings": settlement.findings,
        "evidenced": settlement.evidenced,
        "kept": findings,
        "published": settlement.published.iter().map(StepRef::describe).collect::<Vec<_>>(),
        "artifacts": settlement.artifacts.iter().map(StepRef::describe).collect::<Vec<_>>(),
        "artifactsAccepted": settlement.artifacts_accepted,
        "missing": settlement.missing,
        "graphRevision": graph_revision,
        "subagentStoppedSeq": settlement.event_seq,
        "superseded": settlement.superseded,
    });
    if let Err(error) = deps.events.settle_job(
        &record.job_id,
        JobStatus::from_child(&status),
        &result,
        &settlement.result_hash,
        &at,
    ) {
        log::warn!("[orchestrator] job {} could not be settled: {error}", record.job_id);
    }
    update_plan(deps, &record.run_id, |plan| task_plan::job_settled(plan, &record.step_id, &settlement, &at));
}

/// The `subagent_stopped` event this child left, by sequence.
fn stopped_event_seq(events: &TaskEventLog, run_id: &str, child_id: &str) -> Option<i64> {
    events.events_since(run_id, 0).ok()?.events.iter().rev().find_map(|event| {
        (event.event_type == TaskEventType::SubagentStopped
            && event.payload.get("childId").and_then(Value::as_str) == Some(child_id))
        .then_some(event.seq)
    })
}

// ─── Status and cancellation ───────────────────────────────────────────────

/// `agent.status`.
pub(super) async fn status(
    deps: &Arc<RuntimeDeps>,
    call: &CallParams,
    _session: &Session,
    tool_call: &ToolCall,
) -> Result<String, String> {
    let wait = Duration::from_secs(
        tool_call
            .integer("wait_seconds")
            .map(|seconds| seconds.max(0) as u64)
            .unwrap_or(0)
            .min(MAX_STATUS_WAIT_SECONDS),
    );
    status_text(deps, &call.run_id, tool_call.text("job"), wait).await
}

async fn status_text(
    deps: &Arc<RuntimeDeps>,
    run_id: &str,
    job_id: Option<&str>,
    wait: Duration,
) -> Result<String, String> {
    let jobs = deps.events.jobs_for_run(run_id)?;
    let selected: Vec<&JobRecord> = match job_id {
        Some(id) => {
            let found: Vec<&JobRecord> = jobs.iter().filter(|job| job.job_id == id).collect();
            if found.is_empty() {
                // The same answer for a job that exists in another run as for
                // one that does not exist at all.
                return Err(format!("This task has no job {id:?}."));
            }
            found
        }
        None => jobs.iter().collect(),
    };

    if !wait.is_zero() {
        let waiters: Vec<watch::Receiver<bool>> = selected
            .iter()
            .filter(|job| job.status.is_live())
            .filter_map(|job| deps.jobs.get(&job.job_id).map(|live| live.done))
            .collect();
        if !waiters.is_empty() {
            let _ = tokio::time::timeout(wait, async move {
                for mut waiter in waiters {
                    let _ = waiter.wait_for(|done| *done).await;
                }
            })
            .await;
        }
    }

    let jobs = deps.events.jobs_for_run(run_id)?;
    let mut out = String::new();
    for job in jobs.iter().filter(|job| job_id.is_none_or(|id| job.job_id == id)) {
        out.push_str(&render_job(job));
    }
    if out.is_empty() {
        out.push_str("This task has dispatched no jobs.\n");
    }
    if let Some(plan) = deps.events.latest_plan(run_id)? {
        out.push('\n');
        out.push_str(&task_plan::project(&plan));
    }
    Ok(out)
}

fn render_job(job: &JobRecord) -> String {
    let mut out = format!(
        "Job {} — step {}, {} ({} v{}, {}), attempt {}, {}: {}\n",
        job.job_id,
        job.step_id,
        job.role,
        job.definition_id,
        job.definition_version.map(|v| v.to_string()).unwrap_or_else(|| "-".into()),
        job.definition_origin,
        job.attempt,
        job.mode,
        job.status.as_str(),
    );
    if let Some(model) = &job.model_id {
        out.push_str(&format!("  model {model}; lease: {}\n", job.lease));
    }
    if let Some(result) = &job.result {
        if let Some(summary) = result.get("summary").and_then(Value::as_str) {
            out.push_str(&format!("  {summary}\n"));
        }
        if let Some(kept) = result.get("kept").and_then(Value::as_array) {
            for finding in kept {
                let statement = finding.get("statement").and_then(Value::as_str).unwrap_or_default();
                let evidence: Vec<&str> = finding
                    .get("evidence")
                    .and_then(Value::as_array)
                    .map(|items| items.iter().filter_map(Value::as_str).collect())
                    .unwrap_or_default();
                if evidence.is_empty() {
                    out.push_str(&format!("  - {statement} (no citation)\n"));
                } else {
                    out.push_str(&format!("  - {statement} [{}]\n", evidence.join("; ")));
                }
            }
        }
        for (key, label) in [("published", "published"), ("artifacts", "artifacts"), ("missing", "not done")] {
            let items: Vec<&str> = result
                .get(key)
                .and_then(Value::as_array)
                .map(|items| items.iter().filter_map(Value::as_str).collect())
                .unwrap_or_default();
            if !items.is_empty() {
                out.push_str(&format!("  {label}: {}\n", items.join(", ")));
            }
        }
        if let Some(revision) = result.get("graphRevision").and_then(Value::as_i64) {
            out.push_str(&format!("  shared memory at graph revision {revision}\n"));
        }
    }
    out
}

/// `agent.cancel`.
pub(super) fn cancel(
    deps: &Arc<RuntimeDeps>,
    call: &CallParams,
    _session: &Session,
    tool_call: &ToolCall,
) -> Result<String, String> {
    let job_id = tool_call.text("job").ok_or("`job` names the job to stop.")?.trim().to_string();
    let job = deps
        .events
        .job(&job_id)?
        .filter(|job| job.run_id == call.run_id)
        .ok_or_else(|| format!("This task has no job {job_id:?}."))?;
    if !job.status.is_live() {
        return Ok(format!(
            "Job {job_id} already ended ({}); nothing was stopped.",
            job.status.as_str()
        ));
    }
    if deps.jobs.cancel(&job_id, false) {
        Ok(format!(
            "Job {job_id} was told to stop{}. It settles as cancelled within a few seconds; \
             agent.status shows the final state and the step it leaves pending or cancelled.",
            tool_call.text("reason").map(|reason| format!(" ({reason})")).unwrap_or_default()
        ))
    } else {
        Err(format!(
            "Job {job_id} is recorded as {} but this process is not running it; it will be \
             marked interrupted when the application next starts.",
            job.status.as_str()
        ))
    }
}

// ─── Corrections, stops and crashes ────────────────────────────────────────

/// A person's correction arriving while the run works.
///
/// Recorded on the plan (it outranks every step), published to the task's
/// shared memory as an operator correction, and propagated: every job of the
/// run still running was dispatched before the correction existed, so each is
/// stopped and its step goes back to pending without the attempt counting.
/// Returns the jobs stopped.
pub fn record_correction(deps: &Arc<RuntimeDeps>, run_id: &str, text: &str, user_id: &str) -> Vec<String> {
    record_correction_in(&deps.events, &deps.jobs, deps.memory_graph.as_deref(), run_id, text, user_id)
}

/// [`record_correction`], over the parts a command holds.
pub fn record_correction_in(
    events: &TaskEventLog,
    board: &JobBoard,
    graph: Option<&crate::knowledge::graph::runtime_store::MemoryGraph>,
    run_id: &str,
    text: &str,
    user_id: &str,
) -> Vec<String> {
    let at = now();
    let mut stopped = Vec::new();
    let written = update_plan_in(events, board, run_id, |plan| {
        Some(task_plan::correction_recorded(plan, text, user_id, &at).0)
    });
    let Some(plan) = written else {
        return stopped;
    };
    for step in &plan.steps {
        if let Some(job_id) = &step.job_id {
            if board.cancel(job_id, true) {
                stopped.push(job_id.clone());
            }
        }
    }
    if let (Some(graph), Some(correction)) = (graph, plan.corrections.last()) {
        let item = memory_item(
            run_id,
            &format!("mi-correction-{run_id}-{}", correction.id),
            MemoryKind::Correction,
            format!("Correction {} from a person: {}", correction.id, correction.text),
            Provenance::Operator { user_id: user_id.to_string() },
            user_id,
            &at,
        );
        if let Err(error) = graph.commit(item, None, &[]) {
            log::warn!(
                "[orchestrator] run {run_id}: the correction was not published to memory: {}",
                error.explain()
            );
        }
    }
    stopped
}

/// Stops every job of a run that has ended or been stopped.
pub fn stop_run(deps: &Arc<RuntimeDeps>, run_id: &str) -> usize {
    deps.jobs.cancel_run(run_id)
}

/// At start, before any job of this process exists: every job a previous
/// process left queued or running is marked interrupted, and its step settled
/// -- failed and reopenable for a reader, blocked for a person for a writer,
/// whose effects are unknown. Returns the job ids.
pub fn recover_interrupted_jobs(events: &TaskEventLog) -> Vec<String> {
    let at = now();
    let board = JobBoard::default();
    let interrupted = match events.interrupt_live_jobs(&at) {
        Ok(jobs) => jobs,
        Err(error) => {
            log::warn!("[orchestrator] interrupted jobs could not be read: {error}");
            return Vec::new();
        }
    };
    for job in &interrupted {
        let writer = job.mode == DelegationMode::Writer.as_str();
        update_plan_in(events, &board, &job.run_id, |plan| {
            task_plan::job_interrupted(plan, &job.step_id, &job.job_id, writer, &at)
        });
    }
    interrupted.into_iter().map(|job| job.job_id).collect()
}

// ─── Review ────────────────────────────────────────────────────────────────

/// `task.request_review`: backend acceptance checks over a step's outputs, then
/// an independent reviewer, recorded on the plan as a review receipt.
pub(super) async fn request_review(
    deps: &Arc<RuntimeDeps>,
    call: &CallParams,
    session: &Session,
    tool_call: &ToolCall,
) -> Result<String, String> {
    let step_id = tool_call.text("step").ok_or("`step` names the step to review.")?.trim().to_string();
    let plan = deps.events.latest_plan(&call.run_id)?.ok_or("This run has no plan, so there is no step to review.")?;
    let step = plan.step(&step_id).ok_or_else(|| format!("The plan has no step {step_id:?}."))?.clone();

    // What is under review: the step's own outputs, or for a review step the
    // outputs of what it depends on.
    let targets: Vec<&task_plan::Step> = if step.kind == StepKind::Review {
        let unmet: Vec<String> = step
            .depends_on
            .iter()
            .filter_map(|id| plan.step(id))
            .filter(|dep| !matches!(dep.status, StepStatus::Completed | StepStatus::AwaitingReview))
            .map(|dep| format!("{} ({})", dep.id, dep.status.as_str()))
            .collect();
        if !unmet.is_empty() {
            return Err(format!("Review step {step_id} waits on {}. Nothing was reviewed.", unmet.join(", ")));
        }
        step.depends_on.iter().filter_map(|id| plan.step(id)).collect()
    } else {
        if step.status != StepStatus::AwaitingReview {
            return Err(format!(
                "Step {step_id} is {}; a step is reviewed once its work is done by receipt and it \
                 waits for review. Nothing was reviewed.",
                step.status.as_str()
            ));
        }
        vec![&step]
    };
    let mut artifacts: Vec<(String, u32)> = targets
        .iter()
        .flat_map(|target| target.outputs.iter())
        .filter_map(|output| match output {
            StepRef::Artifact { artifact_id, version } => Some((artifact_id.clone(), *version)),
            _ => None,
        })
        .collect();
    // Versions the call names -- the coordinator's own deliverable, for a
    // direct step -- each checked to exist for this owner before anything runs.
    if let Some(named) = tool_call.arguments.get("artifacts") {
        let named = named.as_array().ok_or("`artifacts` is a list of art-…@N references. Nothing was reviewed.")?;
        for item in named {
            let raw = item.as_str().unwrap_or_default().trim();
            let Some((artifact_id, version)) = raw
                .rsplit_once('@')
                .and_then(|(id, version)| Some((id.to_string(), version.parse::<u32>().ok()?)))
            else {
                return Err(format!("{raw:?} is not an exact art-…@N reference. Nothing was reviewed."));
            };
            artifact_input(deps, call, session, &artifact_id, version)
                .map_err(|_| format!("The artifact {raw} does not exist or is not readable here. Nothing was reviewed."))?;
            if !artifacts.contains(&(artifact_id.clone(), version)) {
                artifacts.push((artifact_id, version));
            }
        }
    }
    let memory: Vec<(String, u64)> = targets
        .iter()
        .flat_map(|target| target.outputs.iter())
        .filter_map(|output| match output {
            StepRef::Memory { item_id, revision } => Some((item_id.clone(), *revision)),
            _ => None,
        })
        .collect();

    let mut lines: Vec<String> = Vec::new();
    let mut passed = true;

    // 1. Backend acceptance: the P04 ladder for every artifact, run now.
    for (artifact_id, version) in &artifacts {
        let args = json!({ "artifact": format!("{artifact_id}@{version}") });
        let check_call = ToolCall::new(ToolName::ArtifactValidate.as_str().to_string(), args);
        let (deps_c, call_c, session_c) = (Arc::clone(deps), call.clone(), session.clone());
        let outcome = tokio::task::spawn_blocking(move || {
            super::artifact_tools::validate_version(&deps_c, &call_c, &session_c, &check_call)
        })
        .await
        .unwrap_or_else(|error| Err(format!("the check stopped: {error}")));
        let accepted = deps
            .conversation_artifacts
            .latest_validation(
                &session.user.id,
                &crate::artifacts::conversation_store::ArtifactRef { artifact_id: artifact_id.clone(), version: *version },
            )
            .ok()
            .flatten()
            .is_some_and(|validation| validation.accepted);
        passed &= accepted;
        lines.push(format!(
            "backend: {artifact_id}@{version} {}",
            match (&outcome, accepted) {
                (_, true) => "accepted by artifact validation".to_string(),
                (Ok(_), false) => "not accepted by artifact validation (see artifact.validate for the rungs)".to_string(),
                (Err(reason), false) => format!("could not be validated: {reason}"),
            }
        ));
    }
    // 1b. Backend acceptance for memory outputs: still current, not stale.
    if let Some(problem) = memory_not_ready(deps, session, &memory) {
        passed = false;
        lines.push(format!("backend: {problem}"));
    } else if !memory.is_empty() {
        lines.push(format!("backend: {} published item(s) still current", memory.len()));
    }
    if artifacts.is_empty() && memory.is_empty() {
        passed = false;
        lines.push("backend: the step produced nothing to review".into());
    }

    // 2. Independent review, by a role that did not produce the work.
    let mut reviewer_line = String::from("independent: not applicable (no artifact to review)");
    if !artifacts.is_empty() {
        let reviewer = "artifact-reviewer";
        let performable = deps
            .subagents
            .resolve(reviewer)
            .map(|resolved| deps.subagents.has_worker(&resolved.capability) || deps.subagents.has_worker(&resolved.profile.name))
            .unwrap_or(false);
        let produced_by_reviewer = targets.iter().any(|target| target.role.as_deref() == Some(reviewer));
        if produced_by_reviewer {
            passed = false;
            reviewer_line = "independent: refused — the reviewer role produced this work".into();
        } else if !performable {
            passed = false;
            reviewer_line = "independent: blocked — this build has no artifact-reviewer worker".into();
        } else {
            let (parent_model, attempt_id) = parent_model_and_attempt(deps, &call.run_id);
            let outcome = match (parent_model, deps.registry.as_deref(), deps.root_for(&call.run_id)) {
                (Some(parent_model), Some(registry), Some(workspace)) => {
                    let resolved = deps.subagents.resolve(reviewer).map_err(|r| r.explain())?;
                    let role = resolved.profile.model_role;
                    let candidates: Vec<_> = registry.all().iter().filter(|e| e.enabled && e.serves(role)).map(|e| (e, None)).collect();
                    let decision = crate::subagents::certification::choose(role, &parent_model, &candidates);
                    let permitted = super::registered_plan_tools(deps, &call.run_id).unwrap_or_default();
                    let inherited = crate::subagents::InheritedPolicy::of_run(session, crate::policy::Classification::Internal, &workspace, &permitted);
                    let inputs: Vec<InputRef> = artifacts
                        .iter()
                        .filter_map(|(id, version)| artifact_input(deps, call, session, id, *version).ok())
                        .collect();
                    let mut dispatch = Dispatch::for_task(reviewer, call.run_id.clone())
                        .delivering(tool_call.text("criteria").unwrap_or("each artifact is sound and says what the task asked for"))
                        .scoped_to(format!("review {step_id} v{}", plan.version))
                        .within(Duration::from_secs(REVIEWER_DEADLINE_SECONDS));
                    if let Some(attempt_id) = attempt_id {
                        dispatch = dispatch.in_attempt(attempt_id);
                    }
                    deps.subagents
                        .spawn(reviewer, &inherited, &format!("Review the outputs of step {step_id}: {}", step.title), inputs, decision, &dispatch)
                        .await
                        .map_err(|refusal| refusal.explain())
                }
                _ => Err("the run has no model, registry or workspace to route a reviewer with".into()),
            };
            match outcome {
                Ok(spawned) => {
                    let child = spawned.result();
                    let clean = child.status.is_complete()
                        && child.uncertainty.is_empty()
                        && child.findings.len() >= artifacts.len()
                        && !child.receipts.is_empty();
                    passed &= clean;
                    reviewer_line = format!(
                        "independent: {} ({}; {} uncertainty note(s))",
                        if clean { "passed" } else { "did not pass" },
                        child.describe(),
                        child.uncertainty.len()
                    );
                    for note in child.uncertainty.iter().take(4) {
                        reviewer_line.push_str(&format!("\n  - {note}"));
                    }
                }
                Err(reason) => {
                    passed = false;
                    reviewer_line = format!("independent: could not run — {reason}");
                }
            }
        }
    }
    lines.push(reviewer_line);

    let review_id = format!("rev-{}", &uuid::Uuid::new_v4().simple().to_string()[..12]);
    let summary = lines.join("; ");
    let at = now();
    // A review step's verdict settles the steps it reviewed as well: a target
    // waiting for review completes on a pass, and becomes partial on a failure,
    // under the same review receipt. One plan version for all of it.
    let reviewed: Vec<String> = if step.kind == StepKind::Review {
        targets.iter().map(|target| target.id.clone()).collect()
    } else {
        Vec::new()
    };
    let recorded = update_plan(deps, &call.run_id, |plan| {
        let mut next = task_plan::review_settled(plan, &step_id, &review_id, passed, &summary, None, &at)?;
        for target in &reviewed {
            if next.step(target).is_some_and(|held| held.status == StepStatus::AwaitingReview) {
                next = task_plan::review_settled(&next, target, &review_id, passed, &summary, None, &at)?;
            }
        }
        next.version = plan.version + 1;
        Some(next)
    });
    let mut out = format!(
        "Review {review_id} of step {step_id}: {}.\n{}\n",
        if passed { "passed" } else { "did not pass" },
        lines.iter().map(|line| format!("- {line}")).collect::<Vec<_>>().join("\n")
    );
    if let Some(plan) = recorded {
        out.push('\n');
        out.push_str(&task_plan::project(&plan));
    }
    Ok(out)
}

// ─── Shared memory ─────────────────────────────────────────────────────────

fn memory_item(
    run_id: &str,
    item_id: &str,
    kind: MemoryKind,
    content: String,
    provenance: Provenance,
    owner: &str,
    at: &str,
) -> MemoryItem {
    let classification = crate::policy::Classification::Internal;
    MemoryItem {
        item_id: item_id.to_string(),
        revision: 1,
        kind,
        agent_id: "orchestrator".into(),
        scope: MemoryScope::Task { task_id: run_id.to_string() },
        classification,
        acl: crate::agent_runtime::memory::Acl {
            cleared_roles: classification.cleared_roles().to_vec(),
            project_id: None,
            owner: Some(owner.to_string()),
        },
        creator_model_id: None,
        creator_run_id: Some(run_id.to_string()),
        provenance,
        content,
        sources: Vec::new(),
        artifacts: Vec::new(),
        confidence: None,
        status: ItemStatus::Proposed,
        valid_from: at.to_string(),
        valid_until: None,
        supersedes: None,
        conflicts_with: Vec::new(),
        causal_parents: Vec::new(),
        idempotency_key: None,
        created_at: at.to_string(),
        updated_at: at.to_string(),
        basis: None,
        depends_on: Vec::new(),
        revoked_readers: Vec::new(),
        authority: Authority::Graph,
    }
}

/// Publishes the run's goal, constraints, plan, decisions and open questions
/// to the task's shared memory, on the receipt of the orchestrator call that
/// just returned them.
///
/// The receipt is the provenance: the item is what that call returned, and the
/// event log holds its output hash. So the plan reaches the next round's
/// compiled context (`context.refresh`) as a mandatory block, at the version
/// the coordinator was last shown -- no fresher and no staler.
pub(super) fn publish_plan(
    deps: &Arc<RuntimeDeps>,
    session: &Session,
    run_id: &str,
    tool: ToolName,
    receipt: Option<(i64, String)>,
) {
    let (Some(graph), Some((event_seq, output_sha256))) = (deps.memory_graph.as_ref(), receipt) else {
        return;
    };
    let Ok(Some(plan)) = deps.events.latest_plan(run_id) else {
        return;
    };
    let at = now();
    let provenance = Provenance::ToolReceipt {
        run_id: run_id.to_string(),
        tool: tool.as_str().to_string(),
        event_seq,
        output_sha256: Some(output_sha256),
    };
    let owner = session.user.id.as_str();
    let mut items: Vec<(String, MemoryKind, String)> = vec![
        (format!("mi-goal-{run_id}"), MemoryKind::Goal, format!("Goal: {}", plan.goal)),
        (format!("mi-plan-{run_id}"), MemoryKind::Plan, task_plan::project(&plan)),
    ];
    for (index, constraint) in plan.constraints.iter().enumerate() {
        items.push((format!("mi-constraint-{run_id}-{}", index + 1), MemoryKind::Constraint, format!("Constraint: {constraint}")));
    }
    for decision in &plan.decisions {
        items.push((
            format!("mi-decision-{run_id}-{}", decision.id),
            MemoryKind::Decision,
            format!(
                "Decision {}: {} (rests on {})",
                decision.id,
                decision.text,
                if decision.evidence.is_empty() { "nothing named".to_string() } else { decision.evidence.join(", ") }
            ),
        ));
    }
    for question in &plan.questions {
        let item_id = format!("mi-question-{run_id}-{}", question.id);
        if question.answer.is_some() {
            let _ = graph.tombstone(&item_id, &at);
            continue;
        }
        items.push((item_id, MemoryKind::OpenQuestion, format!("Open question {}: {}", question.id, question.text)));
    }
    for (item_id, kind, content) in items {
        let held = graph.versions_of(session, &item_id, None).unwrap_or_default();
        if held.last().is_some_and(|version| version.item.content == content) {
            continue;
        }
        let mut item = memory_item(run_id, &item_id, kind, content, provenance.clone(), owner, &at);
        item.idempotency_key = Some(format!("{item_id}@plan-v{}", plan.version));
        let expected = held.last().map(|version| version.revision);
        if let Err(error) = graph.commit(item, expected, &[]) {
            log::warn!("[orchestrator] run {run_id}: {item_id} was not published: {}", error.explain());
        }
    }
}

#[cfg(test)]
#[path = "delegation_tests.rs"]
mod tests;
