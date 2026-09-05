//! Supervising the agent runtime, and answering it.
//!
//! The agent loop lives in a Node child process (`agent-runtime/`), built from
//! OpenClaw's `agent-core`. This module owns that process and serves the two
//! questions it asks: *may this tool call happen*, and *please perform it*.
//!
//! ## Why the loop is over there and the decisions are here
//!
//! The loop needs streaming, compaction, steering and abort recovery, which
//! OpenClaw already has and this project would otherwise have to grow. The
//! decisions need the user's permissions, the workspace boundary, the
//! sovereignty invariant and the audit record, which live here and should not
//! be copied into a second process to be re-derived.
//!
//! So the split is not by convenience but by authority: **the runtime may
//! request; only this side decides.** Nothing in the child process can widen
//! what a run is permitted to do, because it does not hold the information that
//! would let it.
//!
//! ## The two questions
//!
//! - `tool.authorize` puts a call through [`ToolGateway`] and, on an allow,
//!   returns a single-use grant bound to that exact call (see [`grants`]).
//! - `tool.execute` redeems the grant, *re-derives the verdict independently*,
//!   and only then runs the tool through [`LocalToolRunner`] — the same runner
//!   the retired Rust executor used, unchanged.
//!
//! Checking twice is deliberate. The grant covers a compromised runtime; the
//! re-check covers a bug in the grant. Neither alone is worth the claim being
//! made.

pub mod approval;
pub mod artifacts;
pub mod audit_health;
pub mod completion;
pub mod context_api;
pub mod conversations;
pub mod events;
pub mod grants;
pub mod memory;
pub mod memory_api;
pub mod model_transition;
pub mod outcome;
pub mod planning;
pub mod protocol;
pub mod recording;
pub mod resume;
pub mod retrieval;
pub mod stages;
pub mod task_driver;
pub mod tasks;
pub mod tool_policy;
pub mod workspace;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::{mpsc, oneshot};

#[cfg(target_os = "windows")]
#[cfg(target_os = "windows")]
const CREATE_NO_WINDOW: u32 = 0x08000000;

use crate::identity::Session;
use crate::knowledge::KnowledgeIndex;
use crate::orchestrator::approvals::ApprovalQueue;
use crate::orchestrator::calculation::CalculationRecord;
use crate::orchestrator::gateway::{GatewayVerdict, TaskContext, ToolGateway};
use crate::orchestrator::plan::Continuation;
use crate::orchestrator::runner::LocalToolRunner;
use crate::subagents::InheritedPolicy;
use crate::orchestrator::executor::ToolRunner;
use crate::orchestrator::tools::{ToolCall, ToolName};
use crate::policy::ApprovalState;
use grants::GrantLedger;
use protocol::{code, Frame, Outgoing, WireError};
use recording::{refused, remember_loop_event, remember_outcome, remember_refusal};

/// Event name the UI listens on for the loop's own progress.
///
/// One channel for every run; the payload carries the run id so a listener can
/// filter. Best-effort by design — a dropped event costs a progress line —
/// which is exactly why it is not the channel a client reconciles against.
pub const AGENT_EVENT: &str = "agent://event";

/// Event name the UI listens on for the durable history.
///
/// Every message here corresponds to a row that is on disk, and carries the
/// sequence number of that row. A client that sees a gap in those numbers knows
/// it missed something and asks for a snapshot; a client watching only
/// [`AGENT_EVENT`] cannot tell a quiet run from a lost message.
pub const AGENT_DURABLE_EVENT: &str = "agent://durable";

/// Everything a handler needs that is not on the wire.
///
/// Held rather than reached for so the handlers stay testable without a Tauri
/// app: the tests at the bottom build one of these directly.
pub struct RuntimeDeps {
    pub index: Arc<KnowledgeIndex>,
    pub session: Arc<std::sync::RwLock<Option<Session>>>,
    /// Where a run's files live, keyed by run id.
    ///
    /// Per run rather than per process: a shared scratch directory would let one
    /// task read what an unrelated task left behind, and the audit record would
    /// show a legitimate read of a permitted path. See [`workspace`].
    pub workspaces: Arc<Mutex<HashMap<String, workspace::Workspace>>>,
    /// Where proposed actions go to be seen by a person.
    pub approvals: Arc<ApprovalQueue>,
    /// Calculations a run has performed, in order, keyed by run id.
    ///
    /// Accumulated rather than recomputed because `create_xlsx` writes the
    /// run's whole working — PS 26117 asks for *"calculations with steps
    /// shown"*, and a workbook rebuilt from the model's recollection of what it
    /// computed would be exactly the thing the calculation engine exists to
    /// avoid.
    pub calculations: Arc<Mutex<HashMap<String, Vec<CalculationRecord>>>>,
    /// Passages a run has retrieved, in the order its citation markers refer to.
    ///
    /// Kept for the same reason as the calculations: the verifier resolves each
    /// `[En]` in the final answer against what was actually retrieved, and it
    /// cannot do that against passages nobody kept. See [`retrieval`].
    pub passages: retrieval::RunPassages,
    /// Files a run has produced, so each can be re-opened and checked when the
    /// run ends rather than taken on the model's word. See [`artifacts`].
    pub produced: artifacts::RunArtifacts,
    /// Every tool call a run has made, in order.
    ///
    /// Kept here rather than reconstructed from the event stream, which is
    /// best-effort: a dropped event costs a progress line, and it should not
    /// also cost a line in the permanent record of what the run did.
    pub calls: Arc<Mutex<HashMap<String, Vec<tasks::ToolCallRecord>>>>,
    /// The plan each run is being held to, keyed by run id.
    ///
    /// The budget inside is fixed by [`planning`] before the model is told
    /// anything, and nothing on the runtime's side of the wire can reach it.
    pub plans: Arc<Mutex<HashMap<String, crate::orchestrator::plan::PlanRun>>>,
    /// The durable record of what each run has done, in order.
    ///
    /// Written *as* the run happens, unlike the task record in [`tasks`], which
    /// is written once at the end. The difference is what a window that
    /// remounted mid-run, or a process that starts after one died mid-run, has
    /// to read: after the fact there is nothing to reconstruct from, so the
    /// reconstruction has to be written on the way past. See [`events`].
    pub events: Arc<events::TaskEventLog>,
    /// The skills installed on this machine.
    ///
    /// Held so `capability.search` can answer without a round trip through the
    /// UI. A skill is guidance, not permission — see [`crate::skills`] — so
    /// this is a source of *descriptions*, and nothing reached through it can
    /// widen what a run may do.
    pub skills: Arc<crate::skills::SkillRegistry>,
    /// The deterministic checks this deployment runs at its lifecycle points.
    ///
    /// Built from code at start-up and never from anything a prompt, a skill or
    /// a retrieved document can reach — see [`crate::hooks`]. Held here so the
    /// handlers can consult them without a global, which is also what keeps a
    /// test able to install its own and drive the refusal path.
    pub hooks: Arc<crate::hooks::HookRegistry>,
    /// What this machine remembers, and for whom.
    ///
    /// Scoped and access-controlled in [`memory`]; reachable by a model only
    /// through the two methods in [`memory_api`], which fill in the identity,
    /// project, classification and approval from this side. Held here rather
    /// than reached for so the handlers stay drivable with no Tauri app.
    pub memory: memory::SharedMemory,
    /// The parts of a checkpoint that are fixed for the life of an attempt.
    ///
    /// Held so the deep loop can take a checkpoint after every tool result
    /// without re-deriving the policy, plan and workspace hashes it would need
    /// to do that — those are established once, when the run starts, from state
    /// this side of the wire does not otherwise carry.
    ///
    /// A run with no seed is a run started before this existed, or one whose
    /// start did not complete. Both mean no checkpoint is taken, which is the
    /// honest answer: a checkpoint assembled from defaults would claim a world
    /// nobody observed.
    pub checkpoints: Arc<Mutex<HashMap<String, resume::CheckpointSeed>>>,
    /// Where run events go.
    ///
    /// The loop publishes its own events over the wire; these are the ones this
    /// side decides — a step spent, a plan exhausted. They travel the same
    /// channel because an operator watching a run should see one sequence of
    /// what happened, not two interleaved by luck.
    ///
    /// Injected rather than reached for, so this module keeps no dependency on
    /// Tauri. That is the same reason [`AgentRuntime::spawn`] takes an emitter,
    /// and it is what lets the tests drive all of this with no app running.
    pub emit: Arc<dyn Fn(Value) + Send + Sync>,
    /// Where durable events go, once they are on disk.
    ///
    /// Separate from [`Self::emit`] because the two make different promises. A
    /// message on that channel is a progress line that may be dropped; a
    /// message on this one names a row that exists, and carries the sequence
    /// number a client reconciles against.
    pub emit_durable: Arc<dyn Fn(Value) + Send + Sync>,
    /// The workers a run may delegate a read-only sub-task to.
    ///
    /// `agent.delegate_readonly` is the model-facing surface, and it used to
    /// answer "Subagents are not available on this machine" for every call:
    /// the runner was constructed with `subagents: None` on the agent path, so
    /// the manager the application had built was never handed to it.
    pub subagents: Arc<crate::subagents::SubagentManager>,
    /// The page-region and table half of the knowledge index.
    ///
    /// Backs `knowledge.multimodal_retrieve`. Never constructed outside tests
    /// before, so that tool ran with nothing behind it.
    ///
    /// Wiring this fixed half the problem. The other half outlived it: no plan
    /// permitted the tool, so once the index existed the catalogue still never
    /// offered it and the gateway still refused every call. See
    /// `planning::derive`, which now permits it wherever it permits a text
    /// search, and the reachability test that keeps it there.
    pub multimodal: Arc<crate::knowledge::MultimodalIndex>,
    /// Whether this installation can still record what it does.
    ///
    /// Consulted before a tool with a side effect is authorised. A read is
    /// still allowed when the record is broken — nothing it does needs writing
    /// down beyond the event that names it — but a write to disk, a produced
    /// file or a sandboxed command must not happen where there is nowhere to
    /// record that it happened. That is the whole of ARJUN's claim: an effect
    /// with no provenance is worse than no effect.
    ///
    /// See [`audit_health`].
    pub audit_health: Arc<audit_health::AuditHealth>,
}

impl RuntimeDeps {
    /// The directories a given run may touch. None until the run has one, which
    /// makes every path-taking tool refuse rather than reach somewhere shared.
    fn roots_for(&self, run_id: &str) -> Vec<PathBuf> {
        self.workspaces
            .lock()
            .ok()
            .and_then(|table| table.get(run_id).map(workspace::Workspace::roots))
            .unwrap_or_default()
    }

    /// The run's workspace, for naming a produced file relative to it.
    fn root_for(&self, run_id: &str) -> Option<PathBuf> {
        self.roots_for(run_id).into_iter().next()
    }

    /// Publishes one event about a run, in the shape the UI already listens for.
    fn publish(&self, run_id: &str, event: Value) {
        (self.emit)(json!({ "runId": run_id, "event": event }));
    }

    fn session(&self) -> Result<Session, WireError> {
        self.session
            .read()
            .ok()
            .and_then(|guard| guard.clone())
            .ok_or_else(|| {
                WireError::new(
                    code::REFUSED,
                    "No one is signed in, so no tool call can be attributed to a person. Sign in and start the task again.",
                )
            })
    }

    /// Whether confidential work is permitted right now.
    ///
    /// Read from the broker at the moment of the call rather than captured when
    /// the run started: switching the workbench into provisioning mode mid-run
    /// must stop the next tool call, not just the next run.
    fn confidential_work_permitted(&self) -> bool {
        crate::sovereignty::global_broker()
            .guard_confidential("agent tool call")
            .is_ok()
    }
}

/// A live agent runtime process.
pub struct AgentRuntime {
    outbound: mpsc::UnboundedSender<String>,
    pending: Arc<Mutex<HashMap<String, oneshot::Sender<Result<Value, WireError>>>>>,
    next_id: AtomicU64,
    child: Mutex<Option<Child>>,
}

#[derive(Debug, thiserror::Error)]
pub enum RuntimeError {
    #[error("the agent runtime bundle is missing at {0}. Run `npm run build` in agent-runtime/.")]
    BundleMissing(PathBuf),
    #[error("the agent runtime could not be started: {0}")]
    Spawn(#[source] std::io::Error),
    #[error("the agent runtime went away before answering")]
    Closed,
    #[error("{}", .0.message)]
    Remote(WireError),
}

/// Where the bundle lives when the caller has nothing better.
///
/// `ARJUN_AGENT_RUNTIME` wins, which is what the tests use. Otherwise the
/// development layout, so a checkout runs with no configuration. A packaged
/// build resolves its own resource directory and passes that to [`spawn`] --
/// this module does not depend on Tauri, which is what lets the tests drive a
/// real child process with no application running.
pub fn default_bundle_path() -> PathBuf {
    if let Ok(explicit) = std::env::var("ARJUN_AGENT_RUNTIME") {
        return PathBuf::from(explicit);
    }
    // `CARGO_MANIFEST_DIR` is src-tauri/; the runtime is its sibling.
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(|root| root.join("agent-runtime/dist/arjun-agent-runtime.mjs"))
        .unwrap_or_default()
}

impl AgentRuntime {
    /// Starts the child and the three tasks that keep it fed.
    ///
    /// `emit` receives every `run.event` the runtime publishes. Injected rather
    /// than taking an `AppHandle` so this module does not depend on Tauri, which
    /// is what lets the tests drive a real child process with no app running.
    pub fn spawn(
        deps: Arc<RuntimeDeps>,
        emit: Arc<dyn Fn(Value) + Send + Sync>,
        bundle: PathBuf,
    ) -> Result<Arc<Self>, RuntimeError> {
        if !bundle.exists() {
            return Err(RuntimeError::BundleMissing(bundle));
        }

        // `ARJUN_NODE` when an offline deployment pack laid one down, the bare
        // name otherwise. See `crate::deployment`, which like this module has
        // no Tauri dependency, so the tests still drive a real child process.
        let mut child = Command::new(crate::deployment::program("node"));
        child
            .arg(&bundle)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            // The runtime reaches only the loopback inference endpoint the
            // router chose. It has no use for inherited proxy configuration, and
            // an inherited `HTTP_PROXY` would be a way out of the machine that
            // nothing in this codebase put there.
            .env_remove("HTTP_PROXY")
            .env_remove("HTTPS_PROXY")
            .env_remove("http_proxy")
            .env_remove("https_proxy")
            .env_remove("ALL_PROXY")
            .env_remove("NPM_CONFIG_PROXY")
            .kill_on_drop(true);
        // Without `CREATE_NO_WINDOW`, Windows opens a console window for
        // the spawned `node.exe` every time the user sends a chat message,
        // because the Tauri release build is `windows_subsystem = "windows"`
        // and the OS does not auto-suppress console windows for the children
        // it spawns. The same flag is set in the document and memory
        // sidecars; this is the matching setting for the agent runtime.
        #[cfg(target_os = "windows")]
        {
            child.creation_flags(CREATE_NO_WINDOW);
        }
        let mut child = child.spawn().map_err(RuntimeError::Spawn)?;

        let stdin = child.stdin.take().expect("stdin was piped");
        let stdout = child.stdout.take().expect("stdout was piped");
        let stderr = child.stderr.take().expect("stderr was piped");

        let (outbound, mut outbox) = mpsc::unbounded_channel::<String>();
        let pending: Arc<Mutex<HashMap<String, oneshot::Sender<Result<Value, WireError>>>>> =
            Arc::new(Mutex::new(HashMap::new()));

        let runtime = Arc::new(Self {
            outbound: outbound.clone(),
            pending: pending.clone(),
            next_id: AtomicU64::new(1),
            child: Mutex::new(Some(child)),
        });

        // Writer: one task owns stdin, so writes cannot interleave mid-line.
        tokio::spawn(async move {
            let mut stdin = stdin;
            while let Some(line) = outbox.recv().await {
                if stdin.write_all(line.as_bytes()).await.is_err() {
                    break;
                }
                if stdin.flush().await.is_err() {
                    break;
                }
            }
        });

        // Diagnostics. The runtime writes every log line here precisely so that
        // stdout stays parseable; forwarding them keeps that decision from
        // costing us the ability to debug.
        tokio::spawn(async move {
            let mut lines = BufReader::new(stderr).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                log::info!("[agent-runtime] {line}");
            }
        });

        // Reader: the only place inbound frames are interpreted.
        let reader_deps = deps.clone();
        let reader_outbound = outbound.clone();
        tokio::spawn(async move {
            let mut lines = BufReader::new(stdout).lines();
            // Approval waits must not monopolize the reader: it still has to
            // observe EOF, cancellations and replies while an RPC is suspended.
            let mut handlers = tokio::task::JoinSet::new();
            loop {
                let line = tokio::select! {
                    line = lines.next_line() => match line { Ok(Some(line)) => line, _ => break },
                    _ = handlers.join_next(), if !handlers.is_empty() => continue,
                };
                if line.trim().is_empty() {
                    continue;
                }
                let frame = match Frame::parse(&line) {
                    Ok(frame) => frame,
                    Err(error) => {
                        // Fatal for the channel: past a frame we cannot read,
                        // neither end knows what the other said.
                        log::error!("[agent-runtime] unparseable frame, closing channel: {error}");
                        break;
                    }
                };
                if let Frame::Request { id, method, params } = frame {
                    if handlers.len() >= 64 {
                        let _ = reader_outbound.send(Outgoing::Error { id,
                            error: WireError::new(code::REFUSED, "Too many in-flight core requests") }.encode());
                        continue;
                    }
                    let deps = reader_deps.clone();
                    let outbound = reader_outbound.clone();
                    let pending = pending.clone();
                    let emit = emit.clone();
                    handlers.spawn(async move {
                        dispatch(Frame::Request { id, method, params }, &deps, &outbound, &pending, &emit).await;
                    });
                } else {
                    dispatch(frame, &reader_deps, &reader_outbound, &pending, &emit).await;
                }
            }
            // Drop suspended authorizations with the dead worker. If a dispatched
            // effect is interrupted, its durable intent remains unsettled for
            // reconciliation; cancellation never manufactures a successful receipt.
            handlers.abort_all();
            while handlers.join_next().await.is_some() {}
            // Stream ended. Fail every caller still waiting rather than leaving
            // them to hang on a process that is gone.
            let waiting: Vec<_> = pending.lock().map(|mut p| p.drain().collect()).unwrap_or_default();
            for (_, sender) in waiting {
                let _ = sender.send(Err(WireError::new(
                    code::INTERNAL,
                    "the agent runtime stopped",
                )));
            }
        });

        Ok(runtime)
    }

    /// Sends a request and waits for its reply.
    pub async fn request(&self, method: &str, params: Value) -> Result<Value, RuntimeError> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed).to_string();
        let (tx, rx) = oneshot::channel();
        self.pending
            .lock()
            .map_err(|_| RuntimeError::Closed)?
            .insert(id.clone(), tx);

        let frame = Outgoing::Request {
            id: id.clone(),
            method: method.to_string(),
            params,
        };
        if self.outbound.send(frame.encode()).is_err() {
            self.pending.lock().ok().and_then(|mut p| p.remove(&id));
            return Err(RuntimeError::Closed);
        }

        match rx.await {
            Ok(Ok(value)) => Ok(value),
            Ok(Err(error)) => Err(RuntimeError::Remote(error)),
            Err(_) => Err(RuntimeError::Closed),
        }
    }

    /// Stops the child. Idempotent, so shutdown paths can call it freely.
    pub async fn shutdown(&self) {
        let child = self.child.lock().ok().and_then(|mut slot| slot.take());
        if let Some(mut child) = child {
            let _ = child.kill().await;
        }
    }
}

/// Routes one inbound frame.
async fn dispatch(
    frame: Frame,
    deps: &Arc<RuntimeDeps>,
    outbound: &mpsc::UnboundedSender<String>,
    pending: &Arc<Mutex<HashMap<String, oneshot::Sender<Result<Value, WireError>>>>>,
    emit: &Arc<dyn Fn(Value) + Send + Sync>,
) {
    match frame {
        Frame::Request { id, method, params } => {
            let reply = match handle(&method, params, deps).await {
                Ok(result) => Outgoing::Result { id, result },
                Err(error) => Outgoing::Error { id, error },
            };
            let _ = outbound.send(reply.encode());
        }
        Frame::Result { id, result } => {
            if let Some(sender) = pending.lock().ok().and_then(|mut p| p.remove(&id)) {
                let _ = sender.send(Ok(result));
            }
        }
        Frame::Error { id, error } => {
            if let Some(sender) = pending.lock().ok().and_then(|mut p| p.remove(&id)) {
                let _ = sender.send(Err(error));
            }
        }
        Frame::Notification { method, params } => {
            if method == "run.event" {
                // Two of the loop's own events are kept as well as shown. The
                // rest are progress that a recovered trace can do without; a
                // turn count and a compaction are not. The compaction in
                // particular is a caveat on everything the run says afterwards,
                // and a trace that lost it would overstate its own grounding.
                //
                // `message_start` / `message_update` / `message_end` are
                // *streaming* events: not durable, but forwarded on the live
                // channel so a chat surface can show tokens as they arrive. A
                // half-streamed sentence that arrives on remount as text is
                // misleading; the durable record is the final `Message.content`
                // row, not these deltas.
                let event_type = params
                    .get("event")
                    .and_then(|v| v.get("type"))
                    .and_then(Value::as_str)
                    .unwrap_or("");
                if matches!(
                    event_type,
                    "message_start"
                        | "message_update"
                        | "message_end"
                        | "model_thinking"
                        // `context_ledger` is a reading of where the window
                        // stands, emitted every turn so the meter a person
                        // watches describes this run rather than the last one.
                        // Live-only for the same reason as `model_thinking`: a
                        // durable row per turn would put the whole itemisation
                        // on disk dozens of times to preserve a figure the next
                        // turn supersedes — and the reading that outlives the
                        // run is already kept, once, on the task record.
                        | "context_ledger"
                ) {
                    // Live channel only. Drop on slow consumer.
                    //
                    // `model_thinking` is here for a second reason as well as
                    // the first: it says the model is reasoning and how much
                    // of it there has been, and that is a progress line, not a
                    // fact about the run worth keeping. Recording it would put
                    // a row on disk every second of every reasoning pass to
                    // preserve a number nobody reads afterwards.
                    emit(params);
                } else {
                    remember_loop_event(deps, &params);
                    emit(params);
                }
            } else {
                log::debug!("[agent-runtime] unhandled notification {method}");
            }
        }
    }
}

/// The methods this side serves.
async fn handle(
    method: &str,
    params: Value,
    deps: &Arc<RuntimeDeps>,
) -> Result<Value, WireError> {
    if matches!(method, "tool.authorize" | "tool.execute" | "tool.catalogue" | "capability.search" | "skill.load" | "memory.recall_authorized" | "memory.promote_approved") {
        context_api::validate_attempt(&params, deps, false)?;
    }
    match method {
        "context.commit" => context_api::commit(params, deps),
        "context.load" => context_api::load(params, deps),
        "tool.authorize" => authorize(params, deps).await,
        "tool.execute" => execute(params, deps).await,
        "tool.catalogue" => tool_catalogue(params, deps),
        "capability.search" => capability_search(params, deps),
        "skill.load" => skill_load(params, deps),
        // The whole of a model's reach into memory. Both fill in identity,
        // project, classification and approval on this side; neither takes them
        // from the caller. See [`memory_api`].
        "memory.recall_authorized" => memory_api::recall_authorized(params, deps),
        "memory.promote_approved" => memory_api::promote_approved(params, deps),
        other => Err(WireError::new(
            code::UNKNOWN_METHOD,
            format!("no handler for {other}"),
        )),
    }
}

/// The tools a registered plan permits, or `None` when there is no such plan.
///
/// A poisoned table reads as no plan. That is the fail-closed reading and the
/// correct one: the budget could not be consulted, so nothing may be authorised
/// against it.
fn registered_plan_tools(deps: &Arc<RuntimeDeps>, run_id: &str) -> Option<Vec<ToolName>> {
    deps.plans
        .lock()
        .ok()?
        .get(run_id)
        .map(|plan| plan.budget.permitted_tools.clone())
}

/// Whether this run has a plan registered on this side.
fn has_registered_plan(deps: &Arc<RuntimeDeps>, run_id: &str) -> bool {
    deps.plans
        .lock()
        .map(|plans| plans.contains_key(run_id))
        .unwrap_or(false)
}

/// The wire error for a call that names a run this side does not know.
///
/// Deliberately says nothing about which runs *do* exist. A caller probing ids
/// learns only that this one is not one of them.
fn no_plan_error(run_id: &str) -> WireError {
    WireError::new(
        code::REFUSED,
        format!(
            "There is no plan registered for run {run_id}, so nothing may be done under it. A              run id this side has not issued, or one whose run has already ended, has no              authority to catalogue, authorise or execute any tool."
        ),
    )
}

/// Which tools this run may be offered, as metadata rather than as schemas.
///
/// ## Why the runtime asks instead of knowing
///
/// The child process holds a static table of every tool's parameter schema, and
/// it would be simpler for it to hand the model all of them. That is what it did
/// before, and it costs twice over: the tool definitions are the second largest
/// fixed thing in the context window after the system prompt, and a model shown
/// a tool it may not use spends turns being refused by the gateway for asking.
///
/// So the eligible set is decided here, where the plan and the mode are, and the
/// runtime loads schemas only for the names that come back. The plan is fixed
/// before the model is told anything, so this cannot be widened by anything the
/// model does afterwards.
///
/// ## Why a run with no plan gets nothing
///
/// The plan is the authority for what a run may do, so a run id this side has
/// never heard of has no authority at all. It used to get the read-only set,
/// justified by the runtime's health probe — which does not ask for a
/// catalogue. `health` is its own RPC, served in `agent-runtime/src/main.ts`
/// and answered without touching a tool, so the exception protected nothing and
/// cost a great deal: any string in the `runId` field returned a working
/// catalogue of every read-only tool in the product, and a search runs against
/// the organisation's own documents.
///
/// So it fails closed. If some future probe genuinely needs tools, the answer
/// is a dedicated probe context with a plan of its own — a signed, bounded,
/// auditable thing — not a hole that any unrecognised id falls through.
fn tool_catalogue(params: Value, deps: &Arc<RuntimeDeps>) -> Result<Value, WireError> {
    use crate::orchestrator::tools::spec_for;

    let run_id = params
        .get("runId")
        .and_then(Value::as_str)
        .unwrap_or_default();

    let Some(planned) = registered_plan_tools(deps, run_id) else {
        return Err(no_plan_error(run_id));
    };

    let mode = crate::sovereignty::global_broker().mode();

    // Whether any declared subagent role can actually be performed here.
    //
    // A profile is a declaration and a worker is what performs it, and a build
    // with profiles but no workers refuses every delegation with "the role is
    // declared but this build has no worker for it". Offering the tool anyway
    // is not neutral: the model reads five role names in the tool description,
    // delegates, is refused, and has spent a turn out of a budget the plan
    // fixed before it started. On a multi-step task it does that repeatedly and
    // the run ends having done nothing but ask.
    //
    // So the tool is withheld when nothing behind it can run. A tool a model is
    // never shown is one it cannot spend a turn being refused for asking about
    // — the same reasoning the mode filter below is written on.
    let delegation_possible = deps
        .subagents
        .profiles()
        .any(|profile| deps.subagents.has_worker(&profile.name));

    let eligible: Vec<ToolName> = planned
        .into_iter()
        .filter(|tool| *tool != ToolName::AgentDelegateReadonly || delegation_possible)
    // Applied again here even though the plan was already filtered when it was
    // made. The two are not the same check: a plan is fixed at the start of a
    // run, and this is asked whenever the runtime starts a loop — including
    // after a resumption, which may be happening in a different mode from the
    // one the plan was written in.
    .filter(|tool| spec_for(*tool).network.permitted_in(mode))
    .collect();

    let tools: Vec<Value> = eligible
        .iter()
        .map(|tool| {
            let spec = spec_for(*tool);
            json!({
                "name": tool.as_str(),
                "summary": tool.describe(),
                // What decides whether the runtime may run it beside another.
                "readOnly": tool.is_read_only(),
                "approvalClass": spec.approval_class,
                "approvalNote": spec.approval_class.describe(),
                "network": spec.network,
                "networkNote": spec.network.describe(),
                "maxResponseBytes": spec.max_response_bytes,
                "timeoutSeconds": spec.timeout.as_secs(),
            })
        })
        .collect();

    Ok(json!({
        "tools": tools,
        "mode": mode.label(),
        "note": "Metadata only. Load the parameter schema for a name in this list; \
                 a name absent from it is refused by the gateway however it is called.",
    }))
}

/// Concise metadata for the skills this run could use.
///
/// Deliberately **not** a tool. It takes no grant, appears in no catalogue and
/// spends no step, because it does nothing: it reads local metadata that has
/// already been validated and filters it by what the signed-in person may see.
/// Making it a tool would put a read of a description behind the same gate as
/// writing a document, which teaches an operator that the gate is noise.
///
/// It returns cards — a name, a description, a version, a tool list — and never
/// a skill's instructions. Loading those is a separate, deliberate step. That
/// split is requirement 10, and it is what stops every skill on the machine
/// reaching every prompt.
fn capability_search(params: Value, deps: &Arc<RuntimeDeps>) -> Result<Value, WireError> {
    let session = deps.session()?;
    let run_id = params
        .get("runId")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let query = params
        .get("query")
        .and_then(Value::as_str)
        .unwrap_or_default();

    // The run's own tool list, so a card can be read against what this task can
    // actually do.
    //
    // A run with no plan is refused rather than answered with an empty permit
    // list. Skill *descriptions* are not secret, but they name what this
    // deployment can do and who it is for, and a surface that answers to any id
    // at all is a surface worth probing. The same rule the catalogue follows.
    let Some(permits) = registered_plan_tools(deps, run_id) else {
        return Err(no_plan_error(run_id));
    };

    let context = crate::skills::SkillContext {
        session: &session,
        // Read at the moment of the call rather than captured when the run
        // started: switching the workbench into provisioning mode must change
        // which skills are offered, not only which ones start.
        mode: crate::sovereignty::global_broker().mode(),
        run_permits: &permits,
    };

    let found = deps.skills.search(query, &context);
    Ok(json!({
        "skills": found,
        // Said explicitly so a caller does not have to infer it from the shape.
        "note": "Metadata only. Ask for a skill by name to read its instructions.",
    }))
}

/// Loads one skill's instructions into a run, by name.
///
/// ## Why this exists
///
/// `capability.search` returns *cards* — a name, a description, a version, a
/// tool list — and never a skill's instructions. That split is deliberate: it
/// is what stops every skill on the machine reaching every prompt. Loading the
/// body is the separate, deliberate second step.
///
/// `SkillRegistry::load` has always performed every check that step needs:
/// registry identity, the trust list's hash against the bytes on disk *now*,
/// the ARJUN version requirement, the required binaries, the signed-in
/// person's clearance, and the sovereignty mode. Nothing called it. The
/// checking was complete and the capability did not exist, so a skill could be
/// listed and never used.
///
/// ## What loading a skill can and cannot do
///
/// It can narrow. [`crate::skills::narrowing::narrow`] builds the resulting
/// tool set by *filtering the run's own list*, so a tool the run does not hold
/// cannot enter it by any path — whatever the skill's manifest asked for. The
/// refused list is returned so the model is told what it did not get rather
/// than discovering it one refusal at a time.
///
/// It cannot widen anything: not the tool set, not the classification, not the
/// approval class. A skill is guidance, not permission.
fn skill_load(params: Value, deps: &Arc<RuntimeDeps>) -> Result<Value, WireError> {
    let session = deps.session()?;
    let run_id = params
        .get("runId")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let name = params
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim();

    if name.is_empty() {
        return Err(WireError::new(
            code::BAD_PARAMS,
            "skill.load needs the name of a skill",
        ));
    }

    // Fail closed on a run this side does not know, the same as every other
    // method that acts under a run's authority. A skill body is instructions
    // put in front of a model; an unrecognised run has no authority to ask for
    // any.
    let Some(permits) = registered_plan_tools(deps, run_id) else {
        return Err(no_plan_error(run_id));
    };

    let context = crate::skills::SkillContext {
        session: &session,
        // Read now rather than when the run started: switching the workbench
        // into provisioning mode must change which skills may be loaded, not
        // only which ones could have been.
        mode: crate::sovereignty::global_broker().mode(),
        run_permits: &permits,
    };

    let loaded = match deps.skills.load(name, &context) {
        Ok(loaded) => loaded,
        Err(refusal) => {
            let reason = refusal.explain();
            deps.remember(
                run_id,
                events::TaskEventType::SkillRefused,
                json!({ "skill": name, "reason": reason }),
            );
            return Err(WireError::new(code::REFUSED, reason));
        }
    };

    // Bounded before it goes anywhere near a context window. A skill body is
    // capped at read time by `MAX_SKILL_BYTES`, and this is the second cap: the
    // one that keeps a large-but-legal skill from displacing the task's own
    // instructions. Truncation is reported rather than silent — a model acting
    // on half a procedure should know it has half.
    let (body, truncated) = bound_skill_body(&loaded.body);

    deps.remember(
        run_id,
        events::TaskEventType::SkillLoaded,
        json!({
            "skill": loaded.manifest.name,
            "version": loaded.manifest.version,
            // The hash the bytes were checked against, so the record says which
            // revision of the instructions this run was given.
            "sha256": loaded.manifest.sha256,
            "bodyChars": body.chars().count(),
            "truncated": truncated,
            "toolsAllowed": loaded
                .narrowed
                .tools
                .iter()
                .map(|tool| tool.as_str())
                .collect::<Vec<_>>(),
            "toolsRefused": loaded
                .narrowed
                .refused
                .iter()
                .map(|tool| tool.as_str())
                .collect::<Vec<_>>(),
        }),
    );

    Ok(json!({
        "name": loaded.manifest.name,
        "version": loaded.manifest.version,
        "sha256": loaded.manifest.sha256,
        // The instructions themselves. The only place in the product that
        // returns them, and only after every check above passed.
        "body": body,
        "truncated": truncated,
        // What the run may use from here. Never wider than it already was.
        "tools": loaded
            .narrowed
            .tools
            .iter()
            .map(|tool| tool.as_str())
            .collect::<Vec<_>>(),
        "refused": loaded
            .narrowed
            .refused
            .iter()
            .map(|tool| tool.as_str())
            .collect::<Vec<_>>(),
        "note": "These instructions are guidance. They do not grant any tool this run did not                  already hold, and every call is still put to the gateway.",
    }))
}

/// Caps a skill body at what may reasonably enter a context window.
///
/// Separate from the read cap, which is about not loading a huge file at all.
/// This is about not letting a large-but-legal skill displace the task's own
/// instructions. Cut on a character boundary and reported.
fn bound_skill_body(body: &str) -> (String, bool) {
    /// Roughly four thousand tokens of guidance — long enough for a real
    /// procedure, short enough to leave the window to the work.
    const MAX_BODY_CHARS: usize = 16_000;

    if body.chars().count() <= MAX_BODY_CHARS {
        return (body.to_string(), false);
    }
    let cut: String = body.chars().take(MAX_BODY_CHARS).collect();
    (cut, true)
}

/// Fields both tool methods need off the wire.
pub struct CallParams {
    pub run_id: String,
    pub tool_call_id: String,
    pub tool: String,
    pub args: Value,
    /// The model driving this run, stamped onto anything it produces so a
    /// reader of the document knows what wrote it. Absent when the runtime did
    /// not say, which is recorded as "unrecorded" rather than guessed.
    pub model: Option<String>,
}

fn read_call(params: &Value) -> Result<CallParams, WireError> {
    let field = |name: &str| -> Result<String, WireError> {
        params
            .get(name)
            .and_then(Value::as_str)
            .map(str::to_string)
            .ok_or_else(|| {
                WireError::new(code::BAD_PARAMS, format!("missing string field {name:?}"))
            })
    };
    Ok(CallParams {
        run_id: field("runId")?,
        tool_call_id: field("toolCallId")?,
        tool: field("tool")?,
        args: params.get("args").cloned().unwrap_or(Value::Null),
        model: params
            .get("model")
            .and_then(Value::as_str)
            .map(str::to_string),
    })
}

/// Grants outstanding across every run this process is serving.
///
/// Process-wide because a grant is meaningless outside the ledger that issued
/// it, and one ledger keeps "issued here, redeemed here" true no matter how many
/// runtimes are running.
fn ledger() -> &'static Mutex<GrantLedger> {
    static LEDGER: std::sync::OnceLock<Mutex<GrantLedger>> = std::sync::OnceLock::new();
    LEDGER.get_or_init(|| Mutex::new(GrantLedger::new()))
}

/// Decides one call, issuing a grant if the answer is yes.
///
/// When the gateway says a person must look first, this raises the request and
/// **waits** rather than refusing. From the loop's side that is simply a slow
/// authorisation; from the operator's, it appears on the approvals screen and
/// the run continues when they decide. Neither side has to model the other's
/// idea of waiting.
async fn authorize(params: Value, deps: &Arc<RuntimeDeps>) -> Result<Value, WireError> {
    let call = read_call(&params)?;
    let operation = context_api::operation(&params, deps)?;
    if operation.as_ref().is_some_and(|(_, operation)| operation.receipt.is_some()) {
        let grant = ledger().lock().map_err(|_| WireError::new(code::INTERNAL,"Grant ledger unavailable."))?
            .issue(&call.run_id,&call.tool_call_id,&call.tool,&call.args);
        return Ok(json!({ "outcome": "allow", "tool": call.tool, "grant": grant, "replayed": true }));
    }
    let result = authorize_impl(params,deps,operation.as_ref().map(|(_, operation)| operation.id.as_str())).await;
    let refusal = match &result {
        Ok(value) if value["outcome"] != "allow" => Some(WireError::new(code::REFUSED,value["reason"].as_str().unwrap_or("The requested action was refused."))),
        Err(error) => Some(error.clone()),
        _ => None,
    };
    if let (Some((seed, operation)),Some(refusal))=(operation,refusal) {
        let core=serde_json::to_value(context_api::capture(deps,&seed)?).map_err(|_| WireError::new(code::INTERNAL,"The refused action could not be checkpointed."))?;
        deps.events.decline_operation(&seed.lease,&deps.session()?.user.id,&operation.id,refusal,&core)
            .map_err(|error| { deps.audit_health.writes_failed(&error); WireError::new(code::INTERNAL,error) })?;
    }
    result
}

async fn authorize_impl(params: Value, deps: &Arc<RuntimeDeps>, operation_id: Option<&str>) -> Result<Value, WireError> {
    let call=read_call(&params)?;

    // Is this run over? Asked first, and asked of the durable record rather
    // than of anything in memory.
    //
    // This is where a cancellation actually lands. Telling the child to abort
    // is a request that takes effect whenever the loop next looks; this is the
    // boundary it cannot cross. A tool already executing is left to finish —
    // interrupting it is what creates an effect nobody can account for — but
    // nothing new starts. So "stop" means "no further actions", which is the
    // promise a person pressing the button is actually making.
    if let Some(ending) = deps.events.ending(&call.run_id) {
        let reason = format!(
            "This task has ended ({}), so no further tool calls will be made. Stop and report \\
             what was completed.",
            ending.as_str().replace('_', " ")
        );
        return Ok(refused(deps, &call, reason));
    }

    // A run this side has no plan for has no authority to do anything.
    //
    // Asked first, before the ending check even, because the two answer
    // different questions: "this run finished" presumes the run existed. An
    // invented id belongs to no run at all, and the honest answer to it is not
    // a refusal message about a task — it is that there is no task.
    //
    // Before this, `plan_refusal` returned `None` for a missing plan (an early
    // `?` on the table lookup), which the caller read as "no objection". The
    // call then went to the gateway, which grants on the signed-in person's
    // permissions alone — so any string in `runId` could authorise a search of
    // the organisation's documents, outside every budget, with no step counted
    // and no plan to hold it to.
    if !has_registered_plan(deps, &call.run_id) {
        return Err(no_plan_error(&call.run_id));
    }

    // Nothing with a side effect happens that cannot be written down.
    //
    // A read is still allowed: nothing it does needs recording beyond the event
    // naming it, and refusing reads would leave a degraded installation unable
    // even to explain itself. A write, a produced file or a sandboxed command
    // is different — an effect on this machine with no provenance is worse than
    // no effect, and provenance is exactly what is not working.
    //
    // Checked here rather than only at `agent_start_run` because storage can
    // stop working *during* a run: the run that was already in flight when the
    // disk filled is the one that would otherwise write a document nobody can
    // account for.
    if let Some(reason) = durability_refusal(&call, deps) {
        return Ok(refused(deps, &call, reason));
    }

    // The plan is consulted before the gateway. A task that is out of time
    // should not be asking about permissions, and "you have run out of steps"
    // is a more useful thing to tell a model than "that path is fine, but
    // nothing further will happen".
    if let Some(reason) = plan_refusal(&call, deps) {
        return Ok(refused(deps, &call, reason));
    }

    // The deployment's own checks, before the gateway and outside anything the
    // model can address. A hook here can only refuse — it never issues a grant
    // and never widens what the gateway would have allowed — so running it
    // first costs nothing and means a deployment-specific rule is applied even
    // to a call the gateway would have waved through.
    if let Some(reason) = hook_refusal(&call, deps) {
        return Ok(refused(deps, &call, reason));
    }

    // The reservation is live from here, so a failure that leaves by `?` has to
    // give it back explicitly — `refused` covers the paths that produce a
    // verdict, and these do not.
    let verdict = match decide(&call, deps, ApprovalState::NotRequested) {
        Ok(verdict) => verdict,
        Err(error) => {
            release_reservation(deps, &call.run_id, &call.tool_call_id);
            return Err(error);
        }
    };

    let (tool, resolved_path) = match verdict {
        GatewayVerdict::Allow {
            tool,
            resolved_path,
        } => (tool, resolved_path),
        GatewayVerdict::Refuse { reason } => return Ok(refused(deps, &call, reason)),
        GatewayVerdict::NeedsApproval {
            tool,
            summary,
            resolved_path,
        } => {
            let session = match deps.session() {
                Ok(session) => session,
                Err(error) => {
                    release_reservation(deps, &call.run_id, &call.tool_call_id);
                    return Err(error);
                }
            };
            let target = resolved_path
                .as_ref()
                .map(|path| path.display().to_string())
                .unwrap_or_else(|| call.tool.clone());
            deps.remember(
                &call.run_id,
                events::TaskEventType::ApprovalRequested,
                json!({
                    "toolCallId": call.tool_call_id,
                    "tool": call.tool,
                    "target": target,
                }),
            );

            let outcome = approval::await_decision(
                &deps.approvals,
                &deps.events,
                &session,
                &call.run_id,
                tool,
                summary,
                target,
                operation_id.unwrap_or(&call.tool_call_id),
                &call.args,
            )
            .await;

            let decided = matches!(outcome, approval::ApprovalOutcome::Approved { .. });
            deps.remember(
                &call.run_id,
                events::TaskEventType::ApprovalDecided,
                json!({
                    "toolCallId": call.tool_call_id,
                    "tool": call.tool,
                    "approved": decided,
                }),
            );

            match outcome {
                approval::ApprovalOutcome::Approved { .. } => (tool, resolved_path),
                approval::ApprovalOutcome::Unavailable { detail } => {
                    release_reservation(deps, &call.run_id, &call.tool_call_id);
                    deps.audit_health.writes_failed(&detail);
                    return Err(WireError::new(code::INTERNAL, detail));
                }
                other => return Ok(refused(deps, &call, other.refusal())),
            }
        }
    };

    context_api::validate_attempt(&params, deps, false)?;
    let grant = match ledger().lock() {
        Ok(mut ledger) => ledger.issue(&call.run_id, &call.tool_call_id, &call.tool, &call.args),
        Err(_) => {
            release_reservation(deps, &call.run_id, &call.tool_call_id);
            return Err(WireError::new(code::INTERNAL, "grant ledger is poisoned"));
        }
    };

    deps.remember(
        &call.run_id,
        events::TaskEventType::ToolAuthorized,
        json!({
            "toolCallId": call.tool_call_id,
            "tool": tool.as_str(),
            // A reference, so two events about one call can be matched up
            // without either of them carrying what the call was about.
            "argsFingerprint": events::args_fingerprint(&call.args),
        }),
    );

    Ok(json!({
        "outcome": "allow",
        "tool": tool.as_str(),
        "grant": grant,
        "resolvedPath": resolved_path,
    }))
}

/// Puts a call to the run's plan, and says why not when the answer is no.
///
/// Two shapes of refusal, deliberately different:
///
/// - **A tool outside the plan** is refused without stopping the run. The model
///   reads the refusal and can do the rest of the work, or say plainly what it
///   could not do. Stopping there would turn one wrong guess by [`planning`]
///   into a lost run, which is far too high a price for a keyword miss.
/// - **Out of steps, out of time, or going in circles** stops the run, and
///   every later call is refused with the same sentence. Those are the
///   conditions PS Part C asks to be stopped at, and a limit a model could keep
///   retrying against would not be a limit.
///
/// A run with no plan is allowed through. That is not a hole: the only caller
/// that starts a run registers a plan first, and refusing every tool call for a
/// run this table has never heard of would break the runtime's own health check
/// rather than enforce anything.
/// Runs the deployment's checks for one tool call, and records what they said.
///
/// ## Why this is separate from the gateway
///
/// The gateway answers "is this call permitted by the product's rules?" — a
/// question with one right answer that every deployment shares. Hooks answer
/// "does *this site* also forbid it?", which is a different question with a
/// different owner. Folding them together would mean a site-specific rule and a
/// product rule producing the same refusal text, and an operator being unable to
/// tell which one they need to change.
///
/// ## Why an unknown tool passes
///
/// A name that resolves to no tool is refused by the gateway a moment later,
/// with a message naming the tools that do exist. Refusing it here instead would
/// replace that useful sentence with a hook's, and the model would lose the list
/// it needs to correct itself.
///
/// The report is written to the durable record only when a hook had something to
/// say. A run whose checks all passed silently produces no events, which is what
/// keeps the ones that did fire worth reading.
fn hook_refusal(call: &CallParams, deps: &Arc<RuntimeDeps>) -> Option<String> {
    let tool = ToolName::from_str(&call.tool)?;

    let input = crate::hooks::HookInput::Tool {
        run_id: call.run_id.clone(),
        tool,
        // Unresolved on purpose. The gateway resolves paths against the run's
        // roots and has not run yet, and a second resolution here would be a
        // weaker copy of that check — see the note on `hooks::policy`. A hook
        // needing a resolved path belongs at `BeforeArtifactWrite`, which runs
        // after the gateway has produced one.
        path: None,
        mode: crate::sovereignty::global_broker().mode(),
        succeeded: None,
    };

    let report = deps
        .hooks
        .dispatch(crate::hooks::HookPoint::BeforeToolAuthorize, &input);

    if report.blocked || !report.failed.is_empty() || !report.notes.is_empty() {
        deps.remember(
            &call.run_id,
            events::TaskEventType::HookEvaluated,
            serde_json::to_value(&report).unwrap_or_else(|_| {
                // The report is plain data with no exotic types, so this cannot
                // happen; recording that it did beats writing nothing, because
                // a missing hook event reads as a check that never ran.
                json!({
                    "point": "before_tool_authorize",
                    "blocked": report.blocked,
                    "note": "the hook report could not be serialised",
                })
            }),
        );
    }

    report.refusal()
}

/// Refuses a side-effecting call while this installation cannot record it.
///
/// Read-only calls pass. The distinction is the one the whole product rests on:
/// a search that is not written down costs a line in a trace, and a document
/// written to disk that is not written down is an artefact with no provenance —
/// which is the thing an engineer would be asked to sign, and the thing nobody
/// could then stand behind.
fn durability_refusal(call: &CallParams, deps: &Arc<RuntimeDeps>) -> Option<String> {
    // An unknown tool is not this check's business; `decide` refuses it by
    // name. Treating it as side-effecting here would produce a misleading
    // reason for a call that was never going to run.
    let tool = ToolName::from_str(&call.tool)?;
    if tool.is_read_only() {
        return None;
    }
    let refusal = deps.audit_health.refusal()?;
    Some(format!(
        "{} needs to be recorded before it can be done, and this installation cannot record it. {refusal}",
        call.tool
    ))
}

fn plan_refusal(call: &CallParams, deps: &Arc<RuntimeDeps>) -> Option<String> {
    let stopped = {
        // A poisoned table is a panic that happened while the budget was being
        // read, and carrying on would mean running with no budget at all. That
        // is the one case here that fails closed: an unbounded run is worse
        // than a stopped one.
        let Ok(mut plans) = deps.plans.lock() else {
            return Some(
                "This task's plan cannot be read, so there is no budget to hold the work to and \
                 nothing further will be run. Start the task again."
                    .to_string(),
            );
        };
        // A missing plan is not reachable here: `authorize` refuses a run it has
        // no plan for before this is called, and `execute` does the same. The
        // `?` is what remains of the old health-probe exception, which used to
        // read a missing plan as "no objection" — kept only so a table that
        // changed under this call cannot panic, and no longer a way in.
        let plan = plans.get_mut(&call.run_id)?;

        // Checked before `may_call`, which halts the whole plan on an
        // unpermitted tool. Here that is one refused call and no more.
        let permitted = ToolName::from_str(&call.tool)
            .map(|tool| plan.budget.permits(tool))
            .unwrap_or(false);
        if !permitted {
            let allowed: Vec<&str> = plan
                .budget
                .permitted_tools
                .iter()
                .map(|tool| tool.as_str())
                .collect();
            return Some(format!(
                "{} is not one of the tools this task was planned to use. The plan allows: {}. \
                 Do what you can with those, and say plainly what you could not do.",
                call.tool,
                allowed.join(", ")
            ));
        }

        // Reserved, not merely checked.
        //
        // The slot is taken here, under the lock that just decided there was
        // one, and before any grant is issued. The runtime runs read-only tools
        // in parallel, so four searches in a turn arrive here as four
        // concurrent authorisations; when this only *asked* whether there was
        // room, all four read the same unchanged figure and all four were let
        // through. Everything downstream now either settles this lease or gives
        // it back — see `release_reservation`.
        match plan.reserve(
            &call.tool_call_id,
            &ToolCall::new(call.tool.clone(), call.args.clone()),
        ) {
            Continuation::Proceed => return None,
            Continuation::Stop(reason) => reason,
        }
    };

    // Published so the trace says why the run went quiet. Emitted outside the
    // lock: the handler is arbitrary code, and holding the plan table across it
    // would let a slow listener block every other run's authorisation.
    deps.publish(
        &call.run_id,
        json!({
            "type": "plan_stopped",
            "reason": stopped.explain(),
            "tool": call.tool,
        }),
    );
    deps.remember(
        &call.run_id,
        events::TaskEventType::PlanStopped,
        json!({ "reason": stopped.explain(), "tool": call.tool }),
    );
    Some(stopped.explain())
}

/// Renders arguments the way an approver will read them.
///
/// Values are truncated: a write's `content` can be a whole document, and an
/// approval screen that makes somebody scroll past 30 KB to find the path is one
/// where they stop reading and start clicking yes.
fn render_arguments(args: &Value) -> Vec<String> {
    const MAX_VALUE_CHARS: usize = 200;
    let Some(object) = args.as_object() else {
        return Vec::new();
    };
    object
        .iter()
        .map(|(key, value)| {
            let rendered = match value {
                Value::String(text) => text.clone(),
                other => other.to_string(),
            };
            if rendered.chars().count() > MAX_VALUE_CHARS {
                let head: String = rendered.chars().take(MAX_VALUE_CHARS).collect();
                format!("{key} = {head}… ({} characters)", rendered.chars().count())
            } else {
                format!("{key} = {rendered}")
            }
        })
        .collect()
}

/// Puts a call through the gateway. Shared by both methods so the two answers
/// cannot diverge.
fn decide(
    call: &CallParams,
    deps: &Arc<RuntimeDeps>,
    approval: ApprovalState,
) -> Result<GatewayVerdict, WireError> {
    let session = deps.session()?;
    let roots = deps.roots_for(&call.run_id);
    let tool_call = anchor_path(ToolCall::new(call.tool.clone(), call.args.clone()), &roots);
    let context = TaskContext {
        session: &session,
        workspace_roots: &roots,
        confidential_work_permitted: deps.confidential_work_permitted(),
        // What the caller already holds. `authorize` raises the request and
        // waits; `execute` passes `Granted` because it only runs after that
        // wait returned yes, and re-deciding as `NotRequested` would ask the
        // same person the same question a second time.
        approval,
    };
    Ok(ToolGateway::decide(&tool_call, &context))
}


/// Resolves a relative `path` argument against the run's workspace.
///
/// The gateway's containment check compares a path against the permitted roots,
/// so a bare `"note.txt"` fails it — it is not under any root, it is under
/// nothing. That would make every relative path a refusal, which matters
/// because relative is exactly what the model is told to use: an absolute path
/// is a temp directory with a UUID in it, and a 7B model asked to reproduce one
/// verbatim across a dozen calls will not.
///
/// So the anchoring happens here, before the gateway sees the call, and the
/// gateway's check is unchanged. Traversal is still refused: `../../etc/passwd`
/// joined onto the root still normalises to somewhere outside it, and
/// `resolve_within` still says no. This makes relative paths *expressible*, not
/// permitted — the containment decision stays exactly where it was.
fn anchor_path(call: ToolCall, roots: &[PathBuf]) -> ToolCall {
    let Some(root) = roots.first() else {
        // No workspace, so nothing to anchor against. The gateway refuses every
        // path-taking tool in that state, which is the correct outcome.
        return call;
    };
    // Copied out before `arguments` is moved: `text` borrows the call, and the
    // rewrite below takes ownership of what it borrows from.
    let Some(raw) = call.text("path").map(str::to_string) else {
        return call;
    };
    // Only a *purely* relative path is anchored. Anything carrying a root is
    // passed through to be judged as written, because `Path::join` replaces
    // rather than appends when the argument has one: on Windows
    // `C:\runs\<id>`.join("/etc/passwd") is `C:/etc/passwd` — outside the
    // workspace, silently. The gateway refuses that, but anchoring should not
    // be manufacturing paths that depend on a later check to be safe.
    let candidate = Path::new(&raw);
    if candidate.is_absolute() || candidate.has_root() {
        return call;
    }

    let mut arguments = call.arguments;
    if let Some(object) = arguments.as_object_mut() {
        object.insert(
            "path".to_string(),
            Value::String(root.join(&raw).display().to_string()),
        );
    }
    ToolCall {
        tool: call.tool,
        arguments,
    }
}

/// Redeems the grant, re-derives the verdict, then runs the tool.
async fn execute(params: Value, deps: &Arc<RuntimeDeps>) -> Result<Value, WireError> {
    let Some((seed, operation)) = context_api::operation(&params, deps)? else {
        return execute_untracked(params, deps).await;
    };
    // Consume the transport grant even for a receipt replay. A replay restores
    // prior work, and must not consume another logical plan step.
    let grant = params.get("grant").and_then(Value::as_str).ok_or_else(|| WireError::new(code::REFUSED, "No authorization grant was presented."))?;
    let call = read_call(&params)?;
    if operation.receipt.is_some() {
        ledger().lock().map_err(|_| WireError::new(code::INTERNAL, "Grant ledger unavailable."))?
            .redeem(grant,&call.run_id,&call.tool_call_id,&call.tool,&call.args)
            .map_err(|error| WireError::new(code::REFUSED,error.to_string()))?;
        release_reservation(deps,&call.run_id,&call.tool_call_id);
        return operation.receipt.unwrap().into_result();
    }
    let approval_id = approval::approval_id(&call.run_id,&operation.id);
    if let Some(approval) = deps.events.approval(&approval_id).map_err(|error| WireError::new(code::REFUSED,error))? {
        approval.authorises(&json!({ "tool": operation.tool, "args": call.args }),chrono::Utc::now())
            .map_err(|error| WireError::new(code::REFUSED,error.explain()))?;
    }
    if operation.status == "running" && operation.fence_token != seed.lease.fence_token {
        let tool = crate::orchestrator::tools::ToolName::from_str(&operation.tool)
            .ok_or_else(|| WireError::new(code::REFUSED, "The saved tool is unknown."))?;
        let policy = tool_policy::retry_policy_of(tool);
        if policy.safe_to_retry && operation.attempts <= u32::from(policy.max_retries) {
            tokio::time::sleep(std::time::Duration::from_secs(policy.backoff_for(operation.attempts.min(255) as u8))).await;
            context_api::validate_attempt(&params, deps, true)?;
        }
    }
    if let Some(receipt) = deps.events.start_operation(&seed.lease,&operation.id).map_err(|error| WireError::new(code::REFUSED,error))? {
        release_reservation(deps,&call.run_id,&call.tool_call_id);
        return receipt.into_result();
    }
    let mut scoped = params;
    scoped["idempotencyKey"] = json!(operation.id);
    let result = execute_untracked(scoped,deps).await;
    let stored = (|| {
        let core = serde_json::to_value(context_api::capture(deps,&seed)?).map_err(|_| WireError::new(code::INTERNAL,"Run resources could not be serialized."))?;
        deps.events.finish_operation(&seed.lease,&deps.session()?.user.id,&operation.id,
            &events::operations::ToolReceipt::from_result(&result),&core)
            .map_err(|error| WireError::new(code::INTERNAL,error))
    })();
    if let Err(error) = stored {
        deps.audit_health.writes_failed(&error.message);
        let _ = deps.events.release_claim(&call.run_id,&seed.lease.owner,seed.lease.fence_token);
        return Err(error);
    }
    result
}

async fn execute_untracked(params: Value, deps: &Arc<RuntimeDeps>) -> Result<Value, WireError> {
    let call = read_call(&params)?;
    let execution_seed = context_api::validate_attempt(&params,deps,false)?;
    let grant = params
        .get("grant")
        .and_then(Value::as_str)
        .ok_or_else(|| WireError::new(code::REFUSED, "no authorisation grant was presented"))?;

    // Asked again here, and not only in `authorize`.
    //
    // A grant is single-use and bound to its call, so in the ordinary course
    // nothing reaches this without having passed the same check a moment ago.
    // But the two are separated in time, and what changes in between is exactly
    // what matters: a run that ended between authorisation and execution has
    // had its plan released, and a call redeemed against a plan that is no
    // longer registered is an action outside every budget this side can name.
    if !has_registered_plan(deps, &call.run_id) {
        return Err(no_plan_error(&call.run_id));
    }

    ledger()
        .lock()
        .map_err(|_| WireError::new(code::INTERNAL, "grant ledger is poisoned"))?
        .redeem(
            grant,
            &call.run_id,
            &call.tool_call_id,
            &call.tool,
            &call.args,
        )
        .map_err(|error| WireError::new(code::REFUSED, error.to_string()))?;

    // Independent of the grant. A grant proves the gateway said yes once; this
    // asks it again, because the state it decides against — the signed-in user,
    // the sovereignty mode — can have changed since.
    //
    // `Granted` because the grant is itself the evidence a person already said
    // yes: re-deciding as `NotRequested` would put the same request in front of
    // the same approver a second time, for an action they have just approved.
    //
    // A refusal here is a call that will not run, so its reservation goes back:
    // the slot was taken when the gateway said yes a moment ago, and the state
    // it decides against has changed since. Charging the run for a step the
    // policy then stopped would let a rule the model kept running into exhaust
    // a run that had done nothing.
    let verdict = match decide(&call, deps, ApprovalState::Granted) {
        Ok(verdict) => verdict,
        Err(error) => {
            release_reservation(deps, &call.run_id, &call.tool_call_id);
            return Err(error);
        }
    };
    let (tool, resolved_path) = match verdict {
        GatewayVerdict::Allow {
            tool,
            resolved_path,
        } => (tool, resolved_path),
        GatewayVerdict::NeedsApproval { summary, .. } => {
            release_reservation(deps, &call.run_id, &call.tool_call_id);
            return Err(WireError::new(code::REFUSED, summary));
        }
        GatewayVerdict::Refuse { reason } => {
            release_reservation(deps, &call.run_id, &call.tool_call_id);
            return Err(WireError::new(code::REFUSED, reason));
        }
    };

    let session = match deps.session() {
        Ok(session) => session,
        Err(error) => {
            release_reservation(deps, &call.run_id, &call.tool_call_id);
            return Err(error);
        }
    };
    // Anchored the same way the gateway saw it, so the tool acts on exactly the
    // path that was judged rather than on the raw argument.
    let tool_call = anchor_path(
        ToolCall::new(call.tool.clone(), call.args.clone()),
        &deps.roots_for(&call.run_id),
    );

    // Has this exact side effect already happened, or is it happening now, or
    // did the lights go out in the middle of it? Asked before the tool runs and
    // answered from disk, because the case it exists for is the one where the
    // process that ran it the first time is gone. See [`events::idempotency`].
    let effect = events::is_side_effecting(tool).then(|| {
        // The runtime may supply a key. Accepting one is safe because the
        // recorded tool and argument fingerprint are checked against the call
        // being made — a key that names a different call is refused, not
        // replayed — and deriving one here means this works with a runtime
        // bundle that has never heard of idempotency keys.
        let key = params
            .get("idempotencyKey")
            .and_then(Value::as_str)
            .map(str::to_string)
            .unwrap_or_else(|| events::derive_key(&call.run_id, tool.as_str(), &call.args));
        (key, events::args_fingerprint(&call.args))
    });

    if let Some((key, fingerprint)) = &effect {
        // A reference to what is being acted on, so a person reconciling an
        // unknown effect later is told which file to go and look at. A name,
        // never contents.
        let target = resolved_path
            .as_deref()
            .and_then(Path::file_name)
            .and_then(|name| name.to_str())
            .unwrap_or_else(|| tool.as_str())
            .to_string();

        let lookup = match &execution_seed {
            Some(seed) => deps.events.begin_effect_fenced(&seed.lease,key,tool.as_str(),fingerprint,&target),
            None => deps.events.begin_effect(&call.run_id,key,tool.as_str(),fingerprint,&target),
        };
        match lookup {
            // Nothing has happened under this key. The intent is now on disk,
            // so a process that dies during the next few lines leaves evidence
            // it was trying rather than leaving nothing at all.
            events::EffectLookup::Fresh => {
                deps.remember(
                    &call.run_id,
                    events::TaskEventType::ToolEffectPending,
                    json!({
                        "toolCallId": call.tool_call_id,
                        "tool": tool.as_str(),
                        "target": target,
                        "idempotencyKey": key,
                    }),
                );
            }

            // Already settled. Return what it did; do not do it again.
            events::EffectLookup::Settled(recorded) => {
                deps.remember(
                    &call.run_id,
                    events::TaskEventType::ToolReplayed,
                    json!({
                        "toolCallId": call.tool_call_id,
                        "tool": tool.as_str(),
                        "firstRunAt": recorded.at,
                        "succeeded": recorded.succeeded(),
                    }),
                );
                let outcome = recorded.replay();
                record_call(deps, &call.run_id, tool.as_str(), &outcome);
                // Counted like any other call. A replay still costs a turn and
                // a slice of the context window, and a budget that did not
                // count it is one a model repeating itself never reaches.
                record_step(deps, &call.run_id, &call.tool_call_id, tool);
                return match outcome {
                    // Cut and sanitised exactly as a fresh result is. A replay
                    // that came back longer than the same call did the first
                    // time would make the two disagree about what happened.
                    Ok(text) => Ok(json!({
                        "text": crate::orchestrator::tools::truncate_response(tool, text),
                        "details": { "tool": tool.as_str(), "replayed": true },
                    })),
                    Err(reason) => Err(WireError::new(
                        code::TOOL_FAILED,
                        crate::orchestrator::tools::sanitise_failure(&reason),
                    )),
                };
            }

            // Two attempts at one side effect at the same moment. Refused
            // rather than serialised: whichever finished last would win, which
            // is not a decision anybody made.
            events::EffectLookup::InFlight(recorded) => {
                let reason = format!(
                    "another attempt at this exact action is already under way ({} on {}), so it \
                     was not started a second time.",
                    recorded.tool, recorded.target
                );
                remember_refusal(deps, &call, &reason);
                // Not started, so not charged.
                release_reservation(deps, &call.run_id, &call.tool_call_id);
                return Err(WireError::new(code::REFUSED, reason));
            }

            // The one that matters. A side effect was in flight when the
            // process went away, and nobody can say whether it took. Repeating
            // it could do it twice; assuming it happened could mean it never
            // does. Both are worse than stopping and asking.
            events::EffectLookup::Unknown(recorded) => {
                let reason = recorded.unknown_refusal();
                remember_refusal(deps, &call, &reason);
                // Not started, so not charged.
                release_reservation(deps, &call.run_id, &call.tool_call_id);
                return Err(WireError::new(code::REFUSED, reason));
            }

            events::EffectLookup::Conflict(conflict) => {
                let reason = conflict.to_string();
                remember_refusal(deps, &call, &reason);
                // Not started, so not charged.
                release_reservation(deps, &call.run_id, &call.tool_call_id);
                return Err(WireError::new(code::REFUSED, reason));
            }
            events::EffectLookup::Unavailable { reason } => {
                deps.audit_health.writes_failed(&reason);
                release_reservation(deps, &call.run_id, &call.tool_call_id);
                remember_refusal(deps, &call, &reason);
                return Err(WireError::new(code::REFUSED, reason));
            }
        }
    }

    // Four tools are handled here rather than in `LocalToolRunner` because each
    // needs the run's accumulated state — its calculations, its evidence, the
    // files it has produced — and the runner is built fresh per call, so it
    // cannot hold any of it.
    let outcome = match tool {
        ToolName::CreateDocx => {
            artifacts::create_docx(&call, resolved_path.as_deref(), &session, &tool_call)
        }
        ToolName::CreateXlsx => {
            artifacts::create_xlsx(resolved_path.as_deref(), &deps.calculations, &call.run_id)
        }
        ToolName::CreatePptx => {
            artifacts::create_pptx(&call, resolved_path.as_deref(), &tool_call)
        }
        // Recorded as the run's evidence on the way past, and numbered once
        // across the whole run so a citation means one passage. See
        // [`retrieval`].
        ToolName::SearchDocuments => LocalToolRunner::new(deps.index.as_ref(), &session)
            .search_hits(&tool_call)
            .map(|(query, hits)| retrieval::record(&deps.passages, &call.run_id, &query, &hits)),
        // Handled here for the same reason as search: a page pulled back later
        // is this run's evidence and has to be numbered against the same table,
        // or the marker the model cites will resolve to a different passage.
        ToolName::LoadMoreEvidence => LocalToolRunner::new(deps.index.as_ref(), &session)
            .region_hits(&tool_call)
            .map(|(_, from_page, to_page, hits)| {
                let name = hits
                    .first()
                    .map(|hit| hit.document_name.clone())
                    .unwrap_or_else(|| "that document".to_string());
                retrieval::record_region(
                    &deps.passages,
                    &call.run_id,
                    &name,
                    from_page,
                    to_page,
                    &hits,
                )
            }),
        // Served through the same boundary the RPC methods use, so a model
        // reaching memory by tool and a runtime reaching it by method get
        // identical policy. Two paths with two implementations would be two
        // places for the entitlement check to drift.
        ToolName::MemoryRecallAuthorized => memory_api::recall_authorized(
            {
                let mut memory_params=call.args.clone();
                memory_params["runId"]=json!(call.run_id);
                memory_params["attemptId"]=params["attemptId"].clone();
                memory_params["fenceToken"]=params["fenceToken"].clone();
                memory_params
            },
            deps,
        )
        .map(|value| render_memory(&value))
        .map_err(|error| error.message),
        ToolName::MemoryPromoteApproved => memory_api::promote_approved(
            json!({
                "runId": call.run_id,
                "key": tool_call.text("key").unwrap_or_default(),
                "approvalId": tool_call.text("approvalId").unwrap_or_default(),
            }),
            deps,
        )
        .map(|value| render_memory(&value))
        .map_err(|error| error.message),
        // Served through the same handler the RPC uses, for exactly the reason
        // memory is: two paths with two implementations are two places for the
        // plan and session filtering to drift.
        //
        // It used to fall through to `LocalToolRunner`, which refuses it —
        // "served on the agent path, not by this runner" — so a model that
        // called `capability.search` got an error saying the tool was
        // available somewhere else. It was in the catalogue, it was in the
        // plan, and it could not be called.
        ToolName::CapabilitySearch => capability_search(
            json!({
                "runId": call.run_id,
                "query": tool_call.text("query").unwrap_or_default(),
            }),
            deps,
        )
        .map(|value| render_capabilities(&value))
        .map_err(|error| error.message),
        ToolName::ValidateArtifact => {
            validate(deps, &call.run_id, resolved_path.as_deref(), &session, &tool_call).await
        }
        _ => {
            // Built with everything the run has, rather than with the index
            // alone.
            //
            // The agent path used to construct this with `LocalToolRunner::new`,
            // leaving `multimodal`, `subagents`, `inherited` and `run_workspace`
            // all `None`. Three tools in the catalogue were therefore
            // unreachable: `knowledge.multimodal_retrieve` had no index,
            // `agent.delegate_readonly` answered "subagents are not available
            // on this machine", and any worker that had started would have had
            // no workspace to be confined to.
            let workspace = deps.root_for(&call.run_id);
            let inherited =
                inherited_policy_for(deps, &session, &call.run_id, workspace.as_deref());
            let runner = runner_for(deps, &session, inherited.as_ref(), workspace.as_deref());
            let result = runner.run(tool, &tool_call, resolved_path.as_deref()).await;
            // A successful calculation is kept, so the workbook can show the
            // working rather than the model's memory of it.
            if tool == ToolName::RunCalculation && result.is_ok() {
                if let Ok(record) =
                    crate::orchestrator::calculation::evaluate(tool_call.text("expression").unwrap_or_default())
                {
                    if let Ok(mut table) = deps.calculations.lock() {
                        table.entry(call.run_id.clone()).or_default().push(record);
                    }
                }
            }
            result
        }
    };

    // Settled whichever way it went, and before anything is returned to the
    // loop. The intent went down before the tool ran; this is the other half.
    // A side effect that happened and was never settled stays `pending`, and
    // the next start promotes it to `unknown` — which is the correct answer
    // when nobody can say what happened, and the wrong one when somebody could
    // have.
    if let Some((key, _)) = &effect {
        let settled = match &execution_seed {
            Some(seed) => deps.events.settle_effect_fenced(&seed.lease,key,&outcome),
            None => deps.events.settle_effect(&call.run_id,key,&outcome),
        };
        if let Err(reason) = settled {
            deps.audit_health.writes_failed(&reason);
            // The intent stays pending (unknown on recovery). Do not report a
            // successful result or authorize more effects without a record.
            record_step(deps, &call.run_id, &call.tool_call_id, tool);
            return Err(WireError::new(code::INTERNAL, reason));
        }
    }

    if outcome.is_ok() {
        remember_if_produced(deps, &call.run_id, tool, resolved_path.as_deref(), &tool_call);
    }
    record_call(deps, &call.run_id, tool.as_str(), &outcome);
    remember_outcome(deps, &call, tool, resolved_path.as_deref(), &outcome);

    // Counted whatever the tool returned. A failed call cost the same wall
    // clock and the same context window as a successful one, and a budget that
    // only counts successes is one a model going in circles never reaches.
    record_step(deps, &call.run_id, &call.tool_call_id, tool);

    match outcome {
        // Cut to the tool's own ceiling on the way out, in one place. Doing it
        // inside each tool would mean a tool added later returning whatever it
        // liked, and the ceiling is what stops one call taking half the window.
        Ok(text) => Ok(json!({
            "text": crate::orchestrator::tools::truncate_response(tool, text),
            "details": { "tool": tool.as_str() },
        })),
        // A tool that fails says why, in words the model can act on. Returned as
        // an error frame so the runtime turns it into an error tool result
        // rather than passing it off as an answer.
        //
        // Sanitised first: a failure carrying a backtrace or the operator's home
        // directory is a failure the model will quote into a document.
        Err(reason) => Err(WireError::new(
            code::TOOL_FAILED,
            crate::orchestrator::tools::sanitise_failure(&reason),
        )),
    }
}

/// Re-opens a file this run produced and says what is actually in it.
///
/// The runner's own check asks whether the path exists and is not empty, which
/// is all it *can* ask: it is built fresh per call and does not know the file
/// was rendered from the `approval_note` template. This does, because the run
/// remembered it — so `validate_artifact` on a document opens the package and
/// checks the sections are really there, which is what ARJUN design rule 30 asks for and
/// what the tool's own description promises the model.
async fn validate(
    deps: &Arc<RuntimeDeps>,
    run_id: &str,
    resolved_path: Option<&Path>,
    session: &Session,
    tool_call: &ToolCall,
) -> Result<String, String> {
    let known = resolved_path.and_then(|path| {
        artifacts::for_run(&deps.produced, run_id)
            .into_iter()
            .find(|produced| Path::new(&produced.path) == path)
    });

    let Some(produced) = known else {
        // Not something this run produced, so there is no template to check it
        // against and no claim to make beyond what is on disk. The runner's
        // existence-and-size check is then the honest answer.
        let runner = LocalToolRunner::new(deps.index.as_ref(), session);
        return runner.run(ToolName::ValidateArtifact, tool_call, resolved_path).await;
    };

    let report = artifacts::check(&produced);
    if report.sound {
        Ok(format!("{}: {}", report.name, report.detail))
    } else {
        Err(format!(
            "{} did not pass its check: {}. Correct it and produce it again.",
            report.name,
            report.problems.join("; ")
        ))
    }
}

/// Records a file the call has just produced, so it can be re-opened later.
fn remember_if_produced(
    deps: &Arc<RuntimeDeps>,
    run_id: &str,
    tool: ToolName,
    resolved_path: Option<&Path>,
    tool_call: &ToolCall,
) {
    let kind = match tool {
        ToolName::CreateDocx => artifacts::Kind::Document,
        ToolName::CreateXlsx => artifacts::Kind::Workbook,
        ToolName::CreatePptx => artifacts::Kind::Deck,
        ToolName::WriteScopedFile => artifacts::Kind::Text,
        _ => return,
    };
    let Some(path) = resolved_path else { return };

    let template = if tool == ToolName::CreateDocx {
        tool_call.text("template").map(str::to_string)
    } else {
        None
    };
    let root = deps.root_for(run_id);
    artifacts::remember(
        &deps.produced,
        run_id,
        artifacts::produced_from(path, root.as_deref(), kind, template),
    );
}

/// Keeps what a tool call did, for the run's record.
///
/// A refusal is recorded as its own outcome rather than as a failure. The two
/// look the same to a naive reader and mean opposite things: a failure is the
/// tool going wrong, a refusal is the policy working, and a Tasks screen that
/// paints every refusal red teaches people to skip the ones that matter.
fn record_call(
    deps: &Arc<RuntimeDeps>,
    run_id: &str,
    tool: &str,
    outcome: &Result<String, String>,
) {
    let record = match outcome {
        Ok(text) => tasks::ToolCallRecord::new(tool, tasks::CallOutcome::Succeeded, text),
        Err(reason) => {
            // The gateway and the plan both refuse in this wording; a tool that
            // simply went wrong does not. Read from the reason rather than
            // threaded through, because every refusal path already produces a
            // sentence and none of them produces a code.
            let refused = reason.contains("not permitted")
                || reason.contains("planned to use")
                || reason.contains("permitted steps")
                || reason.contains("was not approved")
                || reason.contains("going in circles");
            let kind = if refused {
                tasks::CallOutcome::Refused
            } else {
                tasks::CallOutcome::Failed
            };
            tasks::ToolCallRecord::new(tool, kind, reason)
        }
    };

    if let Ok(mut table) = deps.calls.lock() {
        table.entry(run_id.to_string()).or_default().push(record);
    }
}

/// Marks a step spent and publishes how far through the plan the run is.
/// Gives back a slot reserved for a call that will not run.
///
/// Called from every path that refuses *after* the plan admitted the call: the
/// durability check, a deployment hook, the gateway itself, a declined
/// approval, a grant that was never redeemed. None of those spent anything, and
/// a budget that charged for them would let a policy the model kept running
/// into exhaust a run that had done no work.
fn release_reservation(deps: &Arc<RuntimeDeps>, run_id: &str, tool_call_id: &str) {
    if let Ok(mut plans) = deps.plans.lock() {
        if let Some(plan) = plans.get_mut(run_id) {
            plan.release(tool_call_id);
        }
    }
}

fn record_step(deps: &Arc<RuntimeDeps>, run_id: &str, tool_call_id: &str, tool: ToolName) {
    // Built inside the lock, published outside it: the handler is arbitrary
    // code, and holding the plan table across a slow listener would stall every
    // other run's authorisation.
    let progress = {
        let Ok(mut plans) = deps.plans.lock() else {
            return;
        };
        let Some(plan) = plans.get_mut(run_id) else {
            return;
        };
        // Settles the slot this call reserved when it was authorised, rather
        // than incrementing a counter. Two things follow from that and neither
        // is true of a bare increment: a call cannot be charged twice, because
        // the second settlement finds no lease; and a call that never reserved
        // one -- which should not happen, and would mean a grant issued outside
        // the plan -- is not silently charged to a budget it never entered.
        //
        // A step, not a checklist tick: one planned step can take several tool
        // calls, and ticking a step off per call would report a document as
        // produced and checked after four searches.
        if !plan.settle(tool_call_id) {
            log::warn!(
                "[tasks] run {run_id}: {} settled without holding a reservation; the budget was                  not charged for it",
                tool.as_str()
            );
            return;
        }
        json!({
            "type": "plan_step",
            "tool": tool.as_str(),
            "stepsTaken": plan.steps_taken(),
            "maxSteps": plan.budget.max_steps,
            "stepsPlanned": plan.steps.len(),
            // What else is in flight. A person watching a parallel batch sees
            // the counter jump and wants to know which calls did it.
            "inFlight": plan.leases_outstanding(),
        })
    };
    deps.publish(run_id, progress.clone());
    // The same figures, kept. A run recovered after a restart should show how
    // far through its budget it got, and the plan table that knows is in memory.
    deps.remember(
        run_id,
        events::TaskEventType::PlanStep,
        json!({
            "tool": tool.as_str(),
            "stepsTaken": progress.get("stepsTaken").cloned().unwrap_or(Value::Null),
            "maxSteps": progress.get("maxSteps").cloned().unwrap_or(Value::Null),
        }),
    );
}

/// Builds the tool runner with everything this run carries.
///
/// The inherited policy is derived here rather than stored, because it is a
/// function of things that can change during a run: who is signed in, what the
/// plan still permits, and where the run is writing. A policy captured at start
/// would be one a narrowing mid-run could not tighten.
///
/// Every field narrows a child and none widens it — `InheritedPolicy::of_run`
/// takes the run's *permitted tools*, so a worker cannot be handed a tool its
/// parent does not hold, and takes the run's workspace, so it cannot reach a
/// file outside it.
fn runner_for<'a>(
    deps: &'a Arc<RuntimeDeps>,
    session: &'a Session,
    inherited: Option<&'a InheritedPolicy>,
    workspace: Option<&'a std::path::Path>,
) -> LocalToolRunner<'a> {
    let mut runner = LocalToolRunner::with_multimodal(
        deps.index.as_ref(),
        deps.multimodal.as_ref(),
        session,
    );
    runner.subagents = Some(deps.subagents.as_ref());
    runner.inherited = inherited;
    runner.run_workspace = workspace;
    runner
}

/// The policy a child of this run would inherit.
///
/// Derived per call rather than stored, because it is a function of things that
/// can change during a run: who is signed in, what the plan still permits, and
/// where the run is writing. A policy captured at start-up would be one that a
/// narrowing mid-run could not tighten.
///
/// `None` when the run has no plan — a run this side does not know has no
/// permitted tools to pass on, and a child inheriting an empty set could do
/// nothing anyway. Returning `None` makes the delegation refuse with the
/// reason rather than start a worker that can do nothing.
fn inherited_policy_for(
    deps: &Arc<RuntimeDeps>,
    session: &Session,
    run_id: &str,
    workspace: Option<&std::path::Path>,
) -> Option<InheritedPolicy> {
    let permitted = registered_plan_tools(deps, run_id)?;
    let root = workspace?;
    Some(InheritedPolicy::of_run(
        session,
        // The ceiling a child may not exceed. Taken as the most restrictive
        // this deployment recognises rather than the parent's own, because a
        // read-only worker has no business seeing more than it must.
        crate::policy::Classification::Internal,
        root,
        &permitted,
    ))
}

/// Turns the skill cards into the prose the model reads.
///
/// Metadata only, and said so: a card carries a name, a description, a version
/// and a tool list, never a skill's instructions. Loading those is the separate
/// `skill.load` step, and a model told the difference here does not have to
/// infer it.
fn render_capabilities(value: &Value) -> String {
    let cards = value.get("skills").and_then(Value::as_array);
    let Some(cards) = cards.filter(|cards| !cards.is_empty()) else {
        // Said plainly. A model told nothing came back asks a different next
        // question from one told the search failed.
        return "No installed skill matches that, so nothing was loaded. Carry on with the \
                tools this task already has."
            .to_string();
    };

    let mut out = String::from(
        "These installed skills match. This is their description only; ask for one by name to \
         read its instructions.\n\n",
    );
    for card in cards {
        let name = card.get("name").and_then(Value::as_str).unwrap_or("unnamed");
        let description = card
            .get("description")
            .and_then(Value::as_str)
            .unwrap_or("no description");
        let version = card.get("version").and_then(Value::as_str).unwrap_or("?");
        out.push_str(&format!("- {name} (v{version}): {description}"));
        // Whether it can actually be used right now. A card for a quarantined
        // skill is still worth showing -- a model that cannot see it will keep
        // looking for it -- but presenting it as usable would waste a step.
        if card.get("available").and_then(Value::as_bool) == Some(false) {
            let why = card
                .get("unavailableBecause")
                .and_then(Value::as_str)
                .unwrap_or("it is not available");
            out.push_str(&format!(" — not usable: {why}"));
        }
        out.push_str("
");
    }
    out
}

/// Turns a memory result into the prose the model reads.
///
/// Written here rather than in [`memory_api`] because that module answers an
/// RPC whose caller is a program, and this answers a tool call whose caller is a
/// model. The same JSON serves both; only the rendering differs.
fn render_memory(value: &Value) -> String {
    if value.get("promoted").and_then(Value::as_bool) == Some(true) {
        let key = value.get("key").and_then(Value::as_str).unwrap_or("that fact");
        return format!(
            "Recorded {key} in the project's memory under the approval you were granted.              Changing the value later needs a new approval."
        );
    }

    let scope = value.get("scope").and_then(Value::as_str).unwrap_or("that scope");
    let items = value.get("items").and_then(Value::as_array);
    let Some(items) = items.filter(|items| !items.is_empty()) else {
        // Said explicitly. A model told nothing came back asks a different
        // question; one told nothing at all assumes memory is unavailable and
        // answers from its own recollection instead.
        return format!(
            "Nothing is remembered for {scope}. Do not treat that as evidence either way —              search the documents for anything you need to assert."
        );
    };

    let mut out = format!("{} remembered item(s) for {scope}.

", items.len());
    for item in items {
        let key = item.get("key").and_then(Value::as_str).unwrap_or("");
        let body = item.get("value").and_then(Value::as_str).unwrap_or("");
        out.push_str(&format!("- {key}: {body}
"));
    }
    out.push_str(
        "
These are this deployment's own notes, not retrieved passages. A claim that needs a          citation still needs a search.
",
    );
    out
}

/// Names the tool catalogue exposes. Used by the absence test in `tests/`.
pub fn catalogue() -> Vec<&'static str> {
    ToolName::ALL.iter().map(|tool| tool.as_str()).collect()
}


#[cfg(test)]
mod conversations_tests;
#[cfg(test)]
mod memory_boundary_tests;
#[cfg(test)]
mod tests;
