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
pub mod cancellation;
pub mod completion;
pub mod conversations;
pub mod doc_pipeline;
pub mod documents;
pub mod events;
pub mod grants;
pub mod memory;
pub mod memory_api;
pub mod outcome;
pub mod planning;
pub mod protocol;
pub mod recording;
pub mod resume;
pub mod retrieval;
pub mod stages;
pub mod tasks;
pub mod tool_policy;
pub mod turn_context;
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
    /// Everything the OCR models have read, kept past the turn that read it.
    ///
    /// Backs `document.read_pages`. Held here rather than reached for because a
    /// read is scoped by the signed-in owner and the conversation the document
    /// was attached to, and `LocalToolRunner` — which is rebuilt per call — knows
    /// neither. See [`documents`].
    pub documents: Arc<documents::DocumentStore>,
    /// Which conversation each live run belongs to.
    ///
    /// The second half of the scope above. A run knows its own id; the
    /// document store answers questions about a *conversation*, and this is
    /// what turns one into the other. A run that is not in the table can read
    /// nothing, which is the safe direction: without a conversation there is no
    /// scope to check a document against.
    pub run_to_conversation: Arc<conversations::RunToConversation>,
    /// The notebook graphs, for `knowledge.build_graph`.
    ///
    /// Held for the same reason `documents` is: every query in
    /// [`crate::knowledge::graph::persist`] takes an owner id, and
    /// `LocalToolRunner` is rebuilt per call knowing nothing about who is
    /// asking. A graph is a summary of what a person documents say, so
    /// reaching one is an entitlement question, not a lookup.
    pub notebooks: Arc<crate::knowledge::NotebookStore>,
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
            while let Ok(Some(line)) = lines.next_line().await {
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
                dispatch(frame, &reader_deps, &reader_outbound, &pending, &emit).await;
            }
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
    match method {
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
                render_arguments(&call.args),
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
                other => return Ok(refused(deps, &call, other.refusal())),
            }
        }
    };

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
    let call = read_call(&params)?;
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

        match deps
            .events
            .begin_effect(&call.run_id, key, tool.as_str(), fingerprint, &target)
        {
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
            json!({ "runId": call.run_id, "scope": tool_call.text("scope").unwrap_or_default() }),
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
        // Served here rather than by `LocalToolRunner` for the same reason
        // memory is: the answer depends on who is asking and which conversation
        // they are in, and the runner is rebuilt per call holding neither.
        ToolName::ReadAttachedPages => read_attached_pages(deps, &call, &session, &tool_call),
        ToolName::SearchAttachedDocuments => {
            search_attached_documents(deps, &call, &session, &tool_call)
        }
        ToolName::BuildDocumentGraph => build_document_graph(deps, &session, &tool_call),
        ToolName::NotebookList => notebook_list(deps, &session),
        ToolName::NotebookCreate => notebook_create(deps, &session, &tool_call),
        ToolName::NotebookRename => notebook_rename(deps, &session, &tool_call),
        ToolName::NotebookDelete => notebook_delete(deps, &session, &tool_call),
        ToolName::NotebookSources => notebook_sources(deps, &session, &tool_call),
        ToolName::NotebookAddSource => notebook_add_source(deps, &call, &session, &tool_call),
        ToolName::NotebookRemoveSource => notebook_remove_source(deps, &session, &tool_call),
        ToolName::CreateChart => create_chart(deps, &call, &tool_call),
        ToolName::CreateDiagram => create_diagram(deps, &call, &tool_call),
        ToolName::CreatePdf => create_pdf(deps, &call, &tool_call),
        ToolName::CreateTable => create_table(deps, &call, &tool_call),
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
        deps.events.settle_effect(&call.run_id, key, &outcome);
    }

    // Every tool call, named, with what it was given and how it ended.
    //
    // This is the only place that sees all of them: the runner path, the agent
    // path and the refusals all funnel through here. Until it existed a tool
    // that quietly did nothing looked exactly like a tool that was never
    // called, and the difference could not be established from outside - which
    // is precisely the position `notebook.create` left us in.
    //
    // Argument *names* and the size of the result, never their contents: this
    // line goes to a log file, and the arguments carry document text, search
    // queries and file paths.
    {
        let mut keys: Vec<&str> = tool_call
            .arguments
            .as_object()
            .map(|fields| fields.keys().map(String::as_str).collect())
            .unwrap_or_default();
        keys.sort_unstable();
        match &outcome {
            Ok(answer) => log::info!(
                "[tool] run={} {} args=[{}] ok bytes={}",
                call.run_id,
                tool.as_str(),
                keys.join(","),
                answer.len()
            ),
            Err(problem) => log::warn!(
                "[tool] run={} {} args=[{}] failed: {problem}",
                call.run_id,
                tool.as_str(),
                keys.join(",")
            ),
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

/// Reads pages of a document attached to this run's conversation.
///
/// ## What this is for
///
/// The OCR budget decides how much of an attachment fits the window, and a
/// large document enters in part or not at all. Every page is stored as it is
/// read — see [`documents`] — and this is the only way back to the rest. Before
/// it existed, the pages the budget dropped were gone the moment the prompt
/// was composed, and the conversation could not recover them either: an
/// assistant's earlier answers contain what the model chose to write about the
/// pages it was shown, which by construction excludes the pages it was not.
///
/// ## The two things that scope it
///
/// **Who** comes from the session, never from the arguments. A model that could
/// name an owner could read another person's attachment by asking for it.
///
/// **Which conversation** comes from the run-to-conversation index, keyed by
/// this run's own id. A run whose conversation is not in the table reads
/// nothing, which is the safe direction: with no conversation there is no scope
/// to check a document against, and an unscoped read of a content-addressed
/// store is a read of every document on the machine.
///
/// Both are enforced inside [`documents::DocumentStore`], which returns the same
/// sentence for "no such document" as for "not yours" — so a refusal cannot be
/// used to discover what somebody else has attached.
fn read_attached_pages(
    deps: &Arc<RuntimeDeps>,
    call: &CallParams,
    session: &Session,
    tool_call: &ToolCall,
) -> Result<String, String> {
    let sha256 = tool_call.text("documentSha256").unwrap_or_default();
    if sha256.is_empty() {
        return Err(
            "documentSha256 is required. It is the id on the <attachment> tag of the document you \
             want to read."
                .to_string(),
        );
    }
    let from_page = tool_call.integer("fromPage").unwrap_or(1);
    // A single page is the common case, and asking for it should not require
    // saying the same number twice.
    let to_page = tool_call.integer("toPage").unwrap_or(from_page);

    let Some(conversation_id) = deps.run_to_conversation.lookup(&call.run_id) else {
        return Err(
            "This run is not attached to a conversation, so it cannot read a conversation's \
             documents."
                .to_string(),
        );
    };

    let read = deps.documents.pages(
        &sha256,
        &session.user.id,
        Some(&conversation_id),
        from_page,
        to_page,
    )?;

    Ok(render_pages(&read))
}

/// Finds a passage in this conversation's documents by what it says.
///
/// ## Why a model needs this and not only page ranges
///
/// After a turn that could afford a third of a forty-page scan, the prompt says
/// which pages were shown and which were not. `document.read_pages` can walk
/// the rest ten pages at a time, which is three calls of guessing for one
/// answer, and a model that guesses wrong twice tends to stop and apologise
/// instead. This answers the question it actually has.
///
/// ## Scope
///
/// The conversation this run belongs to, and the person signed in. Both, from
/// the same check every other read of the store makes — a document somebody
/// else attached, or one this person attached to a different thread, is not
/// searchable here and is not reported as existing.
fn search_attached_documents(
    deps: &Arc<RuntimeDeps>,
    call: &CallParams,
    session: &Session,
    tool_call: &ToolCall,
) -> Result<String, String> {
    let query = tool_call.text("query").unwrap_or_default();
    if query.trim().is_empty() {
        return Err(
            "query is required. Give the words you expect to find in the document — a tag \
             number, a clause title, a part name."
                .to_string(),
        );
    }
    let Some(conversation_id) = deps.run_to_conversation.lookup(&call.run_id) else {
        return Err(
            "This run is not attached to a conversation, so it cannot search a conversation's \
             documents."
                .to_string(),
        );
    };

    let found = deps.documents.search(
        &query,
        &session.user.id,
        &conversation_id,
        documents::MAX_SEARCH_HITS,
    )?;
    Ok(render_search(&found))
}

/// Reads a notebook graph and returns it drawn.
///
/// ## It reads, it does not extract
///
/// The three passes that build a graph - statistical, typing, relations - run
/// from the Notebooks screen, where a person can watch them and where spending
/// four minutes on a model is a thing they asked for. A chat turn calling this
/// gets whatever those passes have already produced. A graph built inside a turn
/// would make the turn take minutes, and would produce edges nothing else could
/// cite afterwards.
///
/// So an empty answer here is a real answer, and it says which pass has not run.
/// "No graph yet" and "a graph with no named relations" are different states
/// needing different actions, and a model told only "nothing found" would go
/// looking for a different notebook.
///
/// ## What comes back
///
/// A fenced Mermaid diagram, and the counts behind it. Fenced because the chat
/// surface draws a `mermaid` fence rather than printing it, and because a model
/// handed bare diagram source tends to reformat it. The diagram is the point - the
/// question this tool answers is "how do these connect", and prose is the worst
/// form for that answer.
/// Works out which notebook a turn is talking about.
///
/// Nobody says "draw notebook 9f2c4e". They say "the supplier contracts", or
/// they say nothing at all because there is only one. Requiring an opaque id
/// would mean the person has to go and find it, which is exactly the step a
/// chat interface exists to remove.
///
/// The ladder, in order, and it never guesses:
///
/// 1. no name given and the person has one notebook - that one,
/// 2. an exact id,
/// 3. an exact name, ignoring case,
/// 4. one name that contains the words given, or is contained by them, so
///    "supplier" finds "Supplier Contracts" and "my refinery notebook" finds
///    "Refinery",
/// 5. anything ambiguous or unmatched - the candidates, named, as an error the
///    model can act on in one more call.
///
/// Step 5 is the important one. Picking the first of three plausible matches
/// would draw a real graph of the wrong documents, and a wrong answer that
/// looks right is the worst outcome this tool can produce. Listing them costs
/// one round trip and cannot mislead.
fn resolve_notebook(
    deps: &Arc<RuntimeDeps>,
    session: &Session,
    query: Option<&str>,
) -> Result<crate::knowledge::Notebook, String> {
    let all = deps
        .notebooks
        .list(&session.user.id)
        .map_err(|error| format!("the notebooks could not be listed: {error}"))?;
    choose_notebook(&all, query)
}

/// The decision itself, with no store behind it.
///
/// Separated so the ladder can be tested directly. Which notebook somebody
/// meant is the part worth getting right, and it should not need a runtime, a
/// session and a SQLite file to exercise.
fn choose_notebook(
    all: &[crate::knowledge::Notebook],
    query: Option<&str>,
) -> Result<crate::knowledge::Notebook, String> {
    if all.is_empty() {
        return Err("There are no notebooks yet. Create one on the Notebooks screen, add \
                    documents to it and run Build graph; then there is a graph to draw."
            .to_string());
    }

    // Named so the same sentence can be produced from three different dead ends.
    let choices = || {
        all.iter()
            .map(|notebook| {
                format!(
                    "\"{}\" ({} document(s))",
                    notebook.name, notebook.document_count
                )
            })
            .collect::<Vec<_>>()
            .join(", ")
    };

    let Some(query) = query.map(str::trim).filter(|text| !text.is_empty()) else {
        // `list` returns most recently updated first, but that is not a strong
        // enough signal to choose on: the notebook someone last added a file to
        // is not necessarily the one they are asking about.
        if all.len() == 1 {
            return Ok(all[0].clone());
        }
        return Err(format!(
            "There are {} notebooks, so say which: {}. Pass its name as `notebook`.",
            all.len(),
            choices()
        ));
    };

    if let Some(found) = all.iter().find(|notebook| notebook.id == query) {
        return Ok(found.clone());
    }

    let wanted = query.to_lowercase();
    if let Some(found) = all
        .iter()
        .find(|notebook| notebook.name.to_lowercase() == wanted)
    {
        return Ok(found.clone());
    }

    let mut loose = all.iter().filter(|notebook| {
        let name = notebook.name.to_lowercase();
        name.contains(&wanted) || wanted.contains(&name)
    });
    match (loose.next(), loose.next()) {
        (Some(only), None) => Ok(only.clone()),
        (Some(_), Some(_)) => Err(format!(
            "\"{query}\" matches more than one notebook: {}. Say which one.",
            choices()
        )),
        _ => Err(format!(
            "No notebook is called \"{query}\". There is: {}.",
            choices()
        )),
    }
}

/// The notebooks, as a turn can see them.
///
/// The same names the system prompt already carries, returned as a tool result
/// so a turn that needs them mid-conversation - after creating one, say - does
/// not have to rely on a prompt composed before that happened.
fn notebook_list(deps: &Arc<RuntimeDeps>, session: &Session) -> Result<String, String> {
    let all = deps
        .notebooks
        .list(&session.user.id)
        .map_err(|error| format!("the notebooks could not be listed: {error}"))?;

    if all.is_empty() {
        return Ok("There are no notebooks yet.".to_string());
    }

    let mut answer = format!("{} notebook(s):\n", all.len());
    for notebook in &all {
        answer.push_str(&format!(
            "- \"{}\" - {} document(s), updated {}\n",
            notebook.name, notebook.document_count, notebook.updated_at
        ));
    }
    Ok(answer)
}

fn notebook_create(
    deps: &Arc<RuntimeDeps>,
    session: &Session,
    tool_call: &ToolCall,
) -> Result<String, String> {
    let name = tool_call.text("name").unwrap_or_default().trim();
    if name.is_empty() {
        return Err("a notebook needs a name; say what to call it".to_string());
    }

    // Refused rather than silently making a second one. Two notebooks with one
    // name is a state every later "which notebook did you mean" has to live
    // with, and it is created here or nowhere.
    let existing = deps
        .notebooks
        .list(&session.user.id)
        .map_err(|error| format!("the notebooks could not be listed: {error}"))?;
    if let Some(clash) = existing
        .iter()
        .find(|notebook| notebook.name.to_lowercase() == name.to_lowercase())
    {
        return Err(format!(
            "There is already a notebook called \"{}\". Use it, or choose another name.",
            clash.name
        ));
    }

    let made = deps
        .notebooks
        .create(&session.user.id, name)
        .map_err(|error| format!("the notebook could not be created: {error}"))?;
    Ok(format!(
        "Created the notebook \"{}\". It has no documents yet - add them from the Notebooks \
         screen, or attach a file to this conversation and ask for it to be put in.",
        made.name
    ))
}

fn notebook_rename(
    deps: &Arc<RuntimeDeps>,
    session: &Session,
    tool_call: &ToolCall,
) -> Result<String, String> {
    let notebook = resolve_notebook(deps, session, tool_call.text("notebook"))?;
    let name = tool_call.text("name").unwrap_or_default().trim();
    if name.is_empty() {
        return Err("say what the notebook should be called".to_string());
    }

    let was = notebook.name.clone();
    let renamed = deps
        .notebooks
        .rename(&notebook.id, &session.user.id, name)
        .map_err(|error| format!("the notebook could not be renamed: {error}"))?;
    Ok(format!("Renamed \"{was}\" to \"{}\".", renamed.name))
}

fn notebook_delete(
    deps: &Arc<RuntimeDeps>,
    session: &Session,
    tool_call: &ToolCall,
) -> Result<String, String> {
    let notebook = resolve_notebook(deps, session, tool_call.text("notebook"))?;

    deps.notebooks
        .delete(&notebook.id, &session.user.id)
        .map_err(|error| format!("the notebook could not be deleted: {error}"))?;
    Ok(format!(
        "Deleted the notebook \"{}\" and the graph built over it. Its {} document(s) are \
         untouched and still attached where they were.",
        notebook.name, notebook.document_count
    ))
}

fn notebook_sources(
    deps: &Arc<RuntimeDeps>,
    session: &Session,
    tool_call: &ToolCall,
) -> Result<String, String> {
    let notebook = resolve_notebook(deps, session, tool_call.text("notebook"))?;
    let documents = deps
        .notebooks
        .documents(&notebook.id, &session.user.id)
        .map_err(|error| format!("the sources could not be listed: {error}"))?;

    if documents.is_empty() {
        return Ok(format!("\"{}\" has no documents in it.", notebook.name));
    }

    let mut answer = format!(
        "\"{}\" holds {} document(s):\n",
        notebook.name,
        documents.len()
    );
    for document in &documents {
        // The sha is what `document.read_pages` and `notebook.remove_source`
        // both take, so it is named here rather than made a second lookup.
        answer.push_str(&format!(
            "- {} ({})\n",
            document.document_name, document.document_sha256
        ));
    }
    Ok(answer)
}

/// Puts a document already attached to this conversation into a notebook.
///
/// Scoped to the conversation's own attachments on purpose. The alternative -
/// letting a turn name any sha it likes - would make a tool that can pull a
/// document out of one conversation and into a notebook by guessing a hash,
/// which is not a capability a chat turn should have even for its own owner.
fn notebook_add_source(
    deps: &Arc<RuntimeDeps>,
    call: &CallParams,
    session: &Session,
    tool_call: &ToolCall,
) -> Result<String, String> {
    let notebook = resolve_notebook(deps, session, tool_call.text("notebook"))?;
    let wanted = tool_call
        .text("document")
        .unwrap_or_default()
        .trim()
        .to_string();
    if wanted.is_empty() {
        return Err("say which attached document to add, by its name".to_string());
    }

    // The run's conversation, looked up the way `search_attached_documents`
    // does: `CallParams` carries the run id, and the mapping to a conversation
    // lives in the runtime rather than on the call.
    let Some(conversation_id) = deps.run_to_conversation.lookup(&call.run_id) else {
        return Err(
            "This run is not attached to a conversation, so it has no attached documents to add."
                .to_string(),
        );
    };
    let attached = deps
        .documents
        .for_conversation(&session.user.id, &conversation_id)
        .map_err(|error| format!("this conversation's documents could not be read: {error}"))?;
    if attached.is_empty() {
        return Err(
            "nothing is attached to this conversation, so there is nothing to add".to_string(),
        );
    }

    let lowered = wanted.to_lowercase();
    let mut matches = attached.iter().filter(|document| {
        document.sha256 == wanted
            || document.name.to_lowercase() == lowered
            || document.name.to_lowercase().contains(&lowered)
    });
    let found = match (matches.next(), matches.next()) {
        (Some(only), None) => only,
        (Some(_), Some(_)) => {
            return Err(format!(
                "\"{wanted}\" matches more than one attached document. Name it exactly."
            ))
        }
        _ => {
            let names: Vec<&str> = attached
                .iter()
                .map(|document| document.name.as_str())
                .collect();
            return Err(format!(
                "Nothing attached here is called \"{wanted}\". Attached: {}.",
                names.join(", ")
            ));
        }
    };

    deps.notebooks
        .add_document(&notebook.id, &session.user.id, &found.sha256, &found.name)
        .map_err(|error| format!("the document could not be added: {error}"))?;
    Ok(format!(
        "Added \"{}\" to the notebook \"{}\". Run Build graph on the Notebooks screen to \
         include it in the graph.",
        found.name, notebook.name
    ))
}

fn notebook_remove_source(
    deps: &Arc<RuntimeDeps>,
    session: &Session,
    tool_call: &ToolCall,
) -> Result<String, String> {
    let notebook = resolve_notebook(deps, session, tool_call.text("notebook"))?;
    let wanted = tool_call
        .text("document")
        .unwrap_or_default()
        .trim()
        .to_string();
    if wanted.is_empty() {
        return Err("say which source to take out, by its name".to_string());
    }

    let documents = deps
        .notebooks
        .documents(&notebook.id, &session.user.id)
        .map_err(|error| format!("the sources could not be listed: {error}"))?;

    let lowered = wanted.to_lowercase();
    let mut matches = documents.iter().filter(|document| {
        document.document_sha256 == wanted
            || document.document_name.to_lowercase() == lowered
            || document.document_name.to_lowercase().contains(&lowered)
    });
    let found = match (matches.next(), matches.next()) {
        (Some(only), None) => only,
        (Some(_), Some(_)) => {
            return Err(format!(
                "\"{wanted}\" matches more than one source in \"{}\". Name it exactly.",
                notebook.name
            ))
        }
        _ => {
            return Err(format!(
                "\"{}\" has no source called \"{wanted}\".",
                notebook.name
            ))
        }
    };

    let name = found.document_name.clone();
    let sha = found.document_sha256.clone();
    deps.notebooks
        .remove_document(&notebook.id, &session.user.id, &sha)
        .map_err(|error| format!("the source could not be removed: {error}"))?;
    Ok(format!(
        "Took \"{name}\" out of \"{}\", along with the graph evidence that came from it. \
         The document itself is untouched.",
        notebook.name
    ))
}

/// Draws a chart of figures the run already has.
///
/// The SVG goes back in the tool result as well as to a file. A chart the
/// person has to open a file manager to see is a chart they will not look at,
/// and the chat surface draws an `svg` fence directly.
///
/// Series arrive as one per line, `Name: 1, 2, 3`. A model that has been given
/// a table can write that without being taught a schema, and the parse either
/// works or says which line it could not read - it never drops a series to make
/// the rest fit, because a chart missing a series still looks like a whole one.
/// Where a produced file goes, and under what name.
///
/// One place, so every producer names its output the same way and a person
/// reading an artifact list is not guessing which tool wrote which file.
fn artifact_path(
    deps: &Arc<RuntimeDeps>,
    call: &CallParams,
    title: &str,
    extension: &str,
) -> Result<(std::path::PathBuf, String), String> {
    let workspace = deps
        .root_for(&call.run_id)
        .ok_or_else(|| "This run has no workspace to write into.".to_string())?;
    let slug: String = title
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect();
    let stem = slug.trim_matches('-').to_string();
    let stem = if stem.is_empty() {
        "artifact".to_string()
    } else {
        stem
    };
    let name = format!("{stem}.{extension}");
    Ok((workspace.join(&name), name))
}

/// Draws a block, process or engineering diagram.
///
/// Blocks arrive one per line as `id | Label | shape | tag`, connections as
/// `from -> to : label`. Both are shapes a model can write from a description
/// without being taught a schema, and both refuse rather than guess: a
/// connection naming a block that was never declared is an error, because a
/// diagram silently missing a line says something false about the plant.
fn create_diagram(
    deps: &Arc<RuntimeDeps>,
    call: &CallParams,
    tool_call: &ToolCall,
) -> Result<String, String> {
    use crate::artifacts::diagram::{DiagramSpec, Direction, Edge, Node, Shape};

    let title = tool_call.text("title").unwrap_or_default().trim().to_string();
    let direction_text = tool_call.text("direction").unwrap_or_default();
    let direction = if direction_text.trim().is_empty() {
        Direction::Across
    } else {
        Direction::parse(direction_text).ok_or_else(|| {
            format!(
                "\"{direction_text}\" is not a direction. Use \"LR\" to read across or \
                 \"TD\" to read down."
            )
        })?
    };

    let mut nodes = Vec::new();
    for line in tool_call.text("blocks").unwrap_or_default().lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split('|').map(str::trim).collect();
        if parts.len() < 2 {
            return Err(format!(
                "Could not read the block \"{line}\". Each line is \
                 \"id | Label | shape | tag\", and shape and tag may be left off."
            ));
        }
        let shape = match parts.get(2).copied().filter(|s| !s.is_empty()) {
            Some(text) => Shape::parse(text).ok_or_else(|| {
                format!(
                    "\"{text}\" is not a shape. Use box, rounded, vessel, valve, instrument \
                     or decision."
                )
            })?,
            None => Shape::Box,
        };
        nodes.push(Node {
            id: parts[0].to_string(),
            label: parts[1].to_string(),
            shape,
            tag: parts
                .get(3)
                .map(|t| t.to_string())
                .filter(|t| !t.is_empty()),
        });
    }

    let mut edges = Vec::new();
    for line in tool_call.text("connections").unwrap_or_default().lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let (pair, label) = match line.split_once(':') {
            Some((pair, label)) => (pair, Some(label.trim().to_string())),
            None => (line, None),
        };
        let (from, to) = pair.split_once("->").ok_or_else(|| {
            format!("Could not read the connection \"{line}\". Each line is \"from -> to\".")
        })?;
        edges.push(Edge {
            from: from.trim().to_string(),
            to: to.trim().to_string(),
            label: label.filter(|l| !l.is_empty()),
        });
    }

    let svg = crate::artifacts::diagram::render_svg(&DiagramSpec {
        title: title.clone(),
        direction,
        nodes,
        edges,
    })?;

    let (path, name) = artifact_path(deps, call, &title, "svg")?;
    std::fs::write(&path, svg.as_bytes())
        .map_err(|error| format!("the diagram could not be written: {error}"))?;

    Ok(format!(
        "Drew \"{title}\" and saved it as {name}. Include the fence below in your reply so \
         the reader sees the diagram.\n\n```svg\n{svg}\n```"
    ))
}

/// Writes a PDF report or note.
///
/// Turns a tool's `body` into document blocks.
///
/// The grammar is markdown-ish on purpose: `#` for a heading, `-` for a bullet,
/// `|` for columns, anything else a paragraph. A model writes that without being
/// taught, and it keeps the tool from needing a structured document schema it
/// would then have to explain in a description that is already long.
///
/// ## Why fenced code is part of it
///
/// Because without it the grammar destroyed the one content people most often
/// ask to be put in a document. Every rule above is actively wrong inside a
/// program:
///
/// - `#include <iostream>` and every Python `#` comment became a **heading**;
/// - a line containing `|` — a pipe, a bitwise or, a table in a docstring —
///   was cut into padded columns;
/// - `line.trim()` deleted the indentation, which in Python *is* the program;
/// - blank lines were dropped, so statements ran together.
///
/// The file opened, and nothing said it was wrong. Worse, it was unfixable from
/// the model's side: there was no way to say "this part is verbatim", so a model
/// asked for a PDF of some code had no correct move available. Watching one
/// deliberate about it — "the body parameter should be a string that starts with
/// # for headings… but actually the body parameter expects a string, I'll just
/// put the entire code in a string with line breaks" — is what led here.
///
/// So a ``` fence turns the rules off until the closing fence. Inside one the
/// line is kept exactly as written apart from trailing whitespace, and is drawn
/// in the fixed-width face that already exists for table rows.
///
/// An unclosed fence runs to the end of the body. That is what the chat renderer
/// does with a half-arrived answer, and it is the safe reading here too: the
/// alternative is to apply the interpreting rules to the rest of the document,
/// which is the mangling this exists to prevent.
fn parse_document_body(body: &str) -> Vec<crate::artifacts::pdf::Block> {
    use crate::artifacts::pdf::Block;

    let mut blocks = Vec::new();
    let mut verbatim = false;

    for raw in body.lines() {
        let line = raw.trim();

        // The fence is a delimiter, never content, so it is not drawn.
        if line.starts_with("```") {
            verbatim = !verbatim;
            continue;
        }

        if verbatim {
            // Only trailing whitespace goes. Leading whitespace is the
            // program's structure, and a blank line inside a listing is a
            // deliberate separation rather than nothing.
            blocks.push(Block::Fixed(raw.trim_end().to_string()));
            continue;
        }

        if line.is_empty() {
            continue;
        }
        if let Some(rest) = line.strip_prefix('#') {
            blocks.push(Block::Heading(
                rest.trim_start_matches('#').trim().to_string(),
            ));
        } else if let Some(rest) = line.strip_prefix("- ") {
            blocks.push(Block::Bullet(rest.trim().to_string()));
        } else if line.contains('|') {
            // A row of a table keeps its columns lined up.
            blocks.push(Block::Fixed(
                line.split('|')
                    .map(|c| format!("{:<14}", c.trim()))
                    .collect::<String>()
                    .trim_end()
                    .to_string(),
            ));
        } else {
            blocks.push(Block::Paragraph(line.to_string()));
        }
    }

    blocks
}

/// Writes a PDF from a title, a classification banner and a body.
///
/// The body grammar, including its fenced-code rule, is [`parse_document_body`].
fn create_pdf(
    deps: &Arc<RuntimeDeps>,
    call: &CallParams,
    tool_call: &ToolCall,
) -> Result<String, String> {
    use crate::artifacts::pdf::{Block, PdfSpec};

    let title = tool_call.text("title").unwrap_or_default().trim().to_string();
    let blocks = parse_document_body(&tool_call.text("body").unwrap_or_default());

    let bytes = crate::artifacts::pdf::render(&PdfSpec {
        title: title.clone(),
        classification: tool_call
            .text("classification")
            .unwrap_or_default()
            .trim()
            .to_string(),
        blocks,
    })?;

    let (path, name) = artifact_path(deps, call, &title, "pdf")?;
    let size = bytes.len();
    std::fs::write(&path, bytes)
        .map_err(|error| format!("the PDF could not be written: {error}"))?;

    Ok(format!(
        "Wrote \"{title}\" as {name} ({size} bytes). It is in this run's artifacts, where it \
         can be previewed and opened."
    ))
}

/// Writes a table as a spreadsheet, and returns it for the chat to draw.
fn create_table(
    deps: &Arc<RuntimeDeps>,
    call: &CallParams,
    tool_call: &ToolCall,
) -> Result<String, String> {
    let title = tool_call.text("title").unwrap_or_default().trim().to_string();
    let header: Vec<String> = tool_call
        .text("header")
        .unwrap_or_default()
        .split('|')
        .map(|c| c.trim().to_string())
        .filter(|c| !c.is_empty())
        .collect();

    let mut rows: Vec<Vec<String>> = Vec::new();
    for line in tool_call.text("rows").unwrap_or_default().lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        rows.push(line.split('|').map(|c| c.trim().to_string()).collect());
    }

    let (path, name) = artifact_path(deps, call, &title, "xlsx")?;
    let classification = tool_call
        .text("classification")
        .unwrap_or_default()
        .trim()
        .to_string();
    crate::artifacts::xlsx::write_table(&path, &title, &header, &rows, &classification)?;

    // The same table as markdown, so the reader sees it in the reply rather
    // than only as a file they have to open.
    let mut markdown = String::new();
    markdown.push_str(&format!("| {} |\n", header.join(" | ")));
    markdown.push_str(&format!(
        "|{}|\n",
        header.iter().map(|_| " --- ").collect::<Vec<_>>().join("|")
    ));
    for row in &rows {
        markdown.push_str(&format!("| {} |\n", row.join(" | ")));
    }

    Ok(format!(
        "Wrote \"{title}\" as {name}. Include the table below in your reply.\n\n{markdown}"
    ))
}

fn create_chart(
    deps: &Arc<RuntimeDeps>,
    call: &CallParams,
    tool_call: &ToolCall,
) -> Result<String, String> {
    use crate::artifacts::chart::{ChartKind, ChartSpec, Series};

    let title = tool_call.text("title").unwrap_or_default().trim().to_string();
    let kind_text = tool_call.text("kind").unwrap_or_default();
    let kind = ChartKind::parse(kind_text).ok_or_else(|| {
        format!(
            "\"{kind_text}\" is not a chart kind this draws. Use \"bar\" for comparing \
             categories or \"line\" for something measured over an ordered axis."
        )
    })?;

    let categories: Vec<String> = tool_call
        .text("categories")
        .unwrap_or_default()
        .split(',')
        .map(|c| c.trim().to_string())
        .filter(|c| !c.is_empty())
        .collect();

    let mut series = Vec::new();
    for line in tool_call.text("series").unwrap_or_default().lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let (name, values) = line.split_once(':').ok_or_else(|| {
            format!(
                "Could not read the series \"{line}\". Each line is a name, a colon, then \
                 the values: \"Actual: 120, 96, 143\"."
            )
        })?;
        let mut parsed = Vec::new();
        for value in values.split(',') {
            let value = value.trim();
            if value.is_empty() {
                continue;
            }
            parsed.push(value.parse::<f64>().map_err(|_| {
                format!(
                    "\"{value}\" in series \"{}\" is not a number.",
                    name.trim()
                )
            })?);
        }
        series.push(Series {
            name: name.trim().to_string(),
            values: parsed,
        });
    }

    let spec = ChartSpec {
        kind,
        title: title.clone(),
        categories,
        series,
        value_label: tool_call
            .text("valueLabel")
            .unwrap_or_default()
            .trim()
            .to_string(),
    };

    // Every refusal comes from the renderer, which checks the data rather than
    // the arguments: a series with the wrong number of values is a fact about
    // the chart, not about the call.
    let svg = crate::artifacts::chart::render_svg(&spec)?;

    // Written into the run's own workspace, like every other artifact, so it
    // appears in the artifact list with preview and reveal.
    let workspace = deps
        .root_for(&call.run_id)
        .ok_or_else(|| "This run has no workspace to write a chart into.".to_string())?;
    let slug: String = title
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect();
    let name = format!("{}.svg", slug.trim_matches('-'));
    let path = workspace.join(&name);
    std::fs::write(&path, svg.as_bytes())
        .map_err(|error| format!("the chart could not be written: {error}"))?;

    Ok(format!(
        "Drew \"{title}\" and saved it as {name}. Include the fence below in your reply so \
         the reader sees the chart.\n\n```svg\n{svg}\n```"
    ))
}

fn build_document_graph(
    deps: &Arc<RuntimeDeps>,
    session: &Session,
    tool_call: &ToolCall,
) -> Result<String, String> {
    // `notebook` is what the catalogue asks for - a name in the person's own
    // words. `notebookId` is still read because a model that has seen one in an
    // earlier turn will pass it, and refusing a correct id to insist on a name
    // would be pedantry.
    let asked = tool_call
        .text("notebook")
        .or_else(|| tool_call.text("notebookId"))
        .unwrap_or_default();
    let notebook = resolve_notebook(deps, session, Some(asked))?;
    let notebook_id = notebook.id.clone();

    let document = tool_call.text("documentSha256").unwrap_or_default();
    let document = if document.trim().is_empty() {
        None
    } else {
        Some(document)
    };
    let focus = tool_call.text("focus").unwrap_or_default();
    let focus = if focus.trim().is_empty() {
        None
    } else {
        Some(focus)
    };

    let view = deps
        .notebooks
        .graph(
            notebook_id.trim(),
            &session.user.id,
            document.as_deref(),
            focus.as_deref(),
            2,
            1,
            true,
        )
        .map_err(|error| format!("the graph could not be read: {error}"))?;

    if view.nodes.is_empty() {
        return Ok(format!(
            "Notebook \"{}\" has no graph yet. Open the Notebooks screen and run Build graph \
             over its {} document(s) first; nothing is drawn from an unbuilt notebook.",
            notebook.name, notebook.document_count
        ));
    }

    let nodes: Vec<crate::knowledge::graph::render::RenderNode> = view
        .nodes
        .iter()
        .map(|node| (node.label.clone(), node.node_type.clone(), node.occurrences))
        .collect();
    let label_of = |id: &str| {
        view.nodes
            .iter()
            .find(|node| node.id == id)
            .map(|node| node.label.clone())
    };
    let edges: Vec<crate::knowledge::graph::render::RenderEdge> = view
        .edges
        .iter()
        .filter_map(|edge| {
            Some((
                label_of(&edge.source)?,
                label_of(&edge.target)?,
                edge.weight,
                edge.relation.clone(),
            ))
        })
        .collect();

    let named = view
        .edges
        .iter()
        .filter(|edge| edge.relation.is_some())
        .count();
    let diagram = crate::knowledge::graph::render::render_mermaid(&nodes, &edges);

    // The counts are stated because the diagram cannot state them. A picture of
    // forty nodes drawn from a graph of nine hundred looks like the whole
    // library unless something says otherwise.
    Ok(format!(
        "```mermaid\n{diagram}```\n\nShowing {} of {} terms and {} of {} links. {}",
        nodes.len(),
        view.total_terms,
        edges.len(),
        view.total_edges,
        if named == 0 {
            "No link is named: the relation pass has not run over this notebook, so a line \
             means only that two terms appear in the same passage. Do not describe an \
             unnamed link as a relationship."
        } else {
            "A solid arrow is a named relation read out of the documents; a dashed line \
             means only that two terms share a passage."
        }
    ))
}

/// Turns a search into the prose the model reads.
///
/// Every passage carries its document, page and heading trail, because that is
/// what makes it citable — and because a run that quotes a passage without
/// saying where it came from produces an answer nobody can check.
///
/// Finding nothing says so, and says how much was looked at. "No passage
/// matched across 128 sections of 2 documents" and "there were no documents to
/// search" are different facts, and a model told only the first will keep
/// rephrasing the query against a conversation that has no documents in it.
fn render_search(found: &documents::SearchOutcome) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    if found.hits.is_empty() {
        let _ = writeln!(
            out,
            "No passage matched \"{}\" across {} section(s) of {} document(s) attached to this \
             conversation.",
            found.query, found.chunks_searched, found.documents_searched
        );
        if found.documents_searched == 0 {
            out.push_str(
                "There are no documents attached to this conversation. Do not answer as though \
                 there were.\n",
            );
        } else {
            out.push_str(
                "Try different words, or read a page range with document.read_pages. Do not \
                 describe a passage you have not been shown.\n",
            );
        }
        return out;
    }

    let _ = writeln!(
        out,
        "{} passage(s) matching \"{}\", from {} section(s) of {} document(s):",
        found.hits.len(),
        found.query,
        found.chunks_searched,
        found.documents_searched
    );
    for hit in &found.hits {
        let where_from = if hit.section_path.is_empty() {
            format!("{} — page {}", hit.name, hit.page)
        } else {
            format!(
                "{} — {}, page {}",
                hit.name,
                hit.section_path.join(" › "),
                hit.page
            )
        };
        let _ = write!(out, "\n--- {where_from} ---\n{}\n", hit.text);
    }
    if found.truncated {
        out.push_str(
            "\nThis result reached its size limit before every match was included. Narrow the \
             query to see the rest.\n",
        );
    }
    out
}

/// Turns a page read into the prose the model reads.
///
/// Says what was found *and* what was not, on separate lines. A range that
/// silently returns four pages when five were asked for tells the model nothing
/// about the fifth — and "this page is blank" and "nobody could read this page"
/// lead to opposite conclusions about whether a clause exists.
fn render_pages(read: &documents::PageRead) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    let _ = writeln!(
        out,
        "{} — pages {}-{} of {}",
        read.name, read.from_page, read.to_page, read.pages
    );
    for page in &read.found {
        let _ = write!(out, "\n--- page {} of {} ---\n{}\n", page.page, read.pages, page.text);
    }
    if read.found.is_empty() {
        out.push_str("\nNo text was stored for any page in this range.\n");
    }
    if !read.unread.is_empty() {
        let listed = read
            .unread
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        let _ = write!(
            out,
            "\nNot returned: page(s) {listed}. These pages have no stored text, or this call \
             reached its size limit before them. Do not describe or quote a page that was not \
             returned; ask for it in a narrower range, or say it could not be read.\n"
        );
    }
    // Two different truncations, with two different remedies, so they are two
    // different sentences. Telling a model that a complete document was cut
    // short at extraction time — because *this call* hit its own size ceiling —
    // is false, and it withholds the one recovery that would have worked.
    if read.truncated {
        out.push_str(
            "\nThis call reached its size limit before the end of the range. Ask for a narrower \
             range to see the rest.\n",
        );
    }
    if read.source_truncated {
        out.push_str(
            "\nThis document was cut short when it was first read, so pages beyond that point may \
             not exist in the store at all. No narrower range will find them.\n",
        );
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
mod journey_tests;
#[cfg(test)]
mod large_document_tests;
#[cfg(test)]
mod memory_boundary_tests;
#[cfg(test)]
mod tests;

#[cfg(test)]
mod notebook_choice_tests {
    use super::choose_notebook;
    use crate::knowledge::Notebook;

    fn notebook(id: &str, name: &str) -> Notebook {
        Notebook {
            id: id.to_string(),
            name: name.to_string(),
            created_at: "2026-01-01T00:00:00Z".to_string(),
            updated_at: "2026-01-01T00:00:00Z".to_string(),
            document_count: 3,
        }
    }

    /// The whole point of the change: one notebook, no name given, no id to
    /// copy. "Draw the graph" has to work.
    #[test]
    fn one_notebook_needs_no_name_at_all() {
        let all = vec![notebook("nb-1", "Unit Four")];
        assert_eq!(choose_notebook(&all, None).unwrap().id, "nb-1");
        assert_eq!(choose_notebook(&all, Some("")).unwrap().id, "nb-1");
        assert_eq!(choose_notebook(&all, Some("   ")).unwrap().id, "nb-1");
    }

    #[test]
    fn a_name_is_matched_however_it_is_cased() {
        let all = vec![notebook("nb-1", "Supplier Contracts")];
        assert_eq!(
            choose_notebook(&all, Some("supplier contracts")).unwrap().id,
            "nb-1"
        );
    }

    /// What a person actually says. "the supplier one" and "my refinery
    /// notebook" both have to land, from either direction.
    #[test]
    fn a_partial_name_matches_from_either_direction() {
        let all = vec![
            notebook("nb-1", "Supplier Contracts"),
            notebook("nb-2", "Refinery"),
        ];

        assert_eq!(choose_notebook(&all, Some("supplier")).unwrap().id, "nb-1");
        assert_eq!(
            choose_notebook(&all, Some("my refinery notebook")).unwrap().id,
            "nb-2"
        );
    }

    #[test]
    fn an_id_still_works_for_a_model_that_has_one() {
        let all = vec![notebook("nb-1", "Unit Four"), notebook("nb-2", "Refinery")];
        assert_eq!(choose_notebook(&all, Some("nb-2")).unwrap().id, "nb-2");
    }

    /// The refusal that matters. Two plausible matches and a guess would draw a
    /// real graph of the wrong documents - a wrong answer that looks right.
    #[test]
    fn an_ambiguous_name_is_refused_and_names_the_candidates() {
        let all = vec![
            notebook("nb-1", "Supplier Contracts 2025"),
            notebook("nb-2", "Supplier Contracts 2026"),
        ];

        let problem = choose_notebook(&all, Some("supplier contracts")).unwrap_err();
        assert!(problem.contains("2025"), "{problem}");
        assert!(problem.contains("2026"), "{problem}");
    }

    #[test]
    fn several_notebooks_and_no_name_asks_rather_than_picking() {
        let all = vec![notebook("nb-1", "Unit Four"), notebook("nb-2", "Refinery")];

        let problem = choose_notebook(&all, None).unwrap_err();
        assert!(problem.contains("Unit Four"), "{problem}");
        assert!(problem.contains("Refinery"), "{problem}");
        // Never a bare id: the person has no way to use one.
        assert!(!problem.contains("nb-1"), "{problem}");
    }

    #[test]
    fn an_unknown_name_says_what_there_is() {
        let all = vec![notebook("nb-1", "Unit Four")];
        let problem = choose_notebook(&all, Some("payroll")).unwrap_err();
        assert!(problem.contains("Unit Four"), "{problem}");
    }

    #[test]
    fn no_notebooks_says_so_rather_than_failing_obscurely() {
        let problem = choose_notebook(&[], Some("anything")).unwrap_err();
        assert!(problem.contains("no notebooks"), "{problem}");
        assert!(problem.contains("Build graph"), "{problem}");
    }

}

/// The grammar every generated document's body is read with.
#[cfg(test)]
mod document_body_tests {
    use super::parse_document_body;

    /// Blocks in a comparable shape. `Block` carries no `PartialEq`, and giving
    /// it one for a test would be the test changing the type it is testing.
    fn described(body: &str) -> Vec<String> {
        use crate::artifacts::pdf::Block;
        parse_document_body(body)
            .iter()
            .map(|block| match block {
                Block::Heading(text) => format!("H:{text}"),
                Block::Paragraph(text) => format!("P:{text}"),
                Block::Bullet(text) => format!("B:{text}"),
                Block::Fixed(text) => format!("F:{text}"),
            })
            .collect()
    }

    /// Indentation is the program. Losing it is losing the code.
    #[test]
    fn fenced_python_keeps_its_indentation_comments_and_blank_lines() {
        let body = "```python\nclass Node:\n    # the payload\n    def __init__(self):\n\n        self.next = None\n```";

        assert_eq!(
            described(body),
            vec![
                "F:class Node:",
                "F:    # the payload",
                "F:    def __init__(self):",
                "F:",
                "F:        self.next = None",
            ],
            "inside a fence every line is verbatim: indentation kept, a # comment \
             is a comment and not a heading, and a blank line is a real line"
        );
    }

    /// The C++ case from the report: `#include` is not a heading.
    #[test]
    fn an_include_line_is_code_and_not_a_heading() {
        let lines = described("```cpp\n#include <iostream>\nint a = b | c;\n```");
        assert_eq!(lines, vec!["F:#include <iostream>", "F:int a = b | c;"]);
        assert!(
            !lines.iter().any(|line| line.starts_with("H:")),
            "an include line became a document heading"
        );
    }

    /// The prose grammar is untouched outside a fence.
    #[test]
    fn prose_outside_a_fence_still_reads_as_before() {
        assert_eq!(
            described("# Findings\n- The valve was replaced.\nOrdinary prose.\nItem | Qty"),
            vec![
                "H:Findings",
                "B:The valve was replaced.",
                "P:Ordinary prose.",
                "F:Item          Qty",
            ]
        );
    }

    /// A fence the model never closed must not hand the rest of the listing
    /// back to the interpreting rules — that is the mangling, arriving late.
    #[test]
    fn an_unclosed_fence_runs_to_the_end_rather_than_reverting() {
        assert_eq!(
            described("Intro paragraph.\n```py\n# not a heading\n    indented"),
            vec!["P:Intro paragraph.", "F:# not a heading", "F:    indented"]
        );
    }

    /// Prose, then code, then prose again.
    #[test]
    fn the_fence_closes_and_the_prose_rules_come_back() {
        assert_eq!(
            described("# Title\n```\n# code\n```\n# Heading again"),
            vec!["H:Title", "F:# code", "H:Heading again"]
        );
    }
}
