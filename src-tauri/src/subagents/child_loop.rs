//! A child's own model loop, on the runtime the parent is already using.
//!
//! ## Why a child can share the parent's runtime
//!
//! Because the Node side was already built for it, which is worth stating
//! plainly rather than assuming. `peer.ts` dispatches with
//! `this.#dispatch(frame).catch(...)` — *not* awaited — so a second request is
//! served while the first is still in flight. And `main.ts` keeps its runs in
//! `active: Map<runId, run>` and calls `startRun` per request, so two runs are
//! two entries rather than one shared object.
//!
//! So a child needs no new process, no second bundle and no change on the far
//! side. It needs a run id of its own, a plan registered against it, and a model
//! endpoint. That is what this assembles.
//!
//! ## The plan is the enforcement, not a formality
//!
//! Every RPC method the runtime can call checks `has_registered_plan`, and
//! `tool_catalogue` offers exactly `budget.permitted_tools`. So registering the
//! child's plan with its already-narrowed [`EffectivePolicy`] tool set is what
//! makes the gateway refuse anything wider — the child's model is *offered* only
//! what its profile asked for and its parent held, and a call outside that set
//! is refused by the same code that refuses the parent.
//!
//! That is the whole security argument for letting a child drive a model at all:
//! the loop is untrusted, and it is untrusted inside a plan it cannot edit.
//!
//! ## What the child is told, and what it is not
//!
//! Its system prompt is the profile's own Markdown body — see
//! `AgentProfile::instructions` — plus the objective, the deliverable and the
//! *references* from the packet. `history` is deliberately empty: a child does
//! not receive the parent's transcript, which is the property
//! `subagents::packet` exists to hold and which a shared runtime would otherwise
//! make easy to break.
//!
//! ## Findings come from receipts, not from prose
//!
//! The loop returns text; this returns the **tool calls the child actually
//! made**, read from the run-tool-call table that `agent_runtime` writes as each
//! call settles. A worker builds its findings from those. The difference matters
//! for the reason `certification`'s header gives: a model asked to summarise a
//! search it did not finish will summarise it anyway, and a receipt cannot.

use std::path::PathBuf;
use std::sync::Arc;

use serde_json::json;

use crate::agent_runtime::tasks::ToolCallRecord;
use crate::agent_runtime::workspace::Workspace;
use crate::agent_runtime::AgentRuntime;
use crate::orchestrator::plan::{Budget, PlanRun};
use crate::registry::ModelRegistry;
use crate::serving::ModelServers;

use super::inherit::EffectivePolicy;
use super::packet::{ChildTaskPacket, InputRef};

/// How many times one child may repeat the same call before the plan stops it.
///
/// Two. A worker that asks the same question three times has stopped making
/// progress, and a bounded child is the whole point of the role.
const REPEAT_LIMIT: u32 = 2;

/// What a child's own loop did.
#[derive(Debug, Clone)]
pub struct LoopReport {
    /// How the runtime said the run ended, in its own words.
    pub outcome: String,
    /// Whether the runtime reported an ending this side reads as success.
    ///
    /// Never inferred from the payload being non-empty — see
    /// `RunOutcome::from_runtime`, which exists because a runtime that answered
    /// without saying how a run ended must not be read as success.
    pub completed: bool,
    /// Every tool call the child actually made, as the gateway settled them.
    /// This is what a worker's findings are built from.
    pub tool_calls: Vec<ToolCallRecord>,
    /// The model it ran on, and where.
    pub model_id: String,
    pub base_url: String,
    /// The window the server was actually started with.
    pub served_window: u32,
    /// The passages this child retrieved, as the index returned them.
    pub passages: Vec<crate::knowledge::SearchResult>,
}

impl LoopReport {
    /// The calls that succeeded, which are the only ones a finding may rest on.
    pub fn receipts(&self) -> impl Iterator<Item = &ToolCallRecord> {
        self.tool_calls.iter().filter(|call| {
            matches!(
                call.outcome,
                crate::agent_runtime::tasks::CallOutcome::Succeeded
            )
        })
    }
}

/// Everything needed to run one child through the shared runtime.
pub struct ChildLoop {
    /// The lazily-spawned runtime the parent is using. Empty before any run has
    /// started, which is the one case a child cannot use a model.
    pub runtime: crate::commands::agent::AgentRuntimeHandle,
    pub plans: crate::commands::agent::RunPlans,
    pub calls: crate::commands::agent::RunToolCalls,
    pub workspaces: crate::commands::agent::RunWorkspaces,
    /// The passages each run retrieved, in the order its markers refer to.
    ///
    /// A child's evidence comes from here rather than from its prose: these are
    /// what `search_documents` actually returned, recorded by the gateway as it
    /// returned them. A finding built from one cites a document and a page that
    /// exist; a finding built from the model's text cites whatever it wrote.
    pub passages: crate::agent_runtime::retrieval::RunPassages,
    pub servers: Arc<ModelServers>,
    pub registry: Arc<ModelRegistry>,
    pub models_dir: PathBuf,
    /// The one lease service. The worker holds the child's lease around this
    /// loop; the loop asks the same service to serve the model, so admission
    /// happens only under that lease — and the child's own model rounds, which
    /// go through `context.refresh`, re-enter it rather than queue behind it.
    pub scheduler: Arc<super::scheduling::ModelScheduler>,
}

impl ChildLoop {
    /// Whether a model loop can be run at all right now.
    ///
    /// `false` before the runtime has started — which on this product means
    /// before any run has been driven, because the runtime is spawned lazily by
    /// `commands::agent::runtime`.
    pub fn available(&self) -> bool {
        self.runtime
            .lock()
            .map(|slot| slot.is_some())
            .unwrap_or(false)
    }

    fn runtime(&self) -> Option<Arc<AgentRuntime>> {
        self.runtime.lock().ok()?.clone()
    }

    /// Runs the child, and cleans up whatever it registered.
    ///
    /// The clean-up is unconditional. A plan left behind would let anything
    /// holding that run id keep calling tools under it, and a workspace left
    /// behind would leave a directory nothing owns.
    pub async fn run(
        &self,
        packet: &ChildTaskPacket,
        policy: &EffectivePolicy,
        instructions: &str,
    ) -> Result<LoopReport, String> {
        let runtime = self.runtime().ok_or(
            "the agent runtime has not started on this machine, so a worker cannot run a model \
             loop. Nothing was done.",
        )?;
        let model_id = packet.model_id.as_deref().ok_or(
            "this worker was not routed to a model, so it cannot run a model loop. Nothing was \
             done.",
        )?;
        let entry = self
            .registry
            .find(model_id)
            .cloned()
            .ok_or_else(|| format!("{model_id} is not in the model registry on this machine."))?;

        // The endpoint, served under the lease the worker already holds for
        // this child. Loading without it could evict a model in the middle of
        // somebody else's generation, which is why `ensure_served` refuses an
        // owner that does not hold the card.
        let rebind = match self.scheduler.ensure_served(&packet.child_id, &entry.id).await {
            Ok(rebind) => rebind,
            Err(crate::subagents::scheduling::SchedulingRefusal::LoadFailed { failure }) => {
                // Honest about what happens next: the job's model was pinned at
                // dispatch and nothing else is loaded in its place.
                let decision = crate::serving::fallback::decide(
                    &entry,
                    &failure,
                    crate::serving::fallback::RunPhase::PinnedJob,
                    entry.roles.first().copied().unwrap_or(crate::registry::ModelRole::Reasoning),
                    self.registry.all(),
                );
                return Err(decision.because().to_string());
            }
            Err(refusal) => return Err(refusal.explain()),
        };
        let served_window = rebind.served_window.max(1);

        // The child's own model rounds lease as this child, so they re-enter
        // the worker's hold instead of waiting behind it.
        self.scheduler.bind_run(
            &packet.child_id,
            crate::subagents::scheduling::RunBinding {
                model_id: entry.id.clone(),
                class: crate::subagents::scheduling::LeaseClass::Child,
                parent: (!packet.parent_run_id.is_empty()).then(|| packet.parent_run_id.clone()),
                eligible: packet
                    .model_policy
                    .as_ref()
                    .map(|policy| policy.eligible_model_ids.clone())
                    .unwrap_or_default(),
            },
        );

        // Registered before the loop is asked for anything, and removed however
        // this returns.
        let _registered = self.register(packet, policy)?;

        let deadline_ms = (packet.deadline - chrono::Utc::now())
            .num_milliseconds()
            .max(1_000) as u64;

        let params = json!({
            "runId": packet.child_id,
            // A child's message id is its own. The chat surface never sees it —
            // nothing routes a child's tokens to a cell — but the runtime
            // refuses a request without one.
            "messageId": packet.child_id,
            "attemptId": packet.child_id,
            "agentId": packet.agent_id,
            // The definition this child was dispatched under, so the per-round
            // context boundary compiles for the pinned version and not for
            // "version zero". An unpinned bundled profile has none, and says so.
            "definitionVersion": packet.definition_version.unwrap_or(0),
            // What this child does, so an activated procedure scoped to a
            // capability reaches exactly the children it was written for.
            "capability": packet.capability,
            "prompt": objective_prompt(packet),
            "systemPrompt": system_prompt(packet, policy, instructions),
            // Deliberately empty. A child receives an objective and references,
            // never the parent's transcript — see `subagents::packet`. Sharing
            // the runtime makes this easy to get wrong, so it is written out
            // rather than omitted.
            "history": [],
            "historyDropped": 0,
            "model": {
                "id": rebind.served_model_id,
                "provider": provider_label(entry.runtime),
                "baseUrl": rebind.base_url,
                "contextWindow": served_window,
                "maxTokens": policy.limits.max_output_tokens,
                "reasoning": false,
                "hasReasoningToggle": false,
            },
            "deadlineMs": deadline_ms,
        });

        let budget = std::time::Duration::from_millis(deadline_ms);
        let answered = tokio::time::timeout(budget, runtime.request("run.start", params)).await;

        // Whatever happened, the calls the child actually made are the record.
        // Read before the error is propagated, so a child that failed half way
        // still reports the three searches it did finish.
        let tool_calls = self
            .calls
            .lock()
            .ok()
            .and_then(|table| table.get(&packet.child_id).cloned())
            .unwrap_or_default();
        let passages = self
            .passages
            .lock()
            .ok()
            .and_then(|table| table.get(&packet.child_id).cloned())
            .unwrap_or_default();

        let (outcome, completed) = match answered {
            Ok(Ok(value)) => match crate::agent_runtime::outcome::RunOutcome::from_runtime(&value) {
                Some(reported) => {
                    let completed = matches!(
                        reported,
                        crate::agent_runtime::outcome::RunOutcome::Completed
                    );
                    (
                        reported
                            .detail()
                            .map(str::to_string)
                            .unwrap_or_else(|| "it finished".to_string()),
                        completed,
                    )
                }
                // A runtime that answered without saying how the run ended is
                // one this build cannot read, and calling that success is the
                // defect `RunOutcome` exists to remove.
                None => (
                    "the runtime finished without saying how the run ended".to_string(),
                    false,
                ),
            },
            Ok(Err(error)) => (error.to_string(), false),
            Err(_) => {
                // Told to stop. The deadline expiring here does not reach the
                // loop on its own, and a child left running would hold a model
                // server for work nobody is waiting for.
                let _ = runtime
                    .request(
                        "run.abort",
                        json!({
                            "runId": packet.child_id,
                            "reason": "the child's deadline passed",
                        }),
                    )
                    .await;
                (
                    format!(
                        "it reached its {} second limit and was stopped",
                        policy.limits.max_duration_seconds
                    ),
                    false,
                )
            }
        };

        Ok(LoopReport {
            outcome,
            completed,
            tool_calls,
            model_id: entry.id,
            base_url: rebind.base_url,
            served_window,
            passages,
        })
    }

    /// Registers the plan and the workspace this child runs under.
    ///
    /// Returns a guard that removes both. A plan outliving its child would let
    /// anything holding that run id keep calling tools under a budget nobody is
    /// watching.
    fn register(
        &self,
        packet: &ChildTaskPacket,
        policy: &EffectivePolicy,
    ) -> Result<Registered<'_>, String> {
        let budget = Budget {
            max_steps: policy.limits.max_turns,
            max_duration: std::time::Duration::from_secs(policy.limits.max_duration_seconds.max(1)),
            // The narrowed set, and nothing else. This is what `tool_catalogue`
            // offers the child's model and what `execute` authorises against —
            // so a profile asking for a tool its parent does not hold is
            // refused by the same gateway that refuses the parent, rather than
            // by a check written here.
            permitted_tools: policy.tools.clone(),
            repeat_limit: REPEAT_LIMIT,
        };
        self.plans
            .lock()
            .map_err(|_| "the plan table is poisoned".to_string())?
            .insert(
                packet.child_id.clone(),
                PlanRun::new(&packet.child_id, vec![packet.objective.clone()], budget),
            );

        // The run's own directory, not a new one. A child writes under its
        // parent's workspace or nowhere — `InheritedPolicy::workspace_root` is
        // where that is decided, and this only makes it reachable by run id.
        if let Ok(mut workspaces) = self.workspaces.lock() {
            workspaces.insert(
                packet.child_id.clone(),
                Workspace::at(policy.inherited.workspace_root.clone()),
            );
        }

        Ok(Registered {
            owner: self,
            child_id: packet.child_id.clone(),
        })
    }
}

/// Removes what a child registered, however its loop ended.
struct Registered<'a> {
    owner: &'a ChildLoop,
    child_id: String,
}

impl Drop for Registered<'_> {
    fn drop(&mut self) {
        if let Ok(mut plans) = self.owner.plans.lock() {
            plans.remove(&self.child_id);
        }
        if let Ok(mut workspaces) = self.owner.workspaces.lock() {
            workspaces.remove(&self.child_id);
        }
        // The passages were read into the report before this runs; keeping them
        // would be an unbounded table keyed by a run id nothing will ask for
        // again.
        if let Ok(mut passages) = self.owner.passages.lock() {
            passages.remove(&self.child_id);
        }
        // The binding goes; the lease stays with the worker that took it, and
        // is released when the worker's guard drops.
        self.owner.scheduler.unbind_run(&self.child_id);
    }
}

/// What the child is asked to do.
///
/// The objective and the references, and nothing else. Every input is named
/// rather than resolved: the child fetches what it needs itself, under its own
/// clearance, through the same gateway.
fn objective_prompt(packet: &ChildTaskPacket) -> String {
    let mut out = packet.objective.clone();
    if !packet.deliverable.trim().is_empty() {
        out.push_str("\n\nWhat to hand back: ");
        out.push_str(packet.deliverable.trim());
    }
    if !packet.inputs.is_empty() {
        out.push_str("\n\nYou have been pointed at these. Fetch each one yourself:\n");
        for input in &packet.inputs {
            out.push_str("- ");
            out.push_str(&input.describe());
            if let InputRef::Expression { expression } = input {
                out.push_str(&format!(" (check this: {expression})"));
            }
            out.push('\n');
        }
    }
    out
}

/// The child's own instructions, and the bounds it is held to.
///
/// The profile's body first, because that is what the role *is*. The bounds
/// after it, stated so the model is not surprised by a refusal — the refusals
/// still happen in the gateway whatever this says, and a model that knows its
/// limits spends fewer turns discovering them.
fn system_prompt(packet: &ChildTaskPacket, policy: &EffectivePolicy, instructions: &str) -> String {
    let tools = policy
        .tools
        .iter()
        .map(|tool| tool.as_str())
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "{instructions}\n\n\
         --- The bounds of this task ---\n\
         You are a worker on one bounded piece of a larger task. You do not have the parent's \
         conversation and you do not need it.\n\
         Tools you may call: {tools}. Anything else is refused.\n\
         You have at most {turns} turn(s) and must finish within {seconds} second(s).\n\
         Return what you actually found. If you found nothing, say so — that is an answer. Do \
         not describe work you did not do, and do not cite a source you did not retrieve.\n\
         The shape expected of you is: {schema}.",
        turns = policy.limits.max_turns,
        seconds = policy.limits.max_duration_seconds,
        schema = packet.required_schema.as_str(),
    )
}

fn provider_label(runtime: crate::registry::Runtime) -> &'static str {
    match runtime {
        crate::registry::Runtime::LlamaCpp => "llama.cpp",
        _ => "openai-compatible",
    }
}
