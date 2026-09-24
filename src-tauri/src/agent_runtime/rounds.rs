//! The boundary every model round passes through, in the order it has to.
//!
//! ## Why a round has a boundary of its own
//!
//! A model round is the one moment a run needs the GPU, the one moment its
//! context has to be current, and the one moment its request has to fit. Before
//! this module those were three unrelated facts decided in three places — or,
//! for the first two, not decided at all: the parent run held no lease, its
//! endpoint was fixed at run start even if a child's admission had since
//! stopped the server, and the "does it fit" question was answered by a
//! characters-over-four estimate of a request nobody had rendered.
//!
//! So the runtime now asks three things of Rust around every model call:
//!
//! | RPC | When | What Rust does |
//! |---|---|---|
//! | `context.refresh` | before the round, before compaction | takes the one heavy-call lease, re-binds the model endpoint (starting the server again if it was stopped to make room), compiles the four-scope context against the window the server *actually* has, refuses outright if the mandatory state cannot fit, and records the manifest |
//! | `context.count` | with the exact outgoing request | renders it through the served model's own template, counts it with its own tokenizer, accounts for images, checks the result against the served window less the generation and safety reserves, checks every authorised block reached the wire, and refuses a request that does not fit |
//! | `context.settle` | when the round's stream ends | gives the lease back and reconciles the count against what the server reported, learning a model's per-image cost from the difference |
//!
//! A tool call is a second, independent release point (`tool.authorize`): a
//! round that issued tool calls has finished generating, so the card is given
//! back before any tool — a delegation to a child in particular — runs.
//!
//! ## What is refused, and what degrades
//!
//! A round that cannot get the card, whose model cannot be served, whose
//! mandatory state does not fit, or whose rendered request does not fit is
//! refused with [`code::ROUND_REFUSED`] and the numbers somebody needs to act.
//! The runtime fails the run on that code rather than calling the model anyway.
//! Everything else — no graph on this deployment, a server that cannot render
//! for counting — degrades, and the record says exactly how.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use serde_json::{json, Value};

use super::context_compiler::{
    record_projection, ContextBlock, ContextCompiler, FrozenScope, Reserves,
};
use super::context_manifest::{ContextManifest, HistoryBinding};
use super::events::TaskEventType;
use super::protocol::{code, WireError};
use super::RuntimeDeps;
use crate::subagents::scheduling::{LeaseRequest, SchedulingRefusal};

// ─────────────────────────────────────────────────────────────────────────────
// Counting what the server will count
// ─────────────────────────────────────────────────────────────────────────────

/// Renders and counts a request the way the served model will.
///
/// A trait so a test can supply two tokenizers of different density and show a
/// request that fits one model's window and overflows another's.
#[async_trait::async_trait]
pub trait RenderedCounter: Send + Sync {
    /// Tokens in the request as rendered by the model's own chat template,
    /// tool schemas included, image parts excluded.
    async fn count(&self, base_url: &str, payload: &Value) -> Result<u32, String>;
}

/// The served model's own `/apply-template` and `/tokenize`.
pub struct ServerCounter;

#[async_trait::async_trait]
impl RenderedCounter for ServerCounter {
    async fn count(&self, base_url: &str, payload: &Value) -> Result<u32, String> {
        crate::serving::probe::count_rendered(base_url, payload).await
    }
}

/// What one run's current round was compiled against.
#[derive(Debug, Clone, Default)]
pub struct RoundState {
    /// The registry id of the model the round leases.
    pub model_id: String,
    pub served_model_id: String,
    pub base_url: Option<String>,
    pub served_window: u32,
    pub reserved_output: u32,
    pub reserved_safety: u32,
    pub manifest: Option<ContextManifest>,
    pub blocks: Vec<ContextBlock>,
    /// The rendered text tokens of the last counted request, for settling.
    pub counted_text: Option<u32>,
    pub images: u32,
    /// How many rounds this run has opened.
    pub rounds: u32,
}

/// Per-run round state, and what this process has measured about images.
pub struct RoundBook {
    rounds: Mutex<HashMap<String, RoundState>>,
    /// Measured tokens per image, by served model. Learned only from a server's
    /// own usage report; never assumed.
    image_costs: Mutex<HashMap<String, u32>>,
    counter: Arc<dyn RenderedCounter>,
}

impl Default for RoundBook {
    fn default() -> Self {
        Self::with_counter(Arc::new(ServerCounter))
    }
}

impl RoundBook {
    pub fn with_counter(counter: Arc<dyn RenderedCounter>) -> Self {
        Self {
            rounds: Mutex::new(HashMap::new()),
            image_costs: Mutex::new(HashMap::new()),
            counter,
        }
    }

    pub fn state(&self, run_id: &str) -> Option<RoundState> {
        self.rounds.lock().ok()?.get(run_id).cloned()
    }

    fn put(&self, run_id: &str, state: RoundState) {
        if let Ok(mut rounds) = self.rounds.lock() {
            rounds.insert(run_id.to_string(), state);
        }
    }

    /// Drops the state of every run that is no longer live. Called as each
    /// round opens, so the table holds only runs a plan still exists for — a
    /// finished run's plan is removed when it ends.
    fn prune(&self, live: impl Fn(&str) -> bool) {
        if let Ok(mut rounds) = self.rounds.lock() {
            rounds.retain(|run_id, _| live(run_id));
        }
    }

    /// Drops a finished run's state.
    pub fn forget(&self, run_id: &str) {
        if let Ok(mut rounds) = self.rounds.lock() {
            rounds.remove(run_id);
        }
    }

    pub fn image_cost(&self, served_model_id: &str) -> Option<u32> {
        self.image_costs.lock().ok()?.get(served_model_id).copied()
    }

    fn learn_image_cost(&self, served_model_id: &str, per_image: u32) {
        if let Ok(mut costs) = self.image_costs.lock() {
            costs.insert(served_model_id.to_string(), per_image);
        }
    }
}

/// How many image parts a chat payload carries.
pub fn images_in(payload: &Value) -> u32 {
    payload
        .get("messages")
        .and_then(Value::as_array)
        .map(|messages| {
            messages
                .iter()
                .filter_map(|message| message.get("content").and_then(Value::as_array))
                .flatten()
                .filter(|part| {
                    matches!(
                        part.get("type").and_then(Value::as_str),
                        Some("image_url") | Some("input_image") | Some("image")
                    )
                })
                .count() as u32
        })
        .unwrap_or(0)
}

/// Every message's text, one message per line, for the projection check.
pub fn text_of(payload: &Value) -> String {
    let Some(messages) = payload.get("messages").and_then(Value::as_array) else {
        return String::new();
    };
    let mut out = String::new();
    for message in messages {
        match message.get("content") {
            Some(Value::String(text)) => out.push_str(text),
            Some(Value::Array(parts)) => {
                for part in parts {
                    if let Some(text) = part.get("text").and_then(Value::as_str) {
                        out.push_str(text);
                        out.push('\n');
                    }
                }
            }
            _ => {}
        }
        out.push('\n');
    }
    out
}

/// Whether a counted request fits the window it is going to.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "camelCase", tag = "fit")]
pub enum Fit {
    /// Counted, images included, and within the limit.
    Fits { total: u32, limit: u32 },
    /// Counted and over. The round is refused.
    Overflows { total: u32, limit: u32, over: u32 },
    /// The text fits; images are present whose cost this model has not been
    /// measured for yet. Allowed, and recorded as not fully counted — the
    /// server's usage report after the round supplies the measurement.
    ImagesUnmeasured { text: u32, limit: u32, images: u32 },
    /// The server could not render the request for counting.
    Uncounted { because: String },
}

impl Fit {
    pub fn counted_by(&self) -> String {
        match self {
            Fit::Fits { .. } | Fit::Overflows { .. } => "tokenizer".into(),
            Fit::ImagesUnmeasured { images, .. } => {
                format!("tokenizer; {images} image(s) not yet measured for this model")
            }
            Fit::Uncounted { .. } => "uncounted".into(),
        }
    }
}

/// The limit a rendered request must fit: the served window less what the
/// model's own reply needs and the safety reserve. Framing and tool schemas are
/// *in* the rendered count, so they are not subtracted a second time.
pub fn judge(
    text: Result<u32, String>,
    images: u32,
    image_cost: Option<u32>,
    served_window: u32,
    max_output: u32,
    safety: u32,
) -> Fit {
    let limit = served_window.saturating_sub(max_output).saturating_sub(safety);
    let text = match text {
        Ok(text) => text,
        Err(because) => return Fit::Uncounted { because },
    };
    if text > limit {
        return Fit::Overflows {
            total: text,
            limit,
            over: text - limit,
        };
    }
    if images == 0 {
        return Fit::Fits { total: text, limit };
    }
    match image_cost {
        Some(per_image) => {
            let total = text.saturating_add(per_image.saturating_mul(images));
            if total > limit {
                Fit::Overflows {
                    total,
                    limit,
                    over: total - limit,
                }
            } else {
                Fit::Fits { total, limit }
            }
        }
        None => Fit::ImagesUnmeasured {
            text,
            limit,
            images,
        },
    }
}

fn round_refused(detail: String) -> WireError {
    WireError::new(code::ROUND_REFUSED, detail)
}

// ─────────────────────────────────────────────────────────────────────────────
// context.refresh
// ─────────────────────────────────────────────────────────────────────────────

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct RefreshRequest {
    run_id: String,
    task_id: String,
    agent_id: String,
    #[serde(default)]
    definition_version: u64,
    model_id: String,
    served_window: u32,
    #[serde(default)]
    question: String,
    /// Content hashes the round already carries. A block whose hash is here is
    /// not injected a second time.
    #[serde(default)]
    already_carried: Vec<String>,
    #[serde(default)]
    project_id: Option<String>,
    #[serde(default)]
    template_id: Option<String>,
    #[serde(default)]
    capability: Option<String>,
    #[serde(default)]
    reserved_tool_schemas: u32,
    #[serde(default)]
    reserved_output: u32,
    #[serde(default)]
    reserved_framing: u32,
    #[serde(default)]
    reserved_safety: u32,
}

/// Opens one model round. See the module header for the order.
pub(super) async fn refresh(params: Value, deps: &Arc<RuntimeDeps>) -> Result<Value, WireError> {
    let request: RefreshRequest = serde_json::from_value(params).map_err(|error| {
        WireError::new(
            code::BAD_PARAMS,
            format!("context.refresh needs a typed request: {error}"),
        )
    })?;

    // The same gate every other method here passes.
    if !super::has_registered_plan(deps, &request.run_id) {
        return Err(super::no_plan_error(&request.run_id));
    }
    let run_id = request.run_id.clone();

    // ── The card, and the endpoint behind it ─────────────────────────────
    let leases = deps.leases.as_ref().and_then(|leases| {
        leases
            .binding(&run_id)
            .map(|binding| (Arc::clone(leases), binding))
    });
    let (lease, rebind) = match leases {
        Some((leases, binding)) => {
            let cancel = deps
                .cancellations
                .as_ref()
                .and_then(|table| table.token(&run_id))
                .unwrap_or_default();
            let mut ask = LeaseRequest::new(run_id.as_str(), binding.class, binding.model_id.as_str())
                .eligible(binding.eligible.clone())
                .cancelled_by(cancel);
            if let Some(parent) = &binding.parent {
                ask = ask.child_of(parent.as_str());
            }
            let admitted = leases.acquire_round(ask).await.map_err(|refusal| {
                round_refused(format!(
                    "This round was not given the GPU, so the model was not called: {}",
                    refusal.explain()
                ))
            })?;
            match leases.ensure_served(&run_id, &binding.model_id).await {
                Ok(rebind) => {
                    let lease = json!({
                        "status": "held",
                        "class": binding.class,
                        "grant": admitted.grant,
                        "waitedMs": admitted.waited.as_millis() as u64,
                        "reentered": admitted.reentered,
                        "resumedAfterMs": admitted.resumed_after.map(|after| after.as_millis() as u64),
                        "suspended": admitted.suspended,
                    });
                    (lease, Some(rebind))
                }
                Err(SchedulingRefusal::LoadFailed { failure }) => {
                    // No model, so no round: the card goes back at once.
                    leases.release_round(&run_id);
                    return Err(load_failed(deps, &run_id, &binding.model_id, failure));
                }
                Err(refusal) => {
                    leases.release_round(&run_id);
                    return Err(round_refused(format!(
                        "This round's model could not be served, so it was not called: {}",
                        refusal.explain()
                    )));
                }
            }
        }
        // A run nobody bound to a model — a caller outside the two production
        // drivers. Said, not hidden: nothing here holds the card for it.
        None => (
            json!({
                "status": "unbound",
                "because": "this run was not bound to a model by the driver that started it, so \
                            no GPU lease was taken for its rounds",
            }),
            None,
        ),
    };
    let served_window = rebind
        .as_ref()
        .map(|rebind| rebind.served_window)
        .unwrap_or(request.served_window);

    // From here on the round may hold the card. Every way out that is not an
    // opened round gives it back at once, rather than leaving it to the run's
    // next tool boundary or its end. Idempotent, and a no-op for an unbound run.
    let give_back = |error: WireError| {
        if let Some(leases) = deps.leases.as_ref() {
            leases.release_round(&run_id);
        }
        error
    };
    // A context that cannot be compiled is a refused round, not a degraded
    // one. The runtime degrades on any other error and calls the model anyway,
    // which here would mean a call without the card and without the round's
    // corrections and receipts: exactly the loss this boundary exists to stop.
    let not_compiled = |error: WireError| {
        give_back(round_refused(format!(
            "This round's context could not be compiled, so the model was not called: {}",
            error.message
        )))
    };

    // ── The context, against the window the server actually has ─────────
    let compiled = match deps.memory_graph.as_ref() {
        Some(graph) => {
            let session = deps.session().map_err(not_compiled)?;
            let scope = FrozenScope {
                task_id: request.task_id.clone(),
                agent_id: request.agent_id.clone(),
                definition_version: request.definition_version,
                // "Now": the cursor comes from the read itself.
                graph_revision: 0,
                model_id: request.model_id.clone(),
                template_id: request.template_id.clone(),
                served_window,
                project_id: request.project_id.clone(),
                capability: request.capability.clone(),
                now: chrono::Utc::now().to_rfc3339(),
            };
            let base = ContextManifest::new(
                &run_id,
                "",
                "",
                "",
                &request.model_id,
                served_window,
                Vec::new(),
                None,
                HistoryBinding {
                    carried: 0,
                    dropped: 0,
                    tokens: 0,
                    pinned: Vec::new(),
                    omitted_pins: Vec::new(),
                },
            );
            let reserves = Reserves {
                tool_schemas: request.reserved_tool_schemas,
                output: request.reserved_output,
                framing: request.reserved_framing,
                safety: request.reserved_safety,
                // The blocks are sized by estimate here; the whole request is
                // counted by the model's own tokenizer at `context.count`.
                counted_by: "estimate".to_string(),
                window_source: rebind.as_ref().map(|rebind| rebind.window_source.clone()),
            };
            let compiled = ContextCompiler::new(graph)
                .compile(
                    &session,
                    &scope,
                    base,
                    &request.question,
                    &request.already_carried.iter().cloned().collect(),
                    reserves.clone(),
                )
                .map_err(|error| not_compiled(WireError::new(code::INTERNAL, error.explain())))?;

            deps.remember(
                &run_id,
                TaskEventType::ContextCompiled,
                json!({
                    "manifest": compiled.manifest,
                    "mandatoryOverflowed": compiled.mandatory_overflowed,
                }),
            );

            if compiled.mandatory_overflowed {
                return Err(give_back(round_refused(format!(
                    "The mandatory state for this round — the objective, corrections, plan, \
                     completed-effect receipts and pending approvals — needs {} tokens, and the \
                     {}-token window this model is actually serving affords {} after reserving {} \
                     for tool schemas, the reply, framing and safety. Nothing was dropped and the \
                     model was not called. Split the task, or use a configuration with a larger \
                     served window.",
                    compiled.mandatory_tokens(),
                    served_window,
                    reserves.available(served_window),
                    reserves.total()
                ))));
            }
            Some(compiled)
        }
        None => None,
    };

    // ── What the count and the settle will need ─────────────────────────
    deps.rounds
        .prune(|other| other == run_id || super::has_registered_plan(deps, other));
    let previous = deps.rounds.state(&run_id).unwrap_or_default();
    deps.rounds.put(
        &run_id,
        RoundState {
            model_id: rebind
                .as_ref()
                .map(|rebind| rebind.model_id.clone())
                .unwrap_or_else(|| request.model_id.clone()),
            served_model_id: rebind
                .as_ref()
                .map(|rebind| rebind.served_model_id.clone())
                .unwrap_or_else(|| request.model_id.clone()),
            base_url: rebind.as_ref().map(|rebind| rebind.base_url.clone()),
            served_window,
            reserved_output: request.reserved_output,
            reserved_safety: request.reserved_safety,
            manifest: compiled.as_ref().map(|compiled| compiled.manifest.clone()),
            blocks: compiled
                .as_ref()
                .map(|compiled| compiled.blocks.clone())
                .unwrap_or_default(),
            counted_text: None,
            images: 0,
            rounds: previous.rounds.saturating_add(1),
        },
    );

    let endpoint = rebind.as_ref().map(|rebind| {
        json!({
            "baseUrl": rebind.base_url,
            "modelId": rebind.served_model_id,
            "servedWindow": rebind.served_window,
            "windowSource": rebind.window_source,
            "warm": rebind.warm,
            "restarted": rebind.restarted,
            "cache": rebind.cache,
        })
    });

    Ok(match compiled {
        Some(compiled) => json!({
            "graphRevision": compiled.cursor(),
            "manifestHash": compiled.manifest.manifest_hash,
            "mandatoryOverflowed": false,
            "blocks": compiled.blocks,
            "contentHashes": compiled.manifest.content_hashes,
            "omissions": compiled.manifest.omissions,
            "retrieval": compiled.manifest.retrieval,
            "budget": compiled.manifest.budget,
            "scopes": compiled.manifest.scopes,
            "graph": { "available": true },
            "endpoint": endpoint,
            "lease": lease,
        }),
        // Said plainly rather than answered with an empty set that reads as
        // "the graph holds nothing for this task".
        None => json!({
            "graphRevision": Value::Null,
            "manifestHash": "",
            "mandatoryOverflowed": false,
            "blocks": [],
            "contentHashes": [],
            "omissions": [],
            "graph": {
                "available": false,
                "because": "this deployment has no runtime memory graph, so no task memory was \
                            compiled for this round",
            },
            "endpoint": endpoint,
            "lease": lease,
        }),
    })
}

/// A model that would not come up, recorded with an honest decision.
fn load_failed(
    deps: &Arc<RuntimeDeps>,
    run_id: &str,
    model_id: &str,
    failure: crate::serving::fallback::LoadFailure,
) -> WireError {
    use crate::serving::fallback::{decide, FallbackDecision, RunPhase};

    // Work already done under this model means the run keeps it; a run that
    // has not yet made a call is a job whose model was pinned when it started.
    let did_work = deps
        .calls
        .lock()
        .map(|calls| calls.get(run_id).is_some_and(|made| !made.is_empty()))
        .unwrap_or(true);
    let phase = if did_work {
        RunPhase::MidTask
    } else {
        RunPhase::PinnedJob
    };
    let decision = match deps
        .registry
        .as_ref()
        .and_then(|registry| registry.find(model_id).cloned().map(|entry| (registry, entry)))
    {
        Some((registry, entry)) => decide(
            &entry,
            &failure,
            phase,
            entry
                .roles
                .first()
                .copied()
                .unwrap_or(crate::registry::ModelRole::Reasoning),
            registry.all(),
        ),
        None => FallbackDecision::Blocked {
            because: format!(
                "{} could not be loaded ({}): {}. It is not in the registry this run can see, so \
                 no decision beyond stopping could be taken.",
                model_id,
                failure.kind.as_str(),
                failure.detail
            ),
        },
    };
    deps.remember(
        run_id,
        TaskEventType::ModelLoadFailed,
        json!({
            "modelId": model_id,
            "kind": failure.kind.as_str(),
            "detail": failure.detail,
            "phase": phase,
            "decision": decision.as_str(),
            "because": decision.because(),
            "recoverable": decision.recoverable(),
        }),
    );
    round_refused(format!(
        "{} The run's notes, receipts and artifacts are unchanged, and nothing was reported as \
         done that was not.",
        decision.because()
    ))
}

// ─────────────────────────────────────────────────────────────────────────────
// context.count
// ─────────────────────────────────────────────────────────────────────────────

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct CountRequest {
    run_id: String,
    /// The request body exactly as the provider is about to send it.
    payload: Value,
    #[serde(default)]
    call_index: u32,
}

/// Counts the outgoing request and refuses one that does not fit.
pub(super) async fn count(params: Value, deps: &Arc<RuntimeDeps>) -> Result<Value, WireError> {
    let request: CountRequest = serde_json::from_value(params).map_err(|error| {
        WireError::new(
            code::BAD_PARAMS,
            format!("context.count needs a typed request: {error}"),
        )
    })?;
    if !super::has_registered_plan(deps, &request.run_id) {
        return Err(super::no_plan_error(&request.run_id));
    }
    let run_id = request.run_id.clone();
    let Some(mut state) = deps.rounds.state(&run_id) else {
        return Ok(json!({
            "fit": { "fit": "uncounted", "because": "no round was opened for this run" },
            "countedBy": "uncounted",
        }));
    };

    let images = images_in(&request.payload);
    let text = match state.base_url.as_deref() {
        Some(base_url) => deps.rounds.counter.count(base_url, &request.payload).await,
        None => Err("this run's endpoint was not bound through the lease service".to_string()),
    };
    let max_output = request
        .payload
        .get("max_completion_tokens")
        .or_else(|| request.payload.get("max_tokens"))
        .and_then(Value::as_u64)
        .map(|tokens| tokens as u32)
        .unwrap_or(state.reserved_output);
    let fit = judge(
        text.clone(),
        images,
        deps.rounds.image_cost(&state.served_model_id),
        state.served_window,
        max_output,
        state.reserved_safety,
    );
    let total = match &fit {
        Fit::Fits { total, .. } | Fit::Overflows { total, .. } => *total,
        Fit::ImagesUnmeasured { text, .. } => *text,
        Fit::Uncounted { .. } => 0,
    };

    // Did what the compiler authorised reach the wire, and did nothing
    // block-shaped arrive that it did not?
    let projection = state.manifest.as_ref().map(|manifest| {
        record_projection(
            manifest,
            &state.blocks,
            &text_of(&request.payload),
            request.call_index,
            total,
        )
    });

    state.counted_text = text.as_ref().ok().copied();
    state.images = images;
    deps.rounds.put(&run_id, state.clone());

    let input_hash = super::context_compiler::hash_of(&request.payload.to_string());
    deps.remember(
        &run_id,
        TaskEventType::ModelRequested,
        json!({
            "callIndex": request.call_index,
            "modelId": state.model_id,
            "inputHash": input_hash,
            "inputTokens": total,
            "countedBy": fit.counted_by(),
            "fit": fit,
            "images": images,
            "window": state.served_window,
            "reservedOutput": max_output,
            "reservedSafety": state.reserved_safety,
            "manifestHash": state.manifest.as_ref().map(|manifest| manifest.manifest_hash.clone()),
            "projection": projection,
        }),
    );

    if let Fit::Overflows { total, limit, over } = &fit {
        if let Some(leases) = deps.leases.as_ref() {
            leases.release_round(&run_id);
        }
        return Err(round_refused(format!(
            "The request for this round is {total} tokens as {}'s own template and tokenizer \
             count it, {over} over the {limit} this {}-token window leaves after reserving \
             {max_output} for the reply and {} for safety. The model was not called. The \
             transcript needs compacting further, or the task a model with a larger served \
             window.",
            state.model_id, state.served_window, state.reserved_safety
        )));
    }

    Ok(json!({
        "fit": fit,
        "countedBy": fit.counted_by(),
        "inputTokens": total,
        "window": state.served_window,
        "projection": projection,
    }))
}

// ─────────────────────────────────────────────────────────────────────────────
// context.settle
// ─────────────────────────────────────────────────────────────────────────────

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct SettleRequest {
    run_id: String,
    #[serde(default)]
    input_tokens: Option<u32>,
    #[serde(default)]
    output_tokens: Option<u32>,
    #[serde(default)]
    stop_reason: Option<String>,
}

/// Ends a round: the card goes back, and the count meets the server's report.
pub(super) fn settle(params: Value, deps: &Arc<RuntimeDeps>) -> Result<Value, WireError> {
    let request: SettleRequest = serde_json::from_value(params).map_err(|error| {
        WireError::new(
            code::BAD_PARAMS,
            format!("context.settle needs a typed request: {error}"),
        )
    })?;
    if !super::has_registered_plan(deps, &request.run_id) {
        return Err(super::no_plan_error(&request.run_id));
    }
    let run_id = request.run_id.clone();

    let released = deps
        .leases
        .as_ref()
        .map(|leases| leases.release_round(&run_id))
        .unwrap_or(false);

    let state = deps.rounds.state(&run_id).unwrap_or_default();
    // What the images cost, measured: the server's own prompt count less the
    // text the template rendered. Learned once per model and used from then on;
    // never inferred from anything but a report.
    let mut learned = None;
    if let (Some(reported), Some(text)) = (request.input_tokens, state.counted_text) {
        if state.images > 0
            && reported > text
            && deps.rounds.image_cost(&state.served_model_id).is_none()
        {
            let per_image = (reported - text) / state.images;
            if per_image > 0 {
                deps.rounds.learn_image_cost(&state.served_model_id, per_image);
                learned = Some(per_image);
            }
        }
    }
    let delta = match (request.input_tokens, state.counted_text) {
        (Some(reported), Some(text)) if state.images == 0 => Some(i64::from(reported) - i64::from(text)),
        _ => None,
    };

    deps.remember(
        &run_id,
        TaskEventType::ModelResponded,
        json!({
            "modelId": state.model_id,
            "reportedInputTokens": request.input_tokens,
            "outputTokens": request.output_tokens,
            "countedTextTokens": state.counted_text,
            "images": state.images,
            "deltaFromCount": delta,
            "imageCostLearned": learned,
            "stopReason": request.stop_reason,
            "leaseReleased": released,
        }),
    );

    Ok(json!({ "released": released, "imageCostLearned": learned }))
}

#[cfg(test)]
#[path = "rounds_tests.rs"]
pub(crate) mod tests;
