//! The P00 baseline, driven through the production task driver.
//!
//! ## What "the actual task driver" means here
//!
//! [`AgentRuntime::spawn`] starts the same Node child process the application
//! starts, from the same bundle, over the same stdio protocol, against real
//! `RuntimeDeps` backed by real stores in a `TempDir`. Nothing on that path is
//! stubbed. A case that passes here passed against the thing that ships.
//!
//! What is *not* the driver, and is labelled so in the record: the model. Three
//! of the cases below never reach one, and they say `deterministic-transport`
//! rather than pretending the absence does not matter. The plan permits exactly
//! that — "allow deterministic transport fixtures for contract tests but mark
//! them as such" — and the marking is the condition.
//!
//! ## Why it prints JSON
//!
//! `scripts/agent-baseline.mjs` reads these lines and folds them into one
//! report alongside the fixture and coding-task groups. Each line carries the
//! case's kind, its status, the run and agent identity it ran under, the model
//! identity if any, and hashes of anything it produced — which is the evidence
//! §11.1 asks to be stored, rather than a count of green ticks.
//!
//! ## The rule
//!
//! A prerequisite this machine does not have produces `blocked`, never `passed`
//! and never `failed`. A blocked case names the exact command that would
//! unblock it. Nothing here converts "could not look" into "works".

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex, RwLock};

use sarathi_lib::agent_runtime::workspace::Workspace;
use sarathi_lib::agent_runtime::{default_bundle_path, AgentRuntime, RuntimeDeps};
use sarathi_lib::identity::{Role, Session, User};
use sarathi_lib::knowledge::KnowledgeIndex;
use sarathi_lib::orchestrator::approvals::ApprovalQueue;
use sarathi_lib::orchestrator::plan::{Budget, PlanRun};
use sarathi_lib::orchestrator::tools::ToolName;
use serde_json::{json, Value};

/// The marker `scripts/agent-baseline.mjs` greps for. Distinctive on purpose:
/// a substring that could appear in ordinary test output would let a failure
/// message be parsed as a result.
const MARKER: &str = "ARJUN_BASELINE_CASE";

const RUN_ID: &str = "p00-baseline-run-1";
const AGENT_ID: &str = "ag-p00-baseline";

/// One case's record. Printed, not asserted on: the assertions are in the cases.
struct Record {
    id: &'static str,
    /// `deterministic-transport` for a case that never reaches a model,
    /// `real-driver` for one that exercises the driver's own behaviour,
    /// `real-model` for one that generated tokens.
    kind: &'static str,
    status: &'static str,
    detail: String,
    model_id: Option<String>,
    trace: Vec<String>,
    artifacts: BTreeMap<String, String>,
    unblock: Option<String>,
}

impl Record {
    fn new(id: &'static str, kind: &'static str) -> Self {
        Self {
            id,
            kind,
            status: "failed",
            detail: String::new(),
            model_id: None,
            trace: Vec::new(),
            artifacts: BTreeMap::new(),
            unblock: None,
        }
    }

    fn passed(mut self, detail: impl Into<String>) -> Self {
        self.status = "passed";
        self.detail = detail.into();
        self
    }

    fn blocked(mut self, detail: impl Into<String>, unblock: impl Into<String>) -> Self {
        self.status = "blocked";
        self.detail = detail.into();
        self.unblock = Some(unblock.into());
        self
    }

    fn emit(self) {
        let line = json!({
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "detail": self.detail,
            "runId": RUN_ID,
            "agentId": AGENT_ID,
            "modelId": self.model_id,
            "trace": self.trace,
            "artifactHashes": self.artifacts,
            "unblockCommand": self.unblock,
            "driver": "sarathi_lib::agent_runtime::AgentRuntime -> agent-runtime/dist/arjun-agent-runtime.mjs",
        });
        // stdout, and `--nocapture` is how the harness reads it. A test that
        // printed nothing is reported by the harness as `skipped`, which is the
        // correct reading of a case that produced no record.
        println!("{MARKER} {line}");
    }
}

/// Node has to be on PATH to start the driver at all.
fn node_present() -> bool {
    std::process::Command::new("node")
        .arg("--version")
        .output()
        .is_ok()
}

fn bundle_present() -> Option<std::path::PathBuf> {
    let path = default_bundle_path();
    path.exists().then_some(path)
}

/// Real stores in a temp directory, and the run the cases drive.
fn deps() -> (Arc<RuntimeDeps>, tempfile::TempDir) {
    let dir = tempfile::tempdir().expect("temp dir");
    let index = KnowledgeIndex::open(dir.path()).expect("index opens");

    let workspaces = Arc::new(Mutex::new(HashMap::new()));
    workspaces.lock().expect("fresh lock").insert(
        RUN_ID.to_string(),
        Workspace::create(dir.path(), RUN_ID).expect("workspace"),
    );

    let plans: Arc<Mutex<HashMap<String, PlanRun>>> = Arc::default();
    plans.lock().expect("fresh lock").insert(
        RUN_ID.to_string(),
        PlanRun::new(
            RUN_ID,
            vec!["establish the P00 baseline".to_string()],
            Budget::standard(ToolName::ALL.to_vec()),
        ),
    );

    let index = Arc::new(index);
    let deps = RuntimeDeps {
        memory_graph: None,
        index: index.clone(),
        session: Arc::new(RwLock::new(Some(Session::open(User::new(
            "baseline",
            "P00 Baseline",
            vec![Role::Employee],
        ))))),
        workspaces,
        approvals: Arc::new(ApprovalQueue::new()),
        calculations: Arc::default(),
        passages: Arc::default(),
        produced: Arc::default(),
        calls: Arc::default(),
        plans,
        events: Arc::new(
            sarathi_lib::agent_runtime::events::TaskEventLog::open(dir.path())
                .expect("a task event log"),
        ),
        skills: Arc::new(sarathi_lib::skills::SkillRegistry::open(
            dir.path().join("__no_skills__"),
        )),
        hooks: Arc::new(sarathi_lib::hooks::HookRegistry::default()),
        memory: Arc::new(sarathi_lib::agent_runtime::memory::MemoryStore::open(
            dir.path(),
        )),
        checkpoints: Arc::default(),
        emit: Arc::new(|_| {}),
        emit_durable: Arc::new(|_| {}),
        subagents: Arc::new(sarathi_lib::subagents::SubagentManager::new(
            Vec::new(),
            Arc::new(
                sarathi_lib::agent_runtime::events::TaskEventLog::in_memory()
                    .expect("an event log"),
            ),
        )),
        registry: None,
        multimodal: Arc::new(
            sarathi_lib::knowledge::MultimodalIndex::open(dir.path()).expect("a multimodal index"),
        ),
        audit_health: Arc::new(sarathi_lib::agent_runtime::audit_health::AuditHealth::durable()),
        documents: Arc::new(
            sarathi_lib::agent_runtime::documents::DocumentStore::open(dir.path())
                .expect("an extraction store"),
        ),
        run_to_conversation: Arc::new(
            sarathi_lib::agent_runtime::conversations::RunToConversation::new(),
        ),
        conversation_artifacts: Arc::new(
            sarathi_lib::artifacts::conversation_store::ConversationArtifacts::open(dir.path())
                .expect("the artifact store opens"),
        ),
        notebooks: Arc::new(
            sarathi_lib::knowledge::NotebookStore::open(dir.path()).expect("notebook store opens"),
        ),
        jobs: Arc::default(),
        extraction: Arc::new(sarathi_lib::extraction::service::ExtractionService::without_models(
            &dir.path().join("documents"),
            "no OCR or vision model in this test",
        )),
        retrieval: Arc::new(sarathi_lib::knowledge::service::RetrievalService::lexical_only(
            index.clone(),
            "no embedding model in this test",
        )),
    };

    (Arc::new(deps), dir)
}

/// Starts the real driver, or explains why it could not and returns `Err`.
///
/// The explanation is a blocked record rather than a bare `return`, because a
/// machine with no Node and no bundle cannot run this baseline and saying so is
/// the result. A silent return would leave the harness reading zero cases,
/// which it treats as a broken run and not as a pass.
async fn driver(
    id: &'static str,
    kind: &'static str,
    events: Arc<Mutex<Vec<String>>>,
) -> Result<(Arc<AgentRuntime>, Arc<RuntimeDeps>, tempfile::TempDir), Record> {
    if !node_present() {
        return Err(Record::new(id, kind).blocked(
            "node is not on PATH, so the agent runtime child cannot be started",
            "Install Node 24 and put it on PATH, then re-run.",
        ));
    }
    let Some(bundle) = bundle_present() else {
        return Err(Record::new(id, kind).blocked(
            format!(
                "the agent runtime bundle is not built at {}",
                default_bundle_path().display()
            ),
            "npm run runtime:build",
        ));
    };

    let (deps, dir) = deps();
    let sink = Arc::clone(&events);
    let emit = Arc::new(move |value: Value| {
        if let Ok(mut held) = sink.lock() {
            // The event *type*, not its payload: a trace is what happened, and
            // copying payloads here would put run content in an evidence file.
            //
            // `run.event` params are `{ runId, event: { type, ... } }` — see
            // `agent_runtime::mod` around the `run.event` branch, and the TS
            // side's own tests. Reading `type` off the top level finds nothing
            // and records seven events called "unnamed", which is a trace that
            // proves only that something happened seven times.
            let name = value
                .pointer("/event/type")
                .or_else(|| value.get("type"))
                .and_then(Value::as_str)
                .unwrap_or("unnamed")
                .to_string();
            held.push(name);
        }
    });

    match AgentRuntime::spawn(Arc::clone(&deps), emit, bundle) {
        Ok(runtime) => Ok((runtime, deps, dir)),
        Err(error) => Err(Record::new(id, kind).blocked(
            format!("the agent runtime child did not start: {error}"),
            "npm run runtime:build, then check that node runs.",
        )),
    }
}

// ── Cases ───────────────────────────────────────────────────────────────────

#[tokio::test]
async fn transport_01_health() {
    const ID: &str = "transport-01-health";
    let events: Arc<Mutex<Vec<String>>> = Arc::default();
    let (runtime, _deps, _dir) =
        match driver(ID, "deterministic-transport", Arc::clone(&events)).await {
            Ok(started) => started,
            Err(blocked) => return blocked.emit(),
        };

    let answer = runtime
        .request("health", json!({}))
        .await
        .expect("the driver answers health");
    runtime.shutdown().await;

    let mut record = Record::new(ID, "deterministic-transport");
    record.trace = events.lock().map(|held| held.clone()).unwrap_or_default();
    record
        .passed(format!(
            "the production driver started and answered health: {answer}"
        ))
        .emit();
}

#[tokio::test]
async fn transport_02_unknown_method_is_refused() {
    const ID: &str = "transport-02-unknown-method-refused";
    let events: Arc<Mutex<Vec<String>>> = Arc::default();
    let (runtime, _deps, _dir) =
        match driver(ID, "deterministic-transport", Arc::clone(&events)).await {
            Ok(started) => started,
            Err(blocked) => return blocked.emit(),
        };

    let outcome = runtime.request("no.such.method", json!({})).await;
    runtime.shutdown().await;

    let mut record = Record::new(ID, "deterministic-transport");
    record.trace = events.lock().map(|held| held.clone()).unwrap_or_default();
    match outcome {
        Err(error) => record
            .passed(format!(
                "an unknown method is refused rather than absorbed: {error}"
            ))
            .emit(),
        Ok(value) => {
            record.detail =
                format!("an unknown method returned a result instead of an error: {value}");
            record.emit();
            panic!("the driver answered a method it does not implement");
        }
    }
}

#[tokio::test]
async fn driver_01_refuses_a_run_with_no_model_server() {
    // The honesty case, and the reason this target exists rather than a mock.
    //
    // `run.start` is given a loopback address nothing is listening on. The
    // required behaviour is an error naming the failure. A driver that returned
    // a plausible assistant message here would be fabricating the whole answer,
    // which is the single failure mode this repository's rules are written
    // against — and no amount of downstream checking would catch it, because
    // the fabrication would carry a real run id and a real trace.
    const ID: &str = "driver-01-no-model-server-fails-honestly";
    let events: Arc<Mutex<Vec<String>>> = Arc::default();
    let (runtime, _deps, _dir) = match driver(ID, "real-driver", Arc::clone(&events)).await {
        Ok(started) => started,
        Err(blocked) => return blocked.emit(),
    };

    // Port 1 on loopback: privileged, unbindable by an ordinary process, and so
    // reliably not serving anything. A high random port could collide with
    // something a developer happens to be running.
    let outcome = runtime
        .request(
            "run.start",
            json!({
                "runId": RUN_ID,
                "agentId": AGENT_ID,
                "messageId": "p00-msg-1",
                "prompt": "What is the governing ultrasonic measurement for PV-2201?",
                "systemPrompt": "Answer only from retrieved sources.",
                "model": {
                    "id": "no-such-model",
                    "provider": "sovereign-local",
                    "baseUrl": "http://127.0.0.1:1/v1",
                    "maxTokens": 64
                }
            }),
        )
        .await;
    runtime.shutdown().await;

    let mut record = Record::new(ID, "real-driver");
    record.model_id = Some("no-such-model".to_string());
    record.trace = events.lock().map(|held| held.clone()).unwrap_or_default();

    match outcome {
        Err(error) => record
            .passed(format!(
                "the driver reported the unreachable model rather than inventing an answer: {error}"
            ))
            .emit(),
        Ok(value) => {
            let text = value.to_string();
            // A structured result that itself reports the failure is also
            // correct. What is not correct is a result carrying content.
            let honest = text.contains("error")
                || text.contains("failed")
                || text.contains("ECONNREFUSED")
                || text.contains("fetch");
            if honest {
                record
                    .passed(format!("the driver returned a failure outcome: {text}"))
                    .emit();
            } else {
                record.detail =
                    format!("the driver returned content with no model reachable: {text}");
                record.emit();
                panic!("the driver produced an answer without a model");
            }
        }
    }
}

#[tokio::test]
async fn model_01_real_generation() {
    // The target-machine gate, held open.
    //
    // There is no default endpoint and no fallback to a model this test picks
    // for itself. An endpoint the operator did not choose is an endpoint whose
    // identity the record cannot state, and a baseline that names the wrong
    // model is worse than one that names none.
    const ID: &str = "model-01-real-generation";
    let Ok(base_url) = std::env::var("ARJUN_BASELINE_MODEL_URL") else {
        return Record::new(ID, "real-model")
            .blocked(
                "no served model endpoint was supplied, so no tokens were generated on this machine",
                "Start a server, e.g.  llama-server -m <weights.gguf> --port 8080 -c 4096  \
                 then set ARJUN_BASELINE_MODEL_URL=http://127.0.0.1:8080/v1 and re-run \
                 `cargo test --manifest-path src-tauri/Cargo.toml --test agent_baseline -- --nocapture`.",
            )
            .emit();
    };
    let model_id = std::env::var("ARJUN_BASELINE_MODEL_ID").unwrap_or_else(|_| "unnamed".into());

    let events: Arc<Mutex<Vec<String>>> = Arc::default();
    let (runtime, _deps, _dir) = match driver(ID, "real-model", Arc::clone(&events)).await {
        Ok(started) => started,
        Err(blocked) => return blocked.emit(),
    };

    let outcome = runtime
        .request(
            "run.start",
            json!({
                "runId": RUN_ID,
                "agentId": AGENT_ID,
                "messageId": "p00-msg-model-1",
                "prompt": "Reply with exactly the word: acknowledged",
                "systemPrompt": "Answer in one word.",
                "model": {
                    "id": model_id,
                    "provider": "sovereign-local",
                    "baseUrl": base_url,
                    "maxTokens": 32
                }
            }),
        )
        .await;
    runtime.shutdown().await;

    let mut record = Record::new(ID, "real-model");
    record.model_id = Some(model_id);
    record.trace = events.lock().map(|held| held.clone()).unwrap_or_default();

    let value = match outcome {
        Ok(value) => value,
        Err(error) => {
            // A supplied endpoint that does not work is a failure, not a block:
            // the operator said it was there.
            record.detail = format!("the supplied model endpoint did not answer: {error}");
            record.emit();
            panic!("ARJUN_BASELINE_MODEL_URL was set and the transport call did not return");
        }
    };

    // `run.start` returning `Ok` means the driver answered, not that the run
    // worked. The first version of this case checked only the former, and
    // reported `passed` for a run whose own outcome was
    // `failed: request (8852 tokens) exceeds the available context size (4096)`.
    // That is the exact shape of the fabricated pass the plan forbids: a green
    // case sitting on top of a failed run, with a real trace underneath it.
    //
    // So the outcome the driver reports is what decides this case.
    let kind = value.pointer("/outcome/kind").and_then(Value::as_str);
    let detail = value
        .pointer("/outcome/detail")
        .and_then(Value::as_str)
        .unwrap_or("");
    let text = value.get("text").and_then(Value::as_str).unwrap_or("");
    let stop = value.get("stopReason").and_then(Value::as_str).unwrap_or("");

    match kind {
        Some("completed") if !text.trim().is_empty() => record
            .passed(format!(
                "a real model generated {} character(s) through the production driver \
                 (stopReason {stop:?})",
                text.chars().count()
            ))
            .emit(),
        Some("completed") => {
            record.detail =
                format!("the run reported completed and produced no text (stopReason {stop:?})");
            record.emit();
            panic!("the run completed with empty output");
        }
        _ => {
            // Not a panic. A model that cannot fit one turn on this machine is
            // a measurement, and the measurement is the point of the baseline.
            // It is `failed`, it names the reason, and the reason is actionable.
            record.status = "failed";
            record.detail = format!(
                "the run did not complete: outcome {} — {detail} (stopReason {stop:?})",
                kind.unwrap_or("absent")
            );
            record.unblock = Some(
                "Serve this model with a context large enough for the tool schemas the runtime \
                 sends. Measure the schema cost first: the context_ledger event on this run \
                 reports it."
                    .to_string(),
            );
            record.emit();
        }
    }
}
