//! P08 end to end: the calculation tools through the runtime's own
//! `authorize` → `execute`, the production calculation-checker worker
//! dispatched by `agent.delegate_readonly`, and two agents' deliverables
//! citing one exact verified calculation — which go stale together when an
//! input is corrected.
//!
//! ## Evidence labels
//!
//! Everything here is a **deterministic test**: the engine, the store, the
//! memory graph, the artifact registry and the worker are the production
//! code, and no model runs anywhere. That is the point of the calculation
//! path — a model formulates, the engine computes — and a model's
//! formulation is what the P10 journey grades.

use std::sync::Arc;

use serde_json::{json, Value};

use super::*;
use crate::calculation::store::CalculationStore;
use crate::identity::{Role, Session, User};
use crate::knowledge::graph::runtime_memory::{ItemStatus, MemoryItem, MemoryKind, MemoryScope, Provenance};
use crate::knowledge::graph::runtime_store::MemoryGraph;
use crate::knowledge::service::RetrievalService;
use crate::registry::{ModelManifest, ModelRegistry, ModelRole};
use crate::subagents::{load_profiles, SpecialistWorker, SubagentManager, WorkerServices};

const OWNER: &str = "priya";
const RUN: &str = "r";
const DECK_RUN: &str = "r-deck";
const CONVERSATION: &str = "c-p08";

fn session() -> Session {
    Session::open(User::new(OWNER, "Priya Sharma", vec![Role::Employee]))
}

struct World {
    deps: Arc<RuntimeDeps>,
    graph: Arc<MemoryGraph>,
    store: Arc<CalculationStore>,
    _dir: tempfile::TempDir,
}

impl World {
    fn new() -> Self {
        let signed_in = Arc::new(std::sync::RwLock::new(Some(session())));
        let (base, dir) = super::tests::deps_with(signed_in.clone());
        let graph = Arc::new(MemoryGraph::in_memory().expect("a graph").with_receipts(base.events.clone()));
        let store = Arc::new(CalculationStore::in_memory().expect("a calculation store"));
        let retrieval = Arc::new(RetrievalService::lexical_only(base.index.clone(), "no embedding model in this test"));
        let mut parent = crate::registry::tests::entry("model-parent", 9.0, vec![ModelRole::Reasoning]);
        parent.supports_structured_output = true;
        let registry = Arc::new(
            ModelRegistry::from_manifest(ModelManifest { models: vec![parent] }, std::path::PathBuf::from("models/registry.json"))
                .expect("a manifest"),
        );
        let cancellations = Arc::new(crate::agent_runtime::cancellation::RunCancellations::new());
        let services = Arc::new(WorkerServices {
            index: base.index.clone(),
            graph: Some(graph.clone()),
            events: base.events.clone(),
            scheduler: Arc::new(crate::subagents::ModelScheduler::new(registry.clone(), Arc::new(crate::serving::ModelServers::new()))),
            session: signed_in.clone(),
            child_loop: None,
            cancellations: cancellations.clone(),
            analyst: None,
            retrieval: retrieval.clone(),
            // One store: what the parent computed is what the checker reads.
            calculation_store: store.clone(),
        });
        let profiles = load_profiles(&std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../agents")).profiles;
        let mut manager = SubagentManager::new(profiles.clone(), base.events.clone()).with_cancellations(cancellations);
        for worker in SpecialistWorker::register_all(&profiles, services) {
            manager = manager.with_worker(worker);
        }
        let deps = Arc::new(RuntimeDeps {
            memory_graph: Some(graph.clone()),
            conversation_artifacts: base.conversation_artifacts.clone(),
            registry: Some(registry),
            index: base.index.clone(),
            session: signed_in,
            workspaces: base.workspaces.clone(),
            approvals: base.approvals.clone(),
            calculations: base.calculations.clone(),
            passages: base.passages.clone(),
            produced: base.produced.clone(),
            calls: base.calls.clone(),
            plans: base.plans.clone(),
            events: base.events.clone(),
            skills: base.skills.clone(),
            hooks: base.hooks.clone(),
            memory: base.memory.clone(),
            checkpoints: base.checkpoints.clone(),
            emit: base.emit.clone(),
            emit_durable: base.emit_durable.clone(),
            subagents: Arc::new(manager),
            multimodal: base.multimodal.clone(),
            audit_health: base.audit_health.clone(),
            documents: base.documents.clone(),
            run_to_conversation: base.run_to_conversation.clone(),
            notebooks: base.notebooks.clone(),
            jobs: base.jobs.clone(),
            extraction: base.extraction.clone(),
            retrieval,
            calculation_store: store.clone(),
        });
        for run in [RUN, DECK_RUN] {
            deps.plans.lock().unwrap().insert(
                run.to_string(),
                crate::orchestrator::plan::PlanRun::new(
                    run,
                    vec!["calculate the wall loss and report it".to_string()],
                    crate::orchestrator::plan::Budget::standard(ToolName::ALL.to_vec()),
                ),
            );
            deps.workspaces
                .lock()
                .unwrap()
                .entry(run.to_string())
                .or_insert_with(|| workspace::Workspace::create(dir.path(), run).expect("workspace"));
            deps.run_to_conversation.bind(run, CONVERSATION);
            deps.checkpoints.lock().unwrap().insert(
                run.to_string(),
                crate::agent_runtime::resume::CheckpointSeed {
                    attempt_id: "attempt-1".into(),
                    plan_hash: "plan".into(),
                    policy_hash: "policy".into(),
                    workspace_hash: "workspace".into(),
                    model_id: "model-parent".into(),
                    committed_notes: Default::default(),
                    manifest: None,
                },
            );
        }
        World { deps, graph, store, _dir: dir }
    }

    /// A sourced figure in the task's memory, as an operator recorded it.
    fn fact(&self, content: &str) -> MemoryItem {
        let mut item = crate::knowledge::graph::runtime_memory::tests::item(MemoryKind::Fact, Provenance::Operator { user_id: "ravi".into() });
        item.scope = MemoryScope::Task { task_id: RUN.into() };
        item.content = content.into();
        let committed = self.graph.commit(item.clone(), None, &[]).expect("committed");
        assert_eq!(committed.status, ItemStatus::Admitted);
        item
    }

    fn status_of(&self, item_id: &str) -> ItemStatus {
        self.graph.versions_of(&session(), item_id, None).unwrap().last().unwrap().item.status
    }
}

async fn call(deps: &Arc<RuntimeDeps>, run: &str, tool: &str, args: Value) -> Result<String, String> {
    let request = json!({ "runId": run, "toolCallId": format!("tc-{tool}-{}", uuid::Uuid::new_v4()), "tool": tool, "args": args });
    let allow = authorize(request.clone(), deps).await.map_err(|e| format!("authorise: {}", e.message))?;
    let grant = allow.get("grant").and_then(Value::as_str).ok_or_else(|| format!("refused: {allow}"))?.to_string();
    let mut spent = request;
    spent["grant"] = json!(grant);
    let result = execute(spent, deps).await.map_err(|e| e.message)?;
    Ok(result["text"].as_str().unwrap_or_default().to_string())
}

fn calc_id(rendered: &str) -> String {
    rendered.split_whitespace().next().filter(|id| id.starts_with("calc-")).expect("a record id first").to_string()
}

fn wall_loss_args(min: &MemoryItem, meas: &MemoryItem) -> Value {
    json!({
        "expression": "wall_loss = (t_min - t_meas) / t_min",
        "inputs": [
            format!("t_min = 9.0 mm [M:{}@1] [note: V-101 shell, inspection plan revision 3]", min.item_id),
            format!("t_meas = 8.2 mm ±0.05 mm [M:{}@1]", meas.item_id),
        ],
        "resultUnit": "%",
    })
}

fn note(calc: &str, display: &str) -> Value {
    json!({
        "path": "note.docx",
        "title": "V-101 shell wall loss at point C",
        "sections": [
            {"heading": "Findings", "level": 1, "blocks": [
                {"kind": "paragraph", "text": format!("Wall loss at point C is {display} of the minimum allowable thickness [C:{calc}].")}
            ]},
            {"heading": "Recommendation", "level": 1, "blocks": [
                {"kind": "paragraph", "text": "Assess point C before the unit returns to service."}
            ]}
        ]
    })
}

fn deck(calc: &str, display: &str) -> Value {
    json!({
        "path": "brief.pptx",
        "content": {
            "title": "V-101 shell at point C",
            "findings": [format!("Wall loss {display} against the minimum [C:{calc}].")],
            "recommendation": ["Assess point C before return to service."],
            "assumptions": ["The minimum allowable thickness is the inspection plan's."],
            "evidence": [format!("Calculation record [C:{calc}].")]
        }
    })
}

/// **Two agents consume one exact verified calculation, and both go stale
/// when an input is corrected.**
///
/// 1. The parent evaluates wall loss from two sourced memory facts; the
///    record is kept and published resting on them.
/// 2. The production calculation-checker, dispatched through
///    `agent.delegate_readonly`, re-reads both facts and recomputes by two
///    paths: VERIFIED, published resting on the record.
/// 3. The document author (run `r`) writes an approval note and a second
///    agent (run `r-deck`, another model) a briefing deck; both cite
///    `[C:calc-…]` and quote the same display string.
/// 4. A person corrects the measured thickness. The record itself is
///    unchanged (immutable); its memory item, the verification and both
///    deliverables are stale, and each tool says so. A second check reports
///    STALE and recomputes with the corrected value as a new record.
#[tokio::test]
async fn two_agents_consume_one_verified_calculation_and_both_go_stale_when_an_input_is_corrected() {
    let world = World::new();
    let min = world.fact("V-101 minimum allowable thickness at point C: 9.0 mm per inspection plan revision 3");
    let meas = world.fact("Governing UT reading at point C: 8.2 mm, survey of August 2026");

    // 1. The calculation.
    let evaluated = call(&world.deps, RUN, "calculation.evaluate_with_units", wall_loss_args(&min, &meas)).await.expect("evaluates");
    println!("--- calculation.evaluate_with_units ---\n{evaluated}");
    let id = calc_id(&evaluated);
    assert!(evaluated.contains("· computed ·"), "{evaluated}");
    assert!(evaluated.contains("Result wall_loss: 8.889 % ±0.56 %"), "{evaluated}");
    assert!(evaluated.contains("Substituted: (9.0 mm - 8.2 mm) / (9.0 mm)"), "{evaluated}");
    let record = world.store.get(&id, OWNER).expect("kept in the store");
    let record_sha = record.sha256();
    let calc_item = world.store.graph_item(&id, OWNER).expect("published into memory");
    let published = world.graph.versions_of(&session(), &calc_item, None).unwrap();
    let published = &published.last().unwrap().item;
    assert_eq!(published.kind, MemoryKind::ToolObservation);
    assert_eq!(published.status, ItemStatus::Admitted, "the tool receipt admits it");
    assert!(matches!(published.provenance, Provenance::ToolReceipt { .. }));
    assert_eq!(published.depends_on.len(), 2, "it rests on both inputs: {:?}", published.depends_on);

    // 2. The independent check, by the production worker.
    let checked = call(
        &world.deps,
        RUN,
        "agent.delegate_readonly",
        json!({ "profile": "calculation-checker", "task": "check the wall loss figure", "expressions": [id.clone()] }),
    )
    .await
    .expect("the checker runs");
    println!("--- calculation-checker ---\n{checked}");
    assert!(checked.contains("status: completed"), "{checked}");
    assert!(checked.contains(&format!("{id} VERIFIED")), "{checked}");
    let checks = world.store.checks(&id);
    assert_eq!(checks.len(), 1);
    assert_eq!(checks[0].verdict, "verified");
    let verification = world
        .graph
        .snapshot(&session(), &MemoryScope::Task { task_id: RUN.into() }, None)
        .unwrap()
        .into_iter()
        .find(|item| item.content.starts_with(&format!("check of {id}: VERIFIED")))
        .expect("the verification is in the task's memory");
    assert_eq!(verification.agent_id, "calculation-checker");
    assert!(verification.depends_on.iter().any(|d| d.item_id == calc_item), "it rests on the record: {:?}", verification.depends_on);

    // 3. Two agents consume the exact figure.
    let display = record.first().unwrap().display.clone();
    call(&world.deps, RUN, "artifact.create_approval_note", note(&id, &display)).await.expect("the note is written");
    call(&world.deps, DECK_RUN, "artifact.create_briefing_deck", deck(&id, &display)).await.expect("the deck is written");
    let artifacts = world.deps.conversation_artifacts.list(OWNER, CONVERSATION).expect("lists");
    assert_eq!(artifacts.len(), 2, "{artifacts:#?}");
    for artifact in &artifacts {
        let dependencies = world.deps.conversation_artifacts.dependencies(OWNER, &artifact.reference()).expect("reads");
        let cited = dependencies.iter().find(|d| d.id == id).unwrap_or_else(|| panic!("{} does not rest on {id}: {dependencies:#?}", artifact.title));
        assert_eq!(cited.version.as_deref(), Some(display.as_str()), "the exact displayed figure");
        assert_eq!(cited.sha256.as_deref(), Some(record_sha.as_str()), "the exact record");
        let evidence = call(&world.deps, RUN, "artifact.resolve_evidence", json!({ "artifact": artifact.reference().to_string() })).await.unwrap();
        assert!(evidence.contains("everything it rests on is current"), "{evidence}");
    }

    // 4. A person corrects the measurement.
    let mut correction = crate::knowledge::graph::runtime_memory::tests::item(MemoryKind::Correction, Provenance::Operator { user_id: "ravi".into() });
    correction.scope = MemoryScope::Task { task_id: RUN.into() };
    correction.content = "Governing UT reading at point C corrected to 7.9 mm after recalibration".into();
    world.graph.correct(correction, &meas.item_id).expect("corrected");

    assert_eq!(world.store.get(&id, OWNER).unwrap().sha256(), record_sha, "the record itself never changes");
    assert_eq!(world.status_of(&calc_item), ItemStatus::Stale, "the correction did not reach the calculation");
    assert_eq!(world.status_of(&verification.item_id), ItemStatus::Stale, "the verification still stands");
    for artifact in world.deps.conversation_artifacts.list(OWNER, CONVERSATION).unwrap() {
        let node = artifact.graph_item_id.as_deref().expect("each version is in the graph");
        assert_eq!(world.status_of(node), ItemStatus::Stale, "{} did not go stale", artifact.title);
        let evidence = call(&world.deps, RUN, "artifact.resolve_evidence", json!({ "artifact": artifact.reference().to_string() })).await.unwrap();
        println!("--- {} after the correction ---\n{evidence}", artifact.title);
        // The calculation's own binding says it is stale, not just the node.
        assert!(evidence.contains(&format!("calculation {id} — stale")), "{evidence}");
    }

    // A second check says so, and recomputes with the corrected value.
    let again = call(
        &world.deps,
        RUN,
        "agent.delegate_readonly",
        json!({ "profile": "calculation-checker", "task": "check the wall loss figure again", "expressions": [id.clone()] }),
    )
    .await
    .expect("the checker runs");
    println!("--- calculation-checker after the correction ---\n{again}");
    assert!(again.contains(&format!("{id} STALE")), "{again}");
    assert!(again.contains("12.22 %"), "recomputed with 7.9 mm: {again}");
    assert_eq!(world.store.checks(&id).last().unwrap().verdict, "stale");

    // The recomputation is a new record that cites the correction, not the
    // superseded reading -- and checking it verifies.
    let corrected_id = again
        .split("recomputed with the corrected input(s): ")
        .nth(1)
        .and_then(|rest| rest.split_whitespace().next())
        .expect("the corrected record's id")
        .to_string();
    let corrected = world.store.get(&corrected_id, OWNER).expect("the corrected record is kept");
    let measured = corrected.inputs.iter().find(|i| i.id == "t_meas").unwrap();
    assert_eq!(measured.value.as_deref(), Some("7.9"));
    assert!(
        matches!(&measured.source, crate::calculation::InputSource::Memory { item_id, .. } if *item_id != meas.item_id),
        "it cites the correction: {:?}",
        measured.source
    );
    let rechecked = call(
        &world.deps,
        RUN,
        "agent.delegate_readonly",
        json!({ "profile": "calculation-checker", "task": "check the corrected figure", "expressions": [corrected_id.clone()] }),
    )
    .await
    .expect("the checker runs");
    assert!(rechecked.contains(&format!("{corrected_id} VERIFIED")), "{rechecked}");
}

#[tokio::test]
async fn a_refused_calculation_publishes_nothing_and_says_what_kind_of_refusal() {
    let world = World::new();
    let refused = call(
        &world.deps,
        RUN,
        "calculation.evaluate_with_units",
        json!({ "expression": "(t_min - t_meas) / t_min", "inputs": ["t_min = 0.0 mm [user]", "t_meas = 6.4 mm [user]"], "resultUnit": "%" }),
    )
    .await
    .expect_err("a zero minimum is refused");
    println!("--- refused ---\n{refused}");
    assert!(refused.contains("division_by_zero"), "{refused}");
    assert!(refused.contains("No number was produced"), "{refused}");
    let id = calc_id(&refused);
    assert!(world.store.get(&id, OWNER).is_some(), "the refusal is kept on the record");
    assert!(world.store.graph_item(&id, OWNER).is_none(), "nothing to build on, so nothing published");
    let mismatch = call(
        &world.deps,
        RUN,
        "calculation.validate_dimensions",
        json!({ "expression": "p + t", "inputs": ["p = ? bar", "t = ? K"] }),
    )
    .await
    .expect_err("a pressure plus a temperature");
    assert!(mismatch.contains("dimension_mismatch") && mismatch.contains("a pressure"), "{mismatch}");
}

#[tokio::test]
async fn the_other_families_run_through_the_runtime_and_keep_their_records() {
    let world = World::new();
    let solved = call(
        &world.deps,
        RUN,
        "calculation.solve",
        json!({ "equations": ["p = 2 * s * t / d"], "unknowns": ["t mm"], "inputs": ["p = 10 MPa [user]", "s = 138 MPa [user]", "d = 219.1 mm [user]"] }),
    )
    .await
    .expect("solves");
    assert!(solved.contains("Result t: 7.938 mm"), "{solved}");

    let compared = call(
        &world.deps,
        RUN,
        "calculation.compare",
        json!({
            "value": "t_meas", "relation": ">=", "limit": "t_req",
            "inputs": ["t_meas = 8.2 mm [user]", "t_req = 7.938 mm [standard: ASME B31.3-2022 §304.1.2] [user]"]
        }),
    )
    .await
    .expect("compares");
    println!("--- compare ---\n{compared}");
    assert!(compared.contains("HOLDS") && compared.contains("Conclusion:") && compared.contains("2022"), "{compared}");

    let evaluated = call(
        &world.deps,
        RUN,
        "calculation.evaluate_with_units",
        json!({ "expression": "(t_min - t_meas) / t_min", "inputs": ["t_min = 9.0 mm [user]", "t_meas = 8.2 mm ±0.05 mm [user]"], "resultUnit": "%" }),
    )
    .await
    .expect("evaluates");
    let id = calc_id(&evaluated);
    let sensitivity = call(&world.deps, RUN, "calculation.sensitivity", json!({ "calculationId": id })).await.expect("sensitivity");
    assert!(sensitivity.contains("Sensitivity t_meas") && sensitivity.contains("elasticity"), "{sensitivity}");

    // The workbook shows the record behind each figure.
    call(&world.deps, RUN, "artifact.create_calculation_workbook", json!({ "path": "working.xlsx" })).await.expect("writes");
    let table = world.deps.calculations.lock().unwrap().get(RUN).cloned().unwrap_or_default();
    assert!(table.iter().any(|row| row.id == id), "the run's table holds the record's projection");
    let listed = world.deps.conversation_artifacts.list(OWNER, CONVERSATION).unwrap();
    let rests_on_it = listed.iter().any(|artifact| {
        world
            .deps
            .conversation_artifacts
            .dependencies(OWNER, &artifact.reference())
            .unwrap_or_default()
            .iter()
            .any(|d| d.id == id)
    });
    assert!(rests_on_it, "the workbook does not rest on {id}: {listed:#?}");

    // The same computation asked again is the same record.
    let again = call(
        &world.deps,
        RUN,
        "calculation.evaluate_with_units",
        json!({ "expression": "(t_min - t_meas) / t_min", "inputs": ["t_min = 9.0 mm [user]", "t_meas = 8.2 mm ±0.05 mm [user]"], "resultUnit": "%" }),
    )
    .await
    .expect("evaluates");
    assert_eq!(calc_id(&again), id);
}

#[tokio::test]
async fn an_unsourced_input_is_provisional_and_its_check_is_unresolved() {
    let world = World::new();
    let evaluated = call(
        &world.deps,
        RUN,
        "calculation.evaluate_with_units",
        json!({ "expression": "a * b", "inputs": ["a = 3 m", "b = 2 [user]"] }),
    )
    .await
    .expect("evaluates");
    assert!(evaluated.contains("PROVISIONAL") && evaluated.contains("a = 3 m has no source"), "{evaluated}");
    let id = calc_id(&evaluated);
    let checked = call(
        &world.deps,
        RUN,
        "agent.delegate_readonly",
        json!({ "profile": "calculation-checker", "task": "check it", "expressions": [id.clone()] }),
    )
    .await
    .expect("runs");
    assert!(checked.contains(&format!("{id} UNRESOLVED")), "{checked}");
    assert!(checked.contains("UNSOURCED"), "{checked}");
}
