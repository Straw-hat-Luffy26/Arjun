//! P09 end to end: the Document Author's tools through the runtime's own
//! `authorize` → `execute`, and the production `document-author` worker
//! dispatched by `agent.delegate`, calling the same gateway through its tool
//! port under a plan narrowed to its granted tools.
//!
//! ## Evidence labels
//!
//! **Deterministic test.** The writer, the patcher, the checks, the artifact
//! registry, the memory graph, the event log and the worker are production
//! code, and no model runs anywhere: the worker's no-model path assembles the
//! specification from the task's shared memory. Where this machine has the
//! pinned renderers (LibreOffice + PyMuPDF) the pages are laid out for real
//! and the render checks run; where it does not, each test asserts that the
//! render checks are *unavailable*, not passed.
//!
//! With `ARJUN_EVIDENCE_OUT` set, the versions written and their rendered
//! pages are copied there, for the evidence folder.

use std::sync::Arc;

use serde_json::{json, Value};

use super::*;
use crate::agent_runtime::events::plans::JobStatus;
use crate::calculation::store::CalculationStore;
use crate::identity::{Role, Session, User};
use crate::knowledge::graph::runtime_memory::{EdgeKind, ItemStatus, MemoryItem, MemoryKind, MemoryScope, Provenance};
use crate::knowledge::graph::runtime_store::MemoryGraph;
use crate::knowledge::service::RetrievalService;
use crate::registry::{ModelManifest, ModelRegistry, ModelRole};
use crate::subagents::tool_port::ToolPortSlot;
use crate::subagents::{load_profiles, SpecialistWorker, SubagentManager, WorkerServices};

const OWNER: &str = "priya";
const RUN: &str = "r";
const CONVERSATION: &str = "c-p09";

fn session() -> Session {
    Session::open(User::new(OWNER, "Priya Sharma", vec![Role::Employee, Role::Administrator]))
}

fn reviewer() -> Session {
    Session::open(User::new("ravi", "Ravi Menon", vec![Role::Administrator]))
}

fn renderers_usable() -> bool {
    let inventory = crate::artifacts::render::inventory();
    inventory.office.usable() && inventory.rasteriser.usable()
}

struct World {
    deps: Arc<RuntimeDeps>,
    graph: Arc<MemoryGraph>,
    store: Arc<CalculationStore>,
    dir: tempfile::TempDir,
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
        // The slot the runtime fills when it starts, exactly as `lib.rs`
        // hands it to the workers and `commands::agent::runtime` fills it.
        let tools: ToolPortSlot = Default::default();
        let services = Arc::new(WorkerServices {
            index: base.index.clone(),
            graph: Some(graph.clone()),
            events: base.events.clone(),
            scheduler: Arc::new(crate::subagents::ModelScheduler::new(registry.clone(), Arc::new(crate::serving::ModelServers::new()))),
            session: signed_in.clone(),
            child_loop: None,
            cancellations: cancellations.clone(),
            analyst: Some(crate::subagents::AnalystServices {
                extraction: base.extraction.clone(),
                documents: base.documents.clone(),
                conversations: base.run_to_conversation.clone(),
            }),
            retrieval: retrieval.clone(),
            calculation_store: store.clone(),
            tools: tools.clone(),
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
            jobs: Arc::default(),
            extraction: base.extraction.clone(),
            retrieval,
            calculation_store: store.clone(),
        });
        *tools.write().unwrap() = Some(Arc::new(super::tool_port::RuntimeToolPort::new(&deps)));
        deps.run_to_conversation.bind(RUN, CONVERSATION);
        deps.checkpoints.lock().unwrap().insert(
            RUN.to_string(),
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
        World { deps, graph, store, dir }
    }

    /// Something a person recorded in the task's memory.
    fn remember(&self, kind: MemoryKind, content: &str) -> MemoryItem {
        let mut item = crate::knowledge::graph::runtime_memory::tests::item(kind, Provenance::Operator { user_id: "ravi".into() });
        item.scope = MemoryScope::Task { task_id: RUN.into() };
        item.content = content.into();
        let committed = self.graph.commit(item.clone(), None, &[]).expect("committed");
        assert_eq!(committed.status, ItemStatus::Admitted);
        item
    }

    fn correct(&self, old: &MemoryItem, content: &str) -> MemoryItem {
        let mut correction = crate::knowledge::graph::runtime_memory::tests::item(MemoryKind::Correction, Provenance::Operator { user_id: "ravi".into() });
        correction.scope = MemoryScope::Task { task_id: RUN.into() };
        correction.content = content.into();
        self.graph.correct(correction.clone(), &old.item_id).expect("corrected");
        correction
    }

    fn bytes(&self, reference: &str) -> Vec<u8> {
        let (id, version) = reference.rsplit_once('@').unwrap();
        let record = self.deps.conversation_artifacts.get(OWNER, id, Some(version.parse().unwrap())).unwrap().unwrap();
        self.deps.conversation_artifacts.read(OWNER, &record.reference()).unwrap().unwrap().1
    }

    fn sha(&self, reference: &str) -> String {
        let (id, version) = reference.rsplit_once('@').unwrap();
        self.deps.conversation_artifacts.get(OWNER, id, Some(version.parse().unwrap())).unwrap().unwrap().sha256
    }

    /// Edges leaving an item, inside what the signed-in person may read.
    fn edges_from(&self, item_id: &str) -> Vec<crate::knowledge::graph::runtime_memory::MemoryEdge> {
        let visible = self.graph.snapshot(&session(), &MemoryScope::Task { task_id: RUN.into() }, None).unwrap();
        self.graph.neighbours(item_id, &visible).unwrap().into_iter().filter(|edge| edge.from_item == item_id).collect()
    }

    fn step(&self, id: &str) -> task_plan::Step {
        self.deps.events.latest_plan(RUN).unwrap().unwrap().step(id).cloned().expect("the step exists")
    }

    async fn settled(&self, job_id: &str) -> crate::agent_runtime::events::plans::JobRecord {
        for _ in 0..1500 {
            let job = self.deps.events.job(job_id).unwrap().expect("the job exists");
            if !job.status.is_live() && self.step(&job.step_id).job_id.as_deref() != Some(job_id) {
                return job;
            }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
        panic!("job {job_id} never settled");
    }

    /// Copies a version, and the pages of its latest render, to the evidence
    /// folder when one is named.
    fn keep(&self, reference: &str, name: &str) {
        let Ok(out) = std::env::var("ARJUN_EVIDENCE_OUT") else { return };
        let out = std::path::PathBuf::from(out);
        std::fs::create_dir_all(&out).unwrap();
        std::fs::write(out.join(format!("{name}.docx")), self.bytes(reference)).unwrap();
        let (id, version) = reference.rsplit_once('@').unwrap();
        let reference = crate::artifacts::conversation_store::ArtifactRef::new(id, version.parse().unwrap());
        if let Some(render) = self.deps.conversation_artifacts.renders_for(OWNER, &reference).unwrap().first() {
            let dir = self.deps.conversation_artifacts.renders_dir().join(&render.render_id);
            for entry in std::fs::read_dir(&dir).unwrap().flatten() {
                let file = entry.file_name().to_string_lossy().to_string();
                if file.ends_with(".png") {
                    std::fs::copy(entry.path(), out.join(format!("{name}-{file}"))).unwrap();
                }
            }
            std::fs::write(out.join(format!("{name}-render.json")), &render.outcome).unwrap();
        }
    }
}

fn request(run: &str, tool: &str, args: Value) -> Value {
    json!({ "runId": run, "toolCallId": format!("tc-{tool}-{}", uuid::Uuid::new_v4()), "tool": tool, "args": args, "model": "fixture-model" })
}

async fn call(deps: &Arc<RuntimeDeps>, tool: &str, args: Value) -> Result<String, String> {
    let request = request(RUN, tool, args);
    let allow = authorize(request.clone(), deps).await.map_err(|e| format!("authorise: {}", e.message))?;
    spend(deps, request, allow).await
}

async fn spend(deps: &Arc<RuntimeDeps>, request: Value, allow: Value) -> Result<String, String> {
    let grant = allow
        .get("grant")
        .and_then(Value::as_str)
        .ok_or_else(|| format!("refused: {}", allow.get("reason").and_then(Value::as_str).unwrap_or("?")))?
        .to_string();
    let mut spent = request;
    spent["grant"] = json!(grant);
    let result = execute(spent, deps).await.map_err(|e| e.message)?;
    Ok(result["text"].as_str().unwrap_or_default().to_string())
}

/// Authorise a call a person must approve, by being that person.
async fn approved(deps: &Arc<RuntimeDeps>, tool: &str, args: Value) -> Result<String, String> {
    let request = request(RUN, tool, args);
    let queue = deps.approvals.clone();
    let waiting = tokio::spawn({
        let deps = deps.clone();
        let request = request.clone();
        async move { authorize(request, &deps).await }
    });
    let item = loop {
        if let Some(item) = queue.pending().first().cloned() {
            break item;
        }
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    };
    queue.decide(&reviewer(), &item.request.id, true, None).expect("approved");
    let allow = waiting.await.expect("finished").map_err(|e| format!("authorise: {}", e.message))?;
    spend(deps, request, allow).await
}

fn registered(text: &str) -> String {
    let at = text.find("Registered as ").unwrap_or_else(|| panic!("no registration in: {text}"));
    text[at + "Registered as ".len()..].split_whitespace().next().unwrap().to_string()
}

fn written(text: &str) -> String {
    text.split_whitespace().nth(1).filter(|w| w.starts_with("art-")).unwrap_or_else(|| panic!("no version in: {text}")).to_string()
}

fn job_id_in(text: &str) -> String {
    text.split_whitespace()
        .find(|word| word.starts_with("job-"))
        .map(|word| word.trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '-').to_string())
        .unwrap_or_else(|| panic!("no job id in: {text}"))
}

/// The inspection evidence: a UT reading, the SOP minimum, a recipient and a
/// decision, as a person recorded them; and the wall-loss calculation the
/// engine computed from the first two.
struct Evidence {
    reading: MemoryItem,
    minimum: MemoryItem,
    decision: MemoryItem,
    calc: String,
    display: String,
}

async fn evidence(world: &World) -> Evidence {
    let reading = world.remember(MemoryKind::Fact, "UT survey of V-101, August: governing shell reading at point C is 8.2 mm, pitting noted");
    let minimum = world.remember(MemoryKind::Fact, "SOP-114 rev D section 3.1: minimum allowable shell thickness for V-101 where pitting is recorded is 9.0 mm");
    world.remember(MemoryKind::Fact, "Recipient: Plant Manager, Unit 4");
    let decision = world.remember(MemoryKind::Decision, "Re-survey point C within 14 days and hold V-101 below its reduced operating limit until then");
    let computed = call(
        &world.deps,
        "calculation.evaluate_with_units",
        json!({
            "expression": "wall_loss = (t_min - t_meas) / t_min",
            "inputs": [
                format!("t_min = 9.0 mm [M:{}@1]", minimum.item_id),
                format!("t_meas = 8.2 mm [M:{}@1]", reading.item_id),
            ],
            "resultUnit": "%",
        }),
    )
    .await
    .expect("the calculation runs");
    let calc = computed.split_whitespace().find(|w| w.starts_with("calc-")).expect("a record id").trim_end_matches(|c: char| !c.is_ascii_hexdigit()).to_string();
    let display = world.store.get(&calc, OWNER).expect("kept").first().unwrap().display.clone();
    Evidence { reading, minimum, decision, calc, display }
}

fn note_spec(e: &Evidence) -> Value {
    json!({
        "template": "inspection_approval_note@1",
        "title": "V-101 shell wall loss at point C",
        "audience": "Plant Manager, Unit 4",
        "output": "v101-approval-note.docx",
        "fields": { "recipient": "Plant Manager, Unit 4", "reference": "INSP-2026-081" },
        "sections": [
            { "id": "subject", "blocks": [{ "kind": "paragraph", "text": "Shell thickness of V-101 at point C after the August UT survey." }] },
            { "id": "findings", "blocks": [{ "kind": "bullets", "items": [
                format!("The governing reading at point C is 8.2 mm, with pitting noted [M:{}@1].", e.reading.item_id),
                format!("Where pitting is recorded the minimum allowable thickness is 9.0 mm [M:{}@1].", e.minimum.item_id),
            ] }] },
            { "id": "calculation", "blocks": [{ "kind": "paragraph", "text": format!("Wall loss against the minimum is {} [C:{}].", e.display, e.calc) }] },
            { "id": "recommendation", "blocks": [{ "kind": "paragraph", "text": format!("Re-survey point C within 14 days and hold V-101 below its reduced operating limit until then [M:{}@1].", e.decision.item_id) }] },
            { "id": "assumptions", "blocks": [{ "kind": "paragraph", "text": "The survey instrument was within its calibration interval; no other assumption is made." }] },
        ],
    })
}

// ── Inspection evidence → a cited approval note ─────────────────────────

#[tokio::test]
async fn inspection_evidence_becomes_a_cited_editable_approval_note_that_reopens_and_renders() {
    let world = World::new();
    let e = evidence(&world).await;
    let composed = call(&world.deps, "document.compose", note_spec(&e)).await.expect("composed");
    println!("--- document.compose ---\n{composed}");
    let reference = registered(&composed);
    assert!(composed.contains("candidate") && composed.contains("DRAFT"), "{composed}");

    // Reopened from the store: a Word package with its own styles, numbering,
    // header, footer and page settings, every section at its stable id.
    let bytes = world.bytes(&reference);
    let report = crate::artifacts::package::inspect(&bytes, &crate::artifacts::package::LIMITS);
    assert!(report.safe_to_open(), "{report:?}");
    for part in ["word/styles.xml", "word/numbering.xml", "word/header1.xml", "word/footer1.xml", "word/settings.xml"] {
        assert!(report.parts.iter().any(|p| p == part), "{part}");
    }
    let xml = crate::artifacts::package::read_part(&bytes, "word/document.xml").unwrap();
    let ids: Vec<String> = crate::artifacts::authoring::read_sections(&xml).unwrap().into_iter().map(|s| s.id).collect();
    assert_eq!(ids, ["subject", "findings", "calculation", "recommendation", "assumptions", "references", "provenance"]);
    let text = crate::artifacts::content::extract(&bytes).unwrap().plain_text();
    assert!(text.contains(&format!("[C:{}]", e.calc)) && text.contains("Supporting references"), "{text}");

    // Every citation is bound to what it names, and the calculation to its record.
    let record = world.deps.conversation_artifacts.get(OWNER, reference.split('@').next().unwrap(), Some(1)).unwrap().unwrap();
    assert_eq!(record.template.as_ref().map(|t| t.id.as_str()), Some("inspection_approval_note"));
    let dependencies = world.deps.conversation_artifacts.dependencies(OWNER, &record.reference()).unwrap();
    assert!(dependencies.iter().any(|d| d.id == e.reading.item_id));
    assert!(dependencies.iter().any(|d| d.id == e.calc));

    // Section lineage, as the manifest shows it.
    let manifest = call(&world.deps, "artifact.manifest", json!({ "artifact": reference })).await.unwrap();
    assert!(manifest.contains("\"sectionId\": \"findings\""), "{manifest}");
    assert!(manifest.contains(&format!("[C:{}]", e.calc)), "{manifest}");

    // Graph lineage: the version's item rests on the facts and cites the record.
    let artifact_item = record.graph_item_id.clone().expect("linked to the graph on its receipt");
    let edges = world.edges_from(&artifact_item);
    println!("--- graph edges from {artifact_item} ---");
    for edge in &edges {
        println!("{} -{}-> {}", edge.from_item, edge.kind.as_str(), edge.to_item);
    }
    assert!(edges.iter().any(|edge| edge.to_item == e.reading.item_id && edge.kind == EdgeKind::DerivedFrom));
    assert!(edges.iter().any(|edge| edge.kind == EdgeKind::Cites && edge.to_item.contains(&e.calc)));

    // Validated: the ladder, the document checks and every page rendered.
    let validated = call(&world.deps, "artifact.validate_document", json!({ "artifact": reference })).await.unwrap();
    println!("--- artifact.validate_document ---\n{validated}");
    for check in ["document.sections", "document.gaps", "document.citations_present", "document.citations_resolve", "document.figures_supported", "document.approval_status"] {
        assert!(validated.contains(&format!("- {check} [Blocking]: pass")), "{check}: {validated}");
    }
    if renderers_usable() {
        assert!(validated.contains("- render.text_visible [Blocking]: pass"), "{validated}");
        assert!(validated.contains("Fonts: "), "font versions are recorded: {validated}");
        assert!(validated.contains("Accepted by the backend checks"), "{validated}");
    } else {
        assert!(validated.contains("- render.pages [Blocking]: unavailable"), "{validated}");
        assert!(validated.contains("Not accepted"), "{validated}");
    }
    // Accepted is still not approved: the version stays a candidate.
    let record = world.deps.conversation_artifacts.get(OWNER, &record.artifact_id, Some(1)).unwrap().unwrap();
    assert_eq!(record.stage, crate::artifacts::conversation_store::Stage::Candidate);
    world.keep(&reference, "note-v1");
}

// ── Missing information is a gap, never content ──────────────────────────

#[tokio::test]
async fn missing_fields_and_unsupported_claims_are_refused_or_reported_and_never_filled() {
    let world = World::new();
    let e = evidence(&world).await;
    let before = world.deps.conversation_artifacts.list(OWNER, CONVERSATION).unwrap().len();

    // A mandatory field and a mandatory section not supplied: refused, by name.
    let mut spec = note_spec(&e);
    spec["fields"] = json!({});
    spec["sections"].as_array_mut().unwrap().retain(|s| s["id"] != "recommendation");
    let refused = call(&world.deps, "document.compose", spec).await.expect_err("refused");
    println!("--- missing field and section ---\n{refused}");
    assert!(refused.contains("field_missing") && refused.contains("recipient"), "{refused}");
    assert!(refused.contains("section_missing") && refused.contains("recommendation"), "{refused}");

    // An uncited claim, and a figure no source states: refused.
    let mut spec = note_spec(&e);
    spec["sections"][1]["blocks"][0]["items"].as_array_mut().unwrap().push(json!("The rest of the shell is sound."));
    spec["sections"][1]["blocks"][0]["items"].as_array_mut().unwrap().push(json!(format!("Point D reads 9.4 mm [M:{}@1].", e.reading.item_id)));
    let refused = call(&world.deps, "document.compose", spec).await.expect_err("refused");
    println!("--- unsupported claim and figure ---\n{refused}");
    assert!(refused.contains("unsupported_claim") && refused.contains("rest of the shell"), "{refused}");
    assert!(refused.contains("unsupported_figure") && refused.contains("9.4 mm"), "{refused}");
    assert_eq!(world.deps.conversation_artifacts.list(OWNER, CONVERSATION).unwrap().len(), before, "nothing was written");

    // Declared unknown: written as a visible gap, returned as a clarification
    // request, published as an open question -- and not accepted.
    let mut spec = note_spec(&e);
    spec["fields"]["recipient"] = json!("?");
    spec["sections"][3] = json!({ "id": "recommendation", "gap": "the maintenance lead has not yet decided between repair and de-rating" });
    let composed = call(&world.deps, "document.compose", spec).await.expect("a gap is not a refusal");
    println!("--- with gaps ---\n{composed}");
    assert!(composed.contains("INFORMATION NEEDED (2)"), "{composed}");
    let reference = registered(&composed);
    let text = crate::artifacts::content::extract(&world.bytes(&reference)).unwrap().plain_text();
    assert!(text.contains("INFORMATION NEEDED: the maintenance lead has not yet decided"), "{text}");
    let questions: Vec<MemoryItem> = world
        .graph
        .snapshot(&session(), &MemoryScope::Task { task_id: RUN.into() }, None)
        .unwrap()
        .into_iter()
        .filter(|item| item.kind == MemoryKind::OpenQuestion && item.item_id.starts_with("mi-gap-"))
        .collect();
    assert_eq!(questions.len(), 2, "{questions:#?}");
    let validated = call(&world.deps, "artifact.validate_document", json!({ "artifact": reference, "render": "no" })).await.unwrap();
    assert!(validated.contains("- document.gaps [Blocking]: FAIL"), "{validated}");
    assert!(validated.contains("Not accepted"), "{validated}");
}

// ── A long table ─────────────────────────────────────────────────────────

#[tokio::test]
async fn a_long_table_repeats_its_header_on_every_page_it_runs_onto() {
    let world = World::new();
    let readings: Vec<(String, String)> = (1..=60).map(|i| (format!("TML-{i:03}"), format!("{}.{} mm", 8 + i % 3, i % 10))).collect();
    let sheet = world.remember(
        MemoryKind::Fact,
        &format!("UT thickness survey sheet for V-101: {}", readings.iter().map(|(p, r)| format!("{p} {r}")).collect::<Vec<_>>().join("; ")),
    );
    let rows: Vec<Vec<String>> = readings.iter().map(|(p, r)| vec![p.clone(), format!("{r} [M:{}@1]", sheet.item_id)]).collect();
    let composed = call(
        &world.deps,
        "document.compose",
        json!({
            "template": "authored_document@1",
            "title": "V-101 thickness survey, August",
            "audience": "Inspection engineer",
            "output": "v101-survey.docx",
            "sections": [
                { "id": "readings", "heading": "Readings", "blocks": [
                    { "kind": "table", "header": ["Location", "Thickness"], "rows": rows, "caption": format!("Table 1: UT readings [M:{}@1]", sheet.item_id) },
                    { "kind": "pageBreak" },
                    { "kind": "paragraph", "text": format!("The survey sheet is the source of every reading above [M:{}@1].", sheet.item_id) }
                ] }
            ],
        }),
    )
    .await
    .expect("composed");
    let reference = registered(&composed);
    let validated = call(&world.deps, "artifact.validate_document", json!({ "artifact": reference })).await.unwrap();
    println!("--- long table ---\n{validated}");
    assert!(validated.contains("- document.long_tables [Major]: pass"), "{validated}");
    if renderers_usable() {
        assert!(validated.contains("- render.table_headers [Major]: pass — 1 table(s) run over pages"), "{validated}");
        assert!(validated.contains("- render.page_breaks [Minor]: pass"), "{validated}");
    } else {
        assert!(validated.contains("- render.table_headers [Blocking]: unavailable"), "{validated}");
    }
    world.keep(&reference, "long-table");
}

// ── Exact-version patches ────────────────────────────────────────────────

#[tokio::test]
async fn a_patch_needs_the_exact_version_and_hash_and_changes_one_section_and_nothing_else() {
    let world = World::new();
    let e = evidence(&world).await;
    let v1 = registered(&call(&world.deps, "document.compose", note_spec(&e)).await.unwrap());
    let sha1 = world.sha(&v1);

    // The wrong hash, then a version without one: both refused, nothing written.
    let stale = call(&world.deps, "document.patch_section", json!({ "artifact": v1, "sha256": "0".repeat(64), "section": "recommendation",
        "blocks": [{ "kind": "paragraph", "text": "x" }] })).await.expect_err("stale");
    assert!(stale.contains("stale write refused"), "{stale}");
    let unversioned = call(&world.deps, "document.patch_section", json!({ "artifact": v1.split('@').next().unwrap(), "sha256": sha1, "section": "recommendation" }))
        .await
        .expect_err("no version");
    assert!(unversioned.contains("exact version"), "{unversioned}");

    // The one-section edit.
    let decision = world.remember(MemoryKind::Decision, "Repair point C by weld overlay before restart; re-survey afterwards");
    let patched = call(&world.deps, "document.patch_section", json!({
        "artifact": v1, "sha256": sha1, "section": "recommendation",
        "blocks": [{ "kind": "paragraph", "text": format!("Repair point C by weld overlay before restart, then re-survey [M:{}@1].", decision.item_id) }],
    }))
    .await
    .expect("patched");
    println!("--- document.patch_section ---\n{patched}");
    let v2 = written(&patched);
    assert!(v2.ends_with("@2"), "{v2}");
    assert!(patched.contains("generated references were rewritten"), "the new citation reaches the references: {patched}");

    // v1 is untouched and v1 again is now stale.
    assert_eq!(world.sha(&v1), sha1);
    let again = call(&world.deps, "document.patch_section", json!({ "artifact": v1, "sha256": sha1, "section": "subject",
        "blocks": [{ "kind": "paragraph", "text": "Another subject line for the note." }] })).await.expect_err("stale");
    assert!(again.contains("has moved on to version 2"), "{again}");

    // Byte for byte: every part but the body, and the body outside the two
    // rewritten sections.
    let (before, after) = (world.bytes(&v1), world.bytes(&v2));
    let parts = |bytes: &[u8]| {
        use std::io::Read;
        let mut archive = zip::ZipArchive::new(std::io::Cursor::new(bytes)).unwrap();
        (0..archive.len())
            .map(|i| {
                let mut file = archive.by_index(i).unwrap();
                let mut out = Vec::new();
                file.read_to_end(&mut out).unwrap();
                (file.name().to_string(), out)
            })
            .collect::<std::collections::BTreeMap<_, _>>()
    };
    let (a, b) = (parts(&before), parts(&after));
    assert_eq!(a.keys().collect::<Vec<_>>(), b.keys().collect::<Vec<_>>());
    for (name, content) in &a {
        if name != "word/document.xml" {
            assert_eq!(content, &b[name], "{name} changed");
        }
    }
    let sections = |bytes: &[u8]| {
        let xml = crate::artifacts::package::read_part(bytes, "word/document.xml").unwrap();
        crate::artifacts::authoring::read_sections(&xml).unwrap().into_iter().map(|s| (s.id, s.text)).collect::<std::collections::BTreeMap<_, _>>()
    };
    let (sa, sb) = (sections(&before), sections(&after));
    for id in ["subject", "findings", "calculation", "assumptions", "provenance"] {
        assert_eq!(sa[id], sb[id], "{id} changed");
    }
    assert_ne!(sa["recommendation"], sb["recommendation"]);

    // The diff agrees, the new version derives from the old, and it renders.
    let diff = call(&world.deps, "artifact.diff", json!({ "artifact": v2 })).await.unwrap();
    println!("--- artifact.diff ---\n{diff}");
    let manifest = call(&world.deps, "artifact.manifest", json!({ "artifact": v2 })).await.unwrap();
    assert!(manifest.contains(&format!("\"derivedFrom\": \"{v1}\"")), "{manifest}");
    assert!(manifest.contains(&decision.item_id), "the patched section's lineage names the decision: {manifest}");
    let rendered = call(&world.deps, "document.render_pages", json!({ "artifact": v2 })).await.unwrap();
    println!("--- document.render_pages ---\n{rendered}");
    if renderers_usable() {
        assert!(rendered.contains("- render.text_visible [Blocking]: pass"), "{rendered}");
    }
    world.keep(&v1, "patch-before");
    world.keep(&v2, "patch-after");
}

// ── The compatible path ──────────────────────────────────────────────────

#[tokio::test]
async fn a_compatible_request_goes_through_the_approval_note_writer_and_is_patchable_by_field_key() {
    let world = World::new();
    let e = evidence(&world).await;
    let mut spec = note_spec(&e);
    spec["template"] = json!("approval_note@1");
    spec["sections"].as_array_mut().unwrap().push(json!({ "id": "references", "blocks": [{ "kind": "paragraph", "text": "SOP-114 rev D; UT survey, August" }] }));
    let composed = call(&world.deps, "document.compose", spec).await.expect("composed through the legacy writer");
    assert!(composed.contains("compatible approval_note@1 writer"), "{composed}");
    let reference = registered(&composed);
    let record = world.deps.conversation_artifacts.get(OWNER, reference.split('@').next().unwrap(), Some(1)).unwrap().unwrap();
    assert_eq!(record.template.as_ref().map(|t| t.id.as_str()), Some("approval_note"));
    let patched = call(&world.deps, "document.patch_section", json!({
        "artifact": reference, "sha256": record.sha256, "section": "assumptions",
        "blocks": [{ "kind": "paragraph", "text": "The instrument was calibrated on the survey day." }],
    }))
    .await
    .expect("patched by field key");
    assert!(patched.contains("typed markers"), "{patched}");
}

// ── Through delegation, with a correction during generation ──────────────

/// The parent plans a writing step, delegates it to the production
/// document-author (a person approves: it writes), and the worker composes
/// through its tool port from the task's memory. A person then corrects the
/// reading the note rests on. The first version's check finds the stale
/// citation; the coordinator recomputes the wall loss from the correction;
/// a repair job patches exactly the sections whose sources moved, each as a
/// new version, and the result validates.
#[tokio::test]
async fn a_delegated_author_writes_the_note_and_repairs_it_after_a_correction() {
    let world = World::new();
    let e = evidence(&world).await;
    call(&world.deps, "task.plan_update", json!({
        "base_version": 0,
        "operations": [
            {"op": "set_goal", "text": "Write an approval note on V-101 shell wall loss at point C"},
            {"op": "add_step", "id": "write", "title": "Write the approval note", "kind": "delegate", "role": "document-author"},
            {"op": "add_step", "id": "repair", "title": "Repair the note after review", "kind": "delegate", "role": "document-author", "depends_on": ["write"]},
        ]
    }))
    .await
    .expect("plan");

    let dispatched = approved(&world.deps, "agent.delegate", json!({
        "step": "write", "role": "document-author",
        "objective": "Write an approval note on V-101 shell wall loss at point C",
        "deliverable": "a cited approval note for the plant manager",
        "expressions": [format!("[C:{}]", e.calc)],
        "wait_seconds": 60,
    }))
    .await
    .expect("dispatched");
    let job = world.settled(&job_id_in(&dispatched)).await;
    println!("--- write job ---\n{:#?}", job.result);
    assert_eq!(job.status, JobStatus::Completed, "{:?}", job.result);
    assert_eq!(job.mode, "writer");
    let result = job.result.clone().unwrap();
    let artifacts = result["artifacts"].as_array().cloned().unwrap_or_default();
    let v1 = artifacts.iter().filter_map(|a| a.as_str()).find(|a| a.starts_with("art-")).map(|a| a.split_whitespace().next().unwrap().to_string()).expect("the job names its version");
    println!("write produced {v1}");
    let text = crate::artifacts::content::extract(&world.bytes(&v1)).unwrap().plain_text();
    assert!(text.contains(&format!("[M:{}@1]", e.reading.item_id)), "{text}");
    assert!(text.contains(&format!("[C:{}]", e.calc)), "{text}");
    assert!(text.contains("Plant Manager, Unit 4"), "the recipient a person recorded: {text}");
    // The child's calls ran under its own plan, on its own run, with receipts.
    let child_calls: Vec<String> = world
        .deps
        .calls
        .lock()
        .unwrap()
        .iter()
        .filter(|(run, _)| run.as_str() != RUN)
        .flat_map(|(_, calls)| calls.iter().map(|c| format!("{} {:?} event {:?}", c.tool, c.outcome, c.event_seq)))
        .collect();
    println!("--- the author's own calls ---\n{}", child_calls.join("\n"));
    assert!(child_calls.iter().any(|c| c.starts_with("document.compose Succeeded event Some(")), "{child_calls:?}");

    // A person corrects the reading while the note waits for review.
    let corrected = world.correct(&e.reading, "UT survey of V-101, August, corrected after recalibration: governing shell reading at point C is 7.9 mm, pitting noted");
    let checked = call(&world.deps, "artifact.validate_document", json!({ "artifact": v1, "render": "no" })).await.unwrap();
    println!("--- v1 after the correction ---\n{checked}");
    assert!(checked.contains("- document.citations_resolve [Blocking]: FAIL"), "{checked}");
    assert!(checked.contains("Not accepted"), "{checked}");

    // The coordinator recomputes from the correction (the engine, not a model).
    let minimum = &e.minimum;
    call(&world.deps, "calculation.evaluate_with_units", json!({
        "expression": "wall_loss = (t_min - t_meas) / t_min",
        "inputs": [
            format!("t_min = 9.0 mm [M:{}@1]", minimum.item_id),
            format!("t_meas = 7.9 mm [M:{}@1]", corrected.item_id),
        ],
        "resultUnit": "%",
    }))
    .await
    .expect("recomputed");

    // The repair job, pointed at the exact version.
    let repaired = approved(&world.deps, "agent.delegate", json!({
        "step": "repair", "role": "document-author",
        "objective": "Repair the V-101 approval note after the corrected UT reading",
        "artifacts": [{ "id": v1 }],
        "wait_seconds": 60,
    }))
    .await
    .expect("dispatched");
    let job = world.settled(&job_id_in(&repaired)).await;
    println!("--- repair job ---\n{:#?}", job.result);
    assert_eq!(job.status, JobStatus::Completed, "{:?}", job.result);
    let result = job.result.clone().unwrap();
    let latest = result["artifacts"].as_array().unwrap().iter().filter_map(|a| a.as_str()).map(|a| a.split_whitespace().next().unwrap().to_string()).last().expect("a repaired version");
    assert_ne!(latest, v1, "a repair is a new version");
    let manifest = call(&world.deps, "artifact.manifest", json!({ "artifact": latest })).await.unwrap();
    println!("--- manifest of {latest} ---\n{manifest}");
    assert!(manifest.contains(&corrected.item_id), "the findings now rest on the correction: {manifest}");
    assert!(!manifest.contains(&format!("[M:{}@1]", e.reading.item_id)), "the superseded reading is no longer cited: {manifest}");
    let text = crate::artifacts::content::extract(&world.bytes(&latest)).unwrap().plain_text();
    assert!(text.contains("7.9 mm") && !text.contains("8.2 mm"), "{text}");

    // The repaired version validates, and the chain of versions is in the graph.
    let validated = call(&world.deps, "artifact.validate_document", json!({ "artifact": latest })).await.unwrap();
    println!("--- {latest} validated ---\n{validated}");
    assert!(validated.contains("- document.citations_resolve [Blocking]: pass"), "{validated}");
    assert!(validated.contains("- document.figures_supported [Blocking]: pass"), "{validated}");
    let (id, version) = latest.rsplit_once('@').unwrap();
    let record = world.deps.conversation_artifacts.get(OWNER, id, Some(version.parse().unwrap())).unwrap().unwrap();
    let edges = world.edges_from(record.graph_item_id.as_deref().expect("linked"));
    println!("--- graph edges of {latest} ---");
    for edge in &edges {
        println!("{} -{}-> {}", edge.from_item, edge.kind.as_str(), edge.to_item);
    }
    assert!(edges.iter().any(|edge| edge.kind == EdgeKind::DerivedFrom && edge.to_item.starts_with("mi-art-")), "derived from the previous version");
    assert!(edges.iter().any(|edge| edge.to_item == corrected.item_id), "rests on the correction");
    world.keep(&v1, "delegated-v1");
    world.keep(&latest, "delegated-repaired");
}

/// Without a decision or a recipient in memory the author writes neither:
/// each is a gap, the job is partial, and the gaps are open questions.
#[tokio::test]
async fn a_delegated_author_reports_what_the_memory_lacks_as_gaps_and_a_partial_result() {
    let world = World::new();
    world.remember(MemoryKind::Fact, "UT survey of V-101, August: governing shell reading at point C is 8.2 mm, pitting noted");
    call(&world.deps, "task.plan_update", json!({
        "base_version": 0,
        "operations": [
            {"op": "set_goal", "text": "Write an approval note on V-101 shell wall loss"},
            {"op": "add_step", "id": "write", "title": "Write the approval note", "kind": "delegate", "role": "document-author"},
        ]
    }))
    .await
    .expect("plan");
    let dispatched = approved(&world.deps, "agent.delegate", json!({
        "step": "write", "role": "document-author",
        "objective": "Write an approval note on V-101 shell wall loss", "wait_seconds": 60,
    }))
    .await
    .expect("dispatched");
    let job = world.settled(&job_id_in(&dispatched)).await;
    println!("--- partial job ---\n{:#?}", job.result);
    assert_eq!(job.status, JobStatus::Partial, "{:?}", job.result);
    let result = job.result.clone().unwrap();
    let missing: Vec<String> = result["missing"].as_array().unwrap().iter().filter_map(|m| m.as_str().map(str::to_string)).collect();
    assert!(missing.iter().any(|m| m.starts_with("the recipient")), "{missing:?}");
    assert!(missing.iter().any(|m| m.starts_with("a recommendation")), "{missing:?}");
    // And the document says so where the recommendation would be, rather
    // than saying anything a person did not decide.
    let version = result["artifacts"].as_array().unwrap().iter().filter_map(|a| a.as_str()).next().expect("a version").to_string();
    let xml = crate::artifacts::package::read_part(&world.bytes(&version), "word/document.xml").unwrap();
    let recommendation = crate::artifacts::authoring::read_sections(&xml).unwrap().into_iter().find(|s| s.id == "recommendation").unwrap();
    assert!(recommendation.gap && recommendation.text.contains("INFORMATION NEEDED: a recommendation"), "{}", recommendation.text);
}

/// The author cannot publish its own note: the gateway refuses a tool its
/// narrowed plan does not hold, whoever asks.
#[tokio::test]
async fn the_author_cannot_register_its_own_note_as_final() {
    let world = World::new();
    let profiles = load_profiles(&std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../agents")).profiles;
    let author = profiles.iter().find(|p| p.name == "document-author").expect("the profile ships");
    assert!(!author.allowed_tools.contains(&ToolName::ArtifactRegisterVersion));
    assert!(author.disallowed_tools.contains(&ToolName::ArtifactRegisterVersion));
    let port = super::tool_port::RuntimeToolPort::new(&world.deps);
    use crate::subagents::tool_port::ToolPort;
    port.open("child-author", RUN, world.dir.path(), &author.allowed_tools, 6, std::time::Duration::from_secs(60)).unwrap();
    let refused = port
        .call("child-author", ToolName::ArtifactRegisterVersion, json!({ "artifact": "art-x@1", "stage": "final" }))
        .await
        .expect_err("refused");
    println!("--- register_version from the author's plan ---\n{refused}");
    assert!(refused.contains("refused"), "{refused}");
    port.close("child-author");
}
