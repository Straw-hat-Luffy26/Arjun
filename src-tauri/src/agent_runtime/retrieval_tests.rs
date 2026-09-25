//! P07 end to end: the knowledge connector, hybrid retrieval, the four
//! retrieval tools, the Knowledge Retriever worker and the per-round context
//! compiler, driven the way the main run drives them.
//!
//! ## Evidence labels
//!
//! - The connector runs the **real** document sidecar (`documents::DocumentService`)
//!   over a folder of labelled Markdown fixtures, the real index, the real
//!   version history and the real memory graph: **deterministic test**.
//! - The semantic half runs the real client ([`LocalEmbedder`]), the real
//!   provider ([`EndpointEmbeddings`]), the real windowing, qualification,
//!   fusion and dense floor — against a **fixture embedding server** whose
//!   vectors come from a small concept lexicon written in this file. That is
//!   **deterministic transport**: it proves prefixes are sent, windows are
//!   halved on refusal, vectors are stored per space, fused, floored and cited.
//!   It proves nothing about any model's quality. The lexicon is not a model.
//! - Model quality (Recall@k of a real multilingual-e5-small, nomic or Qwen3
//!   embedding on this corpus) is a target-machine gate: `tests/retrieval_live.rs`.

use std::collections::BTreeSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use serde_json::{json, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

use super::*;
use crate::identity::{Role, Session, User};
use crate::knowledge::connector::{Collection, SourceKind};
use crate::knowledge::embedding::{
    qualify, EmbeddingIdentity, QualificationProbe, MULTILINGUAL_E5_SMALL,
};
use crate::knowledge::graph::runtime_memory::{
    EdgeKind, ItemStatus, MemoryEdge, MemoryKind, MemoryScope, Provenance, SourceRef,
};
use crate::knowledge::graph::runtime_store::MemoryGraph;
use crate::knowledge::provider::{EmbeddingSource, EndpointEmbeddings};
use crate::knowledge::service::{NotebookCorpus, RetrievalService};
use crate::knowledge::CollectionStore;
use crate::policy::Classification;
use crate::registry::{ModelManifest, ModelRegistry, ModelRole};
use crate::subagents::{load_profiles, SpecialistWorker, SubagentManager, WorkerServices};

const RUN: &str = "r";

fn admin() -> Session {
    Session::open(User::new("priya", "Priya Sharma", vec![Role::Employee, Role::Administrator]))
}

fn employee() -> Session {
    Session::open(User::new("kiran", "Kiran Rao", vec![Role::Employee]))
}

// ── The fixture embedding server ─────────────────────────────────────────────

/// Concept → the markers that put a text on that concept's axis.
///
/// A transport double's lookup table, and nothing more: it is how this file
/// says "this question is about the same thing as that passage" without a
/// model. English and Hindi markers sit on the same axis so the multilingual
/// plumbing can be exercised; whether a real model does that is measured on
/// the target machine.
const LEXICON: &[(usize, &[&str])] = &[
    (1, &["cavitat", "npsh", "suction head", "कैविटेशन"]),
    (2, &["safety valve", "psv", "lift at", "सुरक्षा वाल्व"]),
    (3, &["isolat", "block valve", "bleed", "flange"]),
    (4, &["mechanical seal", "seal cartridge", "सील"]),
    (5, &["hot work", "welding", "countersign", "gas test"]),
    (6, &["retirement thickness", "corroded", "spool"]),
    (7, &["flare", "nitrogen purge"]),
    (8, &["anti-surge", "surge-line"]),
    (9, &["overtime"]),
    (10, &["cooling tower", "drift eliminator"]),
];

fn fixture_vector(text: &str) -> Vec<f32> {
    let lower = text.to_lowercase();
    let mut vector = vec![0.0f32; 384];
    // Every text shares a small common component, so no vector is zero and
    // unrelated texts have a small, non-zero similarity -- as real ones do.
    vector[383] = 0.05;
    for (axis, markers) in LEXICON {
        if markers.iter().any(|marker| lower.contains(marker)) {
            vector[*axis] += 1.0;
        }
    }
    vector
}

/// An OpenAI-compatible `/v1/embeddings` on loopback that refuses an input
/// longer than `limit` characters, as llama-server refuses one over its batch.
struct EmbeddingServer {
    base_url: String,
    inputs: Arc<Mutex<Vec<String>>>,
    refused: Arc<AtomicUsize>,
}

impl EmbeddingServer {
    async fn start(limit: usize) -> Self {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.expect("a port");
        let base_url = format!("http://{}/v1", listener.local_addr().unwrap());
        let inputs: Arc<Mutex<Vec<String>>> = Arc::default();
        let refused: Arc<AtomicUsize> = Arc::default();
        let (seen, turned_away) = (inputs.clone(), refused.clone());
        tokio::spawn(async move {
            loop {
                let Ok((mut socket, _)) = listener.accept().await else { return };
                let (seen, turned_away) = (seen.clone(), turned_away.clone());
                tokio::spawn(async move {
                    let mut raw = Vec::new();
                    let mut buffer = [0u8; 65536];
                    let (head_end, length) = loop {
                        let read = socket.read(&mut buffer).await.unwrap_or(0);
                        if read == 0 {
                            return;
                        }
                        raw.extend_from_slice(&buffer[..read]);
                        if let Some(at) = raw.windows(4).position(|w| w == b"\r\n\r\n") {
                            let head = String::from_utf8_lossy(&raw[..at]).to_ascii_lowercase();
                            let length = head
                                .lines()
                                .find_map(|l| l.strip_prefix("content-length:"))
                                .and_then(|v| v.trim().parse::<usize>().ok())
                                .unwrap_or(0);
                            break (at + 4, length);
                        }
                    };
                    while raw.len() < head_end + length {
                        let read = socket.read(&mut buffer).await.unwrap_or(0);
                        if read == 0 {
                            break;
                        }
                        raw.extend_from_slice(&buffer[..read]);
                    }
                    let body: Value = serde_json::from_slice(&raw[head_end..]).unwrap_or(Value::Null);
                    let texts: Vec<String> = body["input"]
                        .as_array()
                        .map(|items| items.iter().filter_map(|i| i.as_str().map(str::to_string)).collect())
                        .unwrap_or_default();
                    seen.lock().unwrap().extend(texts.iter().cloned());
                    let response = if texts.iter().any(|text| text.chars().count() > limit) {
                        turned_away.fetch_add(1, Ordering::SeqCst);
                        let payload = json!({"error": {"message": "input is too large to process. increase the physical batch size"}}).to_string();
                        format!(
                            "HTTP/1.1 500 Internal Server Error\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{payload}",
                            payload.len()
                        )
                    } else {
                        // Out of order on purpose: the client must pair by index.
                        let data: Vec<Value> = texts
                            .iter()
                            .enumerate()
                            .rev()
                            .map(|(index, text)| json!({"index": index, "embedding": fixture_vector(text)}))
                            .collect();
                        let payload = json!({"data": data}).to_string();
                        format!(
                            "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{payload}",
                            payload.len()
                        )
                    };
                    let _ = socket.write_all(response.as_bytes()).await;
                    let _ = socket.shutdown().await;
                });
            }
        });
        Self { base_url, inputs, refused }
    }
}

/// Probes written against the fixture lexicon — the qualification this
/// transport can meaningfully be put through. The bundled probes are for real
/// models.
fn fixture_probes() -> Vec<QualificationProbe> {
    let probe = |id: &str, query: &str, relevant: &str, distractor: &str| QualificationProbe {
        id: id.into(),
        language: "en".into(),
        query: query.into(),
        relevant: relevant.into(),
        distractor: distractor.into(),
    };
    vec![
        probe("f-cavitation", "Why does the pump cavitate?", "Keep NPSH margin above 1.5 m.", "The pump is painted green."),
        probe("f-relief", "When does the PSV open?", "The safety valve is set to lift at 14 barg.", "The vessel is painted grey."),
        probe("f-seal", "How long does a mechanical seal last?", "Renew the mechanical seal every 8000 h.", "The workshop opens at eight."),
        probe("f-flare", "What purges the flare?", "A nitrogen purge keeps oxygen out of the flare.", "The drawing was revised."),
    ]
}

// ── The world ────────────────────────────────────────────────────────────────

fn model_registry() -> ModelRegistry {
    let mut parent = crate::registry::tests::entry("model-parent", 9.0, vec![ModelRole::Reasoning]);
    parent.supports_structured_output = true;
    ModelRegistry::from_manifest(
        ModelManifest { models: vec![parent] },
        std::path::PathBuf::from("models/registry.json"),
    )
    .expect("a well-formed manifest")
}

fn corpus_source() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/knowledge/testdata/retrieval")
}

fn copy_tree(from: &std::path::Path, to: &std::path::Path) {
    std::fs::create_dir_all(to).unwrap();
    for entry in std::fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        let target = to.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            copy_tree(&entry.path(), &target);
        } else {
            std::fs::copy(entry.path(), target).unwrap();
        }
    }
}

struct World {
    deps: Arc<RuntimeDeps>,
    base: Arc<RuntimeDeps>,
    graph: Arc<MemoryGraph>,
    retrieval: Arc<RetrievalService>,
    server: EmbeddingServer,
    collections: Arc<CollectionStore>,
    dir: tempfile::TempDir,
}

impl World {
    /// `qualified`: whether the fixture embedder is put through qualification
    /// (and so used), or left compatible and unmeasured (and so not used).
    async fn new(qualified: bool) -> Self {
        let server = EmbeddingServer::start(700).await;
        let session = Arc::new(std::sync::RwLock::new(Some(admin())));
        let (base, dir) = super::tests::deps_with(session.clone());
        let retrieval_dir = dir.path().join("retrieval");
        let identity = EmbeddingIdentity::new("fixture-e5", &MULTILINGUAL_E5_SMALL, &"f1".repeat(32));
        let source = EndpointEmbeddings::new(server.base_url.clone(), identity, &retrieval_dir).expect("loopback");
        if qualified {
            let embedder = source.open(true).await.expect("opens for qualification");
            let record = qualify(embedder.as_ref(), &fixture_probes(), &server.base_url).await;
            assert!(record.passed, "the fixture transport failed qualification: {:#?}", record.checks);
            source.store().save(&record).expect("saved");
        }
        let retrieval = Arc::new(
            RetrievalService::new(base.index.clone(), Arc::new(source)).with_notebooks(NotebookCorpus {
                notebooks: base.notebooks.clone(),
                documents: base.documents.clone(),
            }),
        );

        // The corpus, as two collections on disk.
        let corpus = dir.path().join("corpus");
        for folder in ["sops", "manuals", "correspondence"] {
            copy_tree(&corpus_source().join(folder), &corpus.join("plant").join(folder));
        }
        copy_tree(&corpus_source().join("hr"), &corpus.join("hr"));
        let collections = Arc::new(CollectionStore::open(dir.path()).expect("collections"));
        for (id, restricted) in [("plant", Vec::new()), ("hr", vec![Role::Administrator])] {
            collections
                .upsert(Collection {
                    id: id.into(),
                    name: id.into(),
                    kind: SourceKind::LocalFolder,
                    root: corpus.join(id),
                    owner: "Document control".into(),
                    classification: Classification::Internal,
                    restricted_to_roles: restricted,
                    retention_days: None,
                    enabled: true,
                })
                .expect("saved");
        }

        let graph = Arc::new(MemoryGraph::in_memory().expect("a graph").with_receipts(base.events.clone()));
        let registry = Arc::new(model_registry());
        let cancellations = Arc::new(crate::agent_runtime::cancellation::RunCancellations::new());
        let services = Arc::new(WorkerServices {
            index: base.index.clone(),
            graph: Some(graph.clone()),
            events: base.events.clone(),
            scheduler: Arc::new(crate::subagents::ModelScheduler::new(
                registry.clone(),
                Arc::new(crate::serving::ModelServers::new()),
            )),
            session: base.session.clone(),
            child_loop: None,
            cancellations: cancellations.clone(),
            analyst: None,
            calculation_store: Arc::new(crate::calculation::CalculationStore::in_memory().expect("a calculation store")),
            retrieval: retrieval.clone(),
        });
        let profiles_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../agents");
        let profiles = load_profiles(&profiles_dir).profiles;
        let mut manager = SubagentManager::new(profiles.clone(), base.events.clone()).with_cancellations(cancellations);
        for worker in SpecialistWorker::register_all(&profiles, services) {
            manager = manager.with_worker(worker);
        }
        base.checkpoints.lock().unwrap().insert(
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
        base.run_to_conversation.bind(RUN, "conv-p07");
        let deps = rebuilt(&base, Some(graph.clone()), Some(registry), Arc::new(manager), retrieval.clone(), session);
        World { deps, base, graph, retrieval, server, collections, dir }
    }

    /// The same runtime, as somebody else.
    fn as_user(&self, session: Session) -> Arc<RuntimeDeps> {
        rebuilt(
            &self.base,
            Some(self.graph.clone()),
            self.deps.registry.clone(),
            self.deps.subagents.clone(),
            self.retrieval.clone(),
            Arc::new(std::sync::RwLock::new(Some(session))),
        )
    }

    /// Syncs one collection through the real document sidecar.
    fn sync(&self, id: &str) -> crate::knowledge::ingest::SyncReport {
        let collection = self.collections.get(id).unwrap().expect("defined");
        let reader = crate::documents::DocumentService::spawn(self.dir.path()).expect("the document sidecar starts");
        let originals = crate::documents::DocumentStore::open(self.dir.path()).expect("the original store opens");
        crate::knowledge::ingest::sync_collection(
            &collection,
            &self.collections,
            &reader,
            &originals,
            &self.deps.index,
            Some(&self.graph),
            None,
            "priya",
        )
        .expect("the sync runs")
    }

    async fn sync_all(&self) {
        for id in ["plant", "hr"] {
            let report = self.sync(id);
            assert!(report.outcome.failures.is_empty(), "{id}: {:#?}", report.outcome.failures);
        }
        if self.retrieval.status().state == crate::knowledge::provider::ProviderState::Qualified {
            let pass = self.retrieval.reindex_once(10_000).await.expect("the embedding pass runs");
            assert_eq!(pass.remaining, 0, "{pass:?}");
        }
    }

    fn sha_of(&self, relative: &str) -> String {
        crate::extraction::regions::sha256_file(&self.dir.path().join("corpus/plant").join(relative)).unwrap()
    }
}

#[allow(clippy::too_many_arguments)]
fn rebuilt(
    base: &Arc<RuntimeDeps>,
    graph: Option<Arc<MemoryGraph>>,
    registry: Option<Arc<ModelRegistry>>,
    subagents: Arc<SubagentManager>,
    retrieval: Arc<RetrievalService>,
    session: Arc<std::sync::RwLock<Option<Session>>>,
) -> Arc<RuntimeDeps> {
    Arc::new(RuntimeDeps {
        memory_graph: graph,
        conversation_artifacts: base.conversation_artifacts.clone(),
        registry,
        index: base.index.clone(),
        session,
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
        subagents,
        multimodal: base.multimodal.clone(),
        audit_health: base.audit_health.clone(),
        documents: base.documents.clone(),
        run_to_conversation: base.run_to_conversation.clone(),
        notebooks: base.notebooks.clone(),
        jobs: base.jobs.clone(),
        extraction: base.extraction.clone(),
        retrieval,
        calculation_store: base.calculation_store.clone(),
    })
}

fn request(tool: &str, args: Value) -> Value {
    json!({
        "runId": RUN,
        "toolCallId": format!("tc-{tool}-{}", uuid::Uuid::new_v4()),
        "tool": tool,
        "args": args,
    })
}

async fn call(deps: &Arc<RuntimeDeps>, tool: &str, args: Value) -> Result<String, String> {
    let request = request(tool, args);
    let allow = authorize(request.clone(), deps).await.map_err(|e| format!("authorise: {}", e.message))?;
    let grant = allow
        .get("grant")
        .and_then(Value::as_str)
        .ok_or_else(|| format!("refused: {allow}"))?
        .to_string();
    let mut spent = request;
    spent["grant"] = json!(grant);
    let result = execute(spent, deps).await.map_err(|e| e.message)?;
    Ok(result["text"].as_str().unwrap_or_default().to_string())
}

/// The first `[En] … · found by …` line of a hybrid rendering.
fn first_hit(rendered: &str) -> &str {
    rendered.lines().find(|line| line.starts_with("[E")).unwrap_or("")
}

// ── The connector ────────────────────────────────────────────────────────────

#[tokio::test]
async fn the_connector_reads_a_real_folder_into_versioned_authorised_passages() {
    let world = World::new(false).await;
    let plant = world.sync("plant");
    println!("--- sync plant ---\n{plant:#?}");
    assert_eq!(plant.outcome.documents_read, 8, "{plant:#?}");
    assert!(plant.outcome.chunks_indexed >= 12, "{plant:#?}");
    assert!(world.deps.index.current_paths("plant").unwrap().len() == 8);

    // Nothing changed: the second sync reads nothing again.
    let again = world.sync("plant");
    assert_eq!(again.outcome.documents_read, 0, "{again:#?}");

    // The whole path, literal: keyword search finds the tag.
    let found = call(&world.deps, "knowledge.search_authorized", json!({"query": "PSV-1011"}))
        .await
        .expect("searches");
    assert!(found.contains("vessel-V101.md"), "{found}");
}

// ── Literal, paraphrase, multilingual ────────────────────────────────────────

#[tokio::test]
async fn a_literal_query_is_found_by_keyword_and_a_paraphrase_only_by_meaning() {
    let world = World::new(true).await;
    world.sync_all().await;

    let literal = call(&world.deps, "knowledge.hybrid_search", json!({"query": "PSV-1011 set pressure"}))
        .await
        .expect("searches");
    println!("--- literal ---\n{literal}");
    assert!(first_hit(&literal).contains("vessel-V101.md"), "{literal}");
    assert!(first_hit(&literal).contains("keyword"), "{literal}");
    assert!(literal.contains("Retrieval: hybrid"), "{literal}");

    let question = "What stops the charge pump from cavitating?";
    let exact = call(&world.deps, "knowledge.search_authorized", json!({"query": question}))
        .await
        .expect("searches");
    assert!(!exact.contains("NPSH"), "the literal search found the paraphrase, so this proves nothing: {exact}");

    let hybrid = call(&world.deps, "knowledge.hybrid_search", json!({"query": question}))
        .await
        .expect("searches");
    println!("--- paraphrase ---\n{hybrid}");
    let top = first_hit(&hybrid);
    assert!(top.contains("charge-pump-P101.md") && top.contains("Suction conditions"), "{hybrid}");
    assert!(top.contains("semantic"), "{hybrid}");
    assert!(hybrid.contains("semantic cosine"), "the score is not labelled: {hybrid}");

    // Every stored passage and every question went out with its prefix.
    let sent = world.server.inputs.lock().unwrap().clone();
    assert!(sent.iter().any(|input| input == &format!("query: {question}")), "{sent:?}");
    assert!(sent.iter().filter(|input| !input.starts_with("query: ")).all(|input| input.starts_with("passage: ")));
}

#[tokio::test]
async fn a_hindi_question_reaches_the_english_passage_through_the_semantic_half() {
    let world = World::new(true).await;
    world.sync_all().await;
    let hybrid = call(
        &world.deps,
        "knowledge.hybrid_search",
        json!({"query": "पोत का सुरक्षा वाल्व किस दबाव पर खुलता है?"}),
    )
    .await
    .expect("searches");
    println!("--- multilingual (transport) ---\n{hybrid}");
    assert!(first_hit(&hybrid).contains("vessel-V101.md"), "{hybrid}");
    assert!(first_hit(&hybrid).contains("semantic"), "{hybrid}");
}

#[tokio::test]
async fn an_unqualified_model_is_not_used_and_the_search_says_why() {
    let world = World::new(false).await;
    world.sync_all().await;
    let hybrid = call(&world.deps, "knowledge.hybrid_search", json!({"query": "What stops the charge pump from cavitating?"}))
        .await
        .expect("searches");
    println!("--- unqualified ---\n{hybrid}");
    assert!(hybrid.contains("Retrieval: lexical. Keyword only because"), "{hybrid}");
    assert!(hybrid.contains("has not been qualified"), "{hybrid}");
    assert!(world.server.inputs.lock().unwrap().is_empty(), "an unqualified model was sent text");
}

// ── Stale, conflicting, none, oversized ──────────────────────────────────────

#[tokio::test]
async fn a_revised_sop_answers_its_old_version_is_flagged_and_claims_on_it_go_stale() {
    let world = World::new(true).await;
    world.sync_all().await;
    let old_sha = world.sha_of("sops/exchanger-isolation.md");

    // A claim resting on revision A, as a retriever would have published it.
    let mut claim = crate::knowledge::graph::runtime_memory::tests::item(
        MemoryKind::Fact,
        Provenance::Operator { user_id: "priya".into() },
    );
    claim.scope = MemoryScope::Task { task_id: RUN.into() };
    claim.content = "exchanger isolation: vent to atmosphere, per SOP-OPS-014".into();
    claim.sources = vec![SourceRef { sha256: old_sha.clone(), locator: "page 1".into(), extraction_revision: Some("v1".into()) }];
    world.graph.commit(claim.clone(), None, &[]).expect("committed");

    // Revision B lands in the folder, and the next sync reads it.
    std::fs::copy(
        corpus_source().join("revisions/exchanger-isolation-rev-b.md"),
        world.dir.path().join("corpus/plant/sops/exchanger-isolation.md"),
    )
    .unwrap();
    let modified = std::time::SystemTime::now() + std::time::Duration::from_secs(5);
    std::fs::File::options()
        .write(true)
        .open(world.dir.path().join("corpus/plant/sops/exchanger-isolation.md"))
        .unwrap()
        .set_modified(modified)
        .unwrap();
    let report = world.sync("plant");
    assert_eq!(report.outcome.superseded_sha256s, vec![old_sha.clone()], "{report:#?}");
    assert_eq!(report.memory_stale, 1, "{report:#?}");
    world.retrieval.reindex_once(1_000).await.expect("embeds the revision");

    let hybrid = call(&world.deps, "knowledge.hybrid_search", json!({"query": "isolation before opening the exchanger"}))
        .await
        .expect("searches");
    println!("--- stale SOP ---\n{hybrid}");
    assert!(hybrid.contains("closed drain"), "revision B did not answer: {hybrid}");
    assert!(!hybrid.contains("vent to atmosphere. Wait"), "the superseded version still answered: {hybrid}");
    assert!(hybrid.contains("version 2 (current)"), "{hybrid}");

    let version = call(&world.deps, "knowledge.source_version", json!({"documentSha256": old_sha}))
        .await
        .expect("reads");
    println!("--- source_version of revision A ---\n{version}");
    assert!(version.contains("is superseded") && version.contains("This is not the current version"), "{version}");

    let now = world.graph.snapshot(&admin(), &MemoryScope::Task { task_id: RUN.into() }, None).unwrap();
    let stale = now.iter().find(|item| item.item_id == claim.item_id).expect("still held");
    assert_eq!(stale.status, ItemStatus::Stale, "{stale:?}");
}

#[tokio::test]
async fn two_current_sources_that_disagree_are_both_returned_with_their_own_sources() {
    let world = World::new(true).await;
    world.sync_all().await;
    let hybrid = call(
        &world.deps,
        "knowledge.hybrid_search",
        json!({"query": "How often should the P-101 mechanical seal be replaced?"}),
    )
    .await
    .expect("searches");
    println!("--- conflicting sources ---\n{hybrid}");
    assert!(hybrid.contains("8000 running hours"), "{hybrid}");
    assert!(hybrid.contains("6000 running hours"), "{hybrid}");
    assert!(hybrid.contains("charge-pump-P101.md") && hybrid.contains("2026-03-seal-vendor-letter.md"));
}

#[tokio::test]
async fn a_question_nothing_answers_is_said_to_have_no_answer() {
    let world = World::new(true).await;
    world.sync_all().await;
    let hybrid = call(
        &world.deps,
        "knowledge.hybrid_search",
        json!({"query": "What is the drift eliminator spacing in the cooling tower?"}),
    )
    .await
    .expect("searches");
    println!("--- no answer ---\n{hybrid}");
    assert!(hybrid.contains("No passage you may read answers"), "{hybrid}");
    assert!(!hybrid.contains("[E"), "{hybrid}");
}

#[tokio::test]
async fn an_oversized_passage_is_embedded_in_windows_and_found_by_its_last_one() {
    let world = World::new(true).await;
    world.sync_all().await;
    assert!(world.server.refused.load(Ordering::SeqCst) > 0, "no window was ever refused, so halving was not exercised");
    let hybrid = call(&world.deps, "knowledge.hybrid_search", json!({"query": "How fast must the anti-surge valve open?"}))
        .await
        .expect("searches");
    println!("--- oversized passage ---\n{hybrid}");
    let top = first_hit(&hybrid);
    assert!(top.contains("compressor-K301.md"), "{hybrid}");
    assert!(hybrid.contains("within 1.2 seconds"), "the excerpt missed the answering sentence: {hybrid}");
}

// ── Access ───────────────────────────────────────────────────────────────────

#[tokio::test]
async fn a_restricted_collection_and_a_revoked_document_are_neither_retrieved_nor_disclosed() {
    let world = World::new(true).await;
    world.sync_all().await;
    let kiran = world.as_user(employee());
    kiran.plans.lock().unwrap().entry(RUN.into()).or_insert_with(|| {
        crate::orchestrator::plan::PlanRun::new(
            RUN,
            vec!["p07".into()],
            crate::orchestrator::plan::Budget::standard(ToolName::ALL.to_vec()),
        )
    });

    let denied = call(&kiran, "knowledge.hybrid_search", json!({"query": "overtime rate"})).await.expect("searches");
    println!("--- restricted collection, as an employee ---\n{denied}");
    // Whatever else "rate" matches, nothing of the restricted collection is
    // named, excerpted or counted.
    assert!(!denied.contains("overtime-policy.md") && !denied.contains("double the ordinary"), "{denied}");
    assert!(!denied.contains("semantic cosine"), "a sub-floor passage was given a semantic score: {denied}");
    let overtime_sha = crate::extraction::regions::sha256_file(&world.dir.path().join("corpus/hr/overtime-policy.md")).unwrap();
    let version = call(&kiran, "knowledge.source_version", json!({"documentSha256": overtime_sha})).await.unwrap();
    assert!(version.contains("is visible to you"), "{version}");
    let nonexistent = call(&kiran, "knowledge.source_version", json!({"documentSha256": "0".repeat(64)})).await.unwrap();
    assert_eq!(
        version.replace(&overtime_sha, "X"),
        nonexistent.replace(&"0".repeat(64), "X"),
        "a hidden document reads differently from one that does not exist"
    );
    let allowed = call(&world.deps, "knowledge.hybrid_search", json!({"query": "overtime rate"})).await.unwrap();
    assert!(allowed.contains("overtime-policy.md"), "{allowed}");

    // Revocation: a claim on the permit SOP, then the SOP revoked.
    let permit_sha = world.sha_of("sops/hot-work-permit.md");
    let mut claim = crate::knowledge::graph::runtime_memory::tests::item(
        MemoryKind::Fact,
        Provenance::Operator { user_id: "priya".into() },
    );
    claim.scope = MemoryScope::Task { task_id: RUN.into() };
    claim.content = "hot work: area authority and fire watch countersign".into();
    claim.sources = vec![SourceRef { sha256: permit_sha.clone(), locator: "page 1".into(), extraction_revision: Some("v1".into()) }];
    world.graph.commit(claim.clone(), None, &[]).unwrap();
    world.deps.index.revoke_document(&permit_sha).unwrap();
    world.graph.invalidate_source(&permit_sha, "2026-09-25T00:00:00Z").unwrap();
    let after = call(&world.deps, "knowledge.hybrid_search", json!({"query": "hot work permit countersign"})).await.unwrap();
    println!("--- after revocation ---\n{after}");
    assert!(!after.contains("hot-work-permit.md"), "{after}");
    let now = world.graph.snapshot(&admin(), &MemoryScope::Task { task_id: RUN.into() }, None).unwrap();
    assert_eq!(now.iter().find(|i| i.item_id == claim.item_id).unwrap().status, ItemStatus::Rejected);
}

#[tokio::test]
async fn a_run_scoped_to_someone_elses_notebook_does_not_read_it_and_says_so() {
    let world = World::new(false).await;
    world.sync_all().await;
    let theirs = world.deps.notebooks.create("ravi", "Ravi's notes").expect("created");
    world.retrieval.pin(
        RUN,
        crate::knowledge::service::RunScope {
            pinned_revision: None,
            notebook: Some(crate::knowledge::service::NotebookPin {
                notebook_id: theirs.id.clone(),
                source_sha256s: vec!["a".repeat(64)],
            }),
        },
    );
    let hybrid = call(&world.deps, "knowledge.hybrid_search", json!({"query": "flare header purge"})).await.unwrap();
    println!("--- denied notebook ---\n{hybrid}");
    assert!(hybrid.contains("was not searched: that notebook does not exist"), "{hybrid}");
    assert!(hybrid.contains("flare-purge.md"), "the index half should still answer: {hybrid}");
    assert!(!hybrid.contains("Ravi"), "{hybrid}");
}

// ── Queued work, the worker, the compiler, citations ─────────────────────────

#[tokio::test]
async fn a_retrieval_job_is_pinned_when_queued_and_its_findings_reach_the_parents_next_round() {
    let world = World::new(true).await;
    world.sync_all().await;
    let revision = world.deps.index.index_revision().unwrap();

    let written = call(
        &world.deps,
        "task.plan_update",
        json!({"base_version": 0, "operations": [
            {"op": "set_goal", "text": "Find the anti-surge response requirement"},
            {"op": "add_step", "id": "find", "title": "Find the anti-surge valve requirement", "kind": "delegate",
             "role": "knowledge-retriever", "acceptance": ["childCompleted", "memoryPublished"]}
        ]}),
    )
    .await
    .expect("planned");
    assert!(written.contains("Plan version 1"), "{written}");
    let started = call(
        &world.deps,
        "agent.delegate",
        json!({"step": "find", "role": "knowledge-retriever", "objective": "anti-surge valve opening time", "wait_seconds": 60}),
    )
    .await
    .expect("delegated");
    println!("--- agent.delegate ---\n{started}");
    assert!(started.contains("completed"), "{started}");

    // The packet carried the pin, as the manifest event records it.
    let events = world.deps.events.events_since(RUN, 0).unwrap().events;
    let pinned = format!("retrieval scope at index revision {revision}");
    assert!(
        events.iter().any(|event| event.payload.to_string().contains(&pinned)),
        "no child manifest records {pinned:?}"
    );

    // What the worker published: a versioned citation on the search's own
    // receipt, and the search's coverage.
    let memory = world.graph.snapshot(&admin(), &MemoryScope::Task { task_id: RUN.into() }, None).unwrap();
    let citation = memory
        .iter()
        .find(|item| item.kind == MemoryKind::Fact && item.content.contains("compressor-K301.md"))
        .unwrap_or_else(|| panic!("no citation published: {memory:#?}"));
    assert_eq!(citation.status, ItemStatus::Admitted, "{citation:?}");
    assert!(citation.content.contains("[v1 current]"), "{}", citation.content);
    assert_eq!(citation.sources[0].extraction_revision.as_deref(), Some("v1"));
    assert!(memory.iter().any(|item| item.kind == MemoryKind::ToolObservation && item.content.starts_with("search coverage:")));

    // The parent's next round, compiled by the production handler.
    let refreshed = context_refresh_handler(
        json!({"runId": RUN, "taskId": RUN, "agentId": "orchestrator", "modelId": "model-parent",
               "servedWindow": 16384, "question": "How fast must the anti-surge valve open?",
               "reservedToolSchemas": 2000, "reservedOutput": 2048, "reservedFraming": 256}),
        &world.deps,
    )
    .await
    .expect("compiled");
    let blocks = refreshed["blocks"].as_array().unwrap();
    println!("--- context.refresh ---\n{}", serde_json::to_string_pretty(&refreshed).unwrap());
    assert!(
        blocks.iter().any(|b| b["content"].as_str().unwrap_or_default().contains("[v1 current]")),
        "the retriever's published citation did not reach the parent's round"
    );
    let evidence = blocks
        .iter()
        .find(|b| b["kind"] == "evidence")
        .expect("no retrieved passage in the round");
    assert!(evidence["content"].as_str().unwrap().contains("[E"), "{evidence}");
    assert_eq!(refreshed["retrieval"]["mode"], "hybrid", "{}", refreshed["retrieval"]);
    assert!(refreshed["retrieval"]["embeddingModelId"].as_str().unwrap().starts_with("multilingual-e5-small@"));
}

/// The retriever's ceiling is `internal`. A passage the parent's session may
/// read but classified above that ceiling is found, counted, and **not**
/// published into the task's shared memory at the worker's lower
/// classification — not its text, not its document name.
#[tokio::test]
async fn a_passage_above_the_retrievers_ceiling_is_counted_not_published() {
    let world = World::new(false).await;
    world.sync_all().await;
    world
        .deps
        .index
        .index_document(
            "vendor-terms-surge-valve.md",
            Classification::VendorNegotiation,
            &[crate::knowledge::Chunk {
                id: "vendor-0".into(),
                document_sha256: "d".repeat(64),
                ordinal: 0,
                text: "Anti-surge valve UV-301 unit price is 4.2 lakh with a 16 week lead time.".into(),
                page: 1,
                section_path: Vec::new(),
                kind: crate::knowledge::ChunkKind::Prose,
                char_count: 72,
            }],
        )
        .unwrap();
    let direct = call(&world.deps, "knowledge.hybrid_search", json!({"query": "anti-surge valve UV-301"})).await.unwrap();
    assert!(direct.contains("vendor-terms-surge-valve.md"), "the parent may read it: {direct}");
    call(
        &world.deps,
        "task.plan_update",
        json!({"base_version": 0, "operations": [
            {"op": "set_goal", "text": "Find the anti-surge valve information"},
            {"op": "add_step", "id": "find", "title": "Find the anti-surge valve information", "kind": "delegate",
             "role": "knowledge-retriever", "acceptance": ["childCompleted"]}
        ]}),
    )
    .await
    .expect("planned");
    let result = call(
        &world.deps,
        "agent.delegate",
        json!({"step": "find", "role": "knowledge-retriever", "objective": "anti-surge valve UV-301", "wait_seconds": 60}),
    )
    .await
    .expect("delegated");
    println!("--- ceiling ---\n{result}");
    // Counted, in the result the parent reads.
    assert!(result.contains("uncertain: 1 passage(s) found are classified above this worker's"), "{result}");
    let memory = world.graph.snapshot(&admin(), &MemoryScope::Task { task_id: RUN.into() }, None).unwrap();
    assert!(
        memory.iter().all(|item| !item.content.contains("vendor-terms") && !item.content.contains("lakh")),
        "a passage above the ceiling reached shared memory: {memory:#?}"
    );
    assert!(memory.iter().any(|item| item.content.contains("compressor-K301.md")), "the internal passage was not published");
}

#[tokio::test]
async fn a_hybrid_passage_is_cited_exactly_by_marker_and_reranked_locally() {
    let world = World::new(true).await;
    world.sync_all().await;
    let found = call(&world.deps, "knowledge.hybrid_search", json!({"query": "charge pump mechanical seal"})).await.unwrap();
    let marked = crate::agent_runtime::retrieval::for_run(&world.deps.passages, RUN);
    assert!(!marked.is_empty());
    // Each [En] line resolves to exactly the passage the table holds under n.
    for line in found.lines().filter(|line| line.starts_with("[E")) {
        let marker: usize = line[2..line.find(']').unwrap()].parse().unwrap();
        let held = &marked[marker - 1];
        assert!(line.contains(&held.document_name), "[E{marker}] names another passage: {line}");
    }
    let reranked = call(&world.deps, "knowledge.rerank", json!({"query": "mechanical seal replacement interval"})).await.unwrap();
    println!("--- rerank ---\n{reranked}");
    assert!(reranked.contains("lexical-proximity reranker (deterministic, no model)"), "{reranked}");
    assert!(call(&world.deps, "knowledge.rerank", json!({"query": "x", "markers": [99]})).await.is_err());
}

#[tokio::test]
async fn memory_neighbours_walks_only_what_the_reader_may_see() {
    let world = World::new(false).await;
    let make = |content: &str, owner: Option<&str>| {
        let mut item = crate::knowledge::graph::runtime_memory::tests::item(
            MemoryKind::Fact,
            Provenance::Operator { user_id: "priya".into() },
        );
        item.scope = MemoryScope::Task { task_id: RUN.into() };
        item.content = content.into();
        item.acl.owner = owner.map(str::to_string);
        world.graph.commit(item.clone(), None, &[]).unwrap();
        item
    };
    let claim = make("seal interval: 8000 h", None);
    let support = make("seal interval: manual section 3 says 8000 h", None);
    let hidden = make("seal interval: ravi's private note says 6000 h", Some("ravi"));
    for (from, kind) in [(&support, EdgeKind::Supports), (&hidden, EdgeKind::Contradicts)] {
        world
            .graph
            .link(MemoryEdge {
                edge_id: crate::knowledge::graph::runtime_memory::edge_id(&from.item_id, kind, &claim.item_id),
                from_item: from.item_id.clone(),
                to_item: claim.item_id.clone(),
                kind,
                agent_id: "test".into(),
                scope: MemoryScope::Task { task_id: RUN.into() },
                created_at: "2026-09-25T00:00:00Z".into(),
            })
            .unwrap();
    }
    let walked = call(&world.deps, "memory.neighbours", json!({"itemId": claim.item_id, "depth": 2})).await.unwrap();
    println!("--- memory.neighbours ---\n{walked}");
    assert!(walked.contains(&support.item_id) && walked.contains("supports"), "{walked}");
    assert!(!walked.contains(&hidden.item_id) && !walked.contains("contradicts") && !walked.contains("6000"), "{walked}");
    let missing = call(&world.deps, "memory.neighbours", json!({"itemId": hidden.item_id})).await.unwrap();
    let nonexistent = call(&world.deps, "memory.neighbours", json!({"itemId": "mi-nope"})).await.unwrap();
    assert_eq!(missing.replace(&hidden.item_id, "X"), nonexistent.replace("mi-nope", "X"));
}

// ── The P00 pack's SOP: the passages sop-01 must cite ─────────────────────────

/// `sop-01-applicable-minimum` must cite section 3 and the 3.1 carve-out of
/// Revision D, and `sop-02` must surface that two values exist. Composing that
/// answer needs a model (a target run); finding and exactly citing the two
/// passages does not, and is checked here on the pack's own file through the
/// real sidecar, keyword-only.
#[tokio::test]
async fn the_packs_sop_question_retrieves_both_sections_it_must_cite() {
    let world = World::new(false).await;
    let pack = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../fixtures/agent-system/v1/sources/sop");
    world
        .collections
        .upsert(Collection {
            id: "pack-sop".into(),
            name: "P00 pack SOP".into(),
            kind: SourceKind::LocalFolder,
            root: pack,
            owner: "fixtures".into(),
            classification: Classification::Internal,
            restricted_to_roles: Vec::new(),
            retention_days: None,
            enabled: true,
        })
        .unwrap();
    let synced = world.sync("pack-sop");
    assert_eq!(synced.outcome.documents_read, 1, "{synced:#?}");
    let found = call(
        &world.deps,
        "knowledge.hybrid_search",
        json!({"query": "minimum allowable thickness hydrocarbon vessel course with pitting recorded", "maxResults": 8}),
    )
    .await
    .unwrap();
    println!("--- P00 pack SOP (keyword-only) ---\n{found}");
    assert!(found.contains("3. Minimum allowable thickness"), "section 3 was not retrieved: {found}");
    assert!(found.contains("3.1 Carve-out"), "the 3.1 carve-out was not retrieved: {found}");
    assert!(found.contains("maintenance-sop-rev-d.md"), "{found}");
}

// ── Measurement against the labels ───────────────────────────────────────────

#[derive(Debug, serde::Deserialize)]
struct Labels {
    cases: Vec<Case>,
}

#[derive(Debug, serde::Deserialize)]
struct Case {
    id: String,
    kind: String,
    query: String,
    relevant: Vec<Relevant>,
}

#[derive(Debug, serde::Deserialize)]
struct Relevant {
    path: String,
    phrase: String,
}

/// Recall@k and MRR per method, over the labelled cases.
///
/// A hit counts when it is from the labelled document and contains the
/// labelled phrase — the exact passage, not merely the right file. The keyword
/// methods are measured on this machine for real; the hybrid figures are
/// through the fixture transport and are **not** a quality claim about any
/// model (see the module header).
#[tokio::test]
async fn labelled_retrieval_measurement() {
    let world = World::new(true).await;
    world.sync_all().await;
    let labels: Labels =
        serde_json::from_str(&std::fs::read_to_string(corpus_source().join("labels.json")).unwrap()).unwrap();
    const K: usize = 5;
    let session = admin();
    let lexical_only = RetrievalService::lexical_only(world.deps.index.clone(), "measurement: keyword half alone");

    let mut rows = Vec::new();
    let mut totals: std::collections::BTreeMap<&str, (f64, f64, usize, usize)> = Default::default();
    for case in &labels.cases {
        let literal = world.deps.index.search(&session, &case.query, K).unwrap();
        let request = crate::knowledge::hybrid::HybridRequest { query: case.query.clone(), limit: K, ..Default::default() };
        let keyword = lexical_only.search(&session, &request).await.unwrap().hits;
        let hybrid = world.retrieval.search(&session, &request).await.unwrap().hits;
        let as_passages = |hits: &[crate::knowledge::hybrid::EvidenceHit]| hits.iter().map(|h| h.passage.clone()).collect::<Vec<_>>();
        for (method, hits) in [
            ("keyword-all-terms (search_authorized)", literal.clone()),
            ("keyword-any-term (hybrid, no model)", as_passages(&keyword)),
            ("hybrid via fixture transport", as_passages(&hybrid)),
        ] {
            let matches = |hit: &crate::knowledge::SearchResult, relevant: &Relevant| {
                hit.document_name.ends_with(&relevant.path) && hit.text.contains(&relevant.phrase)
            };
            let (recall, reciprocal, correct_no_answer) = if case.relevant.is_empty() {
                (None, None, Some(hits.is_empty()))
            } else {
                let found = case.relevant.iter().filter(|r| hits.iter().take(K).any(|h| matches(h, r))).count();
                let first = hits.iter().position(|h| case.relevant.iter().any(|r| matches(h, r)));
                (
                    Some(found as f64 / case.relevant.len() as f64),
                    Some(first.map(|rank| 1.0 / (rank + 1) as f64).unwrap_or(0.0)),
                    None,
                )
            };
            let entry = totals.entry(method).or_default();
            if let (Some(recall), Some(reciprocal)) = (recall, reciprocal) {
                entry.0 += recall;
                entry.1 += reciprocal;
                entry.2 += 1;
            }
            if correct_no_answer == Some(true) {
                entry.3 += 1;
            }
            rows.push(json!({
                "case": case.id, "kind": case.kind, "method": method,
                "recallAtK": recall, "reciprocalRank": reciprocal, "noAnswerCorrect": correct_no_answer,
                "top": hits.iter().take(K).map(|h| format!("{} p{}", h.document_name, h.page)).collect::<Vec<_>>(),
            }));
        }
    }
    let answerable = labels.cases.iter().filter(|c| !c.relevant.is_empty()).count();
    let unanswerable = labels.cases.len() - answerable;
    let summary: Vec<Value> = totals
        .iter()
        .map(|(method, (recall, reciprocal, n, no_answer))| {
            json!({
                "method": method,
                "recallAtK": recall / *n as f64,
                "mrr": reciprocal / *n as f64,
                "answerableCases": n,
                "noAnswerCorrect": format!("{no_answer} of {unanswerable}"),
            })
        })
        .collect();
    let report = json!({
        "k": K,
        "cases": labels.cases.len(),
        "labels": "src-tauri/src/knowledge/testdata/retrieval/labels.json",
        "label": "keyword rows: deterministic test on this machine; hybrid row: deterministic transport (fixture concept lexicon, not a model)",
        "summary": summary,
        "rows": rows,
    });
    println!("--- labelled measurement ---\n{}", serde_json::to_string_pretty(&report).unwrap());
    if let Ok(out) = std::env::var("ARJUN_P07_MEASUREMENT") {
        std::fs::write(out, serde_json::to_string_pretty(&report).unwrap()).unwrap();
    }

    // What the measurement must show for the pipeline to be doing its job.
    let recall = |method: &str| {
        let (r, _, n, _) = totals[method];
        r / n as f64
    };
    assert!(recall("hybrid via fixture transport") >= recall("keyword-any-term (hybrid, no model)"));
    assert!(recall("keyword-any-term (hybrid, no model)") > recall("keyword-all-terms (search_authorized)"));
    assert_eq!(totals["hybrid via fixture transport"].3, unanswerable, "a no-answer case was answered");
    let _ = BTreeSet::<String>::new();
}
