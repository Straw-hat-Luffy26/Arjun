//! The Document & Vision Analyst end to end (plan P06).
//!
//! ## What is real here and what is not
//!
//! - **Real:** the documents (eight files drawn by PyMuPDF with their ink
//!   positions measured, `sidecars/document_sidecar/tests/p06_fixtures.py`);
//!   the page analyser (`render_pages.py` layout, crop and skew, run as a
//!   process); every tool call, through the runtime's own `authorize` and
//!   `execute`; `stream_ocr` over a real loopback socket; the region store, the
//!   OCR cache, the task event log and the memory graph; the production
//!   document-extractor worker dispatched by `agent.delegate`.
//! - **Not a model:** the OCR and vision *answers*. A fixture server on
//!   127.0.0.1 replies in llama-server's own streaming format with output in
//!   Unlimited-OCR's grounded-line shape, keyed by the SHA-256 of the image it
//!   was sent. So these tests measure routing, coordinate mapping, the loop /
//!   truncation / malformed checks, caching, coverage and publication — and
//!   say nothing about how well a real model reads. That is the target-machine
//!   gate (`tests/extraction_live.rs`, ignored here).
//!
//! Evidence label: deterministic test with deterministic transport.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use async_trait::async_trait;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

use super::*;
use crate::agent_runtime::cancellation::RunCancellations;
use crate::agent_runtime::documents::{NewExtraction, Sighting};
use crate::agent_runtime::events::plans::JobStatus;
use crate::ai_engine::ocr_profile::OcrDetent;
use crate::extraction::ocr::{HeldCard, OcrIdentity, OcrService, OcrSession};
use crate::extraction::regions::{BBox, EvidenceRegion, Method, RegionStatus};
use crate::extraction::service::ExtractionService;
use crate::extraction::vision::{ReadinessRecord, ReadyModel, VisionEndpoint, VisionService, VisionSession};
use crate::identity::{Role, Session, User};
use crate::knowledge::graph::runtime_memory::{ItemStatus, MemoryKind, MemoryScope};
use crate::knowledge::graph::runtime_store::MemoryGraph;
use crate::registry::{ModelManifest, ModelRegistry, ModelRole};
use crate::subagents::{load_profiles, SpecialistWorker, SubagentManager, WorkerServices};

const RUN: &str = "r";
const CONVERSATION: &str = "conv-analyst";

fn session() -> Session {
    Session::open(User::new("priya", "Priya Sharma", vec![Role::Employee, Role::Administrator]))
}

// -- The labelled documents ------------------------------------------------

struct Fixtures {
    dir: std::path::PathBuf,
    manifest: Value,
}

/// Drawn once per test binary, by the checked-in generator.
fn fixtures() -> &'static Fixtures {
    static FIXTURES: OnceLock<Fixtures> = OnceLock::new();
    FIXTURES.get_or_init(|| {
        let dir = tempfile::tempdir().expect("a fixture dir").keep();
        let script = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../sidecars/document_sidecar/tests/p06_fixtures.py");
        let output = std::process::Command::new(crate::deployment::program("python"))
            .arg(&script)
            .arg(&dir)
            .output()
            .expect("the fixture generator runs");
        assert!(
            output.status.success(),
            "the fixture generator failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let manifest: Value = serde_json::from_slice(&output.stdout).expect("a JSON manifest");
        Fixtures { dir, manifest }
    })
}

fn doc(name: &str) -> &'static Value {
    &fixtures().manifest["documents"][name]
}

/// The measured box of a labelled string.
fn truth(name: &str, text: &str) -> (u32, BBox) {
    let entry = doc(name)["truth"]
        .as_array()
        .unwrap()
        .iter()
        .find(|t| t["text"].as_str() == Some(text))
        .unwrap_or_else(|| panic!("{text} is not labelled in {name}"));
    let b: Vec<f64> = entry["bbox"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    (entry["page"].as_u64().unwrap() as u32, BBox::new(b[0], b[1], b[2], b[3]))
}

fn sha_of(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

// -- The fixture model server ------------------------------------------------

#[derive(Clone)]
struct Reply {
    text: String,
    /// `length` when the canned read ran to the decode cap.
    finish: &'static str,
}

#[derive(Default)]
struct Served {
    replies: Mutex<HashMap<String, Reply>>,
    requests: AtomicUsize,
    images: Mutex<Vec<String>>,
}

/// A loopback server speaking llama-server's chat-completions dialect.
struct FixtureServer {
    base_url: String,
    served: Arc<Served>,
}

impl FixtureServer {
    async fn start() -> Self {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.expect("a port");
        let base_url = format!("http://{}/v1", listener.local_addr().unwrap());
        let served = Arc::new(Served::default());
        let state = served.clone();
        tokio::spawn(async move {
            loop {
                let Ok((mut socket, _)) = listener.accept().await else { return };
                let state = state.clone();
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
                    let url = body
                        .pointer("/messages/0/content/0/image_url/url")
                        .and_then(Value::as_str)
                        .unwrap_or_default();
                    let encoded = url.split_once(",").map(|(_, b)| b).unwrap_or_default();
                    let bytes = base64::Engine::decode(&base64::engine::general_purpose::STANDARD, encoded)
                        .unwrap_or_default();
                    let image = sha_of(&bytes);
                    state.requests.fetch_add(1, Ordering::SeqCst);
                    state.images.lock().unwrap().push(image.clone());
                    let reply = state
                        .replies
                        .lock()
                        .unwrap()
                        .get(&image)
                        .cloned()
                        .unwrap_or(Reply { text: String::new(), finish: "stop" });
                    let streaming = body.get("stream").and_then(Value::as_bool).unwrap_or(false);
                    let response = if streaming {
                        let mut frames = String::new();
                        let chars: Vec<char> = reply.text.chars().collect();
                        for piece in chars.chunks(9) {
                            let delta: String = piece.iter().collect();
                            frames.push_str(&format!(
                                "data: {}\n\n",
                                json!({"choices":[{"delta":{"content": delta}}]})
                            ));
                        }
                        frames.push_str(&format!(
                            "data: {}\n\ndata: [DONE]\n\n",
                            json!({"choices":[{"delta":{},"finish_reason": reply.finish}]})
                        ));
                        format!(
                            "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\nconnection: close\r\n\r\n{frames}"
                        )
                    } else {
                        let payload = json!({"choices":[{"message":{"content": reply.text}}]}).to_string();
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
        Self { base_url, served }
    }

    fn reply(&self, image_sha256: &str, text: String, finish: &'static str) {
        self.served
            .replies
            .lock()
            .unwrap()
            .insert(image_sha256.to_string(), Reply { text, finish });
    }

    fn requests(&self) -> usize {
        self.served.requests.load(Ordering::SeqCst)
    }
}

/// OCR through the fixture server. Labelled as what it is.
struct FixtureOcr {
    base_url: String,
    /// Every detent a read was opened at, in order.
    opened: Arc<Mutex<Vec<OcrDetent>>>,
}

#[async_trait]
impl OcrService for FixtureOcr {
    fn identity(&self, detent: OcrDetent) -> Result<OcrIdentity, String> {
        Ok(OcrIdentity {
            model_id: "fixture-ocr".into(),
            weights_file: "none (deterministic transport)".into(),
            weights_sha256: None,
            weights_bytes: 0,
            projector_file: None,
            projector_bytes: None,
            detent: detent.label().into(),
            request_fingerprint: "0".repeat(64),
            prompt: "Free OCR.".into(),
        })
    }

    async fn open(&self, detent: OcrDetent, _: Option<&HeldCard>, _: Duration) -> Result<OcrSession, String> {
        self.opened.lock().unwrap().push(detent);
        Ok(OcrSession {
            base_url: self.base_url.clone(),
            served_model_id: "fixture-ocr".into(),
            residency: "fixture server; no card reserved".into(),
            _lease: None,
        })
    }

    fn transport(&self) -> &'static str {
        "deterministic fixture transport (canned Unlimited-OCR-format output; no model)"
    }
}

/// A vision service whose one model is "ready" by a fixture record.
struct FixtureVision {
    base_url: Option<String>,
}

#[async_trait]
impl VisionService for FixtureVision {
    fn ready_model(&self) -> Result<ReadyModel, String> {
        if self.base_url.is_none() {
            return Err("no model on this machine has passed an image probe".into());
        }
        Ok(ReadyModel {
            model_id: "fixture-vision".into(),
            record: ReadinessRecord {
                model_id: "fixture-vision".into(),
                weights_file: "none".into(),
                weights_bytes: 0,
                weights_sha256: None,
                projector_file: "fixture".into(),
                projector_bytes: 0,
                projector_sha256: "0".repeat(64),
                probe_token: "FIXTURE".into(),
                prompt: String::new(),
                answer_excerpt: String::new(),
                passed: true,
                reason: "deterministic fixture".into(),
                elapsed_ms: 0,
                at: "2026-09-24T00:00:00Z".into(),
                transport: "deterministic fixture transport".into(),
                actor: "test".into(),
            },
        })
    }

    async fn open(&self, _: &str, _: Duration) -> Result<VisionSession, String> {
        Ok(VisionSession {
            endpoint: VisionEndpoint {
                base_url: self.base_url.clone().expect("checked by ready_model"),
                served_model_id: "fixture-vision".into(),
            },
            residency: "fixture".into(),
            _lease: None,
        })
    }

    fn transport(&self) -> &'static str {
        "deterministic fixture transport (canned answer; no model)"
    }
}

// -- The world ---------------------------------------------------------------

struct World {
    deps: Arc<RuntimeDeps>,
    graph: Arc<MemoryGraph>,
    server: FixtureServer,
    dir: tempfile::TempDir,
    /// The detents OCR was opened at.
    opened: Arc<Mutex<Vec<OcrDetent>>>,
}

fn model_registry() -> ModelRegistry {
    let mut parent = crate::registry::tests::entry("model-parent", 9.0, vec![ModelRole::Reasoning]);
    parent.supports_structured_output = true;
    // An uncertified OCR-role model, as the shipped Unlimited-OCR rows are.
    let mut ocr = crate::registry::tests::entry("unlimited-ocr-q4-k-m", 1.0, vec![ModelRole::DocumentOcr]);
    ocr.supports_structured_output = true;
    ModelRegistry::from_manifest(
        ModelManifest { models: vec![parent, ocr] },
        std::path::PathBuf::from("models/registry.json"),
    )
    .expect("a well-formed manifest")
}

impl World {
    async fn new(vision: bool) -> Self {
        let server = FixtureServer::start().await;
        let (base, dir) = super::tests::deps_with(Arc::new(std::sync::RwLock::new(Some(session()))));
        let opened: Arc<Mutex<Vec<OcrDetent>>> = Arc::default();
        let extraction = Arc::new(ExtractionService::new(
            &dir.path().join("documents"),
            Arc::new(FixtureOcr { base_url: server.base_url.clone(), opened: opened.clone() }),
            Arc::new(FixtureVision { base_url: vision.then(|| server.base_url.clone()) }),
            crate::extraction::sidecar::Sidecar::resolve(),
        ));
        let events = base.events.clone();
        let graph = Arc::new(MemoryGraph::in_memory().expect("a graph").with_receipts(events.clone()));
        let registry = Arc::new(model_registry());
        let cancellations = Arc::new(RunCancellations::new());
        let services = Arc::new(WorkerServices {
            index: base.index.clone(),
            graph: Some(graph.clone()),
            events: events.clone(),
            scheduler: Arc::new(crate::subagents::ModelScheduler::new(
                registry.clone(),
                Arc::new(crate::serving::ModelServers::new()),
            )),
            session: base.session.clone(),
            child_loop: None,
            cancellations: cancellations.clone(),
            analyst: Some(crate::subagents::AnalystServices {
                extraction: extraction.clone(),
                documents: base.documents.clone(),
                conversations: base.run_to_conversation.clone(),
            }),
            retrieval: base.retrieval.clone(),
        });
        let profiles_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../agents");
        let profiles = load_profiles(&profiles_dir).profiles;
        let mut manager = SubagentManager::new(profiles.clone(), events.clone()).with_cancellations(cancellations);
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
        base.run_to_conversation.bind(RUN, CONVERSATION);
        let deps = Arc::new(RuntimeDeps {
            memory_graph: Some(graph.clone()),
            conversation_artifacts: base.conversation_artifacts.clone(),
            registry: Some(registry),
            index: base.index.clone(),
            session: base.session.clone(),
            workspaces: base.workspaces.clone(),
            approvals: base.approvals.clone(),
            calculations: base.calculations.clone(),
            passages: base.passages.clone(),
            produced: base.produced.clone(),
            calls: base.calls.clone(),
            plans: base.plans.clone(),
            events,
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
            extraction,
            retrieval: base.retrieval.clone(),
        });
        World { deps, graph, server, dir, opened }
    }

    /// Attaches a fixture the way `read_attachment` stores one: the bytes under
    /// their hash, and an extraction record sighted in this conversation.
    fn attach(&self, name: &str, owner: &str, conversation: &str) -> String {
        let file = doc(name)["file"].as_str().unwrap();
        let bytes = std::fs::read(fixtures().dir.join(file)).expect("the fixture exists");
        let sha = sha_of(&bytes);
        let extension = file.rsplit('.').next().unwrap();
        let stored = if extension == "pdf" { format!("source.{extension}") } else { format!("page-1.{extension}") };
        let folder = self.dir.path().join("documents").join("attachments").join(&sha);
        std::fs::create_dir_all(&folder).unwrap();
        std::fs::write(folder.join(stored), &bytes).unwrap();
        let pages = doc(name)["pages"].as_u64().unwrap() as u32;
        let mut page_text = std::collections::BTreeMap::new();
        if name == "typed" {
            page_text.insert(1, "INSPECTION REPORT\nTag: PT-2201\nDesign pressure: 10 bar".to_string());
            page_text.insert(2, "Material: SS316\nLine size: 4 in".to_string());
        }
        self.deps
            .documents
            .record(NewExtraction {
                sha256: sha.clone(),
                name: file.to_string(),
                kind: if extension == "pdf" { "pdf".into() } else { "image".into() },
                pages,
                truncated: false,
                page_text,
                sighting: Sighting {
                    owner_user_id: owner.to_string(),
                    conversation_id: conversation.to_string(),
                    message_id: "m-1".into(),
                    run_id: RUN.into(),
                    at: chrono::Utc::now().to_rfc3339(),
                    ocr_model_id: None,
                    ocr_detent: None,
                },
            })
            .expect("recorded");
        sha
    }

    fn regions(&self, sha: &str) -> Vec<EvidenceRegion> {
        self.deps.extraction.regions.load(sha).unwrap().regions.into_values().collect()
    }

    /// Renders the page the analyst will send to OCR, and programs the fixture
    /// server's answer for exactly those pixels.
    fn program(&self, sha: &str, page: u32, lines: &[(&str, Option<BBox>, &str)], finish: &'static str) {
        self.program_crop(sha, page, None, lines, finish);
    }

    /// As [`Self::program`], for a crop of the page rather than all of it. The
    /// reply's boxes are then relative to the crop, as a model's would be.
    fn program_crop(
        &self,
        sha: &str,
        page: u32,
        region: Option<BBox>,
        lines: &[(&str, Option<BBox>, &str)],
        finish: &'static str,
    ) {
        let doc = self
            .deps
            .extraction
            .authorise(&self.deps.documents, sha, "priya", CONVERSATION)
            .expect("authorised");
        let crop = self.deps.extraction.render_crop(&doc, page, region, None).expect("rendered");
        let mut text = String::new();
        for (label, bbox, body) in lines {
            match bbox {
                Some(b) => {
                    // The page box, as Unlimited-OCR reports it: 0..999 over the
                    // image it was shown.
                    let n = |v: f64, origin: f64, extent: u32| {
                        (((v - origin) * crop.pixels_per_unit) / extent as f64 * 999.0).round() as i32
                    };
                    text.push_str(&format!(
                        "{label} [{}, {}, {}, {}]{body}\n",
                        n(b.x0, crop.bbox.x0, crop.width),
                        n(b.y0, crop.bbox.y0, crop.height),
                        n(b.x1, crop.bbox.x0, crop.width),
                        n(b.y1, crop.bbox.y0, crop.height)
                    ));
                }
                None => text.push_str(&format!("{body}\n")),
            }
        }
        self.server.reply(&crop.image_sha256, text, finish);
    }
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
        .ok_or_else(|| format!("refused: {}", allow.get("reason").and_then(Value::as_str).unwrap_or("?")))?
        .to_string();
    let mut spent = request;
    spent["grant"] = json!(grant);
    let result = execute(spent, deps).await.map_err(|e| e.message)?;
    Ok(result["text"].as_str().unwrap_or_default().to_string())
}

/// The region whose text contains `needle`, read by `method`.
fn region_with<'a>(regions: &'a [EvidenceRegion], needle: &str, method: Method) -> &'a EvidenceRegion {
    regions
        .iter()
        .find(|r| r.method == method && r.text.contains(needle))
        .unwrap_or_else(|| panic!("no {method:?} region holds {needle:?}: {regions:#?}"))
}

fn close(a: &BBox, b: &BBox, tolerance: f64) -> bool {
    (a.x0 - b.x0).abs() <= tolerance
        && (a.y0 - b.y0).abs() <= tolerance
        && (a.x1 - b.x1).abs() <= tolerance
        && (a.y1 - b.y1).abs() <= tolerance
}

// -- Typed PDF -----------------------------------------------------------------

#[tokio::test]
async fn a_typed_pdf_is_read_from_its_text_layer_with_boxes_where_the_ink_is() {
    let world = World::new(false).await;
    let sha = world.attach("typed", "priya", CONVERSATION);

    let layout = call(&world.deps, "document.layout_map", json!({"documentSha256": sha, "fromPage": 1, "toPage": 2}))
        .await
        .expect("laid out");
    println!("--- typed.pdf: document.layout_map ---\n{layout}");
    assert!(layout.contains("pdf-points"), "{layout}");
    assert!(layout.contains("the text layer is adequate; no OCR is needed"), "{layout}");
    assert!(!layout.contains("NOT READ"), "a typed page was reported unread: {layout}");

    // The region the tag was read in covers the tag's measured ink.
    let regions = world.regions(&sha);
    let (page, ink) = truth("typed", "Tag: PT-2201");
    let tag = region_with(&regions, "Tag: PT-2201", Method::EmbeddedText);
    assert_eq!(tag.page, page);
    assert!(ink.covered_by(&tag.bbox) > 0.95, "{:?} does not cover {:?}", tag.bbox, ink);
    assert!(layout.contains(&tag.region_id), "the id is cited in the answer");

    // The table, cell by cell, each with the box of its own cell.
    let tables = call(&world.deps, "document.extract_tables", json!({"documentSha256": sha, "fromPage": 1}))
        .await
        .expect("tables");
    assert!(tables.contains("3 x 3 cells, embedded table"), "{tables}");
    assert!(tables.contains("r1: A | 9.4 | ok"), "{tables}");
    let table = regions.iter().find(|r| r.method == Method::EmbeddedTable).expect("a table region");
    let cell = table.cells.iter().find(|c| c.text == "9.4").expect("the 9.4 cell");
    let (_, ink) = truth("typed", "9.4");
    assert!(ink.covered_by(cell.bbox.as_ref().expect("a cell box")) > 0.95);

    // Fields, found in the text layer; OCR never asked.
    let findings = call(
        &world.deps,
        "media.extract_findings",
        json!({"documentSha256": sha, "fromPage": 1, "toPage": 2, "fields": ["tag", "design pressure", "material", "flow rate"]}),
    )
    .await
    .expect("findings");
    assert!(findings.contains("Field \"tag\": FOUND"), "{findings}");
    assert!(findings.contains("\"PT-2201\""), "{findings}");
    assert!(findings.contains("\"10 bar\""), "{findings}");
    assert!(findings.contains("\"SS316\""), "{findings}");
    assert!(findings.contains("embedded text layer"), "{findings}");
    assert!(findings.contains("Field \"flow rate\": NOT FOUND on the pages that were read"), "{findings}");
    assert!(!findings.contains("confidence:"), "no confidence number is invented");

    let refused = call(&world.deps, "document.ocr_regions", json!({"documentSha256": sha, "pages": [1]}))
        .await
        .expect("answered");
    assert!(refused.contains("carries its own text layer"), "{refused}");
    assert_eq!(world.server.requests(), 0, "a typed page was sent to OCR");

    // `document.read_pages` now says how each page was read.
    let pages = call(&world.deps, "document.read_pages", json!({"documentSha256": sha, "fromPage": 1, "toPage": 2}))
        .await
        .expect("read");
    assert!(pages.contains("page 1: the file's own text layer"), "{pages}");
    assert!(pages.contains("Coverage: 2 of 2 page(s)"), "{pages}");
}

// -- Scan -------------------------------------------------------------------

#[tokio::test]
async fn a_scan_is_read_by_local_ocr_with_boxes_mapped_back_onto_the_page_and_cached() {
    let world = World::new(false).await;
    let sha = world.attach("scan", "priya", CONVERSATION);
    let (_, title) = truth("scan", "INSPECTION REPORT");
    let (_, tag) = truth("scan", "Tag: PT-3301");
    let (_, pressure) = truth("scan", "Design pressure: 16 bar");
    let (_, material) = truth("scan", "Material: CS A106");
    world.program(
        &sha,
        1,
        &[
            ("title", Some(title), "INSPECTION REPORT"),
            ("text", Some(tag), "Tag: PT-3301"),
            ("text", Some(pressure), "Design pressure: 16 bar"),
        ],
        "stop",
    );
    world.program(&sha, 2, &[("text", Some(material), "Material: CS A106")], "stop");

    let layout = call(&world.deps, "document.layout_map", json!({"documentSha256": sha, "fromPage": 1, "toPage": 2}))
        .await
        .expect("laid out");
    assert!(layout.contains("a scan or picture: read it with document.ocr_regions"), "{layout}");
    assert!(layout.contains("NOT READ"), "{layout}");

    let findings = call(
        &world.deps,
        "media.extract_findings",
        json!({"documentSha256": sha, "fromPage": 1, "toPage": 2, "fields": ["tag", "design pressure", "material"]}),
    )
    .await
    .expect("findings");
    println!("--- scan.pdf: media.extract_findings ---\n{findings}");
    assert!(findings.contains("\"PT-3301\""), "{findings}");
    assert!(findings.contains("\"16 bar\""), "{findings}");
    assert!(findings.contains("\"CS A106\""), "{findings}");
    assert!(findings.contains("local OCR transcription"), "{findings}");
    assert!(findings.contains("Scanned page(s) 1, 2 were read by local OCR"), "{findings}");
    // The layout's picture placeholders were read by OCR, so they are not
    // listed as regions nobody could read.
    assert!(!findings.contains("could not be quoted"), "{findings}");
    assert_eq!(world.server.requests(), 2, "one request per scanned page");

    // Every mapped box lands on the ink it names, within a point.
    let regions = world.regions(&sha);
    for (needle, ink) in [("Tag: PT-3301", tag), ("Design pressure: 16 bar", pressure), ("Material: CS A106", material)] {
        let region = region_with(&regions, needle, Method::Ocr);
        assert!(close(&region.bbox, &ink, 1.0), "{needle}: {:?} vs {:?}", region.bbox, ink);
        assert_eq!(region.status, RegionStatus::Read);
        assert!(region.cache_key.is_some() && region.image_sha256.is_some() && region.crop_id.is_some());
    }

    // Again: the regions are in the store, so no page is sent twice.
    let again = call(&world.deps, "document.ocr_regions", json!({"documentSha256": sha, "pages": [1, 2]}))
        .await
        .expect("read again");
    assert!(again.contains("from the cache (same bytes, crop, model and settings)"), "{again}");
    assert_eq!(world.server.requests(), 2, "a cached page went back to the model");
    assert!(again.contains("deterministic fixture transport"), "the transport is named: {again}");
}

#[tokio::test]
async fn a_region_crop_is_read_and_its_boxes_land_on_the_page_not_on_the_crop() {
    let world = World::new(false).await;
    let sha = world.attach("scan", "priya", CONVERSATION);
    let (_, tag) = truth("scan", "Tag: PT-3301");
    // A crop well away from the page origin, so a box left in crop space would
    // be off by the crop's offset.
    let region = BBox::new(60.0, 100.0, 320.0, 150.0);
    world.program_crop(&sha, 1, Some(region), &[("text", Some(tag), "Tag: PT-3301")], "stop");
    let read = call(
        &world.deps,
        "document.ocr_regions",
        json!({"documentSha256": sha, "page": 1, "regions": ["60,100,320,150"]}),
    )
    .await
    .expect("read");
    assert!(read.contains("Page 1 crop cr-"), "{read}");
    let regions = world.regions(&sha);
    let found = region_with(&regions, "PT-3301", Method::Ocr);
    assert!(close(&found.bbox, &tag, 1.0), "{:?} is not where the ink is ({:?})", found.bbox, tag);
    let crop = world.deps.extraction.regions.load(&sha).unwrap().crops[found.crop_id.as_ref().unwrap()].clone();
    assert!(close(&crop.bbox, &region, 0.01), "the crop is not the region asked for: {:?}", crop.bbox);
    assert!(crop.bbox.x0 > 0.0 && crop.bbox.y0 > 0.0);
}

// -- Skewed page ---------------------------------------------------------------

#[tokio::test]
async fn a_skewed_page_is_measured_and_its_boxes_say_so() {
    let world = World::new(false).await;
    let sha = world.attach("skewed", "priya", CONVERSATION);
    // Where the rotated line actually lies is not an axis-aligned box; the
    // reply gives the box a model would, around the line.
    world.program(&sha, 1, &[("text", Some(BBox::new(70.0, 110.0, 160.0, 135.0)), "Tag: PT-4401")], "stop");
    let read = call(&world.deps, "document.ocr_regions", json!({"documentSha256": sha, "pages": [1]}))
        .await
        .expect("read");
    let expected = doc("skewed")["skewDegrees"].as_f64().unwrap();
    let measured = world.deps.extraction.regions.load(&sha).unwrap().pages[&1].skew_degrees.expect("measured");
    assert!((measured - expected).abs() <= 1.0, "measured {measured}, drawn {expected}");
    assert!(read.contains("is skewed by about"), "{read}");
    let regions = world.regions(&sha);
    let region = region_with(&regions, "PT-4401", Method::Ocr);
    assert!(region.notes.iter().any(|n| n.contains("skewed")), "{:?}", region.notes);
}

// -- Handwriting -------------------------------------------------------------

#[tokio::test]
async fn handwriting_is_located_in_image_pixels_and_a_scribble_is_unreadable_not_guessed() {
    let world = World::new(false).await;
    let sha = world.attach("handwriting", "priya", CONVERSATION);
    let (_, checked) = truth("handwriting", "Checked by R. Menon");
    let (_, date) = truth("handwriting", "Date 12/09");
    let scribble = {
        let b: Vec<f64> = doc("handwriting")["truth"][2]["bbox"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
        BBox::new(b[0], b[1], b[2], b[3])
    };
    world.program(
        &sha,
        1,
        &[
            ("text", Some(checked), "Checked by R. Menon"),
            ("text", Some(date), "Date 12/09"),
            ("text", Some(scribble), ""),
        ],
        "stop",
    );
    let findings = call(
        &world.deps,
        "media.extract_findings",
        json!({"documentSha256": sha, "fromPage": 1, "fields": ["checked by", "date"]}),
    )
    .await
    .expect("findings");
    assert!(findings.contains("\"R. Menon\""), "{findings}");
    assert!(findings.contains("\"12/09\""), "{findings}");
    assert!(findings.contains("image-pixels"), "{findings}");
    assert!(findings.contains("Regions that could not be quoted"), "{findings}");
    assert!(findings.contains("never supply a likely value"), "{findings}");

    let regions = world.regions(&sha);
    let unreadable = regions
        .iter()
        .find(|r| r.method == Method::Ocr && r.status == RegionStatus::Unreadable)
        .expect("the scribble is an unreadable region");
    assert!(close(&unreadable.bbox, &scribble, 2.0), "{:?} vs {:?}", unreadable.bbox, scribble);
    assert!(unreadable.text.is_empty());
    let name = region_with(&regions, "Menon", Method::Ocr);
    assert!(close(&name.bbox, &checked, 2.0), "{:?} vs {:?}", name.bbox, checked);
}

// -- Long repeated output ------------------------------------------------------

#[tokio::test]
async fn a_long_repeated_output_is_cut_and_the_page_is_reported_incomplete() {
    let world = World::new(false).await;
    let sha = world.attach("long", "priya", CONVERSATION);
    let (_, title) = truth("long", "SITE SURVEY");
    let (_, tag) = truth("long", "Tag: PT-5501");
    let mut lines: Vec<(&str, Option<BBox>, &str)> =
        vec![("title", Some(title), "SITE SURVEY"), ("text", Some(tag), "Tag: PT-5501")];
    // The observed failure: short caption lines cycling a few values, their
    // boxes creeping down the page, to the decode cap.
    for i in 0..200 {
        let y = 200.0 + (i % 40) as f64;
        lines.push(("image_caption", Some(BBox::new(300.0, y, 360.0, y + 10.0)), ["OVO", "AUDIO", "AVO"][i % 3]));
    }
    world.program(&sha, 1, &lines, "length");

    let read = call(&world.deps, "document.ocr_regions", json!({"documentSha256": sha, "pages": [1]}))
        .await
        .expect("read");
    assert!(read.contains("INCOMPLETE"), "{read}");
    assert!(read.contains("repetition"), "{read}");
    let regions = world.regions(&sha);
    assert_eq!(region_with(&regions, "PT-5501", Method::Ocr).status, RegionStatus::Read);
    assert!(regions.iter().any(|r| r.status == RegionStatus::Looped), "{regions:#?}");
    assert!(regions.len() < 20, "the loop's regions were kept: {}", regions.len());

    let findings = call(&world.deps, "media.extract_findings", json!({"documentSha256": sha, "fromPage": 1, "fields": ["tag"]}))
        .await
        .expect("findings");
    assert!(findings.contains("\"PT-5501\""), "{findings}");
    assert!(findings.contains("OCR looped and was cut"), "{findings}");
}

// -- Table ---------------------------------------------------------------------

#[tokio::test]
async fn an_ocr_table_is_returned_cell_by_cell_and_two_readings_are_both_reported() {
    let world = World::new(false).await;
    let sha = world.attach("table", "priya", CONVERSATION);
    let (_, table) = truth("table", "table");
    world.program(
        &sha,
        1,
        &[(
            "table",
            Some(table),
            "<table><tr><td>Phase</td><td>Reading</td><td>Limit</td></tr><tr><td>R</td><td>412 MOhm</td><td>100 MOhm</td></tr><tr><td>Y</td><td>388 MOhm</td><td>100 MOhm</td></tr></table>",
        )],
        "stop",
    );
    let before = call(&world.deps, "document.extract_tables", json!({"documentSha256": sha, "fromPage": 1}))
        .await
        .expect("answered");
    assert!(before.contains("have not been OCR-read"), "an unread scan is not a page with no tables: {before}");

    call(&world.deps, "document.ocr_regions", json!({"documentSha256": sha, "pages": [1]})).await.expect("read");
    let tables = call(&world.deps, "document.extract_tables", json!({"documentSha256": sha, "fromPage": 1}))
        .await
        .expect("tables");
    assert!(tables.contains("3 x 3 cells, local OCR transcription"), "{tables}");
    assert!(tables.contains("r2: Y | 388 MOhm | 100 MOhm"), "{tables}");
    assert!(tables.contains("OCR locates the table, not its cells"), "{tables}");

    let findings = call(&world.deps, "media.extract_findings", json!({"documentSha256": sha, "fromPage": 1, "fields": ["reading"]}))
        .await
        .expect("findings");
    assert!(findings.contains("CONFLICTING VALUES"), "{findings}");
    assert!(findings.contains("\"412 MOhm\"") && findings.contains("\"388 MOhm\""), "{findings}");
    let region = world.regions(&sha).into_iter().find(|r| r.label == "table").unwrap();
    assert!(close(&region.bbox, &table, 1.0), "{:?} vs {:?}", region.bbox, table);
}

// -- Photo -----------------------------------------------------------------------

#[tokio::test]
async fn a_photo_is_transcribed_and_its_interpretation_is_a_labelled_proposal() {
    let world = World::new(true).await;
    let sha = world.attach("photo", "priya", CONVERSATION);
    let (_, name) = truth("photo", "PUMP P-101");
    // A line before any region header has no box; after one, it would be read
    // as that region's continuation.
    world.program(&sha, 1, &[("text", None, "7.5 kW 2900 rpm"), ("text", Some(name), "PUMP P-101")], "stop");
    // The same pixels answer the interpretation question.
    let findings = call(
        &world.deps,
        "media.extract_findings",
        json!({"documentSha256": sha, "fromPage": 1, "fields": ["pump"], "question": "What equipment is this nameplate on?"}),
    )
    .await
    .expect("findings");
    assert!(findings.contains("\"P-101\""), "{findings}");
    assert!(findings.contains("PROPOSAL — vision-model inference by fixture-vision"), "{findings}");
    let regions = world.regions(&sha);
    let proposal = regions.iter().find(|r| r.method == Method::VisionInference).expect("a proposal region");
    assert!(proposal.notes[0].contains("not a transcription"));
    // The line with no box is kept, and located only to the crop.
    let loose = region_with(&regions, "7.5 kW", Method::Ocr);
    assert_eq!(loose.status, RegionStatus::Malformed);
    // A proposal is never a source of a field value.
    assert!(!findings.contains("vision-model inference (proposal, not transcription); read"), "{findings}");
}

#[tokio::test]
async fn with_no_vision_ready_model_interpretation_is_refused_by_name_and_nothing_is_guessed() {
    let world = World::new(false).await;
    let sha = world.attach("photo", "priya", CONVERSATION);
    let (_, name) = truth("photo", "PUMP P-101");
    world.program(&sha, 1, &[("text", Some(name), "PUMP P-101")], "stop");
    let findings = call(
        &world.deps,
        "media.extract_findings",
        json!({"documentSha256": sha, "fromPage": 1, "question": "What is it?"}),
    )
    .await
    .expect("findings");
    assert!(findings.contains("Interpretation was not attempted: no model on this machine has passed an image probe"), "{findings}");
    assert!(!findings.contains("PROPOSAL"), "{findings}");
    assert!(world.regions(&sha).iter().all(|r| r.method != Method::VisionInference));
}

// -- P&ID with an unreadable label ----------------------------------------------

#[tokio::test]
async fn a_pid_with_a_smudged_label_reports_it_unreadable_at_its_box() {
    let world = World::new(true).await;
    let sha = world.attach("pid", "priya", CONVERSATION);
    let (_, pt) = truth("pid", "PT-101");
    let (_, fv) = truth("pid", "FV-102");
    let (_, tt) = truth("pid", "TT-103");
    world.program(&sha, 1, &[("text", Some(pt), "PT-101"), ("text", Some(fv), "FV-102"), ("text", Some(tt), "")], "stop");
    let findings = call(
        &world.deps,
        "media.extract_findings",
        json!({"documentSha256": sha, "fromPage": 1, "question": "Which instruments are on the line?"}),
    )
    .await
    .expect("findings");
    println!("--- pid.png: media.extract_findings with a question ---\n{findings}");
    assert!(findings.contains("could not be quoted"), "{findings}");
    let regions = world.regions(&sha);
    let smudged = regions
        .iter()
        .find(|r| r.method == Method::Ocr && r.status == RegionStatus::Unreadable)
        .expect("the smudged tag is unreadable");
    assert!(close(&smudged.bbox, &tt, 2.0), "{:?} vs {:?}", smudged.bbox, tt);
    assert!(!regions.iter().any(|r| r.method.is_transcription() && r.text.contains("TT-103")), "a label nobody could read was supplied");
    assert!(close(&region_with(&regions, "PT-101", Method::Ocr).bbox, &pt, 2.0));
}

// -- Authorisation ---------------------------------------------------------------

#[tokio::test]
async fn a_document_from_another_owner_or_thread_is_not_there_and_changed_bytes_are_refused() {
    let world = World::new(false).await;
    let theirs = world.attach("typed", "someone-else", CONVERSATION);
    let other_thread = world.attach("scan", "priya", "another-thread");
    for sha in [&theirs, &other_thread] {
        let refused = call(&world.deps, "document.layout_map", json!({"documentSha256": sha, "fromPage": 1}))
            .await
            .expect_err("not attached here");
        assert!(refused.contains("No document with that id has been attached to this conversation."), "{refused}");
    }
    let mine = world.attach("pid", "priya", CONVERSATION);
    let stored = world.dir.path().join("documents/attachments").join(&mine).join("page-1.png");
    std::fs::write(&stored, b"not the attached bytes").unwrap();
    let refused = call(&world.deps, "document.layout_map", json!({"documentSha256": mine, "fromPage": 1}))
        .await
        .expect_err("tampered");
    assert!(refused.contains("no longer hash to its id"), "{refused}");
    let wide = call(&world.deps, "document.layout_map", json!({"documentSha256": world.attach("typed", "priya", CONVERSATION), "fromPage": 1, "toPage": 11}))
        .await
        .expect_err("too wide");
    assert!(wide.contains("at most 10"), "{wide}");
}

// -- A real parent task --------------------------------------------------------

#[tokio::test]
async fn extraction_from_a_parent_task_commits_observations_and_proposals_to_the_graph() {
    let world = World::new(true).await;
    let deps = &world.deps;
    let scan = world.attach("scan", "priya", CONVERSATION);
    let (_, tag) = truth("scan", "Tag: PT-3301");
    let (_, pressure) = truth("scan", "Design pressure: 16 bar");
    world.program(&scan, 1, &[("text", Some(tag), "Tag: PT-3301"), ("text", Some(pressure), "Design pressure: 16 bar")], "stop");
    world.program(&scan, 2, &[("text", None, "")], "stop");
    let pid = world.attach("pid", "priya", CONVERSATION);
    let (_, pt) = truth("pid", "PT-101");
    let (_, tt) = truth("pid", "TT-103");
    world.program(&pid, 1, &[("text", Some(pt), "PT-101"), ("text", Some(tt), "")], "stop");

    call(
        deps,
        "task.plan_update",
        json!({"base_version": 0, "operations": [
            {"op": "set_goal", "text": "Record the inspection data from the attached scan and drawing"},
            {"op": "add_step", "id": "scan", "title": "Extract the scan's fields", "kind": "delegate",
             "role": "document-extractor", "acceptance": ["childCompleted", "memoryPublished"]},
            {"op": "add_step", "id": "pid", "title": "Read the P&ID", "kind": "delegate",
             "role": "document-extractor", "acceptance": ["childCompleted", "memoryPublished"]},
        ]}),
    )
    .await
    .expect("the plan");
    let revision_before = world.graph.graph_revision().unwrap();

    let first = call(
        deps,
        "agent.delegate",
        json!({"step": "scan", "role": "document-extractor",
               "objective": "Extract fields: tag, design pressure.", "documents": [scan], "wait_seconds": 60}),
    )
    .await
    .expect("dispatched");
    let job = first.split_whitespace().find(|w| w.starts_with("job-")).unwrap().trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '-').to_string();
    let settled = loop {
        let record = deps.events.job(&job).unwrap().expect("the job");
        if !record.status.is_live() {
            break record;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    };
    assert_eq!(settled.status, JobStatus::Completed, "{:?}", settled.result);
    // Routing, as this build does it: an OCR model with no certification pack
    // is not a cheaper child model, so the extractor child runs on the
    // coordinator's model and OCR is a service call that reserves its own card
    // (plan §5.2: OCR is a service within this agent's workflow).
    assert_eq!(settled.model_id.as_deref(), Some("model-parent"), "{}", settled.lease);

    let second = call(
        deps,
        "agent.delegate",
        json!({"step": "pid", "role": "document-extractor",
               "objective": "Read the drawing's tags.\nquestion: Which instruments are drawn on the line?",
               "documents": [pid], "wait_seconds": 60}),
    )
    .await
    .expect("dispatched");
    let job = second.split_whitespace().find(|w| w.starts_with("job-")).unwrap().trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '-').to_string();
    loop {
        let record = deps.events.job(&job).unwrap().expect("the job");
        if !record.status.is_live() {
            assert_eq!(record.status, JobStatus::Completed, "{:?}", record.result);
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }

    // So the pages were read at the analyst's own detent, through the service.
    let opened = world.opened.lock().unwrap().clone();
    assert!(!opened.is_empty() && opened.iter().all(|d| *d == OcrDetent::Detailed), "{opened:?}");

    // Committed graph changes, read back from the graph itself.
    assert!(world.graph.graph_revision().unwrap() > revision_before, "nothing was committed");
    let items = world.graph.snapshot(&session(), &MemoryScope::Task { task_id: RUN.into() }, None).expect("readable");
    println!(
        "--- the task's shared memory after two extraction jobs (graph revision {} -> {}) ---",
        revision_before,
        world.graph.graph_revision().unwrap()
    );
    for item in &items {
        println!(
            "{:?} {:?} {} | {}",
            item.kind,
            item.status,
            item.sources.first().map(|s| s.locator.as_str()).unwrap_or("-"),
            item.content.chars().take(200).collect::<String>()
        );
    }
    let tag_item = items
        .iter()
        .find(|i| i.content.contains("tag = \"PT-3301\""))
        .unwrap_or_else(|| panic!("the tag was not published: {:#?}", items.iter().map(|i| &i.content).collect::<Vec<_>>()));
    assert_eq!(tag_item.kind, MemoryKind::ToolObservation);
    assert_eq!(tag_item.status, ItemStatus::Admitted, "a checked value is admitted on its receipt");
    assert!(tag_item.confidence.is_none(), "no confidence number is invented");
    let source = &tag_item.sources[0];
    assert_eq!(source.sha256, scan);
    assert!(source.locator.starts_with("page 1 region rg-") && source.locator.contains("pdf-points"), "{}", source.locator);
    let region_id = source.locator.split_whitespace().nth(3).unwrap();
    let region = world.deps.extraction.regions.load(&scan).unwrap().regions[region_id].clone();
    assert!(close(&region.bbox, &tag, 1.0), "the cited region is where the tag is");

    // Page 2 of the scan read as nothing: named, not assumed blank.
    assert!(
        items.iter().any(|i| i.content.starts_with("coverage —") && i.content.contains("p.2 local OCR (1 unreadable region(s))")),
        "page 2 read as nothing was not said to be unreadable"
    );

    // The P&ID's smudged tag is an admitted observation that it is unreadable.
    let unreadable = items
        .iter()
        .find(|i| i.content.contains("unreadable") && i.sources.iter().any(|s| s.sha256 == pid))
        .expect("the unreadable label is published");
    assert_eq!(unreadable.status, ItemStatus::Admitted);
    // The interpretation is a proposal, labelled, never admitted.
    let proposal = items.iter().find(|i| i.content.starts_with("PROPOSAL")).expect("the interpretation is published");
    assert_ne!(proposal.status, ItemStatus::Admitted, "a model's description was admitted as fact");
    assert!(proposal.content.contains("not a transcription"));
}
