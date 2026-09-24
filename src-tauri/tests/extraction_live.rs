//! The Document & Vision Analyst against the real local models (plan P06).
//!
//! **Target-machine gate. Ignored by default**, because it needs this machine's
//! installed Unlimited-OCR weights, their projector and a `llama-server`, and
//! the coding computer has none of them. Run it on the target:
//!
//! ```text
//! set ARJUN_APP_DATA=C:\Users\<you>\AppData\Local\com.arjun.workbench
//! cargo test --manifest-path src-tauri/Cargo.toml --test extraction_live -- --ignored --nocapture --test-threads=1
//! ```
//!
//! ## What it establishes that the deterministic tests cannot
//!
//! `agent_runtime::extraction_tests` drives the whole analyst path with canned
//! OCR answers, so it proves routing, coordinate mapping, the loop / truncation
//! / malformed checks, caching and publication — and nothing about how well the
//! installed model reads. The Unlimited-OCR GGUF is a third-party conversion
//! whose architecture string was rewritten to reach llama.cpp's DeepSeek-OCR
//! path (`config/ocr-model-registry.json`); a server that *loads* it has proved
//! only that it loads. This reads the eight labelled documents
//! (`sidecars/document_sidecar/tests/p06_fixtures.py`) through the production
//! service and scores, separately:
//!
//! - **field extraction** — whether each labelled string was transcribed;
//! - **citation location** — whether the region it was found in overlaps where
//!   the ink is (intersection over union against the measured box);
//! - **checks** — loops, decode caps, malformed spans, per page;
//!
//! and writes everything, with the exact weights, projector and server build,
//! to `evidence/agent-system/P06/target-ocr-<timestamp>.json`. Nothing here is
//! a canned number: an unread label scores zero and says so.
//!
//! The vision half probes every model with an image projector and records the
//! result. A text-only model (Gemma 4 E4B and Qwen3.5 9B as inventoried at P00)
//! is recorded as not probed, with why; binding a projector and probing again
//! is the operator's step, not this test's.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use sarathi_lib::agent_runtime::cancellation::CancelToken;
use sarathi_lib::agent_runtime::documents::{DocumentStore, NewExtraction, Sighting};
use sarathi_lib::ai_engine::ocr_profile::OcrDetent;
use sarathi_lib::extraction::ocr::{OcrService, ServedOcr};
use sarathi_lib::extraction::regions::{BBox, Method, RegionStatus};
use sarathi_lib::extraction::service::ExtractionService;
use sarathi_lib::extraction::sidecar::Sidecar;
use sarathi_lib::extraction::vision::NoVision;
use sarathi_lib::registry::{ModelRegistry, ModelRole};
use sarathi_lib::serving::ModelServers;
use sarathi_lib::subagents::scheduling::ModelScheduler;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

const OWNER: &str = "target-gate";
const CONVERSATION: &str = "p06-target-gate";

fn app_data() -> PathBuf {
    PathBuf::from(std::env::var("ARJUN_APP_DATA").unwrap_or_else(|_| {
        panic!(
            "ARJUN_APP_DATA is not set. Point it at the application data directory that holds \
             models/registry.json on the target machine; this gate does not guess a path."
        )
    }))
}

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..")
}

fn fixtures(dir: &Path) -> Value {
    let output = std::process::Command::new(sarathi_lib::deployment::program("python"))
        .arg(repo().join("sidecars/document_sidecar/tests/p06_fixtures.py"))
        .arg(dir)
        .output()
        .expect("the fixture generator runs");
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    serde_json::from_slice(&output.stdout).expect("a manifest")
}

fn iou(a: &BBox, b: &BBox) -> f64 {
    let x0 = a.x0.max(b.x0);
    let y0 = a.y0.max(b.y0);
    let x1 = a.x1.min(b.x1);
    let y1 = a.y1.min(b.y1);
    if x1 <= x0 || y1 <= y0 {
        return 0.0;
    }
    let inter = (x1 - x0) * (y1 - y0);
    inter / (a.area() + b.area() - inter)
}

fn squeeze(text: &str) -> String {
    text.chars().filter(|c| c.is_alphanumeric()).collect::<String>().to_lowercase()
}

fn sha_file(path: &Path) -> Option<String> {
    let bytes = std::fs::read(path).ok()?;
    Some(hex::encode(Sha256::digest(bytes)))
}

fn llama_server_version() -> String {
    let program = sarathi_lib::deployment::program("llama-server");
    std::process::Command::new(&program)
        .arg("--version")
        .output()
        .map(|o| {
            let text = format!("{}{}", String::from_utf8_lossy(&o.stdout), String::from_utf8_lossy(&o.stderr));
            text.lines().filter(|l| !l.trim().is_empty()).collect::<Vec<_>>().join(" | ")
        })
        .unwrap_or_else(|error| format!("{program} --version could not run: {error}"))
}

fn write_evidence(name: &str, value: &Value) -> PathBuf {
    let dir = repo().join("evidence/agent-system/P06");
    std::fs::create_dir_all(&dir).expect("the evidence directory");
    let path = dir.join(format!("{name}-{}.json", chrono::Utc::now().format("%Y%m%dT%H%M%SZ")));
    std::fs::write(&path, serde_json::to_vec_pretty(value).unwrap()).expect("evidence written");
    path
}

#[tokio::test]
#[ignore = "target-machine gate: needs the installed Unlimited-OCR weights, projector and llama-server"]
async fn local_unlimited_ocr_reads_the_labelled_pages_on_this_machine() {
    let app_data = app_data();
    let registry = Arc::new(ModelRegistry::load_with_discovery(&app_data).expect("the registry loads"));
    let servers = Arc::new(ModelServers::new());
    let scheduler = Arc::new(ModelScheduler::new(registry.clone(), servers.clone()));
    let work = tempfile::tempdir().expect("a work dir");
    let documents_root = work.path().join("documents");
    let store = DocumentStore::open(work.path()).expect("a store");
    let ocr = ServedOcr { registry: registry.clone(), servers: servers.clone(), scheduler };
    let detent = OcrDetent::Detailed;
    let identity = ocr.identity(detent).expect("the OCR model is registered");

    // What the installed files actually are, measured here.
    let models_dir = registry.models_dir().to_path_buf();
    let entry = registry.find(&identity.model_id).expect("registered").clone();
    let weights = models_dir.join(&entry.path);
    let projector = entry.projector.as_ref().map(|p| models_dir.join(p));
    let projector_header = projector
        .as_deref()
        .map(|p| match sarathi_lib::extraction::projector::inspect_projector(p) {
            Ok(header) => serde_json::to_value(header).unwrap(),
            Err(error) => json!({"unreadable": error}),
        });
    let model_header = sarathi_lib::ai_engine::gguf_meta::read_gguf_scalars(&weights)
        .map(|kv| {
            json!({
                "general.architecture": kv.get("general.architecture").and_then(|v| v.as_str()),
                "general.name": kv.get("general.name").and_then(|v| v.as_str()),
            })
        })
        .unwrap_or_else(|error| json!({"unreadable": format!("{error:#}")}));

    let service = Arc::new(ExtractionService::new(
        &documents_root,
        Arc::new(ocr),
        Arc::new(NoVision("not exercised by the OCR gate".into())),
        Sidecar::resolve(),
    ));
    let manifest = fixtures(&work.path().join("fixtures"));
    let mut documents = Vec::new();
    let (mut labels, mut transcribed, mut located) = (0u32, 0u32, 0u32);

    for (key, fixture) in manifest["documents"].as_object().unwrap() {
        if key == "typed" {
            continue; // Read from its text layer; OCR is not asked.
        }
        let file = fixture["file"].as_str().unwrap();
        let bytes = std::fs::read(work.path().join("fixtures").join(file)).unwrap();
        let sha = hex::encode(Sha256::digest(&bytes));
        let extension = file.rsplit('.').next().unwrap();
        let folder = documents_root.join("attachments").join(&sha);
        std::fs::create_dir_all(&folder).unwrap();
        let stored = if extension == "pdf" { format!("source.{extension}") } else { format!("page-1.{extension}") };
        std::fs::write(folder.join(stored), &bytes).unwrap();
        let pages = fixture["pages"].as_u64().unwrap() as u32;
        store
            .record(NewExtraction {
                sha256: sha.clone(),
                name: file.into(),
                kind: "fixture".into(),
                pages,
                truncated: false,
                page_text: Default::default(),
                sighting: Sighting {
                    owner_user_id: OWNER.into(),
                    conversation_id: CONVERSATION.into(),
                    message_id: "gate".into(),
                    run_id: "gate".into(),
                    at: chrono::Utc::now().to_rfc3339(),
                    ocr_model_id: None,
                    ocr_detent: None,
                },
            })
            .unwrap();
        let doc = service.authorise(&store, &sha, OWNER, CONVERSATION).expect("authorised");
        let started = Instant::now();
        let outcome = service
            .ocr(
                &doc,
                (1..=pages).map(|p| (p, None)).collect(),
                None,
                detent,
                None,
                &CancelToken::never(),
                Instant::now() + Duration::from_secs(900),
            )
            .await
            .expect("the OCR pass ran");
        let regions: Vec<_> = service.regions.load(&sha).unwrap().regions.into_values().filter(|r| r.method == Method::Ocr).collect();

        let mut scored = Vec::new();
        for label in fixture["truth"].as_array().unwrap() {
            let Some(text) = label["text"].as_str() else { continue };
            if text == "table" {
                continue;
            }
            let b: Vec<f64> = label["bbox"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
            let truth = BBox::new(b[0], b[1], b[2], b[3]);
            let page = label["page"].as_u64().unwrap() as u32;
            let expected_unreadable = label["unreadable"].as_bool().unwrap_or(false);
            let found = regions
                .iter()
                .filter(|r| r.page == page && squeeze(&r.text).contains(&squeeze(text)))
                .max_by(|x, y| iou(&x.bbox, &truth).partial_cmp(&iou(&y.bbox, &truth)).unwrap());
            labels += 1;
            let (read, overlap) = match found {
                Some(region) => (true, iou(&region.bbox, &truth)),
                None => (false, 0.0),
            };
            if read {
                transcribed += 1;
            }
            if overlap >= 0.3 {
                located += 1;
            }
            scored.push(json!({
                "page": page, "label": text, "expectedUnreadable": expected_unreadable,
                "transcribed": read, "iou": (overlap * 1000.0).round() / 1000.0,
                "region": found.map(|r| json!({"id": r.region_id, "bbox": [r.bbox.x0, r.bbox.y0, r.bbox.x1, r.bbox.y1], "status": r.status})),
            }));
        }
        documents.push(json!({
            "document": key, "file": file, "sha256": sha, "pages": pages,
            "elapsedMs": started.elapsed().as_millis() as u64,
            "units": outcome.batch.read.iter().map(|u| json!({
                "page": u.page, "tokens": u.tokens, "elapsedMs": u.elapsed_ms, "cached": u.cached,
                "looped": u.looped, "truncated": u.truncated, "regions": u.regions.len(),
                "unreadable": u.regions.iter().filter(|r| r.status == RegionStatus::Unreadable).count(),
                "malformed": u.regions.iter().filter(|r| r.status == RegionStatus::Malformed).count(),
                "notes": u.notes,
            })).collect::<Vec<_>>(),
            "unread": outcome.batch.unread.iter().map(|u| json!({"page": u.page, "reason": u.reason})).collect::<Vec<_>>(),
            "skew": outcome.skew.iter().map(|(p, d, m)| json!({"page": p, "degrees": d, "method": m})).collect::<Vec<_>>(),
            "labels": scored,
        }));
    }

    let evidence = json!({
        "label": "real-model run (target-machine gate)",
        "at": chrono::Utc::now().to_rfc3339(),
        "machine": {"gpu": sarathi_lib::system_analyzer::gpu_collector::installed_gpus().iter().map(|g| &g.model).collect::<Vec<_>>()},
        "llamaServer": llama_server_version(),
        "ocr": {
            "identity": identity,
            "weightsPath": weights.display().to_string(),
            "weightsSha256Measured": sha_file(&weights),
            "weightsSha256Pinned": entry.sha256,
            "projectorPath": projector.as_ref().map(|p| p.display().to_string()),
            "projectorSha256Measured": projector.as_deref().and_then(sha_file),
            "projectorHeader": projector_header,
            "modelHeader": model_header,
        },
        "scores": {
            "labels": labels,
            "fieldExtraction": if labels == 0 { 0.0 } else { transcribed as f64 / labels as f64 },
            "citationLocationIou0_3": if labels == 0 { 0.0 } else { located as f64 / labels as f64 },
        },
        "documents": documents,
        "reading": "fieldExtraction counts labelled strings transcribed anywhere on their page; \
                    citationLocation counts those whose region overlaps the measured ink with IoU \
                    >= 0.3. A smudged label (expectedUnreadable) should NOT be transcribed; it is \
                    listed so a transcription of it is visible as an invention.",
    });
    let path = write_evidence("target-ocr", &evidence);
    println!("P06 target OCR evidence: {}", path.display());
    println!("{}", serde_json::to_string_pretty(&evidence["scores"]).unwrap());
    assert!(labels > 0, "no labelled string was scored");
}

/// The P00 pack's three degraded scans (`fixtures/agent-system/v1/sources/scan`),
/// read through the production OCR path. Their answers are graded by the
/// baseline harness's cases `scan-01`..`scan-03`; this records what the local
/// model actually transcribed, so those cases can be judged against a real read.
#[tokio::test]
#[ignore = "target-machine gate: needs the installed Unlimited-OCR weights, projector and llama-server"]
async fn local_unlimited_ocr_reads_the_baseline_pack_scans_on_this_machine() {
    let app_data = app_data();
    let registry = Arc::new(ModelRegistry::load_with_discovery(&app_data).expect("the registry loads"));
    let servers = Arc::new(ModelServers::new());
    let scheduler = Arc::new(ModelScheduler::new(registry.clone(), servers.clone()));
    let work = tempfile::tempdir().expect("a work dir");
    let documents_root = work.path().join("documents");
    let store = DocumentStore::open(work.path()).expect("a store");
    let service = Arc::new(ExtractionService::new(
        &documents_root,
        Arc::new(ServedOcr { registry: registry.clone(), servers, scheduler }),
        Arc::new(NoVision("not exercised by the OCR gate".into())),
        Sidecar::resolve(),
    ));
    // What each case needs to find on its page, as the pack's expected answers state it.
    let cases = [
        ("scan-01-governing-reading", "page-01-readings.png", vec!["8.2", "governing"]),
        ("scan-02-pitting-present", "page-02-condition.png", vec!["pitting"]),
        ("scan-03-no-internal-inspection", "page-02-condition.png", vec!["not opened", "internal"]),
        ("scan-03-illegible-signature", "page-03-signature-illegible.png", vec![]),
    ];
    let mut results = Vec::new();
    for (case, file, needles) in cases {
        let path = repo().join("fixtures/agent-system/v1/sources/scan").join(file);
        let bytes = std::fs::read(&path).expect("the pack scan exists");
        let sha = hex::encode(Sha256::digest(&bytes));
        let folder = documents_root.join("attachments").join(&sha);
        std::fs::create_dir_all(&folder).unwrap();
        std::fs::write(folder.join("page-1.png"), &bytes).unwrap();
        store
            .record(NewExtraction {
                sha256: sha.clone(),
                name: file.into(),
                kind: "image".into(),
                pages: 1,
                truncated: false,
                page_text: Default::default(),
                sighting: Sighting {
                    owner_user_id: OWNER.into(),
                    conversation_id: CONVERSATION.into(),
                    message_id: "gate".into(),
                    run_id: "gate".into(),
                    at: chrono::Utc::now().to_rfc3339(),
                    ocr_model_id: None,
                    ocr_detent: None,
                },
            })
            .unwrap();
        let doc = service.authorise(&store, &sha, OWNER, CONVERSATION).expect("authorised");
        let outcome = service
            .ocr(&doc, vec![(1, None)], None, OcrDetent::Detailed, None, &CancelToken::never(),
                 Instant::now() + Duration::from_secs(600))
            .await
            .expect("the OCR pass ran");
        let regions: Vec<_> = service.regions.load(&sha).unwrap().regions.into_values().filter(|r| r.method == Method::Ocr).collect();
        let text: String = regions.iter().filter(|r| r.status == RegionStatus::Read).map(|r| r.text.to_lowercase()).collect::<Vec<_>>().join("\n");
        results.push(json!({
            "case": case, "file": file, "sha256": sha,
            "found": needles.iter().map(|n| json!({"needle": n, "transcribed": text.contains(&n.to_lowercase())})).collect::<Vec<_>>(),
            "regions": regions.len(),
            "unreadable": regions.iter().filter(|r| r.status != RegionStatus::Read).map(|r| json!({"id": r.region_id, "status": r.status, "bbox": [r.bbox.x0, r.bbox.y0, r.bbox.x1, r.bbox.y1], "notes": r.notes})).collect::<Vec<_>>(),
            "units": outcome.batch.read.iter().map(|u| json!({"tokens": u.tokens, "elapsedMs": u.elapsed_ms, "looped": u.looped, "truncated": u.truncated, "notes": u.notes})).collect::<Vec<_>>(),
            "unread": outcome.batch.unread.iter().map(|u| u.reason.clone()).collect::<Vec<_>>(),
            "transcript": text,
        }));
    }
    let evidence = json!({
        "label": "real-model run (target-machine gate)",
        "at": chrono::Utc::now().to_rfc3339(),
        "llamaServer": llama_server_version(),
        "results": results,
        "reading": "Whether the local model transcribed what each baseline scan case needs. The \
                    illegible signature on page 3 should come back unreadable or malformed, never \
                    as a name.",
    });
    let path = write_evidence("target-pack-scans", &evidence);
    println!("P06 pack scan evidence: {}", path.display());
}

#[tokio::test]
#[ignore = "target-machine gate: probes every model with an image projector on this machine"]
async fn every_model_with_a_projector_is_probed_with_an_image_on_this_machine() {
    let app_data = app_data();
    let registry = Arc::new(ModelRegistry::load_with_discovery(&app_data).expect("the registry loads"));
    let servers = Arc::new(ModelServers::new());
    let scheduler = ModelScheduler::new(registry.clone(), servers.clone());
    let sidecar = Sidecar::resolve().expect("the page rasteriser resolves");
    let work = tempfile::tempdir().expect("a work dir");
    let mut results = Vec::new();
    for entry in registry.all() {
        let wanted = entry.serves(ModelRole::Vision)
            || entry.serves(ModelRole::Reasoning)
            || entry.projector.is_some();
        if !wanted || !entry.enabled {
            continue;
        }
        if entry.projector.is_none() {
            results.push(json!({"modelId": entry.id, "probed": false,
                "why": "no image projector is bound; it is text-only here until a verified projector is bound and it passes this probe"}));
            continue;
        }
        let outcome = sarathi_lib::extraction::vision::probe_registered(
            &registry, &servers, &scheduler, &entry.id, &sidecar, work.path(), "target-gate",
        )
        .await;
        results.push(match outcome {
            Ok(record) => json!({"modelId": entry.id, "probed": true, "record": record}),
            Err(why) => json!({"modelId": entry.id, "probed": false, "why": why}),
        });
    }
    let evidence = json!({
        "label": "real-model run (target-machine gate)",
        "at": chrono::Utc::now().to_rfc3339(),
        "llamaServer": llama_server_version(),
        "results": results,
        "reading": "A model is vision-ready only when a record here says passed: true, which means \
                    its answer contained a random token drawn into the probe image on this run.",
    });
    let path = write_evidence("target-vision-probe", &evidence);
    println!("P06 vision probe evidence: {}", path.display());
}
