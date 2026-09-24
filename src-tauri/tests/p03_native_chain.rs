//! P03 native gate: Spark → OCR → Qwen → reviewer → Spark on real servers.
//!
//! ## What this proves, on the machine it runs on
//!
//! Real `llama-server` processes, real weights, the production lease service
//! (`ModelScheduler::new`, whose backend is `serving::admission` plus
//! `ModelServers`), and each model's own chat template and tokenizer counting
//! the request it would be sent. Five hops, one heavy call at a time; each hop
//! compiles the task's context for the window its server *actually* came up
//! with, from a memory graph holding a file already written and — from the
//! third hop on — an operator's correction. Every hop must carry each exactly
//! once. The report records served windows, load times, cache state, exact
//! token counts and the holders seen at each hop.
//!
//! ## What it does not prove
//!
//! That the specialist *roles* work. No model generates here: the extraction,
//! drafting and review roles are built in P06, P09 and P10, and the full native
//! chain with those roles is P10's gate, repeated in P16. This is the serving,
//! scheduling and context half of it — the half P03 owns.
//!
//! ## How it is run
//!
//! Never by `cargo test` on its own: it is `#[ignore]`d, and the only runner is
//! `node scripts/p03-native-gate.mjs`, which checks every prerequisite first and
//! reports the gate **blocked** — not passed — when any is missing.

use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::Instant;

use sarathi_lib::agent_runtime::context_compiler::{ContextCompiler, FrozenScope, Reserves};
use sarathi_lib::agent_runtime::context_manifest::{ContextManifest, HistoryBinding};
use sarathi_lib::agent_runtime::memory::Acl;
use sarathi_lib::identity::{Role, Session, User};
use sarathi_lib::knowledge::graph::runtime_memory::{
    item_id, ArtifactRef, ItemStatus, MemoryItem, MemoryKind, MemoryScope, Provenance,
};
use sarathi_lib::knowledge::graph::runtime_store::MemoryGraph;
use sarathi_lib::policy::Classification;
use sarathi_lib::registry::ModelRegistry;
use sarathi_lib::serving::ModelServers;
use sarathi_lib::subagents::{LeaseClass, LeaseRequest, ModelScheduler};
use serde_json::{json, Value};

const TASK: &str = "p03-native-chain";
const NOTE_SHA: &str = "4be1c6a95e9d6c3b27f0e2a1d8c7b6a594837261504f3e2d1c0b9a8f7e6d5c4b";

fn required(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| {
        panic!("{name} is not set. Run this through `node scripts/p03-native-gate.mjs`.")
    })
}

fn item(kind: MemoryKind, content: &str, provenance: Provenance) -> MemoryItem {
    MemoryItem {
        item_id: item_id(),
        revision: 1,
        kind,
        agent_id: "ag-spark".into(),
        scope: MemoryScope::Task {
            task_id: TASK.into(),
        },
        classification: Classification::Internal,
        acl: Acl::for_classification(Classification::Internal, None),
        creator_model_id: None,
        creator_run_id: Some(TASK.into()),
        provenance,
        content: content.into(),
        sources: Vec::new(),
        artifacts: Vec::new(),
        confidence: None,
        status: ItemStatus::Proposed,
        valid_from: "2026-09-24T00:00:00Z".into(),
        valid_until: None,
        supersedes: None,
        conflicts_with: Vec::new(),
        causal_parents: Vec::new(),
        idempotency_key: None,
        applies_to: None,
        created_at: "2026-09-24T00:00:00Z".into(),
        updated_at: "2026-09-24T00:00:00Z".into(),
    }
}

fn vram_used_mib() -> Option<u64> {
    let output = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .next()?
        .trim()
        .parse()
        .ok()
}

#[test]
#[ignore = "native: run through `node scripts/p03-native-gate.mjs`, which checks prerequisites"]
fn spark_ocr_qwen_reviewer_spark_on_real_servers() {
    let models_dir = std::path::PathBuf::from(required("ARJUN_MODELS_DIR"));
    let spark = required("ARJUN_P03_SPARK");
    let ocr = required("ARJUN_P03_OCR");
    let qwen = required("ARJUN_P03_QWEN");
    let reviewer = required("ARJUN_P03_REVIEWER");
    let report_path = std::path::PathBuf::from(required("ARJUN_P03_REPORT"));

    let registry = Arc::new(ModelRegistry::load(&models_dir).expect("the registry loads"));
    for id in [&spark, &ocr, &qwen, &reviewer] {
        assert!(registry.find(id).is_some(), "{id} is not in {}", models_dir.display());
    }
    let servers = Arc::new(ModelServers::new());
    let scheduler = Arc::new(ModelScheduler::new(Arc::clone(&registry), Arc::clone(&servers)));

    let store = tempfile::tempdir().expect("a directory");
    let graph = MemoryGraph::open(store.path()).expect("the graph opens");
    let session = Session::open(User::new("gate", "P03 gate", vec![Role::Employee]));
    let mut receipt = item(
        MemoryKind::ToolObservation,
        "approval-note.docx was written",
        Provenance::ToolReceipt {
            run_id: TASK.into(),
            tool: "artifact.create_approval_note".into(),
            event_seq: 1,
        },
    );
    receipt.artifacts = vec![ArtifactRef {
        artifact_id: "art-note".into(),
        revision: 1,
        sha256: NOTE_SHA.into(),
    }];
    graph.commit(receipt, None, &[]).expect("receipt");
    graph
        .commit(
            item(
                MemoryKind::Goal,
                "Produce the approval note from the scanned inspection report",
                Provenance::Operator {
                    user_id: "gate".into(),
                },
            ),
            None,
            &[],
        )
        .expect("goal");
    let exact_receipt = format!("art-note@1 sha256:{NOTE_SHA}");
    let correction = "design pressure is 10 bar";

    let hops: [(&str, &str, LeaseClass, Option<&str>); 5] = [
        ("spark", spark.as_str(), LeaseClass::Parent, None),
        ("ocr", ocr.as_str(), LeaseClass::Ocr, Some("spark")),
        ("qwen", qwen.as_str(), LeaseClass::Child, Some("spark")),
        ("reviewer", reviewer.as_str(), LeaseClass::Child, Some("spark")),
        ("spark", spark.as_str(), LeaseClass::Parent, None),
    ];

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("a runtime");
    let mut records: Vec<Value> = Vec::new();
    let mut failures: Vec<String> = Vec::new();

    for (index, (owner, model, class, parent)) in hops.iter().enumerate() {
        // The correction arrives after the OCR hop, while the chain runs.
        if index == 2 {
            graph
                .commit(
                    item(
                        MemoryKind::Correction,
                        "The design pressure is 10 bar; the 150 psi reading is superseded",
                        Provenance::Operator {
                            user_id: "gate".into(),
                        },
                    ),
                    None,
                    &[],
                )
                .expect("correction");
        }
        let record = runtime.block_on(async {
            if let Some(parent) = parent {
                scheduler.suspend(parent, Some(owner), "the gate delegated");
            }
            let mut request = LeaseRequest::new(*owner, *class, *model);
            if let Some(parent) = parent {
                request = request.child_of(*parent);
            }
            let guard = scheduler.acquire(request).await.expect("the lease");
            let holders = scheduler.snapshot().holders.len();

            let started = Instant::now();
            let served = scheduler.ensure_served(owner, model).await;
            let load_ms = started.elapsed().as_millis() as u64;
            let rebind = match served {
                Ok(rebind) => rebind,
                Err(refusal) => {
                    drop(guard);
                    scheduler.forget(owner);
                    return json!({ "hop": index, "owner": owner, "model": model, "loadFailed": refusal.explain() });
                }
            };

            let compiled = ContextCompiler::new(&graph)
                .compile(
                    &session,
                    &FrozenScope {
                        task_id: TASK.into(),
                        agent_id: format!("ag-{owner}"),
                        definition_version: 1,
                        graph_revision: 0,
                        model_id: rebind.model_id.clone(),
                        template_id: None,
                        served_window: rebind.served_window,
                        project_id: None,
                        capability: None,
                        now: chrono::Utc::now().to_rfc3339(),
                    },
                    ContextManifest::new(
                        TASK,
                        "gate",
                        "",
                        "",
                        &rebind.model_id,
                        rebind.served_window,
                        Vec::new(),
                        None,
                        HistoryBinding {
                            carried: 0,
                            dropped: 0,
                            tokens: 0,
                            pinned: Vec::new(),
                            omitted_pins: Vec::new(),
                        },
                    ),
                    "approval note design pressure",
                    &BTreeSet::new(),
                    Reserves {
                        tool_schemas: 0,
                        output: 1_024,
                        framing: 256,
                        safety: (rebind.served_window / 33).max(256),
                        counted_by: "estimate".into(),
                        window_source: Some(rebind.window_source.clone()),
                    },
                )
                .expect("compiles");
            let joined = compiled
                .blocks
                .iter()
                .map(|block| block.content.clone())
                .collect::<Vec<_>>()
                .join("\n");
            let payload = json!({
                "model": rebind.served_model_id,
                "messages": [
                    { "role": "system", "content": format!("--- TASK MEMORY ---\n{joined}") },
                    { "role": "user", "content": "What is the design pressure for the approval note?" }
                ],
                "max_tokens": 1_024,
            });
            let counted = sarathi_lib::serving::probe::count_rendered(&rebind.base_url, &payload).await;
            let receipt_times = joined.matches(&exact_receipt).count();
            let correction_times = joined.matches(correction).count();

            drop(guard);
            if parent.is_some() {
                scheduler.forget(owner);
            } else {
                scheduler.release_round(owner);
            }
            json!({
                "hop": index,
                "owner": owner,
                "class": class,
                "model": model,
                "servedModelId": rebind.served_model_id,
                "servedWindow": rebind.served_window,
                "windowSource": rebind.window_source,
                "warm": rebind.warm,
                "restarted": rebind.restarted,
                "cache": rebind.cache,
                "loadMs": load_ms,
                "holdersWhileHeld": holders,
                "vramUsedMiB": vram_used_mib(),
                "renderedTokens": counted.as_ref().ok(),
                "countError": counted.as_ref().err(),
                "mandatoryTokens": compiled.mandatory_tokens(),
                "mandatoryOverflowed": compiled.mandatory_overflowed,
                "receiptCarried": receipt_times,
                "correctionCarried": correction_times,
                "manifestHash": compiled.manifest.manifest_hash,
                "graphRevision": compiled.cursor(),
            })
        });

        if let Some(reason) = record.get("loadFailed") {
            failures.push(format!("hop {index} ({model}) did not load: {reason}"));
        } else {
            if record["holdersWhileHeld"].as_u64() != Some(1) {
                failures.push(format!("hop {index}: {} holders while held", record["holdersWhileHeld"]));
            }
            if record["receiptCarried"].as_u64() != Some(1) {
                failures.push(format!("hop {index}: the prior write was carried {} time(s)", record["receiptCarried"]));
            }
            let expected_correction = if index >= 2 { 1 } else { 0 };
            if record["correctionCarried"].as_u64() != Some(expected_correction) {
                failures.push(format!(
                    "hop {index}: the correction was carried {} time(s), expected {expected_correction}",
                    record["correctionCarried"]
                ));
            }
        }
        records.push(record);
    }

    let idle = scheduler.snapshot().is_idle();
    runtime.block_on(servers.stop_all());
    if !idle {
        failures.push("a lease was still held after the chain".into());
    }

    let report = json!({
        "gate": "P03 native: Spark -> OCR -> Qwen -> reviewer -> Spark",
        "labelled": "real-model serving run; no model generated text, the specialist roles are P06/P09/P10",
        "modelsDir": models_dir,
        "hops": records,
        "failures": failures,
        "passed": failures.is_empty(),
    });
    std::fs::write(&report_path, serde_json::to_vec_pretty(&report).expect("serialises"))
        .expect("the report is written");
    assert!(failures.is_empty(), "{failures:#?}");
}
