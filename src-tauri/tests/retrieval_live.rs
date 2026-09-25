//! P07 target-machine gates for retrieval. **Ignored here**, and not counted as
//! coverage: nothing in them can run without the installed embedding models.
//!
//! On the target machine:
//!
//! ```text
//! set ARJUN_APP_DATA=C:\Users\<you>\AppData\Roaming\com.arjun.workbench
//! cargo test --manifest-path src-tauri/Cargo.toml --test retrieval_live -- --ignored --nocapture --test-threads=1
//! ```
//!
//! 1. `qualify_every_installed_embedding_model` serves each embedding-only model
//!    in the registry (CPU, `--embedding`, the profile's pooling), measures it
//!    on the bundled labelled probes (English, and Hindi questions against
//!    English passages), writes the record where the application reads it —
//!    so a model that passes becomes the semantic half — and writes
//!    `evidence/agent-system/P07/target-embedding-qualification-<model>.json`.
//! 2. `labelled_retrieval_on_the_target` indexes the labelled corpus through the
//!    real document sidecar, embeds it with the qualified model, and measures
//!    Recall@5, MRR and no-answer correctness for keyword-only and hybrid
//!    retrieval, into `target-retrieval-measurement.json`.
//!
//! Neither writes a number it did not measure. A model that fails is recorded
//! as failing, and the measurement refuses to run without a qualified model.

use std::path::PathBuf;
use std::sync::Arc;

use sarathi_lib::identity::{Role, Session, User};
use sarathi_lib::knowledge::embedding::{bundled_probes, qualify};
use sarathi_lib::knowledge::provider::{is_embedding_only, EmbeddingSource, ProviderState, ServedEmbeddings};
use sarathi_lib::knowledge::service::RetrievalService;
use sarathi_lib::knowledge::KnowledgeIndex;
use sarathi_lib::registry::ModelRegistry;
use sarathi_lib::serving::ModelServers;
use serde_json::{json, Value};

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

fn evidence(name: &str, value: &Value) {
    let dir = repo().join("evidence/agent-system/P07");
    std::fs::create_dir_all(&dir).expect("the evidence folder");
    std::fs::write(dir.join(name), serde_json::to_string_pretty(value).unwrap()).expect("written");
    println!("wrote evidence/agent-system/P07/{name}");
}

#[tokio::test]
#[ignore = "target machine: needs the installed embedding models and llama-server"]
async fn qualify_every_installed_embedding_model() {
    let app_data = app_data();
    let registry = Arc::new(ModelRegistry::load_with_discovery(&app_data).expect("the registry loads"));
    let servers = Arc::new(ModelServers::new());
    let models: Vec<String> = registry
        .all()
        .iter()
        .filter(|entry| entry.enabled && is_embedding_only(entry))
        .map(|entry| entry.id.clone())
        .collect();
    assert!(!models.is_empty(), "no embedding-only model is registered in {}", app_data.display());
    let probes = bundled_probes();
    for model in models {
        let source = ServedEmbeddings::for_model(registry.clone(), servers.clone(), app_data.join("retrieval"), &model);
        let status = source.status();
        println!("{model}: {:?} — {}", status.state, status.detail);
        if status.state == ProviderState::Unavailable {
            evidence(&format!("target-embedding-qualification-{model}.json"), &json!({"model": model, "unavailable": status.detail}));
            continue;
        }
        let started = std::time::Instant::now();
        let embedder = source.open(true).await.unwrap_or_else(|error| panic!("{model} could not be served: {error}"));
        let record = qualify(embedder.as_ref(), &probes, "served by ARJUN on loopback (target gate)").await;
        source.store().save(&record).expect("the record is saved where the application reads it");
        evidence(
            &format!("target-embedding-qualification-{model}.json"),
            &json!({"model": model, "record": record, "elapsedMs": started.elapsed().as_millis()}),
        );
        servers.stop(&model).await;
    }
}

#[tokio::test]
#[ignore = "target machine: needs a qualified embedding model"]
async fn labelled_retrieval_on_the_target() {
    let app_data = app_data();
    let registry = Arc::new(ModelRegistry::load_with_discovery(&app_data).expect("the registry loads"));
    let servers = Arc::new(ModelServers::new());
    let source = Arc::new(ServedEmbeddings::new(registry, servers, app_data.join("retrieval")));
    let status = source.status();
    assert_eq!(status.state, ProviderState::Qualified, "run qualify_every_installed_embedding_model first: {}", status.detail);

    let work = tempfile::tempdir().expect("a work dir");
    let index = Arc::new(KnowledgeIndex::open(work.path()).expect("an index"));
    let corpus = repo().join("src-tauri/src/knowledge/testdata/retrieval");
    let collections = sarathi_lib::knowledge::CollectionStore::open(work.path()).expect("collections");
    collections
        .upsert(sarathi_lib::knowledge::connector::Collection {
            id: "plant".into(),
            name: "plant".into(),
            kind: sarathi_lib::knowledge::connector::SourceKind::LocalFolder,
            root: corpus.clone(),
            owner: "Document control".into(),
            classification: sarathi_lib::policy::Classification::Internal,
            restricted_to_roles: Vec::new(),
            retention_days: None,
            enabled: true,
        })
        .expect("saved");
    let reader = sarathi_lib::documents::DocumentService::spawn(work.path()).expect("the document sidecar starts");
    let originals = sarathi_lib::documents::DocumentStore::open(work.path()).expect("the original store");
    let collection = collections.get("plant").unwrap().unwrap();
    let synced = sarathi_lib::knowledge::ingest::sync_collection(&collection, &collections, &reader, &originals, &index, None, None, "target-gate")
        .expect("synced");
    println!("{}", synced.plan);

    let retrieval = RetrievalService::new(index.clone(), source.clone());
    let started = std::time::Instant::now();
    let pass = retrieval.reindex_once(10_000).await.expect("the embedding pass runs");
    let embed_ms = started.elapsed().as_millis();
    assert_eq!(pass.remaining, 0, "{pass:?}");

    let labels: Value = serde_json::from_str(&std::fs::read_to_string(corpus.join("labels.json")).unwrap()).unwrap();
    let session = Session::open(User::new("target-gate", "Target gate", vec![Role::Employee, Role::Administrator]));
    let lexical = RetrievalService::lexical_only(index.clone(), "measurement: keyword half alone");
    const K: usize = 5;
    let mut rows = Vec::new();
    for case in labels["cases"].as_array().unwrap() {
        let query = case["query"].as_str().unwrap().to_string();
        let relevant: Vec<(String, String)> = case["relevant"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| (r["path"].as_str().unwrap().to_string(), r["phrase"].as_str().unwrap().to_string()))
            .collect();
        let request = sarathi_lib::knowledge::HybridRequest { query: query.clone(), limit: K, ..Default::default() };
        for (method, response) in [
            ("keyword", lexical.search(&session, &request).await.unwrap()),
            ("hybrid", retrieval.search(&session, &request).await.unwrap()),
        ] {
            let hit = |h: &sarathi_lib::knowledge::EvidenceHit, (path, phrase): &(String, String)| {
                h.passage.document_name.replace('\\', "/").ends_with(path.as_str()) && h.passage.text.contains(phrase.as_str())
            };
            let found = relevant.iter().filter(|r| response.hits.iter().any(|h| hit(h, r))).count();
            let first = response.hits.iter().position(|h| relevant.iter().any(|r| hit(h, r)));
            rows.push(json!({
                "case": case["id"], "kind": case["kind"], "method": method,
                "recallAtK": if relevant.is_empty() { Value::Null } else { json!(found as f64 / relevant.len() as f64) },
                "reciprocalRank": if relevant.is_empty() { Value::Null } else { json!(first.map(|r| 1.0 / (r + 1) as f64).unwrap_or(0.0)) },
                "noAnswerCorrect": if relevant.is_empty() { json!(response.hits.is_empty()) } else { Value::Null },
                "mode": response.coverage.mode,
                "top": response.hits.iter().map(|h| format!("{} ({})", h.passage.document_name, h.method.label())).collect::<Vec<_>>(),
            }));
        }
    }
    evidence(
        "target-retrieval-measurement.json",
        &json!({
            "k": K,
            "provider": status,
            "embeddingPass": pass,
            "embeddingPassMs": embed_ms,
            "rows": rows,
            "label": "real-model run on the target machine",
        }),
    );
}
