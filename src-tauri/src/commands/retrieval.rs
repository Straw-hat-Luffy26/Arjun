//! The commands behind the Knowledge screen's collections and retrieval panel
//! (P07).
//!
//! ## What was missing
//!
//! A connector with no door. `knowledge::connector` could walk a folder,
//! `knowledge::ingest` could take it in and `knowledge::collections` could
//! remember it — and not one of them had a caller, so the index behind
//! `knowledge.search_authorized` was empty on every installation. These are the
//! door: define a collection, sync it, see what retrieval can do on this
//! machine, qualify the embedding model, and run the embedding pass.
//!
//! Every decision stays where it was: clearance in the index's SQL, versions in
//! the index, invalidation in the memory graph, qualification in the provider.
//! These commands establish who is asking and hand that down.

use std::sync::Arc;

use serde::Serialize;
use tauri::{AppHandle, Manager, State};

use crate::commands::governance::{require_permission, require_session, CurrentSession};
use crate::identity::Permission;
use crate::knowledge::connector::Collection;
use crate::knowledge::embedding::QualificationRecord;
use crate::knowledge::index::EmbeddingCoverage;
use crate::knowledge::ingest::SyncReport;
use crate::knowledge::provider::{ProviderState, ProviderStatus};
use crate::knowledge::service::ReindexReport;
use crate::knowledge::CollectionStore;

use super::agent::RetrievalState;

/// Most passages one embedding pass started from the screen embeds.
const REINDEX_BATCH: usize = 5_000;

/// One collection as the screen lists it.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CollectionView {
    #[serde(flatten)]
    pub collection: Collection,
    pub last_synced_at: Option<String>,
    /// Current sources in the index.
    pub current_sources: usize,
}

/// What retrieval can do on this machine, for the signed-in reader.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RetrievalStatus {
    pub provider: ProviderStatus,
    /// Over the passages this reader may see, never the whole index.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub coverage: Option<EmbeddingCoverage>,
    pub reindexing: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_reindex: Option<ReindexReport>,
    pub index_revision: i64,
}

fn service(app: &AppHandle) -> Result<Arc<crate::knowledge::service::RetrievalService>, String> {
    app.try_state::<RetrievalState>()
        .map(|state| Arc::clone(&state.0))
        .ok_or_else(|| "retrieval is not configured in this process".to_string())
}

fn graph(app: &AppHandle) -> Option<Arc<crate::knowledge::graph::runtime_store::MemoryGraph>> {
    app.try_state::<Arc<crate::knowledge::graph::runtime_store::MemoryGraph>>()
        .map(|graph| Arc::clone(&graph))
}

/// The collections this person may read, or every one for an administrator.
#[tauri::command]
pub async fn knowledge_collections(
    app: AppHandle,
    collections: State<'_, Arc<CollectionStore>>,
    session: State<'_, CurrentSession>,
) -> Result<Vec<CollectionView>, String> {
    let signed_in = require_session(&session)?;
    let all = collections.list().map_err(|error| error.to_string())?;
    let index = service(&app)?.index.clone();
    let administers = signed_in.user.roles.contains(&crate::identity::Role::Administrator);
    Ok(all
        .into_iter()
        .filter(|collection| administers || collection.readable_by(&signed_in.user.roles))
        .map(|collection| CollectionView {
            last_synced_at: collections.last_synced_at(&collection.id),
            current_sources: index.current_paths(&collection.id).map(|paths| paths.len()).unwrap_or(0),
            collection,
        })
        .collect())
}

/// Defines or changes a collection. Its access takes effect immediately.
#[tauri::command]
pub async fn knowledge_collection_save(
    app: AppHandle,
    collection: Collection,
    collections: State<'_, Arc<CollectionStore>>,
    session: State<'_, CurrentSession>,
) -> Result<(), String> {
    require_permission(&session, Permission::ModifyPolicy)?;
    if collection.id.trim().is_empty() || collection.name.trim().is_empty() {
        return Err("A collection needs an id and a name.".into());
    }
    let index = service(&app)?.index.clone();
    // Access before anything else, so narrowing a collection is in force for
    // the next search rather than the next sync.
    index
        .set_collection_access(&collection.id, collection.enabled, &collection.restricted_to_roles)
        .map_err(|error| error.to_string())?;
    collections.upsert(collection).map_err(|error| error.to_string())
}

/// Removes a collection: its sources stop answering and claims resting on
/// them are rejected. The passages stay traceable.
#[tauri::command]
pub async fn knowledge_collection_remove(
    app: AppHandle,
    collection_id: String,
    collections: State<'_, Arc<CollectionStore>>,
    session: State<'_, CurrentSession>,
) -> Result<usize, String> {
    require_permission(&session, Permission::ModifyPolicy)?;
    let index = service(&app)?.index.clone();
    let graph = graph(&app);
    let withdrawn = crate::knowledge::ingest::withdraw_collection(&collection_id, &index, graph.as_deref())
        .map_err(|error| error.to_string())?;
    collections.remove(&collection_id).map_err(|error| error.to_string())?;
    Ok(withdrawn)
}

/// Reads a collection's folder into the index.
///
/// On the blocking pool: the document sidecar is a Python process read over
/// stdin, and a share of thousands of files is minutes of work. An embedding
/// pass follows in the background when a qualified model is installed.
#[tauri::command]
pub async fn knowledge_collection_sync(
    app: AppHandle,
    collection_id: String,
    collections: State<'_, Arc<CollectionStore>>,
    session: State<'_, CurrentSession>,
) -> Result<SyncReport, String> {
    let signed_in = require_permission(&session, Permission::UploadDocument)?;
    let collection = collections
        .get(&collection_id)
        .map_err(|error| error.to_string())?
        .ok_or_else(|| format!("There is no collection {collection_id}."))?;
    let retrieval = service(&app)?;
    let graph = graph(&app);
    let audit = app
        .try_state::<Arc<crate::audit::AuditService>>()
        .map(|audit| Arc::clone(&audit));
    let data_dir = app.path().app_data_dir().map_err(|error| error.to_string())?;
    let store = Arc::clone(&collections);
    let index = retrieval.index.clone();
    let actor = signed_in.user.id.clone();

    let report = tokio::task::spawn_blocking(move || -> Result<SyncReport, String> {
        let reader = crate::documents::DocumentService::spawn(&data_dir)
            .map_err(|error| format!("the document reader could not start: {error}"))?;
        let originals = crate::documents::DocumentStore::open(&data_dir)
            .map_err(|error| format!("the document store could not open: {error}"))?;
        crate::knowledge::ingest::sync_collection(
            &collection,
            &store,
            &reader,
            &originals,
            &index,
            graph.as_deref(),
            audit.as_deref(),
            &actor,
        )
        .map_err(|error| error.to_string())
    })
    .await
    .map_err(|error| format!("the sync stopped unexpectedly: {error}"))??;

    if retrieval.status().state == ProviderState::Qualified && report.outcome.chunks_indexed > 0 {
        let background = Arc::clone(&retrieval);
        tauri::async_runtime::spawn(async move {
            if let Err(reason) = background.reindex_once(REINDEX_BATCH).await {
                log::warn!("[retrieval] the embedding pass after a sync did not run: {reason}");
            }
        });
    }
    Ok(report)
}

/// Which half of retrieval runs here, and how much it covers for this reader.
#[tauri::command]
pub async fn knowledge_retrieval_status(
    app: AppHandle,
    session: State<'_, CurrentSession>,
) -> Result<RetrievalStatus, String> {
    let signed_in = require_session(&session)?;
    let retrieval = service(&app)?;
    let provider = retrieval.status();
    let coverage = provider.space_key.as_ref().and_then(|space| {
        retrieval
            .index
            .embedding_coverage(&signed_in, space, &Default::default())
            .ok()
    });
    Ok(RetrievalStatus {
        provider,
        coverage,
        reindexing: retrieval.is_reindexing(),
        last_reindex: retrieval.last_reindex(),
        index_revision: retrieval.index.index_revision().unwrap_or(0),
    })
}

/// Measures the registered embedding model on the bundled labelled probes and
/// keeps the record, passed or failed.
///
/// The only way a model becomes usable for retrieval. Nothing is assumed: the
/// model is served, embedded against, and judged on what it returned.
#[tauri::command]
pub async fn knowledge_embedding_qualify(
    app: AppHandle,
    session: State<'_, CurrentSession>,
) -> Result<QualificationRecord, String> {
    require_permission(&session, Permission::ImportModel)?;
    let retrieval = service(&app)?;
    let embedder = retrieval.embeddings.open(true).await?;
    let probes = crate::knowledge::embedding::bundled_probes();
    let record = crate::knowledge::embedding::qualify(embedder.as_ref(), &probes, "served by ARJUN on loopback").await;
    retrieval
        .embeddings
        .store()
        .save(&record)
        .map_err(|error| format!("the qualification was measured and could not be saved: {error}"))?;
    Ok(record)
}

/// Runs one bounded embedding pass now.
#[tauri::command]
pub async fn knowledge_reindex(app: AppHandle, session: State<'_, CurrentSession>) -> Result<ReindexReport, String> {
    require_permission(&session, Permission::UploadDocument)?;
    service(&app)?.reindex_once(REINDEX_BATCH).await
}
