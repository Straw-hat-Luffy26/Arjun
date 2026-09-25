//! Retrieval as one service: the index, the embedding provider, the embedding
//! pass, and the scope each run was pinned to.
//!
//! Held once and shared by everything that retrieves — the agent's tools, the
//! Knowledge Retriever worker, and the per-round context compiler — so all
//! three see the same qualification state, the same vector space and the same
//! pins. Three copies of "is the dense half on?" would be three places for the
//! answer to disagree.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};

use super::embedding::{embed_passages, Embedder, BATCH_SIZE};
use super::hybrid::{HybridRequest, HybridResponse, HybridRetriever};
use super::index::{KnowledgeIndex, SearchScope};
use super::provider::{EmbeddingSource, ProviderState, ProviderStatus};
use crate::identity::Session;

/// The notebook a run's question was scoped to, as resolved when it started.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NotebookPin {
    pub notebook_id: String,
    /// The sources selected when the run started, by content hash.
    pub source_sha256s: Vec<String>,
}

/// What one run may retrieve from, fixed when it starts or is queued.
///
/// Authorisation is not in here and never is: every read re-checks the
/// reader's clearance at the moment it runs, so a pin can narrow what a job
/// reads and cannot preserve access that has since been revoked.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RunScope {
    /// The index clock when the work was queued. `None` for live work.
    #[serde(default)]
    pub pinned_revision: Option<i64>,
    #[serde(default)]
    pub notebook: Option<NotebookPin>,
}

impl RunScope {
    pub fn search_scope(&self) -> SearchScope {
        SearchScope {
            pinned_revision: self.pinned_revision,
            ..Default::default()
        }
    }
}

/// What one embedding pass did.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReindexReport {
    pub space_key: String,
    pub embedded: usize,
    pub failed: usize,
    /// Passages still waiting when the pass stopped.
    pub remaining: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stopped_because: Option<String>,
}

/// Embeds what a space has not embedded yet, until done, cancelled or capped.
///
/// Resumable because the work list *is* the table: each batch asks the index
/// what is still missing, so a pass stopped halfway — shutdown, crash, cancel —
/// continues where it stopped the next time it runs. A passage the model
/// cannot embed is recorded as failed rather than retried forever, and counted.
pub async fn reindex(
    index: &KnowledgeIndex,
    embedder: &dyn Embedder,
    max_passages: usize,
    cancel: &AtomicBool,
) -> anyhow::Result<ReindexReport> {
    let space = embedder.identity().space_key();
    index.register_space(embedder.identity())?;
    let mut report = ReindexReport {
        space_key: space.clone(),
        ..Default::default()
    };
    while report.embedded + report.failed < max_passages {
        if cancel.load(Ordering::SeqCst) {
            report.stopped_because = Some("cancelled".into());
            break;
        }
        let batch = index.passages_needing_embeddings(&space, BATCH_SIZE)?;
        if batch.is_empty() {
            break;
        }
        let bodies: Vec<String> = batch.iter().map(|pending| pending.body.clone()).collect();
        let results = embed_passages(embedder, &bodies).await;
        for (pending, result) in batch.iter().zip(results) {
            match result {
                Ok(windows) => {
                    if index.store_embeddings(&space, &pending.chunk_id, &pending.body, &windows)? {
                        report.embedded += 1;
                    }
                }
                Err(error) => {
                    index.record_embedding_failure(&space, &pending.chunk_id, &error.to_string())?;
                    report.failed += 1;
                }
            }
        }
    }
    if report.embedded + report.failed >= max_passages {
        report.stopped_because.get_or_insert_with(|| format!("reached the {max_passages}-passage cap for one pass"));
    }
    report.remaining = index.count_needing_embeddings(&space)?;
    Ok(report)
}

/// A person's notebooks and the documents in them: the second corpus a run can
/// be scoped to, owner-authorised in every read.
pub struct NotebookCorpus {
    pub notebooks: Arc<super::NotebookStore>,
    pub documents: Arc<crate::agent_runtime::documents::DocumentStore>,
}

/// The retrieval service.
pub struct RetrievalService {
    pub index: Arc<KnowledgeIndex>,
    pub embeddings: Arc<dyn EmbeddingSource>,
    notebook_corpus: Option<NotebookCorpus>,
    scopes: Mutex<HashMap<String, RunScope>>,
    reindexing: AtomicBool,
    cancel: AtomicBool,
    last_reindex: Mutex<Option<ReindexReport>>,
}

impl RetrievalService {
    pub fn new(index: Arc<KnowledgeIndex>, embeddings: Arc<dyn EmbeddingSource>) -> Self {
        Self {
            index,
            embeddings,
            notebook_corpus: None,
            scopes: Mutex::new(HashMap::new()),
            reindexing: AtomicBool::new(false),
            cancel: AtomicBool::new(false),
            last_reindex: Mutex::new(None),
        }
    }

    pub fn status(&self) -> ProviderStatus {
        self.embeddings.status()
    }

    /// A qualified embedder, or why there is none.
    pub async fn embedder(&self) -> Result<Arc<dyn Embedder>, String> {
        let status = self.embeddings.status();
        if status.state != ProviderState::Qualified {
            return Err(status.detail);
        }
        self.embeddings.open(false).await
    }

    /// Hybrid search, with the dense half only when qualified.
    pub async fn search(&self, session: &Session, request: &HybridRequest) -> anyhow::Result<HybridResponse> {
        match self.embedder().await {
            Ok(embedder) => {
                let dense_floor = self
                    .embeddings
                    .store()
                    .get(embedder.identity())
                    .and_then(|record| record.dense_floor);
                HybridRetriever {
                    index: &self.index,
                    embedder: Some(embedder.as_ref()),
                    dense_unavailable: None,
                    dense_floor,
                }
                .search(session, request)
                .await
            }
            Err(why) => HybridRetriever::lexical_only(&self.index, why).search(session, request).await,
        }
    }

    /// A service with no embedding model: every search is keyword-only and
    /// says `reason`.
    pub fn lexical_only(index: Arc<KnowledgeIndex>, reason: &str) -> Self {
        Self::new(
            index,
            Arc::new(super::provider::NoEmbeddings::new(reason, std::path::Path::new(""))),
        )
    }

    pub fn with_notebooks(mut self, corpus: NotebookCorpus) -> Self {
        self.notebook_corpus = Some(corpus);
        self
    }

    /// Hybrid search inside a run's scope: the organisation's index at the
    /// pinned revision, and the pinned notebook's sources when there are any.
    ///
    /// The notebook half is keyword-ranked (its passages are not embedded) and
    /// says so on every hit it contributes; the two rankings are fused by rank
    /// exactly as the keyword and semantic halves are.
    pub async fn search_scoped(
        &self,
        session: &Session,
        request: &HybridRequest,
        scope: &RunScope,
    ) -> anyhow::Result<HybridResponse> {
        let mut request = request.clone();
        if request.scope.pinned_revision.is_none() {
            request.scope.pinned_revision = scope.pinned_revision;
        }
        let mut response = self.search(session, &request).await?;
        let Some(pin) = &scope.notebook else {
            return Ok(response);
        };
        let Some(corpus) = &self.notebook_corpus else {
            response.coverage.partial.push(
                "this work was scoped to a notebook, and notebooks cannot be read on this path"
                    .into(),
            );
            return Ok(response);
        };
        match self.notebook_passages(corpus, session, pin, &request.query) {
            Err(reason) => response
                .coverage
                .partial
                .push(format!("the notebook this work was scoped to was not searched: {reason}")),
            Ok((passages, notes)) => {
                response.coverage.partial.extend(notes);
                merge_notebook(&mut response, passages, request.limit.clamp(1, super::hybrid::MAX_RESULTS));
            }
        }
        response.coverage.no_answer = response.hits.is_empty();
        Ok(response)
    }

    /// The pinned notebook's passages, re-authorised now.
    ///
    /// Ownership first, then membership: a source removed from the notebook
    /// after the work was queued is not read, and is counted; a notebook the
    /// reader does not own reads exactly as one that does not exist.
    fn notebook_passages(
        &self,
        corpus: &NotebookCorpus,
        session: &Session,
        pin: &NotebookPin,
        query: &str,
    ) -> Result<(Vec<super::SearchResult>, Vec<String>), String> {
        let owner = &session.user.id;
        let notebook = corpus
            .notebooks
            .get(&pin.notebook_id, owner)
            .map_err(|error| format!("the notebook could not be read: {error}"))?
            .ok_or_else(|| "that notebook does not exist".to_string())?;
        let members: std::collections::BTreeSet<String> = corpus
            .notebooks
            .documents(&notebook.id, owner)
            .map_err(|error| format!("the notebook's sources could not be read: {error}"))?
            .into_iter()
            .map(|member| member.document_sha256)
            .collect();
        let present: Vec<String> = pin
            .source_sha256s
            .iter()
            .filter(|sha| members.contains(*sha))
            .cloned()
            .collect();
        let mut notes = Vec::new();
        let removed = pin.source_sha256s.len() - present.len();
        if removed > 0 {
            notes.push(format!(
                "{removed} source(s) were removed from \"{}\" after this work was scoped and were \
                 not read",
                notebook.name
            ));
        }
        if present.is_empty() {
            return Ok((Vec::new(), notes));
        }
        let research = super::ResearchScope {
            notebook_id: notebook.id.clone(),
            sources: Some(super::SourceSelection::subset(present)),
            ..Default::default()
        };
        let resolved = super::notebook_retrieval::resolve(&corpus.notebooks, owner, &research)?;
        let found = super::notebook_retrieval::retrieve(&corpus.documents, owner, &resolved, query);
        notes.extend(found.limitations);
        Ok((found.passages, notes))
    }

    // ── Pins ────────────────────────────────────────────────────────────

    pub fn pin(&self, run_id: &str, scope: RunScope) {
        if let Ok(mut scopes) = self.scopes.lock() {
            scopes.insert(run_id.to_string(), scope);
        }
    }

    pub fn scope_for(&self, run_id: &str) -> RunScope {
        self.scopes
            .lock()
            .ok()
            .and_then(|scopes| scopes.get(run_id).cloned())
            .unwrap_or_default()
    }

    pub fn release(&self, run_id: &str) {
        if let Ok(mut scopes) = self.scopes.lock() {
            scopes.remove(run_id);
        }
    }

    // ── The embedding pass ──────────────────────────────────────────────

    /// Runs one bounded embedding pass, unless one is already running.
    pub async fn reindex_once(&self, max_passages: usize) -> Result<ReindexReport, String> {
        if self.reindexing.swap(true, Ordering::SeqCst) {
            return Err("an embedding pass is already running".into());
        }
        self.cancel.store(false, Ordering::SeqCst);
        let outcome = match self.embedder().await {
            Ok(embedder) => reindex(&self.index, embedder.as_ref(), max_passages, &self.cancel)
                .await
                .map_err(|error| error.to_string()),
            Err(why) => Err(why),
        };
        self.reindexing.store(false, Ordering::SeqCst);
        if let Ok(report) = &outcome {
            log::info!(
                "[retrieval] embedding pass over {}: {} embedded, {} failed, {} remaining",
                report.space_key,
                report.embedded,
                report.failed,
                report.remaining
            );
            if let Ok(mut last) = self.last_reindex.lock() {
                *last = Some(report.clone());
            }
        }
        outcome
    }

    pub fn cancel_reindex(&self) {
        self.cancel.store(true, Ordering::SeqCst);
    }

    pub fn is_reindexing(&self) -> bool {
        self.reindexing.load(Ordering::SeqCst)
    }

    pub fn last_reindex(&self) -> Option<ReindexReport> {
        self.last_reindex.lock().ok().and_then(|last| last.clone())
    }
}

/// Fuses notebook passages into a response by rank.
fn merge_notebook(response: &mut HybridResponse, passages: Vec<super::SearchResult>, limit: usize) {
    use super::hybrid::{excerpt, reciprocal_rank_fusion, EvidenceHit, HitMethod, HitScores};
    if passages.is_empty() {
        return;
    }
    let index_ranking: Vec<String> = response.hits.iter().map(|hit| hit.passage.chunk_id.clone()).collect();
    let notebook_ranking: Vec<String> = passages.iter().map(|passage| passage.chunk_id.clone()).collect();
    let fused = reciprocal_rank_fusion(&[index_ranking, notebook_ranking]);
    let mut from_index: HashMap<String, super::hybrid::EvidenceHit> = response
        .hits
        .drain(..)
        .map(|hit| (hit.passage.chunk_id.clone(), hit))
        .collect();
    let mut from_notebook: HashMap<String, super::SearchResult> = passages
        .into_iter()
        .map(|passage| (passage.chunk_id.clone(), passage))
        .collect();
    for (chunk_id, score) in fused {
        if response.hits.len() >= limit {
            break;
        }
        if let Some(mut hit) = from_index.remove(&chunk_id) {
            hit.scores.fused_rank = score;
            response.hits.push(hit);
        } else if let Some(passage) = from_notebook.remove(&chunk_id) {
            response.hits.push(EvidenceHit {
                handle: format!("nb:{}", passage.chunk_id),
                excerpt: excerpt(&passage.text, &response.query, super::hybrid::EXCERPT_CHARS),
                passage,
                method: HitMethod::Lexical,
                scores: HitScores {
                    fused_rank: score,
                    ..Default::default()
                },
                source: None,
                duplicates: Vec::new(),
            });
        }
    }
    response.coverage.partial.push(
        "passages from the notebook this work was scoped to were ranked by keyword only; its \
         sources are not embedded"
            .into(),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge::embedding::{EmbedError, EmbeddingIdentity, EmbeddingProfile, MULTILINGUAL_E5_SMALL};
    use crate::knowledge::{Chunk, ChunkKind};
    use crate::policy::Classification;

    struct Flaky {
        identity: EmbeddingIdentity,
    }

    #[async_trait::async_trait]
    impl Embedder for Flaky {
        fn identity(&self) -> &EmbeddingIdentity {
            &self.identity
        }
        fn profile(&self) -> &'static EmbeddingProfile {
            &MULTILINGUAL_E5_SMALL
        }
        async fn embed_raw(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>, EmbedError> {
            // Refuses anything mentioning "poison" as the server would refuse
            // an input it cannot process; everything else gets a vector.
            if inputs.iter().any(|input| input.contains("poison")) {
                return Err(EmbedError::Failed("refused".into()));
            }
            Ok(inputs.iter().map(|_| { let mut v = vec![0.0; 384]; v[0] = 1.0; v }).collect())
        }
    }

    fn corpus(n: usize) -> (tempfile::TempDir, KnowledgeIndex) {
        let dir = tempfile::tempdir().unwrap();
        let index = KnowledgeIndex::open(dir.path()).unwrap();
        let chunks: Vec<Chunk> = (0..n)
            .map(|i| Chunk {
                id: format!("c{i:03}"),
                document_sha256: "doc".into(),
                ordinal: i as u32,
                text: if i == 3 { "poison passage".into() } else { format!("passage number {i}") },
                page: 1,
                section_path: Vec::new(),
                kind: ChunkKind::Prose,
                char_count: 10,
            })
            .collect();
        index.index_document("Doc", Classification::Internal, &chunks).unwrap();
        (dir, index)
    }

    #[tokio::test]
    async fn an_interrupted_pass_resumes_where_it_stopped_and_counts_failures() {
        let (_dir, index) = corpus(40);
        let embedder = Flaky {
            identity: EmbeddingIdentity::new("e5", &MULTILINGUAL_E5_SMALL, &"ab".repeat(32)),
        };
        let cancel = AtomicBool::new(false);
        let first = reindex(&index, &embedder, 20, &cancel).await.unwrap();
        assert!(first.embedded + first.failed >= 20);
        assert!(first.remaining > 0 && first.stopped_because.is_some());

        let second = reindex(&index, &embedder, 1000, &cancel).await.unwrap();
        assert_eq!(second.remaining, 0);
        assert_eq!(first.embedded + second.embedded, 39);
        assert_eq!(first.failed + second.failed, 1, "the refused passage was retried or lost");

        cancel.store(true, Ordering::SeqCst);
        let third = reindex(&index, &embedder, 1000, &cancel).await.unwrap();
        assert_eq!(third.embedded, 0);
    }
}
