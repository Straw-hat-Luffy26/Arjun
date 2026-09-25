//! Which embedding model this machine retrieves with, and whether it may.
//!
//! ## Three states, never two
//!
//! - **Unavailable**: no registered model matches a pinned profile, or its
//!   weights cannot be read. Retrieval is keyword-only and says why.
//! - **Unqualified**: a compatible model is registered, and nobody has measured
//!   it. It can be opened *for qualification* and for nothing else. Retrieval
//!   stays keyword-only and says which model is waiting and what command
//!   measures it.
//! - **Qualified**: [`super::embedding::qualify`] measured exactly this
//!   identity — these weights, this profile version — and it passed. Only now
//!   does the dense half run.
//!
//! A two-state design (installed / not installed) is how a deployment comes to
//! run semantic search on a model nobody checked, because "it is installed"
//! reads as "it works".
//!
//! ## Not a conversational model, by construction
//!
//! An embedding model is served with `llama-server --embedding`, on the CPU,
//! with the profile's pooling (see [`crate::serving::plan_launch`]). A server in
//! that mode refuses chat completions, and child routing refuses a model whose
//! only role is embedding (see [`crate::subagents::certification`]). The only
//! thing this module hands out is an [`Embedder`], which has no method that
//! generates text.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde::Serialize;

use super::embedding::{
    profile_for_weights, weights_sha256, Embedder, EmbeddingIdentity, EmbeddingProfile,
    LocalEmbedder, QualificationRecord, QualificationStore, PROFILES,
};

/// Where the dense half stands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum ProviderState {
    Qualified,
    Unqualified,
    Unavailable,
}

/// What a screen, a manifest or a tool result says about the dense half.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ProviderStatus {
    pub state: ProviderState,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub identity: Option<EmbeddingIdentity>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub space_key: Option<String>,
    /// One sentence a person can act on.
    pub detail: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub qualification: Option<QualificationRecord>,
}

impl ProviderStatus {
    pub fn unavailable(detail: impl Into<String>) -> Self {
        Self {
            state: ProviderState::Unavailable,
            identity: None,
            space_key: None,
            detail: detail.into(),
            qualification: None,
        }
    }

    fn for_identity(identity: EmbeddingIdentity, store: &QualificationStore) -> Self {
        let record = store.get(&identity);
        let (state, detail) = match &record {
            Some(record) if record.passed => (
                ProviderState::Qualified,
                format!(
                    "{} is qualified for retrieval ({} of {} labelled probes ordered correctly, \
                     measured {})",
                    identity.model_id, record.ordered_correctly, record.probes, record.measured_at
                ),
            ),
            Some(record) => (
                ProviderState::Unqualified,
                format!(
                    "{} failed qualification on {} ({}), so retrieval is keyword-only",
                    identity.model_id,
                    record.measured_at,
                    record
                        .checks
                        .iter()
                        .filter(|check| !check.passed)
                        .map(|check| format!("{}: {}", check.name, check.detail))
                        .collect::<Vec<_>>()
                        .join("; ")
                ),
            ),
            None => (
                ProviderState::Unqualified,
                format!(
                    "{} is installed and matches the {} profile, but has not been qualified on \
                     this machine, so retrieval is keyword-only. Qualify it from the Knowledge \
                     screen or with the retrieval_live test.",
                    identity.model_id, identity.profile
                ),
            ),
        };
        Self {
            state,
            space_key: Some(identity.space_key()),
            identity: Some(identity),
            detail,
            qualification: record,
        }
    }
}

/// Something that can hand out an embedder.
#[async_trait]
pub trait EmbeddingSource: Send + Sync {
    /// Where the dense half stands. Cheap after the first call.
    fn status(&self) -> ProviderStatus;
    /// An embedder for retrieval. Refused unless qualified — or, when
    /// `for_qualification` is set, when merely compatible, which is how the
    /// qualification itself gets a model to measure.
    async fn open(&self, for_qualification: bool) -> Result<Arc<dyn Embedder>, String>;
    fn store(&self) -> &QualificationStore;
}

/// A deployment with no embedding model at all.
pub struct NoEmbeddings {
    reason: String,
    store: QualificationStore,
}

impl NoEmbeddings {
    pub fn new(reason: impl Into<String>, retrieval_dir: &std::path::Path) -> Self {
        Self {
            reason: reason.into(),
            store: QualificationStore::new(retrieval_dir),
        }
    }
}

#[async_trait]
impl EmbeddingSource for NoEmbeddings {
    fn status(&self) -> ProviderStatus {
        ProviderStatus::unavailable(self.reason.clone())
    }
    async fn open(&self, _: bool) -> Result<Arc<dyn Embedder>, String> {
        Err(self.reason.clone())
    }
    fn store(&self) -> &QualificationStore {
        &self.store
    }
}

/// An embedding server already running on loopback, started by someone else.
///
/// For an operator who serves the model themselves, and for the target-machine
/// qualification run. The identity is still measured from the weights file —
/// the endpoint's word for what it is serving is not taken.
pub struct EndpointEmbeddings {
    base_url: String,
    identity: EmbeddingIdentity,
    store: QualificationStore,
}

impl EndpointEmbeddings {
    pub fn new(base_url: String, identity: EmbeddingIdentity, retrieval_dir: &std::path::Path) -> Result<Self, String> {
        crate::serving::probe::check_loopback(&base_url).map_err(|outcome| outcome.explain(&base_url))?;
        Ok(Self {
            base_url,
            identity,
            store: QualificationStore::new(retrieval_dir),
        })
    }

    /// From a weights file on disk and the loopback URL serving it.
    pub fn for_weights(
        base_url: String,
        model_id: &str,
        weights: &std::path::Path,
        retrieval_dir: &std::path::Path,
    ) -> Result<Self, String> {
        let profile = profile_for_weights(weights).ok_or_else(|| {
            format!("{} matches no pinned embedding profile", weights.display())
        })?;
        let sha = weights_sha256(weights, retrieval_dir).map_err(|error| error.to_string())?;
        Self::new(base_url, EmbeddingIdentity::new(model_id, profile, &sha), retrieval_dir)
    }
}

#[async_trait]
impl EmbeddingSource for EndpointEmbeddings {
    fn status(&self) -> ProviderStatus {
        ProviderStatus::for_identity(self.identity.clone(), &self.store)
    }
    async fn open(&self, for_qualification: bool) -> Result<Arc<dyn Embedder>, String> {
        if !for_qualification && !self.store.is_qualified(&self.identity) {
            return Err(self.status().detail);
        }
        LocalEmbedder::new(self.base_url.clone(), self.identity.clone())
            .map(|embedder| Arc::new(embedder) as Arc<dyn Embedder>)
            .map_err(|error| error.to_string())
    }
    fn store(&self) -> &QualificationStore {
        &self.store
    }
}

/// The registry's embedding model, served by ARJUN on this machine.
pub struct ServedEmbeddings {
    registry: Arc<crate::registry::ModelRegistry>,
    servers: Arc<crate::serving::ModelServers>,
    retrieval_dir: PathBuf,
    store: QualificationStore,
    /// An operator's choice, by registry id. Otherwise the plan's order.
    preferred: Option<String>,
    resolved: Mutex<Option<Resolved>>,
}

#[derive(Clone)]
struct Resolved {
    entry: crate::registry::ModelEntry,
    profile: &'static EmbeddingProfile,
    identity: EmbeddingIdentity,
    fingerprint: (u64, u64),
}

impl ServedEmbeddings {
    pub fn new(
        registry: Arc<crate::registry::ModelRegistry>,
        servers: Arc<crate::serving::ModelServers>,
        retrieval_dir: PathBuf,
    ) -> Self {
        Self {
            store: QualificationStore::new(&retrieval_dir),
            registry,
            servers,
            retrieval_dir,
            preferred: std::env::var("ARJUN_EMBEDDING_MODEL").ok().filter(|id| !id.trim().is_empty()),
            resolved: Mutex::new(None),
        }
    }

    /// The same, for one named model — how a qualification run measures each
    /// installed embedding model in turn.
    pub fn for_model(
        registry: Arc<crate::registry::ModelRegistry>,
        servers: Arc<crate::serving::ModelServers>,
        retrieval_dir: PathBuf,
        model_id: &str,
    ) -> Self {
        let mut served = Self::new(registry, servers, retrieval_dir);
        served.preferred = Some(model_id.to_string());
        served
    }

    fn fingerprint(path: &std::path::Path) -> Option<(u64, u64)> {
        let metadata = std::fs::metadata(path).ok()?;
        let modified = metadata
            .modified()
            .ok()?
            .duration_since(std::time::UNIX_EPOCH)
            .ok()?
            .as_secs();
        Some((metadata.len(), modified))
    }

    /// The model to use, and its measured identity.
    ///
    /// Candidates are enabled registry entries whose roles are embedding only
    /// and whose weights match a pinned profile. Among them: the operator's
    /// choice; otherwise a qualified one in the plan's profile order; otherwise
    /// the first compatible one, which is then reported as unqualified.
    fn resolve(&self) -> Result<Resolved, String> {
        let models_dir = self.registry.models_dir().to_path_buf();
        if let Ok(held) = self.resolved.lock() {
            if let Some(resolved) = held.as_ref() {
                let weights = models_dir.join(&resolved.entry.path);
                if Self::fingerprint(&weights) == Some(resolved.fingerprint) {
                    return Ok(resolved.clone());
                }
            }
        }

        let mut compatible: Vec<Resolved> = Vec::new();
        let mut skipped: Vec<String> = Vec::new();
        for entry in self.registry.all() {
            if !entry.enabled || !is_embedding_only(entry) {
                continue;
            }
            if let Some(preferred) = &self.preferred {
                if &entry.id != preferred {
                    continue;
                }
            }
            let weights = models_dir.join(&entry.path);
            let Some(profile) = profile_for_weights(&weights) else {
                skipped.push(format!("{} (no pinned profile matches its header)", entry.id));
                continue;
            };
            let Some(fingerprint) = Self::fingerprint(&weights) else {
                skipped.push(format!("{} (its weights file cannot be read)", entry.id));
                continue;
            };
            let sha = match weights_sha256(&weights, &self.retrieval_dir) {
                Ok(sha) => sha,
                Err(error) => {
                    skipped.push(format!("{} ({error})", entry.id));
                    continue;
                }
            };
            compatible.push(Resolved {
                entry: entry.clone(),
                profile,
                identity: EmbeddingIdentity::new(&entry.id, profile, &sha),
                fingerprint,
            });
        }
        if compatible.is_empty() {
            return Err(match (&self.preferred, skipped.is_empty()) {
                (Some(preferred), _) => format!(
                    "ARJUN_EMBEDDING_MODEL names {preferred}, which is not a usable embedding \
                     model here{}",
                    if skipped.is_empty() { String::new() } else { format!(": {}", skipped.join(", ")) }
                ),
                (None, true) => "no embedding model is registered on this machine, so retrieval \
                                 is keyword-only"
                    .to_string(),
                (None, false) => format!(
                    "no registered embedding model matches a pinned profile ({}), so retrieval is \
                     keyword-only",
                    skipped.join(", ")
                ),
            });
        }
        let rank = |resolved: &Resolved| {
            PROFILES
                .iter()
                .position(|profile| profile.key == resolved.profile.key)
                .unwrap_or(usize::MAX)
        };
        compatible.sort_by_key(|resolved| (!self.store.is_qualified(&resolved.identity), rank(resolved)));
        let chosen = compatible.remove(0);
        if let Ok(mut held) = self.resolved.lock() {
            *held = Some(chosen.clone());
        }
        Ok(chosen)
    }
}

/// Whether a registry entry is an embedding model and nothing else.
pub fn is_embedding_only(entry: &crate::registry::ModelEntry) -> bool {
    !entry.roles.is_empty()
        && entry
            .roles
            .iter()
            .all(|role| *role == crate::registry::ModelRole::Embedding)
}

#[async_trait]
impl EmbeddingSource for ServedEmbeddings {
    fn status(&self) -> ProviderStatus {
        match self.resolve() {
            Ok(resolved) => ProviderStatus::for_identity(resolved.identity, &self.store),
            Err(reason) => ProviderStatus::unavailable(reason),
        }
    }

    async fn open(&self, for_qualification: bool) -> Result<Arc<dyn Embedder>, String> {
        let resolved = self.resolve()?;
        if !for_qualification && !self.store.is_qualified(&resolved.identity) {
            return Err(ProviderStatus::for_identity(resolved.identity, &self.store).detail);
        }
        // CPU, at the profile's window. Not admitted against the card: an
        // embedding pass must never be the reason a generation is evicted.
        let plan = crate::ai_engine::vram_planner::GpuOffloadPlan {
            gpu_layers: 0,
            full_offload: false,
            context_length: resolved.profile.max_tokens,
            reason: "embedding model: CPU only, at its profile's window (plan §4)".to_string(),
        };
        let endpoint = self
            .servers
            .endpoint_for(&resolved.entry, self.registry.models_dir(), &plan)
            .await
            .map_err(|error| error.to_string())?;
        crate::serving::probe::check_loopback(&endpoint.base_url)
            .map_err(|outcome| format!("refusing to embed off-machine: {}", outcome.explain(&endpoint.base_url)))?;
        LocalEmbedder::new(endpoint.base_url, resolved.identity)
            .map(|embedder| Arc::new(embedder) as Arc<dyn Embedder>)
            .map_err(|error| error.to_string())
    }

    fn store(&self) -> &QualificationStore {
        &self.store
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge::embedding::MULTILINGUAL_E5_SMALL;

    #[tokio::test]
    async fn an_unqualified_endpoint_opens_for_qualification_and_nothing_else() {
        let dir = tempfile::tempdir().unwrap();
        let identity = EmbeddingIdentity::new("e5", &MULTILINGUAL_E5_SMALL, &"ab".repeat(32));
        let source = EndpointEmbeddings::new("http://127.0.0.1:9/v1".into(), identity.clone(), dir.path()).unwrap();
        assert_eq!(source.status().state, ProviderState::Unqualified);
        assert!(source.open(false).await.is_err(), "an unmeasured model was handed to retrieval");
        assert!(source.open(true).await.is_ok());

        let record = QualificationRecord {
            identity: identity.clone(),
            space_key: identity.space_key(),
            passed: true,
            checks: Vec::new(),
            ordered_correctly: 10,
            probes: 10,
            relevant_mean: Some(0.8),
            distractor_mean: Some(0.4),
            dense_floor: Some(0.6),
            measured_at: "2026-09-25T00:00:00Z".into(),
            endpoint: "http://127.0.0.1:9/v1".into(),
        };
        source.store().save(&record).unwrap();
        assert_eq!(source.status().state, ProviderState::Qualified);
        assert!(source.open(false).await.is_ok());
    }

    #[test]
    fn an_endpoint_off_this_machine_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let identity = EmbeddingIdentity::new("e5", &MULTILINGUAL_E5_SMALL, &"ab".repeat(32));
        assert!(EndpointEmbeddings::new("http://10.1.2.3:8080/v1".into(), identity, dir.path()).is_err());
    }
}
