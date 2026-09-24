//! Whether a model can see, established by making it look.
//!
//! ## The rule
//!
//! A model is **vision-ready** only after an actual image call on this machine
//! returned text that could only have come from the image. A projector on disk,
//! a role in a manifest, a name containing "VL", a successful server start —
//! none of these is that. `llama-server` given a mismatched or missing
//! projector starts, accepts image requests and answers from the text alone,
//! fluently. The only test that separates a model that saw the image from one
//! that guessed is to put something in the image that cannot be guessed.
//!
//! So [`qualify`] draws a fresh random token into a PNG (via the page
//! rasteriser, [`super::sidecar::Sidecar::probe_image`]), asks the model to read
//! it, and records a pass only when the answer contains the token. The record
//! keeps the token, the answer excerpt, the projector's measured SHA-256 and
//! both files' sizes; a record stops applying when the weights or the projector
//! on disk no longer match it.
//!
//! ## What a ready model is used for
//!
//! Interpretation — "what does this symbol connect to", "which valve is this
//! label next to" — over a crop the analyst selected. Its answer becomes a
//! region with [`Method::VisionInference`]: a labelled proposal, never a
//! transcription, never an observation.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use base64::Engine as _;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::registry::{Modality, ModelEntry, ModelRole};

use super::regions::{region_id, sha256_file, CropRecord, EvidenceRegion, Extractor, Method, RegionStatus};

pub const READINESS_FILE: &str = "vision-readiness.json";
const READINESS_SCHEMA: u32 = 1;

/// The question the probe asks. Plain, so a model that can see has no reason
/// to answer anything but the token.
pub const PROBE_PROMPT: &str = "Read the text in this image. Reply with exactly that text and nothing else.";

/// Characters a probe token is drawn from: no 0/O, 1/I, 5/S, 8/B, 2/Z pairs a
/// model could confuse, so a failure means it did not see rather than it
/// misread.
const TOKEN_ALPHABET: &[u8] = b"ACDEFHJKLMNPRTUVWXY3479";
const TOKEN_LENGTH: usize = 7;

/// One qualification attempt, pass or fail.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReadinessRecord {
    pub model_id: String,
    pub weights_file: String,
    pub weights_bytes: u64,
    /// The registry's pinned hash, when it pins one.
    pub weights_sha256: Option<String>,
    pub projector_file: String,
    pub projector_bytes: u64,
    /// Measured when the probe ran, not taken from any manifest.
    pub projector_sha256: String,
    pub probe_token: String,
    pub prompt: String,
    /// The model's answer, bounded.
    pub answer_excerpt: String,
    pub passed: bool,
    pub reason: String,
    pub elapsed_ms: u64,
    pub at: String,
    /// How the call reached the model.
    pub transport: String,
    pub actor: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ReadinessFile {
    schema_version: u32,
    records: Vec<ReadinessRecord>,
}

/// Every recorded attempt, newest last.
pub fn load(models_dir: &Path) -> Vec<ReadinessRecord> {
    std::fs::read(models_dir.join(READINESS_FILE))
        .ok()
        .and_then(|raw| serde_json::from_slice::<ReadinessFile>(&raw).ok())
        .filter(|file| file.schema_version == READINESS_SCHEMA)
        .map(|file| file.records)
        .unwrap_or_default()
}

/// Records an attempt. The latest attempt per model is what counts: a model
/// that passed once and failed since is not ready.
pub fn record(models_dir: &Path, attempt: &ReadinessRecord) -> Result<(), String> {
    let mut records = load(models_dir);
    records.retain(|r| r.model_id != attempt.model_id);
    records.push(attempt.clone());
    std::fs::create_dir_all(models_dir)
        .map_err(|error| format!("the models directory could not be created: {error}"))?;
    let path = models_dir.join(READINESS_FILE);
    let bytes = serde_json::to_vec_pretty(&ReadinessFile {
        schema_version: READINESS_SCHEMA,
        records,
    })
    .map_err(|error| format!("the readiness record could not be encoded: {error}"))?;
    let temporary = path.with_extension("json.tmp");
    std::fs::write(&temporary, bytes)
        .map_err(|error| format!("{} could not be written: {error}", path.display()))?;
    std::fs::rename(&temporary, &path)
        .map_err(|error| format!("{} could not be replaced: {error}", path.display()))
}

/// Why a model is or is not vision-ready right now.
pub fn readiness(models_dir: &Path, entry: &ModelEntry) -> Result<ReadinessRecord, String> {
    let Some(projector) = entry.projector.as_ref() else {
        return Err(format!(
            "{} has no image projector bound, so it cannot be shown an image",
            entry.id
        ));
    };
    let Some(attempt) = load(models_dir).into_iter().rev().find(|r| r.model_id == entry.id) else {
        return Err(format!(
            "{} has a projector and has never been shown an image on this machine; it is not \
             vision-ready until a probe image call passes",
            entry.id
        ));
    };
    if !attempt.passed {
        return Err(format!(
            "{}'s last image probe failed ({}): {}",
            entry.id, attempt.at, attempt.reason
        ));
    }
    let resolved = models_dir.join(projector);
    let projector_file = resolved
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default();
    let projector_bytes = std::fs::metadata(&resolved).map(|m| m.len()).ok();
    if projector_file != attempt.projector_file || projector_bytes != Some(attempt.projector_bytes) {
        return Err(format!(
            "{} passed its image probe with {} ({} bytes), and the projector now is {} ({}); \
             probe it again",
            entry.id,
            attempt.projector_file,
            attempt.projector_bytes,
            projector_file,
            projector_bytes
                .map(|b| format!("{b} bytes"))
                .unwrap_or_else(|| "missing".into())
        ));
    }
    let weights_bytes = std::fs::metadata(models_dir.join(&entry.path)).map(|m| m.len()).ok();
    if weights_bytes != Some(attempt.weights_bytes) {
        return Err(format!(
            "{}'s weights on disk are not the file that passed the image probe; probe it again",
            entry.id
        ));
    }
    Ok(attempt)
}

/// Gives a model whose current files passed a probe the vision role and image
/// modality. Nothing else grants them through this path.
pub fn apply(models_dir: &Path, entries: &mut [ModelEntry]) -> Vec<String> {
    let mut said = Vec::new();
    for entry in entries.iter_mut() {
        if let Ok(attempt) = readiness(models_dir, entry) {
            let mut changed = false;
            if !entry.roles.contains(&ModelRole::Vision) {
                entry.roles.push(ModelRole::Vision);
                changed = true;
            }
            if !entry.modalities.contains(&Modality::Image) {
                entry.modalities.push(Modality::Image);
                changed = true;
            }
            if changed {
                said.push(format!(
                    "{} is vision-ready: it read probe token {} from an image at {}",
                    entry.id, attempt.probe_token, attempt.at
                ));
            }
        }
    }
    said
}

/// A fresh probe token. Random, so no answer can be prepared for it.
pub fn probe_token() -> String {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    (0..TOKEN_LENGTH)
        .map(|_| TOKEN_ALPHABET[rng.gen_range(0..TOKEN_ALPHABET.len())] as char)
        .collect()
}

/// Whether an answer contains the token, ignoring case, spaces and punctuation
/// a model may wrap it in.
pub fn answer_contains(answer: &str, token: &str) -> bool {
    let squeeze = |s: &str| {
        s.chars()
            .filter(|c| c.is_ascii_alphanumeric())
            .map(|c| c.to_ascii_uppercase())
            .collect::<String>()
    };
    squeeze(answer).contains(&squeeze(token))
}

/// A loopback chat endpoint with a model that takes images.
#[derive(Debug, Clone)]
pub struct VisionEndpoint {
    pub base_url: String,
    pub served_model_id: String,
}

fn vision_client() -> &'static reqwest::Client {
    static CLIENT: std::sync::OnceLock<reqwest::Client> = std::sync::OnceLock::new();
    CLIENT.get_or_init(|| {
        reqwest::Client::builder() // arjun-egress-ok: loopback only; `image_call` checks the host first
            .connect_timeout(Duration::from_secs(10))
            .no_proxy()
            .build()
            .expect("the vision http client builds from constants")
    })
}

/// One non-streaming image call. Returns the answer text.
pub async fn image_call(
    endpoint: &VisionEndpoint,
    image: &Path,
    prompt: &str,
    max_tokens: u32,
    limit: Duration,
) -> Result<String, String> {
    crate::serving::probe::check_loopback(&endpoint.base_url).map_err(|outcome| {
        format!(
            "refusing to send an image off-machine: {}",
            outcome.explain(&endpoint.base_url)
        )
    })?;
    let bytes = tokio::fs::read(image)
        .await
        .map_err(|error| format!("{} could not be read: {error}", image.display()))?;
    let body = serde_json::json!({
        "model": endpoint.served_model_id,
        "messages": [{
            "role": "user",
            "content": [
                { "type": "image_url", "image_url": {
                    "url": format!("data:image/png;base64,{}",
                        base64::engine::general_purpose::STANDARD.encode(&bytes)) } },
                { "type": "text", "text": prompt },
            ],
        }],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": false,
    });
    let url = format!("{}/chat/completions", endpoint.base_url.trim_end_matches('/'));
    let response = tokio::time::timeout(limit, vision_client().post(&url).json(&body).send())
        .await
        .map_err(|_| format!("the image call to {url} did not answer within {}s", limit.as_secs()))?
        .map_err(|error| format!("the image call to {url} failed: {error}"))?;
    let status = response.status();
    let payload: serde_json::Value = tokio::time::timeout(limit, response.json())
        .await
        .map_err(|_| "the image call's answer did not arrive in time".to_string())?
        .map_err(|error| format!("the image call's answer was not JSON: {error}"))?;
    if !status.is_success() {
        return Err(format!("the image call returned {status}: {payload}"));
    }
    payload
        .pointer("/choices/0/message/content")
        .and_then(|content| content.as_str())
        .map(str::to_string)
        .ok_or_else(|| format!("the image call's answer had no message content: {payload}"))
}

/// Runs the probe against a started endpoint and builds the record.
///
/// Recording it is the caller's decision; this only measures.
#[allow(clippy::too_many_arguments)]
pub async fn qualify(
    endpoint: &VisionEndpoint,
    entry: &ModelEntry,
    models_dir: &Path,
    sidecar: &super::sidecar::Sidecar,
    work_dir: &Path,
    actor: &str,
    transport: &str,
) -> Result<ReadinessRecord, String> {
    let projector = entry
        .projector
        .as_ref()
        .map(|p| models_dir.join(p))
        .ok_or_else(|| format!("{} has no projector bound; bind one first", entry.id))?;
    let projector_bytes = std::fs::metadata(&projector)
        .map_err(|error| format!("{} could not be measured: {error}", projector.display()))?
        .len();
    let projector_sha256 = {
        let path = projector.clone();
        tokio::task::spawn_blocking(move || sha256_file(&path))
            .await
            .map_err(|error| format!("hashing the projector did not finish: {error}"))??
    };
    let weights_bytes = std::fs::metadata(models_dir.join(&entry.path))
        .map(|m| m.len())
        .map_err(|error| format!("the weights could not be measured: {error}"))?;

    let token = probe_token();
    std::fs::create_dir_all(work_dir)
        .map_err(|error| format!("the probe directory could not be created: {error}"))?;
    let image = work_dir.join(format!("vision-probe-{token}.png"));
    {
        let (sidecar, image, token) = (sidecar.clone(), image.clone(), token.clone());
        tokio::task::spawn_blocking(move || sidecar.probe_image(&image, &token))
            .await
            .map_err(|error| format!("drawing the probe did not finish: {error}"))??;
    }

    let started = Instant::now();
    let answer = image_call(endpoint, &image, PROBE_PROMPT, 64, Duration::from_secs(90)).await;
    let elapsed_ms = started.elapsed().as_millis() as u64;
    let _ = std::fs::remove_file(&image);
    let (passed, reason, excerpt) = match answer {
        Ok(answer) => {
            let excerpt: String = answer.chars().take(200).collect();
            if answer_contains(&answer, &token) {
                (true, "the answer contains the probe token".to_string(), excerpt)
            } else {
                (
                    false,
                    "the answer does not contain the probe token: the model answered without \
                     reading the image"
                        .to_string(),
                    excerpt,
                )
            }
        }
        Err(error) => (false, error, String::new()),
    };
    Ok(ReadinessRecord {
        model_id: entry.id.clone(),
        weights_file: entry
            .path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default(),
        weights_bytes,
        weights_sha256: entry.sha256.clone(),
        projector_file: projector
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default(),
        projector_bytes,
        projector_sha256,
        probe_token: token,
        prompt: PROBE_PROMPT.to_string(),
        answer_excerpt: excerpt,
        passed,
        reason,
        elapsed_ms,
        at: chrono::Utc::now().to_rfc3339(),
        transport: transport.to_string(),
        actor: actor.to_string(),
    })
}

/// Probes a registered model end to end and records the attempt.
///
/// Reserves the card through the shared scheduler, starts (or reuses) the
/// model's loopback server with its projector, runs [`qualify`], and writes the
/// record whether it passed or failed — a failed probe is evidence too, and a
/// model that passed once and fails now must stop being treated as ready.
#[allow(clippy::too_many_arguments)]
pub async fn probe_registered(
    registry: &crate::registry::ModelRegistry,
    servers: &crate::serving::ModelServers,
    scheduler: &crate::subagents::scheduling::ModelScheduler,
    model_id: &str,
    sidecar: &super::sidecar::Sidecar,
    work_dir: &Path,
    actor: &str,
) -> Result<ReadinessRecord, String> {
    let entry = registry
        .find(model_id)
        .cloned()
        .ok_or_else(|| format!("{model_id} is not in the model registry"))?;
    if entry.projector.is_none() {
        return Err(format!(
            "{model_id} has no image projector. Bind a verified projector first; it takes effect \
             at the next start."
        ));
    }
    let _lease = scheduler
        .reserve(model_id, Duration::from_secs(60))
        .await
        .map_err(|refusal| refusal.explain())?;
    let plan = crate::serving::admission::admit(servers, &entry, registry.models_dir())
        .await
        .map_err(|error| error.to_string())?
        .plan;
    let endpoint = servers
        .endpoint_for(&entry, registry.models_dir(), &plan)
        .await
        .map_err(|error| error.to_string())?;
    let endpoint = VisionEndpoint {
        base_url: endpoint.base_url,
        served_model_id: endpoint.served_model_id,
    };
    let attempt = qualify(
        &endpoint,
        &entry,
        registry.models_dir(),
        sidecar,
        work_dir,
        actor,
        "local llama-server (loopback)",
    )
    .await?;
    record(registry.models_dir(), &attempt)?;
    Ok(attempt)
}

/// A vision-ready model and the record that makes it so.
#[derive(Debug, Clone)]
pub struct ReadyModel {
    pub model_id: String,
    pub record: ReadinessRecord,
}

/// A model endpoint opened for interpretation, holding the card.
pub struct VisionSession {
    pub endpoint: VisionEndpoint,
    pub residency: String,
    pub _lease: Option<crate::subagents::scheduling::ModelLease>,
}

/// Where interpretation comes from on this deployment.
#[async_trait]
pub trait VisionService: Send + Sync {
    /// The model interpretation would use, or why there is none.
    fn ready_model(&self) -> Result<ReadyModel, String>;
    async fn open(&self, model_id: &str, wait: Duration) -> Result<VisionSession, String>;
    fn transport(&self) -> &'static str;
}

/// Models the plan names for interpretation, preferred in this order.
const PREFERRED: &[&str] = &["gemma-4-e4b", "gemma4-e4b", "qwen3.5-9b", "qwen3-5-9b", "qwen35-9b"];

/// The production service: the registry's vision-ready models, served locally.
pub struct ServedVision {
    pub registry: Arc<crate::registry::ModelRegistry>,
    pub servers: Arc<crate::serving::ModelServers>,
    pub scheduler: Arc<crate::subagents::scheduling::ModelScheduler>,
}

impl ServedVision {
    /// Ready models, preferred first. OCR models are not interpreters.
    pub fn candidates(&self) -> Vec<(ModelEntry, Result<ReadinessRecord, String>)> {
        let models_dir = self.registry.models_dir();
        let mut found: Vec<(ModelEntry, Result<ReadinessRecord, String>)> = self
            .registry
            .all()
            .iter()
            .filter(|entry| entry.enabled && !entry.roles.contains(&ModelRole::DocumentOcr))
            .filter(|entry| entry.projector.is_some() || entry.roles.contains(&ModelRole::Vision))
            .map(|entry| (entry.clone(), readiness(models_dir, entry)))
            .collect();
        let rank = |entry: &ModelEntry| {
            let id = entry.id.to_ascii_lowercase();
            PREFERRED
                .iter()
                .position(|token| id.contains(token))
                .unwrap_or(PREFERRED.len())
        };
        found.sort_by_key(|(entry, ready)| (ready.is_err(), rank(entry), entry.weights_bytes));
        found
    }
}

#[async_trait]
impl VisionService for ServedVision {
    fn ready_model(&self) -> Result<ReadyModel, String> {
        let candidates = self.candidates();
        if let Some((entry, Ok(record))) = candidates.iter().find(|(_, ready)| ready.is_ok()) {
            return Ok(ReadyModel {
                model_id: entry.id.clone(),
                record: record.clone(),
            });
        }
        if candidates.is_empty() {
            return Err(
                "no model on this machine has an image projector, so nothing can interpret an \
                 image; bind a verified projector to Gemma 4 E4B or Qwen3.5 9B and probe it"
                    .to_string(),
            );
        }
        Err(format!(
            "no model is vision-ready: {}",
            candidates
                .iter()
                .filter_map(|(_, ready)| ready.as_ref().err().cloned())
                .collect::<Vec<_>>()
                .join("; ")
        ))
    }

    async fn open(&self, model_id: &str, wait: Duration) -> Result<VisionSession, String> {
        let entry = self
            .registry
            .find(model_id)
            .cloned()
            .ok_or_else(|| format!("{model_id} is not in the model registry"))?;
        let lease = self
            .scheduler
            .reserve(model_id, wait)
            .await
            .map_err(|refusal| refusal.explain())?;
        let residency = lease.describe();
        let plan = crate::serving::admission::admit(&self.servers, &entry, self.registry.models_dir())
            .await
            .map_err(|error| error.to_string())?
            .plan;
        let endpoint = self
            .servers
            .endpoint_for(&entry, self.registry.models_dir(), &plan)
            .await
            .map_err(|error| error.to_string())?;
        Ok(VisionSession {
            endpoint: VisionEndpoint {
                base_url: endpoint.base_url,
                served_model_id: endpoint.served_model_id,
            },
            residency,
            _lease: Some(lease),
        })
    }

    fn transport(&self) -> &'static str {
        "local llama-server (loopback)"
    }
}

/// A deployment with no vision model.
pub struct NoVision(pub String);

#[async_trait]
impl VisionService for NoVision {
    fn ready_model(&self) -> Result<ReadyModel, String> {
        Err(self.0.clone())
    }

    async fn open(&self, _: &str, _: Duration) -> Result<VisionSession, String> {
        Err(self.0.clone())
    }

    fn transport(&self) -> &'static str {
        "none"
    }
}

/// The prompt an interpretation is asked under. It asks for what is visible
/// and for unreadable things to be said to be unreadable; it cannot make the
/// model comply, which is why the answer is a proposal.
pub fn interpretation_prompt(question: &str, fields: &[String]) -> String {
    let mut prompt = format!(
        "You are looking at a crop of an engineering document. {question}\n\
         Describe only what is visible. If a label, tag or value cannot be read, write \
         UNREADABLE for it rather than guessing. Do not infer connections that are not drawn."
    );
    if !fields.is_empty() {
        prompt.push_str(&format!(
            "\nReport these if they are visible: {}.",
            fields.join(", ")
        ));
    }
    prompt
}

/// Turns an interpretation into a proposal region over the crop it was asked of.
pub fn proposal_region(
    crop: &CropRecord,
    ready: &ReadyModel,
    prompt: &str,
    answer: &str,
) -> EvidenceRegion {
    let discriminator = hex::encode(Sha256::digest(
        format!("{}|{}|{}|{}", ready.model_id, ready.record.at, prompt, crop.image_sha256).as_bytes(),
    ));
    let text: String = answer.trim().chars().take(4000).collect();
    let mut notes = vec![
        "vision-model inference: a proposal about what the image shows, not a transcription and \
         not verified against the page"
            .to_string(),
    ];
    if text.to_ascii_uppercase().contains("UNREADABLE") {
        notes.push("the model reported part of this crop as unreadable".to_string());
    }
    EvidenceRegion {
        region_id: region_id(
            &crop.document_sha256,
            crop.page,
            Method::VisionInference,
            "interpretation",
            &crop.bbox,
            &discriminator,
            0,
        ),
        document_sha256: crop.document_sha256.clone(),
        page: crop.page,
        bbox: crop.bbox,
        coord_space: crop.coord_space,
        label: "interpretation".to_string(),
        method: Method::VisionInference,
        status: if text.is_empty() {
            RegionStatus::Unreadable
        } else {
            RegionStatus::Read
        },
        text,
        cells: Vec::new(),
        notes,
        crop_id: Some(crop.crop_id.clone()),
        image_sha256: Some(crop.image_sha256.clone()),
        extractor: Extractor {
            name: ready.model_id.clone(),
            version: format!("vision-ready since {}", ready.record.at),
            model_id: Some(ready.model_id.clone()),
            weights_sha256: ready.record.weights_sha256.clone(),
            projector: Some(format!(
                "{} (sha256 {})",
                ready.record.projector_file,
                &ready.record.projector_sha256[..16.min(ready.record.projector_sha256.len())]
            )),
            profile: Some(format!("prompt sha256 {}", &hex::encode(Sha256::digest(prompt.as_bytes()))[..12])),
        },
        cache_key: None,
    }
}

/// The directory probe images are drawn in.
pub fn probe_dir(documents_root: &Path) -> PathBuf {
    documents_root.join("vision-probe")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(dir: &Path) -> ModelEntry {
        std::fs::write(dir.join("m.gguf"), b"weights").unwrap();
        std::fs::write(dir.join("p.gguf"), b"projector").unwrap();
        let mut entry: ModelEntry = serde_json::from_value(serde_json::json!({
            "id": "gemma-4-e4b-q4", "name": "g", "version": "1", "license": "x", "sha256": null,
            "runtime": "llamaCpp", "roles": ["reasoning"], "quantization": "Q4_K_M",
            "parametersB": 4.0, "contextLength": 8192, "weightsBytes": 7, "path": "m.gguf"
        }))
        .unwrap();
        entry.projector = Some(PathBuf::from("p.gguf"));
        entry
    }

    fn attempt(passed: bool) -> ReadinessRecord {
        ReadinessRecord {
            model_id: "gemma-4-e4b-q4".into(),
            weights_file: "m.gguf".into(),
            weights_bytes: 7,
            weights_sha256: None,
            projector_file: "p.gguf".into(),
            projector_bytes: 9,
            projector_sha256: "a".repeat(64),
            probe_token: "K7XWM4P".into(),
            prompt: PROBE_PROMPT.into(),
            answer_excerpt: if passed { "K7XWM4P".into() } else { "A tidy office.".into() },
            passed,
            reason: String::new(),
            elapsed_ms: 1,
            at: "2026-09-24T00:00:00Z".into(),
            transport: "test".into(),
            actor: "admin".into(),
        }
    }

    #[test]
    fn a_projector_alone_is_not_vision_ready() {
        let dir = tempfile::tempdir().unwrap();
        let e = entry(dir.path());
        let why = readiness(dir.path(), &e).unwrap_err();
        assert!(why.contains("never been shown an image"), "{why}");
        let mut entries = vec![e];
        apply(dir.path(), &mut entries);
        assert!(!entries[0].roles.contains(&ModelRole::Vision));
    }

    #[test]
    fn only_a_passing_probe_on_the_current_files_grants_vision() {
        let dir = tempfile::tempdir().unwrap();
        let e = entry(dir.path());
        record(dir.path(), &attempt(false)).unwrap();
        assert!(readiness(dir.path(), &e).unwrap_err().contains("failed"));

        record(dir.path(), &attempt(true)).unwrap();
        let mut entries = vec![e.clone()];
        apply(dir.path(), &mut entries);
        assert!(entries[0].roles.contains(&ModelRole::Vision));
        assert!(entries[0].modalities.contains(&Modality::Image));

        // The projector changed after the probe.
        std::fs::write(dir.path().join("p.gguf"), b"another projector").unwrap();
        assert!(readiness(dir.path(), &e).unwrap_err().contains("probe it again"));
    }

    #[test]
    fn the_token_check_ignores_wrapping_and_rejects_a_guess() {
        assert!(answer_contains("The text reads: \"k7x wm4p\".", "K7XWM4P"));
        assert!(!answer_contains("The image shows some letters.", "K7XWM4P"));
        let token = probe_token();
        assert_eq!(token.len(), TOKEN_LENGTH);
        assert_ne!(token, probe_token(), "two probes drew the same token");
    }
}
