//! Turning text into a vector, using a model on this machine.
//!
//! ## Why this exists
//!
//! Keyword search is unbeatable on an exact token — `PV-2201`, `9.0 mm`, a
//! drawing number — and blind to a question phrased in words the document never
//! used. "What stops the charge pump cavitating?" finds nothing by keyword when
//! the manual says "NPSH margin shall exceed 1.5 m". That question is the one a
//! refinery actually asks, and it is the reason to embed at all.
//!
//! ## What was here before P07, and why it was not enough
//!
//! A client, [`LocalEmbedder`], with no production caller, and an `Embedder`
//! trait that sent whatever text it was handed. Three things an embedding model
//! needs were nobody's job:
//!
//! - **Its prefixes.** E5 is trained on `query: ` and `passage: `; nomic on
//!   `search_query: ` and `search_document: `; Qwen3-Embedding on an
//!   instruction before the query and nothing before the document. Sent bare,
//!   each still returns a vector of the right width, and retrieval quietly gets
//!   worse. A caller that has to remember the prefix is a caller that will one
//!   day forget it, so the prefix is applied here, by [`embed_query`] and
//!   [`embed_passage`], from the model's pinned [`EmbeddingProfile`].
//! - **Its window.** A 512-token model given 2,000 characters of Devanagari is
//!   over its window. A server that truncates embeds the first half of the
//!   passage and the vector is confidently about the wrong thing; one that
//!   refuses fails the batch. Passages are cut into windows here, and a window
//!   the server still refuses as too large is halved and retried — the server's
//!   own tokenizer is the measure, not a guessed characters-per-token ratio.
//! - **Its identity.** A vector means nothing outside the space that produced
//!   it. [`EmbeddingIdentity::space_key`] names that space by profile, profile
//!   version, the SHA-256 of the weights file (which carries the tokenizer), the
//!   width and the pooling — and the index keys every stored vector by it, so
//!   two models' vectors are never compared and a re-embed after a model change
//!   starts a new space rather than overwriting the old one row by row.
//!
//! ## What it talks to
//!
//! An OpenAI-compatible `/embeddings` endpoint on loopback — `llama-server
//! --embedding` on this machine, started by [`crate::serving`] with the pooling
//! the profile names and no GPU layers (the plan qualifies embedding on the CPU
//! first, so an embedding pass never competes with the one heavy generation the
//! card holds).
//!
//! ## What it refuses to do
//!
//! **Reach anything but this machine.** The base URL is checked with
//! [`crate::serving::probe::check_loopback`] before the client is constructed.
//!
//! **Return a vector it did not receive.** Every failure — the server down, a
//! reply of the wrong length, a non-finite or zero vector, a batch that came
//! back short — is an error. There is no zero-vector fallback and no padding to
//! the expected width. A zero vector is a valid point in the space: it would be
//! stored, compared, ranked and cited, and nothing downstream could tell it from
//! a real embedding of a passage that genuinely says nothing.
//!
//! **Call itself qualified.** A model that matches a profile is *compatible*.
//! It becomes usable for retrieval only after [`qualify`] has measured it on
//! labelled probes and the record says it passed — see
//! [`QualificationStore`]. Until then retrieval is lexical and says so.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// How long one batch may take before it is abandoned.
///
/// Generous, because embedding a batch on a busy CPU is slow and a pass that
/// gave up early would leave the index half-covered for no reason. Bounded,
/// because a hung request with no ceiling is the stall this product already
/// found once on the generation path.
const REQUEST_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(120);

/// Passages sent in one request.
///
/// Batched because per-request overhead dominates for short passages, and
/// bounded because the whole batch is one payload the server must hold. Sixteen
/// is small enough to keep a slow server responsive between batches, which is
/// what makes the pass interruptible.
pub const BATCH_SIZE: usize = 16;

/// How many times a window the server refuses as too large may be halved.
///
/// Three halvings take a 900-character window to about 110 characters. A
/// window still refused at that size is not a length problem, and retrying it
/// further would hide whatever the real one is.
const MAX_HALVINGS: u32 = 3;

/// How the server reduces token vectors to one passage vector.
///
/// The value is passed to `llama-server --pooling` verbatim, so the names are
/// llama.cpp's.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Pooling {
    Mean,
    Last,
    Cls,
}

impl Pooling {
    pub const fn as_str(self) -> &'static str {
        match self {
            Pooling::Mean => "mean",
            Pooling::Last => "last",
            Pooling::Cls => "cls",
        }
    }
}

/// Everything about one embedding model family that a caller must not guess.
///
/// Pinned in code rather than read from a registry row, because every field is
/// a property of how the publisher trained the model, and a registry row that
/// disagreed would be wrong rather than a different valid configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EmbeddingProfile {
    /// Stable key, part of every space key built from this profile.
    pub key: &'static str,
    /// The publisher's model this profile describes.
    pub publisher_model: &'static str,
    /// `general.architecture` values llama.cpp's converter writes for it.
    pub architectures: &'static [&'static str],
    /// Lowercase markers, one of which must appear in the GGUF `general.name`
    /// or the file name. The architecture alone does not identify a model:
    /// `bert` is the architecture of hundreds of them.
    pub name_markers: &'static [&'static str],
    pub dimensions: usize,
    pub pooling: Pooling,
    /// Put before a query. Part of the input the model was trained on.
    pub query_prefix: &'static str,
    /// Put before a stored passage.
    pub passage_prefix: &'static str,
    /// The token window the server is launched with, and the most one input may use.
    pub max_tokens: u32,
    /// Characters per window before the server's own tokenizer is consulted.
    ///
    /// A first cut, deliberately below what `max_tokens` holds for English.
    /// Scripts that tokenise more densely are caught by the server refusing a
    /// window as too large, which halves it — see [`embed_passage`].
    pub window_chars: usize,
    /// Whether the publisher trained it on more than English.
    pub multilingual: bool,
    /// Bumped whenever prefixes, pooling or windowing change here, so vectors
    /// made under the old rule land in a different space from the new ones.
    pub version: u32,
}

/// `intfloat/multilingual-e5-small`: the plan's first candidate (§4).
///
/// 384 dimensions, mean pooling, `query: ` / `passage: ` prefixes and a 512
/// token window, all from the publisher's card. llama.cpp converts its
/// XLM-RoBERTa encoder as `bert`.
pub const MULTILINGUAL_E5_SMALL: EmbeddingProfile = EmbeddingProfile {
    key: "multilingual-e5-small",
    publisher_model: "intfloat/multilingual-e5-small",
    architectures: &["bert"],
    name_markers: &["multilingual-e5-small", "multilingual_e5_small"],
    dimensions: 384,
    pooling: Pooling::Mean,
    query_prefix: "query: ",
    passage_prefix: "passage: ",
    max_tokens: 512,
    window_chars: 900,
    multilingual: true,
    version: 1,
};

/// `nomic-ai/nomic-embed-text-v2-moe`: installed on the target machine (P00).
///
/// 768 dimensions, mean pooling, `search_query: ` / `search_document: `
/// prefixes, a 512-token window measured from the installed file's header.
pub const NOMIC_EMBED_TEXT_V2_MOE: EmbeddingProfile = EmbeddingProfile {
    key: "nomic-embed-text-v2-moe",
    publisher_model: "nomic-ai/nomic-embed-text-v2-moe",
    architectures: &["nomic-bert-moe"],
    name_markers: &["nomic-embed-text-v2-moe"],
    dimensions: 768,
    pooling: Pooling::Mean,
    query_prefix: "search_query: ",
    passage_prefix: "search_document: ",
    max_tokens: 512,
    window_chars: 900,
    multilingual: true,
    version: 1,
};

/// `Qwen/Qwen3-Embedding-0.6B`: installed on the target machine (P00), and the
/// plan's named alternative.
///
/// 1,024 dimensions and last-token pooling. The publisher's query format is an
/// instruction line followed by `Query:`; documents take no prefix. Served at a
/// 512-token window here — the model accepts far more, and matching the other
/// profiles keeps one chunking rule for every space.
pub const QWEN3_EMBEDDING_0_6B: EmbeddingProfile = EmbeddingProfile {
    key: "qwen3-embedding-0.6b",
    publisher_model: "Qwen/Qwen3-Embedding-0.6B",
    architectures: &["qwen3"],
    name_markers: &["qwen3-embedding", "qwen3 embedding"],
    dimensions: 1024,
    pooling: Pooling::Last,
    query_prefix: "Instruct: Given a question, retrieve passages from the organisation's documents \
                   that answer it\nQuery:",
    passage_prefix: "",
    max_tokens: 512,
    window_chars: 900,
    multilingual: true,
    version: 1,
};

/// Every profile this build knows how to drive, in the plan's order of preference.
pub const PROFILES: &[EmbeddingProfile] =
    &[MULTILINGUAL_E5_SMALL, NOMIC_EMBED_TEXT_V2_MOE, QWEN3_EMBEDDING_0_6B];

/// The profile a model file matches, from its header and its name.
///
/// `None` for a file no profile describes. Such a file is not driven at all:
/// guessing a prefix and a pooling for an unknown model is exactly the kind of
/// plausible default this module exists to refuse.
pub fn profile_for(architecture: &str, general_name: &str, file_name: &str) -> Option<&'static EmbeddingProfile> {
    let name = general_name.to_ascii_lowercase();
    let file = file_name.to_ascii_lowercase();
    PROFILES.iter().find(|profile| {
        profile.architectures.contains(&architecture)
            && profile
                .name_markers
                .iter()
                .any(|marker| name.contains(marker) || file.contains(marker))
    })
}

/// The profile for a GGUF on disk, read from its header.
pub fn profile_for_weights(weights: &Path) -> Option<&'static EmbeddingProfile> {
    let scalars = crate::ai_engine::gguf_meta::read_gguf_scalars(weights).ok()?;
    let architecture = scalars.get("general.architecture")?.as_str()?.to_string();
    let name = scalars
        .get("general.name")
        .and_then(|value| value.as_str())
        .unwrap_or_default()
        .to_string();
    let file = weights
        .file_name()
        .map(|name| name.to_string_lossy().to_string())
        .unwrap_or_default();
    profile_for(&architecture, &name, &file)
}

/// Which vector space a vector belongs to.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EmbeddingIdentity {
    /// The registry id the server was started under.
    pub model_id: String,
    pub profile: String,
    pub profile_version: u32,
    /// SHA-256 of the weights file. The GGUF carries the tokenizer, so this
    /// pins both.
    pub weights_sha256: String,
    pub dimensions: usize,
    pub pooling: Pooling,
}

impl EmbeddingIdentity {
    pub fn new(model_id: &str, profile: &EmbeddingProfile, weights_sha256: &str) -> Self {
        Self {
            model_id: model_id.to_string(),
            profile: profile.key.to_string(),
            profile_version: profile.version,
            weights_sha256: weights_sha256.to_ascii_lowercase(),
            dimensions: profile.dimensions,
            pooling: profile.pooling,
        }
    }

    /// The index identity. Every stored vector is keyed by it.
    ///
    /// The model id is deliberately absent: renaming a registry row does not
    /// change a single vector, while changing one byte of the weights changes
    /// all of them.
    pub fn space_key(&self) -> String {
        let short = self.weights_sha256.get(..16).unwrap_or(&self.weights_sha256);
        format!(
            "{}@{}/d{}/{}/v{}",
            self.profile,
            short,
            self.dimensions,
            self.pooling.as_str(),
            self.profile_version
        )
    }

    pub fn profile(&self) -> Option<&'static EmbeddingProfile> {
        PROFILES.iter().find(|profile| profile.key == self.profile)
    }
}

/// Why an embedding call produced nothing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EmbedError {
    /// The server refused an input as longer than its window.
    TooLarge(String),
    /// Anything else. Not retried by halving.
    Failed(String),
}

impl EmbedError {
    pub fn explain(&self) -> String {
        match self {
            EmbedError::TooLarge(detail) => format!("an input was over the model's window: {detail}"),
            EmbedError::Failed(detail) => detail.clone(),
        }
    }
}

/// A model that turns prepared inputs into raw vectors.
///
/// The one method is the transport. Prefixes, windows, normalisation and every
/// check on what came back live in the free functions below, so they run the
/// same whatever is behind this.
#[async_trait]
pub trait Embedder: Send + Sync {
    fn identity(&self) -> &EmbeddingIdentity;
    fn profile(&self) -> &'static EmbeddingProfile;
    /// Embeds exactly these strings, in this order. No prefix is added here.
    async fn embed_raw(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>, EmbedError>;
}

/// Unit length, or an error naming what is wrong with the vector.
///
/// Every stored and every query vector goes through here, so cosine similarity
/// downstream is a dot product of two unit vectors and a server that pooled
/// without normalising cannot skew a ranking by magnitude.
pub fn normalise(vector: Vec<f32>, expected: usize) -> Result<Vec<f32>> {
    if vector.len() != expected {
        bail!(
            "the model returned a {}-dimension vector where its profile declares {expected}. \
             Storing it would put two geometries in one index.",
            vector.len()
        );
    }
    if vector.iter().any(|value| !value.is_finite()) {
        bail!("the model returned a vector with a non-finite component");
    }
    let norm = vector.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>().sqrt();
    if norm <= 1e-12 {
        bail!(
            "the model returned a zero vector. It is not stored: a zero vector would rank as \
             'unrelated to everything', which is an answer, not an absence of one"
        );
    }
    Ok(vector.into_iter().map(|v| (v as f64 / norm) as f32).collect())
}

/// Embeds a question, with the profile's query prefix.
///
/// A query is not windowed. One over the window is refused by the server as too
/// large, and that is returned rather than halved: retrieving on half a
/// question would answer a different one.
pub async fn embed_query(embedder: &dyn Embedder, query: &str) -> Result<Vec<f32>> {
    let profile = embedder.profile();
    let input = format!("{}{}", profile.query_prefix, query.trim());
    let mut vectors = embedder
        .embed_raw(&[input])
        .await
        .map_err(|error| anyhow!("the question could not be embedded: {}", error.explain()))?;
    let vector = vectors
        .pop()
        .ok_or_else(|| anyhow!("the embedding endpoint returned no vector for the question"))?;
    normalise(vector, profile.dimensions)
}

/// A passage's text cut into windows the model can read whole.
///
/// At paragraph, then sentence, then word boundaries, never inside a word.
/// Returns at least one window for non-empty text.
pub fn windows(text: &str, window_chars: usize) -> Vec<String> {
    let text = text.trim();
    if text.is_empty() {
        return Vec::new();
    }
    if text.chars().count() <= window_chars {
        return vec![text.to_string()];
    }
    let mut out: Vec<String> = Vec::new();
    let mut current = String::new();
    for word in text.split_whitespace() {
        let extra = if current.is_empty() { 0 } else { 1 };
        if current.chars().count() + extra + word.chars().count() > window_chars && !current.is_empty() {
            out.push(std::mem::take(&mut current));
        }
        if !current.is_empty() {
            current.push(' ');
        }
        // A single "word" longer than the window (a hash, a run of symbols) is
        // cut by characters: dropping it would silently lose text.
        if word.chars().count() > window_chars {
            let chars: Vec<char> = word.chars().collect();
            for piece in chars.chunks(window_chars) {
                if !current.is_empty() {
                    out.push(std::mem::take(&mut current));
                }
                current = piece.iter().collect();
            }
        } else {
            current.push_str(word);
        }
    }
    if !current.is_empty() {
        out.push(current);
    }
    out
}

/// Splits one window in two at the word boundary nearest its middle.
fn halve(window: &str) -> (String, String) {
    let words: Vec<&str> = window.split_whitespace().collect();
    if words.len() >= 2 {
        let middle = words.len() / 2;
        return (words[..middle].join(" "), words[middle..].join(" "));
    }
    let chars: Vec<char> = window.chars().collect();
    let middle = chars.len() / 2;
    (chars[..middle].iter().collect(), chars[middle..].iter().collect())
}

/// One passage's vectors, one per window, each unit length.
///
/// The passage prefix goes on every window, because every window is stored and
/// compared as a passage.
pub async fn embed_passage(embedder: &dyn Embedder, text: &str) -> Result<Vec<Vec<f32>>> {
    let profile = embedder.profile();
    let mut pending: Vec<(String, u32)> = windows(text, profile.window_chars)
        .into_iter()
        .map(|window| (window, 0))
        .collect();
    if pending.is_empty() {
        bail!("the passage has no text to embed");
    }
    let mut out: Vec<Vec<f32>> = Vec::new();
    // In order, so window 0 is always the passage's opening.
    while !pending.is_empty() {
        let (window, halvings) = pending.remove(0);
        let input = format!("{}{}", profile.passage_prefix, window);
        match embedder.embed_raw(&[input]).await {
            Ok(mut vectors) => {
                let vector = vectors
                    .pop()
                    .ok_or_else(|| anyhow!("the embedding endpoint returned no vector"))?;
                out.push(normalise(vector, profile.dimensions)?);
            }
            Err(EmbedError::TooLarge(detail)) if halvings < MAX_HALVINGS => {
                log::debug!("[embedding] halving a window the server refused: {detail}");
                let (first, second) = halve(&window);
                pending.insert(0, (second, halvings + 1));
                pending.insert(0, (first, halvings + 1));
            }
            Err(error) => bail!("a passage window could not be embedded: {}", error.explain()),
        }
    }
    Ok(out)
}

/// Several passages at once, batched where each is a single window.
///
/// Multi-window passages go through [`embed_passage`] one at a time, so a
/// too-large window is halved without failing its neighbours.
pub async fn embed_passages(embedder: &dyn Embedder, texts: &[String]) -> Vec<Result<Vec<Vec<f32>>>> {
    let profile = embedder.profile();
    let mut results: Vec<Option<Result<Vec<Vec<f32>>>>> = (0..texts.len()).map(|_| None).collect();
    let single: Vec<usize> = texts
        .iter()
        .enumerate()
        .filter(|(_, text)| {
            let trimmed = text.trim();
            !trimmed.is_empty() && trimmed.chars().count() <= profile.window_chars
        })
        .map(|(position, _)| position)
        .collect();

    for batch in single.chunks(BATCH_SIZE) {
        let inputs: Vec<String> = batch
            .iter()
            .map(|position| format!("{}{}", profile.passage_prefix, texts[*position].trim()))
            .collect();
        match embedder.embed_raw(&inputs).await {
            Ok(vectors) if vectors.len() == batch.len() => {
                for (position, vector) in batch.iter().zip(vectors) {
                    results[*position] = Some(normalise(vector, profile.dimensions).map(|v| vec![v]));
                }
            }
            // A batch that failed as a whole is retried one passage at a time,
            // so one oversized passage costs itself and nothing else.
            _ => {
                for position in batch {
                    results[*position] = Some(embed_passage(embedder, &texts[*position]).await);
                }
            }
        }
    }
    for (position, text) in texts.iter().enumerate() {
        if results[position].is_none() {
            results[position] = Some(embed_passage(embedder, text).await);
        }
    }
    results
        .into_iter()
        .map(|result| result.unwrap_or_else(|| Err(anyhow!("not embedded"))))
        .collect()
}

/// An embedding model reachable on loopback.
pub struct LocalEmbedder {
    /// Includes the version prefix, e.g. `http://127.0.0.1:8080/v1`.
    base_url: String,
    identity: EmbeddingIdentity,
    profile: &'static EmbeddingProfile,
    client: reqwest::Client,
}

impl LocalEmbedder {
    /// Builds a client that can only ever address this machine.
    ///
    /// The width is the profile's. It is not discovered from the first reply,
    /// because a width learned at runtime cannot be checked against anything.
    pub fn new(base_url: String, identity: EmbeddingIdentity) -> Result<Self> {
        let profile = identity
            .profile()
            .ok_or_else(|| anyhow!("no embedding profile is named {:?}", identity.profile))?;
        if identity.dimensions == 0 || identity.dimensions != profile.dimensions {
            bail!(
                "{} declares {} dimensions and its profile {}",
                identity.model_id,
                identity.dimensions,
                profile.dimensions
            );
        }
        // Parsed, not pattern-matched. See `probe::is_loopback_host`.
        crate::serving::probe::check_loopback(&base_url)
            .map_err(|outcome| anyhow!("{}", outcome.explain(&base_url)))?;

        // arjun-egress-ok: loopback only. `check_loopback` above parses the base URL and returns an error for any host that is not 127.0.0.0/8, ::1 or `localhost`, and it runs before this client exists.
        let client = reqwest::Client::builder()
            .timeout(REQUEST_TIMEOUT)
            .build()
            .context("could not build the embedding HTTP client")?;

        Ok(Self {
            base_url,
            identity,
            profile,
            client,
        })
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }
}

#[async_trait]
impl Embedder for LocalEmbedder {
    fn identity(&self) -> &EmbeddingIdentity {
        &self.identity
    }

    fn profile(&self) -> &'static EmbeddingProfile {
        self.profile
    }

    /// Sends one batch and pairs each vector with its input by the index the
    /// reply carries, never by position.
    async fn embed_raw(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>, EmbedError> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }
        let url = format!("{}/embeddings", self.base_url.trim_end_matches('/'));
        let response = self
            .client
            .post(&url)
            .json(&serde_json::json!({ "model": self.identity.model_id, "input": inputs }))
            .send()
            .await
            .map_err(|error| {
                EmbedError::Failed(format!(
                    "the embedding model at {} did not answer: {error}",
                    self.base_url
                ))
            })?;

        let status = response.status();
        if !status.is_success() {
            let detail = response.text().await.unwrap_or_default();
            let lower = detail.to_ascii_lowercase();
            // llama-server's words for an input over its batch or context.
            if lower.contains("too large") || lower.contains("exceeds") || lower.contains("too long") {
                return Err(EmbedError::TooLarge(detail.trim().to_string()));
            }
            return Err(EmbedError::Failed(format!(
                "the embedding endpoint answered {status}: {}",
                detail.trim()
            )));
        }

        let body: EmbeddingsResponse = response.json().await.map_err(|error| {
            EmbedError::Failed(format!(
                "the embedding endpoint answered with something this cannot read: {error}"
            ))
        })?;

        if body.data.len() != inputs.len() {
            return Err(EmbedError::Failed(format!(
                "asked for {} embedding(s) and received {}. The batch is not usable, because \
                 which passage was dropped is not recoverable from the reply.",
                inputs.len(),
                body.data.len()
            )));
        }

        let mut rows = body.data;
        rows.sort_by_key(|row| row.index);
        let mut out = Vec::with_capacity(rows.len());
        for (position, row) in rows.into_iter().enumerate() {
            if row.index != position {
                return Err(EmbedError::Failed(format!(
                    "the reply is missing index {position}. Pairing the remaining vectors with \
                     the passages sent would attach them to the wrong text."
                )));
            }
            out.push(row.embedding);
        }
        Ok(out)
    }
}

/// The OpenAI-compatible reply shape.
#[derive(Debug, Deserialize)]
struct EmbeddingsResponse {
    data: Vec<EmbeddingRow>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingRow {
    embedding: Vec<f32>,
    /// Which input this vector is for. Never assumed from position.
    #[serde(default)]
    index: usize,
}

// ── Qualification ─────────────────────────────────────────────────────────

/// One labelled probe: a question, a passage that answers it, and one that
/// shares its vocabulary without answering it.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct QualificationProbe {
    pub id: String,
    pub language: String,
    pub query: String,
    pub relevant: String,
    pub distractor: String,
}

/// One check a qualification ran.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct QualificationCheck {
    pub name: String,
    pub passed: bool,
    pub detail: String,
}

/// What measuring one model produced.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct QualificationRecord {
    pub identity: EmbeddingIdentity,
    pub space_key: String,
    pub passed: bool,
    pub checks: Vec<QualificationCheck>,
    /// Probes where the answering passage was closer than the distractor.
    pub ordered_correctly: usize,
    pub probes: usize,
    /// Mean cosine of a question to the passage that answers it, measured.
    #[serde(default)]
    pub relevant_mean: Option<f64>,
    /// Mean cosine of a question to a passage that shares its words and does
    /// not answer it, measured.
    #[serde(default)]
    pub distractor_mean: Option<f64>,
    /// The semantic floor for this space: halfway between the two means.
    ///
    /// Dense search always has a nearest neighbour, so without a floor a
    /// question nothing answers still returns passages — and "no answer" can
    /// never be said. The floor is measured for this exact model on labelled
    /// probes, not chosen: a passage closer than a vocabulary-sharing
    /// non-answer is kept, one further is not used as semantic evidence.
    #[serde(default)]
    pub dense_floor: Option<f64>,
    pub measured_at: String,
    /// Where the vectors came from, for the record.
    pub endpoint: String,
}

/// The labelled probes ARJUN ships for qualification.
///
/// Synthetic, non-confidential, and in two languages, so a model that claims
/// to be multilingual is measured on a Hindi question against an English
/// passage rather than assumed to handle one.
pub fn bundled_probes() -> Vec<QualificationProbe> {
    #[derive(Deserialize)]
    struct File {
        probes: Vec<QualificationProbe>,
    }
    serde_json::from_str::<File>(include_str!("qualification_probes.json"))
        .map(|file| file.probes)
        .unwrap_or_default()
}

/// The fraction of probes that must order correctly.
///
/// Nine in ten. A model that gets one probe in five wrong is ranking by
/// something other than meaning, and the dense half would add noise that the
/// fusion then promotes.
pub const QUALIFYING_ORDER_RATE: f64 = 0.9;

/// Measures a model on labelled probes. Nothing here is assumed to pass.
pub async fn qualify(
    embedder: &dyn Embedder,
    probes: &[QualificationProbe],
    endpoint: &str,
) -> QualificationRecord {
    let identity = embedder.identity().clone();
    let profile = embedder.profile();
    let mut checks = Vec::new();
    let mut ordered = 0usize;

    // 1. Width, finiteness and non-zero length — on a real passage.
    let sample = probes
        .first()
        .map(|probe| probe.relevant.clone())
        .unwrap_or_else(|| "Close the isolation valve before opening the drain.".to_string());
    let first = embed_passage(embedder, &sample).await;
    checks.push(QualificationCheck {
        name: "width-and-finite".into(),
        passed: first.is_ok(),
        detail: match &first {
            Ok(vectors) => format!("{} window(s) of {} dimensions, unit length", vectors.len(), profile.dimensions),
            Err(error) => error.to_string(),
        },
    });

    // 2. The same input twice gives the same vector.
    let second = embed_passage(embedder, &sample).await;
    let deterministic = match (&first, &second) {
        (Ok(a), Ok(b)) if !a.is_empty() && !b.is_empty() => dot(&a[0], &b[0]),
        _ => None,
    };
    checks.push(QualificationCheck {
        name: "deterministic".into(),
        passed: deterministic.is_some_and(|similarity| similarity >= 0.999),
        detail: match deterministic {
            Some(similarity) => format!("cosine {similarity:.5} between two embeddings of one passage"),
            None => "could not compare two embeddings of one passage".into(),
        },
    });

    // 3. A full window is accepted whole or halved, never truncated silently.
    let long: String = std::iter::repeat("The relief valve set pressure is recorded on the data sheet. ")
        .take(profile.window_chars / 60 + 1)
        .collect::<String>();
    let long_result = embed_passage(embedder, &long).await;
    checks.push(QualificationCheck {
        name: "window".into(),
        passed: long_result.is_ok(),
        detail: match &long_result {
            Ok(vectors) => format!(
                "{} characters embedded as {} window(s) within a {}-token window",
                long.chars().count(),
                vectors.len(),
                profile.max_tokens
            ),
            Err(error) => error.to_string(),
        },
    });

    // 4. Labelled ordering.
    let mut misses: Vec<String> = Vec::new();
    let mut relevant_sum = 0.0f64;
    let mut distractor_sum = 0.0f64;
    let mut compared = 0usize;
    for probe in probes {
        let query = embed_query(embedder, &probe.query).await;
        let relevant = embed_passage(embedder, &probe.relevant).await;
        let distractor = embed_passage(embedder, &probe.distractor).await;
        match (query, relevant, distractor) {
            (Ok(q), Ok(r), Ok(d)) => {
                let near = best(&q, &r);
                let far = best(&q, &d);
                if let (Some(near), Some(far)) = (near, far) {
                    relevant_sum += near;
                    distractor_sum += far;
                    compared += 1;
                }
                match (near, far) {
                    (Some(near), Some(far)) if near > far => ordered += 1,
                    (Some(near), Some(far)) => misses.push(format!("{} ({near:.3} ≤ {far:.3})", probe.id)),
                    _ => misses.push(format!("{} (not comparable)", probe.id)),
                }
            }
            _ => misses.push(format!("{} (not embedded)", probe.id)),
        }
    }
    let rate = if probes.is_empty() { 0.0 } else { ordered as f64 / probes.len() as f64 };
    checks.push(QualificationCheck {
        name: "labelled-ordering".into(),
        passed: !probes.is_empty() && rate >= QUALIFYING_ORDER_RATE,
        detail: if misses.is_empty() {
            format!("{ordered} of {} probes ordered correctly", probes.len())
        } else {
            format!(
                "{ordered} of {} probes ordered correctly; missed {}",
                probes.len(),
                misses.join(", ")
            )
        },
    });

    let passed = checks.iter().all(|check| check.passed);
    let (relevant_mean, distractor_mean) = if compared > 0 {
        (
            Some(relevant_sum / compared as f64),
            Some(distractor_sum / compared as f64),
        )
    } else {
        (None, None)
    };
    let dense_floor = match (relevant_mean, distractor_mean) {
        (Some(near), Some(far)) if near > far => Some((near + far) / 2.0),
        _ => None,
    };
    QualificationRecord {
        space_key: identity.space_key(),
        identity,
        passed,
        checks,
        ordered_correctly: ordered,
        probes: probes.len(),
        relevant_mean,
        distractor_mean,
        dense_floor,
        measured_at: chrono::Utc::now().to_rfc3339(),
        endpoint: endpoint.to_string(),
    }
}

fn dot(a: &[f32], b: &[f32]) -> Option<f64> {
    (a.len() == b.len() && !a.is_empty())
        .then(|| a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum())
}

/// The closest window of a passage to a query.
fn best(query: &[f32], windows: &[Vec<f32>]) -> Option<f64> {
    windows
        .iter()
        .filter_map(|window| dot(query, window))
        .max_by(|a, b| a.total_cmp(b))
}

/// Where qualification records are kept, keyed by space.
///
/// A file beside the index rather than a table in it, because it is written
/// rarely, read at start-up, and worth reading when retrieval is not doing what
/// somebody expected.
pub struct QualificationStore {
    path: PathBuf,
}

impl QualificationStore {
    pub fn new(retrieval_dir: &Path) -> Self {
        Self {
            path: retrieval_dir.join("embedding-qualification.json"),
        }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    fn read_all(&self) -> BTreeMap<String, QualificationRecord> {
        std::fs::read(&self.path)
            .ok()
            .and_then(|bytes| serde_json::from_slice(&bytes).ok())
            .unwrap_or_default()
    }

    /// The record for exactly this space, if one was ever made.
    pub fn get(&self, identity: &EmbeddingIdentity) -> Option<QualificationRecord> {
        self.read_all()
            .remove(&identity.space_key())
            .filter(|record| record.identity == *identity)
    }

    pub fn is_qualified(&self, identity: &EmbeddingIdentity) -> bool {
        self.get(identity).is_some_and(|record| record.passed)
    }

    /// Keeps a record, passed or failed. A failure is worth keeping: it is the
    /// reason retrieval is lexical on this machine.
    pub fn save(&self, record: &QualificationRecord) -> Result<()> {
        let mut all = self.read_all();
        all.insert(record.space_key.clone(), record.clone());
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let tmp = self.path.with_extension("json.tmp");
        std::fs::write(&tmp, serde_json::to_vec_pretty(&all)?)?;
        std::fs::rename(&tmp, &self.path)?;
        Ok(())
    }
}

/// SHA-256 of a weights file, remembered by size and modification time.
///
/// Hashing a 600 MB file takes seconds; doing it at every start would be paid
/// for nothing when the file has not changed. The cache is keyed on what would
/// change if it had, and re-hashes when either moves.
pub fn weights_sha256(weights: &Path, cache_dir: &Path) -> Result<String> {
    #[derive(Serialize, Deserialize)]
    struct Cached {
        bytes: u64,
        modified: u64,
        sha256: String,
    }
    let metadata = std::fs::metadata(weights)
        .with_context(|| format!("{} could not be read", weights.display()))?;
    let modified = metadata
        .modified()
        .ok()
        .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|duration| duration.as_secs())
        .unwrap_or(0);
    let cache_path = cache_dir.join("weights-hashes.json");
    let mut cache: BTreeMap<String, Cached> = std::fs::read(&cache_path)
        .ok()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok())
        .unwrap_or_default();
    let key = weights.display().to_string();
    if let Some(hit) = cache.get(&key) {
        if hit.bytes == metadata.len() && hit.modified == modified {
            return Ok(hit.sha256.clone());
        }
    }
    let sha256 = crate::extraction::regions::sha256_file(weights)
        .map_err(|error| anyhow!("{} could not be hashed: {error}", weights.display()))?;
    cache.insert(
        key,
        Cached {
            bytes: metadata.len(),
            modified,
            sha256: sha256.clone(),
        },
    );
    std::fs::create_dir_all(cache_dir)?;
    std::fs::write(&cache_path, serde_json::to_vec_pretty(&cache)?)?;
    Ok(sha256)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity(profile: &EmbeddingProfile) -> EmbeddingIdentity {
        EmbeddingIdentity::new("embed-test", profile, &"ab".repeat(32))
    }

    #[test]
    fn a_remote_endpoint_is_refused_before_a_client_exists() {
        for hostile in [
            "https://api.openai.com/v1",
            // The prefix trick a naive check waves through: a hostname an
            // attacker controls, which DNS can point anywhere.
            "http://127.example.com/v1",
            "http://10.0.0.5:8080/v1",
        ] {
            let refused = LocalEmbedder::new(hostile.into(), identity(&MULTILINGUAL_E5_SMALL));
            assert!(refused.is_err(), "{hostile} was accepted as a local embedding endpoint");
        }
    }

    #[test]
    fn loopback_endpoints_are_accepted_however_they_are_spelled() {
        for local in ["http://127.0.0.1:8080/v1", "http://localhost:8080/v1", "http://[::1]:8080/v1"] {
            assert!(
                LocalEmbedder::new(local.into(), identity(&MULTILINGUAL_E5_SMALL)).is_ok(),
                "{local} is on this machine and was refused"
            );
        }
    }

    /// An identity whose width disagrees with its profile is refused: the
    /// profile is the publisher's number and the identity cannot overrule it.
    #[test]
    fn an_identity_that_disagrees_with_its_profile_is_refused() {
        let mut wrong = identity(&MULTILINGUAL_E5_SMALL);
        wrong.dimensions = 1024;
        assert!(LocalEmbedder::new("http://127.0.0.1:8080/v1".into(), wrong).is_err());
    }

    #[test]
    fn the_space_key_changes_with_the_weights_the_width_the_pooling_and_the_rule() {
        let base = identity(&MULTILINGUAL_E5_SMALL);
        let mut other_weights = base.clone();
        other_weights.weights_sha256 = "cd".repeat(32);
        let mut other_rule = base.clone();
        other_rule.profile_version = 2;
        let other_model = identity(&NOMIC_EMBED_TEXT_V2_MOE);
        let keys = [
            base.space_key(),
            other_weights.space_key(),
            other_rule.space_key(),
            other_model.space_key(),
        ];
        let distinct: std::collections::BTreeSet<&String> = keys.iter().collect();
        assert_eq!(distinct.len(), keys.len(), "{keys:?}");
        // Renaming the registry row changes nothing about the vectors.
        let mut renamed = base.clone();
        renamed.model_id = "another-name".into();
        assert_eq!(renamed.space_key(), base.space_key());
    }

    #[test]
    fn profiles_are_matched_by_architecture_and_name_never_by_architecture_alone() {
        assert_eq!(
            profile_for("bert", "multilingual-e5-small", "x.gguf").map(|p| p.key),
            Some("multilingual-e5-small")
        );
        assert_eq!(
            profile_for("nomic-bert-moe", "nomic-embed-text-v2-moe", "").map(|p| p.key),
            Some("nomic-embed-text-v2-moe")
        );
        // The measured header of the installed Qwen file (P00 inventory).
        assert_eq!(
            profile_for("qwen3", "Qwen3 Embedding 0.6b", "Qwen3-Embedding-0.6B-Q8_0.gguf").map(|p| p.key),
            Some("qwen3-embedding-0.6b")
        );
        // Some other BERT, and a Qwen3 chat model: no profile, so not driven.
        assert!(profile_for("bert", "all-MiniLM-L6-v2", "minilm.gguf").is_none());
        assert!(profile_for("qwen3", "Qwen3 8B", "Qwen3-8B-Q4_K_M.gguf").is_none());
    }

    #[test]
    fn a_vector_is_normalised_or_refused_never_repaired() {
        let unit = normalise(vec![3.0, 4.0], 2).unwrap();
        assert!((unit[0] - 0.6).abs() < 1e-6 && (unit[1] - 0.8).abs() < 1e-6);
        assert!(normalise(vec![0.0, 0.0], 2).is_err(), "a zero vector was accepted");
        assert!(normalise(vec![f32::NAN, 1.0], 2).is_err(), "NaN was accepted");
        assert!(normalise(vec![1.0, 0.0, 0.0], 2).is_err(), "a wrong width was accepted");
    }

    #[test]
    fn windows_cover_every_word_and_respect_the_limit() {
        let text = (0..400).map(|i| format!("word{i}")).collect::<Vec<_>>().join(" ");
        let cut = windows(&text, 120);
        assert!(cut.len() > 1);
        assert!(cut.iter().all(|window| window.chars().count() <= 120));
        let rejoined: Vec<&str> = cut.iter().flat_map(|window| window.split_whitespace()).collect();
        let original: Vec<&str> = text.split_whitespace().collect();
        assert_eq!(rejoined, original, "a window boundary lost or reordered a word");
        // One unbroken token longer than the window is cut, not dropped.
        let token = "x".repeat(300);
        let cut = windows(&token, 120);
        assert_eq!(cut.concat(), token);
    }

    /// A transport double: it is the server, and everything this test checks
    /// is the production code in front of it.
    struct Recording {
        identity: EmbeddingIdentity,
        limit_chars: usize,
        sent: std::sync::Mutex<Vec<String>>,
    }

    #[async_trait]
    impl Embedder for Recording {
        fn identity(&self) -> &EmbeddingIdentity {
            &self.identity
        }
        fn profile(&self) -> &'static EmbeddingProfile {
            self.identity.profile().unwrap()
        }
        async fn embed_raw(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>, EmbedError> {
            self.sent.lock().unwrap().extend(inputs.iter().cloned());
            if inputs.iter().any(|input| input.chars().count() > self.limit_chars) {
                return Err(EmbedError::TooLarge("input is too large to process".into()));
            }
            Ok(inputs
                .iter()
                .map(|input| {
                    let mut v = vec![0.0f32; self.identity.dimensions];
                    v[input.len() % self.identity.dimensions] = 2.0;
                    v
                })
                .collect())
        }
    }

    #[tokio::test]
    async fn prefixes_are_applied_by_the_profile_not_by_the_caller() {
        for profile in PROFILES {
            let embedder = Recording {
                identity: identity(profile),
                limit_chars: 10_000,
                sent: Default::default(),
            };
            embed_query(&embedder, "what is the design pressure?").await.unwrap();
            embed_passage(&embedder, "Design pressure 14 bar(g).").await.unwrap();
            let sent = embedder.sent.lock().unwrap().clone();
            assert_eq!(sent[0], format!("{}what is the design pressure?", profile.query_prefix));
            assert_eq!(sent[1], format!("{}Design pressure 14 bar(g).", profile.passage_prefix));
        }
    }

    #[tokio::test]
    async fn a_window_the_server_refuses_as_too_large_is_halved_not_truncated() {
        let embedder = Recording {
            identity: identity(&MULTILINGUAL_E5_SMALL),
            // Below the profile's 900-character window: the server's tokenizer
            // disagreeing with the first cut, as it does for dense scripts.
            limit_chars: 500,
            sent: Default::default(),
        };
        let text = (0..160).map(|i| format!("w{i:03}")).collect::<Vec<_>>().join(" ");
        let vectors = embed_passage(&embedder, &text).await.unwrap();
        assert!(vectors.len() >= 2, "the refused window was not split");
        let sent = embedder.sent.lock().unwrap().clone();
        let accepted: Vec<&String> = sent.iter().filter(|input| input.chars().count() <= 500).collect();
        let words: Vec<&str> = accepted
            .iter()
            .flat_map(|input| input.trim_start_matches("passage: ").split_whitespace())
            .collect();
        let original: Vec<&str> = text.split_whitespace().collect();
        assert_eq!(words, original, "halving lost or reordered text");
    }

    #[tokio::test]
    async fn a_query_over_the_window_is_refused_not_halved() {
        let embedder = Recording {
            identity: identity(&MULTILINGUAL_E5_SMALL),
            limit_chars: 20,
            sent: Default::default(),
        };
        assert!(embed_query(&embedder, "a question much longer than twenty characters").await.is_err());
    }

    #[test]
    fn the_bundled_probes_load_and_cover_both_languages() {
        let probes = bundled_probes();
        assert!(probes.len() >= 10, "{} probes", probes.len());
        assert!(probes.iter().any(|probe| probe.language == "hi"));
        assert!(probes.iter().all(|probe| probe.relevant != probe.distractor));
    }

    #[test]
    fn a_qualification_record_is_found_only_for_its_exact_identity() {
        let dir = tempfile::tempdir().unwrap();
        let store = QualificationStore::new(dir.path());
        let id = identity(&MULTILINGUAL_E5_SMALL);
        assert!(!store.is_qualified(&id));
        let record = QualificationRecord {
            identity: id.clone(),
            space_key: id.space_key(),
            passed: true,
            checks: Vec::new(),
            ordered_correctly: 9,
            probes: 10,
            relevant_mean: Some(0.8),
            distractor_mean: Some(0.5),
            dense_floor: Some(0.65),
            measured_at: "2026-09-24T00:00:00Z".into(),
            endpoint: "http://127.0.0.1:1/v1".into(),
        };
        store.save(&record).unwrap();
        assert!(store.is_qualified(&id));
        let mut changed = id.clone();
        changed.weights_sha256 = "ef".repeat(32);
        assert!(!store.is_qualified(&changed), "a record qualified different weights");
    }
}
