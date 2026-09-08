//! Turning text into a vector, using a model on this machine.
//!
//! ## Why this exists
//!
//! [`super::hybrid`] has always described a search with two halves and shipped
//! one. The trait was declared, the fusion was written and tested, and the
//! sentence in [`super::index`] said the plain truth: *"the two model-backed
//! halves are not built because no embedding model is installed."* This is the
//! half that was missing.
//!
//! Keyword search is unbeatable on an exact token — `PV-2201`, `9.0 mm`, a
//! drawing number — and blind to a question phrased in words the document never
//! used. "What stops the charge pump cavitating?" finds nothing by keyword when
//! the manual says "NPSH margin shall exceed 1.5 m". That question is the one a
//! refinery actually asks, and it is the reason to embed at all.
//!
//! ## What it talks to
//!
//! An OpenAI-compatible `/embeddings` endpoint on loopback — the same shape
//! `llama-server`, vLLM and SGLang all expose, so the model is a registry entry
//! rather than a code change, exactly as it is for
//! [`crate::ai_engine::vision_bridge`].
//!
//! ## What it refuses to do
//!
//! **Reach anything but this machine.** The base URL is checked with
//! [`crate::serving::probe::check_loopback`] before the client is constructed,
//! so a non-loopback endpoint fails at construction and there is no later path
//! on which a request could be sent. That check parses the address rather than
//! matching its prefix, which is what stops a hostname like `127.example.com`
//! from passing as local.
//!
//! **Return a vector it did not receive.** Every failure — the server down, a
//! reply of the wrong length, a batch that came back short — is an error. There
//! is no zero-vector fallback and no padding to the expected width. A zero
//! vector is a valid point in the space: it would be stored, compared, ranked
//! and cited, and nothing downstream could tell it from a real embedding of a
//! passage that genuinely says nothing. That is the failure this repository has
//! a standing rule against, and it is why every path here fails loudly.

use anyhow::{anyhow, bail, Context, Result};
use async_trait::async_trait;
use serde::Deserialize;

use super::hybrid::Embedder;

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

/// An embedding model reachable on loopback.
pub struct LocalEmbedder {
    /// Includes the version prefix, e.g. `http://127.0.0.1:8080/v1`.
    base_url: String,
    model_id: String,
    dimensions: usize,
    client: reqwest::Client,
}

impl LocalEmbedder {
    /// Builds a client that can only ever address this machine.
    ///
    /// `dimensions` is what the registry says this model produces. It is not
    /// discovered from the first reply, because a width learned at runtime
    /// cannot be checked against anything: the first reply would define the
    /// truth and every later disagreement would look like the reply being
    /// wrong. Declared up front, a model returning a different width is caught
    /// on its first passage instead of silently seeding an index with two
    /// incompatible geometries.
    pub fn new(base_url: String, model_id: String, dimensions: usize) -> Result<Self> {
        if dimensions == 0 {
            bail!("an embedding model must declare how many dimensions it produces");
        }
        // Parsed, not pattern-matched. See `probe::is_loopback_host`.
        crate::serving::probe::check_loopback(&base_url)
            .map_err(|outcome| anyhow!("{}", outcome.explain(&base_url)))?;

        // arjun-egress-ok: loopback only. `check_loopback` above parses the
        // base URL and returns an error for any host that is not 127.0.0.0/8,
        // ::1 or `localhost`, and it runs before this client exists — so the
        // only host this client can address is a model server on this machine.
        let client = reqwest::Client::builder()
            .timeout(REQUEST_TIMEOUT)
            .build()
            .context("could not build the embedding HTTP client")?;

        Ok(Self {
            base_url,
            model_id,
            dimensions,
            client,
        })
    }

    /// Embeds several passages in one request, in the order they were given.
    ///
    /// The order matters and is not assumed: the reply is sorted by the `index`
    /// each row carries. An OpenAI-compatible server is permitted to answer out
    /// of order, and a batch zipped positionally against its input would attach
    /// every vector to the wrong passage — which is invisible, because each
    /// vector is individually valid and the search simply returns the wrong
    /// paragraph.
    pub async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let url = format!("{}/embeddings", self.base_url.trim_end_matches('/'));
        let response = self
            .client
            .post(&url)
            .json(&serde_json::json!({ "model": self.model_id, "input": texts }))
            .send()
            .await
            .with_context(|| {
                format!(
                    "the embedding model at {} did not answer. Start it, or clear the \
                     embedding model from the registry so search reports itself as \
                     keyword-only rather than failing.",
                    self.base_url
                )
            })?;

        let status = response.status();
        if !status.is_success() {
            let detail = response.text().await.unwrap_or_default();
            bail!("the embedding endpoint answered {status}: {}", detail.trim());
        }

        let body: EmbeddingsResponse = response
            .json()
            .await
            .context("the embedding endpoint answered with something this cannot read")?;

        if body.data.len() != texts.len() {
            bail!(
                "asked for {} embedding(s) and received {}. The batch is not usable, because \
                 which passage was dropped is not recoverable from the reply.",
                texts.len(),
                body.data.len()
            );
        }

        let mut rows = body.data;
        rows.sort_by_key(|row| row.index);

        let mut out = Vec::with_capacity(rows.len());
        for (position, row) in rows.into_iter().enumerate() {
            if row.index != position {
                bail!(
                    "the reply is missing index {position}. Pairing the remaining vectors with \
                     the passages sent would attach them to the wrong text."
                );
            }
            if row.embedding.len() != self.dimensions {
                bail!(
                    "model {} returned a {}-dimension vector where the registry declares {}. \
                     Storing it would put two geometries in one index.",
                    self.model_id,
                    row.embedding.len(),
                    self.dimensions
                );
            }
            out.push(row.embedding);
        }
        Ok(out)
    }
}

#[async_trait]
impl Embedder for LocalEmbedder {
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let batch = self.embed_batch(&[text.to_string()]).await?;
        batch
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("the embedding endpoint returned no vector for the query"))
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn model_id(&self) -> &str {
        &self.model_id
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_remote_endpoint_is_refused_before_a_client_exists() {
        for hostile in [
            "https://api.openai.com/v1",
            // The prefix trick a naive check waves through: a hostname an
            // attacker controls, which DNS can point anywhere.
            "http://127.example.com/v1",
            "http://10.0.0.5:8080/v1",
        ] {
            let refused = LocalEmbedder::new(hostile.into(), "bge-m3".into(), 1024);
            assert!(
                refused.is_err(),
                "{hostile} was accepted as a local embedding endpoint"
            );
        }
    }

    #[test]
    fn loopback_endpoints_are_accepted_however_they_are_spelled() {
        for local in [
            "http://127.0.0.1:8080/v1",
            "http://localhost:8080/v1",
            "http://[::1]:8080/v1",
        ] {
            assert!(
                LocalEmbedder::new(local.into(), "bge-m3".into(), 1024).is_ok(),
                "{local} is on this machine and was refused"
            );
        }
    }

    /// A width of zero would make every stored vector trivially "the right
    /// width", which is the check disabling itself.
    #[test]
    fn a_model_declaring_no_dimensions_is_refused() {
        assert!(LocalEmbedder::new("http://127.0.0.1:8080/v1".into(), "m".into(), 0).is_err());
    }

    /// The reply is trusted for its content, never for its order.
    #[test]
    fn rows_are_paired_with_their_input_by_index_not_by_position() {
        let body: EmbeddingsResponse = serde_json::from_str(
            r#"{"data":[
                {"embedding":[0.0,1.0],"index":1},
                {"embedding":[1.0,0.0],"index":0}
            ]}"#,
        )
        .unwrap();
        let mut rows = body.data;
        rows.sort_by_key(|row| row.index);
        assert_eq!(rows[0].embedding, vec![1.0, 0.0]);
        assert_eq!(rows[1].embedding, vec![0.0, 1.0]);
    }
}
