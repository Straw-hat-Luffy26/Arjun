//! **No production caller. `VisionLanguageBridge::new` is constructed only by
//! this module's own tests.**
//!
//! Chat attachments go through `commands::ocr`, which picks its model by a
//! hardcoded id and streams through `ai_engine::ocr_stream`. Nothing routes to
//! `ModelRole::Vision` and nothing builds this bridge. It is kept because the
//! loopback and modality checks in `new` are the ones a vision path would need
//! and are worth not rewriting — but it is not on any path a turn takes, and a
//! reader tracing how a drawing is read should be in `commands/ocr.rs`.
//!
//! The vision-language bridge: taking image paths plus a query, formatting
//! them into a chat-completion call that a local vision-language model
//! understands, and returning a structured description.
//!
//! ## What this module exists to do
//!
//! PS 26117 asks for a local VLM that can read a P&ID, a scanned page, or a
//! datasheet photograph and answer a question about what is on it. The
//! model exists — Qwen2.5-VL, Llava-1.6, InternVL are all on the registry —
//! but each exposes a different chat schema, and a runtime that hard-codes
//! one schema silently fails the others. So the bridge translates a uniform
//! internal request into the OpenAI-compatible vision schema, which all
//! three speak. A new model is a registry entry, not a code change.
//!
//! ## What this module deliberately does not do
//!
//! It does not embed the image bytes into a transport. The OpenAI vision
//! schema accepts a URL or a base64 data URI; we send a base64 data URI
//! because the images are on the local machine and there is no public URL
//! to point at. A model that can ingest from a URL would still receive
//! bytes — the URL would be a leak vector — and the data URI is a
//! self-contained payload that the model cannot accidentally fetch from
//! somewhere else.
//!
//! It does not call cloud endpoints. The base URL must be loopback (the
//! existing runtime loopback check applies), and the only valid hosts are
//! the ones the runtime manages. A model entry with an `https://`
//! base URL is refused at registry load, not at request time, so the
//! bridge never has to defend against it.
//!
//! It does not do OCR. The image's text content is what the VLM returns
//! in its `text` field; the bridge surfaces it as `extracted_text` for
//! downstream indexing but does not parse it further.

use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::registry::{ModelEntry, Modality, RoutingPreference, Runtime};

/// A vision request: one or more images, plus a user query.
///
/// The images are paths on disk rather than raw bytes because the
/// indexing pipeline that calls this bridge already has the paths from
/// the document service. Reading the file here keeps the bridge the
/// only place that needs to handle base64 encoding, and a test fixture
/// can pass paths without constructing bytes.
#[derive(Debug, Clone)]
pub struct VisionRequest {
    /// What the user wants the model to look at and answer. Optional,
    /// because some calls are pure captioning ("describe this image")
    /// rather than question-answering. An empty prompt is honoured by
    /// the model as "describe what you see" and the response is then
    /// returned as a description with no specific question.
    pub query: String,
    /// Paths to the images, in the order the model should see them. The
    /// model attends to all of them simultaneously, which is how a
    /// P&ID that spans two scans is read as one drawing rather than two
    /// unconnected pages.
    pub image_paths: Vec<PathBuf>,
    /// Maximum tokens the model may generate for the response. Bounded
    /// to keep a runaway generation from filling the context window and
    /// pushing the call itself out.
    pub max_tokens: u32,
}

impl VisionRequest {
    /// A captioning-only request: one image, no question.
    pub fn describe(path: impl AsRef<Path>) -> Self {
        Self {
            query: "Describe what you see in the image in detail. Include any text, \
                    labels, numbers, schematic symbols, and their positions."
                .into(),
            image_paths: vec![path.as_ref().to_path_buf()],
            max_tokens: 1024,
        }
    }
}

/// A vision response, in two parts.
///
/// `description` is the model's free-form text — what it would say if
/// asked "tell me about this image". This is the field a model cites
/// when it answers a question about the image.
///
/// `extracted_text` is the text the model saw *on* the image: a tag
/// printed on a P&ID, a number on a datasheet, a label on a scan. It
/// is the text the multimodal index can search, and the field the
/// downstream P&ID detector and OCR engines feed off.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VisionResponse {
    pub description: String,
    pub extracted_text: String,
    /// The model identifier as reported by the runtime. Stamped onto
    /// the response so a reviewer can see which model produced which
    /// answer when several are registered.
    pub model: String,
    /// Tokens the model reported consuming, when it reported. Absent
    /// for a runtime that does not surface this — the value is then
    /// zero rather than fabricated.
    pub tokens_used: u32,
    /// Wall-clock time the call took. Used by the orchestrator to
    /// detect regressions in the runtime.
    pub generation_time_ms: u64,
}

/// The bridge. Holds the loopback endpoint and a client; the client is
/// cheap to clone, so the bridge is cheap to share across the runtime.
pub struct VisionLanguageBridge {
    /// Loopback base URL of the inference server. The runtime's
    /// residency / activation layer guarantees it is on `127.0.0.1`,
    /// not a routable host, so the bridge never has to defend against
    /// an outbound destination.
    pub base_url: String,
    /// The vision-capable model to use. Held by id; the bridge does
    /// not own the model, it borrows the inference runtime's view of
    /// which vision model is currently loaded.
    pub model: ModelEntry,
    /// An HTTP client. Reused across calls so the connection pool
    /// stays warm.
    client: reqwest::Client,
}

impl VisionLanguageBridge {
    /// Builds a bridge for a given model.
    ///
    /// The model entry is the same one the registry hands out: vision
    /// role, image modality, and a base URL that the registry has
    /// already verified is loopback. A model that is not vision-capable
    /// is refused here rather than at request time, so a misuse
    /// surfaces at the construction site.
    ///
    /// The base URL is also re-checked here, on the conservative
    /// principle that the runtime layer that handed it in might one
    /// day be wrong. A non-loopback URL is refused at the construction
    /// site so the sovereignty claim in the field doc stays true
    /// even if the upstream check is bypassed.
    pub fn new(base_url: String, model: ModelEntry) -> Result<Self> {
        // The bridge speaks to a local vLLM / llama.cpp server. The
        // residency / activation layer is documented to hand us a
        // 127.0.0.1 URL; this is the defensive re-check that turns
        // the doc into an invariant. The expected shape is
        // scheme + "://" + host + ":" + port + optional path; anything
        // else is refused.
        let scheme_host_port = base_url
            .splitn(2, "://")
            .nth(1)
            .ok_or_else(|| anyhow::anyhow!("vision bridge base URL is missing a scheme"))?;
        let host_port = scheme_host_port
            .split('/')
            .next()
            .unwrap_or(scheme_host_port);
        let host = host_port
            .rsplit_once(':')
            .map(|(h, _)| h)
            .unwrap_or(host_port)
            .trim_matches('[')
            .trim_matches(']');
        let is_loopback = host.eq_ignore_ascii_case("localhost")
            || host == "127.0.0.1"
            || host == "::1";
        if !is_loopback {
            bail!(
                "vision bridge base URL must be loopback (localhost, 127.0.0.1 or ::1); \
                 got {host:?}"
            );
        }

        if !model.modalities.contains(&Modality::Image) {
            bail!(
                "model {} does not declare image modality, so it cannot be used as a vision bridge",
                model.id
            );
        }
        if !model.roles.contains(&crate::registry::ModelRole::Vision) {
            bail!(
                "model {} is not registered with the Vision role, so it cannot be used as a vision bridge",
                model.id
            );
        }
        // The runtime says it loads GGUF; vision models served by a
        // Python sidecar are out of scope for this bridge, which speaks
        // the OpenAI vision schema over HTTP. A vLLM / llama.cpp server
        // both expose that schema, and the registry's Runtime field is
        // what tells the bridge which wire format to expect — for now,
        // the OpenAI-compatible path.
        if !matches!(model.runtime, Runtime::LlamaCpp) {
            bail!(
                "vision bridge currently only supports llama.cpp-compatible runtimes; \
                 model {} is on {:?}",
                model.id,
                model.runtime
            );
        }
        // arjun-egress-ok: loopback only. The constructor above
        // refuses any non-loopback base URL, so the only host this
        // client can address is the local vLLM / llama.cpp server
        // (127.0.0.1 / ::1 / localhost). Sovereignty: no remote.
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(120))
            .build()
            .context("could not build the vision HTTP client")?;
        Ok(Self { base_url, model, client })
    }

    /// Sends the request and parses the response.
    ///
    /// The call is synchronous; the orchestrator awaits it. Streaming
    /// would be a separate change, and the multimodal index does not
    /// need partial captions — it stores the final answer.
    pub async fn describe(&self, request: &VisionRequest) -> Result<VisionResponse> {
        if request.image_paths.is_empty() {
            bail!("a vision request needs at least one image");
        }

        // Read each image and base64-encode it. The OpenAI vision schema
        // accepts either a URL or a data URI; the data URI keeps the
        // payload self-contained and stops the runtime from fetching
        // the image from anywhere we did not intend.
        let mut content_parts: Vec<Value> = Vec::new();
        for path in &request.image_paths {
            let bytes = tokio::fs::read(path)
                .await
                .with_context(|| format!("could not read image {}", path.display()))?;
            let mime = guess_mime(path);
            let encoded = BASE64.encode(&bytes);
            content_parts.push(json!({
                "type": "image_url",
                "image_url": { "url": format!("data:{};base64,{}", mime, encoded) },
            }));
        }
        // The text part comes last in the OpenAI schema, so a model
        // attends to the images first and the question afterwards.
        if !request.query.is_empty() {
            content_parts.push(json!({ "type": "text", "text": request.query }));
        }

        let body = json!({
            "model": self.model.id,
            "messages": [{
                "role": "user",
                "content": content_parts,
            }],
            "max_tokens": request.max_tokens,
            // The model is asked to be specific. ``temperature: 0`` is
            // what the multimodal index relies on for stable, repeated
            // calls to return the same description for the same image.
            "temperature": 0.0,
            "stream": false,
        });

        let url = format!("{}/v1/chat/completions", self.base_url.trim_end_matches('/'));
        let started = std::time::Instant::now();
        let response = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .with_context(|| format!("vision request to {} failed", url))?;
        let status = response.status();
        let payload: Value = response
            .json()
            .await
            .context("vision response was not parseable as JSON")?;
        if !status.is_success() {
            bail!(
                "vision request to {} returned {}: {}",
                url,
                status,
                payload
            );
        }

        // The OpenAI schema puts the assistant's text in
        // ``choices[0].message.content``. Anything else is a runtime
        // that does not speak the schema, and refusing here is the
        // honest outcome.
        let description = payload
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|arr| arr.first())
            .and_then(|choice| choice.get("message"))
            .and_then(|msg| msg.get("content"))
            .and_then(Value::as_str)
            .ok_or_else(|| {
                anyhow!(
                    "vision response did not contain a choices[0].message.content string: {}",
                    payload
                )
            })?
            .to_string();

        let model_name = payload
            .get("model")
            .and_then(Value::as_str)
            .unwrap_or(&self.model.id)
            .to_string();
        let tokens_used = payload
            .get("usage")
            .and_then(|u| u.get("completion_tokens"))
            .and_then(Value::as_u64)
            .unwrap_or(0) as u32;

        // Extract anything that looks like on-image text. We do not
        // claim certainty here — a model that returns a description
        // mentioning "PT-2201" has the right ``extracted_text``, and
        // a model that returns prose without identifiable text
        // returns an empty string, which the index correctly handles
        // as "no indexable text on this image".
        let extracted_text = extract_text_from_description(&description);

        Ok(VisionResponse {
            description,
            extracted_text,
            model: model_name,
            tokens_used,
            generation_time_ms: started.elapsed().as_millis() as u64,
        })
    }
}

/// The MIME type we tell the runtime the image is. A wrong MIME is
/// not catastrophic (the model inspects the bytes) but a correct one
/// avoids some servers rejecting the upload. We do the matching on
/// the file extension rather than the magic number — image files
/// are written by the document sidecar, which always names them
/// canonically, and magic-number sniffing would add a dependency
/// for no gain.
fn guess_mime(path: &Path) -> &'static str {
    match path.extension().and_then(|e| e.to_str()) {
        Some(ext) => match ext.to_ascii_lowercase().as_str() {
            "png" => "image/png",
            "jpg" | "jpeg" => "image/jpeg",
            "gif" => "image/gif",
            "webp" => "image/webp",
            "bmp" => "image/bmp",
            // Some VLMs accept PDFs as "image"-like documents. We
            // surface the same MIME and let the runtime negotiate.
            "pdf" => "application/pdf",
            _ => "application/octet-stream",
        },
        None => "application/octet-stream",
    }
}

/// Pulls likely on-image text out of the model's free-form description.
///
/// This is a heuristic, not a parser. The idea is: the model, when
/// asked to describe an image, often quotes the on-page text inline —
/// "the tag reads 'PT-2201'", "labelled 'Design pressure 14 bar'".
/// Those quoted fragments are exactly what the index wants. We extract
/// them by walking the string and pulling out anything between matched
/// quote characters (curly or straight, single or double), then
/// dedupe. Long fragments are dropped — a passage longer than 200
/// characters is prose, not a tag.
///
/// A model that does not quote is not wrong; an empty ``extracted_text``
/// is the honest answer, and the index handles it.
fn extract_text_from_description(description: &str) -> String {
    let mut seen: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    // The set of bytes that can start a quoted region. Curly quotes are
    // multi-byte in UTF-8, so we look for the leading byte 0xE2 and
    // match the full three-byte sequence.
    let bytes = description.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        // Identify the opener and the closer. A straight quote opens
        // and closes with itself; a curly open matches its curly close.
        let (opener_len, closer_bytes): (usize, &[u8]) = if bytes[i] == b'"' {
            (1, b"\"")
        } else if bytes[i] == b'\'' {
            (1, b"'")
        } else if bytes[i..].starts_with("\u{201C}".as_bytes()) {
            (3, "\u{201D}".as_bytes())
        } else if bytes[i..].starts_with("\u{2018}".as_bytes()) {
            (3, "\u{2019}".as_bytes())
        } else {
            i += 1;
            continue;
        };
        let start = i + opener_len;
        // Find the matching close.
        let mut end = start;
        while end + closer_bytes.len() <= bytes.len() {
            if &bytes[end..end + closer_bytes.len()] == closer_bytes {
                break;
            }
            // Tolerate a straight close after a curly open, and vice versa.
            if closer_bytes == b"\"" && bytes[end..].starts_with("\u{201D}".as_bytes()) {
                break;
            }
            if closer_bytes == "\u{201D}".as_bytes() && bytes[end] == b'"' {
                break;
            }
            end += 1;
        }
        if end > start && end + closer_bytes.len() <= bytes.len() {
            if let Ok(s) = std::str::from_utf8(&bytes[start..end]) {
                let trimmed = s.trim();
                if !trimmed.is_empty() && trimmed.len() < 200 {
                    seen.insert(trimmed.to_string());
                }
            }
        }
        i = end + closer_bytes.len();
    }
    seen.into_iter().collect::<Vec<_>>().join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::ModelRole;

    fn fixture_model() -> ModelEntry {
        ModelEntry {
            id: "qwen2.5-vl-3b-instruct".into(),
            name: "Qwen2.5-VL 3B Instruct".into(),
            version: "1".into(),
            license: "Apache-2.0".into(),
            sha256: None,
            runtime: Runtime::LlamaCpp,
            roles: vec![ModelRole::Vision],
            modalities: vec![Modality::Text, Modality::Image],
            quantization: Some("Q4_K_M".into()),
            parameters_b: 3.0,
            active_parameters_b: None,
            context_length: 32768,
            weights_bytes: 2_000_000_000,
            supports_structured_output: false,
            permitted_classifications: vec![crate::policy::Classification::Internal],
            path: std::path::PathBuf::from("models/qwen2.5-vl-3b-instruct.gguf"),
            projector: None,
            load: None,
            serving: None,
            required_runtime_profile: None,
            enabled: true,
            routing: RoutingPreference::default(),
        }
    }

    #[test]
    fn non_vision_model_is_refused() {
        let mut model = fixture_model();
        model.modalities = vec![Modality::Text];
        let err = VisionLanguageBridge::new("http://127.0.0.1:8080".into(), model)
            .err()
            .expect("non-vision model must be refused");
        assert!(err.to_string().contains("image modality"));
    }

    #[test]
    fn non_vision_role_is_refused() {
        let mut model = fixture_model();
        model.roles = vec![ModelRole::Reasoning];
        let err = VisionLanguageBridge::new("http://127.0.0.1:8080".into(), model)
            .err()
            .expect("non-vision-role model must be refused");
        assert!(err.to_string().contains("Vision role"));
    }

    #[test]
    fn non_llamacpp_runtime_is_refused() {
        let mut model = fixture_model();
        model.runtime = Runtime::PythonSidecar;
        let err = VisionLanguageBridge::new("http://127.0.0.1:8080".into(), model)
            .err()
            .expect("non-llama.cpp model must be refused");
        assert!(err.to_string().contains("llama.cpp"));
    }

    #[test]
    fn mime_guessing_handles_known_types() {
        assert_eq!(guess_mime(Path::new("a.png")), "image/png");
        assert_eq!(guess_mime(Path::new("a.JPG")), "image/jpeg");
        assert_eq!(guess_mime(Path::new("a.webp")), "image/webp");
        assert_eq!(guess_mime(Path::new("a.pdf")), "application/pdf");
        assert_eq!(guess_mime(Path::new("a.bin")), "application/octet-stream");
    }

    #[test]
    fn extracted_text_collects_quoted_fragments() {
        let description = "The pump is labelled 'P-101'. The tag \"PT-2201\" sits next to it. \
                          The line is named '23-101-4N3'.";
        let extracted = extract_text_from_description(description);
        assert!(extracted.contains("P-101"));
        assert!(extracted.contains("PT-2201"));
        assert!(extracted.contains("23-101-4N3"));
    }

    #[test]
    fn extracted_text_dedupes() {
        let description = "'PT-2201' and 'PT-2201' and 'PT-2201'.";
        let extracted = extract_text_from_description(description);
        // One occurrence in the joined result, not three.
        assert_eq!(extracted.matches("PT-2201").count(), 1);
    }

    #[test]
    fn extracted_text_handles_no_quotes() {
        let description = "A pump with an instrument attached to the discharge line.";
        let extracted = extract_text_from_description(description);
        assert!(extracted.is_empty());
    }
}
