//! Binding an image projector to a model, by what the files say rather than
//! what they are called.
//!
//! ## Why a name is not enough
//!
//! Discovery pairs weights with a sibling `mmproj-*.gguf`, which is the
//! convention and nothing more. A projector saved as `gemma-4-e4b-vision.gguf`
//! is invisible to it, and a `mmproj-*` file beside the wrong model is paired
//! anyway. `llama-server` given a projector whose output width does not match
//! the model's embedding fails at load, or — worse — loads and answers text-only.
//!
//! So a binding is verified from both headers before it is recorded:
//!
//! 1. the projector is a `clip` GGUF,
//! 2. it declares a vision encoder (`clip.has_vision_encoder`),
//! 3. it declares its output width (`clip.vision.projection_dim`), and
//! 4. that width equals the model's `{arch}.embedding_length`.
//!
//! Any filename is accepted. A verified binding is still **not** a model that
//! can see: that takes an actual image call, which is [`super::vision`]'s job.
//! Binding sets the file `--mmproj` is given and nothing else — no role, no
//! modality.
//!
//! ## Where bindings live
//!
//! `<models>/projector-bindings.json`, beside the manifest, applied when the
//! registry loads and re-checked there (header and size) so a file swapped
//! after binding is not silently used. The registry is immutable for the
//! process's life, so a new binding is in force at the next start, and the
//! command that records one says so.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::ai_engine::gguf_meta::{read_gguf_metadata, read_gguf_scalars};

pub const BINDINGS_FILE: &str = "projector-bindings.json";
const BINDINGS_SCHEMA: u32 = 1;

/// What a projector's header says it is.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProjectorHeader {
    pub architecture: String,
    pub has_vision_encoder: Option<bool>,
    pub projector_type: Option<String>,
    pub projection_dim: Option<u64>,
    pub image_size: Option<u64>,
    pub name: Option<String>,
}

/// Reads the header of a would-be projector.
pub fn inspect_projector(path: &Path) -> Result<ProjectorHeader, String> {
    let kv = read_gguf_scalars(path).map_err(|error| format!("{error:#}"))?;
    let text = |key: &str| kv.get(key).and_then(|v| v.as_str()).map(str::to_string);
    let number = |key: &str| kv.get(key).and_then(|v| v.as_u64());
    Ok(ProjectorHeader {
        architecture: text("general.architecture").unwrap_or_default(),
        has_vision_encoder: kv.get("clip.has_vision_encoder").and_then(|v| v.as_bool()),
        projector_type: text("clip.projector_type").or_else(|| text("clip.vision.projector_type")),
        projection_dim: number("clip.vision.projection_dim"),
        image_size: number("clip.vision.image_size"),
        name: text("general.name"),
    })
}

/// A binding that passed every check, as recorded.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProjectorBinding {
    pub model_id: String,
    /// Relative to the models directory when inside it, absolute otherwise —
    /// resolved exactly as a manifest's `projector` is.
    pub projector: PathBuf,
    pub projector_bytes: u64,
    pub projector_type: Option<String>,
    pub projection_dim: u64,
    pub model_architecture: String,
    pub model_embedding: u64,
    pub verified_at: String,
    pub verified_by: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BindingsFile {
    schema_version: u32,
    bindings: Vec<ProjectorBinding>,
}

/// Checks a projector against a model's weights. Every refusal names the key
/// or the numbers that failed.
pub fn verify(weights: &Path, projector: &Path) -> Result<(ProjectorHeader, String, u64), String> {
    if !projector.is_file() {
        return Err(format!("{} is not a file on this machine", projector.display()));
    }
    let file = projector
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default();
    let header = inspect_projector(projector)
        .map_err(|error| format!("{file} could not be read as a GGUF: {error}"))?;
    if header.architecture != "clip" {
        return Err(format!(
            "{file} is a {:?} GGUF, not an image projector (a projector's architecture is \"clip\")",
            header.architecture
        ));
    }
    match header.has_vision_encoder {
        Some(true) => {}
        Some(false) => {
            return Err(format!(
                "{file} is a projector with no vision encoder (clip.has_vision_encoder is false); \
                 it cannot give a model images"
            ))
        }
        None => {
            return Err(format!(
                "{file} does not say whether it has a vision encoder (no clip.has_vision_encoder), \
                 so it cannot be verified as an image projector"
            ))
        }
    }
    let projection = header.projection_dim.ok_or_else(|| {
        format!(
            "{file} does not declare its output width (no clip.vision.projection_dim), so it \
             cannot be checked against the model"
        )
    })?;
    let model = read_gguf_metadata(weights)
        .map_err(|error| format!("the model's weights could not be read: {error:#}"))?;
    let embedding = u64::from(model.embedding_length);
    if embedding == 0 {
        return Err(format!(
            "the model does not declare {}.embedding_length, so no projector can be checked \
             against it",
            model.architecture
        ));
    }
    if projection != embedding {
        return Err(format!(
            "{file} projects images to width {projection}, and this model's embedding width is \
             {embedding} ({}); they are not a pair",
            model.architecture
        ));
    }
    Ok((header, model.architecture, embedding))
}

/// Verifies and records a binding. Replaces an earlier binding for the model.
pub fn bind(
    models_dir: &Path,
    entry: &crate::registry::ModelEntry,
    projector: &Path,
    actor: &str,
) -> Result<ProjectorBinding, String> {
    let resolved = if projector.is_absolute() {
        projector.to_path_buf()
    } else {
        models_dir.join(projector)
    };
    let weights = models_dir.join(&entry.path);
    let (header, architecture, embedding) = verify(&weights, &resolved)?;
    let bytes = std::fs::metadata(&resolved)
        .map_err(|error| format!("{} could not be measured: {error}", resolved.display()))?
        .len();
    let stored = resolved
        .strip_prefix(models_dir)
        .map(Path::to_path_buf)
        .unwrap_or_else(|_| resolved.clone());
    let binding = ProjectorBinding {
        model_id: entry.id.clone(),
        projector: stored,
        projector_bytes: bytes,
        projector_type: header.projector_type,
        projection_dim: embedding,
        model_architecture: architecture,
        model_embedding: embedding,
        verified_at: chrono::Utc::now().to_rfc3339(),
        verified_by: actor.to_string(),
    };
    let mut held = load(models_dir)?;
    held.retain(|b| b.model_id != binding.model_id);
    held.push(binding.clone());
    write(models_dir, &held)?;
    Ok(binding)
}

/// The recorded bindings. A missing file is none.
pub fn load(models_dir: &Path) -> Result<Vec<ProjectorBinding>, String> {
    let path = models_dir.join(BINDINGS_FILE);
    let raw = match std::fs::read(&path) {
        Ok(raw) => raw,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(format!("{} could not be read: {error}", path.display())),
    };
    let file: BindingsFile = serde_json::from_slice(&raw)
        .map_err(|error| format!("{} could not be parsed: {error}", path.display()))?;
    if file.schema_version != BINDINGS_SCHEMA {
        return Err(format!(
            "{} is schema {}, and this build reads {BINDINGS_SCHEMA}",
            path.display(),
            file.schema_version
        ));
    }
    Ok(file.bindings)
}

fn write(models_dir: &Path, bindings: &[ProjectorBinding]) -> Result<(), String> {
    std::fs::create_dir_all(models_dir)
        .map_err(|error| format!("the models directory could not be created: {error}"))?;
    let path = models_dir.join(BINDINGS_FILE);
    let bytes = serde_json::to_vec_pretty(&BindingsFile {
        schema_version: BINDINGS_SCHEMA,
        bindings: bindings.to_vec(),
    })
    .map_err(|error| format!("the bindings could not be encoded: {error}"))?;
    let temporary = path.with_extension("json.tmp");
    std::fs::write(&temporary, bytes)
        .map_err(|error| format!("{} could not be written: {error}", path.display()))?;
    std::fs::rename(&temporary, &path)
        .map_err(|error| format!("{} could not be replaced: {error}", path.display()))
}

/// Applies recorded bindings to loaded entries, re-checking each.
///
/// Returns one line per binding saying what happened, for the log. A binding
/// whose projector changed size or no longer verifies is not applied: the file
/// it was verified against is not the file on disk.
pub fn apply(models_dir: &Path, entries: &mut [crate::registry::ModelEntry]) -> Vec<String> {
    let bindings = match load(models_dir) {
        Ok(bindings) => bindings,
        Err(error) => return vec![format!("projector bindings were not applied: {error}")],
    };
    let mut said = Vec::new();
    for binding in bindings {
        let Some(entry) = entries.iter_mut().find(|e| e.id == binding.model_id) else {
            said.push(format!(
                "a projector binding names {}, which is not in the registry; not applied",
                binding.model_id
            ));
            continue;
        };
        let resolved = models_dir.join(&binding.projector);
        let size = std::fs::metadata(&resolved).map(|m| m.len()).ok();
        if size != Some(binding.projector_bytes) {
            said.push(format!(
                "{}'s bound projector {} is {} now and was {} bytes when verified; not applied",
                binding.model_id,
                resolved.display(),
                size.map(|s| format!("{s} bytes")).unwrap_or_else(|| "missing".into()),
                binding.projector_bytes
            ));
            continue;
        }
        match verify(&models_dir.join(&entry.path), &resolved) {
            Ok(_) => {
                if entry.projector.as_ref() != Some(&binding.projector) {
                    said.push(format!(
                        "{} uses bound projector {} (was {})",
                        binding.model_id,
                        binding.projector.display(),
                        entry
                            .projector
                            .as_ref()
                            .map(|p| p.display().to_string())
                            .unwrap_or_else(|| "none".into())
                    ));
                }
                entry.projector = Some(binding.projector.clone());
            }
            Err(reason) => said.push(format!(
                "{}'s bound projector no longer verifies and was not applied: {reason}",
                binding.model_id
            )),
        }
    }
    said
}

/// Synthetic GGUF headers for tests. No tensors: the checks read headers only.
#[cfg(test)]
pub(crate) mod fixture {
    use std::io::Write;
    use std::path::Path;

    pub enum V<'a> {
        U32(u32),
        Bool(bool),
        Str(&'a str),
    }

    pub fn write_gguf(path: &Path, entries: &[(&str, V)]) {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&(entries.len() as u64).to_le_bytes());
        let string = |out: &mut Vec<u8>, s: &str| {
            out.extend_from_slice(&(s.len() as u64).to_le_bytes());
            out.extend_from_slice(s.as_bytes());
        };
        for (key, value) in entries {
            string(&mut bytes, key);
            match value {
                V::U32(v) => {
                    bytes.extend_from_slice(&4u32.to_le_bytes());
                    bytes.extend_from_slice(&v.to_le_bytes());
                }
                V::Bool(v) => {
                    bytes.extend_from_slice(&7u32.to_le_bytes());
                    bytes.push(u8::from(*v));
                }
                V::Str(v) => {
                    bytes.extend_from_slice(&8u32.to_le_bytes());
                    string(&mut bytes, v);
                }
            }
        }
        std::fs::File::create(path).unwrap().write_all(&bytes).unwrap();
    }

    /// A text model with the given embedding width.
    pub fn model(path: &Path, arch: &str, embedding: u32) {
        let block = format!("{arch}.block_count");
        let width = format!("{arch}.embedding_length");
        let heads = format!("{arch}.attention.head_count");
        write_gguf(
            path,
            &[
                ("general.architecture", V::Str(arch)),
                (&block, V::U32(34)),
                (&width, V::U32(embedding)),
                (&heads, V::U32(8)),
            ],
        );
    }

    /// A vision projector with the given output width.
    pub fn projector(path: &Path, projection: u32) {
        write_gguf(
            path,
            &[
                ("general.architecture", V::Str("clip")),
                ("clip.has_vision_encoder", V::Bool(true)),
                ("clip.projector_type", V::Str("gemma3")),
                ("clip.vision.projection_dim", V::U32(projection)),
                ("clip.vision.image_size", V::U32(896)),
            ],
        );
    }
}

#[cfg(test)]
mod tests {
    use super::fixture::*;
    use super::*;

    fn entry(id: &str, path: &str) -> crate::registry::ModelEntry {
        let mut entry: crate::registry::ModelEntry = serde_json::from_value(serde_json::json!({
            "id": id, "name": id, "version": "1", "license": "x", "sha256": null,
            "runtime": "llamaCpp", "roles": ["reasoning"], "quantization": "Q4_K_M",
            "parametersB": 4.0, "contextLength": 8192, "weightsBytes": 1, "path": path
        }))
        .unwrap();
        entry.projector = None;
        entry
    }

    #[test]
    fn a_projector_with_any_name_binds_when_its_header_matches_the_model() {
        let dir = tempfile::tempdir().unwrap();
        model(&dir.path().join("gemma-4-e4b-Q4_K_M.gguf"), "gemma4", 2560);
        // Deliberately not `mmproj-*`: discovery would never pair this file.
        projector(&dir.path().join("gemma-4-e4b-vision-f16.gguf"), 2560);
        let e = entry("gemma-4-e4b", "gemma-4-e4b-Q4_K_M.gguf");

        let bound = bind(dir.path(), &e, Path::new("gemma-4-e4b-vision-f16.gguf"), "admin").unwrap();
        assert_eq!(bound.projector, PathBuf::from("gemma-4-e4b-vision-f16.gguf"));
        assert_eq!(bound.model_embedding, 2560);

        let mut entries = vec![e];
        apply(dir.path(), &mut entries);
        assert_eq!(entries[0].projector, Some(PathBuf::from("gemma-4-e4b-vision-f16.gguf")));
        // Binding a projector is not the same as being able to see.
        assert!(!entries[0].roles.contains(&crate::registry::ModelRole::Vision));
    }

    #[test]
    fn a_projector_for_another_width_is_refused_with_both_numbers() {
        let dir = tempfile::tempdir().unwrap();
        model(&dir.path().join("qwen.gguf"), "qwen35", 4096);
        projector(&dir.path().join("mmproj-other.gguf"), 2560);
        let error = bind(dir.path(), &entry("qwen", "qwen.gguf"), Path::new("mmproj-other.gguf"), "a")
            .unwrap_err();
        assert!(error.contains("2560") && error.contains("4096"), "{error}");
        assert!(load(dir.path()).unwrap().is_empty(), "a refused binding was recorded");
    }

    #[test]
    fn a_text_model_or_an_audio_projector_is_not_an_image_projector() {
        let dir = tempfile::tempdir().unwrap();
        model(&dir.path().join("m.gguf"), "gemma4", 2560);
        model(&dir.path().join("not-a-projector.gguf"), "gemma4", 2560);
        write_gguf(
            &dir.path().join("audio.gguf"),
            &[
                ("general.architecture", V::Str("clip")),
                ("clip.has_vision_encoder", V::Bool(false)),
                ("clip.vision.projection_dim", V::U32(2560)),
            ],
        );
        let e = entry("m", "m.gguf");
        assert!(bind(dir.path(), &e, Path::new("not-a-projector.gguf"), "a")
            .unwrap_err()
            .contains("not an image projector"));
        assert!(bind(dir.path(), &e, Path::new("audio.gguf"), "a")
            .unwrap_err()
            .contains("no vision encoder"));
    }

    #[test]
    fn a_projector_changed_after_binding_is_not_applied() {
        let dir = tempfile::tempdir().unwrap();
        model(&dir.path().join("m.gguf"), "gemma4", 2560);
        projector(&dir.path().join("p.gguf"), 2560);
        let e = entry("m", "m.gguf");
        bind(dir.path(), &e, Path::new("p.gguf"), "a").unwrap();
        // Swapped for a different file after verification.
        std::fs::write(dir.path().join("p.gguf"), b"GGUF-not-the-same").unwrap();
        let mut entries = vec![e];
        let said = apply(dir.path(), &mut entries);
        assert_eq!(entries[0].projector, None);
        assert!(said.iter().any(|line| line.contains("not applied")), "{said:?}");
    }
}
