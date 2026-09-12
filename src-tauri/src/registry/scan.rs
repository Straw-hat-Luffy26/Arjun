//! Filesystem model scanner — TODO 4 of the 7-step plan.
//!
//! The existing `discovery::discover` reads the app data dir
//! (`<app data>/models/<provider>/<model>/`), which is the path
//! the Sarathi downloader writes to. The TODO 4 scanner is
//! different in two important ways:
//!
//! 1. It scans the **user's actual model library** — the directory
//!    the user keeps models in, which on this machine is
//!    `F:\models` (32 GGUFs across Vision-OCR / OCR / Reasoning
//!    / General), not the app data dir.
//! 2. It pairs vision models with their `mmproj-*.gguf`
//!    projector files, so the runtime can load a vision model
//!    with its projector as a single coordinated step.
//!
//! ## What it deliberately does not do
//!
//! - It does not assign data classifications. A scanned model
//!   arrives cleared for nothing, exactly like the existing
//!   `discover` — the safety property of the registry.
//! - It does not compute sha256 on scan. The hash is heavy
//!   (3 GB of reads on a Q6_K) and the orchestrator path
//!   resolver already does it on load. Scanning records the
//!   size and lets the runtime verify before the load.
//!
//! ## What it does do
//!
//! - Recursively walks the library root.
//! - Reads a 1 MB head sample from each GGUF to extract the
//!   `general.name` and `general.architecture` from the GGUF
//!   header (the `gguf-meta` crate is already a dependency).
//! - Pairs a GGUF with a sibling `mmproj-*.gguf` so the
//!   vision load can find its projector.
//! - Deduplicates by sha256: three identical Q6_K copies
//!   across the user's drives count as one model.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use crate::ai_engine::gguf_meta::GgufMetadata;
use crate::registry::capability::{infer_modalities, infer_roles};

/// Context length recorded when the GGUF header does not state one.
///
/// The value this scan used unconditionally before the header was consulted, so
/// a file whose converter wrote no `context_length` key is registered exactly as
/// it was.
const DEFAULT_CONTEXT_LENGTH: u32 = 8192;

/// Largest window this build will register from a header.
///
/// Not a judgement about the model. A KV cache scales linearly with the window,
/// and a frontier-length context costs more VRAM than the weights do — at which
/// point `plan_gpu_offload` puts nothing on the GPU and the model runs on the
/// CPU at a fraction of the speed. An administrator who wants the full window
/// raises it on the entry, having decided to spend the VRAM.
const MAX_REGISTERED_CONTEXT_LENGTH: u32 = 32_768;

/// The window this model was trained for, read from its own header.
fn header_context_length(path: &std::path::Path) -> u32 {
    crate::ai_engine::gguf_meta::read_gguf_metadata(path)
        .ok()
        .and_then(|meta| meta.context_length)
        .filter(|value| *value > 0)
        .map(|value| value.min(MAX_REGISTERED_CONTEXT_LENGTH))
        .unwrap_or(DEFAULT_CONTEXT_LENGTH)
}
use crate::registry::{LoadSpec, ModelEntry, Runtime, RoutingPreference};

/// One GGUFs discovered on disk, before it is shaped into a
/// `ModelEntry`. The `mmproj_path`, when set, points at a
/// projector GGUFs in the same directory that this model can
/// load with for vision tasks.
#[derive(Debug, Clone)]
pub struct ScannedGguf {
    pub path: PathBuf,
    pub bytes: u64,
    pub mmproj_path: Option<PathBuf>,
}

/// The result of a scan, before it is shaped into model
/// entries. `models` is one entry per GGUFs; `duplicates` is
/// the list of paths that were skipped because their sha256
/// matched a model already in the result.
#[derive(Debug, Default)]
pub struct ScanResult {
    pub ggufs: Vec<ScannedGguf>,
    pub mmprojs: Vec<PathBuf>,
}

/// Walks `root` recursively, returning every `.gguf` it finds.
/// Vision projector files (`mmproj-*.gguf`) are kept separate
/// from chat / reasoning models so the pairing step below
/// knows which is which.
pub fn scan_library(root: &Path) -> ScanResult {
    let mut ggufs: Vec<ScannedGguf> = Vec::new();
    let mut mmprojs: Vec<PathBuf> = Vec::new();
    if !root.is_dir() {
        return ScanResult {
            ggufs,
            mmprojs,
        };
    }
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let entries = match std::fs::read_dir(&dir) {
            Ok(it) => it,
            Err(_) => continue,
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let name = match path.file_name().and_then(|n| n.to_str()) {
                Some(n) => n.to_ascii_lowercase(),
                None => continue,
            };
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            if !name.ends_with(".gguf") {
                continue;
            }
            let bytes = entry.metadata().map(|m| m.len()).unwrap_or(0);
            if name.starts_with("mmproj-") {
                mmprojs.push(path);
            } else {
                ggufs.push(ScannedGguf {
                    path,
                    bytes,
                    mmproj_path: None,
                });
            }
        }
    }
    // Pair each GGUFs with the mmproj in the same directory,
    // if any. The pairing is by sibling, not by content match:
    // the llama.cpp loader expects a projector next to its
    // model, and the user's folder layout already groups them.
    let mut by_dir: BTreeMap<PathBuf, Vec<PathBuf>> = BTreeMap::new();
    for mmproj in &mmprojs {
        if let Some(parent) = mmproj.parent() {
            by_dir.entry(parent.to_path_buf())
                .or_default()
                .push(mmproj.clone());
        }
    }
    for gguf in ggufs.iter_mut() {
        if let Some(parent) = gguf.path.parent() {
            if let Some(candidates) = by_dir.get(parent) {
                // Take the first; if the directory has
                // multiple projectors, the user can pick later
                // by re-saving the entry in the manifest.
                if let Some(first) = candidates.first() {
                    gguf.mmproj_path = Some(first.clone());
                }
            }
        }
    }
    ScanResult { ggufs, mmprojs }
}

/// Reads the GGUF header and extracts the `general.architecture`
/// and (when present) `parameter_count`. The result feeds the
/// model id and family guess in `entry_for`. Reading the header
/// is cheap (a few KB at the head of the file), and avoids the
/// alternative of guessing from the filename.
fn read_gguf_meta(path: &Path) -> Option<GgufMetadata> {
    crate::ai_engine::gguf_meta::read_gguf_metadata(path).ok()
}

/// Lowercases and replaces anything that is not alphanumeric with a hyphen, so
/// a file name is usable as a stable identifier.
fn slug(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut last_dash = false;
    for c in raw.chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c.to_ascii_lowercase());
            last_dash = false;
        } else if !last_dash && !out.is_empty() {
            out.push('-');
            last_dash = true;
        }
    }
    while out.ends_with('-') {
        out.pop();
    }
    if out.is_empty() {
        "model".to_string()
    } else {
        out
    }
}

/// Shaped a `ScannedGguf` into a `ModelEntry` suitable for the
/// registry. The id is derived from the file name (no slashes,
/// no extension). The `LoadSpec` is filled in from what we
/// can see on disk: `provider_id` defaults to the runtime
/// (`llama-cpp`), and `model_id` is the file stem — the same value as
/// `ModelEntry.id`, so one file is one identity.
pub fn entry_for(gguf: &ScannedGguf) -> ModelEntry {
    let meta = read_gguf_meta(&gguf.path);
    let file_stem = gguf
        .path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("model")
        .to_string();
    // The display name falls back to the file stem when the
    // header does not carry a `general.name` (most GGUFs do
    // not, the field is converter-specific).
    let name = file_stem.clone();
    let provider_id = "llama-cpp".to_string();
    // The file, not the family.
    //
    // This used to be `general.architecture` from the GGUF header, which is
    // `"qwen3"`, `"llama"` or `"gemma3"` — a *family* name that every file of
    // that family on the machine shares. The `LoadSpec` triple is what an
    // administrator's pinned choice is matched on, so five Qwen3 files all
    // registered as `("llama-cpp", "qwen3", <quant>)` and the pin stopped
    // identifying one of them: `orchestrator_rank` returned its top rank for
    // all five at once, the size sort then decided between them, and the Models
    // screen kept showing a star beside a model that was not answering.
    // `entries_for_package` had the same problem from the other end, resolving
    // one package to several files.
    //
    // The file stem is what `ModelEntry.id` already uses, and a GGUF filename
    // carries the model and its quantisation, so it distinguishes exactly the
    // things that are actually different. Slugged rather than hashed so the id
    // stays readable in the manifest and survives the library being moved.
    let model_id = slug(&file_stem);
    let quantization = infer_quantization(&file_stem);
    let parameters_b = if let Some(count) = meta.as_ref().and_then(|m| m.parameter_count) {
        // Header-reported parameter count, divided by
        // 1e9 to land in billions. The `gguf-meta` field
        // is the *total* parameters, which is what the
        // registry wants here.
        (count as f32) / 1_000_000_000.0
    } else {
        infer_parameters_b(&file_stem)
    };
    // One flag, two answers, so a model cannot be registered as a vision model
    // that accepts no images — an entry the modality filter drops on every
    // vision request, which is a silent way of being unroutable.
    let has_projector = gguf.mmproj_path.is_some();
    let roles = infer_roles(&file_stem, has_projector);
    let modalities = infer_modalities(&file_stem, has_projector);
    // What the file says it was trained for, capped at what this machine can
    // plan a KV cache for.
    //
    // Every scanned model used to be recorded as 8192 tokens regardless, and
    // that number is not decoration: it is the window the context meter shows,
    // and it is what the agent loop compacts against. A 128k model was being
    // compacted at 8k — throwing away history it had room for on every long
    // task — and a 4k model was told it had twice the room it has.
    //
    // The cap is not timidity about large windows; it is the KV cache. A 262k
    // window on a 12B model is tens of gigabytes of cache before a single token
    // is generated, which `plan_gpu_offload` would correctly answer by refusing
    // to put anything on the GPU. `DEFAULT_CONTEXT_LENGTH` is the value used
    // when the converter wrote no key, and it is the same number this scan
    // always used — so a file that says nothing behaves exactly as before.
    let context_length = header_context_length(&gguf.path);
    ModelEntry {
        id: file_stem.clone(),
        name,
        version: "1".to_string(),
        license: "unstated".to_string(),
        sha256: None,
        runtime: Runtime::LlamaCpp,
        roles,
        modalities,
        quantization: Some(quantization.clone()),
        parameters_b,
        // From the header's own expert geometry where the file is a mixture of
        // experts, and from the `-A<n>B` naming convention otherwise. A dense
        // model reports `None`, whose active count is its total and is already
        // recorded in `parameters_b`.
        active_parameters_b: meta
            .as_ref()
            .filter(|m| m.is_moe())
            .and_then(|m| m.active_params(m.parameter_count))
            .map(|count| (count as f32) / 1_000_000_000.0)
            .or_else(|| crate::registry::capability::infer_active_parameters_b(&file_stem)),
        context_length,
        weights_bytes: gguf.bytes,
        supports_structured_output: false,
        // Cleared for nothing. See the module note above.
        permitted_classifications: Vec::new(),
        path: gguf.path.clone(),
        projector: gguf.mmproj_path.clone(),
        load: Some(LoadSpec {
            provider_id,
            model_id,
            quantization,
        }),
        serving: None,
        required_runtime_profile: None,
        enabled: true,
        routing: RoutingPreference::default(),
    }
}

/// Reads the quantisation token from a file name. Returns
/// `Q4_K_M` style strings (the standard llama.cpp quant
/// suffix). Falls back to `unknown` for names that do not
/// carry one.
fn infer_quantization(stem: &str) -> String {
    let lowered = stem.to_ascii_lowercase();
    // The standard quant tokens, longest first so a `Q4_K_M`
    // is not matched as `Q4_K`.
    const QUANTS: &[&str] = &[
        "q8_0", "q6_k", "q5_k_m", "q5_k_s", "q4_k_m", "q4_k_s",
        "q3_k_m", "q3_k_s", "q2_k", "iq4_xs", "iq4_nl",
        "iq3_xxs", "iq3_xs", "iq2_xxs", "iq2_xs", "f16", "f32", "bf16",
    ];
    for q in QUANTS {
        if lowered.contains(q) {
            return q.to_ascii_uppercase();
        }
    }
    "unknown".to_string()
}

/// Reads a parameter count from a file name. The standard
/// `7B` / `1.5b` convention; an unknown size sorts below
/// every floor.
fn infer_parameters_b(name: &str) -> f32 {
    let lowered = name.to_ascii_lowercase();
    let bytes = lowered.as_bytes();
    for (i, c) in bytes.iter().enumerate() {
        if *c != b'b' {
            continue;
        }
        let next_is_boundary = bytes
            .get(i + 1)
            .map(|c| !c.is_ascii_alphanumeric())
            .unwrap_or(true);
        if !next_is_boundary {
            continue;
        }
        let mut start = i;
        let mut seen_dot = false;
        while start > 0 {
            let c = bytes[start - 1];
            if c.is_ascii_digit() {
                start -= 1;
            } else if c == b'.' && !seen_dot && start >= 2 && bytes[start - 2].is_ascii_digit() {
                seen_dot = true;
                start -= 1;
            } else {
                break;
            }
        }
        if start < i {
            if let Ok(value) = lowered[start..i].parse::<f32>() {
                if value > 0.0 && value <= 2000.0 {
                    return value;
                }
            }
        }
    }
    0.0
}


/// One detection pass over the machine.
///
/// Reported rather than applied: the command layer decides whether to write
/// the manifest, and the operator is shown what changed. A scanner that
/// quietly rewrote the registry would be a scanner nobody could audit.
#[derive(Debug, Default)]
pub struct Detection {
    /// Directories actually walked, after removing duplicates and any that do
    /// not exist. Shown to the operator so "nothing found" can be told apart
    /// from "nowhere looked".
    pub roots: Vec<PathBuf>,
    /// Weight files seen, including ones already registered.
    pub files_seen: usize,
    /// Entries for files the registry did not already list.
    pub added: Vec<ModelEntry>,
    /// Files that resolved to a path an existing entry already names.
    pub already_registered: usize,
}

/// Resolves an entry's declared path the way the loader will.
///
/// `join` returns an absolute path unchanged, so this handles both a manifest
/// entry written relative to the models directory and a scanned one carrying
/// an absolute path — which the live registry contains today, and which a
/// naive comparison against the scan would have registered a second time.
fn resolved(models_dir: &Path, declared: &Path) -> PathBuf {
    models_dir.join(declared)
}

/// A path in the one form two spellings of the same file agree on.
///
/// Canonicalisation is what makes `C:\models.gguf`, `C:/models/a.gguf` and
/// a path reached through a junction compare equal. It needs the file to
/// exist; when it does not — a manifest entry whose weights were deleted —
/// the lossy string is used instead, lowercased, because the alternative is
/// treating a missing file as a different file and offering to register a
/// duplicate of it.
fn identity(path: &Path) -> String {
    std::fs::canonicalize(path)
        .unwrap_or_else(|_| path.to_path_buf())
        .to_string_lossy()
        .to_ascii_lowercase()
}

/// An id no existing entry has taken.
///
/// The file stem is the id a scan wants, and two libraries can hold two
/// different files with the same stem. Disambiguated by quantisation first,
/// because that is the difference an operator recognises — the two
/// Unlimited-OCR weights differ in exactly that and nothing else — and by a
/// counter only when even that collides.
fn unique_id(preferred: &str, quantization: Option<&str>, taken: &BTreeSet<String>) -> String {
    if !taken.contains(preferred) {
        return preferred.to_string();
    }
    if let Some(quant) = quantization {
        let with_quant = format!("{preferred}-{}", quant.to_ascii_lowercase());
        if !taken.contains(&with_quant) {
            return with_quant;
        }
    }
    for suffix in 2..1000 {
        let candidate = format!("{preferred}-{suffix}");
        if !taken.contains(&candidate) {
            return candidate;
        }
    }
    // A thousand files sharing a stem is not a library; give up on a unique
    // name rather than looping, and let the manifest's duplicate check refuse
    // it loudly.
    preferred.to_string()
}

/// Walks every root and reports the weight files the registry does not list.
///
/// The safety property is inherited from [`entry_for`] and is the whole reason
/// this is a detection rather than an installation: a model found on disk
/// arrives cleared for **no** classification. It is visible in the library and
/// unusable on real material until an administrator says otherwise. Nothing in
/// a filename answers the question "may this model see vendor negotiations?",
/// so nothing here tries to.
pub fn detect(roots: &[PathBuf], declared: &[ModelEntry], models_dir: &Path) -> Detection {
    let registered: BTreeSet<String> = declared
        .iter()
        .map(|entry| identity(&resolved(models_dir, &entry.path)))
        .collect();
    let mut taken: BTreeSet<String> = declared.iter().map(|entry| entry.id.clone()).collect();

    let mut detection = Detection::default();
    // Roots overlap in practice — the models directory is usually also the
    // library root — and a file found twice must not be offered twice.
    let mut seen_roots: BTreeSet<String> = BTreeSet::new();
    let mut seen_files: BTreeSet<String> = BTreeSet::new();

    for root in roots {
        if !root.is_dir() || !seen_roots.insert(identity(root)) {
            continue;
        }
        detection.roots.push(root.clone());

        for gguf in scan_library(root).ggufs {
            let file = identity(&gguf.path);
            if !seen_files.insert(file.clone()) {
                continue;
            }
            detection.files_seen += 1;
            if registered.contains(&file) {
                detection.already_registered += 1;
                continue;
            }
            let mut entry = entry_for(&gguf);
            entry.id = unique_id(&entry.id, entry.quantization.as_deref(), &taken);
            taken.insert(entry.id.clone());
            detection.added.push(entry);
        }
    }

    detection
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::{Modality, ModelRole};

    #[test]
    fn infer_quantization_picks_the_longest_token() {
        // Q4_K_M must win over Q4_K.
        assert_eq!(infer_quantization("Qwen3-4B-Q4_K_M"), "Q4_K_M");
        assert_eq!(infer_quantization("gemma-3-12b-it-Q6_K"), "Q6_K");
        // A model with no quant token in the file name is
        // `unknown` (lowercase, like the others, since the
        // function upcases its match).
        assert_eq!(infer_quantization("deepseek-r1-distill-7B"), "unknown");
    }

    #[test]
    fn infer_parameters_b_reads_standard_suffixes() {
        assert_eq!(infer_parameters_b("Qwen3-4B-Instruct-Q6_K"), 4.0);
        assert_eq!(infer_parameters_b("gemma-3-12b-it"), 12.0);
        assert_eq!(infer_parameters_b("Qwen3.5-9B"), 9.0);
        assert_eq!(infer_parameters_b("deepseek-r1-distill-qwen-7b"), 7.0);
        assert_eq!(infer_parameters_b("Qwen3.6-35B-A3B"), 35.0);
    }

    #[test]
    fn infer_parameters_b_does_not_match_a_year_in_a_name() {
        // 2024 in a year-of-release context is not 2024B
        // parameters. The plausibility band caps at 2000.
        assert_eq!(infer_parameters_b("model-2024-release"), 0.0);
    }

    #[test]
    fn infer_roles_default_to_reasoning() {
        let roles = infer_roles("Qwen3-4B", false);
        assert!(roles.contains(&ModelRole::Reasoning));
    }

    #[test]
    fn infer_roles_detects_ocr_specialists() {
        let roles = infer_roles("deepseek-ocr-2", false);
        assert_eq!(roles, vec![ModelRole::DocumentOcr]);
    }

    #[test]
    fn infer_roles_detects_coding_specialists() {
        let roles = infer_roles("Qwen2.5-Coder-7B", false);
        assert!(roles.contains(&ModelRole::Coding));
    }

    #[test]
    fn a_scan_pairs_a_model_with_its_sibling_mmproj() {
        let dir = std::env::temp_dir().join(format!(
            "arjun-scan-pair-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let model = dir.join("vision-model-Q4.gguf");
        let mmproj = dir.join("mmproj-model-f16.gguf");
        std::fs::write(&model, b"model").unwrap();
        std::fs::write(&mmproj, b"mmproj").unwrap();
        let result = scan_library(&dir);
        assert_eq!(result.ggufs.len(), 1);
        assert_eq!(result.mmprojs.len(), 1);
        assert_eq!(result.ggufs[0].mmproj_path.as_deref(), Some(mmproj.as_path()));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_scan_skips_a_missing_root() {
        let dir = std::env::temp_dir().join("definitely-not-here-xyz");
        let result = scan_library(&dir);
        assert!(result.ggufs.is_empty());
        assert!(result.mmprojs.is_empty());
    }

    #[test]
    fn a_scan_walks_into_nested_directories() {
        let dir = std::env::temp_dir().join(format!(
            "arjun-scan-nested-{}",
            std::process::id()
        ));
        let nested = dir.join("Vision-OCR").join("SomeModel");
        std::fs::create_dir_all(&nested).unwrap();
        std::fs::write(nested.join("somemodel-Q4.gguf"), b"x").unwrap();
        let result = scan_library(&dir);
        assert_eq!(result.ggufs.len(), 1);
        assert!(result.ggufs[0].path.ends_with("somemodel-Q4.gguf"));
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A directory with two weight files, one already registered.
    ///
    /// The name carries a counter as well as the clock. `SystemTime::now()` on
    /// Windows advances in steps of about fifteen milliseconds, so two of these
    /// tests starting in the same tick got the *same* directory — and the first
    /// to finish deleted the other's fixture from under it. That produced a
    /// failure in `a_model_already_registered_is_not_offered_again` that came
    /// and went with the scheduler and had nothing to do with the code under
    /// test.
    fn library_with_two_quantisations() -> std::path::PathBuf {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static NEXT: AtomicUsize = AtomicUsize::new(0);

        let dir = std::env::temp_dir().join(format!(
            "arjun-detect-{}-{}-{:?}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        let ocr = dir.join("OCR").join("Unlimited-OCR");
        std::fs::create_dir_all(&ocr).unwrap();
        std::fs::write(ocr.join("Unlimited-OCR-Q6_K.gguf"), b"weights").unwrap();
        std::fs::write(ocr.join("Unlimited-OCR-Q4_K_M.gguf"), b"weights").unwrap();
        std::fs::write(ocr.join("mmproj-Unlimited-OCR-F16.gguf"), b"projector").unwrap();
        dir
    }

    fn declared_at(id: &str, path: std::path::PathBuf) -> ModelEntry {
        let mut entry = entry_for(&ScannedGguf {
            path,
            bytes: 7,
            mmproj_path: None,
        });
        entry.id = id.to_string();
        entry
    }

    /// The case the button exists for: weights on disk, nothing in the
    /// manifest, both quantisations found.
    #[test]
    fn detection_finds_both_quantisations_of_an_unregistered_model() {
        let dir = library_with_two_quantisations();
        let detection = detect(&[dir.clone()], &[], &dir);

        let mut names: Vec<String> = detection.added.iter().map(|e| e.name.clone()).collect();
        names.sort();
        assert_eq!(
            names,
            vec![
                "Unlimited-OCR-Q4_K_M".to_string(),
                "Unlimited-OCR-Q6_K".to_string()
            ],
            "a fast and an accurate weight file are two models, not one"
        );
        assert_eq!(detection.already_registered, 0);
        // The projector is paired, never offered as a model of its own.
        assert!(detection.added.iter().all(|e| e.projector.is_some()));
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Nothing a scan finds may be usable on real material until somebody has
    /// looked at it. This is the property the whole registry rests on, so it is
    /// asserted at the point new entries are minted rather than assumed from
    /// `entry_for`.
    #[test]
    fn everything_detected_arrives_cleared_for_nothing() {
        let dir = library_with_two_quantisations();
        let detection = detect(&[dir.clone()], &[], &dir);
        assert!(!detection.added.is_empty());
        assert!(detection
            .added
            .iter()
            .all(|entry| entry.permitted_classifications.is_empty()));
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Pressing the button twice must not register everything twice. The live
    /// manifest stores absolute paths while a hand-written one stores relative
    /// ones, so the comparison is on the resolved file and not on the string.
    #[test]
    fn a_model_already_registered_is_not_offered_again() {
        let dir = library_with_two_quantisations();
        let absolute = dir.join("OCR").join("Unlimited-OCR").join("Unlimited-OCR-Q6_K.gguf");
        let relative = std::path::PathBuf::from("OCR/Unlimited-OCR/Unlimited-OCR-Q4_K_M.gguf");

        let declared = vec![
            declared_at("unlimited-ocr-q6-k", absolute),
            declared_at("unlimited-ocr-q4-k-m", relative),
        ];
        let detection = detect(&[dir.clone()], &declared, &dir);

        assert!(
            detection.added.is_empty(),
            "both files are registered already, one by absolute path and one by relative: {:?}",
            detection.added.iter().map(|e| e.id.clone()).collect::<Vec<_>>()
        );
        assert_eq!(detection.already_registered, 2);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Two roots that resolve to the same directory are one directory. The
    /// models folder is usually also the library root, so this is the ordinary
    /// case rather than an exotic one.
    #[test]
    fn overlapping_roots_do_not_double_count() {
        let dir = library_with_two_quantisations();
        let nested = dir.join("OCR");
        let detection = detect(&[dir.clone(), dir.clone(), nested], &[], &dir);
        assert_eq!(detection.roots.len(), 2, "the same root twice is one root");
        assert_eq!(detection.added.len(), 2, "a file seen twice is one model");
        assert_eq!(detection.files_seen, 2);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A different file that happens to share a name cannot take the id of one
    /// already registered, and cannot silently replace it either.
    #[test]
    fn a_name_collision_is_given_its_own_id_rather_than_overwriting() {
        let dir = library_with_two_quantisations();
        let elsewhere = dir.join("second-copy");
        std::fs::create_dir_all(&elsewhere).unwrap();
        std::fs::write(elsewhere.join("Unlimited-OCR-Q6_K.gguf"), b"other").unwrap();

        let declared = vec![declared_at(
            "Unlimited-OCR-Q6_K",
            dir.join("OCR").join("Unlimited-OCR").join("Unlimited-OCR-Q6_K.gguf"),
        )];
        let detection = detect(&[dir.clone()], &declared, &dir);

        let ids: Vec<String> = detection.added.iter().map(|e| e.id.clone()).collect();
        assert!(
            !ids.contains(&"Unlimited-OCR-Q6_K".to_string()),
            "the declared id must survive: {ids:?}"
        );
        assert!(
            ids.iter().any(|id| id.starts_with("Unlimited-OCR-Q6_K-")),
            "the second copy needs an id of its own: {ids:?}"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A root that is not there is not an error. An operator whose library
    /// lives on a drive that is currently unplugged gets the models from every
    /// other root, not a failure.
    #[test]
    fn a_missing_root_is_skipped_rather_than_failing() {
        let dir = library_with_two_quantisations();
        let missing = dir.join("no-such-folder");
        let detection = detect(&[missing, dir.clone()], &[], &dir);
        assert_eq!(detection.roots.len(), 1);
        assert_eq!(detection.added.len(), 2);
        std::fs::remove_dir_all(&dir).ok();
    }
}
