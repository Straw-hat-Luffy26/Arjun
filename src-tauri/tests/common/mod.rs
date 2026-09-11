//! Finding the models this machine actually has.
//!
//! ## Why this exists
//!
//! Three tests in this directory load a real model and run real inference:
//! `verify_llama_runtime`, `verify_gui_execution_trace` and
//! `verify_real_world_model_switching`. They are the only end-to-end proof of
//! properties this product is judged on — that a model loads through the whole
//! stack, and that a conversation survives a switch from one model to another.
//!
//! Each of them used to name its weights outright:
//!
//! ```text
//! load_installed_model_direct(&app_data, "huggingface", "Qwen/Qwen2.5-7B", "Q4_K_M")
//!     .expect("Failed to load Model A")
//! ```
//!
//! Two things were wrong with that. The obvious one is that nothing puts those
//! files on anybody's machine, so all three failed everywhere — and they failed
//! by `expect`, which reads exactly like the loader is broken. The second is
//! that they were masked: cargo stops at the first failing test binary, and
//! `scan_real_library` failed earlier in the alphabet, so `cargo test --tests`
//! reported one failure and these three never ran at all. `--no-fail-fast` is
//! what found them.
//!
//! The third thing, by the time anybody looked: `huggingface` is a provider
//! this build no longer has. The directory survives as a place weights were
//! once downloaded to, and on this machine it holds two empty folders.
//!
//! So these ask the disk instead. A test that needs two models says so, gets
//! the two that are installed, and if there are not two it prints why and
//! returns rather than asserting against weights nobody has.

#![allow(dead_code)]

use std::path::{Path, PathBuf};

/// A model that is installed and has weights on disk right now.
pub struct InstalledModel {
    /// The provider directory it sits under — `local`, or whatever else is
    /// there. Passed straight back to `load_installed_model_direct`.
    pub provider: String,
    /// The model id in the form the loader expects: the package directory name
    /// with the first `_` restored to `/`, e.g. `Qwen_Qwen3.5-9B` ->
    /// `Qwen/Qwen3.5-9B`.
    pub id: String,
    /// The quantisation, read from the GGUF's own filename.
    pub quantization: String,
    pub weights: PathBuf,
}

pub fn app_data_dir() -> PathBuf {
    let appdata = std::env::var("APPDATA")
        .expect("APPDATA is set on every Windows session this runs in");
    PathBuf::from(appdata).join("com.sarathi.app")
}

/// `...-Q4_K_M.gguf` -> `Q4_K_M`. `None` when the filename does not carry one,
/// because guessing a quantisation gets a different file loaded than the one
/// the test thought it named.
fn quantization_of(file: &Path) -> Option<String> {
    let stem = file.file_stem()?.to_str()?;
    stem.rsplit('-')
        .find(|part| {
            let upper = part.to_ascii_uppercase();
            part.len() >= 2
                && (upper.starts_with('Q') || upper.starts_with("IQ") || upper == "F16" || upper == "BF16")
                && upper.chars().any(|c| c.is_ascii_digit())
        })
        .map(|q| q.to_string())
}

/// Every installed model with weights, newest-looking first is not promised —
/// the order is whatever the filesystem gives, sorted, so a run is repeatable.
///
/// Layout walked: `<app_data>/models/<provider>/<Package_Name>/<variant>/*.gguf`.
pub fn installed_models() -> Vec<InstalledModel> {
    let root = app_data_dir().join("models");
    let mut found = Vec::new();

    let Ok(providers) = std::fs::read_dir(&root) else {
        return found;
    };
    let mut providers: Vec<PathBuf> = providers
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    providers.sort();

    for provider_dir in providers {
        let Some(provider) = provider_dir.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        let Ok(packages) = std::fs::read_dir(&provider_dir) else {
            continue;
        };
        let mut packages: Vec<PathBuf> = packages
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.is_dir())
            .collect();
        packages.sort();

        for package_dir in packages {
            let Some(package) = package_dir.file_name().and_then(|n| n.to_str()) else {
                continue;
            };
            // The loader takes `Owner/Model`; the directory is `Owner_Model`.
            let id = match package.split_once('_') {
                Some((owner, rest)) => format!("{owner}/{rest}"),
                None => package.to_string(),
            };

            let mut weights: Vec<PathBuf> = walk_gguf(&package_dir);
            weights.sort();
            // A projector is not a model. Loading one gets a confusing failure
            // several layers down rather than "there is no model here".
            let weight = weights.into_iter().find(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| !n.to_ascii_lowercase().starts_with("mmproj-"))
            });

            if let Some(weight) = weight {
                if let Some(quantization) = quantization_of(&weight) {
                    found.push(InstalledModel {
                        provider: provider.to_string(),
                        id,
                        quantization,
                        weights: weight,
                    });
                }
            }
        }
    }
    found
}

fn walk_gguf(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(next) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&next) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path
                .extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| e.eq_ignore_ascii_case("gguf"))
            {
                out.push(path);
            }
        }
    }
    out
}

/// The models a test needs, or `None` with the reason printed.
///
/// Returning `None` rather than panicking is the point: "this machine has no
/// weights" and "the loader is broken" are different facts, and a test that
/// conflates them sends somebody debugging the wrong thing. The message names
/// what was found, so a skipped run is visible in the output rather than
/// looking like a pass.
pub fn need_models(count: usize, what: &str) -> Option<Vec<InstalledModel>> {
    let installed = installed_models();
    if installed.len() < count {
        println!(
            "SKIPPED: {what} needs {count} installed model(s) and this machine has {}. \
             Install weights under {}/models/<provider>/<Owner_Model>/<variant>/*.gguf \
             and run again.",
            installed.len(),
            app_data_dir().display()
        );
        for model in &installed {
            println!("  found: {}/{} ({})", model.provider, model.id, model.quantization);
        }
        return None;
    }
    Some(installed)
}
