//! What this build needs on the machine, where it looks, and what it found.
//!
//! ARJUN spawns things it did not compile: `node` runs the agent runtime
//! bundle, `python` runs three sidecars, `llama-server` serves GGUF weights.
//! Until this module existed, each call site invented its own answer to "where
//! is it?" — a bare `Command::new` on a program name, or a hand-rolled list of
//! candidate directories walked from the current working directory.
//!
//! That is survivable in a checkout, where the working directory is the
//! repository and everything is a sibling. It is not survivable in an
//! installed build, and the failure mode is the bad kind: the installer
//! succeeds, the app launches, the panel looks fine, and the first attachment
//! a user opens reports that a Python script "was not found next to the
//! application" — because it was never in the installer to begin with.
//!
//! ## One table, checked by a gate
//!
//! [`DEPENDENCIES`] is the single place that knows what this build needs.
//! `scripts/check-deployment.mjs` reads it and fails the build if a dependency
//! declared [`Packaging::Bundled`] is not covered by `tauri.conf.json`'s
//! `bundle.resources`, so the table and the installer cannot drift apart
//! quietly. The gate is the reason this is a table and not five functions.
//!
//! ## Missing is a verdict, not a fallback
//!
//! [`Resolution::Missing`] carries every path that was tried. There is no
//! branch below that returns a plausible-looking default when a probe fails —
//! the repository rule ("measure, or fail loudly") applies to a filesystem
//! probe exactly as it applies to a benchmark. A dependency that could not be
//! found reports where it looked, and the caller's error message is then
//! something an operator can act on rather than a shrug.
//!
//! ## No Tauri
//!
//! Like [`crate::agent_runtime`], this module does not depend on Tauri, which
//! is what lets its tests run with no application. The packaged resource
//! directory is handed in once at startup through [`set_resource_dir`]; before
//! that call, and in tests, resolution simply skips the packaged candidate.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use serde::{Deserialize, Serialize};

/// The packaged resource directory, as Tauri resolved it.
///
/// A process-wide cell rather than a parameter threaded through a dozen call
/// sites, because the value is a property of the installation and is known
/// before any of those call sites can run. [`set_resource_dir`] is called once
/// from `lib.rs`'s setup hook.
static RESOURCE_DIR: OnceLock<PathBuf> = OnceLock::new();

/// Records where the packaged resources live.
///
/// Idempotent and first-write-wins: a second call is ignored rather than
/// panicking, so a test binary that initialises twice does not abort.
pub fn set_resource_dir(dir: PathBuf) {
    let _ = RESOURCE_DIR.set(dir);
}

/// The packaged resource directory, if this is a packaged build.
pub fn resource_dir() -> Option<&'static Path> {
    RESOURCE_DIR.get().map(PathBuf::as_path)
}

/// How a dependency gets onto the machine.
///
/// Named `Packaging` rather than the more obvious "provisioning" because
/// [`crate::sovereignty::mode::OperatingMode::Provisioning`] already means
/// something else entirely in this codebase — the mode in which the network is
/// reachable — and two unrelated senses of that word would be a trap.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Packaging {
    /// Shipped inside the installer, under `bundle.resources`.
    ///
    /// The gate checks that the declared `bundle_path` is actually covered by
    /// a `tauri.conf.json` resource entry.
    Bundled,
    /// Not shipped: an executable the operator installs, or that the offline
    /// deployment pack lays down beside the app.
    ///
    /// These are the honest part of the picture. ARJUN cannot redistribute a
    /// CPython or a CUDA-linked `llama-server` inside a Tauri bundle without a
    /// deliberate licensing and signing decision, so it does not pretend to.
    External,
}

/// What a dependency is needed for, in the words an operator would use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Criticality {
    /// Chat does not work without it.
    Core,
    /// One feature does not work without it; the rest of the app does.
    Feature,
}

/// One thing this build needs and did not compile.
#[derive(Debug, Clone)]
pub struct Dependency {
    /// Stable identifier, used by the gate and by the preflight report.
    pub id: &'static str,
    /// What an operator would call it.
    pub label: &'static str,
    /// What stops working when it is absent. Written to be shown in a UI.
    pub needed_for: &'static str,
    pub packaging: Packaging,
    pub criticality: Criticality,
    /// Environment variable that overrides resolution, if there is one.
    ///
    /// Checked before every other candidate, so an operator can point a
    /// deployment at a vendored copy without moving files around.
    pub env_override: Option<&'static str>,
    /// Path relative to the resource directory, for [`Packaging::Bundled`].
    ///
    /// Also the path relative to the repository root in a checkout, which is
    /// what makes the development fallback a one-liner rather than a list.
    pub bundle_path: Option<&'static str>,
    /// Program name looked up on `PATH`, for [`Packaging::External`].
    pub program: Option<&'static str>,
    /// What to tell the operator when it is missing.
    ///
    /// Held here rather than written at the call site so the same dependency
    /// produces the same remedy wherever it is discovered to be absent.
    pub remedy: &'static str,
}

/// Everything this build needs on the machine.
///
/// Adding a spawn of some new external program without adding it here is the
/// mistake this table exists to make visible; `scripts/check-deployment.mjs`
/// reads the program names below and fails on a `Command::new` literal that is
/// not among them.
pub const DEPENDENCIES: &[Dependency] = &[
    Dependency {
        id: "agent-runtime",
        label: "Agent runtime bundle",
        needed_for: "every chat turn",
        packaging: Packaging::Bundled,
        criticality: Criticality::Core,
        env_override: Some("ARJUN_AGENT_RUNTIME"),
        bundle_path: Some("agent-runtime/dist/arjun-agent-runtime.mjs"),
        program: None,
        remedy: "Run `npm run runtime:build` and rebuild the installer.",
    },
    Dependency {
        id: "node",
        label: "Node.js",
        needed_for: "executing the agent runtime bundle",
        packaging: Packaging::External,
        criticality: Criticality::Core,
        env_override: Some("ARJUN_NODE"),
        bundle_path: None,
        program: Some("node"),
        remedy: "Install Node.js 20 or newer, or set ARJUN_NODE to a node executable \
                 laid down by the offline deployment pack.",
    },
    Dependency {
        id: "python",
        label: "Python",
        needed_for: "the document, memory and voice sidecars",
        packaging: Packaging::External,
        criticality: Criticality::Core,
        env_override: Some("ARJUN_PYTHON"),
        bundle_path: None,
        program: Some("python"),
        remedy: "Install Python 3.11 or newer, or set ARJUN_PYTHON to an interpreter \
                 laid down by the offline deployment pack.",
    },
    Dependency {
        id: "llama-server",
        label: "llama-server",
        needed_for: "serving GGUF models",
        packaging: Packaging::External,
        criticality: Criticality::Core,
        env_override: Some("ARJUN_LLAMA_SERVER"),
        bundle_path: None,
        program: Some("llama-server"),
        remedy: "Install llama.cpp's server binary, or set ARJUN_LLAMA_SERVER to the \
                 copy shipped in the offline deployment pack.",
    },
    Dependency {
        id: "document-sidecar",
        label: "Document sidecar",
        needed_for: "reading PDFs and Office files",
        packaging: Packaging::Bundled,
        criticality: Criticality::Feature,
        env_override: Some("ARJUN_DOCUMENT_SIDECAR"),
        bundle_path: Some("sidecars/document_sidecar/main.py"),
        program: None,
        remedy: "Reinstall ARJUN; the document sidecar ships inside the installer.",
    },
    Dependency {
        id: "document-extractor",
        label: "Attachment extractor",
        needed_for: "reading files attached to a chat",
        packaging: Packaging::Bundled,
        criticality: Criticality::Feature,
        env_override: Some("ARJUN_DOCUMENT_EXTRACTOR"),
        bundle_path: Some("sidecars/document_sidecar/attachment_extract.py"),
        program: None,
        remedy: "Reinstall ARJUN; the attachment extractor ships inside the installer.",
    },
    Dependency {
        id: "graph-sidecar",
        label: "Knowledge graph sidecar",
        needed_for: "naming the relations between things a document mentions",
        packaging: Packaging::Bundled,
        // A Feature, not Core. Without it the graph still builds: the
        // statistical pass finds the nodes and the co-occurrence edges, and the
        // edges simply stay unnamed. A missing relation extractor costs labels,
        // not the graph.
        criticality: Criticality::Feature,
        env_override: Some("ARJUN_GRAPH_SIDECAR"),
        bundle_path: Some("sidecars/graph_sidecar/main.py"),
        program: None,
        remedy: "Reinstall ARJUN; the graph sidecar ships inside the installer.",
    },
    Dependency {
        id: "memory-sidecar",
        label: "Memory engine sidecar",
        needed_for: "long-term memory recall",
        packaging: Packaging::Bundled,
        criticality: Criticality::Feature,
        env_override: Some("ARJUN_MEMORY_SIDECAR"),
        bundle_path: Some("sidecars/memory_engine_sidecar/main.py"),
        program: None,
        remedy: "Reinstall ARJUN; the memory sidecar ships inside the installer.",
    },
    Dependency {
        id: "voice-sidecar",
        label: "Voice sidecar",
        needed_for: "speech transcription",
        packaging: Packaging::Bundled,
        criticality: Criticality::Feature,
        env_override: Some("ARJUN_VOICE_SIDECAR"),
        bundle_path: Some("sidecars/voice_sidecar/voice_bridge.py"),
        program: None,
        remedy: "Reinstall ARJUN; the voice sidecar ships inside the installer.",
    },
];

/// Looks a dependency up by id.
///
/// Panics on an unknown id, deliberately: ids are compile-time constants used
/// by this crate's own call sites, so a miss is a programming error that
/// should not survive to a release, not a runtime condition to handle.
pub fn dependency(id: &str) -> &'static Dependency {
    DEPENDENCIES
        .iter()
        .find(|d| d.id == id)
        .unwrap_or_else(|| panic!("no dependency is registered under the id {id:?}"))
}

/// Where a dependency was found, and by which rule.
///
/// The rule matters as much as the path. "Found on PATH" and "found in the
/// installer" are the difference between a build that works because it shipped
/// what it needs and one that works because this particular machine happens to
/// have a developer's toolchain on it — and only the second one breaks when it
/// reaches an operator.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "via", rename_all = "camelCase")]
pub enum Resolution {
    /// An environment variable named it.
    EnvOverride { variable: String, path: PathBuf },
    /// Found inside the installed application's resource directory.
    Packaged { path: PathBuf },
    /// Found in the repository, relative to `CARGO_MANIFEST_DIR`.
    ///
    /// Correct in a checkout and a red flag in a packaged build: it means the
    /// installer did not ship something and the machine happened to have the
    /// source tree. [`DependencyStatus::development_only`] says so.
    Checkout { path: PathBuf },
    /// A bare program name that the OS will resolve against `PATH` at spawn.
    ///
    /// Not probed here. Probing means spawning the program, and spawning
    /// `llama-server` to ask whether it exists costs a model load. The spawn
    /// itself is the probe, and its failure carries [`Dependency::remedy`].
    SystemPath { program: String },
    /// Nothing matched.
    Missing { looked_in: Vec<PathBuf> },
}

impl Resolution {
    /// The path to hand to a spawn, if resolution produced one.
    pub fn path(&self) -> Option<&Path> {
        match self {
            Self::EnvOverride { path, .. } | Self::Packaged { path } | Self::Checkout { path } => {
                Some(path)
            }
            Self::SystemPath { .. } | Self::Missing { .. } => None,
        }
    }

    /// The program name to spawn, for an external dependency.
    pub fn program(&self) -> Option<&str> {
        match self {
            Self::SystemPath { program } => Some(program),
            Self::EnvOverride { path, .. } => path.to_str(),
            _ => None,
        }
    }

    pub fn is_missing(&self) -> bool {
        matches!(self, Self::Missing { .. })
    }
}

/// The repository root, for the checkout fallback.
///
/// `CARGO_MANIFEST_DIR` is `src-tauri/`; everything this module resolves is a
/// sibling of it. Deliberately not `current_dir()`: the working directory of a
/// GUI process launched from a Start menu shortcut is whatever the shell felt
/// like, which is how the previous hand-rolled candidate lists came to include
/// four spellings of "maybe up one level".
fn checkout_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_default()
}

/// Finds one dependency.
///
/// The order is fixed and is the whole point: an explicit override, then what
/// the installer shipped, then the checkout, then `PATH`. A packaged build
/// that resolves everything at step two needs nothing from the machine, which
/// is the property the offline deployment pack has to establish.
pub fn resolve(dep: &Dependency) -> Resolution {
    let mut looked_in = Vec::new();

    if let Some(variable) = dep.env_override {
        if let Ok(raw) = std::env::var(variable) {
            let path = PathBuf::from(&raw);
            // An override that names a missing file is an operator mistake
            // worth surfacing, not a reason to quietly fall through to a
            // different copy than the one they asked for.
            if path.exists() {
                return Resolution::EnvOverride {
                    variable: variable.to_string(),
                    path,
                };
            }
            looked_in.push(path);
        }
    }

    if let Some(relative) = dep.bundle_path {
        if let Some(dir) = resource_dir() {
            let candidate = dir.join(relative);
            if candidate.exists() {
                return Resolution::Packaged { path: candidate };
            }
            looked_in.push(candidate);
        }

        let candidate = checkout_root().join(relative);
        if candidate.exists() {
            return Resolution::Checkout { path: candidate };
        }
        looked_in.push(candidate);
    }

    if let Some(program) = dep.program {
        return Resolution::SystemPath {
            program: program.to_string(),
        };
    }

    Resolution::Missing { looked_in }
}

/// Resolves by id, for call sites that only know the name.
pub fn resolve_id(id: &str) -> Resolution {
    resolve(dependency(id))
}

/// The program name to spawn for an external dependency, by id.
///
/// Returns the override when one is set and the declared program name
/// otherwise, so `Command::new(deployment::program("python"))` replaces
/// `Command::new("python")` without changing behaviour on a machine that has no
/// override set — while giving an offline deployment pack a supported way to
/// point at the interpreter it laid down.
pub fn program(id: &str) -> String {
    let dep = dependency(id);
    resolve(dep)
        .program()
        .map(str::to_string)
        .unwrap_or_else(|| dep.program.unwrap_or(dep.id).to_string())
}

/// The message to show when a dependency could not be found.
///
/// One sentence about what is missing and what it costs, one about where the
/// search went, and the remedy. Written here so the wording is identical
/// wherever the absence is noticed.
pub fn missing_message(dep: &Dependency, resolution: &Resolution) -> String {
    let mut message = format!(
        "{} is required for {} and was not found.",
        dep.label, dep.needed_for
    );
    if let Resolution::Missing { looked_in } = resolution {
        if !looked_in.is_empty() {
            let paths: Vec<String> = looked_in.iter().map(|p| p.display().to_string()).collect();
            message.push_str(&format!(" Looked in: {}.", paths.join(", ")));
        }
    }
    message.push(' ');
    message.push_str(dep.remedy);
    message
}

/// Resolves a bundled dependency to a path, or explains why it could not.
///
/// The shape every call site that needs a script wants: a `PathBuf`, or a
/// sentence fit to show the user.
pub fn require_path(id: &str) -> Result<PathBuf, String> {
    let dep = dependency(id);
    let resolution = resolve(dep);
    match resolution.path() {
        Some(path) => Ok(path.to_path_buf()),
        None => Err(missing_message(dep, &resolution)),
    }
}

/// One line of the preflight report.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DependencyStatus {
    pub id: String,
    pub label: String,
    pub needed_for: String,
    pub packaging: Packaging,
    pub criticality: Criticality,
    pub resolution: Resolution,
    /// True when a bundled dependency was found only in the checkout.
    ///
    /// The single most useful line in this report. A packaged build in which
    /// this is true for anything is a build that would fail on a clean
    /// machine, and it is invisible to every other check because on a
    /// developer's laptop it works.
    pub development_only: bool,
    /// The operator-facing remedy, present only when something is wrong.
    pub remedy: Option<String>,
}

/// Resolves every dependency and reports what it found.
///
/// Cheap — filesystem existence checks and environment reads, no spawns — so
/// it is safe to call at startup and from a health panel. Nothing here touches
/// the network, which is the constraint [`crate::health`] documents and which
/// applies with equal force to a report about the installation.
pub fn preflight() -> Vec<DependencyStatus> {
    DEPENDENCIES
        .iter()
        .map(|dep| {
            let resolution = resolve(dep);
            let development_only = dep.packaging == Packaging::Bundled
                && matches!(resolution, Resolution::Checkout { .. });
            let remedy = if resolution.is_missing() || development_only {
                Some(dep.remedy.to_string())
            } else {
                None
            };
            DependencyStatus {
                id: dep.id.to_string(),
                label: dep.label.to_string(),
                needed_for: dep.needed_for.to_string(),
                packaging: dep.packaging,
                criticality: dep.criticality,
                resolution,
                development_only,
                remedy,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every id is unique, because the gate and the report both key on it.
    #[test]
    fn dependency_ids_are_unique() {
        let mut seen = std::collections::HashSet::new();
        for dep in DEPENDENCIES {
            assert!(seen.insert(dep.id), "duplicate dependency id {:?}", dep.id);
        }
    }

    /// A bundled dependency without a bundle path could not be shipped, and an
    /// external one without a program name could not be spawned. Either is a
    /// table entry that cannot do its job.
    #[test]
    fn every_dependency_can_actually_be_located() {
        for dep in DEPENDENCIES {
            match dep.packaging {
                Packaging::Bundled => assert!(
                    dep.bundle_path.is_some(),
                    "{} is bundled but declares no bundle path",
                    dep.id
                ),
                Packaging::External => assert!(
                    dep.program.is_some(),
                    "{} is external but declares no program name",
                    dep.id
                ),
            }
        }
    }

    /// The remedy is what an operator is left holding when something is
    /// missing, so an empty one is a bug in the table.
    #[test]
    fn every_dependency_carries_a_remedy() {
        for dep in DEPENDENCIES {
            assert!(!dep.remedy.trim().is_empty(), "{} has no remedy", dep.id);
        }
    }

    /// The checkout fallback has to actually find the files this repository
    /// ships, or the development path is broken and every developer works
    /// around it.
    #[test]
    fn bundled_dependencies_resolve_in_a_checkout() {
        for dep in DEPENDENCIES {
            if dep.packaging != Packaging::Bundled {
                continue;
            }
            // The agent runtime bundle is a build output, absent until
            // `npm run runtime:build` has run, so its absence here says
            // nothing about the table.
            if dep.id == "agent-runtime" {
                continue;
            }
            let candidate = checkout_root().join(dep.bundle_path.unwrap());
            assert!(
                candidate.exists(),
                "{} declares {:?}, which is not in the repository",
                dep.id,
                dep.bundle_path.unwrap()
            );
        }
    }

    /// A bundled dependency that is nowhere reports every path it tried,
    /// rather than a bare "not found" the operator cannot act on.
    #[test]
    fn a_missing_dependency_reports_where_it_looked() {
        let dep = Dependency {
            id: "test-missing",
            label: "Test",
            needed_for: "the test",
            packaging: Packaging::Bundled,
            criticality: Criticality::Feature,
            env_override: None,
            bundle_path: Some("definitely/not/here.py"),
            program: None,
            remedy: "Reinstall.",
        };
        let resolved = resolve(&dep);
        match &resolved {
            Resolution::Missing { looked_in } => {
                assert!(!looked_in.is_empty(), "no paths were reported");
            }
            other => panic!("expected Missing, got {other:?}"),
        }
        let message = missing_message(&dep, &resolved);
        assert!(message.contains("Looked in:"), "got: {message}");
        assert!(message.contains("Reinstall."), "got: {message}");
    }

    /// The preflight covers the whole table and never spawns anything — a
    /// preflight that started `llama-server` to see whether it existed would
    /// load a model every time the health panel opened.
    #[test]
    fn preflight_covers_every_dependency_without_spawning() {
        let report = preflight();
        assert_eq!(report.len(), DEPENDENCIES.len());
        for status in &report {
            // External dependencies are never probed, so they can only come
            // back as an override or as a name for the OS to resolve.
            if status.packaging == Packaging::External {
                assert!(
                    matches!(
                        status.resolution,
                        Resolution::SystemPath { .. } | Resolution::EnvOverride { .. }
                    ),
                    "{} was probed: {:?}",
                    status.id,
                    status.resolution
                );
            }
        }
    }

    /// In a checkout with no override set, the bundled scripts resolve to the
    /// repository and say so. This is the case that must be distinguishable
    /// from a packaged build, and `development_only` is how.
    #[test]
    fn a_checkout_resolution_is_flagged_as_development_only() {
        let status = preflight()
            .into_iter()
            .find(|s| s.id == "document-extractor")
            .expect("the extractor is in the table");
        // Guard against a developer machine that happens to set the override.
        if matches!(status.resolution, Resolution::EnvOverride { .. }) {
            return;
        }
        assert!(
            status.development_only,
            "expected the checkout copy to be flagged, got {:?}",
            status.resolution
        );
        assert!(status.remedy.is_some(), "a flagged status needs a remedy");
    }
}
