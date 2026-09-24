//! Commands for the model registry and the router.

use std::sync::Arc;

use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager, State};

use crate::commands::governance::{require_permission, require_session, CurrentSession};
use crate::config::ConfigManager;
use crate::core::event_bus::{get_event_bus, SarathiEvent};
use crate::identity::Permission;
use crate::policy::Classification;
use crate::ai_engine::activation::{ActivationOutcome, InferenceLoader, ModelActivator};
use crate::audit::{AuditKind, AuditService};
use crate::registry::router::{ModelRouter, RoutingDecision};
use crate::registry::{ModelEntry, ModelRegistry};
use crate::serving::ModelServers;
use crate::system_analyzer::gpu_collector;
use crate::ai_engine::startup::StartupModelTarget;
use crate::download_manager::traits::InstalledModel;

/// The activator, shared across commands.
pub type SharedActivator = Arc<ModelActivator<InferenceLoader>>;

/// Where a swap narrates itself. One channel, one step per message.
const ORCHESTRATOR_SWAP_EVENT: &str = "models://orchestrator";

/// A routed and loaded model, ready to run.
#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PreparedModel {
    pub routing: RoutingDecision,
    pub activation: ActivationOutcome,
}

/// The orchestrator an administrator chose, or `None` when nobody has.
///
/// Read from the configuration on every call rather than cached at startup:
/// the choice changes while the app is running, and a routing decision made
/// after the change must reflect it. A configuration that cannot be read is
/// treated as "nothing chosen" — the router then picks on capability alone,
/// which is a worse answer than the administrator wanted but a working one.
pub fn configured_orchestrator(app: &AppHandle) -> Option<StartupModelTarget> {
    let config = ConfigManager::load(&ConfigManager::get_config_path(app))
        .map_err(|e| {
            log::warn!(
                "[REGISTRY] the configuration could not be read, so no orchestrator is \
                 being honoured for this decision: {e}"
            );
        })
        .ok()?;
    StartupModelTarget::configured(&config.ai_settings)
}

/// The exact installed model variant selected to run the orchestrator, or
/// `null` when an administrator has not chosen one. No model is compiled in as
/// a default, so "not chosen yet" is a real state the UI has to show.
#[tauri::command]
pub async fn get_orchestrator_model(
    app: AppHandle,
    session: State<'_, CurrentSession>,
) -> Result<Option<StartupModelTarget>, String> {
    require_session(&session)?;
    Ok(configured_orchestrator(&app))
}

/// One step of a swap, as it happens.
///
/// Emitted on `models://orchestrator` so the Models screen can show the change
/// taking place rather than a spinner that says "Saving…" while a several-
/// gigabyte model is read off disk. Every phase names the model it is about:
/// during a release that is the model going away, not the one arriving.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct OrchestratorSwapStep {
    /// `releasing` | `loading` | `ready` | `failed`.
    pub phase: &'static str,
    pub model_id: String,
    pub model_name: String,
    /// Present on `failed`, and on nothing else.
    pub detail: Option<String>,
}

/// What `set_orchestrator_model` did.
///
/// The selection is reported separately from the swap because they can succeed
/// independently: the choice is written to the configuration first and survives
/// a server that then refuses to start, so the next launch still honours it.
/// Reporting one field for both would have to lie about one of them.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct OrchestratorChange {
    /// The coordinates written to the configuration, as the registry states
    /// them — which is not always how the installed package states them.
    pub selected: StartupModelTarget,
    pub model_id: String,
    pub model_name: String,
    /// Models whose servers were stopped to make room. Empty when none ran.
    pub released: Vec<String>,
    /// True when the new orchestrator is up and answering.
    pub serving: bool,
    /// Why it is not, when it is not.
    pub detail: Option<String>,
}

/// Selects any ready installed model as the orchestrator, and swaps to it now.
///
/// Administrator only. Two things happen, in this order and for this reason:
///
/// 1. The choice is persisted, as the *registry* states the coordinates. It has
///    to be the registry's spelling: routing matches an administrator's choice
///    against registry entries, and the installed package describes its own
///    quantisation from the file name, which is a different vocabulary. Storing
///    the package's spelling is what made this setting do nothing at all — the
///    star moved in the Models list and the chat carried on answering from the
///    previous model, because no entry ever matched.
/// 2. The running model server is swapped. Without this the change took effect
///    at the next launch, which is not what choosing a model in a list means.
///
/// Step 1 is not rolled back when step 2 fails. A model that cannot be started
/// right now — no VRAM free, a server that will not come up — is still the
/// model the administrator chose, and discarding that choice would be a second
/// surprise on top of the first.
#[tauri::command]
pub async fn set_orchestrator_model(
    app: AppHandle,
    session: State<'_, CurrentSession>,
    audit: State<'_, Arc<AuditService>>,
    registry: State<'_, Arc<ModelRegistry>>,
    servers: State<'_, Arc<ModelServers>>,
    provider_id: String,
    model_id: String,
    quantization: String,
) -> Result<OrchestratorChange, String> {
    let signed_in = require_permission(&session, Permission::ModifyPolicy)?;
    let requested = StartupModelTarget {
        provider_id: provider_id.trim().to_string(),
        model_id: model_id.trim().to_string(),
        quantization: quantization.trim().to_string(),
    };
    if requested.provider_id.is_empty()
        || requested.model_id.is_empty()
        || requested.quantization.is_empty()
    {
        return Err("Provider, model and quantization are all required.".to_string());
    }

    let app_data = app.path().app_data_dir().map_err(|e| e.to_string())?;
    let installed = crate::model_manager::ModelManager::list_installed_models(&app_data);
    let on_disk = resolve_installed_orchestrator(&installed, &requested)?;
    // The whole fix, in one line: what gets saved is what routing can match.
    let selected = registered_coordinates(&registry, &on_disk)?;

    let entry = registry
        .orchestrator_entry_for(Some(&selected))
        .ok_or_else(|| {
            format!(
                "{} resolved to registry coordinates that then matched no entry. The registry \
                 at {} may have changed while this was being saved.",
                selected.model_id,
                registry.manifest_path().display()
            )
        })?;
    let entry_id = entry.id.clone();
    let entry_name = entry.name.clone();

    let config_path = ConfigManager::get_config_path(&app);
    let mut config = ConfigManager::load(&config_path).map_err(|e| e.to_string())?;
    config.ai_settings.orchestrator_provider_id = selected.provider_id.clone();
    config.ai_settings.orchestrator_model_id = selected.model_id.clone();
    config.ai_settings.orchestrator_quantization = selected.quantization.clone();
    config.ai_settings.auto_load_on_startup = true;
    config.ai_settings.use_gpu = true;
    ConfigManager::save(&config, &config_path).map_err(|e| e.to_string())?;

    get_event_bus().publish(
        SarathiEvent::ConfigChanged,
        Some(serde_json::json!({ "orchestrator": &selected })),
    );

    let swap = swap_to(&app, &registry, &servers, &entry_id, &entry_name).await;

    let _ = audit.record(
        &signed_in.user.id,
        AuditKind::ModelRegistry,
        format!(
            "Set orchestrator to {} ({}){}",
            selected.model_id,
            selected.quantization,
            match (&swap.serving, swap.released.as_slice()) {
                (true, []) => ", now serving".to_string(),
                (true, released) => format!(", now serving, releasing {}", released.join(", ")),
                (false, _) => ", not yet serving".to_string(),
            }
        ),
        Some(serde_json::json!({
            "providerId": &selected.provider_id,
            "modelId": &selected.model_id,
            "quantization": &selected.quantization,
            "registryId": &entry_id,
            "released": &swap.released,
            "serving": swap.serving,
            "detail": &swap.detail,
        })),
    );

    Ok(OrchestratorChange {
        selected,
        model_id: entry_id,
        model_name: entry_name,
        released: swap.released,
        serving: swap.serving,
        detail: swap.detail,
    })
}

/// What the swap half of `set_orchestrator_model` produced.
struct SwapOutcome {
    released: Vec<String>,
    serving: bool,
    detail: Option<String>,
}

/// Stops whatever is serving, starts the new orchestrator, and narrates it.
///
/// The order matters on a machine where both models do not fit at once, which
/// is the ordinary case: releasing first is what makes room. It also means a
/// failure to start leaves nothing serving, so the caller reports `serving:
/// false` rather than the previous model quietly continuing to answer — which
/// is the shape of the bug this whole change is about.
async fn swap_to(
    app: &AppHandle,
    registry: &ModelRegistry,
    servers: &ModelServers,
    entry_id: &str,
    entry_name: &str,
) -> SwapOutcome {
    let step = |phase: &'static str, model_id: &str, model_name: &str, detail: Option<String>| {
        let _ = app.emit(
            ORCHESTRATOR_SWAP_EVENT,
            OrchestratorSwapStep {
                phase,
                model_id: model_id.to_string(),
                model_name: model_name.to_string(),
                detail,
            },
        );
    };

    // Stopping every other server and loading this one is heavy work, and
    // stopping a server mid-generation is exactly what the one lease exists to
    // prevent. So the swap waits for the card and holds it until it returns.
    let _swap_lease = match tauri::Manager::try_state::<Arc<crate::subagents::ModelScheduler>>(app) {
        Some(scheduler) => match scheduler
            .acquire(
                crate::subagents::LeaseRequest::new(
                    format!("swap:{entry_id}"),
                    crate::subagents::LeaseClass::Parent,
                    entry_id,
                )
                .waiting(std::time::Duration::from_secs(120)),
            )
            .await
        {
            Ok(lease) => Some(lease),
            Err(refusal) => {
                let detail = refusal.explain();
                step("failed", entry_id, entry_name, Some(detail.clone()));
                return SwapOutcome {
                    released: Vec::new(),
                    serving: false,
                    detail: Some(detail),
                };
            }
        },
        None => None,
    };

    let mut released = Vec::new();
    for running in servers.running_model_ids() {
        if running == entry_id {
            continue;
        }
        // The display name if the registry knows it, the id if it does not. A
        // server can outlive the entry that started it.
        let name = registry
            .find(&running)
            .map(|entry| entry.name.clone())
            .unwrap_or_else(|| running.clone());
        step("releasing", &running, &name, None);
        servers.stop(&running).await;
        released.push(name);
    }

    step("loading", entry_id, entry_name, None);

    let Some(entry) = registry.find(entry_id) else {
        let detail = format!("{entry_name} is no longer in the registry.");
        step("failed", entry_id, entry_name, Some(detail.clone()));
        return SwapOutcome {
            released,
            serving: false,
            detail: Some(detail),
        };
    };

    // Budgeted against free VRAM with the model's own layer count, and any
    // other server released only if this one will not otherwise fit. See
    // `serving::admission`.
    //
    // A model too large for this machine is reported here, before anything is
    // started. That is the difference between an administrator being told it
    // will not fit in a second, and watching the swap sit on "Loading" for the
    // full three-minute readiness timeout first.
    let plan = match crate::serving::admission::admit(&servers, entry, registry.models_dir()).await
    {
        Ok(admitted) => {
            if !admitted.released.is_empty() {
                log::info!(
                    "[serving] released {} to make room for {}",
                    admitted.released.join(", "),
                    entry.name
                );
            }
            admitted.plan
        }
        Err(error) => {
            let detail = error.to_string();
            step("failed", entry_id, entry_name, Some(detail.clone()));
            return SwapOutcome {
                released,
                serving: false,
                detail: Some(detail),
            };
        }
    };

    match servers
        .endpoint_for(entry, registry.models_dir(), &plan)
        .await
    {
        Ok(_) => {
            step("ready", entry_id, entry_name, None);
            SwapOutcome {
                released,
                serving: true,
                detail: None,
            }
        }
        Err(error) => {
            let detail = error.to_string();
            step("failed", entry_id, entry_name, Some(detail.clone()));
            SwapOutcome {
                released,
                serving: false,
                detail: Some(detail),
            }
        }
    }
}

fn resolve_installed_orchestrator(
    installed: &[InstalledModel],
    requested: &StartupModelTarget,
) -> Result<StartupModelTarget, String> {
    installed
        .iter()
        .find(|model| {
            requested.matches_installed(model) && model.is_ready && model.size_bytes > 0
        })
        .map(StartupModelTarget::from_installed)
        .ok_or_else(|| {
            format!(
                "{} ({}) is not a ready installed model.",
                requested.model_id, requested.quantization
            )
        })
}

/// A placeholder quantisation names the container rather than the weights.
/// `GGUF` is what the package manifest records when the file name declares
/// nothing this build can parse, and it identifies no particular variant.
fn is_placeholder_quantization(quantization: &str) -> bool {
    let trimmed = quantization.trim();
    trimmed.is_empty() || trimmed.eq_ignore_ascii_case("gguf")
}

/// Translates a selection made from what is on disk into the coordinates the
/// router matches on.
///
/// The bug this exists to close: the orchestrator was persisted from the
/// installed package, whose quantisation comes from the file name, while
/// routing compares against the registry, whose quantisation is what an
/// administrator wrote in the manifest. On a machine holding `Q4_K_S` and
/// `UD-Q4_K_XL` files the package side recorded both as "GGUF", so nothing ever
/// matched, and the chat went on answering from whichever model the capability
/// sort reached first — with the Models screen still showing the star beside
/// the model nobody was talking to.
///
/// Selecting a model the registry does not know is refused rather than saved.
/// Routing can only choose registered models, so persisting an unregistered one
/// would store a preference that can never be honoured, which is the silence
/// this function exists to break.
fn registered_coordinates(
    registry: &ModelRegistry,
    installed: &StartupModelTarget,
) -> Result<StartupModelTarget, String> {
    let entries = registry.entries_for_package(&installed.provider_id, &installed.model_id);
    if entries.is_empty() {
        return Err(format!(
            "{} is installed but is not in the model registry, so routing cannot select it. \
             Add it to {} and try again.",
            installed.model_id,
            registry.manifest_path().display()
        ));
    }

    // An exact quantisation match is the unambiguous case and is taken first,
    // whatever else is registered for the same package.
    let exact = entries.iter().find_map(|entry| {
        let load = entry.load.as_ref()?;
        load.quantization
            .eq_ignore_ascii_case(&installed.quantization)
            .then(|| StartupModelTarget::from_load(entry))
            .flatten()
    });
    if let Some(target) = exact {
        return Ok(target);
    }

    // No exact match. A selection carrying a real label means the registry does
    // not hold that variant, and saying so is better than quietly substituting
    // another one.
    if !is_placeholder_quantization(&installed.quantization) {
        return Err(format!(
            "{} is registered, but not at quantisation {}. Registered: {}.",
            installed.model_id,
            installed.quantization,
            describe_quantizations(&entries)
        ));
    }

    // The selection says only "a GGUF of this model". That is enough when the
    // registry holds one variant of the package and not enough when it holds
    // several — guessing between two variants of the same model is the one
    // thing the exact-coordinates design exists to prevent.
    match entries.as_slice() {
        [only] => StartupModelTarget::from_load(only).ok_or_else(|| {
            format!(
                "{} is registered but carries no load coordinates, so the runtime has no file \
                 to open. Give its registry entry a load block.",
                installed.model_id
            )
        }),
        many => Err(format!(
            "{} is registered at {} variants ({}), and the installed copy does not say which \
             one it is. Name the quantisation in the registry entry, or keep only the variant \
             you want installed.",
            installed.model_id,
            many.len(),
            describe_quantizations(many)
        )),
    }
}

/// The registered quantisations of a package, for a message that has to say
/// what the alternatives actually were.
fn describe_quantizations(entries: &[&crate::registry::ModelEntry]) -> String {
    entries
        .iter()
        .filter_map(|entry| entry.load.as_ref().map(|load| load.quantization.clone()))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Every registered model, including disabled ones.
#[tauri::command]
pub async fn list_registered_models(
    registry: State<'_, Arc<ModelRegistry>>,
    session: State<'_, CurrentSession>,
) -> Result<Vec<ModelEntry>, String> {
    // Read-only inspection. Any signed-in user may see the registry; the
    // matrix does not gate read paths for the model list itself.
    require_session(&session)?;
    Ok(registry.all().to_vec())
}

/// One model as the library screen shows it.
///
/// Deliberately not `ModelEntry` itself. The screen needs two things the
/// manifest does not carry — whether the weights are actually where the entry
/// says they are, and how large they are *now* — and an entry that answers
/// neither is how a half-finished download comes to be listed as an installed
/// model.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LibraryModel {
    pub id: String,
    pub name: String,
    pub quantization: Option<String>,
    pub roles: Vec<crate::registry::ModelRole>,
    pub modalities: Vec<crate::registry::Modality>,
    pub parameters_b: f32,
    pub context_length: u32,
    /// What the manifest says the weights weigh.
    pub weights_bytes: u64,
    pub license: String,
    pub enabled: bool,
    /// Empty for everything a scan found, and that is the safety property
    /// rather than an oversight: a model nobody has reviewed is cleared for
    /// nothing and cannot be routed to on real material.
    pub permitted_classifications: Vec<Classification>,
    pub runtime: String,
    /// Resolved against the models directory, so the screen shows the file the
    /// loader will actually open rather than a fragment of a path.
    pub path: String,
    pub projector: Option<String>,
    /// Whether that file exists. A registered model whose weights have been
    /// moved or deleted is still registered, and saying so is more useful than
    /// listing it as though it were ready to run.
    pub present: bool,
    /// Bytes on disk right now, when the file is there. Absent rather than
    /// falling back to the manifest figure — a truncated download has to look
    /// truncated.
    pub bytes_on_disk: Option<u64>,
}

fn library_model(entry: &ModelEntry, models_dir: &std::path::Path) -> LibraryModel {
    // `join` leaves an absolute path alone, so a manifest entry written
    // relative to the models directory and a scanned one carrying a full path
    // both resolve to the file the loader opens.
    let resolved = models_dir.join(&entry.path);
    let metadata = std::fs::metadata(&resolved).ok();
    LibraryModel {
        id: entry.id.clone(),
        name: entry.name.clone(),
        quantization: entry.quantization.clone(),
        roles: entry.roles.clone(),
        modalities: entry.modalities.clone(),
        parameters_b: entry.parameters_b,
        context_length: entry.context_length,
        weights_bytes: entry.weights_bytes,
        license: entry.license.clone(),
        enabled: entry.enabled,
        permitted_classifications: entry.permitted_classifications.clone(),
        runtime: entry.runtime.label().to_string(),
        path: resolved.display().to_string(),
        projector: entry
            .projector
            .as_ref()
            .map(|projector| models_dir.join(projector).display().to_string()),
        present: metadata.is_some(),
        bytes_on_disk: metadata.map(|meta| meta.len()),
    }
}

/// Every directory worth walking for weight files on this machine.
///
/// Four sources, because models arrive four ways, and a library that knows
/// only one of them is why both Unlimited-OCR weights were invisible on the
/// models screen while the chat was reading documents with them:
///
/// 1. `<app data>/models`, where the downloader writes.
/// 2. `<local app data>/models`, where the OCR weights were placed by hand.
/// 3. `ARJUN_MODEL_LIBRARY`, when an operator has pointed ARJUN at an existing
///    library instead of copying it in.
/// 4. The directory beside every model already registered — which is what
///    makes a second quantisation dropped next to the first turn up without
///    anybody having to name the folder.
fn library_roots(
    app: &AppHandle,
    registry: &ModelRegistry,
    extra: &[String],
) -> Vec<std::path::PathBuf> {
    let mut roots: Vec<std::path::PathBuf> = Vec::new();

    if let Ok(app_data) = app.path().app_data_dir() {
        roots.push(app_data.join("models"));
        roots.push(registry.library_root(&app_data));
    }
    if let Ok(local) = app.path().app_local_data_dir() {
        roots.push(local.join("models"));
    }
    for entry in registry.all() {
        if let Some(parent) = registry.models_dir().join(&entry.path).parent() {
            roots.push(parent.to_path_buf());
        }
    }
    roots.extend(
        extra
            .iter()
            .filter(|root| !root.trim().is_empty())
            .map(std::path::PathBuf::from),
    );
    roots
}

/// What one detection pass found, as the screen reports it.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DetectionReport {
    /// Directories actually walked. Reported so "nothing new" can be told
    /// apart from "nowhere was looked".
    pub roots: Vec<String>,
    pub files_seen: usize,
    pub already_registered: usize,
    /// Newly registered models, every one of them cleared for nothing.
    pub added: Vec<LibraryModel>,
    /// The library as it stands after the pass, so the screen does not have to
    /// guess what changed.
    pub models: Vec<LibraryModel>,
    /// True when something was added.
    ///
    /// The routing tables are built at startup from the manifest, and this
    /// command writes the manifest rather than rebuilding them. A model
    /// detected now is listed now and routed to after the next launch. Saying
    /// so plainly beats a screen that lists a model the router cannot yet see.
    pub restart_required: bool,
}

/// Every model in the manifest, read from disk.
///
/// From disk rather than from the registry held in memory, because the two
/// diverge the moment a detection pass writes a new entry — and a library
/// screen that cannot show what was just detected is not a library screen.
#[tauri::command]
pub async fn list_library_models(
    registry: State<'_, Arc<ModelRegistry>>,
    session: State<'_, CurrentSession>,
) -> Result<Vec<LibraryModel>, String> {
    require_session(&session)?;
    let models_dir = registry.models_dir().to_path_buf();
    let on_disk = ModelRegistry::load(&models_dir).map_err(|error| error.to_string())?;
    Ok(on_disk
        .all()
        .iter()
        .map(|entry| library_model(entry, &models_dir))
        .collect())
}

/// Finds every weight file on this machine and registers the ones the manifest
/// does not already list.
///
/// This is the button behind "the library should know what is installed". It
/// exists because registering a model was otherwise a text edit: the scanner
/// underneath it has been in the tree, complete and tested, with no caller at
/// all — so a model that was on disk and working, as both Unlimited-OCR
/// weights were, appeared nowhere in the interface.
///
/// It adds. It never edits and never removes: an entry an administrator wrote
/// by hand wins over anything a filename suggests, which is the rule
/// `discovery::merge` has always followed.
#[tauri::command]
pub async fn detect_system_models(
    app: AppHandle,
    registry: State<'_, Arc<ModelRegistry>>,
    session: State<'_, CurrentSession>,
    extra_roots: Option<Vec<String>>,
) -> Result<DetectionReport, String> {
    // Writing the manifest is model management, gated exactly as installing
    // one is.
    require_permission(&session, Permission::ImportModel)?;

    let models_dir = registry.models_dir().to_path_buf();
    let roots = library_roots(&app, &registry, &extra_roots.unwrap_or_default());

    // Read fresh rather than using the registry in memory: an earlier pass in
    // this same session has already written entries the in-memory copy does
    // not have, and detecting against a stale list would offer to add every
    // one of them again under a disambiguated id.
    let mut declared = ModelRegistry::load(&models_dir).map_err(|error| error.to_string())?;
    let detection = crate::registry::scan::detect(&roots, declared.all(), &models_dir);

    let added: Vec<LibraryModel> = detection
        .added
        .iter()
        .map(|entry| library_model(entry, &models_dir))
        .collect();

    if !detection.added.is_empty() {
        let mut merged = declared.all().to_vec();
        merged.extend(detection.added.iter().cloned());
        declared.replace_and_save(merged)?;
        log::info!(
            "[REGISTRY] detection registered {} model(s) across {} directories; every one is cleared for no classification until an administrator reviews it",
            detection.added.len(),
            detection.roots.len()
        );
    }

    Ok(DetectionReport {
        roots: detection
            .roots
            .iter()
            .map(|root| root.display().to_string())
            .collect(),
        files_seen: detection.files_seen,
        already_registered: detection.already_registered,
        restart_required: !added.is_empty(),
        added,
        models: declared
            .all()
            .iter()
            .map(|entry| library_model(entry, &models_dir))
            .collect(),
    })
}

/// Where the manifest lives, so an administrator can find the file to edit.
///
/// Registering a model is editing this file and restarting — there is no import
/// wizard to go through, and no code change. Showing the path makes that
/// concrete rather than a claim in the documentation.
#[tauri::command]
pub async fn model_manifest_path(
    registry: State<'_, Arc<ModelRegistry>>,
    session: State<'_, CurrentSession>,
) -> Result<String, String> {
    require_session(&session)?;
    Ok(registry.manifest_path().display().to_string())
}

/// Shows which model would handle a prompt, without running anything.
///
/// This is what makes automatic selection visible instead of implicit: the same
/// routing the orchestrator will use, reported before the task starts, with the
/// reasons that produced it.
#[tauri::command]
pub async fn preview_routing(
    app: AppHandle,
    registry: State<'_, Arc<ModelRegistry>>,
    session: State<'_, CurrentSession>,
    prompt: String,
    classification: Option<Classification>,
) -> Result<RoutingDecision, String> {
    // Routing reveals which models exist and what they are cleared for, so it
    // needs a signed-in user like anything else.
    require_session(&session)?;

    // Read from the live hardware rather than a stored figure: the right model
    // on a workstation is the wrong one on a laptop, and adapting to the machine
    // it is actually on is the whole point of this router.
    //
    // The largest GPU wins on a multi-GPU box, matching what the inference
    // engine will use. No GPU at all reports zero, and the planner turns that
    // into a CPU-only plan rather than an error.
    let vram = gpu_collector::installed_gpus()
        .iter()
        .map(|gpu| gpu.dedicated_video_memory_bytes)
        .max()
        .unwrap_or(0);

    // Read the same way a run reads it, so the preview names the model that
    // will actually answer.
    let intent = crate::capability::laya_sidecar::analyze_for_turn(
        crate::capability::laya_sidecar::managed(&app),
        &prompt,
    )
    .await;

    ModelRouter::route_analyzed(
        &registry,
        &intent,
        classification,
        vram,
        None,
        false,
        &[],
        &[],
        configured_orchestrator(&app).as_ref(),
    )
    .map_err(|failure| failure.reason)
}

/// Picks the right model for a prompt and loads it, with no human step.
///
/// This is the automatic selection the problem statement asks to be
/// demonstrated: a coding request and a summarisation request each end with a
/// different model resident, and the trace records the routing reasons and what
/// was evicted to get there.
///
/// Refuses while another task holds the model, rather than swapping underneath
/// it. The refusal names the holder so the wait is explicable.
#[tauri::command]
pub async fn prepare_model_for(
    app: AppHandle,
    registry: State<'_, Arc<ModelRegistry>>,
    activator: State<'_, SharedActivator>,
    session: State<'_, CurrentSession>,
    audit: State<'_, Arc<AuditService>>,
    prompt: String,
    classification: Option<Classification>,
) -> Result<PreparedModel, String> {
    let signed_in = require_permission(&session, Permission::ImportModel)?;

    let vram = gpu_collector::installed_gpus()
        .iter()
        .map(|gpu| gpu.dedicated_video_memory_bytes)
        .max()
        .unwrap_or(0);

    let intent = crate::capability::laya_sidecar::analyze_for_turn(
        crate::capability::laya_sidecar::managed(&app),
        &prompt,
    )
    .await;

    let routing = ModelRouter::route_analyzed(
        &registry,
        &intent,
        classification,
        vram,
        None,
        false,
        &[],
        &[],
        configured_orchestrator(&app).as_ref(),
    )
    .map_err(|failure| failure.reason)?;

    // Loading into the in-process engine takes VRAM like any other load, so
    // it waits for the one lease and holds it for the load only. See
    // `subagents::scheduling`.
    let _load_lease = match tauri::Manager::try_state::<Arc<crate::subagents::ModelScheduler>>(&app) {
        Some(scheduler) => Some(
            scheduler
                .acquire(
                    crate::subagents::LeaseRequest::new(
                        format!("prepare:{}", uuid::Uuid::new_v4()),
                        crate::subagents::LeaseClass::Parent,
                        routing.model_id.as_str(),
                    )
                    .waiting(std::time::Duration::from_secs(120)),
                )
                .await
                .map_err(|refusal| refusal.explain())?,
        ),
        None => None,
    };
    let activation = activator
        .ensure_ready(&registry, &routing.model_id, &signed_in.user.id)
        .map_err(|e| e.message())?;

    // Recorded whether or not a swap happened: "which model answered this" is
    // exactly the question an auditor asks afterwards, and it cannot be
    // reconstructed later from the prompt alone.
    let _ = audit.record(
        &signed_in.user.id,
        AuditKind::ModelRegistry,
        format!(
            "Routed to {} ({}){}",
            routing.model_name,
            routing.role.label(),
            match &activation.evicted {
                Some(evicted) => format!(", releasing {evicted}"),
                None if activation.already_resident => ", already loaded".to_string(),
                None => ", loaded".to_string(),
            }
        ),
        Some(serde_json::json!({
            "modelId": routing.model_id,
            "role": routing.role,
            "intent": routing.intent,
            "confidence": routing.confidence,
            "usedFallback": routing.used_fallback,
            "reasons": routing.reasons,
            "evicted": activation.evicted,
            "alreadyResident": activation.already_resident,
            "tookMs": activation.took_ms,
        })),
    );

    Ok(PreparedModel { routing, activation })
}

/// Which model is loaded right now, and who is using it.
#[tauri::command]
pub async fn model_residency(
    activator: State<'_, SharedActivator>,
) -> Result<serde_json::Value, String> {
    Ok(serde_json::json!({
        "heldBy": activator.current_holder(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn installed(quantization: &str, is_ready: bool) -> InstalledModel {
        InstalledModel {
            id: format!("custom-{quantization}"),
            model_id: "org/custom-orchestrator".to_string(),
            model_name: "Custom Orchestrator".to_string(),
            provider_id: "huggingface".to_string(),
            quantization: quantization.to_string(),
            format: "GGUF".to_string(),
            backend: "llama.cpp (GGUF)".to_string(),
            file_name: "model.gguf".to_string(),
            file_path: "/models/model.gguf".to_string(),
            size_bytes: 1_000,
            installed_at: String::new(),
            is_ready,
            checksum: None,
        }
    }

    #[test]
    fn administrator_selection_resolves_the_exact_ready_variant() {
        let requested = StartupModelTarget::from_installed(&installed("Q6_K", true));
        let selected = resolve_installed_orchestrator(
            &[installed("Q4_K_M", true), installed("Q6_K", true)],
            &requested,
        )
        .expect("the exact requested variant should be selectable");

        assert_eq!(selected, requested);
    }

    #[test]
    fn incomplete_models_cannot_become_the_orchestrator() {
        let model = installed("Q6_K", false);
        let requested = StartupModelTarget::from_installed(&model);
        assert!(resolve_installed_orchestrator(&[model], &requested).is_err());
    }

    /// A registry holding the same package the `installed` helper describes,
    /// registered at each of `quantizations`.
    fn registry_with(quantizations: &[&str]) -> ModelRegistry {
        let entries = quantizations
            .iter()
            .map(|quantization| {
                let mut entry = crate::registry::tests::entry(
                    &format!("custom-{quantization}"),
                    7.0,
                    vec![crate::registry::ModelRole::Reasoning],
                );
                entry.load = Some(crate::registry::LoadSpec {
                    provider_id: "huggingface".into(),
                    model_id: "org/custom-orchestrator".into(),
                    quantization: (*quantization).into(),
                });
                entry
            })
            .collect();
        ModelRegistry::from_manifest(
            crate::registry::ModelManifest { models: entries },
            std::path::PathBuf::from("registry.json"),
        )
        .expect("the manifest is well formed")
    }

    /// The bug, exactly as it was found.
    ///
    /// The package manifest could not read `Q4_K_S` out of the file name and
    /// recorded the container word "GGUF" instead. Persisting that is what made
    /// the setting do nothing: the router compares an administrator's choice
    /// against the registry, which says `Q4_K_S`, so the choice matched no
    /// entry and the chat kept answering from a model nobody had picked — with
    /// the star still showing beside the one they had.
    #[test]
    fn a_placeholder_quantisation_resolves_to_what_the_registry_calls_it() {
        let registry = registry_with(&["Q4_K_S"]);
        let on_disk = StartupModelTarget::from_installed(&installed("GGUF", true));

        let stored = registered_coordinates(&registry, &on_disk).expect("resolvable");

        assert_eq!(stored.quantization, "Q4_K_S", "the registry's spelling wins");
        assert!(
            registry.orchestrator_entry_for(Some(&stored)).is_some(),
            "what is stored has to be what routing can find again"
        );
        // Matching also tolerates the unresolved form, so a choice saved by an
        // earlier build starts being honoured without being re-made. Resolving
        // on the way in is still what keeps a two-variant package unambiguous.
        assert!(
            registry.orchestrator_entry_for(Some(&on_disk)).is_some(),
            "a choice already saved as a placeholder has to start working too"
        );
    }

    #[test]
    fn an_exact_quantisation_is_taken_as_given() {
        let registry = registry_with(&["Q4_K_S", "Q6_K"]);
        let on_disk = StartupModelTarget::from_installed(&installed("Q6_K", true));

        let stored = registered_coordinates(&registry, &on_disk).expect("resolvable");

        assert_eq!(stored.quantization, "Q6_K");
    }

    /// Two variants and nothing to tell them apart is the case the exact
    /// coordinates exist to prevent, so it is refused rather than guessed.
    #[test]
    fn a_placeholder_against_several_variants_is_refused_not_guessed() {
        let registry = registry_with(&["Q4_K_S", "Q6_K"]);
        let on_disk = StartupModelTarget::from_installed(&installed("GGUF", true));

        let error = registered_coordinates(&registry, &on_disk).expect_err("ambiguous");

        assert!(error.contains("Q4_K_S") && error.contains("Q6_K"), "{error}");
    }

    /// Routing can only pick registered models, so storing an unregistered one
    /// would save a preference that can never be honoured.
    #[test]
    fn choosing_a_model_the_registry_does_not_know_is_refused() {
        let registry = registry_with(&[]);
        let on_disk = StartupModelTarget::from_installed(&installed("Q4_K_S", true));

        let error = registered_coordinates(&registry, &on_disk).expect_err("unregistered");

        assert!(error.contains("not in the model registry"), "{error}");
    }
}

/// What a review changed, and what it takes to be in force.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ReviewOutcome {
    pub model_id: String,
    pub permitted_classifications: Vec<Classification>,
    /// True when the running process is still using the pre-review manifest.
    ///
    /// Said rather than hidden. The registry is loaded once at start and held
    /// behind an  with no interior mutability, so a review written now is
    /// read at the next start. An administrator who clears a model and finds an
    /// agent still refusing it deserves to have been told why, rather than
    /// concluding the review did not save.
    pub restart_required: bool,
}

/// Records that an administrator has reviewed a model and what material it may
/// be used on.
///
/// ## Why this exists
///
/// Discovery registers every model it finds with *no* permitted classification
/// — "listed but cleared for no classification until an administrator reviews
/// them". That is the right default: a model nobody has looked at is usable on
/// nothing rather than on everything.
///
/// But nothing in the product could then grant that clearance. The only writes
/// to  outside tests set it to empty, so every
/// binding of an agent to a model was refused with "has not been reviewed for
/// <classification> material", and the agent model-assignment feature was
/// unreachable on any real installation. This is the review step that was
/// missing, not a way around the check.
///
/// ## Why it writes the manifest rather than the loaded registry
///
/// The manifest is the declared record —  says declared
/// entries win over discovered ones — so writing it is what makes a review
/// durable. The in-memory registry is immutable behind its ; changing that
/// is a wider refactor than a review command should carry, and the honest
/// interim is to say a restart is needed rather than to pretend otherwise.
#[tauri::command]
pub async fn registry_review_model(
    model_id: String,
    permitted_classifications: Vec<Classification>,
    session: State<'_, CurrentSession>,
    registry: State<'_, Arc<ModelRegistry>>,
    audit: State<'_, Arc<AuditService>>,
) -> Result<ReviewOutcome, String> {
    // Clearing a model for a classification is a policy decision, so it needs
    // the policy permission — not merely the ability to import a model. The
    // permission check takes the session state, and the refusal it produces is
    // what a direct IPC call from a non-administrator meets.
    require_permission(&session, Permission::ModifyPolicy)?;
    let signed_in = require_session(&session)?;

    if registry.find(&model_id).is_none() {
        return Err(format!(
            "{model_id} is not in the model registry on this machine, so there is nothing to review."
        ));
    }

    let path = registry.manifest_path().to_path_buf();
    let text = std::fs::read_to_string(&path)
        .map_err(|error| format!("the model manifest could not be read: {error}"))?;
    let mut manifest: serde_json::Value = serde_json::from_str(&text)
        .map_err(|error| format!("the model manifest could not be parsed: {error}"))?;

    // Edited as JSON rather than through , so a field this build
    // does not know about survives the round trip instead of being dropped.
    let entries = manifest
        .get_mut("models")
        .and_then(|models| models.as_array_mut())
        .ok_or_else(|| "the model manifest has no models array".to_string())?;
    let entry = entries
        .iter_mut()
        .find(|entry| entry.get("id").and_then(|id| id.as_str()) == Some(model_id.as_str()))
        .ok_or_else(|| format!("{model_id} is not named in the model manifest"))?;
    entry["permittedClassifications"] = serde_json::to_value(&permitted_classifications)
        .map_err(|error| format!("the review could not be encoded: {error}"))?;

    let bytes = serde_json::to_vec_pretty(&manifest)
        .map_err(|error| format!("the model manifest could not be serialised: {error}"))?;
    let temporary = path.with_extension("json.review-tmp");
    std::fs::write(&temporary, &bytes)
        .map_err(|error| format!("the model manifest could not be written: {error}"))?;
    std::fs::rename(&temporary, &path)
        .map_err(|error| format!("the model manifest could not be replaced: {error}"))?;

    let _ = audit.record(
        &signed_in.user.id,
        AuditKind::PolicyDecision,
        format!(
            "reviewed model {model_id}; cleared for {}",
            if permitted_classifications.is_empty() {
                "nothing".to_string()
            } else {
                permitted_classifications
                    .iter()
                    .map(|c| c.label().to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            }
        ),
        Some(serde_json::json!({
            "action": "reviewModel",
            "modelId": model_id,
            "permittedClassifications": permitted_classifications,
        })),
    );

    Ok(ReviewOutcome {
        model_id,
        permitted_classifications,
        restart_required: true,
    })
}
