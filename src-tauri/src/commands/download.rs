//! Models already installed on this machine.
//!
//! ## What this module used to be
//!
//! It was the download surface: `start_model_download`,
//! `pause_model_download`, `resume_model_download`, `cancel_model_download`
//! and `get_active_downloads`, each of which reached the HuggingFace Hub
//! through `DownloadManager`. This build reaches no model catalogue, so they
//! are gone along with the manager, the provider and the token that
//! authenticated them.
//!
//! Weights arrive by a reviewed offline transfer into the model directory and
//! are registered by `registry::scan`. What is left here reads and removes what
//! is already on disk, and touches nothing off this machine.

use tauri::{AppHandle, Manager, State};

use crate::commands::governance::{require_permission, require_session, CurrentSession};
use crate::download_manager::traits::{InstalledModel, StorageSummary};
use crate::identity::Permission;
use crate::model_manager::ModelManager;

#[tauri::command]
pub fn get_installed_models(
    app_handle: AppHandle,
    session: State<'_, CurrentSession>,
) -> Result<Vec<InstalledModel>, String> {
    require_session(&session)?;
    let app_data_dir = app_handle
        .path()
        .app_data_dir()
        .map_err(|e| format!("Failed to resolve AppData directory: {}", e))?;

    Ok(ModelManager::list_installed_models(&app_data_dir))
}

#[tauri::command]
pub fn delete_installed_model(
    app_handle: AppHandle,
    session: State<'_, CurrentSession>,
    provider_id: String,
    model_id: String,
    quantization: String,
) -> Result<(), String> {
    // Deleting a model is the inverse of installing one. Same gate.
    require_permission(&session, Permission::ImportModel)?;

    let app_data_dir = app_handle
        .path()
        .app_data_dir()
        .map_err(|e| format!("Failed to resolve AppData directory: {}", e))?;

    ModelManager::delete_installed_model(&app_data_dir, &provider_id, &model_id, &quantization)
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub fn get_storage_summary(
    app_handle: AppHandle,
    session: State<'_, CurrentSession>,
) -> Result<StorageSummary, String> {
    require_session(&session)?;
    let app_data_dir = app_handle
        .path()
        .app_data_dir()
        .map_err(|e| format!("Failed to resolve AppData directory: {}", e))?;

    Ok(ModelManager::get_storage_summary(&app_data_dir))
}
