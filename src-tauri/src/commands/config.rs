//! Configuration commands

use std::collections::HashMap;
use tauri::AppHandle;
use tauri::Manager;
use tauri::State;
use std::path::PathBuf;

use crate::commands::governance::{require_permission, CurrentSession};
use crate::config::{ConfigManager, SarathiConfig};
use crate::core::event_bus::{get_event_bus, SarathiEvent};
use crate::identity::Permission;

/// Gets the entire configuration as JSON string
#[tauri::command]
pub async fn get_config(app: AppHandle) -> Result<String, String> {
    let path = ConfigManager::get_config_path(&app);
    let config = ConfigManager::load(&path).map_err(|e| e.to_string())?;
    serde_json::to_string(&config).map_err(|e| e.to_string())
}

/// Sets the entire configuration from JSON string. Administrator only —
/// a misconfigured ARJUN instance is broken for everyone, so this is
/// gated on `ModifyPolicy`.
#[tauri::command]
pub async fn set_config(
    app: AppHandle,
    config: String,
    session: State<'_, CurrentSession>,
) -> Result<(), String> {
    require_permission(&session, Permission::ModifyPolicy)?;
    let new_config: SarathiConfig = serde_json::from_str(&config).map_err(|e| e.to_string())?;
    let path = ConfigManager::get_config_path(&app);

    ConfigManager::save(&new_config, &path).map_err(|e| e.to_string())?;

    // Publish config changed event
    let event_bus = get_event_bus();
    if let Ok(value) = serde_json::to_value(&new_config) {
        event_bus.publish(SarathiEvent::ConfigChanged, Some(value));
    }

    Ok(())
}


/// Gets a specific config value by key (simplified implementation)
#[tauri::command]
pub async fn get_config_value(app: AppHandle, key: String) -> Result<Option<String>, String> {
    let config_json = get_config(app).await?;
    let config_val: serde_json::Value = serde_json::from_str(&config_json).map_err(|e| e.to_string())?;
    
    if let Some(val) = config_val.get(&key) {
        Ok(Some(val.to_string()))
    } else {
        Ok(None)
    }
}

/// Sets a specific config value (simplified implementation)
#[tauri::command]
pub async fn set_config_value(
    app: AppHandle,
    key: String,
    value: String,
    session: State<'_, CurrentSession>,
) -> Result<(), String> {
    let config_json = get_config(app.clone()).await?;
    let mut config_val: serde_json::Value = serde_json::from_str(&config_json).map_err(|e| e.to_string())?;

    // Parse value if it's JSON, otherwise treat as string
    let parsed_val = serde_json::from_str(&value).unwrap_or(serde_json::Value::String(value));

    if let Some(obj) = config_val.as_object_mut() {
        obj.insert(key, parsed_val);
    }

    let updated_config_str = serde_json::to_string(&config_val).map_err(|e| e.to_string())?;
    set_config(app, updated_config_str, session).await
}

/// Returns the default config map
#[tauri::command]
pub async fn get_default_config() -> Result<String, String> {
    let config = SarathiConfig::default();
    serde_json::to_string(&config).map_err(|e| e.to_string())
}

/// Resets configuration to defaults
#[tauri::command]
pub async fn reset_config(
    app: AppHandle,
    session: State<'_, CurrentSession>,
) -> Result<(), String> {
    let config = SarathiConfig::default();
    let config_str = serde_json::to_string(&config).map_err(|e| e.to_string())?;
    set_config(app, config_str, session).await
}

/// Returns app paths
#[tauri::command]
pub async fn get_app_paths(app: AppHandle) -> Result<HashMap<String, String>, String> {
    let mut paths = HashMap::new();
    
    if let Ok(data_dir) = app.path().app_data_dir() {
        paths.insert("data_dir".to_string(), data_dir.to_string_lossy().to_string());
    }
    
    if let Ok(log_dir) = app.path().app_log_dir() {
        paths.insert("log_dir".to_string(), log_dir.to_string_lossy().to_string());
    }
    
    let config_path = ConfigManager::get_config_path(&app);
    paths.insert("config_path".to_string(), config_path.to_string_lossy().to_string());
    
    Ok(paths)
}
