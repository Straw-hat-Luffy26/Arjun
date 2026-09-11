//! Default configuration definitions

use serde::{Deserialize, Serialize};

/// Main configuration for Sarathi
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SarathiConfig {
    pub theme: String,
    pub language: String,
    pub backend_url: String,
    pub ollama_url: String,
    pub model_directory: String,
    pub download_directory: String,
    pub cache_directory: String,
    pub log_level: String,
    pub ai_settings: AiSettings,
}

/// AI-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AiSettings {
    pub max_context_length: u32,
    pub default_temperature: f32,
    pub use_gpu: bool,
    pub gpu_layers: u32,

    /// Exact installed package coordinates selected by an administrator for
    /// the agent orchestrator.
    ///
    /// **No model is compiled in as the product default.** Empty means "nobody
    /// has chosen one yet", and that is the shipping state: what is installed
    /// on a machine is discovered at runtime, so naming a model here would only
    /// pin the product to a package the user may never have downloaded. An
    /// administrator picks one in Models → *Set as orchestrator*; until then
    /// startup restores the last session and the router picks per prompt from
    /// whatever is actually installed.
    #[serde(default)]
    pub orchestrator_provider_id: String,
    #[serde(default)]
    pub orchestrator_model_id: String,
    #[serde(default)]
    pub orchestrator_quantization: String,

    /// Whether startup may load a model without being asked.
    ///
    /// Enabled by default so the orchestrator is ready without a manual load.
    /// Existing configurations that explicitly set this to false stay disabled.
    #[serde(default = "default_true")]
    pub auto_load_on_startup: bool,
}

impl AiSettings {
    /// True when an administrator has chosen the orchestrator.
    ///
    /// All three coordinates are required: a provider and model without a
    /// quantization names two different files on disk, and startup must never
    /// guess between two variants of the same model.
    pub fn has_configured_orchestrator(&self) -> bool {
        !self.orchestrator_provider_id.trim().is_empty()
            && !self.orchestrator_model_id.trim().is_empty()
            && !self.orchestrator_quantization.trim().is_empty()
    }
}

fn default_true() -> bool {
    true
}

impl Default for AiSettings {
    fn default() -> Self {
        Self {
            max_context_length: 4096,
            default_temperature: 0.7,
            use_gpu: true,
            gpu_layers: 35,
            orchestrator_provider_id: String::new(),
            orchestrator_model_id: String::new(),
            orchestrator_quantization: String::new(),
            auto_load_on_startup: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_model_is_hardcoded_as_the_orchestrator() {
        let settings = AiSettings::default();
        assert!(settings.orchestrator_provider_id.is_empty());
        assert!(settings.orchestrator_model_id.is_empty());
        assert!(settings.orchestrator_quantization.is_empty());
        assert!(!settings.has_configured_orchestrator());
        // Auto-load stays on: with nothing configured it restores the last
        // session rather than loading a model nobody asked for.
        assert!(settings.auto_load_on_startup);
        assert!(settings.use_gpu);
        assert!(settings.gpu_layers > 0);
    }

    #[test]
    fn an_administrator_choice_is_a_complete_set_of_coordinates() {
        let mut settings = AiSettings::default();
        settings.orchestrator_provider_id = "huggingface".to_string();
        settings.orchestrator_model_id = "nvidia/Nemotron3-Nano-4B".to_string();
        assert!(
            !settings.has_configured_orchestrator(),
            "a model without a quantization names more than one file on disk"
        );
        settings.orchestrator_quantization = "Q4_K_M".to_string();
        assert!(settings.has_configured_orchestrator());
    }

    #[test]
    fn older_config_without_startup_fields_inherits_the_new_defaults() {
        let settings: AiSettings = serde_json::from_str(
            r#"{
                "max_context_length": 4096,
                "default_temperature": 0.7,
                "use_gpu": true,
                "gpu_layers": 35
            }"#,
        )
        .expect("legacy AI settings should still deserialize");

        assert!(!settings.has_configured_orchestrator());
        assert!(settings.auto_load_on_startup);
    }

    #[test]
    fn a_saved_orchestrator_choice_survives_a_round_trip() {
        let settings: AiSettings = serde_json::from_str(
            r#"{
                "max_context_length": 4096,
                "default_temperature": 0.7,
                "use_gpu": true,
                "gpu_layers": 35,
                "orchestrator_provider_id": "huggingface",
                "orchestrator_model_id": "nvidia/Nemotron3-Nano-4B",
                "orchestrator_quantization": "Q4_K_M"
            }"#,
        )
        .expect("a configured orchestrator should deserialize");

        assert!(settings.has_configured_orchestrator());
        assert_eq!(settings.orchestrator_model_id, "nvidia/Nemotron3-Nano-4B");
    }
}

impl Default for SarathiConfig {
    fn default() -> Self {
        Self {
            theme: "dark".to_string(),
            language: "en".to_string(),
            backend_url: "http://localhost:8000".to_string(),
            ollama_url: "http://localhost:11434".to_string(),
            model_directory: "models".to_string(),
            download_directory: "downloads".to_string(),
            cache_directory: "cache".to_string(),
            log_level: "info".to_string(),
            ai_settings: AiSettings::default(),
        }
    }
}
