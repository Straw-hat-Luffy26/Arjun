//! Phase 3: Model Recommendation Engine
//!
//! Takes Phase 2's HardwareProfile and deterministically calculates which
//! LLMs the user's machine can realistically run.
//!
//! Architecture:
//!   HardwareProfile → Budget Calculator → Model Catalog → Memory Estimator
//!   → Multi-Config Evaluator → Deterministic Scorer → Ranked Recommendations
//!
//! This module does NOT download models or launch inference backends.

pub mod traits;
pub mod budget;
pub mod catalog;
pub mod estimator;
pub mod runtime;
pub mod scorer;
pub mod certified_catalog;
pub mod pack_manager;
pub mod runtime_validator;

use std::path::Path;
use crate::system_analyzer::traits::HardwareProfile;
use traits::*;

/// Generate model recommendations from a HardwareProfile and dynamic Hugging Face catalog.
/// This is the main entry point for the recommendation engine.
pub async fn generate_recommendations(
    profile: &HardwareProfile,
    app_data_dir: Option<&Path>,
    force_refresh: bool,
) -> Vec<ModelRecommendation> {
    log::info!("[RECOMMENDATION] 🚀 Starting Model Recommendation Engine");

    // 1. Calculate adaptive resource budget
    let budget_config = BudgetConfig::default();
    let memory_budget = budget::calculate_budget(profile, &budget_config);

    // 2. The catalogue this build ships with.
    //
    // This used to call `HuggingFaceCatalogProvider::fetch_catalog`, which
    // reached the Hub for a live listing and fell back to this same bootstrap
    // list when it could not. The Hub is gone from this build, so the fallback
    // is now the only path — recommendations are made from what is checked in,
    // and `force_refresh` has nothing left to refresh.
    let _ = (app_data_dir, force_refresh);
    let models = catalog::bootstrap_models();
    log::info!("[RECOMMENDATION] Discovered/loaded {} models from catalog", models.len());

    // 3. Evaluate all models against budget and score
    let estimator_config = EstimatorConfig::default();
    let mut recommendations = scorer::generate_all_recommendations(
        &models,
        &memory_budget,
        &estimator_config,
    );

    // 4. Enrich recommendations with certification metadata from PackManager
    if let Some(data_dir) = app_data_dir {
        if let Ok(pack_mgr) = pack_manager::PackManager::new(data_dir) {
            for rec in &mut recommendations {
                rec.certification = pm_cert_lookup(&pack_mgr, &rec.model_id);
            }
        }
    } else {
        let temp_dir = std::env::temp_dir().join("sarathi_pack_tmp");
        if let Ok(pack_mgr) = pack_manager::PackManager::new(&temp_dir) {
            for rec in &mut recommendations {
                rec.certification = pm_cert_lookup(&pack_mgr, &rec.model_id);
            }
        }
    }

    log::info!("[RECOMMENDATION] ✓ Engine complete: {} recommendations generated", recommendations.len());
    recommendations
}

fn pm_cert_lookup(pack_mgr: &pack_manager::PackManager, model_id: &str) -> Option<certified_catalog::PackageCertification> {
    pack_mgr.get_package_certification(model_id)
}
