//! The Document & Vision Analyst's administration (P06).
//!
//! Three commands, each answering one question an operator has:
//!
//! - `document_analyst_status` — what reads pages on this machine right now:
//!   the OCR model and its exact files, the page analyser, and for every model
//!   with an image projector whether it is vision-ready and why not.
//! - `registry_bind_projector` — give a model an image projector, by any
//!   filename, verified from both files' headers. In force at the next start.
//! - `vision_probe_model` — show a model a fresh random token in an image and
//!   record whether it read it. Only a pass makes a model vision-ready.
//!
//! The two that change something need the import permission and are audited.

use std::path::PathBuf;
use std::sync::Arc;

use serde::Serialize;
use tauri::State;

use crate::audit::{AuditKind, AuditService};
use crate::commands::agent::ExtractionState;
use crate::commands::governance::{require_permission, require_session, CurrentSession};
use crate::extraction::ocr::OcrIdentity;
use crate::extraction::projector::ProjectorBinding;
use crate::extraction::vision::ReadinessRecord;
use crate::identity::Permission;
use crate::registry::ModelRegistry;
use crate::serving::ModelServers;
use crate::subagents::ModelScheduler;

/// One model that could see, and whether it can.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VisionCandidate {
    pub model_id: String,
    pub name: String,
    pub projector: Option<String>,
    pub ready: bool,
    /// Why not, when it is not.
    pub reason: Option<String>,
    /// The latest probe, pass or fail.
    pub last_probe: Option<ReadinessRecord>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DocumentAnalystStatus {
    /// The OCR model the analyst reads scans with, exactly, or why there is none.
    pub ocr: Option<OcrIdentity>,
    pub ocr_unavailable: Option<String>,
    pub ocr_transport: String,
    /// The page analyser (PyMuPDF through the page rasteriser), or why not.
    pub page_analyser: Option<String>,
    pub page_analyser_unavailable: Option<String>,
    pub vision: Vec<VisionCandidate>,
    /// The model interpretation would use now, if any.
    pub interpreter: Option<String>,
    pub projector_bindings: Vec<ProjectorBinding>,
}

/// What reads pages on this machine.
#[tauri::command]
pub async fn document_analyst_status(
    session: State<'_, CurrentSession>,
    registry: State<'_, Arc<ModelRegistry>>,
    extraction: State<'_, ExtractionState>,
) -> Result<DocumentAnalystStatus, String> {
    require_session(&session)?;
    let service = &extraction.0;
    let detent = crate::agent_runtime::extraction_tools::ANALYST_DETENT;
    let (ocr, ocr_unavailable) = match service.ocr_service().identity(detent) {
        Ok(identity) => (Some(identity), None),
        Err(why) => (None, Some(why)),
    };
    let (page_analyser, page_analyser_unavailable) = match service.sidecar() {
        Ok(sidecar) => (
            Some(format!("{} {}", sidecar.python, sidecar.script.display())),
            None,
        ),
        Err(why) => (None, Some(why)),
    };
    let models_dir = registry.models_dir();
    let records = crate::extraction::vision::load(models_dir);
    let vision = registry
        .all()
        .iter()
        .filter(|entry| entry.projector.is_some())
        .map(|entry| {
            let ready = crate::extraction::vision::readiness(models_dir, entry);
            VisionCandidate {
                model_id: entry.id.clone(),
                name: entry.name.clone(),
                projector: entry.projector.as_ref().map(|p| p.display().to_string()),
                ready: ready.is_ok(),
                reason: ready.err(),
                last_probe: records.iter().rev().find(|r| r.model_id == entry.id).cloned(),
            }
        })
        .collect();
    Ok(DocumentAnalystStatus {
        ocr,
        ocr_unavailable,
        ocr_transport: service.ocr_service().transport().to_string(),
        page_analyser,
        page_analyser_unavailable,
        vision,
        interpreter: service.vision_service().ready_model().ok().map(|m| m.model_id),
        projector_bindings: crate::extraction::projector::load(models_dir).unwrap_or_default(),
    })
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ProjectorBindOutcome {
    pub binding: ProjectorBinding,
    /// The registry is loaded once at start; the binding is read at the next.
    pub restart_required: bool,
    /// A bound projector is not a model that can see. That takes a probe.
    pub vision_ready: bool,
}

/// Binds an image projector to a model, after checking both headers.
#[tauri::command]
pub async fn registry_bind_projector(
    model_id: String,
    projector_path: String,
    session: State<'_, CurrentSession>,
    registry: State<'_, Arc<ModelRegistry>>,
    audit: State<'_, Arc<AuditService>>,
) -> Result<ProjectorBindOutcome, String> {
    require_permission(&session, Permission::ImportModel)?;
    let signed_in = require_session(&session)?;
    let entry = registry
        .find(&model_id)
        .cloned()
        .ok_or_else(|| format!("{model_id} is not in the model registry on this machine."))?;
    let models_dir = registry.models_dir().to_path_buf();
    let projector = PathBuf::from(projector_path.trim());
    let actor = signed_in.user.id.clone();
    let binding = tokio::task::spawn_blocking(move || {
        crate::extraction::projector::bind(&models_dir, &entry, &projector, &actor)
    })
    .await
    .map_err(|error| format!("the projector check did not finish: {error}"))??;

    let _ = audit.record(
        &signed_in.user.id,
        AuditKind::PolicyDecision,
        format!(
            "bound image projector {} to model {model_id} (verified: projection width {} = model \
             embedding width)",
            binding.projector.display(),
            binding.projection_dim
        ),
        Some(serde_json::json!({
            "action": "bindProjector",
            "modelId": model_id,
            "projector": binding.projector,
            "projectorBytes": binding.projector_bytes,
            "projectorType": binding.projector_type,
            "projectionDim": binding.projection_dim,
        })),
    );
    Ok(ProjectorBindOutcome {
        binding,
        restart_required: true,
        vision_ready: false,
    })
}

/// Shows a model a random token in an image, and records whether it read it.
#[tauri::command]
pub async fn vision_probe_model(
    model_id: String,
    session: State<'_, CurrentSession>,
    registry: State<'_, Arc<ModelRegistry>>,
    servers: State<'_, Arc<ModelServers>>,
    scheduler: State<'_, Arc<ModelScheduler>>,
    extraction: State<'_, ExtractionState>,
    audit: State<'_, Arc<AuditService>>,
) -> Result<ReadinessRecord, String> {
    require_permission(&session, Permission::ImportModel)?;
    let signed_in = require_session(&session)?;
    let sidecar = extraction.0.sidecar()?.clone();
    let work_dir = crate::extraction::vision::probe_dir(extraction.0.documents_root());
    let attempt = crate::extraction::vision::probe_registered(
        registry.inner(),
        servers.inner(),
        scheduler.inner(),
        &model_id,
        &sidecar,
        &work_dir,
        &signed_in.user.id,
    )
    .await?;
    let _ = audit.record(
        &signed_in.user.id,
        AuditKind::PolicyDecision,
        format!(
            "image probe of {model_id}: {} ({})",
            if attempt.passed { "passed" } else { "failed" },
            attempt.reason
        ),
        Some(serde_json::json!({
            "action": "visionProbe",
            "modelId": model_id,
            "passed": attempt.passed,
            "projectorSha256": attempt.projector_sha256,
            "at": attempt.at,
        })),
    );
    Ok(attempt)
}
