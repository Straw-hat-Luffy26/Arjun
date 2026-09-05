//! Rust-owned model identity and the two durable stages of a context transition.
//! A worker can request a boundary, but cannot choose its destination or authority.
use super::{
    events::context::{ContextPhase, StoredContext},
    resume::CheckpointSeed,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ModelContext {
    pub model_id: String,
    pub served_model_id: String,
    pub provider: String,
    pub context_window: u32,
    pub max_tokens: u32,
    pub input: Vec<String>,
}

impl ModelContext {
    pub fn validate(&self) -> Result<(), String> {
        if self.model_id.trim().is_empty()
            || self.served_model_id.trim().is_empty()
            || self.provider.trim().is_empty()
            || self.context_window < 512
            || self.max_tokens > self.context_window.saturating_sub(512)
            || !self.input.iter().any(|s| s == "text")
            || self.input.iter().any(|s| s != "text" && s != "image")
        {
            return Err(
                "The destination model has invalid identity, modalities or context limits.".into(),
            );
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum TransitionStatus {
    Preparing,
    Ready,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelTransition {
    pub schema_version: u32,
    pub transition_id: String,
    pub source_revision: i64,
    pub source_raw_seq: i64,
    pub from: ModelContext,
    pub to: ModelContext,
    pub status: TransitionStatus,
}

/// Validate before any transcript writes. Returning the active source model
/// while preparing keeps a failed/abandoned transition resumable from its source.
pub fn advance(
    previous: Option<&StoredContext>,
    seed: &CheckpointSeed,
    phase: ContextPhase,
    commit_id: &str,
) -> Result<(Option<ModelContext>, Option<ModelTransition>), String> {
    let target = seed.model_context.as_ref();
    if let Some(target) = target {
        target.validate()?;
        if target.model_id != seed.model_id {
            return Err("The destination differs from the routed model.".into());
        }
    }
    let Some(saved) = previous else {
        if matches!(
            phase,
            ContextPhase::ModelTransitionStarted | ContextPhase::ModelTransitionReady
        ) {
            return Err("A model transition requires a source checkpoint.".into());
        }
        return Ok((target.cloned(), None));
    };
    if saved.checkpoint.plan_hash != seed.plan_hash
        || saved.checkpoint.policy_hash != seed.policy_hash
        || saved.checkpoint.workspace_hash != seed.workspace_hash
    {
        return Err(
            "A model transition cannot change the task plan, workspace or authorization.".into(),
        );
    }
    let source = saved.view.model_context.as_ref();
    if source.is_none() && saved.checkpoint.model_id != seed.model_id {
        return Err("This legacy checkpoint has no source model contract; a cross-model transition needs review.".into());
    }
    let changes = source.is_some() && source != target;
    if changes && target.is_none() {
        return Err("The destination model contract is missing.".into());
    }
    let pending = saved
        .view
        .model_transition
        .as_ref()
        .filter(|t| t.status == TransitionStatus::Preparing);
    if pending.is_some_and(|p| Some(&p.to) != target) {
        return Err(
            "Finish the pending model transition before selecting another destination.".into(),
        );
    }
    match phase {
        ContextPhase::ModelTransitionStarted => {
            if !changes {
                return Err(
                    "The destination is already active; no model transition is needed.".into(),
                );
            }
            if let Some(pending) = pending {
                if Some(&pending.to) != target {
                    return Err(
                        "Finish the pending model transition before selecting another destination."
                            .into(),
                    );
                }
                return Ok((source.cloned(), Some(pending.clone())));
            }
            Ok((
                source.cloned(),
                Some(ModelTransition {
                    schema_version: 1,
                    transition_id: commit_id.into(),
                    source_revision: saved.view.revision,
                    source_raw_seq: saved.view.raw_seq,
                    from: source.unwrap().clone(),
                    to: target.unwrap().clone(),
                    status: TransitionStatus::Preparing,
                }),
            ))
        }
        ContextPhase::ModelTransitionReady => {
            let pending =
                pending.ok_or("The destination has no prepared transition checkpoint.")?;
            if Some(&pending.to) != target {
                return Err("The prepared destination changed.".into());
            }
            let mut ready = pending.clone();
            ready.status = TransitionStatus::Ready;
            Ok((target.cloned(), Some(ready)))
        }
        _ if changes => {
            if pending.is_none() || pending.map(|p| &p.to) != target {
                return Err("A model change requires an explicit transition checkpoint.".into());
            }
            if !matches!(
                phase,
                ContextPhase::Observed
                    | ContextPhase::BeforeTool
                    | ContextPhase::AfterTool
                    | ContextPhase::Paused
            ) {
                return Err("The destination context must be admitted before model execution or completion.".into());
            }
            Ok((source.cloned(), pending.cloned()))
        }
        _ => Ok((
            target.cloned().or_else(|| source.cloned()),
            saved.view.model_transition.clone(),
        )),
    }
}
