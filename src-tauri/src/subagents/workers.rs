//! Local specialists backed by real services. No model routing or inference is
//! claimed: these workers perform bounded retrieval, extraction and checks.
use super::{
    ChildResult, ChildTaskPacket, ChildWorker, EffectivePolicy, EvidenceRef, Finding, InputRef,
    SubagentManager,
};
use crate::{
    agent_runtime::{artifacts, retrieval},
    commands::{
        agent::{RunCalculations, RunWorkspaces},
        governance::{require_session, CurrentSession},
    },
    identity::Session,
    knowledge::{KnowledgeIndex, SearchResult},
    orchestrator::{calculation, tools::ToolName},
};
use async_trait::async_trait;
use std::{
    path::{Component, Path, PathBuf},
    sync::Arc,
};

#[derive(Clone)]
pub struct Resources {
    pub index: Arc<KnowledgeIndex>,
    pub events: Arc<crate::agent_runtime::events::TaskEventLog>,
    pub session: CurrentSession,
    pub workspaces: RunWorkspaces,
    pub passages: retrieval::RunPassages,
    pub calculations: RunCalculations,
    pub produced: artifacts::RunArtifacts,
}

pub const PROFILES: [&str; 4] = [
    "knowledge-retriever",
    "document-extractor",
    "calculation-checker",
    "artifact-reviewer",
];

pub fn register(mut manager: SubagentManager, resources: Resources) -> SubagentManager {
    for profile in PROFILES {
        manager = manager.with_worker(Arc::new(Specialist {
            profile,
            resources: resources.clone(),
        }));
    }
    manager
}

struct Specialist {
    profile: &'static str,
    resources: Resources,
}

impl Specialist {
    fn scope(&self, policy: &EffectivePolicy) -> Result<(Session, String), String> {
        let session = require_session(&self.resources.session)?;
        if session.user.id != policy.inherited.user_id
            || session.user.roles != policy.inherited.roles
        {
            return Err("The specialist's signed-in identity or roles changed.".into());
        }
        let run = policy
            .inherited
            .workspace_root
            .file_name()
            .and_then(|s| s.to_str())
            .ok_or("Invalid parent workspace.")?
            .to_string();
        let snapshot = self
            .resources
            .events
            .snapshot(&run)?
            .ok_or("The specialist's parent task is missing.")?;
        if snapshot.actor != session.user.id || snapshot.state.is_terminal() {
            return Err("The specialist's parent task is not active for this operator.".into());
        }
        let classification = snapshot
            .classification
            .as_deref()
            .and_then(|label| {
                crate::policy::Classification::ALL
                    .iter()
                    .copied()
                    .find(|c| c.label() == label)
            })
            .ok_or("The specialist's parent classification is missing.")?;
        if !policy.may_handle(classification) {
            return Err("The specialist cannot handle this parent's classification.".into());
        }
        let workspaces = self
            .resources
            .workspaces
            .lock()
            .map_err(|_| "The workspace table is unavailable.")?;
        let root = workspaces
            .get(&run)
            .ok_or("The parent workspace is unavailable.")?
            .root();
        if root != policy.inherited.workspace_root {
            return Err("The handoff belongs to another workspace.".into());
        }
        Ok((session, run))
    }

    fn path(&self, policy: &EffectivePolicy, relative: &str) -> Result<PathBuf, String> {
        if relative.is_empty()
            || Path::new(relative)
                .components()
                .any(|c| !matches!(c, Component::Normal(_)))
        {
            return Err("A handoff file must be relative to its parent workspace.".into());
        }
        let root = policy
            .inherited
            .workspace_root
            .canonicalize()
            .map_err(|e| e.to_string())?;
        let path = root
            .join(relative)
            .canonicalize()
            .map_err(|e| e.to_string())?;
        if !path.starts_with(&root) || !path.is_file() {
            return Err("The handoff file escaped its parent workspace.".into());
        }
        if std::fs::metadata(&path).map_err(|e| e.to_string())?.len() > 16 * 1024 * 1024 {
            return Err("The handoff file exceeds the specialist's 16 MiB read limit.".into());
        }
        Ok(path)
    }

    fn available(&self, policy: &EffectivePolicy) -> Result<Vec<InputRef>, String> {
        let (session, run) = self.scope(policy)?;
        let mut inputs = Vec::new();
        match self.profile {
            "knowledge-retriever" | "document-extractor" => {
                if !policy.may_call(ToolName::SearchDocuments) {
                    return Err("The specialist has no retrieval permission.".into());
                }
                let passages = self
                    .resources
                    .passages
                    .lock()
                    .map_err(|_| "The evidence table is unavailable.")?;
                for (i, hit) in passages.get(&run).into_iter().flatten().enumerate() {
                    if !policy.may_handle(hit.classification) {
                        continue;
                    }
                    // Re-fetch under current ACLs; an old marker is not a grant.
                    let current = self
                        .resources
                        .index
                        .region(&session, &hit.document_sha256, hit.page, hit.page, 32)
                        .map_err(|e| e.to_string())?;
                    if current
                        .iter()
                        .any(|h| h.chunk_id == hit.chunk_id && policy.may_handle(h.classification))
                    {
                        inputs.push(InputRef::Evidence { marker: i + 1 });
                    }
                }
            }
            "calculation-checker" => {
                if !policy.may_call(ToolName::RunCalculation) {
                    return Err("The specialist has no calculation permission.".into());
                }
                let table = self
                    .resources
                    .calculations
                    .lock()
                    .map_err(|_| "The calculation table is unavailable.")?;
                for calculation in table.get(&run).into_iter().flatten() {
                    let reference = InputRef::Expression {
                        expression: calculation.expression.clone(),
                    };
                    if !inputs.contains(&reference) {
                        inputs.push(reference);
                    }
                }
            }
            "artifact-reviewer" => {
                if !policy.may_call(ToolName::ValidateArtifact)
                    || !policy.may_call(ToolName::ReadScopedFile)
                {
                    return Err(
                        "Artifact review requires validation and scoped-read permissions.".into(),
                    );
                }
                let table = self
                    .resources
                    .produced
                    .lock()
                    .map_err(|_| "The artifact table is unavailable.")?;
                for file in table.get(&run).into_iter().flatten() {
                    let path = Path::new(&file.path)
                        .strip_prefix(&policy.inherited.workspace_root)
                        .map_err(|_| "An artifact belongs to another workspace.")?;
                    inputs.push(InputRef::WorkspaceFile {
                        path: path.to_string_lossy().into_owned(),
                    });
                }
            }
            _ => return Err("Unknown specialist.".into()),
        }
        if inputs.len() > policy.limits.max_turns as usize {
            return Err("The handoff exceeds this specialist's input budget; narrow the parent resources before retrying.".into());
        }
        Ok(inputs)
    }

    fn evidence(&self, run: &str, hits: &[SearchResult]) -> Result<Vec<Finding>, String> {
        let mut table = self
            .resources
            .passages
            .lock()
            .map_err(|_| "The parent evidence table is unavailable.")?;
        let parent = table.entry(run.into()).or_default();
        Ok(hits
            .iter()
            .map(|hit| {
                let marker = match parent.iter().position(|p| p.chunk_id == hit.chunk_id) {
                    Some(i) => i + 1,
                    None => {
                        parent.push(hit.clone());
                        parent.len()
                    }
                };
                Finding {
                    statement: format!(
                        "Authorized excerpt: {}",
                        hit.text.chars().take(320).collect::<String>()
                    ),
                    evidence: vec![EvidenceRef {
                        marker: Some(marker),
                        document_sha256: hit.document_sha256.clone(),
                        page: Some(hit.page),
                        citation: hit.citation(),
                    }],
                }
            })
            .collect())
    }
}

#[cfg(test)]
#[path = "workers_tests.rs"]
mod tests;

#[async_trait]
impl ChildWorker for Specialist {
    fn profile(&self) -> &str {
        self.profile
    }
    fn handoff_inputs(&self, policy: &EffectivePolicy) -> Result<Vec<InputRef>, String> {
        self.available(policy)
    }
    fn validate_inputs(&self, policy: &EffectivePolicy, inputs: &[InputRef]) -> Result<(), String> {
        let authorized = self.available(policy)?;
        if inputs.len() > policy.limits.max_turns as usize
            || inputs.iter().any(|i| !authorized.contains(i))
        {
            return Err(
                "The handoff contains a reference outside this parent's authorized resources."
                    .into(),
            );
        }
        Ok(())
    }

    fn result_evidence(
        &self,
        policy: &EffectivePolicy,
        result: &ChildResult,
    ) -> Result<Vec<SearchResult>, String> {
        let (_, run) = self.scope(policy)?;
        let parent = retrieval::for_run(&self.resources.passages, &run);
        result
            .findings
            .iter()
            .flat_map(|f| &f.evidence)
            .map(|reference| {
                parent
                    .get(
                        reference
                            .marker
                            .ok_or("A specialist evidence marker is missing.")?
                            .saturating_sub(1),
                    )
                    .cloned()
                    .ok_or_else(|| "The specialist evidence was not recorded in its parent.".into())
            })
            .collect()
    }

    fn restore_evidence(
        &self,
        policy: &EffectivePolicy,
        record: &crate::agent_runtime::events::children::ChildRecord,
    ) -> Result<(), String> {
        let (session, run) = self.scope(policy)?;
        let references = record
            .result
            .findings
            .iter()
            .flat_map(|f| &f.evidence)
            .collect::<Vec<_>>();
        if references.len() != record.evidence.len() {
            return Err("The saved specialist evidence is incomplete.".into());
        }
        for (reference, saved) in references.iter().zip(&record.evidence) {
            let hits = self
                .resources
                .index
                .region(&session, &saved.document_sha256, saved.page, saved.page, 32)
                .map_err(|e| e.to_string())?;
            if !hits.iter().any(|h| {
                h.chunk_id == saved.chunk_id
                    && h.text == saved.text
                    && policy.may_handle(h.classification)
            }) {
                return Err(
                    "The specialist's saved evidence is no longer authorized or current.".into(),
                );
            }
            if reference.document_sha256 != saved.document_sha256
                || reference.page != Some(saved.page)
            {
                return Err("The specialist citation differs from its source.".into());
            }
        }
        let mut table = self
            .resources
            .passages
            .lock()
            .map_err(|_| "The parent evidence table is unavailable.")?;
        let mut restored = table.get(&run).cloned().unwrap_or_default();
        for (reference, saved) in references.iter().zip(&record.evidence) {
            let marker = reference
                .marker
                .ok_or("The specialist marker is missing.")?;
            if marker == restored.len() + 1 {
                restored.push(saved.clone());
            } else if marker == 0
                || restored
                    .get(marker - 1)
                    .is_none_or(|p| p.chunk_id != saved.chunk_id || p.text != saved.text)
            {
                return Err(
                    "Restoring the child would change an existing parent citation marker.".into(),
                );
            }
        }
        table.insert(run, restored);
        Ok(())
    }

    async fn run(
        &self,
        packet: &ChildTaskPacket,
        policy: &EffectivePolicy,
    ) -> Result<ChildResult, String> {
        self.validate_inputs(policy, &packet.inputs)?;
        let (session, run) = self.scope(policy)?;
        if run != packet.parent_run_id || packet.policy_hash != policy.inherited_hash {
            return Err("The specialist packet belongs to another run or policy.".into());
        }
        let mut findings = Vec::new();
        let mut uncertainty = Vec::new();
        match self.profile {
            "knowledge-retriever" => {
                let hits = self
                    .resources
                    .index
                    .search(&session, &packet.objective, 4)
                    .map_err(|e| e.to_string())?
                    .into_iter()
                    .filter(|h| policy.may_handle(h.classification))
                    .collect::<Vec<_>>();
                findings = self.evidence(&run, &hits)?;
                uncertainty.push("Bounded keyword retrieval; excerpts are source data, not a verified answer or an exhaustive search.".into());
            }
            "document-extractor" => {
                if packet.inputs.is_empty() {
                    return Err(
                        "Document extraction needs authorized parent evidence references.".into(),
                    );
                }
                let parent = retrieval::for_run(&self.resources.passages, &run);
                for reference in &packet.inputs {
                    let InputRef::Evidence { marker } = reference else {
                        return Err("Extraction requires evidence markers.".into());
                    };
                    let hit = parent
                        .get(marker.saturating_sub(1))
                        .ok_or("The parent evidence marker is missing.")?;
                    let hits = self
                        .resources
                        .index
                        .region(&session, &hit.document_sha256, hit.page, hit.page, 32)
                        .map_err(|e| e.to_string())?;
                    let hit = hits
                        .into_iter()
                        .find(|h| h.chunk_id == hit.chunk_id && policy.may_handle(h.classification))
                        .ok_or("The source is no longer authorized or indexed.")?;
                    findings.extend(self.evidence(&run, &[hit])?);
                }
                uncertainty.push("Extracted indexed text excerpts only; unindexed scans, visual layout and omitted text were not interpreted.".into());
            }
            "calculation-checker" => {
                if packet.inputs.is_empty() {
                    return Err(
                        "Calculation checking needs expressions actually calculated by the parent."
                            .into(),
                    );
                }
                let table = self
                    .resources
                    .calculations
                    .lock()
                    .map_err(|_| "The calculation table is unavailable.")?;
                for reference in &packet.inputs {
                    let InputRef::Expression { expression } = reference else {
                        return Err("Calculation checking requires expression references.".into());
                    };
                    let checked =
                        calculation::evaluate(expression).map_err(|e| format!("{e:?}"))?;
                    let originals = table
                        .get(&run)
                        .ok_or("The parent calculations are missing.")?;
                    if originals
                        .iter()
                        .filter(|c| c.expression == *expression)
                        .any(|c| {
                            c.value != checked.value
                                || c.unit != checked.unit
                                || c.formatted != checked.formatted
                                || !c.deterministic
                        })
                    {
                        return Err(format!(
                            "The saved calculation does not match recomputation: {expression}"
                        ));
                    }
                    findings.push(Finding {
                        statement: format!(
                            "Recomputed {expression} = {} with matching saved value and units.",
                            checked.formatted
                        ),
                        evidence: Vec::new(),
                    });
                }
                uncertainty.push("Arithmetic and unit consistency checked; source measurements and engineering suitability were not verified.".into());
            }
            "artifact-reviewer" => {
                if packet.inputs.is_empty() {
                    return Err(
                        "Artifact review needs files actually produced by the parent.".into(),
                    );
                }
                let produced = artifacts::for_run(&self.resources.produced, &run);
                for reference in &packet.inputs {
                    let InputRef::WorkspaceFile { path } = reference else {
                        return Err("Artifact review requires workspace references.".into());
                    };
                    let resolved = self.path(policy, path)?;
                    let source = produced
                        .iter()
                        .find(|p| {
                            Path::new(&p.path).canonicalize().ok().as_ref() == Some(&resolved)
                        })
                        .ok_or("The file is not a parent artifact.")?;
                    let mut source = source.clone();
                    source.path = resolved.to_string_lossy().into_owned();
                    let report = artifacts::check(&source);
                    if !report.sound {
                        return Err(format!(
                            "{} failed re-opening: {}",
                            source.name, report.detail
                        ));
                    }
                    findings.push(Finding {
                        statement: format!(
                            "{}: {} ({} bytes)",
                            source.name, report.detail, report.bytes
                        ),
                        evidence: Vec::new(),
                    });
                }
                uncertainty.push("File structure and declared template checks only; factual correctness still requires parent answer verification.".into());
            }
            _ => return Err("Unknown specialist.".into()),
        }
        self.scope(policy)?;
        Ok(ChildResult::completed(
            &packet.child_id,
            self.profile,
            packet.required_schema,
            findings,
            1.0,
            uncertainty,
            packet.inputs.len().max(1) as u32,
        ))
    }
}
