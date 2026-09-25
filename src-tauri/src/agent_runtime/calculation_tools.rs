//! The calculation tools on the agent path (P08), and what they leave behind.
//!
//! Each call runs the deterministic engine (`crate::calculation`), keeps the
//! immutable record in the owner's calculation store, projects an evaluation
//! into the run's table (what the workbook and the verifier have always read),
//! and — after the call's receipt is written — publishes the record into the
//! task's shared memory, resting on the memory items its inputs cite. A
//! correction to one of those inputs then reaches every figure built on it
//! through the graph's own staleness, and from there every artifact that
//! cites the figure.
//!
//! [`Sources`] is how a checker goes back to where each input came from: the
//! memory graph, the run's evidence table, the knowledge index and the store
//! itself, all under the reader's clearance.

use std::sync::Arc;

use serde_json::Value;

use crate::calculation::{
    self, CalcInput, CalcRecord, InputSource, Operation, Options, RecordStatus, SourceReader, SourceState,
};
use crate::calculation::decimal::Rounding;
use crate::calculation::store::{CalculationStore, Stored};
use crate::identity::Session;
use crate::knowledge::graph::runtime_memory::{
    edge_id, Authority, Dependency, EdgeKind, ItemStatus, MemoryEdge, MemoryItem, MemoryKind, MemoryScope, Provenance,
    SourceRef,
};
use crate::knowledge::graph::runtime_store::MemoryGraph;
use crate::knowledge::index::{KnowledgeIndex, SearchResult};
use crate::orchestrator::calculation::CalculationRecord;
use crate::orchestrator::tools::{ToolCall, ToolName};

use super::{CallParams, RuntimeDeps};

/// What a calculation call hands back: the text the model reads, and the
/// record to publish once the call's receipt exists.
pub type Outcome = (Result<String, String>, Option<CalcRecord>);

fn inputs_of(tool_call: &ToolCall) -> Result<Vec<CalcInput>, String> {
    match tool_call.arguments.get("inputs") {
        None | Some(Value::Null) => Ok(Vec::new()),
        Some(Value::Array(items)) => calculation::read_inputs(items).map_err(|e| format!("{}. Nothing was calculated.", e.describe())),
        Some(_) => Err("`inputs` is a list of lines such as \"t_min = 9.0 mm [M:mi-…#r1]\". Nothing was calculated.".into()),
    }
}

fn options_of(tool_call: &ToolCall) -> Result<Options, String> {
    let rounding = match tool_call.text("round").map(str::trim).filter(|r| !r.is_empty()) {
        None => None,
        Some(text) => Some(Rounding::parse(text).ok_or_else(|| {
            format!("`round` is written like \"4sf\" or \"2dp\" (1 to 12 figures, 0 to 12 places), not {text:?}. Nothing was calculated.")
        })?),
    };
    Ok(Options {
        result_unit: tool_call.text("resultUnit").map(str::trim).filter(|u| !u.is_empty()).map(str::to_string),
        rounding,
    })
}

fn strings(tool_call: &ToolCall, key: &str) -> Vec<String> {
    tool_call
        .arguments
        .get(key)
        .and_then(Value::as_array)
        .map(|items| items.iter().filter_map(|v| v.as_str().map(str::to_string)).collect())
        .unwrap_or_default()
}

/// Keeps the record, and says what the model should do with it.
fn finish(deps: &Arc<RuntimeDeps>, call: &CallParams, session: &Session, record: CalcRecord) -> Outcome {
    let owner = &session.user.id;
    let mut note = String::new();
    match deps.calculation_store.put(&record, owner, &call.run_id) {
        Ok(Stored::Differs) => {
            note = format!("\n(The store already holds {} with different content; the first record stands.)", record.id)
        }
        Ok(_) => {}
        Err(problem) => note = format!("\n(The record could not be kept: {problem}. Do not cite it.)"),
    }
    // The run's table: what the workbook writes and the verifier reconciles.
    if record.operation == Operation::Evaluate {
        if let Some(projection) = CalculationRecord::from_record(&record) {
            if let Ok(mut table) = deps.calculations.lock() {
                let held = table.entry(call.run_id.clone()).or_default();
                if !held.iter().any(|r| !r.id.is_empty() && r.id == projection.id) {
                    held.push(projection);
                }
            }
        }
    }
    let text = format!("{}{note}", record.render());
    let outcome = match record.status {
        // Returned as a failed call, so the model corrects the input rather
        // than quoting a result there is not.
        RecordStatus::Refused => Err(text),
        _ => Ok(text),
    };
    (outcome, Some(record))
}

fn resolver<'a>(store: &'a CalculationStore, owner: &'a str) -> impl Fn(&str) -> Option<CalcRecord> + 'a {
    move |id: &str| store.get(id, owner)
}

pub fn evaluate(deps: &Arc<RuntimeDeps>, call: &CallParams, session: &Session, tool_call: &ToolCall) -> Outcome {
    let (inputs, options) = match (inputs_of(tool_call), options_of(tool_call)) {
        (Ok(i), Ok(o)) => (i, o),
        (Err(e), _) | (_, Err(e)) => return (Err(e), None),
    };
    let find = resolver(&deps.calculation_store, &session.user.id);
    let record = calculation::evaluate(tool_call.text("expression").unwrap_or_default(), inputs, &options, Some(&find));
    finish(deps, call, session, record)
}

pub fn validate_dimensions(deps: &Arc<RuntimeDeps>, call: &CallParams, session: &Session, tool_call: &ToolCall) -> Outcome {
    let inputs = match inputs_of(tool_call) {
        Ok(i) => i,
        Err(e) => return (Err(e), None),
    };
    let record = calculation::validate_dimensions(tool_call.text("expression").unwrap_or_default(), inputs, None);
    finish(deps, call, session, record)
}

pub fn solve(deps: &Arc<RuntimeDeps>, call: &CallParams, session: &Session, tool_call: &ToolCall) -> Outcome {
    let (inputs, options) = match (inputs_of(tool_call), options_of(tool_call)) {
        (Ok(i), Ok(o)) => (i, o),
        (Err(e), _) | (_, Err(e)) => return (Err(e), None),
    };
    let find = resolver(&deps.calculation_store, &session.user.id);
    let record = calculation::solve(&strings(tool_call, "equations"), &strings(tool_call, "unknowns"), inputs, &options, Some(&find));
    finish(deps, call, session, record)
}

pub fn compare(deps: &Arc<RuntimeDeps>, call: &CallParams, session: &Session, tool_call: &ToolCall) -> Outcome {
    let inputs = match inputs_of(tool_call) {
        Ok(i) => i,
        Err(e) => return (Err(e), None),
    };
    let find = resolver(&deps.calculation_store, &session.user.id);
    let record = calculation::compare(
        tool_call.text("value").unwrap_or_default(),
        tool_call.text("relation").unwrap_or_default(),
        tool_call.text("limit").unwrap_or_default(),
        tool_call.text("tolerance"),
        inputs,
        &Options::default(),
        Some(&find),
    );
    finish(deps, call, session, record)
}

pub fn sensitivity(deps: &Arc<RuntimeDeps>, call: &CallParams, session: &Session, tool_call: &ToolCall) -> Outcome {
    let inputs = match inputs_of(tool_call) {
        Ok(i) => i,
        Err(e) => return (Err(e), None),
    };
    let find = resolver(&deps.calculation_store, &session.user.id);
    let record = calculation::sensitivity(
        tool_call.text("expression").filter(|e| !e.trim().is_empty()),
        tool_call.text("calculationId").filter(|c| !c.trim().is_empty()),
        inputs,
        &Options::default(),
        Some(&find),
    );
    finish(deps, call, session, record)
}

// ── Publication ──────────────────────────────────────────────────────────

/// The memory item a record is published as, for one owner: stable, so a
/// second run computing the same record finds the first publication.
pub fn item_id_for(calc_id: &str, owner: &str) -> String {
    use sha2::{Digest, Sha256};
    let digest = Sha256::digest(owner.as_bytes());
    format!("mi-{calc_id}-{}", digest[..4].iter().map(|b| format!("{b:02x}")).collect::<String>())
}

/// What a record rests on in the graph, and the sources it cites.
pub(crate) fn lineage(
    graph: &MemoryGraph,
    store: &CalculationStore,
    session: &Session,
    passages: &[SearchResult],
    index: &KnowledgeIndex,
    record: &CalcRecord,
) -> (Vec<Dependency>, Vec<SourceRef>) {
    let mut depends_on: Vec<Dependency> = Vec::new();
    let mut sources: Vec<SourceRef> = Vec::new();
    let current = |item_id: &str| graph.versions_of(session, item_id, None).ok().and_then(|v| v.last().map(|l| l.revision));
    for input in &record.inputs {
        match &input.source {
            InputSource::Memory { item_id, revision } => {
                if let Some(revision) = revision.or_else(|| current(item_id)) {
                    depends_on.push(Dependency { item_id: item_id.clone(), revision });
                }
            }
            InputSource::Calculation { calculation_id } => {
                if let Some(item) = store.graph_item(calculation_id, &session.user.id) {
                    if let Some(revision) = current(&item) {
                        depends_on.push(Dependency { item_id: item, revision });
                    }
                }
            }
            InputSource::Evidence { handle } => {
                let hit = match handle.strip_prefix('E').and_then(|n| n.parse::<usize>().ok()) {
                    Some(n) => passages.get(n.saturating_sub(1)).cloned(),
                    None => handle.strip_prefix("ev:").and_then(|chunk| index.passage(session, chunk).ok().flatten()),
                };
                if let Some(hit) = hit {
                    sources.push(SourceRef { sha256: hit.document_sha256, locator: format!("page {} · {}", hit.page, hit.chunk_id), extraction_revision: None });
                }
            }
            InputSource::Document { sha256, locator } => {
                sources.push(SourceRef { sha256: sha256.clone(), locator: locator.clone().unwrap_or_default(), extraction_revision: None })
            }
            _ => {}
        }
    }
    depends_on.dedup_by(|a, b| a.item_id == b.item_id);
    (depends_on, sources)
}

/// One line for the graph: enough to cite and to see what it rests on.
pub(crate) fn summary(record: &CalcRecord) -> String {
    let result = match (record.status, record.first()) {
        (RecordStatus::Unresolved, _) => format!("UNRESOLVED ({})", record.unresolved.join("; ")),
        (_, Some(first)) => format!(
            "{}{}",
            first.display,
            first.uncertainty_display.as_deref().map(|u| format!(" {u}")).unwrap_or_default()
        ),
        (_, None) => "no result".into(),
    };
    let comparison = record
        .comparison
        .as_ref()
        .map(|c| format!("; {} {} {} {}", c.value, c.relation, c.limit, c.verdict.as_str()))
        .unwrap_or_default();
    let inputs: Vec<String> = record.inputs.iter().map(CalcInput::line).collect();
    let mut text = format!(
        "{}: {} = {result}{comparison} [{}] from {} ({})",
        record.id,
        record.equation,
        record.status.as_str(),
        if inputs.is_empty() { "no named inputs".to_string() } else { inputs.join("; ") },
        record.engine.describe()
    );
    if text.chars().count() > 1200 {
        text = text.chars().take(1197).collect::<String>() + "…";
    }
    text
}

/// Publishes `record` into the run's task memory on the call's receipt.
///
/// Only a result, or an explicit gap: a record that computed something is a
/// tool observation; one missing an input is an open question naming it. A
/// refusal is not published — there is nothing anyone could build on.
pub fn publish(
    deps: &Arc<RuntimeDeps>,
    session: &Session,
    run_id: &str,
    tool: ToolName,
    record: &CalcRecord,
    receipt: Option<(i64, String)>,
) -> Option<String> {
    let graph = deps.memory_graph.as_ref()?;
    let owner = &session.user.id;
    if !(record.status.has_result() || record.status == RecordStatus::Unresolved) || record.operation == Operation::ValidateDimensions {
        return None;
    }
    let item_id = item_id_for(&record.id, owner);
    if let Some(held) = deps.calculation_store.graph_item(&record.id, owner) {
        return Some(held);
    }
    let passages = super::retrieval::for_run(&deps.passages, run_id);
    let (depends_on, sources) = lineage(graph, &deps.calculation_store, session, &passages, &deps.index, record);
    let (label, _) = super::artifact_tools::run_classification(deps, run_id);
    let classes = super::artifact_tools::classes_of_label(&label);
    let now = chrono::Utc::now().to_rfc3339();
    let provenance = match receipt {
        Some((event_seq, output_sha256)) => Provenance::ToolReceipt {
            run_id: run_id.to_string(),
            tool: tool.as_str().to_string(),
            event_seq,
            output_sha256: Some(output_sha256),
        },
        None => Provenance::Model { model_id: "unrecorded".into(), run_id: run_id.to_string() },
    };
    let item = MemoryItem {
        item_id: item_id.clone(),
        revision: 1,
        kind: if record.status.has_result() { MemoryKind::ToolObservation } else { MemoryKind::OpenQuestion },
        agent_id: "arjun".into(),
        scope: MemoryScope::Task { task_id: run_id.to_string() },
        classification: classes[0],
        acl: super::artifact_tools::acl_for(&classes, owner),
        creator_model_id: None,
        creator_run_id: Some(run_id.to_string()),
        provenance,
        content: summary(record),
        sources,
        artifacts: Vec::new(),
        confidence: None,
        status: ItemStatus::Proposed,
        valid_from: now.clone(),
        valid_until: None,
        supersedes: None,
        conflicts_with: Vec::new(),
        causal_parents: depends_on.iter().map(|d| d.item_id.clone()).collect(),
        idempotency_key: Some(format!("calc:{}:{owner}", record.id)),
        created_at: now.clone(),
        updated_at: now.clone(),
        basis: None,
        depends_on: depends_on.clone(),
        revoked_readers: Vec::new(),
        authority: Authority::Graph,
    };
    let scope = item.scope.clone();
    match graph.commit(item, None, &[]) {
        Ok(committed) => {
            for dependency in &depends_on {
                let _ = graph.link(MemoryEdge {
                    edge_id: edge_id(&committed.item_id, EdgeKind::DerivedFrom, &dependency.item_id),
                    from_item: committed.item_id.clone(),
                    to_item: dependency.item_id.clone(),
                    kind: EdgeKind::DerivedFrom,
                    agent_id: "arjun".into(),
                    scope: scope.clone(),
                    created_at: now.clone(),
                });
            }
            let _ = deps.calculation_store.link(&record.id, owner, &committed.item_id);
            Some(committed.item_id)
        }
        Err(error) => {
            log::warn!("[calculation] {} was not published: {}", record.id, error.explain());
            None
        }
    }
}

/// A calculation's standing now: `None` when it stands, else why not.
pub(crate) fn standing(graph: Option<&MemoryGraph>, store: &CalculationStore, session: &Session, calc_id: &str) -> Option<String> {
    let graph = graph?;
    let item = store.graph_item(calc_id, &session.user.id)?;
    let versions = graph.versions_of(session, &item, None).ok()?;
    let latest = versions.last()?;
    latest.item.status.withdrawn().then(|| match latest.item.status {
        ItemStatus::Stale => "something it rests on was corrected or withdrawn".to_string(),
        other => format!("its shared-memory record is {}", other.as_str()),
    })
}

// ── Going back to the sources ────────────────────────────────────────────

/// What a checker re-reads inputs through, under one reader's clearance.
pub struct Sources<'a> {
    pub graph: Option<&'a MemoryGraph>,
    pub index: &'a KnowledgeIndex,
    pub store: &'a CalculationStore,
    pub session: &'a Session,
    /// The run's evidence table, for `[E3]`. Empty for a worker, whose
    /// packet cannot carry the parent's markers.
    pub passages: Vec<SearchResult>,
}

impl Sources<'_> {
    fn passage_state(&self, hit: Option<SearchResult>, handle: &str) -> SourceState {
        let Some(hit) = hit else {
            return SourceState::Gone { why: format!("{handle} is not a passage this reader can see") };
        };
        // Current only while its document is the current version.
        match self.index.source_versions(self.session, &hit.document_sha256) {
            Ok(Some(versions)) => {
                let this = versions.iter().find(|v| v.document_sha256 == hit.document_sha256);
                match this.map(|v| v.status.as_str()) {
                    Some("superseded") => {
                        let replacement = this.and_then(|v| v.superseded_by.clone());
                        let text = replacement.as_deref().and_then(|sha| {
                            self.index.region(self.session, sha, hit.page, hit.page, 8).ok().map(|hits| {
                                hits.into_iter().map(|h| h.text).collect::<Vec<_>>().join("\n")
                            })
                        });
                        let now = replacement.as_deref().and_then(|sha| {
                            self.index.region(self.session, sha, hit.page, hit.page, 1).ok()?.into_iter().next()
                        });
                        SourceState::Changed {
                            revision: None,
                            text,
                            why: format!("{} was superseded by a later version", hit.document_name),
                            now: now.map(|h| InputSource::Evidence { handle: format!("ev:{}", h.chunk_id) }),
                        }
                    }
                    Some("withdrawn") | Some("revoked") => SourceState::Gone { why: format!("{} was withdrawn", hit.document_name) },
                    _ => match self.index.passage(self.session, &hit.chunk_id) {
                        Ok(Some(now)) => SourceState::Current { revision: None, text: now.text },
                        _ => SourceState::Gone { why: format!("{handle} is no longer readable") },
                    },
                }
            }
            _ => match self.index.passage(self.session, &hit.chunk_id) {
                Ok(Some(now)) => SourceState::Current { revision: None, text: now.text },
                _ => SourceState::Gone { why: format!("{handle} is no longer readable") },
            },
        }
    }
}

impl SourceReader for Sources<'_> {
    fn memory(&self, item_id: &str, pinned: Option<u64>) -> SourceState {
        let Some(graph) = self.graph else {
            return SourceState::Unreadable { why: "this deployment has no shared memory graph".into() };
        };
        let versions = match graph.versions_of(self.session, item_id, None) {
            Ok(v) => v,
            Err(error) => return SourceState::Unreadable { why: error.explain() },
        };
        let Some(latest) = versions.last() else {
            return SourceState::Gone { why: format!("{item_id} is not there, or not readable by this reader") };
        };
        let item = &latest.item;
        match item.status {
            ItemStatus::Superseded => {
                // What replaced it: the item that names it as superseded.
                let replacement = graph
                    .snapshot(self.session, &item.scope, item.acl.project_id.as_deref())
                    .ok()
                    .and_then(|items| items.into_iter().find(|i| i.supersedes.as_deref() == Some(item_id)));
                SourceState::Changed {
                    revision: Some(latest.revision),
                    text: replacement.as_ref().map(|r| r.content.clone()),
                    why: match &replacement {
                        Some(r) => format!("{item_id} was corrected by {}", r.item_id),
                        None => format!("{item_id} was superseded"),
                    },
                    now: replacement.map(|r| InputSource::Memory { item_id: r.item_id, revision: Some(r.revision) }),
                }
            }
            ItemStatus::Stale => SourceState::Changed {
                revision: Some(latest.revision),
                text: None,
                why: format!("{item_id} is stale: something it rests on changed"),
                now: None,
            },
            ItemStatus::Rejected | ItemStatus::Tombstoned => SourceState::Gone { why: format!("{item_id} is {}", item.status.as_str()) },
            _ => match pinned {
                Some(revision) if revision != latest.revision => SourceState::Changed {
                    revision: Some(latest.revision),
                    text: Some(item.content.clone()),
                    why: format!("{item_id} moved from revision {revision} to {}", latest.revision),
                    now: Some(InputSource::Memory { item_id: item_id.to_string(), revision: Some(latest.revision) }),
                },
                _ => SourceState::Current { revision: Some(latest.revision), text: item.content.clone() },
            },
        }
    }

    fn evidence(&self, handle: &str) -> SourceState {
        let hit = match handle.strip_prefix('E').and_then(|n| n.parse::<usize>().ok()) {
            Some(n) if !self.passages.is_empty() => self.passages.get(n.saturating_sub(1)).cloned(),
            Some(_) => {
                return SourceState::Unreadable {
                    why: format!("{handle} is a marker in the run's own evidence table, which this reader does not hold; cite the passage as ev:<chunk>"),
                }
            }
            None => handle.strip_prefix("ev:").and_then(|chunk| self.index.passage(self.session, chunk).ok().flatten()),
        };
        self.passage_state(hit, handle)
    }

    fn document(&self, sha256: &str, locator: Option<&str>) -> SourceState {
        let page = locator
            .and_then(|l| l.trim_start_matches(['p', 'P']).trim_start_matches("age").trim().split(|c: char| !c.is_ascii_digit()).next().map(str::to_string))
            .and_then(|p| p.parse::<u32>().ok());
        let Some(page) = page else {
            return SourceState::Unreadable { why: "a document source needs a page to be re-read (S:<sha256>@p4)".into() };
        };
        // A prefix names the document when it names exactly one.
        let full = if sha256.len() == 64 {
            Some(sha256.to_string())
        } else {
            self.index.documents(self.session).ok().and_then(|docs| {
                let matching: Vec<_> = docs.into_iter().filter(|d| d.document_sha256.starts_with(sha256)).collect();
                (matching.len() == 1).then(|| matching[0].document_sha256.clone())
            })
        };
        let Some(full) = full else {
            return SourceState::Gone { why: format!("S:{sha256} is not one document this reader can see") };
        };
        match self.index.region(self.session, &full, page, page, 16) {
            Ok(hits) if !hits.is_empty() => {
                let first = hits[0].clone();
                match self.passage_state(Some(first), &format!("S:{sha256}@p{page}")) {
                    SourceState::Current { revision, .. } => SourceState::Current {
                        revision,
                        text: hits.into_iter().map(|h| h.text).collect::<Vec<_>>().join("\n"),
                    },
                    other => other,
                }
            }
            _ => SourceState::Gone { why: format!("page {page} of S:{sha256} is not readable by this reader") },
        }
    }

    fn calculation(&self, calc_id: &str) -> Option<(CalcRecord, Option<String>)> {
        let record = self.store.get(calc_id, &self.session.user.id)?;
        let stale = standing(self.graph, self.store, self.session, calc_id);
        Some((record, stale))
    }
}
