//! The Document Author's tools (plan P09), on the agent path.
//!
//! | Tool | Does |
//! |---|---|
//! | `document.template_list` | the authoring templates, with section ids, citation rules and definition hashes |
//! | `document.compose` | a structured spec checked against its template and the run's evidence, written as a complete Word package, reopened and registered as a candidate version |
//! | `document.patch_section` | one section of an exact version replaced, everything else kept byte for byte, as a new candidate version |
//! | `document.render_pages` | every page laid out and rasterised, with the renderer and font versions, and the render checks |
//! | `artifact.validate_document` | the P04 ladder plus the document checks, recorded against the exact bytes |
//!
//! Nothing here approves anything. A composed or patched version is a
//! candidate; `final` needs an accepted validation and a person
//! (`artifact.register_version`, which the document-author role is not
//! granted), and the independent review is the reviewer's (P10).
//!
//! Citations resolve against what the run can actually read — its retrieved
//! passages, the memory graph, the calculation store and this conversation's
//! artifacts — and, for a patch, the bindings the base version already holds.
//! A marker that resolves to nothing is refused, never trusted.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::artifacts::authoring::{
    self, checking_template, AuthoredSection, CitationResolver, DocumentSpec, Gap, Problem, ResolvedCitation,
    SectionLineage, SpecSection, Stamp,
};
use crate::artifacts::content::{self, CitationTarget};
use crate::artifacts::conversation_store::{ArtifactRecord, NewArtifact, Producer, Stage, VersionDependency, VersionMeta};
use crate::artifacts::doc_model::Block;
use crate::artifacts::document_checks::{self, CheckResult, CheckStatus};
use crate::artifacts::package::{self, DetectedFormat};
use crate::artifacts::{render, section_patch, validation};
use crate::identity::Session;
use crate::knowledge::graph::runtime_memory::{
    edge_id, Authority, EdgeKind, ItemStatus, MemoryEdge, MemoryItem, MemoryKind, MemoryScope, Provenance,
};
use crate::knowledge::SearchResult;
use crate::orchestrator::tools::{ToolCall, ToolName};

use super::artifact_tools::{self, Registered};
use super::{CallParams, RuntimeDeps};

/// What a compose or patch leaves for the step after its receipt: the
/// section lineage of the version it registered, and the gaps it reported.
#[derive(Debug, Clone, Default)]
pub(super) struct Composed {
    pub lineage: Vec<SectionLineage>,
    pub gaps: Vec<Gap>,
}

// ── Resolving citations against the run ──────────────────────────────────

/// What a marker means to this run, and whether it still stands.
pub(crate) struct RunResolver<'a> {
    deps: &'a Arc<RuntimeDeps>,
    session: &'a Session,
    conversation: String,
    passages: Vec<SearchResult>,
    /// A base version's bindings, carried through a patch: `[E3]` was
    /// numbered by the run that wrote it.
    inherited: Vec<VersionDependency>,
}

impl<'a> RunResolver<'a> {
    pub(crate) fn new(deps: &'a Arc<RuntimeDeps>, session: &'a Session, run_id: &str, conversation: &str, inherited: Vec<VersionDependency>) -> Self {
        RunResolver {
            deps,
            session,
            conversation: conversation.to_string(),
            passages: super::retrieval::for_run(&deps.passages, run_id),
            inherited,
        }
    }

    fn memory(&self, item_id: &str, revision: Option<u64>, marker: &str) -> Option<ResolvedCitation> {
        let graph = self.deps.memory_graph.as_ref()?;
        let versions = graph.versions_of(self.session, item_id, None).ok()?;
        let latest = versions.last()?;
        let chosen = match revision {
            Some(wanted) => versions.iter().find(|v| v.revision == wanted)?,
            None => latest,
        };
        let stale_because = if chosen.revision != latest.revision {
            Some(format!("revision {} was replaced by revision {}", chosen.revision, latest.revision))
        } else if latest.item.status.withdrawn() {
            Some(format!("the item is now {}", latest.item.status.as_str()))
        } else {
            None
        };
        Some(ResolvedCitation {
            marker: marker.to_string(),
            label: format!("memory item {item_id}@{}: {}", chosen.revision, chosen.item.content.chars().take(90).collect::<String>()),
            text: chosen.item.content.clone(),
            calculation: false,
            stale_because,
        })
    }

    fn calculation(&self, calculation_id: &str, marker: &str) -> Option<ResolvedCitation> {
        let record = self.deps.calculation_store.get(calculation_id, &self.session.user.id)?;
        if !record.status.has_result() {
            return None;
        }
        Some(ResolvedCitation {
            marker: marker.to_string(),
            label: super::calculation_tools::summary(&record).lines().next().unwrap_or_default().chars().take(140).collect(),
            text: record.render(),
            calculation: true,
            stale_because: super::calculation_tools::standing(
                self.deps.memory_graph.as_deref(),
                &self.deps.calculation_store,
                self.session,
                calculation_id,
            ),
        })
    }

    fn document(&self, sha256: &str, locator: Option<&str>, marker: &str) -> Option<ResolvedCitation> {
        let documents = self.deps.index.documents(self.session).ok()?;
        let matching: Vec<_> = documents.iter().filter(|d| d.document_sha256.starts_with(sha256)).collect();
        let [document] = matching.as_slice() else { return None };
        // The passage a locator names ("page 2 · chunk-id"), else what this run
        // retrieved from the document.
        let chunk = locator.and_then(|l| l.rsplit(" · ").next()).and_then(|id| self.deps.index.passage(self.session, id).ok().flatten());
        let text = match chunk {
            Some(passage) => passage.text,
            None => self.passages.iter().filter(|p| p.document_sha256 == document.document_sha256).map(|p| p.text.clone()).collect::<Vec<_>>().join("\n"),
        };
        Some(ResolvedCitation {
            marker: marker.to_string(),
            label: format!("{}{}", document.document_name, locator.map(|l| format!(", {l}")).unwrap_or_default()),
            text,
            calculation: false,
            stale_because: None,
        })
    }
}

impl CitationResolver for RunResolver<'_> {
    fn resolve(&self, target: &CitationTarget, marker: &str) -> Option<ResolvedCitation> {
        if let Some(kept) = self.inherited.iter().find(|d| d.marker.as_deref() == Some(marker)) {
            use crate::artifacts::conversation_store::DependencyKind;
            return match kept.kind {
                DependencyKind::MemoryItem => self.memory(&kept.id, kept.version.as_deref().and_then(|v| v.parse().ok()), marker),
                DependencyKind::Calculation => self.calculation(&kept.id, marker),
                DependencyKind::Document => self.document(&kept.id, kept.locator.as_deref(), marker).or_else(|| {
                    Some(ResolvedCitation {
                        marker: marker.to_string(),
                        label: kept.label.clone().unwrap_or_else(|| kept.id.clone()),
                        text: String::new(),
                        calculation: false,
                        stale_because: Some("no longer a current document you may read".into()),
                    })
                }),
                _ => Some(ResolvedCitation {
                    marker: marker.to_string(),
                    label: kept.label.clone().unwrap_or_else(|| kept.id.clone()),
                    text: String::new(),
                    calculation: false,
                    stale_because: None,
                }),
            };
        }
        match target {
            CitationTarget::Evidence { number } => {
                let hit = self.passages.get((*number as usize).checked_sub(1)?)?;
                Some(ResolvedCitation {
                    marker: marker.to_string(),
                    label: hit.citation(),
                    text: hit.text.clone(),
                    calculation: false,
                    stale_because: None,
                })
            }
            CitationTarget::Memory { item_id, revision } => self.memory(item_id, *revision, marker),
            CitationTarget::Calculation { calculation_id } => self.calculation(calculation_id, marker),
            CitationTarget::Source { sha256, locator } if sha256.len() >= 8 => self.document(sha256, locator.as_deref(), marker),
            CitationTarget::Source { .. } => None,
            CitationTarget::Artifact { artifact_id, version } => {
                let record = self.deps.conversation_artifacts.get(&self.session.user.id, artifact_id, *version).ok()??;
                if record.conversation_id != self.conversation {
                    return None;
                }
                let text = self
                    .deps
                    .conversation_artifacts
                    .read(&self.session.user.id, &record.reference())
                    .ok()
                    .flatten()
                    .and_then(|(_, bytes)| content::extract(&bytes).ok())
                    .map(|model| model.plain_text().chars().take(40_000).collect())
                    .unwrap_or_default();
                Some(ResolvedCitation {
                    marker: marker.to_string(),
                    label: format!("{} {}", record.title, record.reference()),
                    text,
                    calculation: false,
                    stale_because: None,
                })
            }
        }
    }
}

fn conversation_of(deps: &Arc<RuntimeDeps>, call: &CallParams) -> Result<String, String> {
    deps.run_to_conversation
        .lookup(&call.run_id)
        .ok_or_else(|| "This run is not attached to a conversation, so a document it wrote could not be kept as a version.".to_string())
}

fn describe_problems(problems: &[Problem]) -> String {
    let mut out = format!("Nothing was written: {} problem(s).\n", problems.len());
    for problem in problems.iter().take(30) {
        out.push_str(&format!("- {}\n", problem.describe()));
    }
    out.push_str(
        "Fix what can be fixed from the evidence. Where the information does not exist, declare the section a \
         gap (\"gap\": what is needed) or write \"?\" for a field: a gap is reported and asked for, never filled in.",
    );
    out
}

fn check_lines(results: &[CheckResult]) -> String {
    let mut out = String::new();
    for result in results {
        let status = match result.status {
            CheckStatus::Pass => "pass",
            CheckStatus::Fail => "FAIL",
            CheckStatus::Unavailable => "unavailable",
        };
        out.push_str(&format!("- {} [{:?}]: {status} — {}\n", result.check, result.severity, result.detail));
    }
    out
}

// ── document.template_list ───────────────────────────────────────────────

pub(super) fn template_list() -> Result<String, String> {
    let mut out = String::from("Authoring templates (document.compose):\n");
    for template in authoring::AUTHORING_TEMPLATES {
        out.push_str(&format!("- {} — {}\n  definition sha-256 {}\n", template.key(), template.description, template.sha256()));
        if !template.fields.is_empty() {
            out.push_str(&format!(
                "  fields: {}\n",
                template.fields.iter().map(|f| format!("{}{}", f.key, if f.required { " (required)" } else { "" })).collect::<Vec<_>>().join(", ")
            ));
        }
        for section in template.sections {
            let rule = match (section.generated, section.citations) {
                (true, _) => "generated from the citations; do not supply",
                (_, authoring::CitationRule::Required) => "every paragraph, item and row cites",
                (_, authoring::CitationRule::Calculation) => "every line cites a calculation record [C:calc-…]",
                (_, authoring::CitationRule::Optional) => "citations optional; figures still cited",
            };
            out.push_str(&format!("  section {} \"{}\"{}: {rule}\n", section.id, section.heading, if section.required { " (required)" } else { "" }));
        }
        if template.free_sections {
            out.push_str("  any other section id (lowercase, digits, _), in the order given\n");
        }
    }
    let legacy = crate::artifacts::templates::find("approval_note").map(|t| t.sha256).unwrap_or_default();
    out.push_str(&format!(
        "Compatible: approval_note@1 (definition sha-256 {legacy}) — the note artifact.create_approval_note writes; \
         document.compose with this template goes through that writer. Its sections are patchable by field key.\n"
    ));
    Ok(out)
}

// ── document.compose ─────────────────────────────────────────────────────

fn spec_of(tool_call: &ToolCall) -> Result<DocumentSpec, String> {
    let mut object = serde_json::Map::new();
    for key in ["template", "title", "audience", "output", "fields", "sections"] {
        if let Some(value) = tool_call.arguments.get(key) {
            object.insert(key.to_string(), value.clone());
        }
    }
    serde_json::from_value(Value::Object(object)).map_err(|error| {
        format!(
            "the specification could not be read: {error}. It is {{template, title, audience, output, fields?, sections: \
             [{{id, heading?, level?, blocks: [{{kind: paragraph|bullets|numbered|table|pageBreak, …}}], gap?}}]}}"
        )
    })
}

/// The legacy note's fields, from a spec in its sections' terms.
fn legacy_content(spec: &DocumentSpec) -> Result<BTreeMap<String, String>, String> {
    let mut content = BTreeMap::new();
    content.insert("title".to_string(), spec.title.clone());
    for (key, value) in &spec.fields {
        if value.trim() == "?" {
            return Err(format!("approval_note@1 cannot state that {key:?} is unknown; use inspection_approval_note@1, which reports gaps"));
        }
        content.insert(key.clone(), value.clone());
    }
    for section in &spec.sections {
        if section.gap.is_some() {
            return Err(format!(
                "approval_note@1 has no way to show that {:?} is missing; use inspection_approval_note@1, which reports gaps",
                section.id
            ));
        }
        let mut lines = Vec::new();
        for block in &section.blocks {
            match block {
                Block::Paragraph { text } => lines.push(text.clone()),
                Block::Bullets { items } => lines.extend(items.iter().map(|i| format!("- {i}"))),
                Block::Numbered { items } => lines.extend(items.iter().enumerate().map(|(n, i)| format!("{}. {i}", n + 1))),
                Block::Table { header, rows, caption } => {
                    lines.extend(caption.clone());
                    lines.push(header.join(" | "));
                    lines.extend(rows.iter().map(|r| r.join(" | ")));
                }
                Block::PageBreak => {}
            }
        }
        content.insert(section.id.clone(), lines.join("\n"));
    }
    Ok(content)
}

pub(super) fn compose(
    deps: &Arc<RuntimeDeps>,
    call: &CallParams,
    session: &Session,
    tool_call: &ToolCall,
    written: &mut Option<PathBuf>,
    composed: &mut Option<Composed>,
) -> Result<String, String> {
    let conversation = conversation_of(deps, call)?;
    let root = deps.root_for(&call.run_id).ok_or_else(|| "this run has no workspace to write the document into".to_string())?;
    let spec = spec_of(tool_call)?;
    if !spec.output.to_ascii_lowercase().ends_with(".docx") || spec.output.contains(['/', '\\']) || spec.output.contains("..") {
        return Err(format!("{:?} is not a usable file name: a bare name ending .docx, no folders", spec.output));
    }
    let path = root.join(&spec.output);

    // The compatible path: the legacy note, through its own writer.
    if spec.template.trim_end_matches("@1") == "approval_note" {
        let content = legacy_content(&spec)?;
        let legacy_call = ToolCall::new(ToolName::CreateDocx.as_str(), json!({ "template": "approval_note", "content": content }));
        let passages = super::retrieval::for_run(&deps.passages, &call.run_id);
        let calculations = deps.calculations.lock().ok().and_then(|t| t.get(&call.run_id).cloned()).unwrap_or_default();
        let answer = super::artifacts::create_docx_with_evidence(call, Some(&path), session, &legacy_call, &passages, &calculations)?;
        *written = Some(path);
        *composed = Some(Composed {
            lineage: spec
                .sections
                .iter()
                .map(|s| SectionLineage {
                    section_id: s.id.clone(),
                    heading: s.heading.clone().unwrap_or_else(|| s.id.clone()),
                    markers: s.blocks.iter().flat_map(|b| content::citations_in(&b.text())).map(|c| c.marker).collect(),
                    calculations: s
                        .blocks
                        .iter()
                        .flat_map(|b| content::citations_in(&b.text()))
                        .filter(|c| matches!(c.target, CitationTarget::Calculation { .. }))
                        .map(|c| c.marker)
                        .collect(),
                    gap: None,
                })
                .collect(),
            gaps: Vec::new(),
        });
        return Ok(format!("Through the compatible approval_note@1 writer (artifact.create_approval_note): {answer}"));
    }

    let (classification, _) = artifact_tools::run_classification(deps, &call.run_id);
    let stamp = Stamp {
        task_id: call.run_id.clone(),
        model: call.model.clone().unwrap_or_else(|| "unrecorded".into()),
        created_at: chrono::Utc::now().to_rfc3339(),
    };
    let resolver = RunResolver::new(deps, session, &call.run_id, &conversation, Vec::new());
    let composition = authoring::compose(&spec, &classification, &stamp, &resolver).map_err(|problems| describe_problems(&problems))?;
    let bytes = authoring::write_docx(&composition.document)?;

    // Reopened and checked before it is kept: a package that does not read
    // back is not written at all.
    let template = checking_template(composition.document.template.split('@').next().unwrap_or_default());
    let checks = document_checks::format_checks(&bytes, template, Some(&resolver));
    if checks.iter().any(|c| c.check == "document.reopen" && c.status != CheckStatus::Pass) {
        return Err(format!("the written package did not reopen; nothing was kept.\n{}", check_lines(&checks)));
    }
    std::fs::write(&path, &bytes).map_err(|e| format!("the document could not be written: {e}"))?;
    *written = Some(path);

    let mut out = format!(
        "Wrote {} ({} bytes, sha-256 {}) from {} for {}: {} section(s), {} citation(s) bound. It is registered as a \
         candidate version (DRAFT, not approved).\n",
        spec.output,
        bytes.len(),
        package_sha(&bytes),
        composition.document.template,
        composition.document.audience,
        composition.document.sections.len(),
        composition.citations.len()
    );
    for line in &composition.lineage {
        out.push_str(&format!(
            "- section {} \"{}\": {}\n",
            line.section_id,
            line.heading,
            match &line.gap {
                Some(gap) => format!("GAP — {gap}"),
                None if line.markers.is_empty() => "cites nothing".to_string(),
                None => format!("rests on {}", line.markers.join(" ")),
            }
        ));
    }
    if !composition.gaps.is_empty() {
        out.push_str(&format!(
            "INFORMATION NEEDED ({}): the document is incomplete and will not be accepted until these are answered:\n",
            composition.gaps.len()
        ));
        for gap in &composition.gaps {
            out.push_str(&format!("- {} ({}): {}\n", gap.heading, gap.at, gap.need));
        }
    }
    out.push_str("Checks of the written file:\n");
    out.push_str(&check_lines(&checks));
    out.push_str("Next: artifact.validate_document on the version, which renders every page; then task.request_review.\n");
    *composed = Some(Composed { lineage: composition.lineage, gaps: composition.gaps });
    Ok(out)
}

fn package_sha(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    format!("{:x}", Sha256::digest(bytes))
}

// ── After the receipt: lineage and gaps ──────────────────────────────────

/// Records the section lineage of the version a compose or patch registered,
/// and publishes each gap as an open question resting on it -- on the call's
/// receipt, after the version itself is in the graph.
#[allow(clippy::too_many_arguments)]
pub(super) fn after_receipt(
    deps: &Arc<RuntimeDeps>,
    session: &Session,
    run_id: &str,
    tool: ToolName,
    record: &ArtifactRecord,
    artifact_item: Option<&str>,
    composed: Composed,
    receipt: Option<(i64, String)>,
) {
    let rows: Vec<(String, String)> = composed
        .lineage
        .iter()
        .map(|line| (line.section_id.clone(), serde_json::to_string(line).unwrap_or_default()))
        .collect();
    if let Err(error) = deps.conversation_artifacts.record_sections(&session.user.id, &record.reference(), &rows) {
        log::warn!("[document] {} section lineage was not recorded: {error}", record.reference());
    }
    let Some(graph) = deps.memory_graph.as_ref() else { return };
    let classes = artifact_tools::classes_of_label(&record.classification);
    let now = chrono::Utc::now().to_rfc3339();
    for gap in composed.gaps {
        let item_id = format!("mi-gap-{}-v{}-{}", record.artifact_id, record.version, gap.at.replace(':', "-"));
        let provenance = match &receipt {
            Some((event_seq, output_sha256)) => Provenance::ToolReceipt {
                run_id: run_id.to_string(),
                tool: tool.as_str().to_string(),
                event_seq: *event_seq,
                output_sha256: Some(output_sha256.clone()),
            },
            None => Provenance::Model { model_id: "unrecorded".into(), run_id: run_id.to_string() },
        };
        let item = MemoryItem {
            item_id: item_id.clone(),
            revision: 1,
            kind: MemoryKind::OpenQuestion,
            agent_id: "document-author".into(),
            scope: MemoryScope::Task { task_id: crate::subagents::tool_port::task_of(run_id) },
            classification: classes[0],
            acl: artifact_tools::acl_for(&classes, &session.user.id),
            creator_model_id: None,
            creator_run_id: Some(run_id.to_string()),
            provenance,
            content: format!("Information needed for {} ({}) of {}: {}", gap.heading, gap.at, record.reference(), gap.need),
            sources: Vec::new(),
            artifacts: vec![crate::knowledge::graph::runtime_memory::ArtifactRef {
                artifact_id: record.artifact_id.clone(),
                revision: record.version,
                sha256: record.sha256.clone(),
            }],
            confidence: None,
            status: ItemStatus::Proposed,
            valid_from: now.clone(),
            valid_until: None,
            supersedes: None,
            conflicts_with: Vec::new(),
            causal_parents: Vec::new(),
            idempotency_key: Some(format!("document-gap:{item_id}")),
            created_at: now.clone(),
            updated_at: now.clone(),
            basis: None,
            depends_on: Vec::new(),
            revoked_readers: Vec::new(),
            authority: Authority::Graph,
        };
        let scope = item.scope.clone();
        match graph.commit(item, None, &[]) {
            Ok(committed) => {
                if let Some(artifact_item) = artifact_item {
                    let _ = graph.link(MemoryEdge {
                        edge_id: edge_id(&committed.item_id, EdgeKind::PartOf, artifact_item),
                        from_item: committed.item_id.clone(),
                        to_item: artifact_item.to_string(),
                        kind: EdgeKind::PartOf,
                        agent_id: "document-author".into(),
                        scope,
                        created_at: now.clone(),
                    });
                }
            }
            Err(error) => log::warn!("[document] a gap of {} was not published: {}", record.reference(), error.explain()),
        }
    }
}

// ── document.patch_section ───────────────────────────────────────────────

/// Markers the document's written sections cite, in order, excluding the
/// generated references and the provenance.
fn cited_markers(document_xml: &str) -> Vec<String> {
    let mut out = Vec::new();
    for section in authoring::read_sections(document_xml).unwrap_or_default() {
        if section.id == "references" || section.id == "provenance" {
            continue;
        }
        for citation in content::citations_in(&section.text) {
            if !out.contains(&citation.marker) {
                out.push(citation.marker);
            }
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
pub(super) fn patch_section(
    deps: &Arc<RuntimeDeps>,
    call: &CallParams,
    session: &Session,
    tool_call: &ToolCall,
    effect_key: Option<&str>,
    registered: &mut Option<Registered>,
    composed: &mut Option<Composed>,
) -> Result<String, String> {
    let (id, version) = artifact_tools::parse_reference(tool_call, "artifact")?;
    let Some(version) = version else {
        return Err("name the exact version you are patching (art-…@N) and its sha256, so a write made since is not overwritten".into());
    };
    let expected = tool_call.text("sha256").unwrap_or_default().trim().to_ascii_lowercase();
    let base = artifact_tools::resolve(deps, call, session, &id, Some(version))?;
    let newest = artifact_tools::resolve(deps, call, session, &id, None)?;
    if newest.version != base.version {
        return Err(format!(
            "stale write refused: {id} has moved on to version {} (sha-256 {}) since version {version}. Read {} and patch that.",
            newest.version,
            &newest.sha256[..12],
            newest.reference()
        ));
    }
    if expected != base.sha256 {
        return Err(format!(
            "stale write refused: {} holds sha-256 {}, not {:?}. Name the full hash of the version you read (artifact.manifest).",
            base.reference(),
            base.sha256,
            expected
        ));
    }
    let section_id = tool_call.text("section").unwrap_or_default().trim().to_string();
    let blocks: Vec<Block> = match tool_call.arguments.get("blocks") {
        None | Some(Value::Null) => Vec::new(),
        Some(value) => serde_json::from_value(value.clone()).map_err(|e| format!("\"blocks\" could not be read: {e}"))?,
    };
    let gap = tool_call.text("gap").map(str::trim).filter(|g| !g.is_empty()).map(str::to_string);
    let heading = tool_call.text("heading").map(str::trim).filter(|h| !h.is_empty()).map(str::to_string);

    let bytes = artifact_tools_load(deps, session, &base)?;
    let template = base.template.as_ref().and_then(|t| checking_template(&t.id)).unwrap_or(&authoring::AUTHORED_DOCUMENT);
    if template.sections.iter().any(|s| s.id == section_id && s.generated) {
        return Err(format!("{section_id:?} is generated from the document's citations and is rewritten with them; patch the section that cites"));
    }
    let conversation = conversation_of(deps, call)?;
    let inherited = deps.conversation_artifacts.dependencies(&session.user.id, &base.reference()).map_err(|e| e.to_string())?;
    let resolver = RunResolver::new(deps, session, &call.run_id, &conversation, inherited.clone());
    let spec_section = SpecSection { id: section_id.clone(), heading: heading.clone(), level: None, blocks: blocks.clone(), gap: gap.clone() };
    let mut resolved = BTreeMap::new();
    let (problems, line) = authoring::check_section(template, &spec_section, &resolver, &mut resolved);
    if !problems.is_empty() {
        return Err(describe_problems(&problems));
    }
    let replacement = AuthoredSection { id: section_id.clone(), heading: heading.unwrap_or_default(), level: 0, blocks, gap: gap.clone() };
    let first = section_patch::patch_section(&bytes, &section_id, &replacement)?;

    // The generated references follow the citations: when the patch adds or
    // drops a marker, that section is rewritten from them too.
    let mut outcome_bytes = first.bytes.clone();
    let mut regenerated = None;
    if template.sections.iter().any(|s| s.id == "references" && s.generated) {
        let xml = package::read_part(&first.bytes, "word/document.xml")?;
        let cited = cited_markers(&xml);
        let listed: Vec<String> = authoring::read_sections(&xml)?
            .into_iter()
            .find(|s| s.id == "references")
            .map(|s| content::citations_in(&s.text).into_iter().map(|c| c.marker).collect())
            .unwrap_or_default();
        let mut sorted_cited = cited.clone();
        sorted_cited.sort();
        let mut sorted_listed = listed.clone();
        sorted_listed.sort();
        if sorted_cited != sorted_listed {
            let mut items = Vec::new();
            for marker in &cited {
                let citation = content::citations_in(marker).into_iter().next();
                let label = citation
                    .and_then(|c| resolved.get(marker).cloned().or_else(|| resolver.resolve(&c.target, marker)))
                    .map(|r| r.label)
                    .unwrap_or_else(|| "(binding held by the base version)".into());
                items.push(format!("{marker} {label}"));
            }
            items.sort();
            let references = AuthoredSection {
                id: "references".into(),
                heading: String::new(),
                level: 0,
                blocks: if items.is_empty() { vec![Block::Paragraph { text: "This document cites no source.".into() }] } else { vec![Block::Bullets { items }] },
                gap: None,
            };
            let second = section_patch::patch_section(&first.bytes, "references", &references)?;
            outcome_bytes = second.bytes.clone();
            regenerated = Some(second);
        }
    }
    let checks = document_checks::format_checks(&outcome_bytes, Some(template), Some(&resolver));
    if checks.iter().any(|c| c.check == "document.reopen" && c.status != CheckStatus::Pass) {
        return Err(format!("the patched package did not reopen; nothing was kept.\n{}", check_lines(&checks)));
    }

    let model = content::extract(&outcome_bytes).map_err(|e| format!("the patched file does not read back: {e}"))?;
    let mut dependencies = artifact_tools::bind(deps, &call.run_id, session, &conversation, &model, &inherited);
    for kept in inherited.iter().filter(|d| d.marker.is_none()) {
        dependencies.push(kept.clone());
    }
    let registration = deps
        .conversation_artifacts
        .record_version(
            NewArtifact {
                artifact_id: Some(base.artifact_id.clone()),
                conversation_id: base.conversation_id.clone(),
                owner_user_id: session.user.id.clone(),
                message_id: None,
                run_id: Some(call.run_id.clone()),
                producer: Producer { model_id: call.model.clone(), tool: Some(ToolName::DocumentPatchSection.as_str().to_string()), agent: None },
                kind: base.kind,
                mime: base.mime.clone(),
                title: base.title.clone(),
                filename: base.filename.clone(),
                complete: true,
                derived_from: Some(base.reference()),
                renders: None,
                language: base.language.clone(),
                render_requires: base.render_requires.clone(),
                content: outcome_bytes.clone(),
            },
            VersionMeta {
                stage: Stage::Candidate,
                classification: Some(base.classification.clone()),
                template: base.template.clone(),
                project_id: base.project_id.clone(),
                effect_key: effect_key.map(|k| format!("patch:{k}")),
                dependencies: dependencies.clone(),
            },
        )
        .map_err(|e| format!("the patch was made and could not be recorded: {e}"))?;
    let record = registration.record.clone();
    if let (Some(root), Some(name)) = (deps.root_for(&call.run_id), base.filename.as_deref().and_then(|f| std::path::Path::new(f).file_name())) {
        let _ = std::fs::write(root.join(name), &outcome_bytes);
    }

    // The new version's lineage: the base's, with the patched sections'.
    let mut lineage: Vec<SectionLineage> = deps
        .conversation_artifacts
        .sections(&session.user.id, &base.reference())
        .unwrap_or_default()
        .into_iter()
        .filter_map(|(_, json)| serde_json::from_str::<SectionLineage>(&json).ok())
        .collect();
    let mut replace = |updated: SectionLineage| match lineage.iter_mut().find(|l| l.section_id == updated.section_id) {
        Some(slot) => *slot = updated,
        None => lineage.push(updated),
    };
    replace(SectionLineage { heading: if line.heading == section_id { first.after.lines().next().unwrap_or_default().to_string() } else { line.heading.clone() }, ..line.clone() });
    if regenerated.is_some() {
        let xml = package::read_part(&outcome_bytes, "word/document.xml")?;
        replace(SectionLineage {
            section_id: "references".into(),
            heading: "Supporting references".into(),
            markers: cited_markers(&xml),
            calculations: cited_markers(&xml).into_iter().filter(|m| m.starts_with("[C:")).collect(),
            gap: None,
        });
    }
    *composed = Some(Composed {
        lineage,
        gaps: gap.map(|need| vec![Gap { at: section_id.clone(), heading: line.heading.clone(), need }]).unwrap_or_default(),
    });
    *registered = Some(Registered { record: record.clone(), dependencies, base: Some(base.reference()) });

    let mut out = format!(
        "{} {} (sha-256 {}) from {}: section {section_id:?} replaced.\nBefore: {:?}\nAfter: {:?}\n",
        if registration.duplicate { "Already had" } else { "Wrote" },
        record.reference(),
        record.sha256,
        base.reference(),
        first.before.chars().take(300).collect::<String>(),
        first.after.chars().take(300).collect::<String>()
    );
    out.push_str(&format!(
        "Kept byte for byte: {} other part(s) ({}); {} bytes of the body before the section and {} after; sections unchanged: {}. Lists: {}.\n",
        first.parts_identical.len(),
        first.parts_identical.join(", "),
        first.prefix_bytes,
        first.suffix_bytes,
        first.sections_identical.join(", "),
        first.list_dialect
    ));
    if let Some(second) = &regenerated {
        out.push_str(&format!("The generated references were rewritten from the citations now in the document: {:?}\n", second.after.chars().take(300).collect::<String>()));
    }
    out.push_str("Checks of the written file:\n");
    out.push_str(&check_lines(&checks));
    out.push_str("It is a candidate; validate it (artifact.validate_document) before review.\n");
    Ok(out)
}

fn artifact_tools_load(deps: &Arc<RuntimeDeps>, session: &Session, record: &ArtifactRecord) -> Result<Vec<u8>, String> {
    deps.conversation_artifacts
        .read(&session.user.id, &record.reference())
        .map_err(|error| format!("{}'s content could not be read: {error}", record.reference()))?
        .map(|(_, bytes)| bytes)
        .ok_or_else(|| format!("{} has no stored content", record.reference()))
}

// ── document.render_pages ────────────────────────────────────────────────

/// A render of a version and what it was drawn with, as kept.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct DocumentRender {
    pub render_id: String,
    pub outcome: render::RenderOutcome,
    pub checks: Vec<CheckResult>,
}

pub(crate) fn render_document(deps: &Arc<RuntimeDeps>, session: &Session, record: &ArtifactRecord, bytes: &[u8]) -> DocumentRender {
    let render_id = format!("rnd-{}", uuid::Uuid::new_v4().simple());
    let out_dir = deps.conversation_artifacts.renders_dir().join(&render_id);
    let outcome = render::render(bytes, package::sniff(bytes), &out_dir, 1, render::MAX_PAGES_PER_RENDER, render::inventory());
    let state = serde_json::to_value(outcome.state).ok().and_then(|v| v.as_str().map(str::to_string)).unwrap_or_default();
    let _ = deps.conversation_artifacts.record_render(
        &session.user.id,
        &record.reference(),
        &render_id,
        &record.sha256,
        &state,
        &serde_json::to_string(&outcome).unwrap_or_default(),
    );
    let checks = document_checks::render_checks(bytes, &outcome);
    DocumentRender { render_id, outcome, checks }
}

fn render_lines(rendered: &DocumentRender) -> String {
    let outcome = &rendered.outcome;
    let mut out = format!("Render {}: {:?}. {}\n", rendered.render_id, outcome.state, outcome.detail);
    if !outcome.renderers.is_empty() {
        out.push_str(&format!(
            "Renderers: {}.\n",
            outcome.renderers.iter().map(|r| format!("{} {}", r.name, r.version)).collect::<Vec<_>>().join(" + ")
        ));
    }
    if !outcome.font_files.is_empty() {
        out.push_str(&format!(
            "Fonts: {}.\n",
            outcome
                .font_files
                .iter()
                .map(|f| match &f.version {
                    Some(v) => format!("{} {v} ({})", f.family, std::path::Path::new(&f.file).file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default()),
                    None => format!("{} (no installed file found)", f.family),
                })
                .collect::<Vec<_>>()
                .join(", ")
        ));
    } else if outcome.state == render::RenderState::Rendered {
        out.push_str("Fonts: the installed font versions could not be read on this machine (no fontconfig).\n");
    }
    for problem in &outcome.problems {
        out.push_str(&format!("Problem: {problem}\n"));
    }
    for page in &outcome.pages {
        out.push_str(&format!(
            "- render:{}/page:{} — {}×{} px, {} characters{}{}, image sha-256 {}\n",
            rendered.render_id,
            page.page,
            page.width,
            page.height,
            page.text_characters,
            if page.blank { ", BLANK" } else { "" },
            if page.clipped.is_empty() { String::new() } else { format!(", {} run(s) drawn outside the page", page.clipped.len()) },
            &page.image_sha256[..12]
        ));
    }
    out.push_str("Render checks:\n");
    out.push_str(&check_lines(&rendered.checks));
    out
}

pub(super) fn render_pages(deps: &Arc<RuntimeDeps>, call: &CallParams, session: &Session, tool_call: &ToolCall) -> Result<String, String> {
    let (id, version) = artifact_tools::parse_reference(tool_call, "artifact")?;
    let record = artifact_tools::resolve(deps, call, session, &id, version)?;
    let bytes = artifact_tools_load(deps, session, &record)?;
    if package::sniff(&bytes) != DetectedFormat::Docx {
        return Err(format!("{} is not a Word document; artifact.render lays out other formats", record.reference()));
    }
    let rendered = render_document(deps, session, &record, &bytes);
    let head = format!("{} (sha-256 {}), every page up to {}:\n", record.reference(), &record.sha256[..12], render::MAX_PAGES_PER_RENDER);
    let body = render_lines(&rendered);
    match rendered.outcome.state {
        render::RenderState::Refused | render::RenderState::Failed => Err(format!("{head}{body}")),
        // Not an error: the version exists and says why nobody looked at it.
        _ => Ok(format!("{head}{body}")),
    }
}

// ── artifact.validate_document ───────────────────────────────────────────

/// The P04 ladder and the document checks over one exact version, recorded.
pub(crate) struct DocumentValidation {
    pub validation_id: i64,
    pub accepted: bool,
    pub ladder: validation::ValidationReport,
    pub format: Vec<CheckResult>,
    pub render: Option<DocumentRender>,
}

pub(crate) fn validate_document_version(
    deps: &Arc<RuntimeDeps>,
    session: &Session,
    run_id: &str,
    record: &ArtifactRecord,
    wants_render: bool,
) -> Result<DocumentValidation, String> {
    let bytes = artifact_tools_load(deps, session, record)?;
    let dependencies = deps.conversation_artifacts.dependencies(&session.user.id, &record.reference()).map_err(|e| e.to_string())?;
    let check = artifact_tools::recheck(deps, session, record, &dependencies);
    let bound: std::collections::BTreeSet<String> = dependencies.iter().filter_map(|d| d.marker.clone()).collect();
    let template_id = record.template.as_ref().map(|t| t.id.clone());
    let ladder = validation::validate(
        &bytes,
        &validation::Context {
            recorded_sha256: Some(&record.sha256),
            claimed_mime: &record.mime,
            filename: record.filename.as_deref().or(Some(record.title.as_str())),
            template: template_id.as_deref(),
            bound_markers: (record.stage != Stage::Recorded).then_some(&bound),
            stale_because: Some(check.stale_because()),
        },
        None,
    );
    let resolver = RunResolver::new(deps, session, run_id, &record.conversation_id, dependencies.clone());
    let template = template_id.as_deref().and_then(checking_template);
    let format = document_checks::format_checks(&bytes, template, Some(&resolver));
    let rendered = wants_render.then(|| render_document(deps, session, record, &bytes));
    let render_checks: Vec<CheckResult> = match &rendered {
        Some(r) => r.checks.clone(),
        None => ["render.pages", "render.text_visible", "render.table_headers", "render.page_breaks"]
            .iter()
            .map(|name| CheckResult {
                check: (*name).into(),
                status: CheckStatus::Unavailable,
                severity: document_checks::CheckSeverity::Blocking,
                locations: Vec::new(),
                detail: "rendering was not asked for".into(),
            })
            .collect(),
    };
    // The ladder's own render rung is the document render here, one pass.
    let ladder_passed = [&ladder.file_created, &ladder.format_reopened, &ladder.content_checked].iter().all(|stage| stage.passed());
    let mut all = format.clone();
    all.extend(render_checks.iter().cloned());
    let accepted = ladder_passed && document_checks::accepted(&all);
    let mut report = serde_json::to_value(&ladder).unwrap_or(Value::Null);
    if let Value::Object(map) = &mut report {
        map.insert("documentChecks".into(), serde_json::to_value(&all).unwrap_or(Value::Null));
        map.insert("documentAccepted".into(), Value::Bool(accepted));
        if let Some(r) = &rendered {
            map.insert("documentRender".into(), json!({ "renderId": r.render_id, "state": r.outcome.state, "renderers": r.outcome.renderers, "fontFiles": r.outcome.font_files, "totalPages": r.outcome.total_pages }));
            let state = if r.outcome.state == render::RenderState::Rendered && document_checks::accepted(&render_checks) { "passed" } else if r.outcome.state == render::RenderState::Rendered { "failed" } else { "unavailable" };
            map.insert("renderChecked".into(), json!({ "state": state, "validator": r.outcome.renderers.iter().map(|x| format!("{} {}", x.name, x.version)).collect::<Vec<_>>().join(" + "), "detail": r.outcome.detail }));
        }
    }
    let validation_id = deps
        .conversation_artifacts
        .record_validation(&session.user.id, &record.reference(), &ladder.sha256, accepted, &report.to_string())
        .map_err(|e| format!("the validation ran and could not be recorded: {e}"))?;
    Ok(DocumentValidation { validation_id, accepted, ladder, format: all, render: rendered })
}

pub(super) fn validate_document(deps: &Arc<RuntimeDeps>, call: &CallParams, session: &Session, tool_call: &ToolCall) -> Result<String, String> {
    let (id, version) = artifact_tools::parse_reference(tool_call, "artifact")?;
    let record = artifact_tools::resolve(deps, call, session, &id, version)?;
    let wants_render = !matches!(tool_call.text("render").map(|r| r.trim().to_ascii_lowercase()).as_deref(), Some("no") | Some("false"));
    let result = validate_document_version(deps, session, &call.run_id, &record, wants_render)?;
    let mut out = format!(
        "Validation {} of {} (sha-256 {}, {}):\nFormat ladder:\n",
        result.validation_id,
        record.reference(),
        &record.sha256[..12],
        result.ladder.detected_format.label()
    );
    for (name, stage) in result.ladder.rungs().into_iter().take(3) {
        out.push_str(&format!("- {name}: {} — {}\n", stage.state.as_str(), stage.detail));
    }
    out.push_str("Document checks (format and render reported apart; unavailable is never a pass):\n");
    out.push_str(&check_lines(&result.format));
    if let Some(rendered) = &result.render {
        out.push_str(&render_lines(rendered).lines().filter(|l| !l.starts_with("- render.") && *l != "Render checks:").collect::<Vec<_>>().join("\n"));
        out.push('\n');
    }
    out.push_str(if result.accepted {
        "Accepted by the backend checks. It is still a candidate: the independent review (task.request_review) and a person decide the rest.\n"
    } else {
        "Not accepted. Fix what failed, answer what is missing, or say what could not be checked.\n"
    });
    Ok(out)
}
