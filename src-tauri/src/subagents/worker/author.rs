//! The Document Author (P09).
//!
//! Composes and repairs a Word deliverable **through the runtime's gateway**
//! ([`super::super::tool_port`]): every call is authorised against the plan
//! narrowed to this child's granted tools and recorded as a receipt, and every
//! version is registered by the same code that registers a parent's. The
//! worker never touches the artifact store or the writer directly.
//!
//! ## Without a model
//!
//! The specification is assembled from the task's shared memory, and only
//! from it: current facts and corrections as findings, each citing its item
//! at its revision; calculation records as the calculation section, each
//! citing the record; decisions as the recommendation; constraints as
//! assumptions. Every mandatory part the memory does not supply is a gap —
//! the recipient nobody named, the recommendation nobody decided. The worker
//! writes no business content of its own.
//!
//! ## Repairs
//!
//! Pointed at an existing version, the worker regenerates each section from
//! the memory as it is now and patches, against the exact version and hash,
//! every section whose citations changed — a correction that arrived while
//! the note was being written is how that happens. At most
//! [`MAX_REPAIRS`] patches, each a new version.
//!
//! It never issues a review verdict: it validates what it wrote and reports
//! each check as the tool reported it.

use std::collections::BTreeMap;

use serde_json::{json, Value};

use crate::identity::Session;
use crate::knowledge::graph::runtime_memory::{MemoryItem, MemoryKind};
use crate::orchestrator::tools::ToolName;
use crate::subagents::graph_io::TaskMemory;
use crate::subagents::inherit::EffectivePolicy;
use crate::subagents::packet::{ChildTaskPacket, InputRef};
use crate::subagents::result::{ArtifactVersion, Finding, ReceiptRef, ValidationCheck};
use crate::subagents::tool_port::{port_in, OpenRun, PortAnswer};

use super::{require, SpecialistWorker, Stopping, Work};

/// Bounded repairs: a document that three targeted patches did not settle
/// needs a person, not a fourth.
pub const MAX_REPAIRS: usize = 3;

const TEMPLATE: &str = "inspection_approval_note@1";

/// A section as the memory supplies it: blocks, or the gap that stands in
/// for them.
#[derive(Debug, Clone, PartialEq)]
struct Drafted {
    id: &'static str,
    blocks: Vec<Value>,
    gap: Option<String>,
    markers: Vec<String>,
}

impl Drafted {
    fn spec(&self) -> Value {
        match &self.gap {
            Some(gap) => json!({ "id": self.id, "gap": gap }),
            None => json!({ "id": self.id, "blocks": self.blocks }),
        }
    }
}

fn marker_of(item: &MemoryItem) -> String {
    format!("[M:{}@{}]", item.item_id, item.revision)
}

/// `calc-…` inside a calculation item's id (`mi-calc-…-xxxxxxxx`).
fn calc_id_of(item_id: &str) -> Option<String> {
    let rest = item_id.strip_prefix("mi-calc-")?;
    let id = rest.rsplit_once('-').map(|(id, _)| id).unwrap_or(rest);
    (id.len() == 16 && id.chars().all(|c| c.is_ascii_hexdigit())).then(|| format!("calc-{id}"))
}

/// Words that decide whether an item bears on the objective.
fn significant_words(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric() && c != '-')
        .filter(|w| w.len() >= 4)
        .map(|w| w.to_lowercase())
        .collect()
}

fn has_figure(text: &str) -> bool {
    !crate::artifacts::authoring::figures_in(text).is_empty()
}

/// A file name from the objective, safe for `document.compose`.
fn output_name(objective: &str) -> String {
    let slug: String = objective
        .split_whitespace()
        .take(6)
        .map(|w| w.chars().filter(|c| c.is_ascii_alphanumeric() || *c == '-').collect::<String>().to_lowercase())
        .filter(|w| !w.is_empty())
        .collect::<Vec<_>>()
        .join("-");
    let slug = if slug.is_empty() { "document".to_string() } else { slug.chars().take(60).collect() };
    format!("{slug}-note.docx")
}

/// A subject line from the objective, carrying no figure the evidence would
/// have to support: those belong in the findings, cited.
fn subject_line(objective: &str) -> Option<String> {
    let first = objective.split(". ").next().unwrap_or(objective).lines().next().unwrap_or_default().trim().trim_end_matches('.');
    let line = first.split_whitespace().collect::<Vec<_>>().join(" ");
    (line.chars().filter(|c| c.is_alphabetic()).count() >= 12 && !has_figure(&line)).then(|| {
        let mut chars = line.chars();
        match chars.next() {
            Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
            None => line,
        }
    })
}

/// The sections the memory supplies now.
fn draft(
    worker: &SpecialistWorker,
    packet: &ChildTaskPacket,
    session: &Session,
    items: &[MemoryItem],
) -> (BTreeMap<String, String>, Vec<Drafted>) {
    let wanted = significant_words(&packet.objective);
    let usable: Vec<&MemoryItem> = items.iter().filter(|i| i.status.usable_as_evidence()).collect();

    // Findings: corrections always; facts and observations that bear on the
    // objective or state a figure. Never a calculation or an artifact item.
    let mut findings: Vec<&MemoryItem> = usable
        .iter()
        .copied()
        .filter(|i| matches!(i.kind, MemoryKind::Fact | MemoryKind::Correction | MemoryKind::ToolObservation))
        .filter(|i| calc_id_of(&i.item_id).is_none() && !i.item_id.starts_with("mi-art-") && !i.item_id.starts_with("mi-gap-"))
        .filter(|i| {
            i.kind == MemoryKind::Correction
                || has_figure(&i.content)
                || significant_words(&i.content).iter().any(|w| wanted.contains(w))
        })
        .collect();
    findings.sort_by(|a, b| a.created_at.cmp(&b.created_at).then(a.item_id.cmp(&b.item_id)));
    findings.truncate(20);

    // Calculations: each record, cited exactly. From memory and from the
    // packet's own references.
    let mut calc_ids: Vec<String> = usable.iter().filter_map(|i| calc_id_of(&i.item_id)).collect();
    for input in &packet.inputs {
        if let InputRef::Calculation { calculation_id } = input {
            if !calc_ids.contains(calculation_id) {
                calc_ids.push(calculation_id.clone());
            }
        }
    }
    let mut calculation_lines = Vec::new();
    for id in &calc_ids {
        let Some(record) = worker.services.calculation_store.get(id, &session.user.id) else { continue };
        let Some(first) = record.first() else { continue };
        if !record.status.has_result() {
            continue;
        }
        calculation_lines.push(format!("{} gives {} = {} [C:{}]", record.equation, first.name, first.display, record.id));
    }

    let decisions: Vec<&MemoryItem> = usable.iter().copied().filter(|i| i.kind == MemoryKind::Decision).collect();
    let constraints: Vec<&MemoryItem> = usable.iter().copied().filter(|i| i.kind == MemoryKind::Constraint).collect();

    let bullets = |list: &[&MemoryItem]| -> (Vec<Value>, Vec<String>) {
        let markers: Vec<String> = list.iter().map(|i| marker_of(i)).collect();
        let items: Vec<String> = list.iter().map(|i| format!("{} {}", i.content.trim().chars().take(400).collect::<String>(), marker_of(i))).collect();
        (vec![json!({ "kind": "bullets", "items": items })], markers)
    };

    let mut sections = Vec::new();
    sections.push(match subject_line(&packet.objective) {
        Some(line) => Drafted { id: "subject", blocks: vec![json!({ "kind": "paragraph", "text": line })], gap: None, markers: Vec::new() },
        None => Drafted {
            id: "subject",
            blocks: Vec::new(),
            gap: Some("a subject line: the request does not state one this worker can use without restating a figure".into()),
            markers: Vec::new(),
        },
    });
    sections.push(if findings.is_empty() {
        Drafted { id: "findings", blocks: Vec::new(), gap: Some("findings: the task's memory holds no current fact bearing on this request".into()), markers: Vec::new() }
    } else {
        let (blocks, markers) = bullets(&findings);
        Drafted { id: "findings", blocks, gap: None, markers }
    });
    if !calculation_lines.is_empty() {
        let markers = calc_ids.iter().map(|id| format!("[C:{id}]")).collect();
        sections.push(Drafted { id: "calculation", blocks: vec![json!({ "kind": "bullets", "items": calculation_lines })], gap: None, markers });
    }
    sections.push(if decisions.is_empty() {
        Drafted {
            id: "recommendation",
            blocks: Vec::new(),
            gap: Some("a recommendation: no decision has been recorded for this task, and the author does not make one".into()),
            markers: Vec::new(),
        }
    } else {
        let (blocks, markers) = bullets(&decisions);
        Drafted { id: "recommendation", blocks, gap: None, markers }
    });
    sections.push(if constraints.is_empty() {
        Drafted {
            id: "assumptions",
            blocks: vec![json!({ "kind": "paragraph", "text": "No assumption beyond the cited records was made in writing this note." })],
            gap: None,
            markers: Vec::new(),
        }
    } else {
        let (blocks, markers) = bullets(&constraints);
        Drafted { id: "assumptions", blocks, gap: None, markers }
    });

    // The recipient, only where somebody named one.
    let mut fields = BTreeMap::new();
    let recipient = usable.iter().find_map(|i| {
        let lower = i.content.to_lowercase();
        ["recipient:", "addressed to:"].iter().find_map(|prefix| lower.starts_with(prefix).then(|| i.content[prefix.len()..].trim().to_string()))
    });
    fields.insert("recipient".to_string(), recipient.filter(|r| !r.is_empty()).unwrap_or_else(|| "?".to_string()));
    (fields, sections)
}

/// `art-…@N (sha-256 <64 hex>)`, wherever it appears in a tool's answer.
fn versions_in(text: &str) -> Vec<ArtifactVersion> {
    let mut out = Vec::new();
    let mut rest = text;
    while let Some(at) = rest.find("art-") {
        let candidate = &rest[at..];
        let id_end = candidate.find('@').unwrap_or(0);
        let id = &candidate[..id_end];
        let after = &candidate[id_end.saturating_add(1)..];
        let digits: String = after.chars().take_while(|c| c.is_ascii_digit()).collect();
        let tail = &after[digits.len()..];
        if id_end > 4 && !digits.is_empty() && id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-') {
            if let Some(hash) = tail.strip_prefix(" (sha-256 ").map(|h| h.chars().take(64).collect::<String>()) {
                if hash.len() == 64 && hash.chars().all(|c| c.is_ascii_hexdigit()) {
                    let version = ArtifactVersion { artifact_id: id.to_string(), version: digits.parse().unwrap_or(0), sha256: hash };
                    if !out.contains(&version) {
                        out.push(version);
                    }
                }
            }
        }
        rest = &rest[at + 4..];
    }
    out
}

/// The checks a validation reported, as it reported them.
fn checks_in(text: &str) -> Vec<ValidationCheck> {
    text.lines()
        .filter_map(|line| {
            let line = line.strip_prefix("- ")?;
            let (check, rest) = line.split_once(" [")?;
            if !(check.starts_with("document.") || check.starts_with("render.")) {
                return None;
            }
            let (_, rest) = rest.split_once("]: ")?;
            let (status, detail) = rest.split_once(" — ").unwrap_or((rest, ""));
            let outcome = match status.trim() {
                "pass" => "passed",
                "FAIL" => "failed",
                _ => "blocked",
            };
            Some(ValidationCheck { check_id: check.to_string(), outcome: outcome.into(), detail: detail.chars().take(400).collect() })
        })
        .collect()
}

/// After a model loop: the versions its compose and patch calls registered.
pub(super) fn collect_versions(work: &mut Work) {
    let texts: Vec<String> = work.findings.iter().map(|f| f.statement.clone()).collect();
    for text in texts {
        if text.starts_with("document.compose") || text.starts_with("document.patch_section") {
            for version in versions_in(&text) {
                if !work.artifacts.contains(&version) {
                    work.artifacts.push(version);
                }
            }
        }
    }
}

/// Keeps a child run bound to its parent's conversation for as long as it
/// lives, so the versions a model loop registers land where the parent is.
pub(super) struct BoundConversation {
    conversations: std::sync::Arc<crate::agent_runtime::conversations::RunToConversation>,
    child: String,
}

impl Drop for BoundConversation {
    fn drop(&mut self) {
        self.conversations.unbind(&self.child);
        crate::subagents::tool_port::release(&self.child);
    }
}

impl SpecialistWorker {
    pub(super) fn bind_conversation(&self, packet: &ChildTaskPacket) -> Option<BoundConversation> {
        let conversations = self.services.analyst.as_ref()?.conversations.clone();
        let conversation = conversations.lookup(&packet.parent_run_id)?;
        conversations.bind(&packet.child_id, &conversation);
        crate::subagents::tool_port::adopt(&packet.child_id, &packet.parent_run_id);
        Some(BoundConversation { conversations, child: packet.child_id.clone() })
    }

    /// Composes a note from the task's memory, or repairs the version the
    /// packet names, through the gateway.
    pub(super) async fn author(
        &self,
        packet: &ChildTaskPacket,
        policy: &EffectivePolicy,
        session: &Session,
        memory: Option<&TaskMemory>,
        cancel: &Stopping,
    ) -> Result<Work, String> {
        require(policy, ToolName::DocumentCompose)?;
        require(policy, ToolName::ArtifactValidateDocument)?;
        let memory = memory.ok_or(
            "this deployment has no runtime memory graph, so there is no task memory to write a document from. \
             Nothing was written.",
        )?;
        let port = port_in(&self.services.tools)?;
        port.open(
            &packet.child_id,
            &packet.parent_run_id,
            &policy.inherited.workspace_root,
            &policy.tools,
            policy.limits.max_turns.max(4),
            std::time::Duration::from_secs(policy.limits.max_duration_seconds.max(30)),
        )?;
        let run = OpenRun::new(port, &packet.child_id);
        let mut work = Work::new();
        let items = memory.read(session).map_err(|unavailable| unavailable.explain())?;
        let (fields, sections) = draft(self, packet, session, &items);
        cancel.check()?;

        let receipt = |work: &mut Work, tool: ToolName, answer: &PortAnswer| {
            match answer.event_seq {
                Some(event_seq) if event_seq > 0 => {
                    work.receipts.push(ReceiptRef { run_id: packet.child_id.clone(), tool: tool.as_str().to_string(), event_seq })
                }
                _ => work.uncertainty.push(format!("the {} call's event could not be read back, so it carries no receipt", tool.as_str())),
            }
        };

        let base = packet.inputs.iter().find_map(|input| match input {
            InputRef::Artifact { artifact_id, revision, sha256 } => Some((artifact_id.clone(), *revision, sha256.clone())),
            _ => None,
        });
        let current = match base {
            // ── Repair ───────────────────────────────────────────────────
            Some((artifact_id, revision, sha256)) => {
                let manifest = run
                    .call(ToolName::ArtifactManifest, json!({ "artifact": format!("{artifact_id}@{revision}") }))
                    .await?;
                work.turns += 1;
                receipt(&mut work, ToolName::ArtifactManifest, &manifest);
                let body: Value = manifest
                    .text
                    .split_once('\n')
                    .and_then(|(_, json)| serde_json::from_str(json).ok())
                    .unwrap_or(Value::Null);
                let written: BTreeMap<String, Vec<String>> = body
                    .get("sections")
                    .and_then(Value::as_array)
                    .map(|sections| {
                        sections
                            .iter()
                            .filter_map(|s| {
                                let id = s.get("sectionId")?.as_str()?.to_string();
                                let mut markers: Vec<String> = s.get("markers")?.as_array()?.iter().filter_map(|m| m.as_str().map(str::to_string)).collect();
                                if s.get("gap").is_some_and(|g| !g.is_null()) {
                                    markers.push("gap".into());
                                }
                                Some((id, markers))
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                if written.is_empty() {
                    return Err(format!(
                        "{artifact_id}@{revision} records no section lineage, so which of its sections rest on what cannot be told; \
                         it is not repaired section by section. Compose a new version instead."
                    ));
                }
                // The hash the version holds, from its manifest; a hash the
                // parent named must agree with it, or the parent saw other bytes.
                let recorded = body.get("sha256").and_then(Value::as_str).unwrap_or_default().to_string();
                if !sha256.is_empty() && sha256 != recorded {
                    return Err(format!(
                        "{artifact_id}@{revision} holds sha-256 {recorded}, not the {sha256} this job was given; nothing was repaired"
                    ));
                }
                let mut version = ArtifactVersion { artifact_id: artifact_id.clone(), version: revision, sha256: recorded };
                let mut repairs = 0usize;
                for section in &sections {
                    let Some(before) = written.get(section.id) else { continue };
                    let mut now = section.markers.clone();
                    if section.gap.is_some() {
                        now.push("gap".into());
                    }
                    let (mut a, mut b) = (before.clone(), now.clone());
                    a.sort();
                    b.sort();
                    if a == b {
                        continue;
                    }
                    if repairs == MAX_REPAIRS {
                        work.missing.push(format!(
                            "section {} rests on changed sources and was not repaired: {MAX_REPAIRS} repairs is the limit",
                            section.id
                        ));
                        continue;
                    }
                    cancel.check()?;
                    let mut args = json!({
                        "artifact": format!("{}@{}", version.artifact_id, version.version),
                        "sha256": version.sha256,
                        "section": section.id,
                    });
                    match &section.gap {
                        Some(gap) => args["gap"] = json!(gap),
                        None => args["blocks"] = json!(section.blocks),
                    }
                    let patched = run.call(ToolName::DocumentPatchSection, args).await?;
                    work.turns += 1;
                    repairs += 1;
                    receipt(&mut work, ToolName::DocumentPatchSection, &patched);
                    let next = versions_in(patched.text.lines().next().unwrap_or_default())
                        .into_iter()
                        .find(|v| v.artifact_id == version.artifact_id && v.version > version.version)
                        .ok_or_else(|| format!("the patch of {} did not name the version it wrote", section.id))?;
                    work.findings.push(Finding {
                        statement: format!(
                            "Repaired section {} of {}@{} as {}@{}: it now rests on {}.",
                            section.id,
                            version.artifact_id,
                            version.version,
                            next.artifact_id,
                            next.version,
                            if section.gap.is_some() { "a stated gap".to_string() } else { section.markers.join(" ") }
                        ),
                        evidence: Vec::new(),
                    });
                    version = next;
                }
                // A calculation the base quotes and the memory no longer
                // supports (its record rests on a corrected input and nothing
                // has recomputed it) is not left standing: it becomes a gap.
                let mut orphaned: Vec<Drafted> = Vec::new();
                if written.contains_key("calculation") && !sections.iter().any(|s| s.id == "calculation") {
                    orphaned.push(Drafted {
                        id: "calculation",
                        blocks: Vec::new(),
                        gap: Some("a current calculation: the one quoted rests on an input that has since been corrected, and it has not been recomputed".into()),
                        markers: Vec::new(),
                    });
                }
                for section in &orphaned {
                    if written.get(section.id).is_some_and(|m| m == &vec!["gap".to_string()]) {
                        continue;
                    }
                    if repairs == MAX_REPAIRS {
                        work.missing.push(format!("section {} was not repaired: {MAX_REPAIRS} repairs is the limit", section.id));
                        continue;
                    }
                    let args = json!({
                        "artifact": format!("{}@{}", version.artifact_id, version.version),
                        "sha256": version.sha256,
                        "section": section.id,
                        "gap": section.gap,
                    });
                    let patched = run.call(ToolName::DocumentPatchSection, args).await?;
                    work.turns += 1;
                    repairs += 1;
                    receipt(&mut work, ToolName::DocumentPatchSection, &patched);
                    let next = versions_in(patched.text.lines().next().unwrap_or_default())
                        .into_iter()
                        .find(|v| v.artifact_id == version.artifact_id && v.version > version.version)
                        .ok_or_else(|| format!("the patch of {} did not name the version it wrote", section.id))?;
                    work.findings.push(Finding {
                        statement: format!("Section {} of {}@{} rests on a corrected input; it is now a stated gap in {}@{}.", section.id, version.artifact_id, version.version, next.artifact_id, next.version),
                        evidence: Vec::new(),
                    });
                    work.missing.push(section.gap.clone().unwrap_or_default());
                    version = next;
                }
                if repairs == 0 {
                    work.findings.push(Finding {
                        statement: format!(
                            "{}@{} already rests on the task's current memory; no section needed repair.",
                            version.artifact_id, version.version
                        ),
                        evidence: Vec::new(),
                    });
                }
                version
            }
            // ── Compose ──────────────────────────────────────────────────
            None => {
                let subject = subject_line(&packet.objective).unwrap_or_else(|| "The requested approval note".into());
                let args = json!({
                    "template": TEMPLATE,
                    "title": format!("Approval note: {}", subject.trim_end_matches('.')),
                    "audience": format!("the person who requested task {}", packet.task_id),
                    "output": output_name(&packet.objective),
                    "fields": fields,
                    "sections": sections.iter().map(Drafted::spec).collect::<Vec<_>>(),
                });
                let composed = run.call(ToolName::DocumentCompose, args).await?;
                work.turns += 1;
                receipt(&mut work, ToolName::DocumentCompose, &composed);
                let version = versions_in(&composed.text)
                    .into_iter()
                    .last()
                    .ok_or("document.compose did not say which version it registered")?;
                work.findings.push(Finding {
                    statement: format!(
                        "Composed {}@{} (sha-256 {}) from {TEMPLATE}: {}.",
                        version.artifact_id,
                        version.version,
                        &version.sha256[..12],
                        sections
                            .iter()
                            .map(|s| match &s.gap {
                                Some(_) => format!("{} (gap)", s.id),
                                None if s.markers.is_empty() => s.id.to_string(),
                                None => format!("{} ({})", s.id, s.markers.join(" ")),
                            })
                            .collect::<Vec<_>>()
                            .join(", ")
                    ),
                    evidence: Vec::new(),
                });
                version
            }
        };

        // Gaps, named: the result is partial until each is answered.
        if fields.get("recipient").map(String::as_str) == Some("?") {
            work.missing.push("the recipient: nobody named one, and the author does not guess".into());
        }
        for section in &sections {
            if let Some(gap) = &section.gap {
                work.missing.push(gap.clone());
            }
        }

        // Validated and rendered, every check reported as the tool reported it.
        cancel.check()?;
        let validated = run
            .call(ToolName::ArtifactValidateDocument, json!({ "artifact": format!("{}@{}", current.artifact_id, current.version) }))
            .await?;
        work.turns += 1;
        receipt(&mut work, ToolName::ArtifactValidateDocument, &validated);
        work.validation = checks_in(&validated.text);
        let accepted = validated.text.contains("Accepted by the backend checks");
        work.findings.push(Finding {
            statement: format!(
                "{}@{} {} by the backend checks ({} check(s): {} passed, {} failed, {} unavailable). It is a candidate; \
                 the independent review is the reviewer's.",
                current.artifact_id,
                current.version,
                if accepted { "is accepted" } else { "is not accepted" },
                work.validation.len(),
                work.validation.iter().filter(|c| c.outcome == "passed").count(),
                work.validation.iter().filter(|c| c.outcome == "failed").count(),
                work.validation.iter().filter(|c| c.outcome == "blocked").count(),
            ),
            evidence: Vec::new(),
        });
        work.artifacts.push(current);
        work.confidence = if accepted { 1.0 } else { 0.5 };
        Ok(work)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_version_and_its_hash_are_read_from_a_tool_answer() {
        let hash = "a".repeat(64);
        let text = format!("Wrote art-file-0123@2 (sha-256 {hash}) from art-file-0123@1: section \"findings\" replaced.");
        let found = versions_in(&text);
        assert_eq!(found, vec![ArtifactVersion { artifact_id: "art-file-0123".into(), version: 2, sha256: hash }]);
        assert!(versions_in("art-file-0123@2 (sha-256 abc)").is_empty(), "a short hash is not a hash");
    }

    #[test]
    fn checks_are_read_as_reported_and_unavailable_is_not_passed() {
        let text = "- document.sections [Blocking]: pass — all present\n- render.pages [Blocking]: unavailable — no renderer\n- document.gaps [Blocking]: FAIL — section:recommendation";
        let checks = checks_in(text);
        assert_eq!(checks.iter().map(|c| c.outcome.as_str()).collect::<Vec<_>>(), vec!["passed", "blocked", "failed"]);
    }

    #[test]
    fn a_subject_line_never_restates_a_figure() {
        assert_eq!(subject_line("Write an approval note on V-101 shell wall loss").as_deref(), Some("Write an approval note on V-101 shell wall loss"));
        assert!(subject_line("8.2 mm").is_none());
        assert_eq!(calc_id_of("mi-calc-0123456789abcdef-deadbeef").as_deref(), Some("calc-0123456789abcdef"));
        assert_eq!(output_name("Write an approval note on V-101!"), "write-an-approval-note-on-v-101-note.docx");
    }
}
