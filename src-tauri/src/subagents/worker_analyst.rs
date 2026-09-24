//! The document extractor over attached documents: the Document & Vision
//! Analyst's worker path (P06).
//!
//! A child of `worker` so it shares that module's `Work`, receipt and
//! cancellation plumbing rather than restating them.
//!
//! ## What it publishes, and as what
//!
//! One findings pass per document ([`ExtractionService::findings`]): layout,
//! local OCR of scanned pages (bounded, cached), then the requested fields.
//! The pass is recorded as one `media.extract_findings` receipt on the child's
//! run, and what it established is published to the task's shared memory:
//!
//! | Published | Kind | Receipt | Lands as |
//! |---|---|---|---|
//! | a field value read in a cleanly read region | tool observation | the pass | admitted |
//! | a field not found on the pages read | tool observation | the pass | admitted |
//! | a region that could not be read | tool observation | the pass | admitted |
//! | a value only in a looped, truncated or malformed read | open question | none | proposed |
//! | a vision model's interpretation | fact, headed PROPOSAL | none | proposed |
//!
//! Each carries the document hash and a locator naming the page, the region
//! id, the box and its coordinate space, so the exact evidence is one lookup
//! away. No claim carries a confidence number: none was measured.
//!
//! [`ExtractionService::findings`]: crate::extraction::service::ExtractionService::findings

use std::time::{Duration, Instant};

use crate::agent_runtime::extraction_tools::{render_findings, ANALYST_DETENT};
use crate::extraction::fields::FieldState;
use crate::extraction::ocr::HeldCard;
use crate::extraction::regions::RegionStatus;
use crate::extraction::service::MAX_PAGES_PER_CALL;
use crate::identity::Session;
use crate::knowledge::graph::runtime_memory::{MemoryKind, SourceRef};
use crate::orchestrator::tools::ToolName;

use super::super::packet::{ChildTaskPacket, InputRef};
use super::super::result::{EvidenceRef, Finding};
use super::super::scheduling::ModelLease;
use super::{remaining, require, Claim, EffectivePolicy, SpecialistWorker, Stopping, Work};

/// The fields a parent asked for, written after `fields:` in the objective.
///
/// A plain convention rather than a parse of prose: "Extract fields: design
/// pressure, tag, material" names three fields and nothing else, and a worker
/// that guessed fields from free text would be guessing what to look for.
pub fn requested_fields(objective: &str) -> Vec<String> {
    let lower = objective.to_ascii_lowercase();
    let Some(at) = lower.find("fields:") else {
        return Vec::new();
    };
    let rest = &objective[at + "fields:".len()..];
    let end = rest
        .find(|c| c == '\n' || c == '.')
        .unwrap_or(rest.len());
    rest[..end]
        .split([',', ';'])
        .flat_map(|part| part.split(" and "))
        .map(|part| part.trim().trim_matches('"').to_string())
        .filter(|part| !part.is_empty())
        .collect()
}

/// The detent whose OCR model is `model_id`, preferring the analyst's own.
pub fn detent_for_model(model_id: &str) -> Option<crate::ai_engine::ocr_profile::OcrDetent> {
    use crate::ai_engine::ocr_profile::OcrDetent;
    if crate::commands::ocr::ocr_model_id(ANALYST_DETENT) == model_id {
        return Some(ANALYST_DETENT);
    }
    [OcrDetent::Fast, OcrDetent::Maximum, OcrDetent::Fastest]
        .into_iter()
        .find(|detent| crate::commands::ocr::ocr_model_id(*detent) == model_id)
}

/// A question for a vision-ready model, written after `question:`.
pub fn requested_question(objective: &str) -> Option<String> {
    let lower = objective.to_ascii_lowercase();
    let at = lower.find("question:")?;
    let rest = objective[at + "question:".len()..].trim();
    let end = rest.find('\n').unwrap_or(rest.len());
    let question = rest[..end].trim();
    (!question.is_empty()).then(|| question.to_string())
}

impl SpecialistWorker {
    /// Whether this child is pointed at an attached document this worker can
    /// reach.
    pub(super) fn reads_attached_documents(&self, packet: &ChildTaskPacket) -> bool {
        self.profile == "document-extractor"
            && self.services.analyst.is_some()
            && packet
                .inputs
                .iter()
                .any(|input| matches!(input, InputRef::Document { .. }))
    }

    pub(super) async fn extract_documents(
        &self,
        packet: &ChildTaskPacket,
        policy: &EffectivePolicy,
        session: &Session,
        cancel: &Stopping,
        lease: Option<&ModelLease>,
    ) -> Result<Work, String> {
        require(policy, ToolName::MediaExtractFindings)?;
        let analyst = self
            .services
            .analyst
            .as_ref()
            .ok_or("this deployment has no document analyst service")?;
        let conversation = analyst
            .conversations
            .lookup(&packet.parent_run_id)
            .ok_or_else(|| {
                "the task that sent this worker is not attached to a conversation, so the \
                 documents it names cannot be reached"
                    .to_string()
            })?;
        let wanted = requested_fields(&packet.objective);
        let question = requested_question(&packet.objective);
        let held = lease.map(|lease| HeldCard {
            model_id: lease.model_id.clone(),
            exclusive: lease.exclusive(),
        });
        // Read at the detent whose model this worker already holds, when it
        // holds an OCR model: asking for the other tier would mean waiting on a
        // card this worker has itself, which the OCR service refuses rather
        // than deadlock on.
        let detent = lease
            .and_then(|lease| detent_for_model(&lease.model_id))
            .unwrap_or(ANALYST_DETENT);
        let deadline = Instant::now() + remaining(packet).min(Duration::from_secs(110));

        let mut work = Work::new();
        let (mut pages_asked, mut pages_read) = (0u32, 0u32);
        if wanted.is_empty() {
            work.uncertainty.push(
                "no fields were named (write them after `fields:` in the objective), so this \
                 reports what was read and what was not, and extracts no values"
                    .to_string(),
            );
        }

        for input in &packet.inputs {
            let InputRef::Document { sha256, page } = input else { continue };
            cancel.check()?;
            work.turns += 1;
            let doc = match analyst
                .extraction
                .authorise(&analyst.documents, sha256, &session.user.id, &conversation)
            {
                Ok(doc) => doc,
                Err(why) => {
                    work.uncertainty.push(format!("document {}… was not read: {why}", &sha256[..12.min(sha256.len())]));
                    continue;
                }
            };
            let (from, to) = match page {
                Some(page) => (*page, *page),
                None => (1, doc.pages.min(MAX_PAGES_PER_CALL)),
            };
            if page.is_none() && doc.pages > MAX_PAGES_PER_CALL {
                work.uncertainty.push(format!(
                    "{} has {} pages and pages 1-{MAX_PAGES_PER_CALL} were examined; pages beyond \
                     that were not, so nothing here speaks for them",
                    doc.name, doc.pages
                ));
            }
            pages_asked += to - from + 1;
            let report = match analyst
                .extraction
                .findings(
                    &doc,
                    from,
                    to,
                    &wanted,
                    question.as_deref(),
                    &[],
                    detent,
                    held.as_ref(),
                    &cancel.own,
                    deadline,
                )
                .await
            {
                Ok(report) => report,
                Err(why) => {
                    work.uncertainty.push(format!("{} was not read: {why}", doc.name));
                    continue;
                }
            };

            let rendered = render_findings(&report);
            let receipt = self.receipt(
                packet,
                ToolName::MediaExtractFindings,
                &format!("findings:{}:{from}-{to}", doc.sha256),
                &rendered,
                &session.user.id,
                &mut work,
            );
            let extraction_revision = |region_id: &str| {
                crate::extraction::regions::RegionStore::new(analyst.extraction.documents_root())
                    .load(&doc.sha256)
                    .ok()
                    .and_then(|held| held.regions.get(region_id).cloned())
                    .map(|region| match region.cache_key {
                        Some(key) => format!("ocr cache {}", &key[..16]),
                        None => format!("{} {}", region.extractor.name, region.extractor.version),
                    })
            };
            let key = |suffix: &str| format!("{}:{}:{suffix}", packet.idempotency_key, &doc.sha256[..16]);

            // -- Fields --------------------------------------------------------
            for field in &report.fields {
                match field.state {
                    FieldState::NotFound => {
                        let unread: Vec<String> = report
                            .coverage
                            .iter()
                            .filter(|p| p.route == "unread")
                            .map(|p| p.page.to_string())
                            .collect();
                        let statement = format!(
                            "{}: \"{}\" was not found on the pages read ({from}-{to}){}",
                            doc.name,
                            field.field,
                            if unread.is_empty() {
                                String::new()
                            } else {
                                format!("; page(s) {} were not read, so absence is not established", unread.join(", "))
                            }
                        );
                        work.findings.push(Finding {
                            statement: statement.clone(),
                            evidence: Vec::new(),
                        });
                        work.claims.push(Claim {
                            kind: MemoryKind::ToolObservation,
                            content: statement,
                            sources: vec![SourceRef {
                                sha256: doc.sha256.clone(),
                                locator: format!("pages {from}-{to}"),
                                extraction_revision: None,
                            }],
                            artifacts: Vec::new(),
                            confidence: None,
                            causal_parents: Vec::new(),
                            idempotency_key: key(&format!("absent:{}", field.field)),
                            receipt: receipt.clone(),
                            depends_on: Vec::new(),
                        });
                    }
                    _ => {
                        for value in &field.values {
                            let clean = value.status == RegionStatus::Read;
                            let locator = format!(
                                "page {} region {} {} {}",
                                value.page,
                                value.region_id,
                                value.bbox.describe(),
                                value.coord_space.label()
                            );
                            let statement = format!(
                                "{}{}: {} = \"{}\" ({}; {}{})",
                                if clean { "" } else { "UNCERTAIN — " },
                                doc.name,
                                field.field,
                                value.value,
                                locator,
                                value.method.label(),
                                if clean { String::new() } else { format!("; the read was {}", value.status.label()) }
                            );
                            work.findings.push(Finding {
                                statement: statement.clone(),
                                evidence: vec![EvidenceRef {
                                    marker: None,
                                    document_sha256: doc.sha256.clone(),
                                    page: Some(value.page),
                                    citation: format!("{} {locator}", doc.name),
                                }],
                            });
                            work.claims.push(Claim {
                                // A value from a read that was cut is a question
                                // for a person, not something a tool established.
                                kind: if clean { MemoryKind::ToolObservation } else { MemoryKind::OpenQuestion },
                                content: statement,
                                sources: vec![SourceRef {
                                    sha256: doc.sha256.clone(),
                                    locator,
                                    extraction_revision: extraction_revision(&value.region_id),
                                }],
                                artifacts: Vec::new(),
                                confidence: None,
                                causal_parents: Vec::new(),
                                idempotency_key: key(&format!("{}:{}", field.field, value.region_id)),
                                receipt: if clean { receipt.clone() } else { None },
                                depends_on: Vec::new(),
                            });
                        }
                        if field.state == FieldState::Conflicting {
                            work.uncertainty.push(format!(
                                "{}: \"{}\" has more than one value on these pages; all are \
                                 reported and none is chosen",
                                doc.name, field.field
                            ));
                        }
                    }
                }
            }

            // -- Unreadable regions -------------------------------------------
            for region in report.unreadable.iter().take(24) {
                let locator = format!(
                    "page {} region {} {} {}",
                    region.page,
                    region.region_id,
                    region.bbox.describe(),
                    region.coord_space.label()
                );
                let statement = format!(
                    "{}: {} region at {locator} is {} ({}): {}",
                    doc.name,
                    region.label,
                    region.status.label(),
                    region.method.label(),
                    region.notes.first().cloned().unwrap_or_default()
                );
                work.uncertainty.push(statement.clone());
                work.claims.push(Claim {
                    kind: MemoryKind::ToolObservation,
                    content: statement,
                    sources: vec![SourceRef {
                        sha256: doc.sha256.clone(),
                        locator,
                        extraction_revision: extraction_revision(&region.region_id),
                    }],
                    artifacts: Vec::new(),
                    confidence: None,
                    causal_parents: Vec::new(),
                    idempotency_key: key(&format!("unreadable:{}", region.region_id)),
                    receipt: receipt.clone(),
                    depends_on: Vec::new(),
                });
            }

            // -- Coverage -------------------------------------------------------
            let read_here = report.coverage.iter().filter(|p| p.route != "unread").count() as u32;
            pages_read += read_here;
            let coverage = format!(
                "{}: pages {from}-{to} — {}",
                doc.name,
                report
                    .coverage
                    .iter()
                    .map(|p| format!(
                        "p.{} {}{}{}",
                        p.page,
                        match p.route.as_str() {
                            "embedded-text" => "text layer",
                            "ocr" => "local OCR",
                            _ => "NOT READ",
                        },
                        if p.looped || p.truncated { " (incomplete)" } else { "" },
                        if p.unreadable > 0 {
                            format!(" ({} unreadable region(s))", p.unreadable)
                        } else {
                            String::new()
                        }
                    ))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            work.findings.push(Finding {
                statement: coverage.clone(),
                evidence: Vec::new(),
            });
            work.claims.push(Claim {
                kind: MemoryKind::ToolObservation,
                content: format!("coverage — {coverage}"),
                sources: vec![SourceRef {
                    sha256: doc.sha256.clone(),
                    locator: format!("pages {from}-{to}"),
                    extraction_revision: None,
                }],
                artifacts: Vec::new(),
                confidence: None,
                causal_parents: Vec::new(),
                idempotency_key: key(&format!("coverage:{from}-{to}")),
                receipt: receipt.clone(),
                depends_on: Vec::new(),
            });
            for page in report.coverage.iter().filter(|p| p.route == "unread") {
                work.uncertainty.push(format!(
                    "{} page {} was not read: {}",
                    doc.name,
                    page.page,
                    page.reason.clone().unwrap_or_default()
                ));
            }

            // -- Interpretation: proposals only ---------------------------------
            if let Some(interpretation) = &report.interpretation {
                for proposal in &interpretation.proposals {
                    let locator = format!(
                        "page {} region {} {} {} (crop {})",
                        proposal.page,
                        proposal.region_id,
                        proposal.bbox.describe(),
                        proposal.coord_space.label(),
                        proposal.crop_id.as_deref().unwrap_or("-")
                    );
                    let statement = format!(
                        "PROPOSAL (vision-model inference by {}, not a transcription and not \
                         verified): {} {locator}: {}",
                        interpretation.model_id.as_deref().unwrap_or("an unrecorded model"),
                        doc.name,
                        proposal.text.chars().take(600).collect::<String>()
                    );
                    work.claims.push(Claim {
                        kind: MemoryKind::Fact,
                        content: statement,
                        sources: vec![SourceRef {
                            sha256: doc.sha256.clone(),
                            locator,
                            extraction_revision: Some(proposal.extractor.version.clone()),
                        }],
                        artifacts: Vec::new(),
                        confidence: None,
                        causal_parents: Vec::new(),
                        idempotency_key: key(&format!("proposal:{}", proposal.region_id)),
                        // No receipt: a model's description is not something a
                        // tool established, so it is published as a proposal.
                        receipt: None,
                        depends_on: Vec::new(),
                    });
                }
                if let Some(why) = &interpretation.refused {
                    work.uncertainty.push(format!("no interpretation of {}: {why}", doc.name));
                }
            }
        }

        // Workspace files in the same packet are read the way they always were.
        if packet
            .inputs
            .iter()
            .any(|input| matches!(input, InputRef::WorkspaceFile { .. }))
        {
            match self.extract(packet, policy, session, cancel) {
                Ok(files) => {
                    work.findings.extend(files.findings);
                    work.claims.extend(files.claims);
                    work.uncertainty.extend(files.uncertainty);
                    work.turns += files.turns;
                }
                Err(why) => work.uncertainty.push(why),
            }
        }

        // The result's confidence field is required by its schema. What goes
        // in it is the one thing measured: the fraction of asked-for pages that
        // were read. Said, so nobody takes it for a calibrated probability.
        work.confidence = if pages_asked == 0 {
            0.0
        } else {
            pages_read as f32 / pages_asked as f32
        };
        work.uncertainty.push(format!(
            "result confidence is page coverage ({pages_read} of {pages_asked} page(s) read), \
             not a calibrated probability; no extracted value carries a confidence number"
        ));
        Ok(work)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fields_and_a_question_are_read_from_the_objective_and_only_from_there() {
        assert_eq!(
            requested_fields("Read the P&ID. Fields: design pressure, tag; line size and material."),
            vec!["design pressure", "tag", "line size", "material"]
        );
        assert!(requested_fields("Read the P&ID and tell me the pressure").is_empty());
        assert_eq!(
            requested_question("fields: tag\nquestion: which valve is the label next to?").as_deref(),
            Some("which valve is the label next to?")
        );
        assert_eq!(requested_question("no question here"), None);
    }

    #[test]
    fn a_worker_holding_an_ocr_model_reads_at_that_models_detent() {
        use crate::ai_engine::ocr_profile::OcrDetent;
        assert_eq!(detent_for_model("unlimited-ocr-q6-k"), Some(OcrDetent::Detailed));
        assert_eq!(detent_for_model("unlimited-ocr-q4-k-m"), Some(OcrDetent::Fast));
        assert_eq!(detent_for_model("gemma-4-e4b"), None);
    }
}
