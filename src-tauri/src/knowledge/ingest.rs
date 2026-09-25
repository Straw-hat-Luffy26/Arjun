//! Taking a collection from a folder to something searchable.
//!
//! Five steps per document, in this order and for these reasons:
//!
//! 1. **Hold the original.** Before anything is derived from it, so a citation
//!    always has evidence behind it even if every later step fails.
//! 2. **Read it.** Text, confidence, and which pages could not be read.
//! 3. **Store the extraction** beside the original, so a better engine can
//!    re-read the same bytes later.
//! 4. **Chunk it** at the document's own boundaries.
//! 5. **Index it** under the collection's classification.
//!
//! ## One bad document does not stop a sync
//!
//! A share of ten thousand files will contain a corrupt PDF, a file somebody has
//! open in Word, and something with a `.pdf` extension that is not a PDF. A
//! pipeline that aborts on the first of those never finishes a real collection.
//! Each failure is recorded against its file and the run continues, so the
//! outcome says what was read *and* what was not.
//!
//! ## Nothing is silently good
//!
//! The outcome carries pages that could not be read and documents that contain
//! instruction-like text. Both are counted rather than buried, because a
//! collection that indexed nine hundred documents and failed to read a hundred
//! pages is not the same as one that read everything, and the difference must
//! not be discoverable only by noticing an answer is thin.

use std::path::Path;

use anyhow::Result;
use serde::{Deserialize, Serialize};

use super::connector::{Collection, DiscoveredFile, SyncPlan};
use super::index::{Retirement, SourceChange};
use super::{chunk_document, KnowledgeIndex};
use crate::audit::{AuditKind, AuditService};
use crate::documents::{DocumentStore, ExtractedDocument};

/// Reading one document. A trait so the pipeline can be tested without spawning
/// a Python sidecar — the sidecar is exercised by its own tests.
pub trait DocumentReader {
    fn read(&self, path: &Path) -> Result<ExtractedDocument>;
}

impl DocumentReader for crate::documents::DocumentService {
    fn read(&self, path: &Path) -> Result<ExtractedDocument> {
        self.extract(path)
    }
}

/// A document that could not be taken in, and why.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct IngestFailure {
    pub file: String,
    pub reason: String,
}

/// What a sync actually did.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct IngestOutcome {
    pub collection_id: String,
    pub documents_read: usize,
    pub chunks_indexed: usize,
    /// Pages no available engine could read. Counted, never buried.
    pub pages_needing_review: u32,
    /// Documents containing text aimed at the assistant rather than at a reader.
    pub flagged_for_injection: Vec<String>,
    pub failures: Vec<IngestFailure>,
    /// Documents no longer on the source, whose passages were retired.
    pub retired: Vec<String>,
    /// Content hashes of versions a changed file replaced. Still traceable,
    /// no longer answering questions.
    #[serde(default)]
    pub superseded_sha256s: Vec<String>,
    /// Content hashes that stopped answering because their file is gone. The
    /// caller tells the memory graph, so claims resting on them are
    /// invalidated rather than left looking established.
    #[serde(default)]
    pub withdrawn_sha256s: Vec<String>,
    /// Files whose bytes were unchanged although their size or time moved.
    #[serde(default)]
    pub unchanged_on_reread: usize,
    /// Files present but not of a type anything here can read.
    pub skipped: usize,
}

impl IngestOutcome {
    /// Whether somebody should look at this run before trusting the collection.
    pub fn needs_attention(&self) -> bool {
        self.pages_needing_review > 0
            || !self.flagged_for_injection.is_empty()
            || !self.failures.is_empty()
    }

    /// One honest line for the collection screen.
    pub fn summary(&self) -> String {
        let mut parts = vec![format!(
            "{} document(s) read, {} passage(s) indexed",
            self.documents_read, self.chunks_indexed
        )];

        if self.pages_needing_review > 0 {
            parts.push(format!("{} page(s) could not be read", self.pages_needing_review));
        }
        if !self.failures.is_empty() {
            parts.push(format!("{} document(s) failed", self.failures.len()));
        }
        if !self.flagged_for_injection.is_empty() {
            parts.push(format!(
                "{} document(s) contain instruction-like text",
                self.flagged_for_injection.len()
            ));
        }
        if !self.retired.is_empty() {
            parts.push(format!("{} retired", self.retired.len()));
        }

        parts.join(". ") + "."
    }
}

/// Runs one document all the way through.
fn ingest_one(
    file: &DiscoveredFile,
    collection: &Collection,
    reader: &dyn DocumentReader,
    store: &DocumentStore,
    index: &KnowledgeIndex,
    actor: &str,
    outcome: &mut IngestOutcome,
) -> Result<()> {
    // 1. Hold the original first. If reading fails afterwards, the bytes are
    //    still held and the document can be re-read once an engine exists.
    let held = store.ingest(&file.path, actor, collection.classification, None)?;

    // 2. Read it.
    let extracted = reader.read(&file.path)?;

    // 3. Keep the extraction beside the original.
    store.ingest(&file.path, actor, collection.classification, Some(&extracted))?;

    outcome.pages_needing_review += extracted.pages_needing_review;
    if extracted.contains_injection_attempt() {
        outcome.flagged_for_injection.push(file.relative_path.clone());
    }

    // 4 and 5. Chunk and index. A document that produced no readable text is
    //    still held and still recorded — it simply contributes no passages,
    //    which is the honest outcome rather than an empty chunk that would
    //    dilute every search.
    let chunks = chunk_document(&held.sha256, &extracted);
    let indexed = index.index_collection_document(
        Some(&collection.id),
        &file.relative_path,
        collection.classification,
        &chunks,
    )?;

    // 6. The version history: which bytes this path holds now, and what they
    //    replaced. A revised SOP supersedes its previous version here, in the
    //    same sync that indexed the new one.
    match index.record_source_version(
        &collection.id,
        &file.relative_path,
        &file.relative_path,
        &held.sha256,
    )? {
        SourceChange::Revised { previous_sha256, .. } => {
            outcome.superseded_sha256s.push(previous_sha256)
        }
        SourceChange::Unchanged { .. } => outcome.unchanged_on_reread += 1,
        SourceChange::Added { .. } => {}
    }

    outcome.documents_read += 1;
    outcome.chunks_indexed += indexed;
    Ok(())
}

/// Takes everything a sync plan calls for into the knowledge base.
pub fn ingest_collection(
    collection: &Collection,
    plan: &SyncPlan,
    reader: &dyn DocumentReader,
    store: &DocumentStore,
    index: &KnowledgeIndex,
    audit: Option<&AuditService>,
    actor: &str,
) -> IngestOutcome {
    let mut outcome = IngestOutcome {
        collection_id: collection.id.clone(),
        skipped: plan.skipped.len(),
        ..Default::default()
    };

    for file in plan.added.iter().chain(plan.changed.iter()) {
        if let Err(e) = ingest_one(file, collection, reader, store, index, actor, &mut outcome) {
            // Recorded and stepped over. A corrupt file in a share of ten
            // thousand must not stop the other nine thousand nine hundred.
            log::warn!("[KNOWLEDGE] could not take in {}: {e}", file.relative_path);
            outcome.failures.push(IngestFailure {
                file: file.relative_path.clone(),
                reason: e.to_string(),
            });
        }
    }

    // A document taken off the source stops answering questions, but its
    // passages stay held so a citation made earlier still resolves.
    //
    // This used to push the path onto `retired` and do nothing else, so a
    // withdrawn SOP went on answering: the outcome said "retired" and the
    // index disagreed.
    for gone in &plan.removed {
        match index.retire_source(&collection.id, gone, Retirement::Withdrawn) {
            Ok(Some(sha)) => outcome.withdrawn_sha256s.push(sha),
            Ok(None) => {}
            Err(e) => outcome.failures.push(IngestFailure {
                file: gone.clone(),
                reason: format!("could not be retired: {e}"),
            }),
        }
        outcome.retired.push(gone.clone());
    }

    if let Some(audit) = audit {
        let _ = audit.record(
            actor,
            AuditKind::Knowledge,
            format!("Synced {}: {}", collection.name, outcome.summary()),
            Some(serde_json::json!({
                "collectionId": collection.id,
                "classification": collection.classification,
                "documentsRead": outcome.documents_read,
                "chunksIndexed": outcome.chunks_indexed,
                "pagesNeedingReview": outcome.pages_needing_review,
                "failures": outcome.failures.len(),
                "flaggedForInjection": outcome.flagged_for_injection,
            })),
        );
    }

    outcome
}

/// What one sync of one collection did, end to end.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SyncReport {
    pub plan: String,
    pub outcome: IngestOutcome,
    /// Memory items rejected because a source they rest on was withdrawn.
    pub memory_invalidated: usize,
    /// Memory items marked stale because a source they rest on was revised.
    pub memory_stale: usize,
    /// Files that could not be listed, with why.
    pub unreadable: Vec<String>,
}

/// Reads a collection from its folder into the index, and tells the memory
/// graph what changed (P07).
///
/// The whole connector, in the order that keeps each step honest:
///
/// 1. **Access first.** The collection's enabled state and role restriction
///    are written to the index before any passage is, so a restricted
///    collection's first passage is never searchable by the wrong reader.
/// 2. **Discover and plan** against what the last sync saw.
/// 3. **Ingest**: hold, read, chunk, index, record the version — a revised
///    file supersedes its previous version, a removed file is withdrawn.
/// 4. **Remember** what this sync saw, so the next one is incremental.
/// 5. **Tell the graph**: claims resting on a withdrawn source are rejected;
///    claims resting on a revised one go stale.
#[allow(clippy::too_many_arguments)]
pub fn sync_collection(
    collection: &Collection,
    collections: &super::CollectionStore,
    reader: &dyn DocumentReader,
    store: &DocumentStore,
    index: &KnowledgeIndex,
    graph: Option<&super::graph::runtime_store::MemoryGraph>,
    audit: Option<&AuditService>,
    actor: &str,
) -> Result<SyncReport> {
    index.set_collection_access(&collection.id, collection.enabled, &collection.restricted_to_roles)?;
    let mut report = SyncReport::default();
    if !collection.enabled {
        report.plan = "The collection is disabled: nothing was read, and none of it answers.".into();
        return Ok(report);
    }
    let (discovered, unreadable) = super::connector::discover(collection)?;
    report.unreadable = unreadable;
    let plan = super::connector::plan_sync(&discovered, &collections.previous_state(&collection.id));
    report.plan = plan.summary();
    report.outcome = ingest_collection(collection, &plan, reader, store, index, audit, actor);
    collections.record_sync(&collection.id, &discovered)?;

    if let Some(graph) = graph {
        let at = chrono::Utc::now().to_rfc3339();
        for sha in &report.outcome.withdrawn_sha256s {
            match graph.invalidate_source(sha, &at) {
                Ok(ids) => report.memory_invalidated += ids.len(),
                Err(error) => log::warn!("[KNOWLEDGE] memory resting on {sha} was not invalidated: {}", error.explain()),
            }
        }
        for sha in &report.outcome.superseded_sha256s {
            match graph.source_revised(sha, &at) {
                Ok(ids) => report.memory_stale += ids.len(),
                Err(error) => log::warn!("[KNOWLEDGE] memory resting on {sha} was not marked stale: {}", error.explain()),
            }
        }
    }
    Ok(report)
}

/// Withdraws every current source of a collection and forgets its access row.
///
/// Used when a collection is removed. Its passages stay traceable; nothing in
/// it answers; claims resting on it are rejected.
pub fn withdraw_collection(
    collection_id: &str,
    index: &KnowledgeIndex,
    graph: Option<&super::graph::runtime_store::MemoryGraph>,
) -> Result<usize> {
    let mut withdrawn = 0;
    let at = chrono::Utc::now().to_rfc3339();
    for (path, _) in index.current_paths(collection_id)? {
        if let Some(sha) = index.retire_source(collection_id, &path, Retirement::Withdrawn)? {
            withdrawn += 1;
            if let Some(graph) = graph {
                let _ = graph.invalidate_source(&sha, &at);
            }
        }
    }
    index.set_collection_access(collection_id, false, &[])?;
    Ok(withdrawn)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::documents::{
        EngineCapabilities, EscalationPlan, ExtractedPage, InjectionFinding, InjectionScan,
    };
    use crate::identity::{Role, Session, User};
    use crate::knowledge::connector::SourceKind;
    use crate::policy::Classification;
    use std::path::PathBuf;

    /// A reader that returns prepared results, or fails on demand.
    struct FakeReader {
        text: String,
        pages_needing_review: u32,
        injection: bool,
        fail_on: Option<String>,
    }

    impl FakeReader {
        fn reading(text: &str) -> Self {
            Self {
                text: text.into(),
                pages_needing_review: 0,
                injection: false,
                fail_on: None,
            }
        }
    }

    impl DocumentReader for FakeReader {
        fn read(&self, path: &Path) -> Result<ExtractedDocument> {
            if let Some(bad) = &self.fail_on {
                if path.to_string_lossy().contains(bad) {
                    anyhow::bail!("this file is not a readable PDF");
                }
            }

            let mut scan = InjectionScan::default();
            if self.injection {
                scan.contains_instruction_like_text = true;
                scan.high_severity_count = 1;
                scan.findings.push(InjectionFinding {
                    page: 1,
                    kind: "instruction override".into(),
                    severity: "high".into(),
                    excerpt: "Ignore all previous instructions.".into(),
                    detail: "Quoted, never followed.".into(),
                });
            }

            Ok(ExtractedDocument {
                engine: "fake".into(),
                engine_version: "1".into(),
                pages: vec![ExtractedPage {
                    page: 1,
                    text: self.text.clone(),
                    confidence: 1.0,
                    needs_review: self.pages_needing_review > 0,
                    review_reason: None,
                    char_count: self.text.len() as u32,
                    regions: Vec::new(),
                    read_by: None,
                }],
                capabilities: EngineCapabilities::default(),
                warnings: vec![],
                pages_needing_review: self.pages_needing_review,
                source_path: path.display().to_string(),
                source_bytes: 1,
                injection_scan: scan,
                escalation: EscalationPlan::default(),
            })
        }
    }

    struct Fixture {
        _dir: tempfile::TempDir,
        root: PathBuf,
        store: DocumentStore,
        index: KnowledgeIndex,
    }

    fn fixture() -> Fixture {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().to_path_buf();
        Fixture {
            store: DocumentStore::open(&root).unwrap(),
            index: KnowledgeIndex::open(&root).unwrap(),
            _dir: dir,
            root,
        }
    }

    fn collection(root: &Path) -> Collection {
        Collection {
            id: "sops".into(),
            name: "Maintenance SOPs".into(),
            kind: SourceKind::LocalFolder,
            root: root.to_path_buf(),
            owner: "A. Fernandes".into(),
            classification: Classification::ProcessDiagram,
            restricted_to_roles: Vec::new(),
            retention_days: None,
            enabled: true,
        }
    }

    fn discovered(root: &Path, name: &str) -> DiscoveredFile {
        let path = root.join(name);
        std::fs::write(&path, format!("bytes of {name}")).unwrap();
        DiscoveredFile {
            relative_path: name.into(),
            byte_size: 10,
            modified_at: 1,
            path,
        }
    }

    fn plan_with(added: Vec<DiscoveredFile>) -> SyncPlan {
        SyncPlan {
            added,
            ..Default::default()
        }
    }

    const SOP: &str = "4 Inspection\n\n4.2 Wall Thickness\n\nMinimum acceptable is 9.0 mm.";

    #[test]
    fn a_document_ends_up_searchable_under_its_collections_classification() {
        let f = fixture();
        let plan = plan_with(vec![discovered(&f.root, "sop.pdf")]);

        let outcome = ingest_collection(
            &collection(&f.root),
            &plan,
            &FakeReader::reading(SOP),
            &f.store,
            &f.index,
            None,
            "kbadmin",
        );

        assert_eq!(outcome.documents_read, 1);
        assert!(outcome.chunks_indexed > 0);

        let session = Session::open(User::new("p", "P", vec![Role::Employee]));
        let hits = f.index.search(&session, "wall thickness", 10).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].classification, Classification::ProcessDiagram);
    }

    /// The original is held before anything is derived from it.
    #[test]
    fn the_original_is_held_and_verifiable_afterwards() {
        let f = fixture();
        let file = discovered(&f.root, "sop.pdf");
        let sha = DocumentStore::hash_file(&file.path).unwrap();

        ingest_collection(
            &collection(&f.root),
            &plan_with(vec![file]),
            &FakeReader::reading(SOP),
            &f.store,
            &f.index,
            None,
            "kbadmin",
        );

        assert!(f.store.verify(&sha).unwrap());
        assert!(f.store.derived(&sha).unwrap().is_some());
    }

    /// A share of ten thousand files will contain a corrupt one.
    #[test]
    fn one_unreadable_document_does_not_stop_the_others() {
        let f = fixture();
        let plan = plan_with(vec![
            discovered(&f.root, "good-one.pdf"),
            discovered(&f.root, "corrupt.pdf"),
            discovered(&f.root, "good-two.pdf"),
        ]);

        let mut reader = FakeReader::reading(SOP);
        reader.fail_on = Some("corrupt".into());

        let outcome = ingest_collection(
            &collection(&f.root), &plan, &reader, &f.store, &f.index, None, "kbadmin",
        );

        assert_eq!(outcome.documents_read, 2);
        assert_eq!(outcome.failures.len(), 1);
        assert_eq!(outcome.failures[0].file, "corrupt.pdf");
        assert!(outcome.failures[0].reason.contains("not a readable PDF"));
    }

    /// Even a document that failed to read has its bytes held, so it can be
    /// re-read once a better engine is installed.
    #[test]
    fn a_document_that_failed_to_read_is_still_held() {
        let f = fixture();
        let file = discovered(&f.root, "corrupt.pdf");
        let sha = DocumentStore::hash_file(&file.path).unwrap();

        let mut reader = FakeReader::reading(SOP);
        reader.fail_on = Some("corrupt".into());

        ingest_collection(
            &collection(&f.root), &plan_with(vec![file]), &reader, &f.store, &f.index, None, "kbadmin",
        );

        assert!(f.store.find(&sha).unwrap().is_some(), "the bytes should be held");
        assert!(f.store.derived(&sha).unwrap().is_none(), "but nothing was derived");
    }

    #[test]
    fn unread_pages_are_counted_rather_than_buried() {
        let f = fixture();
        let mut reader = FakeReader::reading(SOP);
        reader.pages_needing_review = 3;

        let outcome = ingest_collection(
            &collection(&f.root),
            &plan_with(vec![discovered(&f.root, "scan.pdf")]),
            &reader,
            &f.store,
            &f.index,
            None,
            "kbadmin",
        );

        assert_eq!(outcome.pages_needing_review, 3);
        assert!(outcome.needs_attention());
        assert!(outcome.summary().contains("3 page(s) could not be read"));
    }

    #[test]
    fn a_poisoned_document_is_named_in_the_outcome() {
        let f = fixture();
        let mut reader = FakeReader::reading(SOP);
        reader.injection = true;

        let outcome = ingest_collection(
            &collection(&f.root),
            &plan_with(vec![discovered(&f.root, "poisoned.pdf")]),
            &reader,
            &f.store,
            &f.index,
            None,
            "kbadmin",
        );

        assert_eq!(outcome.flagged_for_injection, vec!["poisoned.pdf"]);
        assert!(outcome.needs_attention());
        // Still indexed — flagging is not refusing.
        assert!(outcome.chunks_indexed > 0);
    }

    #[test]
    fn a_clean_sync_needs_no_attention_and_says_so_plainly() {
        let f = fixture();
        let outcome = ingest_collection(
            &collection(&f.root),
            &plan_with(vec![discovered(&f.root, "sop.pdf")]),
            &FakeReader::reading(SOP),
            &f.store,
            &f.index,
            None,
            "kbadmin",
        );

        assert!(!outcome.needs_attention());
        // One passage, not two: `4 Inspection` is a heading with no body under it
        // before the next heading, so it correctly contributes no chunk.
        assert_eq!(outcome.summary(), "1 document(s) read, 1 passage(s) indexed.");
    }

    #[test]
    fn removed_documents_are_reported_as_retired() {
        let f = fixture();
        let plan = SyncPlan {
            removed: vec!["withdrawn.pdf".into()],
            ..Default::default()
        };

        let outcome = ingest_collection(
            &collection(&f.root), &plan, &FakeReader::reading(SOP), &f.store, &f.index, None, "kbadmin",
        );

        assert_eq!(outcome.retired, vec!["withdrawn.pdf"]);
    }

    /// Re-reading a changed document replaces its passages rather than adding to
    /// them, so a stale sentence cannot be retrieved beside its replacement.
    #[test]
    fn re_syncing_a_changed_document_replaces_its_passages() {
        let f = fixture();
        let file = discovered(&f.root, "sop.pdf");
        let c = collection(&f.root);

        ingest_collection(
            &c, &plan_with(vec![file.clone()]),
            &FakeReader::reading("4.2 Limit\n\nMinimum is 8.0 mm."),
            &f.store, &f.index, None, "kbadmin",
        );

        ingest_collection(
            &c,
            &SyncPlan { changed: vec![file], ..Default::default() },
            &FakeReader::reading("4.2 Limit\n\nMinimum is 9.0 mm."),
            &f.store, &f.index, None, "kbadmin",
        );

        let session = Session::open(User::new("p", "P", vec![Role::Employee]));
        let hits = f.index.search(&session, "minimum", 10).unwrap();
        assert_eq!(hits.len(), 1, "the old passage should be gone");
        assert!(hits[0].text.contains("9.0 mm"));
    }

    /// P07: "retired" used to be a word in the outcome and nothing in the
    /// index. A withdrawn file must actually stop answering.
    #[test]
    fn a_removed_document_stops_answering_and_its_hash_is_reported() {
        let f = fixture();
        let c = collection(&f.root);
        ingest_collection(
            &c, &plan_with(vec![discovered(&f.root, "old-sop.pdf")]),
            &FakeReader::reading("4.2 Limit\n\nMinimum is 8.0 mm."),
            &f.store, &f.index, None, "kbadmin",
        );
        let session = Session::open(User::new("p", "P", vec![Role::Employee]));
        assert_eq!(f.index.search(&session, "minimum", 10).unwrap().len(), 1);

        let outcome = ingest_collection(
            &c,
            &SyncPlan { removed: vec!["old-sop.pdf".into()], ..Default::default() },
            &FakeReader::reading(SOP), &f.store, &f.index, None, "kbadmin",
        );
        assert_eq!(outcome.withdrawn_sha256s.len(), 1);
        assert!(f.index.search(&session, "minimum", 10).unwrap().is_empty(), "a withdrawn SOP still answered");
    }

    /// A file whose bytes changed supersedes its previous version: the new text
    /// answers, the old version stays traceable and says what replaced it.
    #[test]
    fn a_revised_file_supersedes_its_previous_version() {
        let f = fixture();
        let c = collection(&f.root);
        let first = discovered(&f.root, "sop.pdf");
        ingest_collection(
            &c, &plan_with(vec![first]),
            &FakeReader::reading("4.2 Limit\n\nMinimum is 8.0 mm."),
            &f.store, &f.index, None, "kbadmin",
        );
        std::fs::write(f.root.join("sop.pdf"), b"revision C bytes").unwrap();
        let revised = DiscoveredFile {
            path: f.root.join("sop.pdf"),
            relative_path: "sop.pdf".into(),
            byte_size: 16,
            modified_at: 2,
        };
        let outcome = ingest_collection(
            &c,
            &SyncPlan { changed: vec![revised], ..Default::default() },
            &FakeReader::reading("4.2 Limit\n\nMinimum is 9.0 mm."),
            &f.store, &f.index, None, "kbadmin",
        );
        assert_eq!(outcome.superseded_sha256s.len(), 1);
        let session = Session::open(User::new("p", "P", vec![Role::Employee]));
        let hits = f.index.search(&session, "minimum", 10).unwrap();
        assert_eq!(hits.len(), 1);
        assert!(hits[0].text.contains("9.0 mm"));
        let history = f
            .index
            .source_versions(&session, &outcome.superseded_sha256s[0])
            .unwrap()
            .expect("the old version is traceable");
        assert_eq!(history.iter().map(|v| v.status.as_str()).collect::<Vec<_>>(), vec!["superseded", "current"]);
    }

    #[test]
    fn an_empty_plan_does_nothing_and_says_so() {
        let f = fixture();
        let outcome = ingest_collection(
            &collection(&f.root), &SyncPlan::default(),
            &FakeReader::reading(SOP), &f.store, &f.index, None, "kbadmin",
        );

        assert_eq!(outcome.documents_read, 0);
        assert!(!outcome.needs_attention());
    }
}
