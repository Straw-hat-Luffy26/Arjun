//! Where documents live once they are in the workbench.
//!
//! ARJUN design rule 12 asks for the original file *and* the derived representations to be
//! stored locally. Both halves matter, and for different reasons:
//!
//! - The **original** is the evidence. A citation that points at page 4 of an
//!   inspection report is only checkable if the report still exists, byte for
//!   byte, exactly as it arrived.
//! - The **derived** extraction is the working copy. It is regenerable — a
//!   better engine installed next year should be able to re-read every document
//!   already held — so it is stored beside the original rather than replacing it.
//!
//! ## Content addressing
//!
//! Documents are keyed by the SHA-256 of their bytes. Three things fall out of
//! that, all of them useful here:
//!
//! - The same report attached twice is stored once, and both tasks cite the
//!   same identity.
//! - A citation carries a hash, so anybody can verify the file behind it was
//!   never altered.
//! - The store cannot be confused by two documents that happen to share a
//!   filename, which in an organisation that emails `report.pdf` around is not
//!   an edge case.
//!
//! The original is written once and never rewritten. Re-ingesting identical
//! bytes updates the record's metadata, not the file.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::ExtractedDocument;
use crate::policy::Classification;

/// What the store knows about one document.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StoredDocument {
    /// SHA-256 of the original bytes. The document's identity.
    pub sha256: String,
    /// The name it arrived under, kept for display only — never for lookup.
    pub original_name: String,
    pub byte_size: u64,
    pub ingested_at: DateTime<Utc>,
    /// Who brought it in.
    pub ingested_by: String,
    pub classification: Classification,
    /// Which engine produced the derived copy, so a document read by a weaker
    /// engine can be found and re-read when a better one is installed.
    pub engine: Option<String>,
    pub page_count: u32,
    pub pages_needing_review: u32,
    pub injection_findings: u32,
}

impl StoredDocument {
    /// Whether a person should look at this before its contents are relied on.
    pub fn needs_attention(&self) -> bool {
        self.pages_needing_review > 0 || self.injection_findings > 0
    }
}

pub struct DocumentStore {
    root: PathBuf,
    conn: Arc<Mutex<Connection>>,
}

impl DocumentStore {
    pub fn open(app_data_dir: &Path) -> Result<Self> {
        let root = app_data_dir.join("documents");
        std::fs::create_dir_all(root.join("originals"))
            .with_context(|| format!("could not create {}", root.display()))?;
        std::fs::create_dir_all(root.join("derived"))?;

        let conn = Connection::open(app_data_dir.join("sarathi.db"))
            .context("could not open the document index")?;
        Self::prepare(&conn)?;

        Ok(Self {
            root,
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    fn prepare(conn: &Connection) -> Result<()> {
        // `collection_documents`, not `documents`: `sarathi.db` is shared, and
        // `knowledge::multimodal` already creates an unprefixed `documents`
        // table with a different schema. Whichever opened first used to win
        // and the other's inserts failed -- latent only while this store was
        // never constructed outside tests. P07 constructs it for collection
        // sync, so the name is taken apart here.
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS collection_documents (
                sha256               TEXT PRIMARY KEY,
                original_name        TEXT NOT NULL,
                byte_size            INTEGER NOT NULL,
                ingested_at          TEXT NOT NULL,
                ingested_by          TEXT NOT NULL,
                classification       TEXT NOT NULL,
                engine               TEXT,
                page_count           INTEGER NOT NULL DEFAULT 0,
                pages_needing_review INTEGER NOT NULL DEFAULT 0,
                injection_findings   INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS collection_documents_classification_idx
                ON collection_documents(classification);",
        )?;
        Ok(())
    }

    /// SHA-256 of a file, streamed so a large drawing set does not have to fit
    /// in memory before it can be identified.
    pub fn hash_file(path: &Path) -> Result<String> {
        use std::io::Read;

        let mut file = std::fs::File::open(path)
            .with_context(|| format!("could not read {}", path.display()))?;
        let mut hasher = Sha256::new();
        let mut buffer = [0u8; 64 * 1024];

        loop {
            let read = file.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
        }

        Ok(format!("{:x}", hasher.finalize()))
    }

    fn original_path(&self, sha256: &str, source: &Path) -> PathBuf {
        // The extension is kept so the stored file still opens by double-click
        // during an investigation; the name is the hash, so it is unambiguous.
        let extension = source
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("bin")
            .to_ascii_lowercase();
        self.root.join("originals").join(format!("{sha256}.{extension}"))
    }

    fn derived_path(&self, sha256: &str) -> PathBuf {
        self.root.join("derived").join(format!("{sha256}.json"))
    }

    /// Takes a document into the store and records what is known about it.
    ///
    /// Identical bytes already held are not written again — the existing file is
    /// the same file. The metadata record is refreshed, because who brought it
    /// in and what it was read with can legitimately change.
    pub fn ingest(
        &self,
        source: &Path,
        ingested_by: &str,
        classification: Classification,
        extraction: Option<&ExtractedDocument>,
    ) -> Result<StoredDocument> {
        let sha256 = Self::hash_file(source)?;
        let byte_size = std::fs::metadata(source)?.len();
        let original_name = source
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| sha256.clone());

        let stored_original = self.original_path(&sha256, source);
        if !stored_original.exists() {
            std::fs::copy(source, &stored_original).with_context(|| {
                format!("could not store the original at {}", stored_original.display())
            })?;
        }

        if let Some(extraction) = extraction {
            let derived = self.derived_path(&sha256);
            std::fs::write(&derived, serde_json::to_vec_pretty(extraction)?).with_context(
                || format!("could not store the extraction at {}", derived.display()),
            )?;
        }

        let record = StoredDocument {
            sha256: sha256.clone(),
            original_name,
            byte_size,
            ingested_at: Utc::now(),
            ingested_by: ingested_by.to_string(),
            classification,
            engine: extraction.map(|e| e.engine.clone()),
            page_count: extraction.map(|e| e.pages.len() as u32).unwrap_or(0),
            pages_needing_review: extraction.map(|e| e.pages_needing_review).unwrap_or(0),
            injection_findings: extraction
                .map(|e| e.injection_scan.findings.len() as u32)
                .unwrap_or(0),
        };

        let conn = self.conn.lock().expect("document index lock poisoned");
        conn.execute(
            "INSERT INTO collection_documents
                (sha256, original_name, byte_size, ingested_at, ingested_by, classification,
                 engine, page_count, pages_needing_review, injection_findings)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
             ON CONFLICT(sha256) DO UPDATE SET
                original_name = excluded.original_name,
                ingested_at = excluded.ingested_at,
                ingested_by = excluded.ingested_by,
                classification = excluded.classification,
                engine = excluded.engine,
                page_count = excluded.page_count,
                pages_needing_review = excluded.pages_needing_review,
                injection_findings = excluded.injection_findings",
            params![
                record.sha256,
                record.original_name,
                record.byte_size as i64,
                record.ingested_at.to_rfc3339(),
                record.ingested_by,
                serde_json::to_string(&record.classification)?,
                record.engine,
                record.page_count,
                record.pages_needing_review,
                record.injection_findings,
            ],
        )?;

        log::info!(
            "[DOCUMENTS] held {} ({} bytes) as {}",
            record.original_name,
            record.byte_size,
            &record.sha256[..12]
        );
        Ok(record)
    }

    /// The record for one document, if it is held.
    pub fn find(&self, sha256: &str) -> Result<Option<StoredDocument>> {
        let conn = self.conn.lock().expect("document index lock poisoned");
        let mut stmt = conn.prepare(
            "SELECT sha256, original_name, byte_size, ingested_at, ingested_by, classification,
                    engine, page_count, pages_needing_review, injection_findings
             FROM collection_documents WHERE sha256 = ?1",
        )?;

        let mut rows = stmt.query([sha256])?;
        let Some(row) = rows.next()? else {
            return Ok(None);
        };

        let classification: String = row.get(5)?;
        let ingested_at: String = row.get(3)?;

        Ok(Some(StoredDocument {
            sha256: row.get(0)?,
            original_name: row.get(1)?,
            byte_size: row.get::<_, i64>(2)? as u64,
            ingested_at: DateTime::parse_from_rfc3339(&ingested_at)
                .map(|d| d.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now()),
            ingested_by: row.get(4)?,
            classification: serde_json::from_str(&classification)
                .unwrap_or(Classification::Internal),
            engine: row.get(6)?,
            page_count: row.get(7)?,
            pages_needing_review: row.get(8)?,
            injection_findings: row.get(9)?,
        }))
    }

    /// Reads back the stored extraction.
    ///
    /// Absent means the document was held but never read — which happens when a
    /// scan arrives before an engine that can read it is installed.
    pub fn derived(&self, sha256: &str) -> Result<Option<ExtractedDocument>> {
        let path = self.derived_path(sha256);
        if !path.exists() {
            return Ok(None);
        }
        Ok(Some(serde_json::from_slice(&std::fs::read(path)?)?))
    }

    /// Confirms the held original still hashes to its identity.
    ///
    /// Content addressing makes tampering detectable rather than impossible, so
    /// something has to actually check. This is what a citation's "verified"
    /// state is built on.
    pub fn verify(&self, sha256: &str) -> Result<bool> {
        let Some(record) = self.find(sha256)? else {
            return Ok(false);
        };
        let path = self.original_path(sha256, Path::new(&record.original_name));
        if !path.exists() {
            return Ok(false);
        }
        Ok(Self::hash_file(&path)? == sha256)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::documents::{EngineCapabilities, EscalationPlan, ExtractedPage, InjectionScan};

    fn extraction(pages_needing_review: u32) -> ExtractedDocument {
        ExtractedDocument {
            engine: "text-layer".into(),
            engine_version: "1".into(),
            pages: vec![ExtractedPage {
                page: 1,
                text: "Wall thickness 8.2 mm".into(),
                confidence: 1.0,
                needs_review: pages_needing_review > 0,
                review_reason: None,
                char_count: 21,
                regions: Vec::new(),
                read_by: None,
            }],
            capabilities: EngineCapabilities::default(),
            warnings: vec![],
            pages_needing_review,
            source_path: "report.pdf".into(),
            source_bytes: 12,
            injection_scan: InjectionScan::default(),
            escalation: EscalationPlan::default(),
        }
    }

    struct Fixture {
        _dir: tempfile::TempDir,
        store: DocumentStore,
        root: PathBuf,
    }

    fn fixture() -> Fixture {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().to_path_buf();
        let store = DocumentStore::open(&root).unwrap();
        Fixture { _dir: dir, store, root }
    }

    fn write(root: &Path, name: &str, contents: &[u8]) -> PathBuf {
        let path = root.join(name);
        // Nested fixture paths are common here, so the parent is created rather
        // than left as a trap for the next test that uses one.
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(&path, contents).unwrap();
        path
    }

    #[test]
    fn a_document_is_held_with_its_original_intact() {
        let f = fixture();
        let source = write(&f.root, "report.pdf", b"inspection report bytes");

        let record = f
            .store
            .ingest(&source, "engineer", Classification::ProcessDiagram, None)
            .unwrap();

        assert_eq!(record.original_name, "report.pdf");
        assert_eq!(record.byte_size, 23);
        assert!(f.store.verify(&record.sha256).unwrap());
    }

    /// The identity is the bytes, so the same report under two names is one
    /// document — and two different reports sharing a name are not confused.
    #[test]
    fn identical_bytes_are_one_document_whatever_they_are_called() {
        let f = fixture();
        let first = write(&f.root, "report.pdf", b"same bytes");
        let second = write(&f.root, "report-copy.pdf", b"same bytes");

        let a = f.store.ingest(&first, "engineer", Classification::Internal, None).unwrap();
        let b = f.store.ingest(&second, "reviewer", Classification::Internal, None).unwrap();

        assert_eq!(a.sha256, b.sha256);
    }

    #[test]
    fn different_documents_sharing_a_filename_stay_separate() {
        let f = fixture();
        let one = write(&f.root, "a/report.pdf", b"first report");
        let two = write(&f.root, "b/report.pdf", b"second report");

        let a = f.store.ingest(&one, "engineer", Classification::Internal, None).unwrap();
        let b = f.store.ingest(&two, "engineer", Classification::Internal, None).unwrap();

        assert_ne!(a.sha256, b.sha256);
        assert!(f.store.find(&a.sha256).unwrap().is_some());
        assert!(f.store.find(&b.sha256).unwrap().is_some());
    }

    #[test]
    fn the_derived_extraction_is_stored_and_read_back() {
        let f = fixture();
        let source = write(&f.root, "report.pdf", b"bytes");

        let record = f
            .store
            .ingest(&source, "engineer", Classification::Internal, Some(&extraction(0)))
            .unwrap();

        let derived = f.store.derived(&record.sha256).unwrap().unwrap();
        assert_eq!(derived.engine, "text-layer");
        assert_eq!(derived.pages[0].text, "Wall thickness 8.2 mm");
    }

    /// A scan held before an engine that can read it exists is a real state,
    /// not an error.
    #[test]
    fn a_document_held_without_an_extraction_reports_none() {
        let f = fixture();
        let source = write(&f.root, "scan.pdf", b"bytes");
        let record = f.store.ingest(&source, "engineer", Classification::Internal, None).unwrap();

        assert!(f.store.derived(&record.sha256).unwrap().is_none());
        assert_eq!(record.engine, None);
    }

    #[test]
    fn a_document_needing_review_is_flagged_in_its_record() {
        let f = fixture();
        let source = write(&f.root, "scan.pdf", b"bytes");
        let record = f
            .store
            .ingest(&source, "engineer", Classification::Internal, Some(&extraction(1)))
            .unwrap();

        assert!(record.needs_attention());
        assert_eq!(record.pages_needing_review, 1);
    }

    /// Re-ingesting refreshes what is known without rewriting the evidence.
    #[test]
    fn re_ingesting_updates_the_record_and_leaves_the_original_alone() {
        let f = fixture();
        let source = write(&f.root, "report.pdf", b"bytes");

        let first = f.store.ingest(&source, "engineer", Classification::Internal, None).unwrap();
        let stored = f.store.original_path(&first.sha256, &source);
        let before = std::fs::metadata(&stored).unwrap().len();

        let second = f
            .store
            .ingest(&source, "reviewer", Classification::Financial, Some(&extraction(0)))
            .unwrap();

        assert_eq!(first.sha256, second.sha256);
        assert_eq!(std::fs::metadata(&stored).unwrap().len(), before);

        let reread = f.store.find(&first.sha256).unwrap().unwrap();
        assert_eq!(reread.ingested_by, "reviewer");
        assert_eq!(reread.classification, Classification::Financial);
        assert_eq!(reread.engine.as_deref(), Some("text-layer"));
    }

    /// Content addressing only helps if something checks.
    #[test]
    fn a_tampered_original_fails_verification() {
        let f = fixture();
        let source = write(&f.root, "report.pdf", b"original bytes");
        let record = f.store.ingest(&source, "engineer", Classification::Internal, None).unwrap();
        assert!(f.store.verify(&record.sha256).unwrap());

        let stored = f.store.original_path(&record.sha256, &source);
        std::fs::write(&stored, b"quietly altered").unwrap();

        assert!(!f.store.verify(&record.sha256).unwrap());
    }

    #[test]
    fn an_unknown_document_is_not_found_and_does_not_verify() {
        let f = fixture();
        let absent = "0".repeat(64);
        assert!(f.store.find(&absent).unwrap().is_none());
        assert!(!f.store.verify(&absent).unwrap());
    }

    /// The knowledge index needs full-text search, so this confirms the bundled
    /// SQLite was built with it rather than discovering the gap at query time.
    #[test]
    fn the_bundled_sqlite_supports_full_text_search() {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch("CREATE VIRTUAL TABLE probe USING fts5(body);")
            .expect("FTS5 must be available in the bundled SQLite");
    }

    #[test]
    fn hashing_is_stable_across_reads() {
        let f = fixture();
        let source = write(&f.root, "report.pdf", b"bytes");
        assert_eq!(
            DocumentStore::hash_file(&source).unwrap(),
            DocumentStore::hash_file(&source).unwrap()
        );
    }

    /// Streaming has to produce the same digest as hashing in one go, or a
    /// large document would get a different identity than a small one.
    #[test]
    fn a_document_larger_than_the_read_buffer_hashes_correctly() {
        let f = fixture();
        let big = vec![7u8; 200 * 1024];
        let source = write(&f.root, "drawing.pdf", &big);

        let mut expected = Sha256::new();
        expected.update(&big);

        assert_eq!(
            DocumentStore::hash_file(&source).unwrap(),
            format!("{:x}", expected.finalize())
        );
    }
}
