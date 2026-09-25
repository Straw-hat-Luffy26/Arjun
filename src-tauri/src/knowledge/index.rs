//! Finding the right passage, and never returning one the asker may not see.
//!
//! ARJUN design rule 22 is precise about the ordering: *"The policy gateway filters results
//! by document and user permissions **before** the passages reach the model."*
//! That word does the work. A filter applied afterwards has already let the
//! model see the text, and even discarding it leaks — result counts, ranking
//! positions and "no results" versus "results you cannot see" all say something
//! about material the asker was not cleared for.
//!
//! So clearance is part of the SQL. A passage the asker cannot see is not
//! fetched, not ranked, and not counted.
//!
//! ## Two halves, one clause
//!
//! Keyword search (FTS5 bm25) and dense search (vectors of one qualified
//! embedding space — see [`crate::knowledge::embedding`]) both read through the
//! same authorisation clause: classification clearance, the collection's own
//! role restriction, and whether the collection is enabled. Fusion, reranking
//! and graph expansion happen in [`crate::knowledge::hybrid`], over what these
//! two returned, so nothing downstream ever handles a passage the reader may
//! not see. [`SearchResult::retrieval`] says which half found each passage, so
//! a deployment on keyword-only never looks like one running the full pipeline.
//!
//! ## Versions
//!
//! A source path's versions are kept in `source_versions`. A revised file
//! supersedes its previous version (still traceable, no longer answering); a
//! file gone from its folder is withdrawn; a person can revoke one. The index
//! keeps a clock that each of those moves, and a search can be pinned to a
//! clock value — which is how queued work reads the corpus it was queued
//! against.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use rusqlite::types::Value;
use rusqlite::{params, params_from_iter, Connection};
use serde::{Deserialize, Serialize};

use super::Chunk;
use crate::identity::Session;
use crate::policy::Classification;

/// How a passage was found.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Retrieval {
    /// Full-text keyword match.
    Keyword,
    /// Vector similarity, against a passage embedded by a local model.
    Vector,
}

/// One document the index holds, as a library screen lists it.
///
/// Metadata only. The passages themselves are reached by searching, which is
/// where the citation and the evidence markers come from — a list that carried
/// text would be a second, unaudited way to read the material.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct IndexedDocument {
    pub document_sha256: String,
    pub document_name: String,
    pub classification: Classification,
    /// How many passages this document was split into.
    pub chunks: usize,
    /// The highest page number held for it.
    pub pages: u32,
}

/// One passage, with everything needed to cite and to trust it.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SearchResult {
    pub chunk_id: String,
    pub document_sha256: String,
    pub document_name: String,
    pub text: String,
    pub page: u32,
    pub section_path: Vec<String>,
    pub classification: Classification,
    /// Lower is a better match, following the underlying ranking.
    pub score: f64,
    pub retrieval: Retrieval,
}

impl SearchResult {
    /// How a citation to this passage reads.
    pub fn citation(&self) -> String {
        if self.section_path.is_empty() {
            format!("{}, page {}", self.document_name, self.page)
        } else {
            format!(
                "{} — {}, page {}",
                self.document_name,
                self.section_path.join(" › "),
                self.page
            )
        }
    }
}

/// Turns what a person typed into a query FTS5 will read as they meant it.
///
/// FTS5 has its own query language: `-` means NOT, `*` is a prefix wildcard,
/// `OR`/`NOT`/`NEAR` are operators, and an unbalanced quote is a syntax error.
/// A refinery user does not know that and should not have to — they type
/// `PV-2201`, which FTS5 reads as "PV, but not 2201" and matches nothing at all.
/// Silently returning no results for the single most likely query in the domain
/// would be the worst kind of bug: invisible, and indistinguishable from the tag
/// genuinely not appearing anywhere.
///
/// So every token is wrapped as a quoted phrase. `PV-2201` becomes `"PV-2201"`,
/// which FTS5 matches as the adjacent sequence its tokeniser produced — exactly
/// what the person meant. Internal quotes are doubled, which is how FTS5 escapes
/// them, so no input can break out of the phrase and become an operator.
fn to_match_expression(query: &str) -> Option<String> {
    sanitise_fts(query)
}

/// Wraps an FTS5 query so the user's tokens are matched as adjacent phrases.
///
/// Public so the multimodal index can share the same sanitisation — a
/// refinery user typing `PT-2201` should land on the right page whether
/// the page is a passage, an image region, or a table row, and the FTS5
/// expression that does the matching is the same in all three cases.
pub fn sanitise_fts(query: &str) -> Option<String> {
    let phrases: Vec<String> = query
        .split_whitespace()
        .filter(|token| !token.trim_matches(|c: char| !c.is_alphanumeric()).is_empty())
        .map(|token| format!("\"{}\"", token.replace('"', "\"\"")))
        .collect();

    (!phrases.is_empty()).then(|| phrases.join(" "))
}

/// The words a natural-language question is mostly made of, and which a
/// passage matching on them has not matched on anything.
///
/// Used only by the any-term pass of [`KnowledgeIndex::search_lexical`]. The
/// all-terms pass keeps them, because in a literal query ("shut off the pump")
/// they are part of the phrase a person typed.
const STOPWORDS: &[&str] = &[
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "of", "to", "in", "on", "for",
    "and", "or", "what", "which", "who", "whom", "how", "when", "where", "why", "does", "do",
    "did", "with", "by", "at", "from", "as", "it", "its", "this", "that", "these", "those",
    "should", "must", "can", "could", "would", "will", "shall", "i", "we", "you", "our", "my",
    "me", "any", "there", "their", "into", "per", "than", "then", "if", "about", "has", "have",
    "had", "tell", "please", "give", "show", "find", "say", "says",
];

/// Which lexical pass found a passage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum LexicalPass {
    /// Every term of the query, as `search` matches.
    AllTerms,
    /// At least one significant term. Ranked after every all-terms hit.
    AnyTerm,
}

/// A passage the keyword half found, with the score it was found by.
#[derive(Debug, Clone)]
pub struct LexicalHit {
    pub result: SearchResult,
    pub pass: LexicalPass,
    /// SQLite FTS5 bm25: lower is better, comparable only within one pass of
    /// one query.
    pub bm25: f64,
}

/// A passage the dense half found.
#[derive(Debug, Clone)]
pub struct DenseHit {
    pub result: SearchResult,
    /// Cosine similarity of the closest window to the query, in `[-1, 1]`.
    pub similarity: f64,
    /// Which window of the passage was closest.
    pub window: u32,
}

/// Narrowing a search beyond what the reader is cleared for.
///
/// Every field can only remove passages. Nothing here widens a search past the
/// reader's clearance, and an empty scope is the whole of what they may read.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SearchScope {
    /// Only these documents, by content hash.
    #[serde(default)]
    pub documents: Option<Vec<String>>,
    /// Only these collections.
    #[serde(default)]
    pub collections: Option<Vec<String>>,
    /// The corpus as it stood at this index revision — see
    /// [`KnowledgeIndex::index_revision`].
    ///
    /// For queued work: a job delegated at revision 40 reads the sources that
    /// were current at 40, so a document added at 41 does not change what it
    /// answers from, and a version superseded at 41 is still read and flagged as
    /// superseded since. A source withdrawn or revoked since is **not** read —
    /// revocation always wins over the pin.
    #[serde(default)]
    pub pinned_revision: Option<i64>,
}

/// What a pinned scope means today, resolved against the version history.
#[derive(Debug, Clone, Default)]
pub struct PinnedCorpus {
    /// Current now, but added after the pin: not read.
    pub added_since: BTreeSet<String>,
    /// Superseded after the pin: still read, and reported as superseded since.
    pub superseded_since: BTreeSet<String>,
}

/// One version of a source, as the version history holds it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SourceVersion {
    pub collection_id: String,
    pub relative_path: String,
    pub version: u32,
    pub document_sha256: String,
    pub document_name: String,
    /// `current`, `superseded`, `withdrawn` or `revoked`.
    pub status: String,
    pub added_revision: i64,
    pub retired_revision: Option<i64>,
    pub superseded_by: Option<String>,
    pub indexed_at: String,
    pub retired_at: Option<String>,
}

/// What recording a source did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SourceChange {
    /// First time this path was seen, or seen again after being withdrawn.
    Added { version: u32 },
    /// The path now holds different bytes; the old version is superseded.
    Revised { version: u32, previous_sha256: String },
    /// Same bytes as the current version.
    Unchanged { version: u32 },
}

/// Why a source stopped answering questions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Retirement {
    /// Gone from the source folder.
    Withdrawn,
    /// Taken away by a person: its permission changed, or it must not be used.
    Revoked,
}

impl Retirement {
    fn as_str(self) -> &'static str {
        match self {
            Retirement::Withdrawn => "withdrawn",
            Retirement::Revoked => "revoked",
        }
    }
}

/// A passage the embedding pass still has to do.
#[derive(Debug, Clone)]
pub struct PendingPassage {
    pub chunk_id: String,
    /// The indexed body: section trail and text, exactly what keyword search reads.
    pub body: String,
}

/// How much of what a reader may see a vector space covers.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EmbeddingCoverage {
    pub embedded: usize,
    /// Passages the model refused or could not embed, recorded so the pass
    /// does not retry them forever and the gap is visible.
    pub failed: usize,
    pub total: usize,
}

/// What the reader is cleared for, in the forms the SQL binds.
struct Clearance {
    classifications: Vec<String>,
    roles: Vec<String>,
}

impl Clearance {
    fn of(session: &Session) -> Self {
        let classifications = Classification::ALL
            .iter()
            .filter(|c| {
                c.cleared_roles()
                    .iter()
                    .any(|role| session.user.roles.contains(role))
            })
            .filter_map(|c| serde_json::to_string(c).ok())
            .collect();
        let roles = session
            .user
            .roles
            .iter()
            .filter_map(|role| serde_json::to_string(role).ok())
            .collect();
        Self {
            classifications,
            roles,
        }
    }

    fn is_empty(&self) -> bool {
        self.classifications.is_empty()
    }

    /// The clause every read path shares, with its values pushed in order.
    ///
    /// Classification clearance, then the collection's own state: a disabled
    /// collection answers nobody, and a restricted one answers only the roles
    /// it names. One clause, so search, the dense half, a page region, the
    /// library list and the coverage counts cannot disagree about what a
    /// reader may see.
    fn clause(&self, values: &mut Vec<Value>) -> String {
        let classes = placeholders(self.classifications.len());
        for class in &self.classifications {
            values.push(Value::Text(class.clone()));
        }
        let roles = if self.roles.is_empty() {
            // No role names nothing, and `IN ()` is not SQL.
            "NULL".to_string()
        } else {
            placeholders(self.roles.len())
        };
        for role in &self.roles {
            values.push(Value::Text(role.clone()));
        }
        format!(
            "c.classification IN ({classes})
             AND (c.collection_id IS NULL OR (
                 NOT EXISTS (SELECT 1 FROM collection_access a
                             WHERE a.collection_id = c.collection_id AND a.enabled = 0)
                 AND (NOT EXISTS (SELECT 1 FROM collection_access a
                                  WHERE a.collection_id = c.collection_id AND a.restricted = 1)
                      OR EXISTS (SELECT 1 FROM collection_roles r
                                 WHERE r.collection_id = c.collection_id
                                   AND r.role IN ({roles})))))"
        )
    }
}

fn placeholders(count: usize) -> String {
    vec!["?"; count].join(",")
}

fn json_list<'a>(items: impl IntoIterator<Item = &'a String>) -> Value {
    Value::Text(serde_json::to_string(&items.into_iter().collect::<Vec<_>>()).unwrap_or_else(|_| "[]".into()))
}

fn sha256_text(text: &str) -> String {
    use sha2::{Digest, Sha256};
    format!("{:x}", Sha256::digest(text.as_bytes()))
}

pub struct KnowledgeIndex {
    conn: Arc<Mutex<Connection>>,
}

impl KnowledgeIndex {
    pub fn open(app_data_dir: &Path) -> Result<Self> {
        std::fs::create_dir_all(app_data_dir)?;
        let conn = Connection::open(app_data_dir.join("sarathi.db"))
            .context("could not open the knowledge index")?;
        Self::prepare(&conn)?;
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    fn prepare(conn: &Connection) -> Result<()> {
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS chunks (
                id             TEXT PRIMARY KEY,
                document_sha256 TEXT NOT NULL,
                document_name  TEXT NOT NULL,
                ordinal        INTEGER NOT NULL,
                page           INTEGER NOT NULL,
                section_path   TEXT NOT NULL,
                kind           TEXT NOT NULL,
                classification TEXT NOT NULL,
                superseded     INTEGER NOT NULL DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS chunks_document_idx ON chunks(document_sha256);

            -- The searchable text lives in its own FTS table, joined by id.
            -- Kept external rather than as a contentless index so a passage can
            -- be read back for citation without re-reading the source document.
            CREATE VIRTUAL TABLE IF NOT EXISTS chunk_text USING fts5(
                id UNINDEXED,
                body
            );

            -- Which collections answer whom. A collection with no row here is
            -- enabled and unrestricted, which is what a collection that was
            -- never configured is.
            CREATE TABLE IF NOT EXISTS collection_access (
                collection_id TEXT PRIMARY KEY,
                enabled       INTEGER NOT NULL DEFAULT 1,
                restricted    INTEGER NOT NULL DEFAULT 0,
                updated_at    TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS collection_roles (
                collection_id TEXT NOT NULL,
                role          TEXT NOT NULL,
                PRIMARY KEY (collection_id, role)
            );

            -- Every version a source path has had, and why each one stopped.
            --
            -- A citation made against version 2 must still resolve after
            -- version 3 arrives, and a reader asking which is current must be
            -- told. The revision numbers are the index's own clock, which is
            -- what lets queued work read the corpus as it stood when it was
            -- queued.
            CREATE TABLE IF NOT EXISTS source_versions (
                collection_id   TEXT NOT NULL,
                relative_path   TEXT NOT NULL,
                version         INTEGER NOT NULL,
                document_sha256 TEXT NOT NULL,
                document_name   TEXT NOT NULL,
                status          TEXT NOT NULL,
                added_rev       INTEGER NOT NULL,
                retired_rev     INTEGER,
                superseded_by   TEXT,
                indexed_at      TEXT NOT NULL,
                retired_at      TEXT,
                PRIMARY KEY (collection_id, relative_path, version)
            );
            CREATE INDEX IF NOT EXISTS source_versions_sha_idx
                ON source_versions(document_sha256);

            CREATE TABLE IF NOT EXISTS index_clock (
                id       INTEGER PRIMARY KEY CHECK (id = 1),
                revision INTEGER NOT NULL
            );

            -- One embedding space per model identity. See
            -- `knowledge::embedding::EmbeddingIdentity::space_key`: a vector
            -- is only ever compared with vectors of its own space.
            CREATE TABLE IF NOT EXISTS vector_spaces (
                space_key       TEXT PRIMARY KEY,
                model_id        TEXT NOT NULL,
                profile         TEXT NOT NULL,
                profile_version INTEGER NOT NULL,
                weights_sha256  TEXT NOT NULL,
                dimensions      INTEGER NOT NULL,
                pooling         TEXT NOT NULL,
                created_at      TEXT NOT NULL
            );

            -- A passage's vectors, one per window, under the space that made
            -- them. Keyed by space as well as passage, so embedding with a
            -- second model adds a second space instead of overwriting the
            -- first row by row -- the failure the single-model table this
            -- replaces could not prevent.
            CREATE TABLE IF NOT EXISTS chunk_embeddings (
                space_key   TEXT NOT NULL,
                chunk_id    TEXT NOT NULL,
                window_no   INTEGER NOT NULL,
                text_sha256 TEXT NOT NULL,
                dimensions  INTEGER NOT NULL,
                vector      BLOB NOT NULL,
                PRIMARY KEY (space_key, chunk_id, window_no)
            );
            CREATE INDEX IF NOT EXISTS chunk_embeddings_chunk_idx ON chunk_embeddings(chunk_id);

            CREATE TABLE IF NOT EXISTS embedding_failures (
                space_key TEXT NOT NULL,
                chunk_id  TEXT NOT NULL,
                reason    TEXT NOT NULL,
                at        TEXT NOT NULL,
                PRIMARY KEY (space_key, chunk_id)
            );",
        )?;

        // The collection a passage was read from. Added to an existing table,
        // so a database from before P07 gains the column rather than being
        // rebuilt; its rows read as belonging to no collection, which is how
        // they were searched before.
        let has_collection: bool = conn
            .prepare("SELECT 1 FROM pragma_table_info('chunks') WHERE name = 'collection_id'")?
            .exists([])?;
        if !has_collection {
            conn.execute_batch("ALTER TABLE chunks ADD COLUMN collection_id TEXT;")?;
        }
        conn.execute_batch(
            "CREATE INDEX IF NOT EXISTS chunks_collection_idx ON chunks(collection_id);
             INSERT OR IGNORE INTO index_clock (id, revision) VALUES (1, 0);",
        )?;
        Ok(())
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Connection> {
        self.conn.lock().expect("the index lock is never poisoned")
    }

    /// The index's clock. Every source version added or retired moves it.
    pub fn index_revision(&self) -> Result<i64> {
        let conn = self.lock();
        Ok(conn.query_row("SELECT revision FROM index_clock WHERE id = 1", [], |row| row.get(0))?)
    }

    fn tick(tx: &rusqlite::Transaction<'_>) -> Result<i64> {
        tx.execute("UPDATE index_clock SET revision = revision + 1 WHERE id = 1", [])?;
        Ok(tx.query_row("SELECT revision FROM index_clock WHERE id = 1", [], |row| row.get(0))?)
    }

    /// How many distinct documents are currently retrievable.
    ///
    /// Superseded documents are excluded: they remain traceable, but they are
    /// not current guidance, so counting them would overstate what an answer
    /// can actually be grounded in.
    pub fn document_count(&self) -> Result<usize> {
        let conn = self.lock();
        let count: i64 = conn.query_row(
            "SELECT COUNT(DISTINCT document_sha256) FROM chunks WHERE superseded = 0",
            [],
            |row| row.get(0),
        )?;
        Ok(count as usize)
    }

    /// The documents this person may see.
    ///
    /// Filtered by exactly the clearance rule [`Self::search`] applies, and for
    /// the same reason: a list naming a document somebody cannot retrieve from
    /// would disclose that it exists, which is what the clearance withholds.
    /// Somebody cleared for nothing sees an empty library, indistinguishable
    /// from an empty index.
    ///
    /// Superseded chunks are excluded, so this and [`Self::document_count`]
    /// describe the same set.
    pub fn documents(&self, session: &Session) -> Result<Vec<IndexedDocument>> {
        let clearance = Clearance::of(session);
        if clearance.is_empty() {
            return Ok(Vec::new());
        }
        let mut values = Vec::new();
        let clause = clearance.clause(&mut values);
        let sql = format!(
            "SELECT c.document_sha256, c.document_name, c.classification,
                    COUNT(*) AS chunks, MAX(c.page) AS pages
             FROM chunks c
             WHERE c.superseded = 0
               AND {clause}
             GROUP BY c.document_sha256, c.document_name, c.classification
             ORDER BY c.document_name"
        );

        let conn = self.lock();
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map(params_from_iter(values), |row| {
            let classification: String = row.get(2)?;
            Ok(IndexedDocument {
                document_sha256: row.get(0)?,
                document_name: row.get(1)?,
                classification: serde_json::from_str(&classification)
                    .unwrap_or(Classification::Internal),
                chunks: row.get::<_, i64>(3)? as usize,
                pages: row.get::<_, i64>(4)? as u32,
            })
        })?;

        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    /// Adds a document's chunks to the index, replacing anything held for it.
    ///
    /// Replacing rather than appending means re-reading a document with a better
    /// engine does not leave the old, worse passages behind to be retrieved
    /// alongside the new ones.
    pub fn index_document(
        &self,
        document_name: &str,
        classification: Classification,
        chunks: &[Chunk],
    ) -> Result<usize> {
        self.index_collection_document(None, document_name, classification, chunks)
    }

    /// [`Self::index_document`], for a document read from a collection.
    ///
    /// The collection is recorded on every passage so its own restriction and
    /// enabled state are enforced in the same statement as clearance.
    pub fn index_collection_document(
        &self,
        collection_id: Option<&str>,
        document_name: &str,
        classification: Classification,
        chunks: &[Chunk],
    ) -> Result<usize> {
        let Some(first) = chunks.first() else {
            return Ok(0);
        };
        let sha = first.document_sha256.clone();

        let mut conn = self.lock();
        let tx = conn.transaction()?;

        // Clear the old copy first, inside the same transaction, so a failure
        // never leaves a document half-replaced.
        {
            let mut stale = tx.prepare("SELECT id FROM chunks WHERE document_sha256 = ?1")?;
            let ids: Vec<String> = stale
                .query_map([&sha], |row| row.get(0))?
                .filter_map(Result::ok)
                .collect();
            for id in ids {
                tx.execute("DELETE FROM chunk_text WHERE id = ?1", [&id])?;
                // The vector belongs to the text that produced it. Re-reading a
                // document with a better engine changes the text, and a vector
                // left behind would rank a passage by what it used to say.
                tx.execute("DELETE FROM chunk_embeddings WHERE chunk_id = ?1", [&id])?;
                tx.execute("DELETE FROM embedding_failures WHERE chunk_id = ?1", [&id])?;
            }
        }
        tx.execute("DELETE FROM chunks WHERE document_sha256 = ?1", [&sha])?;

        for chunk in chunks {
            tx.execute(
                "INSERT INTO chunks
                    (id, document_sha256, document_name, ordinal, page, section_path,
                     kind, classification, superseded, collection_id)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, 0, ?9)",
                params![
                    chunk.id,
                    chunk.document_sha256,
                    document_name,
                    chunk.ordinal,
                    chunk.page,
                    serde_json::to_string(&chunk.section_path)?,
                    serde_json::to_string(&chunk.kind)?,
                    serde_json::to_string(&classification)?,
                    collection_id,
                ],
            )?;

            // The heading trail is indexed alongside the body, so a search for
            // "wall thickness" finds a passage sitting under that heading even
            // when the passage itself only says "minimum 9.0 mm".
            let searchable = if chunk.section_path.is_empty() {
                chunk.text.clone()
            } else {
                format!("{}\n{}", chunk.section_path.join(" "), chunk.text)
            };

            tx.execute(
                "INSERT INTO chunk_text (id, body) VALUES (?1, ?2)",
                params![chunk.id, searchable],
            )?;
        }

        tx.commit()?;
        log::info!("[KNOWLEDGE] indexed {} chunk(s) from {document_name}", chunks.len());
        Ok(chunks.len())
    }

    // ── Source versions ──────────────────────────────────────────────────

    /// Records that `relative_path` in a collection now holds `sha`.
    ///
    /// Called after the document's passages are indexed. A path that held other
    /// bytes has its previous version superseded — its passages stay, so a
    /// citation to them still resolves, and stop answering questions.
    pub fn record_source_version(
        &self,
        collection_id: &str,
        relative_path: &str,
        document_name: &str,
        sha: &str,
    ) -> Result<SourceChange> {
        let mut conn = self.lock();
        let tx = conn.transaction()?;
        let current: Option<(u32, String)> = tx
            .query_row(
                "SELECT version, document_sha256 FROM source_versions
                 WHERE collection_id = ?1 AND relative_path = ?2 AND status = 'current'",
                params![collection_id, relative_path],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .ok();
        let now = chrono::Utc::now().to_rfc3339();

        if let Some((version, held)) = &current {
            if held == sha {
                // The same bytes. Make sure they answer, and change nothing else.
                tx.execute(
                    "UPDATE chunks SET superseded = 0, collection_id = ?2 WHERE document_sha256 = ?1",
                    params![sha, collection_id],
                )?;
                tx.commit()?;
                return Ok(SourceChange::Unchanged { version: *version });
            }
        }

        let revision = Self::tick(&tx)?;
        let latest: u32 = tx.query_row(
            "SELECT COALESCE(MAX(version), 0) FROM source_versions
             WHERE collection_id = ?1 AND relative_path = ?2",
            params![collection_id, relative_path],
            |row| row.get(0),
        )?;
        let version = latest + 1;

        if let Some((_, previous)) = &current {
            tx.execute(
                "UPDATE source_versions
                 SET status = 'superseded', retired_rev = ?3, superseded_by = ?4, retired_at = ?5
                 WHERE collection_id = ?1 AND relative_path = ?2 AND status = 'current'",
                params![collection_id, relative_path, revision, sha, now],
            )?;
            Self::retire_passages_unless_current_elsewhere(&tx, previous)?;
        }

        tx.execute(
            "INSERT INTO source_versions
                (collection_id, relative_path, version, document_sha256, document_name, status,
                 added_rev, indexed_at)
             VALUES (?1, ?2, ?3, ?4, ?5, 'current', ?6, ?7)",
            params![collection_id, relative_path, version, sha, document_name, revision, now],
        )?;
        tx.execute(
            "UPDATE chunks SET superseded = 0, collection_id = ?2 WHERE document_sha256 = ?1",
            params![sha, collection_id],
        )?;
        tx.commit()?;

        Ok(match current {
            Some((_, previous)) => SourceChange::Revised {
                version,
                previous_sha256: previous,
            },
            None => SourceChange::Added { version },
        })
    }

    /// Marks a path's passages as no longer answering questions, unless the
    /// same bytes are still current under another path.
    fn retire_passages_unless_current_elsewhere(
        tx: &rusqlite::Transaction<'_>,
        sha: &str,
    ) -> Result<()> {
        let still_current: bool = tx
            .prepare(
                "SELECT 1 FROM source_versions WHERE document_sha256 = ?1 AND status = 'current'",
            )?
            .exists([sha])?;
        if !still_current {
            tx.execute("UPDATE chunks SET superseded = 1 WHERE document_sha256 = ?1", [sha])?;
        }
        Ok(())
    }

    /// Retires a path: gone from its folder, or revoked by a person.
    ///
    /// Returns the content hash that stopped answering, so the caller can tell
    /// the memory graph which claims rested on it.
    pub fn retire_source(
        &self,
        collection_id: &str,
        relative_path: &str,
        why: Retirement,
    ) -> Result<Option<String>> {
        let mut conn = self.lock();
        let tx = conn.transaction()?;
        let sha: Option<String> = tx
            .query_row(
                "SELECT document_sha256 FROM source_versions
                 WHERE collection_id = ?1 AND relative_path = ?2 AND status = 'current'",
                params![collection_id, relative_path],
                |row| row.get(0),
            )
            .ok();
        let Some(sha) = sha else {
            tx.commit()?;
            return Ok(None);
        };
        let revision = Self::tick(&tx)?;
        tx.execute(
            "UPDATE source_versions SET status = ?3, retired_rev = ?4, retired_at = ?5
             WHERE collection_id = ?1 AND relative_path = ?2 AND status = 'current'",
            params![
                collection_id,
                relative_path,
                why.as_str(),
                revision,
                chrono::Utc::now().to_rfc3339()
            ],
        )?;
        Self::retire_passages_unless_current_elsewhere(&tx, &sha)?;
        tx.commit()?;
        Ok(Some(sha))
    }

    /// Revokes every current source with these bytes, wherever it is held.
    pub fn revoke_document(&self, sha: &str) -> Result<usize> {
        let paths: Vec<(String, String)> = {
            let conn = self.lock();
            let mut stmt = conn.prepare(
                "SELECT collection_id, relative_path FROM source_versions
                 WHERE document_sha256 = ?1 AND status = 'current'",
            )?;
            let rows = stmt.query_map([sha], |row| Ok((row.get(0)?, row.get(1)?)))?;
            rows.filter_map(Result::ok).collect()
        };
        for (collection, path) in &paths {
            self.retire_source(collection, path, Retirement::Revoked)?;
        }
        // A document indexed outside any collection has no history row; its
        // passages are retired directly so revocation still takes effect.
        let conn = self.lock();
        conn.execute("UPDATE chunks SET superseded = 1 WHERE document_sha256 = ?1", [sha])?;
        Ok(paths.len())
    }

    /// Every current path of a collection, for a sync to retire what vanished.
    pub fn current_paths(&self, collection_id: &str) -> Result<Vec<(String, String)>> {
        let conn = self.lock();
        let mut stmt = conn.prepare(
            "SELECT relative_path, document_sha256 FROM source_versions
             WHERE collection_id = ?1 AND status = 'current' ORDER BY relative_path",
        )?;
        let rows = stmt.query_map([collection_id], |row| Ok((row.get(0)?, row.get(1)?)))?;
        Ok(rows.filter_map(Result::ok).collect())
    }

    /// Sets who a collection answers.
    ///
    /// Takes effect on the next statement: nothing caches a reader's view of a
    /// collection, so revoking a role is not waiting on anything to expire.
    pub fn set_collection_access(
        &self,
        collection_id: &str,
        enabled: bool,
        restricted_to: &[crate::identity::Role],
    ) -> Result<()> {
        let mut conn = self.lock();
        let tx = conn.transaction()?;
        tx.execute(
            "INSERT INTO collection_access (collection_id, enabled, restricted, updated_at)
             VALUES (?1, ?2, ?3, ?4)
             ON CONFLICT(collection_id) DO UPDATE
               SET enabled = excluded.enabled, restricted = excluded.restricted,
                   updated_at = excluded.updated_at",
            params![
                collection_id,
                enabled as i64,
                (!restricted_to.is_empty()) as i64,
                chrono::Utc::now().to_rfc3339()
            ],
        )?;
        tx.execute("DELETE FROM collection_roles WHERE collection_id = ?1", [collection_id])?;
        for role in restricted_to {
            tx.execute(
                "INSERT OR IGNORE INTO collection_roles (collection_id, role) VALUES (?1, ?2)",
                params![collection_id, serde_json::to_string(role)?],
            )?;
        }
        tx.commit()?;
        Ok(())
    }

    /// The version history of a document, if this reader may see it.
    ///
    /// Visibility is the same clause search uses, *without* the current-only
    /// filter: a superseded version stays traceable to a reader cleared for it,
    /// because a citation made against it must still resolve. A document this
    /// reader may not see returns `None`, exactly as one that was never indexed
    /// does.
    pub fn source_versions(&self, session: &Session, sha: &str) -> Result<Option<Vec<SourceVersion>>> {
        let clearance = Clearance::of(session);
        if clearance.is_empty() {
            return Ok(None);
        }
        let mut values = vec![Value::Text(sha.to_string())];
        let clause = clearance.clause(&mut values);
        let visible: bool = {
            let conn = self.lock();
            let sql = format!(
                "SELECT 1 FROM chunks c WHERE c.document_sha256 = ? AND {clause} LIMIT 1"
            );
            let mut stmt = conn.prepare(&sql)?;
            let exists = stmt.exists(params_from_iter(values))?;
            exists
        };
        if !visible {
            return Ok(None);
        }

        let conn = self.lock();
        // Every path that ever held these bytes, and every version of each.
        let mut stmt = conn.prepare(
            "SELECT v.collection_id, v.relative_path, v.version, v.document_sha256,
                    v.document_name, v.status, v.added_rev, v.retired_rev, v.superseded_by,
                    v.indexed_at, v.retired_at
             FROM source_versions v
             WHERE (v.collection_id, v.relative_path) IN
                   (SELECT collection_id, relative_path FROM source_versions
                    WHERE document_sha256 = ?1)
             ORDER BY v.collection_id, v.relative_path, v.version",
        )?;
        let rows = stmt.query_map([sha], |row| {
            Ok(SourceVersion {
                collection_id: row.get(0)?,
                relative_path: row.get(1)?,
                version: row.get(2)?,
                document_sha256: row.get(3)?,
                document_name: row.get(4)?,
                status: row.get(5)?,
                added_revision: row.get(6)?,
                retired_revision: row.get(7)?,
                superseded_by: row.get(8)?,
                indexed_at: row.get(9)?,
                retired_at: row.get(10)?,
            })
        })?;
        Ok(Some(rows.filter_map(Result::ok).collect()))
    }

    /// What a pinned revision means now.
    pub fn resolve_pin(&self, pinned: i64) -> Result<PinnedCorpus> {
        let conn = self.lock();
        let mut corpus = PinnedCorpus::default();
        {
            // Current now, and every current row was added after the pin.
            let mut stmt = conn.prepare(
                "SELECT document_sha256 FROM source_versions
                 WHERE status = 'current'
                 GROUP BY document_sha256
                 HAVING MIN(added_rev) > ?1",
            )?;
            for sha in stmt.query_map([pinned], |row| row.get::<_, String>(0))?.flatten() {
                corpus.added_since.insert(sha);
            }
        }
        {
            // Current at the pin, superseded since. Withdrawn and revoked
            // versions are deliberately absent: revocation wins over the pin.
            let mut stmt = conn.prepare(
                "SELECT DISTINCT document_sha256 FROM source_versions
                 WHERE status = 'superseded' AND added_rev <= ?1 AND retired_rev > ?1",
            )?;
            for sha in stmt.query_map([pinned], |row| row.get::<_, String>(0))?.flatten() {
                corpus.superseded_since.insert(sha);
            }
        }
        Ok(corpus)
    }

    /// The scope clause: which passages count as "the corpus" for this search.
    fn scope_clause(&self, scope: &SearchScope, values: &mut Vec<Value>) -> Result<String> {
        let mut clause = match scope.pinned_revision {
            None => "c.superseded = 0".to_string(),
            Some(pinned) => {
                let corpus = self.resolve_pin(pinned)?;
                values.push(json_list(corpus.added_since.iter()));
                values.push(json_list(corpus.superseded_since.iter()));
                "((c.superseded = 0 AND c.document_sha256 NOT IN (SELECT value FROM json_each(?)))
                  OR c.document_sha256 IN (SELECT value FROM json_each(?)))"
                    .to_string()
            }
        };
        if let Some(documents) = &scope.documents {
            values.push(json_list(documents.iter()));
            clause.push_str(" AND c.document_sha256 IN (SELECT value FROM json_each(?))");
        }
        if let Some(collections) = &scope.collections {
            values.push(json_list(collections.iter()));
            clause.push_str(" AND c.collection_id IN (SELECT value FROM json_each(?))");
        }
        Ok(clause)
    }

    /// Marks a document superseded.
    ///
    /// Its passages stay in the index and remain traceable — a citation made
    /// last year must still resolve — but they are not returned as current
    /// guidance, which is what ARJUN design rule 11 asks for.
    pub fn supersede(&self, document_sha256: &str) -> Result<()> {
        let conn = self.lock();
        conn.execute(
            "UPDATE chunks SET superseded = 1 WHERE document_sha256 = ?1",
            [document_sha256],
        )?;
        Ok(())
    }

    // ── Keyword ──────────────────────────────────────────────────────────

    /// Searches, returning only what this person is cleared to see.
    ///
    /// Every term must match. That is right for the literal query this was
    /// built for — a tag, a clause number — and wrong for a question, which is
    /// why the hybrid path uses [`Self::search_lexical`] instead.
    ///
    /// The clearance test is a bound parameter in the query. It is not a filter
    /// over results, because by then the passages exist and their number is
    /// already informative.
    pub fn search(&self, session: &Session, query: &str, limit: usize) -> Result<Vec<SearchResult>> {
        let Some(expression) = to_match_expression(query) else {
            return Ok(Vec::new());
        };
        Ok(self
            .fts(session, &expression, limit, &SearchScope::default())?
            .into_iter()
            .map(|(result, _)| result)
            .collect())
    }

    /// The keyword half of hybrid retrieval: all terms first, then any term.
    ///
    /// A question asked in words ("what is the minimum wall thickness for the
    /// vessel?") matches no passage on all of its terms, because no passage
    /// says "what". So after the all-terms pass, the significant terms are
    /// matched individually and those hits are ranked after every all-terms
    /// hit. Two passes rather than one OR query so a passage that contains the
    /// whole phrase is never outranked by one that repeats a single word.
    pub fn search_lexical(
        &self,
        session: &Session,
        query: &str,
        limit: usize,
        scope: &SearchScope,
    ) -> Result<Vec<LexicalHit>> {
        let mut out: Vec<LexicalHit> = Vec::new();
        if let Some(expression) = to_match_expression(query) {
            for (result, bm25) in self.fts(session, &expression, limit, scope)? {
                out.push(LexicalHit {
                    result,
                    pass: LexicalPass::AllTerms,
                    bm25,
                });
            }
        }
        if out.len() < limit {
            if let Some(expression) = any_term_expression(query) {
                for (result, bm25) in self.fts(session, &expression, limit, scope)? {
                    if out.iter().any(|hit| hit.result.chunk_id == result.chunk_id) {
                        continue;
                    }
                    out.push(LexicalHit {
                        result,
                        pass: LexicalPass::AnyTerm,
                        bm25,
                    });
                    if out.len() >= limit {
                        break;
                    }
                }
            }
        }
        Ok(out)
    }

    fn fts(
        &self,
        session: &Session,
        expression: &str,
        limit: usize,
        scope: &SearchScope,
    ) -> Result<Vec<(SearchResult, f64)>> {
        let clearance = Clearance::of(session);
        if clearance.is_empty() {
            // Cleared for nothing. An empty result is the correct and complete
            // answer — and identical to what they would see if nothing matched,
            // which is the point.
            return Ok(Vec::new());
        }
        let mut values = vec![Value::Text(expression.to_string())];
        let scope_sql = self.scope_clause(scope, &mut values)?;
        let clause = clearance.clause(&mut values);
        values.push(Value::Integer(limit as i64));
        let sql = format!(
            "SELECT c.id, c.document_sha256, c.document_name, t.body, c.page,
                    c.section_path, c.classification, bm25(chunk_text) AS score
             FROM chunk_text t
             JOIN chunks c ON c.id = t.id
             WHERE chunk_text MATCH ?
               AND {scope_sql}
               AND {clause}
             ORDER BY score, c.document_sha256, c.ordinal
             LIMIT ?"
        );

        let conn = self.lock();
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map(params_from_iter(values), |row| {
            let section_path: String = row.get(5)?;
            let classification: String = row.get(6)?;
            let score: f64 = row.get(7)?;
            Ok((
                SearchResult {
                    chunk_id: row.get(0)?,
                    document_sha256: row.get(1)?,
                    document_name: row.get(2)?,
                    text: row.get(3)?,
                    page: row.get(4)?,
                    section_path: serde_json::from_str(&section_path).unwrap_or_default(),
                    classification: serde_json::from_str(&classification)
                        .unwrap_or(Classification::Internal),
                    score,
                    retrieval: Retrieval::Keyword,
                },
                score,
            ))
        })?;

        Ok(rows.filter_map(Result::ok).collect())
    }

    // ── Vectors ──────────────────────────────────────────────────────────

    /// Declares a vector space. Idempotent for the same identity.
    pub fn register_space(&self, identity: &crate::knowledge::embedding::EmbeddingIdentity) -> Result<()> {
        let conn = self.lock();
        conn.execute(
            "INSERT OR IGNORE INTO vector_spaces
                (space_key, model_id, profile, profile_version, weights_sha256, dimensions,
                 pooling, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                identity.space_key(),
                identity.model_id,
                identity.profile,
                identity.profile_version,
                identity.weights_sha256,
                identity.dimensions as i64,
                identity.pooling.as_str(),
                chrono::Utc::now().to_rfc3339()
            ],
        )?;
        Ok(())
    }

    fn space_dimensions(conn: &Connection, space_key: &str) -> Result<usize> {
        let dimensions: i64 = conn
            .query_row(
                "SELECT dimensions FROM vector_spaces WHERE space_key = ?1",
                [space_key],
                |row| row.get(0),
            )
            .with_context(|| format!("no vector space {space_key} is registered"))?;
        Ok(dimensions as usize)
    }

    /// Passages this space has not embedded yet, in reading order.
    ///
    /// The work list for the embedding pass, and what makes the pass resumable:
    /// it asks again after every batch, so a pass interrupted halfway — by a
    /// shutdown, a crash, a cancelled job — resumes where it stopped.
    pub fn passages_needing_embeddings(&self, space_key: &str, limit: usize) -> Result<Vec<PendingPassage>> {
        let conn = self.lock();
        let mut stmt = conn.prepare(
            "SELECT c.id, t.body
             FROM chunks c
             JOIN chunk_text t ON t.id = c.id
             WHERE c.superseded = 0
               AND NOT EXISTS (SELECT 1 FROM chunk_embeddings e
                               WHERE e.space_key = ?1 AND e.chunk_id = c.id)
               AND NOT EXISTS (SELECT 1 FROM embedding_failures f
                               WHERE f.space_key = ?1 AND f.chunk_id = c.id)
             ORDER BY c.document_sha256, c.ordinal
             LIMIT ?2",
        )?;
        let rows = stmt.query_map(params![space_key, limit as i64], |row| {
            Ok(PendingPassage {
                chunk_id: row.get(0)?,
                body: row.get(1)?,
            })
        })?;
        Ok(rows.filter_map(Result::ok).collect())
    }

    /// How many passages the embedding pass still has to do in this space.
    pub fn count_needing_embeddings(&self, space_key: &str) -> Result<usize> {
        let conn = self.lock();
        let count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM chunks c
             WHERE c.superseded = 0
               AND NOT EXISTS (SELECT 1 FROM chunk_embeddings e
                               WHERE e.space_key = ?1 AND e.chunk_id = c.id)
               AND NOT EXISTS (SELECT 1 FROM embedding_failures f
                               WHERE f.space_key = ?1 AND f.chunk_id = c.id)",
            [space_key],
            |row| row.get(0),
        )?;
        Ok(count as usize)
    }

    /// Stores one passage's window vectors in a space.
    ///
    /// Refused when the passage's text changed since it was handed out — a
    /// re-read landed while the model was working — because those vectors
    /// describe text that no longer exists. Returns whether anything was stored.
    pub fn store_embeddings(
        &self,
        space_key: &str,
        chunk_id: &str,
        body: &str,
        windows: &[Vec<f32>],
    ) -> Result<bool> {
        let mut conn = self.lock();
        let dimensions = Self::space_dimensions(&conn, space_key)?;
        if windows.is_empty() {
            anyhow::bail!("no vectors to store for {chunk_id}");
        }
        if let Some(wrong) = windows.iter().find(|window| window.len() != dimensions) {
            anyhow::bail!(
                "a {}-dimension vector cannot be stored in {space_key}, which holds {dimensions}",
                wrong.len()
            );
        }
        let tx = conn.transaction()?;
        let held: Option<String> = tx
            .query_row("SELECT body FROM chunk_text WHERE id = ?1", [chunk_id], |row| row.get(0))
            .ok();
        if held.as_deref() != Some(body) {
            tx.commit()?;
            return Ok(false);
        }
        let text_sha = sha256_text(body);
        tx.execute(
            "DELETE FROM chunk_embeddings WHERE space_key = ?1 AND chunk_id = ?2",
            params![space_key, chunk_id],
        )?;
        for (window, vector) in windows.iter().enumerate() {
            tx.execute(
                "INSERT INTO chunk_embeddings
                    (space_key, chunk_id, window_no, text_sha256, dimensions, vector)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                params![
                    space_key,
                    chunk_id,
                    window as i64,
                    text_sha,
                    dimensions as i64,
                    vector_to_bytes(vector)
                ],
            )?;
        }
        tx.execute(
            "DELETE FROM embedding_failures WHERE space_key = ?1 AND chunk_id = ?2",
            params![space_key, chunk_id],
        )?;
        tx.commit()?;
        Ok(true)
    }

    /// Records a passage the model could not embed, so the pass moves on.
    pub fn record_embedding_failure(&self, space_key: &str, chunk_id: &str, reason: &str) -> Result<()> {
        let conn = self.lock();
        conn.execute(
            "INSERT OR REPLACE INTO embedding_failures (space_key, chunk_id, reason, at)
             VALUES (?1, ?2, ?3, ?4)",
            params![space_key, chunk_id, reason, chrono::Utc::now().to_rfc3339()],
        )?;
        Ok(())
    }

    /// How much of what this reader may see a space covers.
    ///
    /// Counted over the reader's own authorised passages, never over the whole
    /// index: "your library is 40% embedded" is about their library, while a
    /// whole-index figure would tell them how much there is that they cannot
    /// see.
    pub fn embedding_coverage(
        &self,
        session: &Session,
        space_key: &str,
        scope: &SearchScope,
    ) -> Result<EmbeddingCoverage> {
        let clearance = Clearance::of(session);
        if clearance.is_empty() {
            return Ok(EmbeddingCoverage::default());
        }
        let mut values = vec![Value::Text(space_key.to_string()), Value::Text(space_key.to_string())];
        let scope_sql = self.scope_clause(scope, &mut values)?;
        let clause = clearance.clause(&mut values);
        let sql = format!(
            "SELECT COUNT(*),
                    SUM(CASE WHEN EXISTS (SELECT 1 FROM chunk_embeddings e
                                          WHERE e.space_key = ? AND e.chunk_id = c.id)
                             THEN 1 ELSE 0 END),
                    SUM(CASE WHEN EXISTS (SELECT 1 FROM embedding_failures f
                                          WHERE f.space_key = ? AND f.chunk_id = c.id)
                             THEN 1 ELSE 0 END)
             FROM chunks c
             WHERE {scope_sql} AND {clause}"
        );
        // The two space-key values are bound before the scope and clearance
        // values, in the order the placeholders appear.
        let conn = self.lock();
        let (total, embedded, failed): (i64, Option<i64>, Option<i64>) = conn.query_row(
            &sql,
            params_from_iter(values),
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )?;
        Ok(EmbeddingCoverage {
            embedded: embedded.unwrap_or(0) as usize,
            failed: failed.unwrap_or(0) as usize,
            total: total as usize,
        })
    }

    /// Searches one vector space, returning only what this person may see.
    ///
    /// ## The clearance rule is the same one, in the same place
    ///
    /// A passage the asker cannot see is never fetched, so its vector is never
    /// compared and it cannot influence the ranking of anything else — which a
    /// filter applied after scoring could not promise.
    ///
    /// ## Why the arithmetic is in Rust and not in SQL
    ///
    /// SQLite has no vector type and no cosine function without an extension,
    /// and an extension is a native dependency an air-gapped installer would
    /// have to carry. The rows are reduced to what this person may read before
    /// any of them are scored, so the work is bounded by the reader's own
    /// library rather than by the whole index.
    ///
    /// A passage with several windows scores as its closest window.
    pub fn search_dense(
        &self,
        session: &Session,
        space_key: &str,
        query: &[f32],
        limit: usize,
        scope: &SearchScope,
    ) -> Result<Vec<DenseHit>> {
        if query.is_empty() {
            return Ok(Vec::new());
        }
        let clearance = Clearance::of(session);
        if clearance.is_empty() {
            // Cleared for nothing, and the same answer keyword search gives:
            // empty, and indistinguishable from nothing having matched.
            return Ok(Vec::new());
        }
        let mut values = vec![Value::Text(space_key.to_string())];
        let scope_sql = self.scope_clause(scope, &mut values)?;
        let clause = clearance.clause(&mut values);
        let sql = format!(
            "SELECT c.id, c.document_sha256, c.document_name, t.body, c.page,
                    c.section_path, c.classification, e.window_no, e.vector, e.dimensions
             FROM chunk_embeddings e
             JOIN chunks c ON c.id = e.chunk_id
             JOIN chunk_text t ON t.id = e.chunk_id
             WHERE e.space_key = ?
               AND {scope_sql}
               AND {clause}"
        );

        let conn = self.lock();
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map(params_from_iter(values), |row| {
            let section_path: String = row.get(5)?;
            let classification: String = row.get(6)?;
            let window: i64 = row.get(7)?;
            let bytes: Vec<u8> = row.get(8)?;
            let dimensions: i64 = row.get(9)?;
            Ok((
                SearchResult {
                    chunk_id: row.get(0)?,
                    document_sha256: row.get(1)?,
                    document_name: row.get(2)?,
                    text: row.get(3)?,
                    page: row.get(4)?,
                    section_path: serde_json::from_str(&section_path).unwrap_or_default(),
                    classification: serde_json::from_str(&classification)
                        .unwrap_or(Classification::Internal),
                    score: 0.0,
                    retrieval: Retrieval::Vector,
                },
                window as u32,
                bytes,
                dimensions as usize,
            ))
        })?;

        let mut best: BTreeMap<String, DenseHit> = BTreeMap::new();
        for (result, window, bytes, dimensions) in rows.filter_map(Result::ok) {
            // A stored vector of a different width cannot be compared, and
            // truncating to the shorter length would still produce a number.
            if dimensions != query.len() {
                continue;
            }
            let Some(vector) = bytes_to_vector(&bytes, dimensions) else {
                continue;
            };
            let Some(similarity) = cosine_similarity(query, &vector) else {
                continue;
            };
            let replace = best
                .get(&result.chunk_id)
                .map_or(true, |held| similarity > held.similarity);
            if replace {
                let mut result = result;
                // A distance, so lower stays better here exactly as for bm25.
                result.score = 1.0 - similarity;
                best.insert(
                    result.chunk_id.clone(),
                    DenseHit {
                        result,
                        similarity,
                        window,
                    },
                );
            }
        }

        let mut scored: Vec<DenseHit> = best.into_values().collect();
        // Nearest first; ties by document and passage so the order is total and
        // a citation cannot move between identical runs.
        scored.sort_by(|a, b| {
            b.similarity
                .total_cmp(&a.similarity)
                .then_with(|| a.result.document_sha256.cmp(&b.result.document_sha256))
                .then_with(|| a.result.chunk_id.cmp(&b.result.chunk_id))
        });
        scored.truncate(limit);
        Ok(scored)
    }

    /// The chunks of one document between two pages, in reading order.
    ///
    /// ## Why a region and not the document
    ///
    /// The obvious way to let a model "read more" is to hand it the document.
    /// On this workbench a document is a 200-page drawing set, and pasting one
    /// into an 8k window does not give the model more context — it ends the run,
    /// because the inference server refuses a prompt at or over its window.
    ///
    /// So the model asks for the part it wants. A search returns a passage and
    /// its page; when that passage is cut off mid-clause, the next thing the
    /// model needs is pages 11 to 13 of that document, not the whole of it.
    ///
    /// ## The checks are the same ones search uses
    ///
    /// Clearance is applied here rather than at the caller, and it is applied
    /// the same way: a reader sees only classifications their roles are cleared
    /// for, only collections that answer them, and superseded documents are
    /// excluded. That matters because this is a second door into the same shelf
    /// — a retrieval path that filtered less than search would be a way to read
    /// by page number what could not be read by searching for it.
    ///
    /// `to_page` is inclusive. A reader who names one page gets that page.
    pub fn region(
        &self,
        session: &Session,
        document_sha256: &str,
        from_page: u32,
        to_page: u32,
        limit: usize,
    ) -> Result<Vec<SearchResult>> {
        // A reversed range is a caller mistake, not a reason to return nothing
        // silently: read the way it was plainly meant.
        let (from_page, to_page) = if from_page <= to_page {
            (from_page, to_page)
        } else {
            (to_page, from_page)
        };

        let clearance = Clearance::of(session);
        if clearance.is_empty() {
            return Ok(Vec::new());
        }
        let mut values = vec![
            Value::Text(document_sha256.to_string()),
            Value::Integer(from_page as i64),
            Value::Integer(to_page as i64),
        ];
        let clause = clearance.clause(&mut values);
        values.push(Value::Integer(limit as i64));
        let sql = format!(
            "SELECT c.id, c.document_sha256, c.document_name, t.body, c.page,
                    c.section_path, c.classification
             FROM chunks c
             JOIN chunk_text t ON c.id = t.id
             WHERE c.document_sha256 = ?
               AND c.page >= ?
               AND c.page <= ?
               AND c.superseded = 0
               AND {clause}
             ORDER BY c.page, c.ordinal
             LIMIT ?"
        );

        let conn = self.lock();
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map(params_from_iter(values), |row| {
            let section_path: String = row.get(5)?;
            let classification: String = row.get(6)?;
            Ok(SearchResult {
                chunk_id: row.get(0)?,
                document_sha256: row.get(1)?,
                document_name: row.get(2)?,
                text: row.get(3)?,
                page: row.get(4)?,
                section_path: serde_json::from_str(&section_path).unwrap_or_default(),
                classification: serde_json::from_str(&classification)
                    .unwrap_or(Classification::Internal),
                // Not a match against a query, so there is no relevance to
                // report. Zero rather than an invented figure: a score the
                // caller might sort by would be a lie about ranking.
                score: 0.0,
                retrieval: Retrieval::Keyword,
            })
        })?;

        Ok(rows.filter_map(Result::ok).collect())
    }

    /// One passage by id, if this reader may see it and it is current.
    ///
    /// How an evidence handle is resolved. The same clause as every other read,
    /// so a handle to a passage the reader may not see resolves to nothing.
    pub fn passage(&self, session: &Session, chunk_id: &str) -> Result<Option<SearchResult>> {
        let clearance = Clearance::of(session);
        if clearance.is_empty() {
            return Ok(None);
        }
        let mut values = vec![Value::Text(chunk_id.to_string())];
        let clause = clearance.clause(&mut values);
        let sql = format!(
            "SELECT c.id, c.document_sha256, c.document_name, t.body, c.page,
                    c.section_path, c.classification
             FROM chunks c JOIN chunk_text t ON c.id = t.id
             WHERE c.id = ? AND c.superseded = 0 AND {clause}"
        );
        let conn = self.lock();
        let mut stmt = conn.prepare(&sql)?;
        let mut rows = stmt.query_map(params_from_iter(values), |row| {
            let section_path: String = row.get(5)?;
            let classification: String = row.get(6)?;
            Ok(SearchResult {
                chunk_id: row.get(0)?,
                document_sha256: row.get(1)?,
                document_name: row.get(2)?,
                text: row.get(3)?,
                page: row.get(4)?,
                section_path: serde_json::from_str(&section_path).unwrap_or_default(),
                classification: serde_json::from_str(&classification)
                    .unwrap_or(Classification::Internal),
                score: 0.0,
                retrieval: Retrieval::Keyword,
            })
        })?;
        Ok(rows.next().transpose()?)
    }
}

/// The significant terms of a question, OR-ed, each quoted as `sanitise_fts`
/// quotes it.
fn any_term_expression(query: &str) -> Option<String> {
    let mut seen: BTreeSet<String> = BTreeSet::new();
    let terms: Vec<String> = query
        .split_whitespace()
        .map(|token| token.trim_matches(|c: char| !c.is_alphanumeric()))
        .filter(|token| token.chars().count() >= 2)
        .filter(|token| !STOPWORDS.contains(&token.to_lowercase().as_str()))
        .filter(|token| seen.insert(token.to_lowercase()))
        .map(|token| format!("\"{}\"", token.replace('"', "\"\"")))
        .collect();
    (!terms.is_empty()).then(|| terms.join(" OR "))
}

/// A vector as it is stored: little-endian `f32`, no header.
///
/// The width is a column, not a prefix, so a malformed blob is caught by the
/// length check in [`bytes_to_vector`] rather than by trusting bytes that came
/// out of the same row that is being checked.
fn vector_to_bytes(vector: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(vector.len() * 4);
    for value in vector {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

/// Reads a stored vector back, or `None` when the blob is not the width the row
/// claims.
///
/// Returning `None` rather than a shorter vector is the point: a truncated read
/// would compare a passage on part of its meaning and score it confidently.
fn bytes_to_vector(bytes: &[u8], dimensions: usize) -> Option<Vec<f32>> {
    if bytes.len() != dimensions * 4 {
        return None;
    }
    Some(
        bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect(),
    )
}

/// Cosine similarity, or `None` when it is not defined for these two.
///
/// `None` for mismatched widths and for a zero-magnitude vector, where the
/// quotient would divide by zero. Both are refused rather than defaulted to
/// 0.0, because a zero similarity is a real answer meaning "unrelated", and an
/// undefined comparison reported as unrelated is a silent wrong answer of
/// exactly the kind that is impossible to notice later.
fn cosine_similarity(a: &[f32], b: &[f32]) -> Option<f64> {
    if a.len() != b.len() || a.is_empty() {
        return None;
    }
    let mut dot = 0f64;
    let mut norm_a = 0f64;
    let mut norm_b = 0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let (x, y) = (*x as f64, *y as f64);
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    if norm_a <= 0.0 || norm_b <= 0.0 {
        return None;
    }
    Some(dot / (norm_a.sqrt() * norm_b.sqrt()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::{Role, User};
    use crate::knowledge::{Chunk, ChunkKind};

    fn chunk(id: &str, sha: &str, ordinal: u32, text: &str, section: Vec<&str>) -> Chunk {
        Chunk {
            id: id.into(),
            document_sha256: sha.into(),
            ordinal,
            char_count: text.len() as u32,
            text: text.into(),
            page: 1,
            section_path: section.into_iter().map(String::from).collect(),
            kind: ChunkKind::Prose,
        }
    }

    struct Fixture {
        _dir: tempfile::TempDir,
        index: KnowledgeIndex,
    }

    fn fixture() -> Fixture {
        let dir = tempfile::tempdir().unwrap();
        let index = KnowledgeIndex::open(dir.path()).unwrap();
        Fixture { _dir: dir, index }
    }

    fn session(roles: Vec<Role>) -> Session {
        Session::open(User::new("kiran", "Kiran", roles))
    }

    fn index_sop(f: &Fixture) {
        f.index
            .index_document(
                "Maintenance SOP rev C",
                Classification::ProcessDiagram,
                &[chunk(
                    "c1",
                    "sop",
                    0,
                    "Minimum acceptable wall thickness is 9.0 mm.",
                    vec!["4 Inspection", "4.2 Wall Thickness"],
                )],
            )
            .unwrap();
    }

    fn page_chunk(id: &str, sha: &str, ordinal: u32, page: u32, text: &str) -> Chunk {
        Chunk {
            id: id.into(),
            document_sha256: sha.into(),
            ordinal,
            char_count: text.len() as u32,
            text: text.into(),
            page,
            section_path: Vec::new(),
            kind: ChunkKind::Prose,
        }
    }

    /// A document with one chunk on each of pages 10 to 14.
    fn index_manual(f: &Fixture, classification: Classification) {
        let chunks: Vec<Chunk> = (10..=14)
            .map(|page| {
                page_chunk(
                    &format!("m{page}"),
                    "manual",
                    page,
                    page,
                    &format!("Manual text on page {page}."),
                )
            })
            .collect();
        f.index
            .index_document("Pump Manual", classification, &chunks)
            .unwrap();
    }

    #[test]
    fn a_region_returns_the_pages_asked_for_and_no_others() {
        // The point of loading a region rather than a document: the model asks
        // for three pages of a 200-page manual and gets three pages.
        let f = fixture();
        index_manual(&f, Classification::Internal);

        let hits = f
            .index
            .region(&session(vec![Role::Employee]), "manual", 11, 13, 50)
            .unwrap();

        assert_eq!(
            hits.iter().map(|hit| hit.page).collect::<Vec<_>>(),
            vec![11, 12, 13]
        );
    }

    #[test]
    fn a_region_of_one_page_is_that_page() {
        let f = fixture();
        index_manual(&f, Classification::Internal);

        let hits = f
            .index
            .region(&session(vec![Role::Employee]), "manual", 12, 12, 50)
            .unwrap();

        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].page, 12);
    }

    #[test]
    fn a_region_reads_the_same_clearance_as_a_search() {
        // The failure this guards: a second door into the same shelf that
        // filters less than the first. Reading by page number must not return
        // what searching for the same words would refuse.
        let f = fixture();
        index_manual(&f, Classification::VendorNegotiation);

        // A legacy role is not cleared for any classification. The
        // assertion that the region read returns nothing catches a
        // regression that re-enables a legacy role.
        let refused = f
            .index
            .region(
                &session(vec![Role::Auditor]),
                "manual",
                11,
                13,
                50,
            )
            .unwrap();
        assert!(refused.is_empty());

        let permitted = f
            .index
            .region(&session(vec![Role::Employee]), "manual", 11, 13, 50)
            .unwrap();
        assert_eq!(permitted.len(), 3);
    }

    #[test]
    fn a_region_of_a_superseded_document_is_empty() {
        // Superseded material stays traceable and stops being current guidance.
        // Reading it by page would be a way to quote a withdrawn revision.
        let f = fixture();
        index_manual(&f, Classification::Internal);
        f.index.supersede("manual").unwrap();

        assert!(f
            .index
            .region(&session(vec![Role::Employee]), "manual", 11, 13, 50)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn a_reversed_range_is_read_the_way_it_was_plainly_meant() {
        let f = fixture();
        index_manual(&f, Classification::Internal);

        let hits = f
            .index
            .region(&session(vec![Role::Employee]), "manual", 13, 11, 50)
            .unwrap();

        assert_eq!(hits.len(), 3);
    }

    #[test]
    fn a_region_is_bounded_by_its_limit_however_dense_the_pages_are() {
        let f = fixture();
        index_manual(&f, Classification::Internal);

        let hits = f
            .index
            .region(&session(vec![Role::Employee]), "manual", 10, 14, 2)
            .unwrap();

        // Two, not five. The ceiling is what actually keeps a dense range from
        // filling the window.
        assert_eq!(hits.len(), 2);
    }

    #[test]
    fn a_region_of_a_document_nobody_indexed_is_empty_rather_than_an_error() {
        // A model naming a document it half-remembers should read "nothing
        // there" and search instead, not see a failure it will try to work
        // around.
        let f = fixture();
        index_manual(&f, Classification::Internal);

        assert!(f
            .index
            .region(&session(vec![Role::Employee]), "not-a-document", 1, 5, 50)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn a_region_comes_back_in_reading_order() {
        let f = fixture();
        index_manual(&f, Classification::Internal);

        let hits = f
            .index
            .region(&session(vec![Role::Employee]), "manual", 10, 14, 50)
            .unwrap();
        let pages: Vec<u32> = hits.iter().map(|hit| hit.page).collect();
        let mut sorted = pages.clone();
        sorted.sort_unstable();

        assert_eq!(pages, sorted);
    }

    #[test]
    fn a_cleared_user_finds_the_passage() {
        let f = fixture();
        index_sop(&f);

        let hits = f.index.search(&session(vec![Role::Employee]), "wall thickness", 10).unwrap();
        assert_eq!(hits.len(), 1);
        assert!(hits[0].text.contains("9.0 mm"));
    }

    /// The passage is found through its heading even though the body never says
    /// those words — which is what the heading trail is indexed for.
    #[test]
    fn a_passage_is_findable_through_the_heading_above_it() {
        let f = fixture();
        f.index
            .index_document(
                "SOP",
                Classification::Internal,
                &[chunk("c1", "sop", 0, "Minimum 9.0 mm.", vec!["4.2 Wall Thickness"])],
            )
            .unwrap();

        let hits = f.index.search(&session(vec![Role::Employee]), "thickness", 10).unwrap();
        assert_eq!(hits.len(), 1);
    }

    // ---- The vector half -------------------------------------------------

    use crate::knowledge::embedding::{EmbeddingIdentity, MULTILINGUAL_E5_SMALL};

    /// A registered 384-wide space, as the provider would declare it.
    fn space(f: &Fixture, weights: &str) -> String {
        let identity = EmbeddingIdentity::new("e5", &MULTILINGUAL_E5_SMALL, &weights.repeat(32));
        f.index.register_space(&identity).unwrap();
        identity.space_key()
    }

    fn unit(axis: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; 384];
        v[axis] = 1.0;
        v
    }

    fn body_of(f: &Fixture, chunk_id: &str) -> String {
        f.index
            .passages_needing_embeddings(&space(f, "zz"), 100)
            .unwrap()
            .into_iter()
            .find(|pending| pending.chunk_id == chunk_id)
            .map(|pending| pending.body)
            .unwrap()
    }

    /// Indexes two passages and gives them vectors pointing in different
    /// directions, so "nearest" has an unambiguous right answer.
    fn index_two_with_vectors(f: &Fixture, space_key: &str) {
        f.index
            .index_document(
                "Pump Manual",
                Classification::Internal,
                &[
                    chunk("c-valve", "man", 0, "The control valve was replaced.", vec![]),
                    chunk("c-pump", "man", 1, "The charge pump was overhauled.", vec![]),
                ],
            )
            .unwrap();
        let valve = body_of(f, "c-valve");
        let pump = body_of(f, "c-pump");
        assert!(f.index.store_embeddings(space_key, "c-valve", &valve, &[unit(0)]).unwrap());
        assert!(f.index.store_embeddings(space_key, "c-pump", &pump, &[unit(1)]).unwrap());
    }

    #[test]
    fn the_nearest_vector_ranks_first_and_the_far_one_still_appears() {
        let f = fixture();
        let key = space(&f, "ab");
        index_two_with_vectors(&f, &key);

        let mut query = unit(0);
        query[1] = 0.1;
        let hits = f
            .index
            .search_dense(&session(vec![Role::Employee]), &key, &query, 10, &SearchScope::default())
            .unwrap();

        assert_eq!(hits.len(), 2, "both passages are cleared and embedded");
        assert_eq!(hits[0].result.chunk_id, "c-valve");
        assert_eq!(hits[1].result.chunk_id, "c-pump");
        assert!(hits[0].similarity > hits[1].similarity);
        assert!(hits.iter().all(|hit| hit.result.retrieval == Retrieval::Vector));
    }

    /// The same guarantee keyword search makes, made by the same means: the
    /// clearance is in the SQL, so an uncleared passage is never fetched and
    /// therefore never scored.
    #[test]
    fn an_uncleared_passage_is_never_compared_by_vector() {
        let f = fixture();
        let key = space(&f, "ab");
        f.index
            .index_document(
                "Vendor terms",
                Classification::VendorNegotiation,
                &[chunk("c1", "deal", 0, "Unit price is 4.2 lakh per valve.", vec![])],
            )
            .unwrap();
        let body = body_of(&f, "c1");
        f.index.store_embeddings(&key, "c1", &body, &[unit(0)]).unwrap();

        let hits = f
            .index
            .search_dense(&session(vec![Role::Auditor]), &key, &unit(0), 10, &SearchScope::default())
            .unwrap();
        assert!(hits.is_empty(), "a passage above the reader clearance was scored and returned");
        assert_eq!(
            f.index
                .embedding_coverage(&session(vec![Role::Auditor]), &key, &SearchScope::default())
                .unwrap(),
            EmbeddingCoverage::default(),
            "coverage counted a passage the reader may not see"
        );
    }

    /// Two models' vectors live in two spaces; neither overwrites the other and
    /// neither is compared with the other's query.
    #[test]
    fn a_second_model_adds_a_space_instead_of_overwriting_the_first() {
        let f = fixture();
        let first = space(&f, "ab");
        let second = space(&f, "cd");
        assert_ne!(first, second);
        index_two_with_vectors(&f, &first);
        let reader = session(vec![Role::Employee]);

        assert!(f
            .index
            .search_dense(&reader, &second, &unit(0), 10, &SearchScope::default())
            .unwrap()
            .is_empty());
        let valve = body_of(&f, "c-valve");
        f.index.store_embeddings(&second, "c-valve", &valve, &[unit(5)]).unwrap();
        let from_first = f
            .index
            .search_dense(&reader, &first, &unit(0), 10, &SearchScope::default())
            .unwrap();
        assert_eq!(from_first[0].result.chunk_id, "c-valve");
        assert!((from_first[0].similarity - 1.0).abs() < 1e-6, "the second space overwrote the first");
    }

    /// A vector of a different width than its space is refused at the door.
    #[test]
    fn a_vector_of_the_wrong_width_is_refused_not_stored() {
        let f = fixture();
        let key = space(&f, "ab");
        index_two_with_vectors(&f, &key);
        let valve = body_of(&f, "c-valve");
        assert!(f.index.store_embeddings(&key, "c-valve", &valve, &[vec![1.0, 0.0]]).is_err());
        assert!(f
            .index
            .search_dense(&session(vec![Role::Employee]), &key, &[1.0, 0.0], 10, &SearchScope::default())
            .unwrap()
            .is_empty());
    }

    /// A vector outliving the text it was made from would rank a passage by
    /// what it used to say.
    #[test]
    fn re_reading_a_document_drops_the_vectors_of_the_text_it_replaced() {
        let f = fixture();
        let key = space(&f, "ab");
        index_two_with_vectors(&f, &key);
        let reader = session(vec![Role::Employee]);
        let coverage = f.index.embedding_coverage(&reader, &key, &SearchScope::default()).unwrap();
        assert_eq!((coverage.embedded, coverage.total), (2, 2));

        f.index
            .index_document(
                "Pump Manual",
                Classification::Internal,
                &[chunk("c-merged", "man", 0, "Valve replaced; pump overhauled.", vec![])],
            )
            .unwrap();

        let coverage = f.index.embedding_coverage(&reader, &key, &SearchScope::default()).unwrap();
        assert_eq!(coverage.total, 1, "the re-read replaced both passages with one");
        assert_eq!(coverage.embedded, 0, "the old vectors did not survive their text");
    }

    /// Vectors computed for text that changed while the model worked are not
    /// stored against the new text.
    #[test]
    fn vectors_for_text_that_changed_meanwhile_are_not_stored() {
        let f = fixture();
        let key = space(&f, "ab");
        f.index
            .index_document("M", Classification::Internal, &[chunk("c1", "m", 0, "old words", vec![])])
            .unwrap();
        let handed_out = body_of(&f, "c1");
        f.index
            .index_document("M", Classification::Internal, &[chunk("c1", "m", 0, "new words", vec![])])
            .unwrap();
        assert!(!f.index.store_embeddings(&key, "c1", &handed_out, &[unit(0)]).unwrap());
        assert_eq!(f.index.passages_needing_embeddings(&key, 10).unwrap().len(), 1);
    }

    /// The work list is what makes the embedding pass resumable, and a passage
    /// the model refused is recorded once rather than retried forever.
    #[test]
    fn the_work_list_shrinks_as_passages_are_embedded_or_fail() {
        let f = fixture();
        let key = space(&f, "ab");
        f.index
            .index_document(
                "Pump Manual",
                Classification::Internal,
                &[
                    chunk("c1", "man", 0, "The control valve was replaced.", vec![]),
                    chunk("c2", "man", 1, "The charge pump was overhauled.", vec![]),
                    chunk("c3", "man", 2, "The seal was inspected.", vec![]),
                ],
            )
            .unwrap();

        let outstanding = f.index.passages_needing_embeddings(&key, 10).unwrap();
        assert_eq!(outstanding.len(), 3);
        assert!(outstanding.iter().any(|p| p.chunk_id == "c1" && p.body.contains("valve")));

        f.index.store_embeddings(&key, "c1", &outstanding[0].body, &[unit(0)]).unwrap();
        f.index.record_embedding_failure(&key, "c2", "refused").unwrap();

        let outstanding = f.index.passages_needing_embeddings(&key, 10).unwrap();
        assert_eq!(outstanding.len(), 1);
        assert_eq!(outstanding[0].chunk_id, "c3");
        let coverage = f
            .index
            .embedding_coverage(&session(vec![Role::Employee]), &key, &SearchScope::default())
            .unwrap();
        assert_eq!(coverage, EmbeddingCoverage { embedded: 1, failed: 1, total: 3 });
    }

    /// Half an index embedded must not look like a whole one, and superseded
    /// passages are not current guidance.
    #[test]
    fn coverage_counts_only_live_passages_in_this_space() {
        let f = fixture();
        let key = space(&f, "ab");
        index_two_with_vectors(&f, &key);
        let reader = session(vec![Role::Employee]);
        let other = space(&f, "cd");
        let none = f.index.embedding_coverage(&reader, &other, &SearchScope::default()).unwrap();
        assert_eq!((none.embedded, none.total), (0, 2));

        f.index.supersede("man").unwrap();
        let gone = f.index.embedding_coverage(&reader, &key, &SearchScope::default()).unwrap();
        assert_eq!((gone.embedded, gone.total), (0, 0));
    }

    // ── P07: versions, collections, pins ─────────────────────────────────

    fn index_version(f: &Fixture, collection: &str, path: &str, sha: &str, text: &str) -> SourceChange {
        f.index
            .index_collection_document(
                Some(collection),
                path,
                Classification::Internal,
                &[chunk(&format!("{sha}-0"), sha, 0, text, vec![])],
            )
            .unwrap();
        f.index.record_source_version(collection, path, path, sha).unwrap()
    }

    #[test]
    fn a_revised_source_supersedes_its_previous_version_which_stays_traceable() {
        let f = fixture();
        let reader = session(vec![Role::Employee]);
        assert_eq!(
            index_version(&f, "sops", "isolation.md", "v1", "Isolate the pump with valve V-1."),
            SourceChange::Added { version: 1 }
        );
        assert_eq!(
            index_version(&f, "sops", "isolation.md", "v1", "Isolate the pump with valve V-1."),
            SourceChange::Unchanged { version: 1 }
        );
        assert_eq!(
            index_version(&f, "sops", "isolation.md", "v2", "Isolate the pump with valves V-1 and V-2."),
            SourceChange::Revised { version: 2, previous_sha256: "v1".into() }
        );

        let hits = f.index.search(&reader, "isolate pump", 10).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].document_sha256, "v2", "the superseded version still answered");

        let history = f.index.source_versions(&reader, "v1").unwrap().expect("traceable");
        assert_eq!(history.len(), 2);
        assert_eq!(history[0].status, "superseded");
        assert_eq!(history[0].superseded_by.as_deref(), Some("v2"));
        assert_eq!(history[1].status, "current");
    }

    #[test]
    fn a_withdrawn_or_revoked_source_stops_answering() {
        let f = fixture();
        let reader = session(vec![Role::Employee]);
        index_version(&f, "sops", "a.md", "sa", "Drain the separator slowly.");
        index_version(&f, "sops", "b.md", "sb", "Drain the knockout drum slowly.");
        assert_eq!(f.index.search(&reader, "drain", 10).unwrap().len(), 2);

        assert_eq!(f.index.retire_source("sops", "a.md", Retirement::Withdrawn).unwrap(), Some("sa".into()));
        assert_eq!(f.index.revoke_document("sb").unwrap(), 1);
        assert!(f.index.search(&reader, "drain", 10).unwrap().is_empty());
        let history = f.index.source_versions(&reader, "sb").unwrap().unwrap();
        assert_eq!(history[0].status, "revoked");
    }

    /// Identical bytes under two paths: retiring one path leaves the other
    /// answering.
    #[test]
    fn the_same_bytes_under_another_path_keep_answering() {
        let f = fixture();
        let reader = session(vec![Role::Employee]);
        index_version(&f, "sops", "a.md", "same", "Purge with nitrogen.");
        f.index.record_source_version("sops", "copy/a.md", "a.md", "same").unwrap();
        f.index.retire_source("sops", "a.md", Retirement::Withdrawn).unwrap();
        assert_eq!(f.index.search(&reader, "purge", 10).unwrap().len(), 1);
    }

    #[test]
    fn a_restricted_or_disabled_collection_answers_only_whom_it_names() {
        let f = fixture();
        index_version(&f, "hr", "pay.md", "pay", "Overtime rate is double time.");
        let employee = session(vec![Role::Employee]);
        let admin = session(vec![Role::Administrator]);
        assert_eq!(f.index.search(&employee, "overtime", 10).unwrap().len(), 1);

        f.index.set_collection_access("hr", true, &[Role::Administrator]).unwrap();
        assert!(f.index.search(&employee, "overtime", 10).unwrap().is_empty());
        assert!(f.index.documents(&employee).unwrap().is_empty());
        assert!(f.index.source_versions(&employee, "pay").unwrap().is_none());
        assert!(f.index.passage(&employee, "pay-0").unwrap().is_none());
        assert_eq!(f.index.search(&admin, "overtime", 10).unwrap().len(), 1);

        f.index.set_collection_access("hr", false, &[]).unwrap();
        assert!(f.index.search(&admin, "overtime", 10).unwrap().is_empty());
    }

    /// A pinned search reads the corpus as it stood: a later addition is not
    /// read, a later revision leaves the pinned version readable, and a later
    /// revocation wins over the pin.
    #[test]
    fn a_pinned_search_reads_the_corpus_it_was_queued_against() {
        let f = fixture();
        let reader = session(vec![Role::Employee]);
        index_version(&f, "sops", "flare.md", "f1", "Flare header purge rate 5 m3/h.");
        index_version(&f, "sops", "vent.md", "vent", "Vent header purge rate 2 m3/h.");
        let pinned = f.index.index_revision().unwrap();

        index_version(&f, "sops", "flare.md", "f2", "Flare header purge rate 7 m3/h.");
        index_version(&f, "sops", "new.md", "late", "Drain header purge rate 1 m3/h.");
        f.index.retire_source("sops", "vent.md", Retirement::Revoked).unwrap();

        let scope = SearchScope { pinned_revision: Some(pinned), ..Default::default() };
        let hits = f.index.search_lexical(&reader, "purge rate", 10, &scope).unwrap();
        let shas: BTreeSet<String> = hits.iter().map(|hit| hit.result.document_sha256.clone()).collect();
        assert_eq!(shas, BTreeSet::from(["f1".to_string()]), "{shas:?}");
        let corpus = f.index.resolve_pin(pinned).unwrap();
        assert!(corpus.superseded_since.contains("f1"));
        assert!(corpus.added_since.contains("late"));
        assert!(corpus.added_since.contains("f2"));
    }

    /// A question in words finds the passage that answers it, even though no
    /// passage contains every word of the question.
    #[test]
    fn a_question_in_words_finds_a_passage_by_its_significant_terms() {
        let f = fixture();
        index_sop(&f);
        let reader = session(vec![Role::Employee, Role::Administrator]);
        let question = "What is the minimum wall thickness allowed?";
        assert!(f.index.search(&reader, question, 5).unwrap().is_empty(), "all-terms matched a question");
        let hits = f.index.search_lexical(&reader, question, 5, &SearchScope::default()).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].pass, LexicalPass::AnyTerm);
    }

    #[test]
    fn a_literal_match_outranks_a_single_repeated_term() {
        let f = fixture();
        f.index
            .index_document(
                "M",
                Classification::Internal,
                &[
                    chunk("noise", "m", 0, "Pump pump pump pump pump pump.", vec![]),
                    chunk("exact", "m", 1, "Stop the charge pump before draining.", vec![]),
                ],
            )
            .unwrap();
        let hits = f
            .index
            .search_lexical(&session(vec![Role::Employee]), "charge pump", 5, &SearchScope::default())
            .unwrap();
        assert_eq!(hits[0].result.chunk_id, "exact");
        assert_eq!(hits[0].pass, LexicalPass::AllTerms);
    }

    #[test]
    fn a_vector_survives_the_round_trip_through_storage() {
        let original = vec![1.5f32, -0.25, 0.0, 1e-8];
        let bytes = vector_to_bytes(&original);
        assert_eq!(bytes_to_vector(&bytes, original.len()), Some(original));
    }

    /// A blob that is not the width its row claims is refused rather than read
    /// short, because a passage compared on part of its meaning still gets a
    /// confident score.
    #[test]
    fn a_blob_of_the_wrong_length_is_refused_rather_than_read_short() {
        let bytes = vector_to_bytes(&[1.0, 2.0, 3.0]);
        assert_eq!(bytes_to_vector(&bytes, 4), None);
        assert_eq!(bytes_to_vector(&bytes, 2), None);
        assert!(bytes_to_vector(&bytes, 3).is_some());
    }

    #[test]
    fn cosine_similarity_is_undefined_rather_than_zero_where_it_has_no_meaning() {
        // Identical direction is 1, opposite is -1, orthogonal is 0.
        assert_eq!(cosine_similarity(&[1.0, 0.0], &[2.0, 0.0]), Some(1.0));
        assert_eq!(cosine_similarity(&[1.0, 0.0], &[-1.0, 0.0]), Some(-1.0));
        assert_eq!(cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]), Some(0.0));

        // The cases that must not quietly become "unrelated".
        assert_eq!(cosine_similarity(&[1.0, 0.0], &[0.0, 0.0]), None);
        assert_eq!(cosine_similarity(&[1.0, 0.0], &[1.0, 0.0, 0.0]), None);
        assert_eq!(cosine_similarity(&[], &[]), None);
    }

    /// The requirement this module exists for: an uncleared passage is never
    /// fetched, so it cannot reach a model even to be discarded.
    #[test]
    fn an_uncleared_passage_is_never_returned() {
        let f = fixture();
        f.index
            .index_document(
                "Vendor terms",
                Classification::VendorNegotiation,
                &[chunk("c1", "deal", 0, "Unit price is 4.2 lakh per valve.", vec![])],
            )
            .unwrap();

        // A legacy role is not cleared for any classification. The
        // assertion that the search returns nothing catches a
        // regression that re-enables a legacy role.
        let hits = f
            .index
            .search(&session(vec![Role::Auditor]), "valve", 10)
            .unwrap();
        assert!(hits.is_empty());

        // An Employee is cleared for the commercial tier.
        let hits = f.index.search(&session(vec![Role::Employee]), "valve", 10).unwrap();
        assert_eq!(hits.len(), 1);
    }

    /// An auditor reads the record of what happened, not the documents.
    #[test]
    fn an_auditor_searching_finds_nothing_at_all() {
        let f = fixture();
        index_sop(&f);
        // The legacy `Auditor` role is the test fixture for "no
        // clearance" in the active product. Pinned here so a regression
        // that re-enables a legacy role is caught at the search level.
        let hits = f.index.search(&session(vec![Role::Auditor]), "thickness", 10).unwrap();
        assert!(hits.is_empty());
    }

    /// "No results" and "results you cannot see" must look identical.
    #[test]
    fn an_uncleared_hit_is_indistinguishable_from_no_hit() {
        let f = fixture();
        f.index
            .index_document(
                "Vendor terms",
                Classification::VendorNegotiation,
                &[chunk("c1", "deal", 0, "Unit price per valve.", vec![])],
            )
            .unwrap();

        // A legacy role that is not cleared for any classification: the
        // vendor-term hit must look identical to a hit on a term that
        // returns no results.
        let uncleared = f
            .index
            .search(&session(vec![Role::Auditor]), "valve", 10)
            .unwrap();
        let absent = f
            .index
            .search(&session(vec![Role::Auditor]), "sasquatch", 10)
            .unwrap();

        assert_eq!(uncleared.len(), absent.len());
    }

    #[test]
    fn a_superseded_document_stays_traceable_but_is_not_returned() {
        let f = fixture();
        index_sop(&f);
        assert_eq!(f.index.search(&session(vec![Role::Employee]), "thickness", 10).unwrap().len(), 1);

        f.index.supersede("sop").unwrap();
        assert!(f.index.search(&session(vec![Role::Employee]), "thickness", 10).unwrap().is_empty());

        // Still held, so a citation made before it was superseded still resolves.
        let conn = f.index.conn.lock().unwrap();
        let held: i64 = conn
            .query_row("SELECT COUNT(*) FROM chunks WHERE document_sha256 = 'sop'", [], |r| r.get(0))
            .unwrap();
        assert_eq!(held, 1);
    }

    /// Re-reading a document must not leave its old passages behind.
    #[test]
    fn re_indexing_replaces_rather_than_accumulates() {
        let f = fixture();
        f.index
            .index_document("SOP", Classification::Internal,
                &[chunk("c1", "sop", 0, "Old text about valves.", vec![])])
            .unwrap();
        f.index
            .index_document("SOP", Classification::Internal,
                &[chunk("c2", "sop", 0, "New text about valves.", vec![])])
            .unwrap();

        let hits = f.index.search(&session(vec![Role::Employee]), "valves", 10).unwrap();
        assert_eq!(hits.len(), 1);
        assert!(hits[0].text.contains("New text"));
    }

    #[test]
    fn results_carry_everything_a_citation_needs() {
        let f = fixture();
        index_sop(&f);
        let hit = &f.index.search(&session(vec![Role::Employee]), "thickness", 10).unwrap()[0];

        assert_eq!(
            hit.citation(),
            "Maintenance SOP rev C — 4 Inspection › 4.2 Wall Thickness, page 1"
        );
        assert_eq!(hit.retrieval, Retrieval::Keyword);
        assert_eq!(hit.document_sha256, "sop");
    }

    #[test]
    fn the_limit_is_respected() {
        let f = fixture();
        let chunks: Vec<_> = (0..20)
            .map(|i| chunk(&format!("c{i}"), "sop", i, "valve inspection notes", vec![]))
            .collect();
        f.index.index_document("SOP", Classification::Internal, &chunks).unwrap();

        let hits = f.index.search(&session(vec![Role::Employee]), "valve", 5).unwrap();
        assert_eq!(hits.len(), 5);
    }

    // ── Queries a person actually types ─────────────────────────────────

    /// The query most likely to be typed in a refinery, and the one FTS5 would
    /// silently read as "PV, but not 2201".
    #[test]
    fn a_tag_number_with_a_hyphen_is_found() {
        let f = fixture();
        f.index
            .index_document(
                "SOP",
                Classification::Internal,
                &[chunk("c1", "sop", 0, "Vessel PV-2201 was inspected in March.", vec![])],
            )
            .unwrap();

        let hits = f.index.search(&session(vec![Role::Employee]), "PV-2201", 10).unwrap();
        assert_eq!(hits.len(), 1, "a hyphenated tag number must be findable");
    }

    /// FTS5 operators typed as ordinary words must not change the query's meaning.
    #[test]
    fn fts_operators_typed_by_a_person_are_treated_as_words() {
        let f = fixture();
        f.index
            .index_document(
                "SOP",
                Classification::Internal,
                &[chunk("c1", "sop", 0, "The valve OR the pump may be isolated.", vec![])],
            )
            .unwrap();

        // Reads as the literal words, not as a boolean operator.
        let hits = f.index.search(&session(vec![Role::Employee]), "valve OR pump", 10).unwrap();
        assert_eq!(hits.len(), 1);
    }

    /// An unbalanced quote is a syntax error in FTS5. It must not surface as one.
    #[test]
    fn punctuation_never_produces_a_query_error() {
        let f = fixture();
        index_sop(&f);

        for awkward in [
            "wall \"thickness",
            "thickness*",
            "9.0 mm -- minimum",
            "NEAR(wall thickness)",
            "^thickness",
            "(unbalanced",
        ] {
            let outcome = f.index.search(&session(vec![Role::Employee]), awkward, 10);
            assert!(outcome.is_ok(), "{awkward:?} should not error: {outcome:?}");
        }
    }

    #[test]
    fn a_query_with_nothing_searchable_matches_nothing_quietly() {
        let f = fixture();
        index_sop(&f);

        for empty in ["", "   ", "!!!", "--"] {
            let hits = f.index.search(&session(vec![Role::Employee]), empty, 10).unwrap();
            assert!(hits.is_empty(), "{empty:?} should match nothing");
        }
    }

    #[test]
    fn a_measurement_with_a_decimal_point_is_found() {
        let f = fixture();
        index_sop(&f);
        let hits = f.index.search(&session(vec![Role::Employee]), "9.0 mm", 10).unwrap();
        assert_eq!(hits.len(), 1);
    }

    #[test]
    fn indexing_an_empty_document_is_harmless() {
        let f = fixture();
        assert_eq!(
            f.index.index_document("Empty", Classification::Internal, &[]).unwrap(),
            0
        );
    }
}
