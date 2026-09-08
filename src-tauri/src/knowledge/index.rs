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
//! ## Keyword search, for now
//!
//! Retrieval here is full-text search over the chunks. The plan calls for a
//! hybrid of keyword and vector search with a reranker over the merged set, and
//! the two model-backed halves are not built because no embedding model is
//! installed. What exists is real and useful on its own — exact terms like
//! `PV-2201` or `9.0 mm` are precisely where keyword search beats embeddings —
//! and [`SearchResult::retrieval`] says which method found each passage, so a
//! deployment on keyword-only never looks like one running the full pipeline.

use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use rusqlite::{params, Connection};
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

            -- One passage's embedding, under the model that produced it.
            --
            -- `model_id` and `dimensions` are stored beside the vector rather
            -- than assumed, because the failure they prevent is silent: swap
            -- the embedding model and every stored vector becomes a point in a
            -- different space, where cosine similarity still returns a
            -- plausible number and the ranking is nonsense. A search asks for
            -- one model's vectors by name, so vectors from another are not
            -- found rather than quietly compared.
            CREATE TABLE IF NOT EXISTS chunk_vectors (
                id         TEXT PRIMARY KEY,
                model_id   TEXT NOT NULL,
                dimensions INTEGER NOT NULL,
                vector     BLOB NOT NULL
            );

            CREATE INDEX IF NOT EXISTS chunk_vectors_model_idx
                ON chunk_vectors(model_id);",
        )?;
        Ok(())
    }

    /// How many distinct documents are currently retrievable.
    ///
    /// Superseded documents are excluded: they remain traceable, but they are
    /// not current guidance, so counting them would overstate what an answer
    /// can actually be grounded in.
    pub fn document_count(&self) -> Result<usize> {
        let conn = self.conn.lock().expect("the index lock is never poisoned");
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
        let cleared: Vec<String> = Classification::ALL
            .iter()
            .filter(|c| {
                c.cleared_roles()
                    .iter()
                    .any(|role| session.user.roles.contains(role))
            })
            .filter_map(|c| serde_json::to_string(c).ok())
            .collect();
        if cleared.is_empty() {
            return Ok(Vec::new());
        }

        let placeholders = cleared.iter().map(|_| "?").collect::<Vec<_>>().join(",");
        let sql = format!(
            "SELECT c.document_sha256, c.document_name, c.classification,
                    COUNT(*) AS chunks, MAX(c.page) AS pages
             FROM chunks c
             WHERE c.superseded = 0
               AND c.classification IN ({placeholders})
             GROUP BY c.document_sha256, c.document_name, c.classification
             ORDER BY c.document_name"
        );

        let conn = self.conn.lock().expect("the index lock is never poisoned");
        let mut stmt = conn.prepare(&sql)?;
        let bound: Vec<&dyn rusqlite::ToSql> =
            cleared.iter().map(|v| v as &dyn rusqlite::ToSql).collect();
        let rows = stmt.query_map(bound.as_slice(), |row| {
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
        let Some(first) = chunks.first() else {
            return Ok(0);
        };
        let sha = first.document_sha256.clone();

        let mut conn = self.conn.lock().expect("index lock poisoned");
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
                tx.execute("DELETE FROM chunk_vectors WHERE id = ?1", [&id])?;
            }
        }
        tx.execute("DELETE FROM chunks WHERE document_sha256 = ?1", [&sha])?;

        for chunk in chunks {
            tx.execute(
                "INSERT INTO chunks
                    (id, document_sha256, document_name, ordinal, page, section_path,
                     kind, classification, superseded)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, 0)",
                params![
                    chunk.id,
                    chunk.document_sha256,
                    document_name,
                    chunk.ordinal,
                    chunk.page,
                    serde_json::to_string(&chunk.section_path)?,
                    serde_json::to_string(&chunk.kind)?,
                    serde_json::to_string(&classification)?,
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

    /// Stores one embedding per passage, replacing any the same model held.
    ///
    /// Separate from [`Self::index_document`] because embedding costs a model
    /// call per passage and indexing must not wait on a model being up. A
    /// document is retrievable by keyword the moment it is indexed; its vectors
    /// arrive when the embedding pass gets to it, and until then it is found by
    /// the half of the search that does not need them.
    pub fn store_vectors(&self, model_id: &str, vectors: &[(String, Vec<f32>)]) -> Result<usize> {
        if vectors.is_empty() {
            return Ok(0);
        }
        let mut conn = self.conn.lock().expect("index lock poisoned");
        let tx = conn.transaction()?;
        let mut written = 0usize;
        for (chunk_id, vector) in vectors {
            if vector.is_empty() {
                continue;
            }
            tx.execute(
                "INSERT OR REPLACE INTO chunk_vectors (id, model_id, dimensions, vector)
                 VALUES (?1, ?2, ?3, ?4)",
                params![chunk_id, model_id, vector.len() as i64, vector_to_bytes(vector)],
            )?;
            written += 1;
        }
        tx.commit()?;
        Ok(written)
    }

    /// Passages this model has not embedded yet, in reading order.
    ///
    /// The work list for the embedding pass, and what makes the pass resumable:
    /// it asks again after every batch, so a run interrupted halfway resumes
    /// where it stopped rather than starting over.
    pub fn chunks_needing_vectors(
        &self,
        model_id: &str,
        limit: usize,
    ) -> Result<Vec<(String, String)>> {
        let conn = self.conn.lock().expect("index lock poisoned");
        let mut stmt = conn.prepare(
            "SELECT c.id, t.body
             FROM chunks c
             JOIN chunk_text t ON t.id = c.id
             LEFT JOIN chunk_vectors v ON v.id = c.id AND v.model_id = ?1
             WHERE v.id IS NULL AND c.superseded = 0
             ORDER BY c.document_sha256, c.ordinal
             LIMIT ?2",
        )?;
        let rows = stmt.query_map(params![model_id, limit as i64], |row| {
            Ok((row.get(0)?, row.get(1)?))
        })?;
        Ok(rows.filter_map(Result::ok).collect())
    }

    /// How much of the index this model has embedded, as (embedded, total).
    ///
    /// Reported rather than assumed so a screen can say "vector search covers
    /// 40 of 900 passages" instead of implying the whole index is searchable
    /// that way. A half-embedded index that presents itself as whole is exactly
    /// the silent degradation this product refuses elsewhere.
    pub fn vector_coverage(&self, model_id: &str) -> Result<(usize, usize)> {
        let conn = self.conn.lock().expect("index lock poisoned");
        let total: i64 =
            conn.query_row("SELECT COUNT(*) FROM chunks WHERE superseded = 0", [], |row| {
                row.get(0)
            })?;
        let embedded: i64 = conn.query_row(
            "SELECT COUNT(*) FROM chunk_vectors v
             JOIN chunks c ON c.id = v.id
             WHERE v.model_id = ?1 AND c.superseded = 0",
            [model_id],
            |row| row.get(0),
        )?;
        Ok((embedded as usize, total as usize))
    }

    /// Searches by vector similarity, returning only what this person may see.
    ///
    /// ## The clearance rule is the same one, in the same place
    ///
    /// ARJUN design rule 22 says the gateway filters by permission *before* the
    /// passages reach the model, and [`Self::search`] honours that by binding
    /// clearance into the SQL. So does this. A passage the asker cannot see is
    /// never fetched, so its vector is never compared and it cannot influence
    /// the ranking of anything else — which a filter applied after scoring
    /// could not promise.
    ///
    /// ## Why the arithmetic is in Rust and not in SQL
    ///
    /// SQLite has no vector type and no cosine function without an extension,
    /// and an extension is a native dependency an air-gapped installer would
    /// have to carry. The rows are reduced to what this person may read before
    /// any of them are scored, so the work is bounded by the reader's own
    /// library rather than by the whole index.
    ///
    /// `score` is `1 - cosine similarity`, a distance, so lower stays better
    /// here exactly as it is for bm25 — letting the two be fused by rank
    /// without either pretending its numbers mean the same thing.
    pub fn search_vectors(
        &self,
        session: &Session,
        model_id: &str,
        query: &[f32],
        limit: usize,
    ) -> Result<Vec<SearchResult>> {
        if query.is_empty() {
            return Ok(Vec::new());
        }

        let cleared: Vec<String> = Classification::ALL
            .iter()
            .filter(|c| {
                c.cleared_roles()
                    .iter()
                    .any(|role| session.user.roles.contains(role))
            })
            .filter_map(|c| serde_json::to_string(c).ok())
            .collect();

        if cleared.is_empty() {
            // Cleared for nothing, and the same answer keyword search gives:
            // empty, and indistinguishable from nothing having matched.
            return Ok(Vec::new());
        }

        let placeholders = cleared.iter().map(|_| "?").collect::<Vec<_>>().join(",");
        let sql = format!(
            "SELECT c.id, c.document_sha256, c.document_name, t.body, c.page,
                    c.section_path, c.classification, v.vector, v.dimensions
             FROM chunk_vectors v
             JOIN chunks c ON c.id = v.id
             JOIN chunk_text t ON t.id = v.id
             WHERE v.model_id = ?1
               AND c.superseded = 0
               AND c.classification IN ({placeholders})"
        );

        let conn = self.conn.lock().expect("index lock poisoned");
        let mut stmt = conn.prepare(&sql)?;

        let mut bound: Vec<&dyn rusqlite::ToSql> = vec![&model_id];
        for value in &cleared {
            bound.push(value);
        }

        let rows = stmt.query_map(bound.as_slice(), |row| {
            let section_path: String = row.get(5)?;
            let classification: String = row.get(6)?;
            let bytes: Vec<u8> = row.get(7)?;
            let dimensions: i64 = row.get(8)?;
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
                bytes,
                dimensions as usize,
            ))
        })?;

        let mut scored: Vec<SearchResult> = Vec::new();
        for (mut result, bytes, dimensions) in rows.filter_map(Result::ok) {
            // A stored vector of a different width came from a different model
            // than the one asked for, or from a corrupted write. Either way it
            // cannot be compared, and skipping it is the only honest choice —
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
            result.score = 1.0 - similarity;
            scored.push(result);
        }

        // Nearest first. `total_cmp` rather than `partial_cmp` so the
        // comparator is total: a NaN that somehow survived the guards above
        // sorts to one end deterministically instead of making the order
        // arbitrary, and a citation that moves between identical runs is a bug
        // report waiting to happen.
        scored.sort_by(|a, b| a.score.total_cmp(&b.score));
        scored.truncate(limit);
        Ok(scored)
    }

    /// Marks a document superseded.
    ///
    /// Its passages stay in the index and remain traceable — a citation made
    /// last year must still resolve — but they are not returned as current
    /// guidance, which is what ARJUN design rule 11 asks for.
    pub fn supersede(&self, document_sha256: &str) -> Result<()> {
        let conn = self.conn.lock().expect("index lock poisoned");
        conn.execute(
            "UPDATE chunks SET superseded = 1 WHERE document_sha256 = ?1",
            [document_sha256],
        )?;
        Ok(())
    }

    /// Searches, returning only what this person is cleared to see.
    ///
    /// The clearance test is a bound parameter in the query. It is not a filter
    /// over results, because by then the passages exist and their number is
    /// already informative.
    pub fn search(&self, session: &Session, query: &str, limit: usize) -> Result<Vec<SearchResult>> {
        // A query with nothing searchable in it matches nothing, rather than
        // reaching FTS5 as a syntax error the caller would have to interpret.
        let Some(expression) = to_match_expression(query) else {
            return Ok(Vec::new());
        };

        let cleared: Vec<String> = Classification::ALL
            .iter()
            .filter(|c| {
                c.cleared_roles()
                    .iter()
                    .any(|role| session.user.roles.contains(role))
            })
            .filter_map(|c| serde_json::to_string(c).ok())
            .collect();

        if cleared.is_empty() {
            // Cleared for nothing. An empty result is the correct and complete
            // answer — and identical to what they would see if nothing matched,
            // which is the point.
            return Ok(Vec::new());
        }

        let placeholders = cleared.iter().map(|_| "?").collect::<Vec<_>>().join(",");
        let sql = format!(
            "SELECT c.id, c.document_sha256, c.document_name, t.body, c.page,
                    c.section_path, c.classification, bm25(chunk_text) AS score
             FROM chunk_text t
             JOIN chunks c ON c.id = t.id
             WHERE chunk_text MATCH ?1
               AND c.superseded = 0
               AND c.classification IN ({placeholders})
             ORDER BY score
             LIMIT ?{}",
            cleared.len() + 2
        );

        let conn = self.conn.lock().expect("index lock poisoned");
        let mut stmt = conn.prepare(&sql)?;

        let mut bound: Vec<&dyn rusqlite::ToSql> = vec![&expression];
        for value in &cleared {
            bound.push(value);
        }
        let limit = limit as i64;
        bound.push(&limit);

        let rows = stmt.query_map(bound.as_slice(), |row| {
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
                score: row.get(7)?,
                retrieval: Retrieval::Keyword,
            })
        })?;

        Ok(rows.filter_map(Result::ok).collect())
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
    /// for, and superseded documents are excluded. That matters because this is
    /// a second door into the same shelf — a retrieval path that filtered less
    /// than search would be a way to read by page number what could not be read
    /// by searching for it.
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

        let cleared: Vec<String> = Classification::ALL
            .iter()
            .filter(|c| {
                c.cleared_roles()
                    .iter()
                    .any(|role| session.user.roles.contains(role))
            })
            .filter_map(|c| serde_json::to_string(c).ok())
            .collect();

        if cleared.is_empty() {
            return Ok(Vec::new());
        }

        let placeholders = cleared.iter().map(|_| "?").collect::<Vec<_>>().join(",");
        let sql = format!(
            "SELECT c.id, c.document_sha256, c.document_name, t.body, c.page,
                    c.section_path, c.classification
             FROM chunks c
             JOIN chunk_text t ON c.id = t.id
             WHERE c.document_sha256 = ?1
               AND c.page >= ?2
               AND c.page <= ?3
               AND c.superseded = 0
               AND c.classification IN ({placeholders})
             ORDER BY c.page, c.ordinal
             LIMIT ?{}",
            cleared.len() + 4
        );

        let conn = self.conn.lock().expect("index lock poisoned");
        let mut stmt = conn.prepare(&sql)?;

        let document = document_sha256.to_string();
        let from = from_page as i64;
        let to = to_page as i64;
        let limit = limit as i64;
        let mut bound: Vec<&dyn rusqlite::ToSql> = vec![&document, &from, &to];
        for value in &cleared {
            bound.push(value);
        }
        bound.push(&limit);

        let rows = stmt.query_map(bound.as_slice(), |row| {
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

    /// Indexes two passages and gives them vectors pointing in different
    /// directions, so "nearest" has an unambiguous right answer.
    fn index_two_with_vectors(f: &Fixture, model: &str) {
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
        f.index
            .store_vectors(
                model,
                &[
                    ("c-valve".to_string(), vec![1.0, 0.0, 0.0]),
                    ("c-pump".to_string(), vec![0.0, 1.0, 0.0]),
                ],
            )
            .unwrap();
    }

    #[test]
    fn the_nearest_vector_ranks_first_and_the_far_one_still_appears() {
        let f = fixture();
        index_two_with_vectors(&f, "bge-m3");

        // Almost exactly the valve vector, so the valve passage must lead.
        let hits = f
            .index
            .search_vectors(&session(vec![Role::Employee]), "bge-m3", &[0.9, 0.1, 0.0], 10)
            .unwrap();

        assert_eq!(hits.len(), 2, "both passages are cleared and embedded");
        assert_eq!(hits[0].chunk_id, "c-valve");
        assert_eq!(hits[1].chunk_id, "c-pump");
        assert!(
            hits[0].score < hits[1].score,
            "score is a distance, so the nearer passage must carry the smaller number"
        );
        assert!(hits.iter().all(|hit| hit.retrieval == Retrieval::Vector));
    }

    /// The same guarantee keyword search makes, made by the same means: the
    /// clearance is in the SQL, so an uncleared passage is never fetched and
    /// therefore never scored.
    #[test]
    fn an_uncleared_passage_is_never_compared_by_vector() {
        let f = fixture();
        f.index
            .index_document(
                "Vendor terms",
                Classification::VendorNegotiation,
                &[chunk("c1", "deal", 0, "Unit price is 4.2 lakh per valve.", vec![])],
            )
            .unwrap();
        f.index
            .store_vectors("bge-m3", &[("c1".to_string(), vec![1.0, 0.0, 0.0])])
            .unwrap();

        // An Auditor is cleared for nothing, so the row is not fetched at all.
        let hits = f
            .index
            .search_vectors(&session(vec![Role::Auditor]), "bge-m3", &[1.0, 0.0, 0.0], 10)
            .unwrap();
        assert!(
            hits.is_empty(),
            "a passage above the reader clearance was scored and returned"
        );
    }

    /// Swapping the embedding model must not silently re-rank the corpus.
    #[test]
    fn vectors_from_another_model_are_not_compared() {
        let f = fixture();
        index_two_with_vectors(&f, "bge-m3");

        let hits = f
            .index
            .search_vectors(&session(vec![Role::Employee]), "e5-large", &[1.0, 0.0, 0.0], 10)
            .unwrap();

        assert!(
            hits.is_empty(),
            "vectors produced by a different model were compared as though they shared a \
             space with this query"
        );
    }

    /// A query of a different width than the stored vectors is not comparable,
    /// and producing a number anyway would be the silent kind of wrong.
    #[test]
    fn a_query_of_the_wrong_width_matches_nothing_rather_than_truncating() {
        let f = fixture();
        index_two_with_vectors(&f, "bge-m3");

        let hits = f
            .index
            .search_vectors(&session(vec![Role::Employee]), "bge-m3", &[1.0, 0.0], 10)
            .unwrap();

        assert!(hits.is_empty());
    }

    /// A vector outliving the text it was made from would rank a passage by
    /// what it used to say.
    #[test]
    fn re_reading_a_document_drops_the_vectors_of_the_text_it_replaced() {
        let f = fixture();
        index_two_with_vectors(&f, "bge-m3");
        assert_eq!(f.index.vector_coverage("bge-m3").unwrap(), (2, 2));

        // The same document, read again by a better engine into one passage.
        f.index
            .index_document(
                "Pump Manual",
                Classification::Internal,
                &[chunk("c-merged", "man", 0, "Valve replaced; pump overhauled.", vec![])],
            )
            .unwrap();

        let (embedded, total) = f.index.vector_coverage("bge-m3").unwrap();
        assert_eq!(total, 1, "the re-read replaced both passages with one");
        assert_eq!(embedded, 0, "the old vectors did not survive their text");
    }

    /// The work list is what makes the embedding pass resumable.
    #[test]
    fn the_work_list_shrinks_as_passages_are_embedded() {
        let f = fixture();
        f.index
            .index_document(
                "Pump Manual",
                Classification::Internal,
                &[
                    chunk("c1", "man", 0, "The control valve was replaced.", vec![]),
                    chunk("c2", "man", 1, "The charge pump was overhauled.", vec![]),
                ],
            )
            .unwrap();

        let outstanding = f.index.chunks_needing_vectors("bge-m3", 10).unwrap();
        assert_eq!(outstanding.len(), 2);
        assert!(
            outstanding.iter().any(|(id, body)| id == "c1" && body.contains("valve")),
            "the work list carries the text to embed, not just the id"
        );

        f.index
            .store_vectors("bge-m3", &[("c1".to_string(), vec![1.0, 0.0, 0.0])])
            .unwrap();

        let outstanding = f.index.chunks_needing_vectors("bge-m3", 10).unwrap();
        assert_eq!(outstanding.len(), 1);
        assert_eq!(outstanding[0].0, "c2");
        assert_eq!(f.index.vector_coverage("bge-m3").unwrap(), (1, 2));
    }

    /// Half an index embedded must not look like a whole one.
    #[test]
    fn coverage_counts_only_this_models_vectors_on_live_passages() {
        let f = fixture();
        index_two_with_vectors(&f, "bge-m3");

        assert_eq!(f.index.vector_coverage("bge-m3").unwrap(), (2, 2));
        assert_eq!(
            f.index.vector_coverage("e5-large").unwrap(),
            (0, 2),
            "a model that has embedded nothing covers nothing, and the total is still the truth"
        );

        f.index.supersede("man").unwrap();
        assert_eq!(
            f.index.vector_coverage("bge-m3").unwrap(),
            (0, 0),
            "superseded passages are not current guidance and are not counted"
        );
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
