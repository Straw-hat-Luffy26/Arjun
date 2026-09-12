//! Storing a notebook's graph, and reading parts of it back.
//!
//! ## Rows are per document, totals are computed on read
//!
//! The obvious schema keeps one row per node per notebook and adds to its count
//! as each document is processed. That schema cannot be rebuilt: running a
//! document through the extractor twice doubles every number it contributed, and
//! nothing in the row records which document put it there, so there is no way to
//! subtract it again.
//!
//! So a row here belongs to `(notebook, document, term)`. Re-processing one
//! document deletes its rows and writes them again — idempotent by construction,
//! whatever went wrong last time — and the notebook's graph is the `GROUP BY`
//! over those rows. Aggregating on read costs a little; being unable to trust a
//! rebuilt count costs much more.
//!
//! ## Every row carries where it came from
//!
//! `graph_evidence` holds the chunks behind each node and each edge. Nothing is
//! written here without it, because the question a person asks of a graph like
//! this is "why do you think those two are connected?", and a graph that cannot
//! answer it is decoration.

use std::collections::{BTreeMap, BTreeSet};

use anyhow::{Context, Result};
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::assertions::{AssertionProvenance, AssertionStatus, EdgeAssertion};
use super::statistical::GraphDraft;
use super::store::NotebookStore;

/// What a node stands for.
///
/// The graph holds two populations. Terms come out of the extractor and are what
/// the passes reason about; documents are the notebook's own files, drawn so a
/// reader can see which file a term came from and which terms two files share.
/// A file is not an extracted claim about anything, so it is a separate kind
/// rather than a term that happens to be named like a file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum NodeKind {
    Term,
    Document,
}

/// A node as the interface sees it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GraphNode {
    /// Stable across rebuilds: derived from the notebook and the term, or from
    /// the notebook and the document's content address.
    pub id: String,
    pub label: String,
    pub kind: NodeKind,
    /// `None` until the typing pass has run, and always `None` for a document.
    /// A real, displayable state — the interface says "not yet typed" rather
    /// than guessing a type.
    pub node_type: Option<String>,
    /// For a term: passages it appears in, summed across the notebook's
    /// documents. For a document: passages of it that contributed a term.
    pub occurrences: u32,
    /// Links to this node within the returned view.
    pub degree: u32,
    /// The content address a document node stands for. `None` for a term.
    ///
    /// Carried so the interface can put a file's name against an evidence row,
    /// which cites a chunk by `document_sha256` and would otherwise be shown as
    /// a bare page number belonging to no visible file.
    pub document_sha256: Option<String>,
    /// For a term: how many of the notebook's documents it was found in. Zero
    /// for a document node.
    pub document_count: u32,
    /// Every type proposed for this term, with how many documents proposed it.
    ///
    /// Usually one entry, or none. More than one means the typing pass reached
    /// different conclusions in different documents, and that disagreement is
    /// carried rather than resolved: `node_type` names the best-supported one
    /// so the canvas has something to colour by, and this says what the other
    /// answers were so the inspector can show them.
    ///
    /// The view used to read `MAX(node_type)`, which picked whichever string
    /// sorted highest and left no trace that anything had been discarded.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub type_candidates: Vec<TypeCandidate>,
}

/// One type proposed for a term, and how much of the notebook proposed it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TypeCandidate {
    pub node_type: String,
    /// Documents whose typing pass proposed this type.
    pub documents: u32,
}

/// What a link between two nodes means.
///
/// Named rather than implied. The statistical pass can only observe that two
/// terms were in the same passage, and an interface that drew that as an
/// unlabelled line invites the reader to supply a relation it never claimed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum EdgeKind {
    /// Two terms sharing passages. Weight is the passage count.
    Cooccurrence,
    /// A document and a term found in it, directed from the file to the term.
    /// Weight is the passage count within that document.
    ///
    /// Directed because it genuinely is: a file yields terms, and the reverse
    /// is not a separate fact. That is also what earns it an arrowhead on the
    /// canvas, where [`Self::Cooccurrence`] gets a plain line — two terms
    /// sharing a passage have no direction, and drawing one would assert an
    /// order of events nothing observed.
    ///
    /// A fact of the corpus rather than an inference, and the only edge in the
    /// graph that never needs typing.
    AppearsIn,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GraphEdge {
    pub source: String,
    pub target: String,
    pub kind: EdgeKind,
    /// Passages containing both, summed across documents. A count, not a score.
    pub weight: u32,
    /// The label to draw on this link, when one claim can stand for it.
    ///
    /// Derived from [`Self::assertions`] rather than stored: it is the accepted
    /// claim if there is one, otherwise the only proposed claim, and `None`
    /// when there are none or when two disagree. A caller that needs to know
    /// *which way round* the claim runs must read `assertions` — this field
    /// cannot express direction and never could.
    pub relation: Option<String>,
    /// Every directed claim about this pair of terms, in the subject-to-object
    /// order the extractor produced.
    ///
    /// Empty for a link the passes have only observed and never named, and
    /// always empty for an `AppearsIn`, whose meaning is fixed.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub assertions: Vec<EdgeAssertion>,
    /// True when two or more unrejected claims about this pair say different
    /// things. Surfaced rather than resolved — see [`super::assertions`].
    #[serde(default)]
    pub contested: bool,
}

/// One view of a notebook's graph, and what it is not showing.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GraphView {
    pub nodes: Vec<GraphNode>,
    pub edges: Vec<GraphEdge>,
    /// Nodes in the notebook's whole graph, before this view's filters.
    ///
    /// The interface needs both numbers to say "showing 40 of 912" honestly.
    /// A view that silently dropped the rest would look like a small graph
    /// rather than a filtered one — which is how a person concludes their
    /// documents contained nothing.
    pub total_nodes: u32,
    pub total_edges: u32,
    /// The two populations behind `total_nodes`, so the interface can say
    /// "7 terms · 11 files" rather than a single number that hides which is
    /// which — and, in particular, can show that a notebook of eleven files
    /// yielded seven terms.
    pub total_terms: u32,
    pub total_documents: u32,
}

/// One passage behind a node or an edge.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EvidenceRow {
    pub chunk_id: String,
    pub document_sha256: String,
    pub page: u32,
    /// The verbatim sentence, once the typing pass has supplied one. The
    /// statistical pass cites a passage without quoting from it.
    pub quote: Option<String>,
}

/// The stable id of a term within a notebook.
///
/// A hash rather than the term itself so an id is safe in a URL, a DOM
/// attribute and a JSON key, and fixed-length regardless of the phrase.
pub fn node_id(notebook_id: &str, normalised: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(notebook_id.as_bytes());
    hasher.update(b":");
    hasher.update(normalised.as_bytes());
    format!("{:x}", hasher.finalize())[..32].to_string()
}

/// The stable id of a document node within a notebook.
///
/// Prefixed, so it is 33 characters where a term id is 32 and the two can never
/// collide however a term happens to be spelled. That matters because a node id
/// is also the key `node_evidence` is asked for: a document id reaching that
/// lookup must miss rather than quietly return some term's passages.
pub fn document_node_id(notebook_id: &str, document_sha256: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(notebook_id.as_bytes());
    hasher.update(b":document:");
    hasher.update(document_sha256.as_bytes());
    format!("d{:x}", hasher.finalize())[..33].to_string()
}

impl NotebookStore {
    pub(super) fn prepare_graph(conn: &Connection) -> Result<()> {
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS graph_nodes (
                notebook_id     TEXT NOT NULL,
                document_sha256 TEXT NOT NULL,
                normalised      TEXT NOT NULL,
                label           TEXT NOT NULL,
                node_type       TEXT,
                type_confidence REAL,
                extractor       TEXT NOT NULL,
                occurrences     INTEGER NOT NULL,
                PRIMARY KEY (notebook_id, document_sha256, normalised)
            );
            CREATE INDEX IF NOT EXISTS graph_nodes_notebook_idx
                ON graph_nodes(notebook_id);

            CREATE TABLE IF NOT EXISTS graph_edges (
                notebook_id     TEXT NOT NULL,
                document_sha256 TEXT NOT NULL,
                source          TEXT NOT NULL,
                target          TEXT NOT NULL,
                relation        TEXT,
                weight          INTEGER NOT NULL,
                extractor       TEXT NOT NULL,
                PRIMARY KEY (notebook_id, document_sha256, source, target)
            );
            CREATE INDEX IF NOT EXISTS graph_edges_notebook_idx
                ON graph_edges(notebook_id);

            CREATE TABLE IF NOT EXISTS graph_evidence (
                notebook_id     TEXT NOT NULL,
                subject         TEXT NOT NULL,
                subject_kind    TEXT NOT NULL,
                chunk_id        TEXT NOT NULL,
                document_sha256 TEXT NOT NULL,
                page            INTEGER NOT NULL DEFAULT 0,
                quote           TEXT,
                PRIMARY KEY (notebook_id, subject, subject_kind, chunk_id)
            );
            CREATE INDEX IF NOT EXISTS graph_evidence_subject_idx
                ON graph_evidence(notebook_id, subject);

            CREATE TABLE IF NOT EXISTS graph_build_units (
                notebook_id       TEXT NOT NULL,
                document_sha256   TEXT NOT NULL,
                pass              TEXT NOT NULL,
                extractor_version INTEGER NOT NULL,
                completed_at      TEXT NOT NULL,
                source_revision   TEXT,
                extractor_id      TEXT,
                PRIMARY KEY (notebook_id, document_sha256, pass, extractor_version)
            );

            CREATE TABLE IF NOT EXISTS graph_schema_meta (
                key        TEXT PRIMARY KEY,
                value      TEXT NOT NULL,
                applied_at TEXT NOT NULL
            );",
        )?;
        Ok(())
    }

    /// Brings a database written by an earlier build up to this one.
    ///
    /// Two things are repaired, both of them data the old schema could not
    /// express. The second half runs once, recorded in `graph_schema_meta`,
    /// because it moves rows and running it twice would move them again.
    pub(super) fn migrate_graph(conn: &Connection) -> Result<()> {
        // ── 1. Build units learn what they were built from ────────────────
        //
        // `ALTER TABLE ADD COLUMN` has no `IF NOT EXISTS` in SQLite, so the
        // columns are read first. A database created by `prepare_graph` above
        // already has them and this does nothing.
        let existing: BTreeSet<String> = {
            let mut statement = conn.prepare("PRAGMA table_info(graph_build_units)")?;
            let rows = statement.query_map([], |row| row.get::<_, String>(1))?;
            rows.collect::<rusqlite::Result<BTreeSet<_>>>()?
        };
        if !existing.contains("source_revision") {
            conn.execute_batch("ALTER TABLE graph_build_units ADD COLUMN source_revision TEXT;")?;
        }
        if !existing.contains("extractor_id") {
            conn.execute_batch("ALTER TABLE graph_build_units ADD COLUMN extractor_id TEXT;")?;
        }

        // ── 2. Relation labels become directed claims ─────────────────────
        //
        // `graph_edges.relation` held a label on a pair whose ends had been
        // sorted alphabetically, so which term was the subject was destroyed at
        // write time and cannot be recovered from the row. Each surviving label
        // therefore becomes an assertion marked `direction_certain = 0` and left
        // `proposed` — the honest reading of "a model said these two are related
        // this way, and we no longer know which way round".
        //
        // Guessing a direction here would be worse than the bug being fixed: it
        // would turn an unknown into a confident, wrong, citable claim.
        let already: Option<String> = conn
            .query_row(
                "SELECT value FROM graph_schema_meta WHERE key = 'relations_to_assertions'",
                [],
                |row| row.get(0),
            )
            .optional()?;
        if already.is_some() {
            return Ok(());
        }

        let legacy: Vec<(String, String, String, String, String, String)> = {
            let mut statement = conn.prepare(
                "SELECT notebook_id, document_sha256, source, target, relation, extractor
                   FROM graph_edges
                  WHERE relation IS NOT NULL AND TRIM(relation) <> ''",
            )?;
            let rows = statement.query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, String>(4)?,
                    row.get::<_, String>(5)?,
                ))
            })?;
            rows.collect::<rusqlite::Result<Vec<_>>>()?
        };

        for (notebook_id, document_sha256, source, target, relation, extractor) in &legacy {
            let subject_label =
                label_in(conn, notebook_id, source)?.unwrap_or_else(|| source.clone());
            let object_label =
                label_in(conn, notebook_id, target)?.unwrap_or_else(|| target.clone());

            // The passages the old schema kept against the edge become the
            // claim's evidence, so a migrated row is still answerable.
            let evidence = {
                let subject_key = format!(
                    "{}|{}",
                    node_id(notebook_id, source),
                    node_id(notebook_id, target)
                );
                let mut statement = conn.prepare(
                    "SELECT chunk_id, document_sha256, page, quote
                       FROM graph_evidence
                      WHERE notebook_id = ?1 AND subject = ?2 AND subject_kind = 'edge'",
                )?;
                let rows = statement.query_map(params![notebook_id, &subject_key], |row| {
                    Ok(super::assertions::AssertionEvidence {
                        chunk_id: row.get(0)?,
                        document_sha256: row.get(1)?,
                        page: row.get::<_, i64>(2)? as u32,
                        quote: row.get(3)?,
                    })
                })?;
                rows.collect::<rusqlite::Result<Vec<_>>>()?
            };

            super::assertions::write_assertion(
                conn,
                notebook_id,
                &super::assertions::NewAssertion {
                    document_sha256: Some(document_sha256.clone()),
                    subject: source.clone(),
                    subject_label,
                    predicate: relation.clone(),
                    object: target.clone(),
                    object_label,
                    provenance: super::assertions::AssertionProvenance::Model,
                    status: super::assertions::AssertionStatus::Proposed,
                    // The whole point of the migration.
                    direction_certain: false,
                    extractor: extractor.clone(),
                    extractor_version: 0,
                    source_revision: None,
                    evidence,
                },
            )?;
        }

        // The label is now held in one place. Leaving a copy on the edge would
        // give the view two answers to the same question, and the view would go
        // on reading the one that cannot express direction.
        conn.execute("UPDATE graph_edges SET relation = NULL", [])?;
        conn.execute(
            "INSERT OR REPLACE INTO graph_schema_meta (key, value, applied_at)
             VALUES ('relations_to_assertions', ?1, ?2)",
            params![legacy.len() as i64, chrono::Utc::now().to_rfc3339()],
        )?;
        Ok(())
    }

    /// Writes one document's contribution, replacing whatever it wrote before.
    ///
    /// The delete-then-insert is the whole reason this is safe to re-run: a
    /// build interrupted halfway leaves rows behind, and the next attempt must
    /// not add to them.
    ///
    /// ## Why it clears the enrichment passes too
    ///
    /// This pass writes `node_type = NULL` and no relation, because that is all
    /// it knows. Re-running it over a document therefore *erased* whatever the
    /// typing and relation passes had added — and it left their rows in
    /// `graph_build_units` untouched, so both passes went on skipping the
    /// document as already done. One rebuild removed the enrichment and made it
    /// unreachable, permanently, with the screen reporting the notebook fully
    /// processed.
    ///
    /// So the completion records for the passes downstream of this one go in
    /// the same transaction as the rows they described. Either the enrichment
    /// and its "this is done" marker both survive, or neither does.
    ///
    /// Reviewed claims are not deleted. They are kept and marked stale if their
    /// terms no longer exist, because a person's judgement is not the
    /// extractor's to discard — see
    /// [`purge_document_contributions`] and [`refresh_assertion_staleness`].
    ///
    /// `source_revision` identifies the extraction this was built from. A
    /// document re-read at a better OCR stop gets a new revision, which is what
    /// makes [`Self::completed_units`] treat the old work as out of date rather
    /// than as done.
    pub fn store_document_graph(
        &self,
        notebook_id: &str,
        owner_user_id: &str,
        document_sha256: &str,
        page_of_chunk: &dyn Fn(&str) -> u32,
        draft: &GraphDraft,
        extractor_version: u32,
        source_revision: &str,
    ) -> Result<()> {
        // The owner check is here rather than in the caller, so there is no way
        // to reach the write without it.
        if self.get(notebook_id, owner_user_id)?.is_none() {
            anyhow::bail!("that notebook does not exist");
        }

        let mut conn = self.conn.lock().expect("notebook store lock poisoned");
        let tx = conn.transaction()?;

        tx.execute(
            "DELETE FROM graph_nodes WHERE notebook_id = ?1 AND document_sha256 = ?2",
            params![notebook_id, document_sha256],
        )?;
        tx.execute(
            "DELETE FROM graph_edges WHERE notebook_id = ?1 AND document_sha256 = ?2",
            params![notebook_id, document_sha256],
        )?;
        tx.execute(
            "DELETE FROM graph_evidence WHERE notebook_id = ?1 AND document_sha256 = ?2",
            params![notebook_id, document_sha256],
        )?;
        // The enrichment this rebuild is about to invalidate, and the markers
        // that would otherwise stop it being redone.
        tx.execute(
            "DELETE FROM graph_build_units
              WHERE notebook_id = ?1 AND document_sha256 = ?2 AND pass <> 'statistical'",
            params![notebook_id, document_sha256],
        )?;
        tx.execute(
            "DELETE FROM graph_assertion_evidence
              WHERE assertion_id IN (
                  SELECT id FROM graph_assertions
                   WHERE notebook_id = ?1 AND document_sha256 = ?2 AND provenance <> 'user')",
            params![notebook_id, document_sha256],
        )?;
        tx.execute(
            "DELETE FROM graph_assertions
              WHERE notebook_id = ?1 AND document_sha256 = ?2 AND provenance <> 'user'",
            params![notebook_id, document_sha256],
        )?;

        for node in &draft.nodes {
            tx.execute(
                "INSERT INTO graph_nodes
                     (notebook_id, document_sha256, normalised, label, node_type,
                      type_confidence, extractor, occurrences)
                 VALUES (?1, ?2, ?3, ?4, NULL, NULL, 'statistical', ?5)",
                params![
                    notebook_id,
                    document_sha256,
                    &node.normalised,
                    &node.label,
                    node.occurrences
                ],
            )?;
            let subject = node_id(notebook_id, &node.normalised);
            for chunk_id in &node.chunk_ids {
                tx.execute(
                    "INSERT OR REPLACE INTO graph_evidence
                         (notebook_id, subject, subject_kind, chunk_id, document_sha256, page, quote)
                     VALUES (?1, ?2, 'node', ?3, ?4, ?5, NULL)",
                    params![
                        notebook_id,
                        &subject,
                        chunk_id,
                        document_sha256,
                        page_of_chunk(chunk_id)
                    ],
                )?;
            }
        }

        for edge in &draft.edges {
            tx.execute(
                "INSERT INTO graph_edges
                     (notebook_id, document_sha256, source, target, relation, weight, extractor)
                 VALUES (?1, ?2, ?3, ?4, NULL, ?5, 'statistical')",
                params![
                    notebook_id,
                    document_sha256,
                    &edge.source,
                    &edge.target,
                    edge.weight
                ],
            )?;
            let subject = format!(
                "{}|{}",
                node_id(notebook_id, &edge.source),
                node_id(notebook_id, &edge.target)
            );
            for chunk_id in &edge.chunk_ids {
                tx.execute(
                    "INSERT OR REPLACE INTO graph_evidence
                         (notebook_id, subject, subject_kind, chunk_id, document_sha256, page, quote)
                     VALUES (?1, ?2, 'edge', ?3, ?4, ?5, NULL)",
                    params![
                        notebook_id,
                        &subject,
                        chunk_id,
                        document_sha256,
                        page_of_chunk(chunk_id)
                    ],
                )?;
            }
        }

        tx.execute(
            "INSERT OR REPLACE INTO graph_build_units
                 (notebook_id, document_sha256, pass, extractor_version, completed_at,
                  source_revision, extractor_id)
             VALUES (?1, ?2, 'statistical', ?3, ?4, ?5, 'statistical')",
            params![
                notebook_id,
                document_sha256,
                extractor_version,
                chrono::Utc::now().to_rfc3339(),
                source_revision
            ],
        )?;

        // A person's claims about terms this rebuild may have removed.
        refresh_assertion_staleness(&tx, notebook_id)?;

        tx.commit()?;
        Ok(())
    }

    /// Documents already processed, at this pass's current identity.
    ///
    /// What makes a build resumable: the caller skips these rather than redoing
    /// work that is already on disk.
    ///
    /// ## What counts as "already processed"
    ///
    /// Three things have to match, not one. The pass at this `extractor_version`
    /// — which is how a change to the algorithm forces a rebuild. The
    /// `extractor_id`, which names the model or code that did it — so a
    /// notebook typed by a 3B model is not reported as done when the machine is
    /// now running a 13B one. And the `source_revision` the unit was built
    /// from, so a document re-read at a better OCR stop is work again rather
    /// than being skipped on the strength of a graph built from worse text.
    ///
    /// A document whose revision is not in `revisions` is treated as not done:
    /// the caller could not tell us what is on disk for it, and claiming
    /// completion on a source we cannot identify is the failure this whole
    /// signature exists to prevent.
    pub fn completed_units(
        &self,
        notebook_id: &str,
        pass: &str,
        extractor_version: u32,
        extractor_id: &str,
        revisions: &BTreeMap<String, String>,
    ) -> Result<BTreeSet<String>> {
        let conn = self.conn.lock().expect("notebook store lock poisoned");
        let mut statement = conn.prepare(
            "SELECT document_sha256, source_revision, extractor_id FROM graph_build_units
              WHERE notebook_id = ?1 AND pass = ?2 AND extractor_version = ?3",
        )?;
        let rows = statement.query_map(params![notebook_id, pass, extractor_version], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, Option<String>>(1)?,
                row.get::<_, Option<String>>(2)?,
            ))
        })?;

        let mut done = BTreeSet::new();
        for row in rows {
            let (sha, recorded_revision, recorded_extractor) = row?;

            // A unit written before these columns existed carries neither. It
            // is honoured on the version alone, because the alternative is
            // silently re-running every pass over every existing notebook on
            // the first launch after this change — minutes per document, and a
            // REBEL sidecar the person never asked for.
            match recorded_extractor {
                None => {}
                Some(recorded) if recorded == extractor_id => {}
                Some(_) => continue,
            }
            match (recorded_revision, revisions.get(&sha)) {
                (None, _) => {}
                (Some(recorded), Some(current)) if &recorded == current => {}
                _ => continue,
            }
            done.insert(sha);
        }
        Ok(done)
    }

    /// A view of the notebook's graph.
    ///
    /// `focus` restricts it to one node and its neighbours out to `depth` - the
    /// local view, which is the one that stays useful. A global graph past a few
    /// hundred nodes is an unreadable hairball, so the caller is expected to
    /// pass a focus most of the time and a `min_weight` when it does not.
    ///
    /// ## Documents are in the graph
    ///
    /// Rows are stored per `(notebook, document, term)`, and the term view is a
    /// `GROUP BY` over them. Grouping is what makes a rebuilt count trustworthy,
    /// but on its own it throws away the one relation a reader of a *notebook*
    /// asks for first: which file did this come from, and what do these two
    /// files have in common. So `with_documents` puts the notebook's files back
    /// in as nodes of their own, joined to their terms by [`EdgeKind::AppearsIn`].
    ///
    /// A file that contributed no terms is still returned, unattached. That is a
    /// finding, not an omission - it says the extractor got nothing out of that
    /// document, which is invisible if the file simply never appears.
    /// `document_sha256` narrows every count and every row to one file.
    ///
    /// Not a filter applied to a notebook-wide result: the rows are already
    /// keyed by `(notebook, document, term)`, so scoping is a `WHERE` clause and
    /// the totals that come back describe *that document*. Asking "what is in
    /// this file" and getting a header that counts the whole library is the kind
    /// of wrong number nobody notices until it matters.
    pub fn graph(
        &self,
        notebook_id: &str,
        owner_user_id: &str,
        document_sha256: Option<&str>,
        focus: Option<&str>,
        depth: u32,
        min_weight: u32,
        with_documents: bool,
    ) -> Result<GraphView> {
        if self.get(notebook_id, owner_user_id)?.is_none() {
            return Ok(GraphView::default());
        }

        let conn = self.conn.lock().expect("notebook store lock poisoned");

        // `MAX(label)` is kept: two documents spelling the same normalised term
        // differently is a cosmetic difference and either spelling is correct.
        // The type is not, so it is read separately below — two documents
        // *typing* the same term differently is a disagreement about what the
        // thing is, and picking the alphabetically larger answer is not a way
        // to settle it.
        let mut node_statement = conn.prepare(
            "SELECT normalised,
                    MAX(label) AS label,
                    SUM(occurrences) AS occurrences,
                    COUNT(DISTINCT document_sha256) AS documents
               FROM graph_nodes
              WHERE notebook_id = ?1
                AND (?2 IS NULL OR document_sha256 = ?2)
              GROUP BY normalised",
        )?;
        let all_terms: Vec<(String, String, u32, u32)> = node_statement
            .query_map(params![notebook_id, document_sha256], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, i64>(2)? as u32,
                    row.get::<_, i64>(3)? as u32,
                ))
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;

        // Every type anybody proposed for each term, with its support.
        let mut type_statement = conn.prepare(
            "SELECT normalised, node_type, COUNT(DISTINCT document_sha256) AS documents
               FROM graph_nodes
              WHERE notebook_id = ?1
                AND (?2 IS NULL OR document_sha256 = ?2)
                AND node_type IS NOT NULL
              GROUP BY normalised, node_type
              ORDER BY documents DESC, node_type ASC",
        )?;
        let mut types_of: BTreeMap<String, Vec<TypeCandidate>> = BTreeMap::new();
        for row in type_statement.query_map(params![notebook_id, document_sha256], |row| {
            Ok((
                row.get::<_, String>(0)?,
                TypeCandidate {
                    node_type: row.get::<_, String>(1)?,
                    documents: row.get::<_, i64>(2)? as u32,
                },
            ))
        })? {
            let (normalised, candidate) = row?;
            types_of.entry(normalised).or_default().push(candidate);
        }
        drop(type_statement);

        let mut edge_statement = conn.prepare(
            "SELECT source, target, SUM(weight) AS weight
               FROM graph_edges
              WHERE notebook_id = ?1
                AND (?2 IS NULL OR document_sha256 = ?2)
              GROUP BY source, target
             HAVING SUM(weight) >= ?3",
        )?;
        let all_term_edges: Vec<(String, String, u32)> = edge_statement
            .query_map(
                params![notebook_id, document_sha256, min_weight.max(1)],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, i64>(2)? as u32,
                    ))
                },
            )?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        drop(edge_statement);

        // The directed claims, looked up by the pair they are about. Read once
        // for the whole view rather than per edge.
        let claims = assertions_by_pair(&conn, notebook_id, document_sha256)?;

        // The notebook's files, and which terms each contributed. Read whether
        // or not they are wanted as nodes, because the totals have to describe
        // the whole graph either way.
        let mut document_statement = conn.prepare(
            "SELECT document_sha256, document_name
               FROM notebook_documents
              WHERE notebook_id = ?1
                AND (?2 IS NULL OR document_sha256 = ?2)
              ORDER BY document_name ASC",
        )?;
        let all_documents: Vec<(String, String)> = document_statement
            .query_map(params![notebook_id, document_sha256], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;

        let mut membership_statement = conn.prepare(
            "SELECT document_sha256, normalised, occurrences
               FROM graph_nodes
              WHERE notebook_id = ?1
                AND (?2 IS NULL OR document_sha256 = ?2)",
        )?;
        let all_membership: Vec<(String, String, u32)> = membership_statement
            .query_map(params![notebook_id, document_sha256], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, i64>(2)? as u32,
                ))
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;

        // Totals describe the whole graph, not the slice that survived the
        // filters. Both populations are counted, so "showing N of M" stays true
        // once files are in it.
        let total_terms = all_terms.len() as u32;
        let total_documents = if with_documents {
            all_documents.len() as u32
        } else {
            0
        };
        let total_nodes = total_terms + total_documents;

        // Every id the view can mention, built before any filtering so the
        // breadth-first walk below can cross from a term to a file and back.
        let mut candidate_edges: Vec<GraphEdge> = all_term_edges
            .iter()
            .map(|(source, target, weight)| {
                let key = if source <= target {
                    (source.clone(), target.clone())
                } else {
                    (target.clone(), source.clone())
                };
                let assertions = claims.get(&key).cloned().unwrap_or_default();
                GraphEdge {
                    source: node_id(notebook_id, source),
                    target: node_id(notebook_id, target),
                    kind: EdgeKind::Cooccurrence,
                    weight: *weight,
                    relation: settled_label(&assertions),
                    contested: is_contested(&assertions),
                    assertions,
                }
            })
            .collect();
        let total_cooccurrence = candidate_edges.len() as u32;

        let mut appears_in = 0u32;
        if with_documents {
            for (document_sha256, normalised, occurrences) in &all_membership {
                candidate_edges.push(GraphEdge {
                    // File first: the edge is directed from the document to the
                    // term it produced, which is the direction the arrow on the
                    // canvas is drawn in and the direction the fact runs.
                    source: document_node_id(notebook_id, document_sha256),
                    target: node_id(notebook_id, normalised),
                    kind: EdgeKind::AppearsIn,
                    weight: *occurrences,
                    relation: None,
                    assertions: Vec::new(),
                    contested: false,
                });
                appears_in += 1;
            }
        }
        let total_edges = total_cooccurrence + appears_in;

        // The local view: breadth-first from the focus, out to `depth`, over
        // every kind of edge. Crossing an `AppearsIn` is the point - focusing a
        // term should reach the files it came from, and focusing a file should
        // reach what was found in it.
        let keep: Option<BTreeSet<String>> = match focus {
            None => None,
            Some(focus_id) => {
                let known = all_terms
                    .iter()
                    .any(|(normalised, ..)| node_id(notebook_id, normalised) == focus_id)
                    || (with_documents
                        && all_documents
                            .iter()
                            .any(|(sha, _)| document_node_id(notebook_id, sha) == focus_id));
                if !known {
                    return Ok(GraphView {
                        nodes: Vec::new(),
                        edges: Vec::new(),
                        total_nodes,
                        total_edges,
                        total_terms,
                        total_documents,
                    });
                }

                let mut reached: BTreeSet<String> = BTreeSet::new();
                reached.insert(focus_id.to_string());
                let mut frontier: Vec<String> = vec![focus_id.to_string()];
                for _ in 0..depth {
                    let mut next: Vec<String> = Vec::new();
                    for edge in &candidate_edges {
                        if frontier.contains(&edge.source) && !reached.contains(&edge.target) {
                            reached.insert(edge.target.clone());
                            next.push(edge.target.clone());
                        } else if frontier.contains(&edge.target) && !reached.contains(&edge.source)
                        {
                            reached.insert(edge.source.clone());
                            next.push(edge.source.clone());
                        }
                    }
                    if next.is_empty() {
                        break;
                    }
                    frontier = next;
                }
                Some(reached)
            }
        };

        let edges: Vec<GraphEdge> = candidate_edges
            .into_iter()
            .filter(|edge| match &keep {
                Some(reached) => reached.contains(&edge.source) && reached.contains(&edge.target),
                None => true,
            })
            .collect();

        let degree_of = |id: &str| {
            edges
                .iter()
                .filter(|edge| edge.source == id || edge.target == id)
                .count() as u32
        };

        let mut nodes: Vec<GraphNode> = all_terms
            .iter()
            .filter_map(|(normalised, label, occurrences, documents)| {
                let id = node_id(notebook_id, normalised);
                match &keep {
                    Some(reached) if !reached.contains(&id) => None,
                    _ => {
                        let candidates = types_of.get(normalised).cloned().unwrap_or_default();
                        Some(GraphNode {
                            label: label.clone(),
                            kind: NodeKind::Term,
                            // The best-supported type, not the alphabetically
                            // largest. `type_candidates` carries the rest.
                            node_type: candidates.first().map(|c| c.node_type.clone()),
                            occurrences: *occurrences,
                            degree: degree_of(&id),
                            document_sha256: None,
                            document_count: *documents,
                            type_candidates: candidates,
                            id,
                        })
                    }
                }
            })
            .collect();

        if with_documents {
            // A file's `occurrences` is the passages of it that contributed a
            // term, counted from the rows rather than from its page count - the
            // number is about this graph, not about the document.
            for (sha, name) in &all_documents {
                let id = document_node_id(notebook_id, sha);
                if let Some(reached) = &keep {
                    if !reached.contains(&id) {
                        continue;
                    }
                }
                let occurrences = all_membership
                    .iter()
                    .filter(|(document_sha256, ..)| document_sha256 == sha)
                    .map(|(_, _, occurrences)| *occurrences)
                    .sum();
                nodes.push(GraphNode {
                    label: name.clone(),
                    kind: NodeKind::Document,
                    node_type: None,
                    occurrences,
                    degree: degree_of(&id),
                    document_sha256: Some(sha.clone()),
                    document_count: 0,
                    type_candidates: Vec::new(),
                    id,
                });
            }
        }

        // Busiest first, then alphabetically: a stable order, and the one a
        // reader wants when the list is longer than the screen.
        nodes.sort_by(|a, b| {
            b.degree
                .cmp(&a.degree)
                .then_with(|| b.occurrences.cmp(&a.occurrences))
                .then_with(|| a.label.cmp(&b.label))
        });

        Ok(GraphView {
            nodes,
            edges,
            total_nodes,
            total_edges,
            total_terms,
            total_documents,
        })
    }

    /// The terms one document contributed, as `(normalised, label)`.
    ///
    /// What the typing pass is allowed to classify. Handing the model this list
    /// — rather than asking it to find entities — is what makes the fourth gate
    /// in [`super::typing`] enforceable: anything outside it has no provenance.
    pub fn document_terms(
        &self,
        notebook_id: &str,
        owner_user_id: &str,
        document_sha256: &str,
    ) -> Result<Vec<(String, String)>> {
        if self.get(notebook_id, owner_user_id)?.is_none() {
            return Ok(Vec::new());
        }
        let conn = self.conn.lock().expect("notebook store lock poisoned");
        let mut statement = conn.prepare(
            "SELECT normalised, label FROM graph_nodes
              WHERE notebook_id = ?1 AND document_sha256 = ?2
              ORDER BY occurrences DESC, normalised ASC",
        )?;
        let rows = statement.query_map(params![notebook_id, document_sha256], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .context("the document's terms could not be read")
    }

    /// Applies a verified typing verdict to one document's rows.
    ///
    /// Only rows this document contributed are touched, so typing is per
    /// document and re-runnable exactly like the statistical pass. Nothing is
    /// inserted: this pass may only *update* what the cheap pass already put
    /// there, which is the storage-level expression of "it types and labels, it
    /// does not add entities".
    pub fn store_document_typing(
        &self,
        notebook_id: &str,
        owner_user_id: &str,
        document_sha256: &str,
        verdict: &super::typing::Verdict,
        typing_version: u32,
        source_revision: &str,
        extractor_id: &str,
    ) -> Result<()> {
        if self.get(notebook_id, owner_user_id)?.is_none() {
            anyhow::bail!("that notebook does not exist");
        }

        let mut conn = self.conn.lock().expect("notebook store lock poisoned");
        let tx = conn.transaction()?;

        for node in &verdict.nodes {
            tx.execute(
                "UPDATE graph_nodes SET node_type = ?4, extractor = 'llm'
                  WHERE notebook_id = ?1 AND document_sha256 = ?2 AND normalised = ?3",
                params![
                    notebook_id,
                    document_sha256,
                    &node.normalised,
                    &node.node_type
                ],
            )?;
            // The quote is stored against the passage the model cited, so the
            // inspector can show the actual sentence rather than only a page
            // number. Verified before it got here; see `typing::verify`.
            tx.execute(
                "UPDATE graph_evidence SET quote = ?4
                  WHERE notebook_id = ?1 AND subject = ?2 AND subject_kind = 'node'
                    AND chunk_id = ?3",
                params![
                    notebook_id,
                    node_id(notebook_id, &node.normalised),
                    &node.chunk_id,
                    &node.quote
                ],
            )?;
        }

        for edge in &verdict.edges {
            // Marked as named by the model; the label itself is not written
            // here, because this row cannot express which way round the claim
            // runs. Matched both ways because the co-occurrence row's ends are
            // sorted and the claim's are not.
            tx.execute(
                "UPDATE graph_edges SET extractor = 'llm'
                  WHERE notebook_id = ?1 AND document_sha256 = ?2
                    AND ((source = ?3 AND target = ?4) OR (source = ?4 AND target = ?3))",
                params![notebook_id, document_sha256, &edge.subject, &edge.object],
            )?;

            let subject_label =
                label_in(&tx, notebook_id, &edge.subject)?.unwrap_or_else(|| edge.subject.clone());
            let object_label =
                label_in(&tx, notebook_id, &edge.object)?.unwrap_or_else(|| edge.object.clone());
            super::assertions::write_assertion(
                &tx,
                notebook_id,
                &super::assertions::NewAssertion {
                    document_sha256: Some(document_sha256.to_string()),
                    subject: edge.subject.clone(),
                    subject_label,
                    predicate: edge.relation.clone(),
                    object: edge.object.clone(),
                    object_label,
                    provenance: AssertionProvenance::Model,
                    status: AssertionStatus::Proposed,
                    direction_certain: true,
                    extractor: extractor_id.to_string(),
                    extractor_version: typing_version,
                    source_revision: Some(source_revision.to_string()),
                    evidence: vec![super::assertions::AssertionEvidence {
                        chunk_id: edge.chunk_id.clone(),
                        document_sha256: document_sha256.to_string(),
                        page: page_of_chunk_in(&tx, notebook_id, &edge.chunk_id)?,
                        quote: Some(edge.quote.clone()),
                    }],
                },
            )?;
            // The co-occurrence evidence key is built from the *sorted* pair,
            // because that is how `store_document_graph` wrote it. Sorting here
            // is a lookup detail and touches nothing about the claim, whose
            // direction is already recorded above.
            let (first, second) = if edge.subject <= edge.object {
                (&edge.subject, &edge.object)
            } else {
                (&edge.object, &edge.subject)
            };
            tx.execute(
                "UPDATE graph_evidence SET quote = ?4
                  WHERE notebook_id = ?1 AND subject = ?2 AND subject_kind = 'edge'
                    AND chunk_id = ?3",
                params![
                    notebook_id,
                    format!(
                        "{}|{}",
                        node_id(notebook_id, first),
                        node_id(notebook_id, second)
                    ),
                    &edge.chunk_id,
                    &edge.quote
                ],
            )?;
        }

        tx.execute(
            "INSERT OR REPLACE INTO graph_build_units
                 (notebook_id, document_sha256, pass, extractor_version, completed_at,
                  source_revision, extractor_id)
             VALUES (?1, ?2, 'typing', ?3, ?4, ?5, ?6)",
            params![
                notebook_id,
                document_sha256,
                typing_version,
                chrono::Utc::now().to_rfc3339(),
                source_revision,
                extractor_id
            ],
        )?;

        tx.commit()?;
        Ok(())
    }

    /// The `(source, target)` pairs one document's graph already holds.
    ///
    /// What the relation pass is allowed to name. Handing
    /// [`super::relations::verify`] this set is what makes its fourth gate
    /// enforceable: a triplet joining two terms that are not already an edge has
    /// no support in the co-occurrence the cheap pass measured, whatever the
    /// model believes.
    pub fn document_edges(
        &self,
        notebook_id: &str,
        owner_user_id: &str,
        document_sha256: &str,
    ) -> Result<BTreeSet<(String, String)>> {
        if self.get(notebook_id, owner_user_id)?.is_none() {
            return Ok(BTreeSet::new());
        }
        let conn = self.conn.lock().expect("notebook store lock poisoned");
        let mut statement = conn.prepare(
            "SELECT source, target FROM graph_edges
              WHERE notebook_id = ?1 AND document_sha256 = ?2",
        )?;
        let rows = statement.query_map(params![notebook_id, document_sha256], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })?;
        rows.collect::<rusqlite::Result<BTreeSet<_>>>()
            .context("the document's edges could not be read")
    }

    /// Applies a verified relation verdict to one document.
    ///
    /// ## Each relation becomes a directed claim
    ///
    /// This used to `UPDATE graph_edges SET relation = ?` against the pair the
    /// verifier had already re-sorted into storage order — so "PV-2201 is
    /// manufactured by Northern Valve Company" was written onto the row
    /// `(northern valve company, pv-2201)` and read back as the company being
    /// manufactured by the valve. The label was preserved and the claim was
    /// inverted, silently, for every pair whose subject sorts after its object.
    ///
    /// The co-occurrence row is still what proves the two terms share a
    /// passage, and it is still checked before anything is written — this pass
    /// names links the cheap pass observed, it does not invent them. But the
    /// *claim* is written to [`super::assertions`], subject and object in the
    /// order REBEL produced them, and that is the only place direction lives.
    ///
    /// A person's earlier review of the same claim survives; see
    /// [`super::assertions::write_assertion`].
    pub fn store_document_relations(
        &self,
        notebook_id: &str,
        owner_user_id: &str,
        document_sha256: &str,
        verdict: &super::relations::Verdict,
        relation_version: u32,
        source_revision: &str,
    ) -> Result<()> {
        if self.get(notebook_id, owner_user_id)?.is_none() {
            anyhow::bail!("that notebook does not exist");
        }

        let mut conn = self.conn.lock().expect("notebook store lock poisoned");
        let tx = conn.transaction()?;

        for relation in &verdict.relations {
            // The link itself is marked as named by REBEL, so a graph built by
            // two passes can still say which named which. The label is not
            // written here — it has a direction and this row cannot hold one.
            tx.execute(
                "UPDATE graph_edges SET extractor = 'rebel'
                  WHERE notebook_id = ?1 AND document_sha256 = ?2
                    AND ((source = ?3 AND target = ?4) OR (source = ?4 AND target = ?3))",
                params![
                    notebook_id,
                    document_sha256,
                    &relation.subject,
                    &relation.object
                ],
            )?;

            let subject_label = label_in(&tx, notebook_id, &relation.subject)?
                .unwrap_or_else(|| relation.subject.clone());
            let object_label = label_in(&tx, notebook_id, &relation.object)?
                .unwrap_or_else(|| relation.object.clone());

            super::assertions::write_assertion(
                &tx,
                notebook_id,
                &super::assertions::NewAssertion {
                    document_sha256: Some(document_sha256.to_string()),
                    subject: relation.subject.clone(),
                    subject_label,
                    predicate: relation.relation.clone(),
                    object: relation.object.clone(),
                    object_label,
                    provenance: AssertionProvenance::Model,
                    status: AssertionStatus::Proposed,
                    // REBEL emits (subject, relation, object) and the verifier
                    // carried them through untouched. The direction is the
                    // model's, and it is a real claim.
                    direction_certain: true,
                    extractor: "rebel".to_string(),
                    extractor_version: relation_version,
                    source_revision: Some(source_revision.to_string()),
                    evidence: vec![super::assertions::AssertionEvidence {
                        chunk_id: relation.chunk_id.clone(),
                        document_sha256: document_sha256.to_string(),
                        page: page_of_chunk_in(&tx, notebook_id, &relation.chunk_id)?,
                        quote: relation.quote.clone(),
                    }],
                },
            )?;
        }

        // Recorded even when nothing was kept. A document REBEL found no usable
        // relation in is *done*, and re-running the pass over it on every
        // rebuild would spend minutes to reach the same empty answer.
        tx.execute(
            "INSERT OR REPLACE INTO graph_build_units
                 (notebook_id, document_sha256, pass, extractor_version, completed_at,
                  source_revision, extractor_id)
             VALUES (?1, ?2, 'relations', ?3, ?4, ?5, 'rebel')",
            params![
                notebook_id,
                document_sha256,
                relation_version,
                chrono::Utc::now().to_rfc3339(),
                source_revision
            ],
        )?;

        tx.commit()?;
        Ok(())
    }

    /// How much of the graph has been typed.
    ///
    /// Returns `(typed, total)` distinct terms. The screen needs both, because
    /// "412 terms, 38 typed" is the honest reading of a pass that mostly
    /// declined to classify — and a screen showing only the 38 would look like a
    /// small graph rather than a mostly-untyped one.
    pub fn typing_coverage(&self, notebook_id: &str, owner_user_id: &str) -> Result<(u32, u32)> {
        if self.get(notebook_id, owner_user_id)?.is_none() {
            return Ok((0, 0));
        }
        let conn = self.conn.lock().expect("notebook store lock poisoned");
        let typed: u32 = conn.query_row(
            "SELECT COUNT(DISTINCT normalised) FROM graph_nodes
              WHERE notebook_id = ?1 AND node_type IS NOT NULL",
            params![notebook_id],
            |row| row.get::<_, i64>(0),
        )? as u32;
        let total: u32 = conn.query_row(
            "SELECT COUNT(DISTINCT normalised) FROM graph_nodes WHERE notebook_id = ?1",
            params![notebook_id],
            |row| row.get::<_, i64>(0),
        )? as u32;
        Ok((typed, total))
    }

    /// The passages behind one node.
    pub fn node_evidence(
        &self,
        notebook_id: &str,
        owner_user_id: &str,
        node: &str,
    ) -> Result<Vec<EvidenceRow>> {
        if self.get(notebook_id, owner_user_id)?.is_none() {
            return Ok(Vec::new());
        }
        let conn = self.conn.lock().expect("notebook store lock poisoned");
        let mut statement = conn.prepare(
            "SELECT chunk_id, document_sha256, page, quote
               FROM graph_evidence
              WHERE notebook_id = ?1 AND subject = ?2 AND subject_kind = 'node'
              ORDER BY document_sha256 ASC, page ASC, chunk_id ASC",
        )?;
        let rows = statement.query_map(params![notebook_id, node], |row| {
            Ok(EvidenceRow {
                chunk_id: row.get(0)?,
                document_sha256: row.get(1)?,
                page: row.get::<_, i64>(2)? as u32,
                quote: row.get(3)?,
            })
        })?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .context("the evidence could not be read")
    }
}

/// The one label that can stand for a link, or `None` when none can.
///
/// An accepted claim wins, because a person said so. Failing that a single
/// proposal stands. Two proposals that disagree produce `None` and
/// [`is_contested`] produces `true`, so the canvas draws an unlabelled line and
/// the inspector explains why rather than the view inventing a winner.
fn settled_label(assertions: &[EdgeAssertion]) -> Option<String> {
    if let Some(accepted) = assertions
        .iter()
        .find(|a| a.status == AssertionStatus::Accepted)
    {
        return Some(accepted.predicate.clone());
    }
    let mut distinct: BTreeSet<&str> = BTreeSet::new();
    for assertion in assertions {
        distinct.insert(assertion.predicate.as_str());
    }
    match distinct.len() {
        1 => assertions.first().map(|a| a.predicate.clone()),
        _ => None,
    }
}

/// Whether the claims about one link disagree.
///
/// Two rows naming the same predicate in opposite directions disagree just as
/// much as two rows naming different predicates — "A supplies B" and "B supplies
/// A" cannot both be what the documents meant.
fn is_contested(assertions: &[EdgeAssertion]) -> bool {
    if assertions.len() < 2 {
        return false;
    }
    let mut seen: BTreeSet<(&str, &str)> = BTreeSet::new();
    for assertion in assertions {
        if assertion.status == AssertionStatus::Accepted {
            // A person has settled it. Whatever else was proposed is history,
            // not a live disagreement.
            return false;
        }
        seen.insert((assertion.predicate.as_str(), assertion.subject.as_str()));
    }
    seen.len() > 1
}

/// The page a chunk sits on, as the statistical pass already recorded it.
///
/// Read back rather than re-derived, so a claim's citation points at the same
/// page a node's citation does. Zero when the chunk is not in the evidence
/// table, which is a real state: the relation pass reads passages the
/// statistical pass may not have kept a row for.
fn page_of_chunk_in(conn: &Connection, notebook_id: &str, chunk_id: &str) -> Result<u32> {
    Ok(conn
        .query_row(
            "SELECT page FROM graph_evidence
              WHERE notebook_id = ?1 AND chunk_id = ?2 LIMIT 1",
            params![notebook_id, chunk_id],
            |row| row.get::<_, i64>(0),
        )
        .optional()?
        .unwrap_or(0) as u32)
}

/// The display label a notebook holds for a normalised term.
fn label_in(conn: &Connection, notebook_id: &str, normalised: &str) -> Result<Option<String>> {
    Ok(conn
        .query_row(
            "SELECT label FROM graph_nodes
              WHERE notebook_id = ?1 AND normalised = ?2
              ORDER BY occurrences DESC LIMIT 1",
            params![notebook_id, normalised],
            |row| row.get::<_, String>(0),
        )
        .optional()?)
}

/// Removes everything one document put into a notebook's graph.
///
/// Called when the document leaves the notebook, and when a rebuild is about to
/// re-derive its contribution from scratch. One function rather than two lists
/// of `DELETE`s, because the two lists drifting is exactly how the original bug
/// happened: the removal path deleted three of the five tables a document
/// writes to.
///
/// Runs inside the caller's transaction. It does not take the store lock and
/// must not: the caller already holds it.
pub(super) fn purge_document_contributions(
    conn: &Connection,
    notebook_id: &str,
    document_sha256: &str,
) -> Result<()> {
    // The claims this document produced, and their evidence. Evidence first —
    // it is keyed by the claim id, and deleting the parent first would strand
    // it.
    conn.execute(
        "DELETE FROM graph_assertion_evidence
          WHERE assertion_id IN (
              SELECT id FROM graph_assertions
               WHERE notebook_id = ?1 AND document_sha256 = ?2)",
        params![notebook_id, document_sha256],
    )?;
    conn.execute(
        "DELETE FROM graph_assertions
          WHERE notebook_id = ?1 AND document_sha256 = ?2
            AND provenance <> 'user'",
        params![notebook_id, document_sha256],
    )?;

    // A claim a *person* wrote about this document is not the extractor's to
    // delete. It keeps its row, loses the evidence that no longer exists, and
    // is marked stale below so the interface can say the source is gone.
    conn.execute(
        "DELETE FROM graph_assertion_evidence
          WHERE document_sha256 = ?1
            AND assertion_id IN (SELECT id FROM graph_assertions WHERE notebook_id = ?2)",
        params![document_sha256, notebook_id],
    )?;

    for table in ["graph_evidence", "graph_edges", "graph_nodes", "graph_build_units"] {
        conn.execute(
            &format!("DELETE FROM {table} WHERE notebook_id = ?1 AND document_sha256 = ?2"),
            params![notebook_id, document_sha256],
        )?;
    }

    refresh_assertion_staleness(conn, notebook_id)?;
    Ok(())
}

/// Marks claims whose terms no longer exist in the notebook, and clears the
/// mark from those whose terms have come back.
///
/// Both directions on purpose. A document removed and then added again should
/// leave the person's reviewed claims exactly as they were, not permanently
/// flagged because of a state they passed through.
pub(super) fn refresh_assertion_staleness(conn: &Connection, notebook_id: &str) -> Result<()> {
    let now = chrono::Utc::now().to_rfc3339();
    conn.execute(
        "UPDATE graph_assertions
            SET stale = 1, updated_at = ?2
          WHERE notebook_id = ?1
            AND stale = 0
            AND (subject NOT IN (SELECT normalised FROM graph_nodes WHERE notebook_id = ?1)
              OR object  NOT IN (SELECT normalised FROM graph_nodes WHERE notebook_id = ?1))",
        params![notebook_id, &now],
    )?;
    conn.execute(
        "UPDATE graph_assertions
            SET stale = 0, updated_at = ?2
          WHERE notebook_id = ?1
            AND stale = 1
            AND subject IN (SELECT normalised FROM graph_nodes WHERE notebook_id = ?1)
            AND object  IN (SELECT normalised FROM graph_nodes WHERE notebook_id = ?1)",
        params![notebook_id, &now],
    )?;
    Ok(())
}

/// Every directed claim in a notebook, keyed by the unordered pair it is about.
///
/// The key is sorted and the *value* is not: the map exists so an edge drawn
/// between two terms can find the claims about it, and each claim keeps the
/// subject and object the extractor gave it. Sorting the key is a lookup
/// convenience; sorting the claim would be the bug.
fn assertions_by_pair(
    conn: &Connection,
    notebook_id: &str,
    document_sha256: Option<&str>,
) -> Result<BTreeMap<(String, String), Vec<EdgeAssertion>>> {
    let mut statement = conn.prepare(
        "SELECT a.id, a.subject, a.subject_label, a.predicate, a.object, a.object_label,
                a.provenance, a.status, a.direction_certain, a.stale,
                (SELECT COUNT(*) FROM graph_assertion_evidence e
                  WHERE e.assertion_id = a.id) AS evidence_count
           FROM graph_assertions a
          WHERE a.notebook_id = ?1
            AND a.status <> 'rejected'
            AND (?2 IS NULL OR a.document_sha256 = ?2 OR a.document_sha256 IS NULL)
          ORDER BY a.subject_label ASC, a.predicate ASC",
    )?;
    let rows = statement.query_map(params![notebook_id, document_sha256], |row| {
        let provenance: String = row.get(6)?;
        let status: String = row.get(7)?;
        Ok((
            row.get::<_, String>(1)?,
            row.get::<_, String>(4)?,
            EdgeAssertion {
                id: row.get(0)?,
                subject: row.get(1)?,
                subject_label: row.get(2)?,
                predicate: row.get(3)?,
                object: row.get(4)?,
                object_label: row.get(5)?,
                provenance: if provenance == "user" {
                    AssertionProvenance::User
                } else {
                    AssertionProvenance::Model
                },
                status: match status.as_str() {
                    "accepted" => AssertionStatus::Accepted,
                    "rejected" => AssertionStatus::Rejected,
                    _ => AssertionStatus::Proposed,
                },
                direction_certain: row.get::<_, i64>(8)? != 0,
                stale: row.get::<_, i64>(9)? != 0,
                evidence_count: row.get::<_, i64>(10)? as u32,
            },
        ))
    })?;

    let mut out: BTreeMap<(String, String), Vec<EdgeAssertion>> = BTreeMap::new();
    for row in rows {
        let (subject, object, assertion) = row?;
        let key = if subject <= object {
            (subject, object)
        } else {
            (object, subject)
        };
        out.entry(key).or_default().push(assertion);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge::chunking::{Chunk, ChunkKind};
    use crate::knowledge::graph::statistical::{extract, EXTRACTOR_VERSION};

    const ALICE: &str = "user-alice";
    const BOB: &str = "user-bob";

    /// The extraction revision every test document is built from.
    ///
    /// One constant, so a test that wants to simulate a re-read says so by
    /// passing something else rather than by accident.
    const REVISION: &str = "2026-01-01T00:00:00Z";
    /// The extractor identity the statistical and typing passes record here.
    const PASS_ID: &str = "statistical";

    /// The revisions map `completed_units` compares against: the one document
    /// these tests use, at the revision it was built from.
    fn revisions() -> BTreeMap<String, String> {
        [("sha-doc".to_string(), REVISION.to_string())]
            .into_iter()
            .collect()
    }

    fn chunk(id: &str, text: &str) -> Chunk {
        Chunk {
            id: id.to_string(),
            document_sha256: "sha-doc".to_string(),
            ordinal: 0,
            text: text.to_string(),
            page: 1,
            section_path: Vec::new(),
            kind: ChunkKind::Prose,
            char_count: text.chars().count() as u32,
        }
    }

    fn sample() -> GraphDraft {
        extract(&[
            chunk("c1", "Northern Valve Company supplied PV-2201 to Unit Four."),
            chunk("c2", "Northern Valve Company serviced PV-2201 in Unit Four."),
        ])
    }

    fn page_one(_: &str) -> u32 {
        1
    }

    #[test]
    fn a_stored_graph_reads_back() {
        let store = NotebookStore::in_memory().unwrap();
        let notebook = store.create(ALICE, "Site A").unwrap();
        store
            .store_document_graph(
                &notebook.id,
                ALICE,
                "sha-doc",
                &page_one,
                &sample(),
                EXTRACTOR_VERSION,
                REVISION,
            )
            .unwrap();

        let view = store.graph(&notebook.id, ALICE, None, None, 1, 1, false).unwrap();
        assert!(!view.nodes.is_empty());
        assert!(!view.edges.is_empty());
        assert_eq!(view.total_nodes, view.nodes.len() as u32);
    }

    #[test]
    fn rebuilding_the_same_document_does_not_double_the_counts() {
        let store = NotebookStore::in_memory().unwrap();
        let notebook = store.create(ALICE, "Site A").unwrap();

        for _ in 0..3 {
            store
                .store_document_graph(
                    &notebook.id,
                    ALICE,
                    "sha-doc",
                    &page_one,
                    &sample(),
                    EXTRACTOR_VERSION,
                    REVISION,
                )
                .unwrap();
        }

        let once = sample();
        let view = store.graph(&notebook.id, ALICE, None, None, 1, 1, false).unwrap();

        let stored = view
            .nodes
            .iter()
            .find(|n| n.label == "PV-2201")
            .expect("the tag should be there");
        let drafted = once
            .nodes
            .iter()
            .find(|n| n.normalised == "pv-2201")
            .unwrap();
        assert_eq!(stored.occurrences, drafted.occurrences);
    }

    #[test]
    fn one_person_cannot_read_another_persons_graph() {
        let store = NotebookStore::in_memory().unwrap();
        let notebook = store.create(ALICE, "Site A").unwrap();
        store
            .store_document_graph(
                &notebook.id,
                ALICE,
                "sha-doc",
                &page_one,
                &sample(),
                EXTRACTOR_VERSION,
                REVISION,
            )
            .unwrap();

        assert_eq!(
            store.graph(&notebook.id, BOB, None, None, 1, 1, false).unwrap(),
            GraphView::default()
        );
        assert!(store
            .store_document_graph(
                &notebook.id,
                BOB,
                "sha-doc",
                &page_one,
                &sample(),
                EXTRACTOR_VERSION,
                REVISION
            )
            .is_err());
    }

    #[test]
    fn a_completed_unit_is_remembered_so_a_build_can_resume() {
        let store = NotebookStore::in_memory().unwrap();
        let notebook = store.create(ALICE, "Site A").unwrap();
        assert!(store
            .completed_units(&notebook.id, "statistical", EXTRACTOR_VERSION, PASS_ID, &revisions())
            .unwrap()
            .is_empty());

        store
            .store_document_graph(
                &notebook.id,
                ALICE,
                "sha-doc",
                &page_one,
                &sample(),
                EXTRACTOR_VERSION,
                REVISION,
            )
            .unwrap();

        assert!(store
            .completed_units(&notebook.id, "statistical", EXTRACTOR_VERSION, PASS_ID, &revisions())
            .unwrap()
            .contains("sha-doc"));
        // A different algorithm version has done nothing yet.
        assert!(store
            .completed_units(&notebook.id, "statistical", EXTRACTOR_VERSION + 1, PASS_ID, &revisions())
            .unwrap()
            .is_empty());
    }

    #[test]
    fn a_focused_view_returns_the_neighbourhood_and_still_reports_the_whole() {
        let store = NotebookStore::in_memory().unwrap();
        let notebook = store.create(ALICE, "Site A").unwrap();
        store
            .store_document_graph(
                &notebook.id,
                ALICE,
                "sha-doc",
                &page_one,
                &sample(),
                EXTRACTOR_VERSION,
                REVISION,
            )
            .unwrap();

        let whole = store.graph(&notebook.id, ALICE, None, None, 1, 1, false).unwrap();
        let focus = whole
            .nodes
            .iter()
            .find(|n| n.label == "PV-2201")
            .unwrap()
            .id
            .clone();

        let local = store
            .graph(&notebook.id, ALICE, None, Some(&focus), 1, 1, false)
            .unwrap();
        assert!(local.nodes.iter().any(|n| n.id == focus));
        assert!(local.nodes.len() <= whole.nodes.len());
        // The denominator describes the whole graph, not the slice.
        assert_eq!(local.total_nodes, whole.total_nodes);
    }

    #[test]
    fn every_node_can_produce_the_passages_behind_it() {
        let store = NotebookStore::in_memory().unwrap();
        let notebook = store.create(ALICE, "Site A").unwrap();
        store
            .store_document_graph(
                &notebook.id,
                ALICE,
                "sha-doc",
                &page_one,
                &sample(),
                EXTRACTOR_VERSION,
                REVISION,
            )
            .unwrap();

        let view = store.graph(&notebook.id, ALICE, None, None, 1, 1, false).unwrap();
        for node in &view.nodes {
            let evidence = store.node_evidence(&notebook.id, ALICE, &node.id).unwrap();
            assert!(!evidence.is_empty(), "{} cites nothing", node.label);
        }
    }

    #[test]
    fn evidence_is_not_readable_by_another_person() {
        let store = NotebookStore::in_memory().unwrap();
        let notebook = store.create(ALICE, "Site A").unwrap();
        store
            .store_document_graph(
                &notebook.id,
                ALICE,
                "sha-doc",
                &page_one,
                &sample(),
                EXTRACTOR_VERSION,
                REVISION,
            )
            .unwrap();

        let view = store.graph(&notebook.id, ALICE, None, None, 1, 1, false).unwrap();
        let id = view.nodes[0].id.clone();
        assert!(store
            .node_evidence(&notebook.id, BOB, &id)
            .unwrap()
            .is_empty());
    }

    /// Two documents, one notebook. Scoping to one must return only its rows -
    /// and, just as importantly, totals that describe only it. A header reading
    /// "7 terms" over a file that contributed three is the kind of wrong number
    /// nobody catches by looking at the picture.
    #[test]
    fn a_document_scope_isolates_both_the_rows_and_the_totals() {
        let store = NotebookStore::in_memory().unwrap();
        let notebook = store.create(ALICE, "Site A").unwrap();

        store
            .store_document_graph(
                &notebook.id,
                ALICE,
                "sha-valves",
                &page_one,
                &sample(),
                EXTRACTOR_VERSION,
                REVISION,
            )
            .unwrap();
        let other = extract(&[
            chunk("d1", "Southern Pump Works supplied PX-9100 to Unit Nine."),
            chunk("d2", "Southern Pump Works serviced PX-9100 in Unit Nine."),
        ]);
        store
            .store_document_graph(
                &notebook.id,
                ALICE,
                "sha-pumps",
                &page_one,
                &other,
                EXTRACTOR_VERSION,
                REVISION,
            )
            .unwrap();

        let whole = store
            .graph(&notebook.id, ALICE, None, None, 1, 1, false)
            .unwrap();
        let scoped = store
            .graph(&notebook.id, ALICE, Some("sha-valves"), None, 1, 1, false)
            .unwrap();

        assert!(scoped.total_terms < whole.total_terms);
        assert_eq!(scoped.total_terms, scoped.nodes.len() as u32);
        assert!(
            scoped
                .nodes
                .iter()
                .all(|node| !node.label.contains("Southern Pump")),
            "a scoped graph leaked the other document"
        );
        assert!(
            whole
                .nodes
                .iter()
                .any(|node| node.label.contains("Southern Pump")),
            "the fixture never had two documents to separate"
        );
    }

    /// The file list is scoped too. A one-file view that draws every file in the
    /// notebook as an unattached box is reporting the library, not the file.
    #[test]
    fn a_document_scope_shows_only_that_file_as_a_node() {
        let store = NotebookStore::in_memory().unwrap();
        let notebook = store.create(ALICE, "Site A").unwrap();
        store.add_document(&notebook.id, ALICE, "sha-valves", "valves.pdf").unwrap();
        store.add_document(&notebook.id, ALICE, "sha-pumps", "pumps.pdf").unwrap();
        store
            .store_document_graph(
                &notebook.id,
                ALICE,
                "sha-valves",
                &page_one,
                &sample(),
                EXTRACTOR_VERSION,
                REVISION,
            )
            .unwrap();

        let scoped = store
            .graph(&notebook.id, ALICE, Some("sha-valves"), None, 1, 1, true)
            .unwrap();

        assert_eq!(scoped.total_documents, 1);
        let files: Vec<_> = scoped
            .nodes
            .iter()
            .filter(|node| node.kind == NodeKind::Document)
            .collect();
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].label, "valves.pdf");
    }

    #[test]
    fn typing_updates_the_type_the_relation_and_the_quote() {
        use crate::knowledge::graph::typing::{TypedEdge, TypedNode, TypingStats, Verdict};

        let store = NotebookStore::in_memory().unwrap();
        let notebook = store.create(ALICE, "Site A").unwrap();
        store
            .store_document_graph(
                &notebook.id,
                ALICE,
                "sha-doc",
                &page_one,
                &sample(),
                EXTRACTOR_VERSION,
                REVISION,
            )
            .unwrap();

        // Untyped to begin with — a real state, not a missing value.
        let before = store.graph(&notebook.id, ALICE, None, None, 1, 1, false).unwrap();
        assert!(before.nodes.iter().all(|node| node.node_type.is_none()));
        assert_eq!(store.typing_coverage(&notebook.id, ALICE).unwrap().0, 0);

        let verdict = Verdict {
            nodes: vec![TypedNode {
                normalised: "pv-2201".into(),
                node_type: "equipment".into(),
                chunk_id: "c1".into(),
                quote: "Northern Valve Company supplied PV-2201".into(),
            }],
            edges: vec![TypedEdge {
                subject: "northern valve company".into(),
                object: "pv-2201".into(),
                relation: "supplies".into(),
                chunk_id: "c1".into(),
                quote: "Northern Valve Company supplied PV-2201".into(),
            }],
            stats: TypingStats::default(),
        };
        store
            .store_document_typing(&notebook.id, ALICE, "sha-doc", &verdict, 1, REVISION, PASS_ID)
            .unwrap();

        let after = store.graph(&notebook.id, ALICE, None, None, 1, 1, false).unwrap();
        let typed = after
            .nodes
            .iter()
            .find(|node| node.label == "PV-2201")
            .unwrap();
        assert_eq!(typed.node_type.as_deref(), Some("equipment"));
        assert!(after
            .edges
            .iter()
            .any(|edge| edge.relation.as_deref() == Some("supplies")));

        // The verified quote reaches the evidence, so the inspector can show
        // the sentence rather than only a page number.
        let evidence = store.node_evidence(&notebook.id, ALICE, &typed.id).unwrap();
        assert!(evidence
            .iter()
            .any(|row| row.quote.as_deref() == Some("Northern Valve Company supplied PV-2201")));

        let (types, total) = store.typing_coverage(&notebook.id, ALICE).unwrap();
        assert_eq!(types, 1);
        assert!(total > types, "the rest is still untyped, and says so");
    }

    #[test]
    fn typing_records_a_unit_so_the_pass_can_resume() {
        use crate::knowledge::graph::typing::Verdict;

        let store = NotebookStore::in_memory().unwrap();
        let notebook = store.create(ALICE, "Site A").unwrap();
        store
            .store_document_graph(
                &notebook.id,
                ALICE,
                "sha-doc",
                &page_one,
                &sample(),
                EXTRACTOR_VERSION,
                REVISION,
            )
            .unwrap();

        assert!(store
            .completed_units(&notebook.id, "typing", 1, PASS_ID, &revisions())
            .unwrap()
            .is_empty());
        store
            .store_document_typing(&notebook.id, ALICE, "sha-doc", &Verdict::default(), 1, REVISION, PASS_ID)
            .unwrap();
        assert!(store
            .completed_units(&notebook.id, "typing", 1, PASS_ID, &revisions())
            .unwrap()
            .contains("sha-doc"));
        // The statistical unit is a separate pass and is unaffected.
        assert!(store
            .completed_units(&notebook.id, "statistical", EXTRACTOR_VERSION, PASS_ID, &revisions())
            .unwrap()
            .contains("sha-doc"));
    }

    #[test]
    fn one_person_cannot_type_another_persons_graph() {
        use crate::knowledge::graph::typing::Verdict;

        let store = NotebookStore::in_memory().unwrap();
        let notebook = store.create(ALICE, "Site A").unwrap();
        assert!(store
            .store_document_typing(&notebook.id, BOB, "sha-doc", &Verdict::default(), 1, REVISION, PASS_ID)
            .is_err());
        assert!(store
            .document_terms(&notebook.id, BOB, "sha-doc")
            .unwrap()
            .is_empty());
    }

    /// The relation a person opens a notebook graph to see.
    ///
    /// Before this, the view grouped `document_sha256` away and the only trace
    /// of a file was a sha256 on an evidence row that the interface had nothing
    /// to resolve against - so a term could not say which of eleven files it
    /// came from.
    #[test]
    fn a_file_is_a_node_joined_to_the_terms_found_in_it() {
        let store = NotebookStore::in_memory().unwrap();
        let notebook = store.create(ALICE, "Site A").unwrap();
        store
            .add_document(&notebook.id, ALICE, "sha-doc", "valves.pdf")
            .unwrap();
        store
            .store_document_graph(
                &notebook.id,
                ALICE,
                "sha-doc",
                &page_one,
                &sample(),
                EXTRACTOR_VERSION,
                REVISION,
            )
            .unwrap();

        let view = store.graph(&notebook.id, ALICE, None, None, 1, 1, true).unwrap();

        let file = view
            .nodes
            .iter()
            .find(|node| node.kind == NodeKind::Document)
            .expect("the file should be a node");
        assert_eq!(file.label, "valves.pdf");
        assert_eq!(file.document_sha256.as_deref(), Some("sha-doc"));

        let terms: Vec<&GraphNode> = view
            .nodes
            .iter()
            .filter(|node| node.kind == NodeKind::Term)
            .collect();
        assert!(!terms.is_empty());

        // Every term is joined to the file it was found in, and says so.
        for term in &terms {
            assert!(
                view.edges.iter().any(|edge| edge.kind == EdgeKind::AppearsIn
                    && edge.source == file.id
                    && edge.target == term.id),
                "{} is not joined to the file it came from",
                term.label
            );
            assert_eq!(term.document_count, 1);
        }

        assert_eq!(view.total_terms, terms.len() as u32);
        assert_eq!(view.total_documents, 1);
        assert_eq!(view.total_nodes, view.total_terms + view.total_documents);
    }

    /// A file the extractor got nothing out of is drawn, unattached.
    ///
    /// Five of the eleven documents in the notebook that prompted this produced
    /// no terms at all. A view that simply omitted them would read as a graph of
    /// six files; drawing them with no links says what actually happened.
    #[test]
    fn a_file_that_contributed_no_terms_is_still_a_node() {
        let store = NotebookStore::in_memory().unwrap();
        let notebook = store.create(ALICE, "Site A").unwrap();
        store
            .add_document(&notebook.id, ALICE, "sha-doc", "valves.pdf")
            .unwrap();
        store
            .add_document(&notebook.id, ALICE, "sha-empty", "scan.jpeg")
            .unwrap();
        store
            .store_document_graph(
                &notebook.id,
                ALICE,
                "sha-doc",
                &page_one,
                &sample(),
                EXTRACTOR_VERSION,
                REVISION,
            )
            .unwrap();

        let view = store.graph(&notebook.id, ALICE, None, None, 1, 1, true).unwrap();

        let empty = view
            .nodes
            .iter()
            .find(|node| node.label == "scan.jpeg")
            .expect("a file that produced nothing is still part of the notebook");
        assert_eq!(empty.kind, NodeKind::Document);
        assert_eq!(empty.degree, 0);
        assert_eq!(empty.occurrences, 0);
        assert_eq!(view.total_documents, 2);
    }

    /// Focusing a file gives what was found in it; focusing a term gives its
    /// files. The walk has to cross an `AppearsIn` in both directions, which a
    /// term-only breadth-first search never did.
    #[test]
    fn focusing_a_file_reaches_the_terms_found_in_it() {
        let store = NotebookStore::in_memory().unwrap();
        let notebook = store.create(ALICE, "Site A").unwrap();
        store
            .add_document(&notebook.id, ALICE, "sha-doc", "valves.pdf")
            .unwrap();
        store
            .store_document_graph(
                &notebook.id,
                ALICE,
                "sha-doc",
                &page_one,
                &sample(),
                EXTRACTOR_VERSION,
                REVISION,
            )
            .unwrap();

        let whole = store.graph(&notebook.id, ALICE, None, None, 1, 1, true).unwrap();
        let file = whole
            .nodes
            .iter()
            .find(|node| node.kind == NodeKind::Document)
            .unwrap();

        let local = store
            .graph(&notebook.id, ALICE, None, Some(&file.id), 1, 1, true)
            .unwrap();
        assert!(
            local
                .nodes
                .iter()
                .filter(|node| node.kind == NodeKind::Term)
                .count()
                > 0,
            "focusing a file should reach the terms found in it"
        );
        // The denominator still describes the whole graph, not the local slice.
        assert_eq!(local.total_nodes, whole.total_nodes);
    }

    /// Asking for the term graph gives exactly what it gave before files
    /// existed - no file nodes, no `AppearsIn`, and totals that count terms
    /// only. The subgraph renderer depends on this: a file is not a term, and
    /// rendering one into a chat turn would be nonsense.
    #[test]
    fn the_term_only_view_contains_no_files() {
        let store = NotebookStore::in_memory().unwrap();
        let notebook = store.create(ALICE, "Site A").unwrap();
        store
            .add_document(&notebook.id, ALICE, "sha-doc", "valves.pdf")
            .unwrap();
        store
            .store_document_graph(
                &notebook.id,
                ALICE,
                "sha-doc",
                &page_one,
                &sample(),
                EXTRACTOR_VERSION,
                REVISION,
            )
            .unwrap();

        let view = store.graph(&notebook.id, ALICE, None, None, 1, 1, false).unwrap();
        assert!(view.nodes.iter().all(|node| node.kind == NodeKind::Term));
        assert!(view
            .edges
            .iter()
            .all(|edge| edge.kind == EdgeKind::Cooccurrence));
        assert_eq!(view.total_documents, 0);
        assert_eq!(view.total_nodes, view.total_terms);
    }

    /// A document id must never be mistaken for a term id, because the same
    /// string is what `node_evidence` is asked for. Different lengths make the
    /// two impossible to confuse however a term happens to be spelled.
    #[test]
    fn a_document_id_cannot_collide_with_a_term_id() {
        let term = node_id("notebook-1", "document:sha-doc");
        let document = document_node_id("notebook-1", "sha-doc");
        assert_ne!(term, document);
        assert_eq!(term.len(), 32);
        assert_eq!(document.len(), 33);
        assert_eq!(
            document,
            document_node_id("notebook-1", "sha-doc"),
            "the id must be stable across rebuilds"
        );
        assert_ne!(
            document,
            document_node_id("notebook-2", "sha-doc"),
            "and scoped to its notebook"
        );
    }

    #[test]
    fn a_node_id_is_stable_and_notebook_scoped() {
        assert_eq!(node_id("nb-1", "pv-2201"), node_id("nb-1", "pv-2201"));
        assert_ne!(node_id("nb-1", "pv-2201"), node_id("nb-2", "pv-2201"));
        assert_eq!(node_id("nb-1", "pv-2201").len(), 32);
    }
}
