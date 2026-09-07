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

use std::collections::BTreeSet;

use anyhow::{Context, Result};
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

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
    /// `None` for a co-occurrence the typing pass has not labelled. Always
    /// `None` for an `AppearsIn`, whose meaning is fixed and needs no model.
    pub relation: Option<String>,
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
                PRIMARY KEY (notebook_id, document_sha256, pass, extractor_version)
            );",
        )?;
        Ok(())
    }

    /// Writes one document's contribution, replacing whatever it wrote before.
    ///
    /// The delete-then-insert is the whole reason this is safe to re-run: a
    /// build interrupted halfway leaves rows behind, and the next attempt must
    /// not add to them.
    pub fn store_document_graph(
        &self,
        notebook_id: &str,
        owner_user_id: &str,
        document_sha256: &str,
        page_of_chunk: &dyn Fn(&str) -> u32,
        draft: &GraphDraft,
        extractor_version: u32,
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
                 (notebook_id, document_sha256, pass, extractor_version, completed_at)
             VALUES (?1, ?2, 'statistical', ?3, ?4)",
            params![
                notebook_id,
                document_sha256,
                extractor_version,
                chrono::Utc::now().to_rfc3339()
            ],
        )?;

        tx.commit()?;
        Ok(())
    }

    /// Documents already processed at this extractor version.
    ///
    /// What makes a build resumable: the caller skips these rather than
    /// redoing work that is already on disk.
    pub fn completed_units(
        &self,
        notebook_id: &str,
        pass: &str,
        extractor_version: u32,
    ) -> Result<BTreeSet<String>> {
        let conn = self.conn.lock().expect("notebook store lock poisoned");
        let mut statement = conn.prepare(
            "SELECT document_sha256 FROM graph_build_units
              WHERE notebook_id = ?1 AND pass = ?2 AND extractor_version = ?3",
        )?;
        let rows = statement.query_map(params![notebook_id, pass, extractor_version], |row| {
            row.get::<_, String>(0)
        })?;
        rows.collect::<rusqlite::Result<BTreeSet<_>>>()
            .context("the completed build units could not be read")
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

        let mut node_statement = conn.prepare(
            "SELECT normalised,
                    MAX(label) AS label,
                    MAX(node_type) AS node_type,
                    SUM(occurrences) AS occurrences,
                    COUNT(DISTINCT document_sha256) AS documents
               FROM graph_nodes
              WHERE notebook_id = ?1
                AND (?2 IS NULL OR document_sha256 = ?2)
              GROUP BY normalised",
        )?;
        let all_terms: Vec<(String, String, Option<String>, u32, u32)> = node_statement
            .query_map(params![notebook_id, document_sha256], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, Option<String>>(2)?,
                    row.get::<_, i64>(3)? as u32,
                    row.get::<_, i64>(4)? as u32,
                ))
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;

        let mut edge_statement = conn.prepare(
            "SELECT source, target, SUM(weight) AS weight, MAX(relation) AS relation
               FROM graph_edges
              WHERE notebook_id = ?1
                AND (?2 IS NULL OR document_sha256 = ?2)
              GROUP BY source, target
             HAVING SUM(weight) >= ?3",
        )?;
        let all_term_edges: Vec<(String, String, u32, Option<String>)> = edge_statement
            .query_map(
                params![notebook_id, document_sha256, min_weight.max(1)],
                |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, i64>(2)? as u32,
                    row.get::<_, Option<String>>(3)?,
                ))
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;

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
            .map(|(source, target, weight, relation)| GraphEdge {
                source: node_id(notebook_id, source),
                target: node_id(notebook_id, target),
                kind: EdgeKind::Cooccurrence,
                weight: *weight,
                relation: relation.clone(),
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
            .filter_map(|(normalised, label, node_type, occurrences, documents)| {
                let id = node_id(notebook_id, normalised);
                match &keep {
                    Some(reached) if !reached.contains(&id) => None,
                    _ => Some(GraphNode {
                        label: label.clone(),
                        kind: NodeKind::Term,
                        node_type: node_type.clone(),
                        occurrences: *occurrences,
                        degree: degree_of(&id),
                        document_sha256: None,
                        document_count: *documents,
                        id,
                    }),
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
            tx.execute(
                "UPDATE graph_edges SET relation = ?5, extractor = 'llm'
                  WHERE notebook_id = ?1 AND document_sha256 = ?2
                    AND source = ?3 AND target = ?4",
                params![
                    notebook_id,
                    document_sha256,
                    &edge.source,
                    &edge.target,
                    &edge.relation
                ],
            )?;
            tx.execute(
                "UPDATE graph_evidence SET quote = ?4
                  WHERE notebook_id = ?1 AND subject = ?2 AND subject_kind = 'edge'
                    AND chunk_id = ?3",
                params![
                    notebook_id,
                    format!(
                        "{}|{}",
                        node_id(notebook_id, &edge.source),
                        node_id(notebook_id, &edge.target)
                    ),
                    &edge.chunk_id,
                    &edge.quote
                ],
            )?;
        }

        tx.execute(
            "INSERT OR REPLACE INTO graph_build_units
                 (notebook_id, document_sha256, pass, extractor_version, completed_at)
             VALUES (?1, ?2, 'typing', ?3, ?4)",
            params![
                notebook_id,
                document_sha256,
                typing_version,
                chrono::Utc::now().to_rfc3339()
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

    /// Applies a verified relation verdict to one document's edges.
    ///
    /// `UPDATE` only, never `INSERT` — the storage-level expression of "this
    /// pass names edges, it does not create them". A relation that reached here
    /// has already been checked against the edges this document actually has,
    /// but writing it as an update means a bug in that check costs a label
    /// rather than a fabricated link.
    ///
    /// The extractor is recorded as `rebel` rather than `llm`, so a graph built
    /// by two different passes can still say which named which. That matters
    /// when the relation is wrong and somebody has to work out why.
    pub fn store_document_relations(
        &self,
        notebook_id: &str,
        owner_user_id: &str,
        document_sha256: &str,
        verdict: &super::relations::Verdict,
        relation_version: u32,
    ) -> Result<()> {
        if self.get(notebook_id, owner_user_id)?.is_none() {
            anyhow::bail!("that notebook does not exist");
        }

        let mut conn = self.conn.lock().expect("notebook store lock poisoned");
        let tx = conn.transaction()?;

        for relation in &verdict.relations {
            tx.execute(
                "UPDATE graph_edges SET relation = ?5, extractor = 'rebel'
                  WHERE notebook_id = ?1 AND document_sha256 = ?2
                    AND source = ?3 AND target = ?4",
                params![
                    notebook_id,
                    document_sha256,
                    &relation.source,
                    &relation.target,
                    &relation.relation
                ],
            )?;
        }

        // Recorded even when nothing was kept. A document REBEL found no usable
        // relation in is *done*, and re-running the pass over it on every
        // rebuild would spend minutes to reach the same empty answer.
        tx.execute(
            "INSERT OR REPLACE INTO graph_build_units
                 (notebook_id, document_sha256, pass, extractor_version, completed_at)
             VALUES (?1, ?2, 'relations', ?3, ?4)",
            params![
                notebook_id,
                document_sha256,
                relation_version,
                chrono::Utc::now().to_rfc3339()
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge::chunking::{Chunk, ChunkKind};
    use crate::knowledge::graph::statistical::{extract, EXTRACTOR_VERSION};

    const ALICE: &str = "user-alice";
    const BOB: &str = "user-bob";

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
                EXTRACTOR_VERSION
            )
            .is_err());
    }

    #[test]
    fn a_completed_unit_is_remembered_so_a_build_can_resume() {
        let store = NotebookStore::in_memory().unwrap();
        let notebook = store.create(ALICE, "Site A").unwrap();
        assert!(store
            .completed_units(&notebook.id, "statistical", EXTRACTOR_VERSION)
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
            )
            .unwrap();

        assert!(store
            .completed_units(&notebook.id, "statistical", EXTRACTOR_VERSION)
            .unwrap()
            .contains("sha-doc"));
        // A different algorithm version has done nothing yet.
        assert!(store
            .completed_units(&notebook.id, "statistical", EXTRACTOR_VERSION + 1)
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
                source: "northern valve company".into(),
                target: "pv-2201".into(),
                relation: "supplies".into(),
                chunk_id: "c1".into(),
                quote: "Northern Valve Company supplied PV-2201".into(),
            }],
            stats: TypingStats::default(),
        };
        store
            .store_document_typing(&notebook.id, ALICE, "sha-doc", &verdict, 1)
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
            )
            .unwrap();

        assert!(store
            .completed_units(&notebook.id, "typing", 1)
            .unwrap()
            .is_empty());
        store
            .store_document_typing(&notebook.id, ALICE, "sha-doc", &Verdict::default(), 1)
            .unwrap();
        assert!(store
            .completed_units(&notebook.id, "typing", 1)
            .unwrap()
            .contains("sha-doc"));
        // The statistical unit is a separate pass and is unaffected.
        assert!(store
            .completed_units(&notebook.id, "statistical", EXTRACTOR_VERSION)
            .unwrap()
            .contains("sha-doc"));
    }

    #[test]
    fn one_person_cannot_type_another_persons_graph() {
        use crate::knowledge::graph::typing::Verdict;

        let store = NotebookStore::in_memory().unwrap();
        let notebook = store.create(ALICE, "Site A").unwrap();
        assert!(store
            .store_document_typing(&notebook.id, BOB, "sha-doc", &Verdict::default(), 1)
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
