//! Directed semantic assertions over a notebook's terms.
//!
//! ## Why this is not a column on `graph_edges`
//!
//! `graph_edges` records a *co-occurrence*: two terms were in the same passage.
//! That observation has no direction, and the row is keyed on the pair with its
//! ends sorted so one pair cannot be stored twice. Sorting is correct for an
//! observation and destructive for an assertion — "PV-2201 is manufactured by
//! Northern Valve Company" sorted alphabetically becomes "northern valve
//! company → pv-2201", which reads as the valve company being manufactured by
//! the valve. The relation label survived; the claim it belonged to did not.
//!
//! So a directed claim lives here, in its own table, keyed by
//! `(subject, predicate, object)` in the order the extractor produced them.
//! Nothing in this module sorts a pair, and nothing that reads it may.
//!
//! ## Conflicting claims are rows, not a `MAX`
//!
//! Two passages can say different things about the same pair. The old edge
//! table held one nullable `relation` column and the view read it with
//! `MAX(relation)`, which is to say it picked whichever string sorted highest
//! and threw the other away — silently, with no record that a disagreement had
//! ever existed.
//!
//! Here each distinct `(subject, predicate, object)` is its own row with its own
//! evidence, so a pair carrying two claims carries both. The interface is told
//! they disagree rather than being handed a winner.
//!
//! ## Review status is part of the claim
//!
//! A relation a model proposed and a relation a person confirmed are different
//! kinds of thing, and an answer built from them should be able to say which it
//! used. [`AssertionStatus`] is therefore stored, never inferred, and
//! [`AssertionProvenance`] records who put the row there in the first place.

use anyhow::{Context, Result};
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::store::NotebookStore;

/// Who put this claim in the graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum AssertionProvenance {
    /// A local extraction model proposed it.
    Model,
    /// A person wrote it, or corrected one the model proposed.
    User,
}

impl AssertionProvenance {
    pub fn as_str(self) -> &'static str {
        match self {
            AssertionProvenance::Model => "model",
            AssertionProvenance::User => "user",
        }
    }

    fn parse(raw: &str) -> Self {
        match raw {
            "user" => AssertionProvenance::User,
            _ => AssertionProvenance::Model,
        }
    }
}

/// Where a claim stands with the person who owns the notebook.
///
/// The three states answer different questions and are deliberately not
/// collapsible into a boolean. `Proposed` is "a model said this and nobody has
/// looked"; `Accepted` is "a person read the evidence and agrees"; `Rejected` is
/// "a person read the evidence and says no" — which is information worth
/// keeping, because deleting it invites the next extraction pass to propose the
/// same wrong thing again with nothing to say it was already refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum AssertionStatus {
    Proposed,
    Accepted,
    Rejected,
}

impl AssertionStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            AssertionStatus::Proposed => "proposed",
            AssertionStatus::Accepted => "accepted",
            AssertionStatus::Rejected => "rejected",
        }
    }

    fn parse(raw: &str) -> Self {
        match raw {
            "accepted" => AssertionStatus::Accepted,
            "rejected" => AssertionStatus::Rejected,
            _ => AssertionStatus::Proposed,
        }
    }

    /// Whether an answer may be built on this claim.
    ///
    /// Stated once, here, rather than re-decided at each call site. A rejected
    /// claim is never retrieved; a proposed one is retrieved and labelled as
    /// unreviewed so the answer can say so.
    pub fn usable_as_evidence(self) -> bool {
        !matches!(self, AssertionStatus::Rejected)
    }
}

/// One passage behind an assertion.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AssertionEvidence {
    pub chunk_id: String,
    pub document_sha256: String,
    pub page: u32,
    /// The verbatim sentence the claim was read out of, when the extractor
    /// supplied one and it was checked against the passage.
    pub quote: Option<String>,
}

/// A directed claim about two terms.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Assertion {
    pub id: String,
    pub notebook_id: String,
    /// The document this was read out of. `None` for a claim a person wrote
    /// that no single source is responsible for.
    pub document_sha256: Option<String>,
    /// Normalised subject term, as `graph_nodes.normalised` holds it.
    pub subject: String,
    pub subject_label: String,
    pub predicate: String,
    pub object: String,
    pub object_label: String,
    pub provenance: AssertionProvenance,
    pub status: AssertionStatus,
    /// False for a claim recovered from the old undirected storage, where the
    /// direction was destroyed before this table existed.
    ///
    /// A false here is shown as "direction unverified" rather than being
    /// guessed. Inventing a direction for a migrated row would manufacture a
    /// claim nobody made.
    pub direction_certain: bool,
    pub extractor: String,
    pub extractor_version: u32,
    /// The extraction revision of the source when this was written. Compared on
    /// rebuild: a claim whose source has been re-read since is marked stale
    /// rather than silently trusted.
    pub source_revision: Option<String>,
    /// True when the supporting evidence no longer stands at the revision this
    /// was made against.
    pub stale: bool,
    /// What the reviewer said, if anything.
    pub note: Option<String>,
    pub evidence: Vec<AssertionEvidence>,
    pub created_at: String,
    pub updated_at: String,
    pub reviewed_at: Option<String>,
}

/// A new claim on its way in.
#[derive(Debug, Clone)]
pub struct NewAssertion {
    pub document_sha256: Option<String>,
    pub subject: String,
    pub subject_label: String,
    pub predicate: String,
    pub object: String,
    pub object_label: String,
    pub provenance: AssertionProvenance,
    pub status: AssertionStatus,
    pub direction_certain: bool,
    pub extractor: String,
    pub extractor_version: u32,
    pub source_revision: Option<String>,
    pub evidence: Vec<AssertionEvidence>,
}

/// The stable id of a claim.
///
/// Derived from everything that makes the claim what it is, so re-running the
/// extractor over an unchanged document updates the row it wrote last time
/// instead of adding a second identical one — and so a *different* predicate
/// between the same two terms is a different row, which is what keeps a
/// disagreement visible.
pub fn assertion_id(
    notebook_id: &str,
    document_sha256: Option<&str>,
    subject: &str,
    predicate: &str,
    object: &str,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(notebook_id.as_bytes());
    hasher.update(b"\x1f");
    hasher.update(document_sha256.unwrap_or("").as_bytes());
    hasher.update(b"\x1f");
    hasher.update(subject.as_bytes());
    hasher.update(b"\x1f");
    hasher.update(predicate.as_bytes());
    hasher.update(b"\x1f");
    hasher.update(object.as_bytes());
    format!("a{:x}", hasher.finalize())[..33].to_string()
}

/// How a set of claims about one pair of terms reads.
///
/// Carried on the edge so the canvas can draw an arrowhead the right way round
/// and the inspector can show that two passages disagree.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EdgeAssertion {
    pub id: String,
    pub subject: String,
    pub subject_label: String,
    pub predicate: String,
    pub object: String,
    pub object_label: String,
    pub provenance: AssertionProvenance,
    pub status: AssertionStatus,
    pub direction_certain: bool,
    pub stale: bool,
    pub evidence_count: u32,
}

impl NotebookStore {
    pub(super) fn prepare_assertions(conn: &Connection) -> Result<()> {
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS graph_assertions (
                id                TEXT PRIMARY KEY,
                notebook_id       TEXT NOT NULL,
                document_sha256   TEXT,
                subject           TEXT NOT NULL,
                subject_label     TEXT NOT NULL,
                predicate         TEXT NOT NULL,
                object            TEXT NOT NULL,
                object_label      TEXT NOT NULL,
                provenance        TEXT NOT NULL,
                status            TEXT NOT NULL,
                direction_certain INTEGER NOT NULL DEFAULT 1,
                extractor         TEXT NOT NULL,
                extractor_version INTEGER NOT NULL DEFAULT 0,
                source_revision   TEXT,
                stale             INTEGER NOT NULL DEFAULT 0,
                note              TEXT,
                created_at        TEXT NOT NULL,
                updated_at        TEXT NOT NULL,
                reviewed_at       TEXT
            );
            CREATE INDEX IF NOT EXISTS graph_assertions_notebook_idx
                ON graph_assertions(notebook_id);
            CREATE INDEX IF NOT EXISTS graph_assertions_document_idx
                ON graph_assertions(notebook_id, document_sha256);
            CREATE INDEX IF NOT EXISTS graph_assertions_pair_idx
                ON graph_assertions(notebook_id, subject, object);

            CREATE TABLE IF NOT EXISTS graph_assertion_evidence (
                assertion_id    TEXT NOT NULL,
                chunk_id        TEXT NOT NULL,
                document_sha256 TEXT NOT NULL,
                page            INTEGER NOT NULL DEFAULT 0,
                quote           TEXT,
                PRIMARY KEY (assertion_id, chunk_id)
            );
            CREATE INDEX IF NOT EXISTS graph_assertion_evidence_idx
                ON graph_assertion_evidence(assertion_id);",
        )?;
        Ok(())
    }

    /// Writes a claim, replacing the one with the same identity.
    ///
    /// A user review already recorded against this id survives an extractor
    /// re-proposing it: the status and the note are kept, and only the evidence
    /// and the revision are refreshed. Losing somebody's "no, that is wrong" to
    /// a rebuild would mean the same wrong claim comes back every time the
    /// documents are re-read.
    pub fn upsert_assertion(
        &self,
        notebook_id: &str,
        owner_user_id: &str,
        new: &NewAssertion,
    ) -> Result<String> {
        if self.get(notebook_id, owner_user_id)?.is_none() {
            anyhow::bail!("that notebook does not exist");
        }
        let mut conn = self.conn.lock().expect("notebook store lock poisoned");
        let tx = conn.transaction()?;
        let id = write_assertion(&tx, notebook_id, new)?;
        tx.commit()?;
        Ok(id)
    }

    /// Every claim in a notebook.
    ///
    /// `document_sha256` narrows to one source. `include_rejected` is off by
    /// default because a rejected claim is not part of the graph; the review
    /// screen turns it on, because a reviewer needs to see what they refused.
    pub fn assertions(
        &self,
        notebook_id: &str,
        owner_user_id: &str,
        document_sha256: Option<&str>,
        include_rejected: bool,
    ) -> Result<Vec<Assertion>> {
        if self.get(notebook_id, owner_user_id)?.is_none() {
            return Ok(Vec::new());
        }
        let conn = self.conn.lock().expect("notebook store lock poisoned");
        let mut statement = conn.prepare(
            "SELECT id, notebook_id, document_sha256, subject, subject_label, predicate,
                    object, object_label, provenance, status, direction_certain,
                    extractor, extractor_version, source_revision, stale, note,
                    created_at, updated_at, reviewed_at
               FROM graph_assertions
              WHERE notebook_id = ?1
                AND (?2 IS NULL OR document_sha256 = ?2)
                AND (?3 = 1 OR status <> 'rejected')
              ORDER BY subject_label ASC, predicate ASC, object_label ASC",
        )?;
        let rows: Vec<Assertion> = statement
            .query_map(
                params![notebook_id, document_sha256, include_rejected as i64],
                row_to_assertion,
            )?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        drop(statement);

        let mut out = Vec::with_capacity(rows.len());
        for mut assertion in rows {
            assertion.evidence = read_evidence(&conn, &assertion.id)?;
            out.push(assertion);
        }
        Ok(out)
    }

    /// One claim, or `None` when it does not exist or is not this person's.
    pub fn assertion(
        &self,
        notebook_id: &str,
        owner_user_id: &str,
        assertion_id: &str,
    ) -> Result<Option<Assertion>> {
        if self.get(notebook_id, owner_user_id)?.is_none() {
            return Ok(None);
        }
        let conn = self.conn.lock().expect("notebook store lock poisoned");
        let found: Option<Assertion> = conn
            .query_row(
                "SELECT id, notebook_id, document_sha256, subject, subject_label, predicate,
                        object, object_label, provenance, status, direction_certain,
                        extractor, extractor_version, source_revision, stale, note,
                        created_at, updated_at, reviewed_at
                   FROM graph_assertions
                  WHERE notebook_id = ?1 AND id = ?2",
                params![notebook_id, assertion_id],
                row_to_assertion,
            )
            .optional()?;
        match found {
            None => Ok(None),
            Some(mut assertion) => {
                assertion.evidence = read_evidence(&conn, &assertion.id)?;
                Ok(Some(assertion))
            }
        }
    }

    /// Records a person's decision about a claim.
    ///
    /// The row is not deleted on a rejection. See [`AssertionStatus`].
    pub fn review_assertion(
        &self,
        notebook_id: &str,
        owner_user_id: &str,
        assertion_id: &str,
        status: AssertionStatus,
        note: Option<&str>,
    ) -> Result<()> {
        if self.get(notebook_id, owner_user_id)?.is_none() {
            anyhow::bail!("that notebook does not exist");
        }
        let now = chrono::Utc::now().to_rfc3339();
        let conn = self.conn.lock().expect("notebook store lock poisoned");
        let changed = conn.execute(
            "UPDATE graph_assertions
                SET status = ?3, note = ?4, reviewed_at = ?5, updated_at = ?5
              WHERE notebook_id = ?1 AND id = ?2",
            params![notebook_id, assertion_id, status.as_str(), note, &now],
        )?;
        if changed == 0 {
            anyhow::bail!("that relationship is not in this notebook");
        }
        Ok(())
    }

    /// Corrects a claim's wording or direction.
    ///
    /// A correction is a person's assertion however it started life, so the
    /// provenance becomes `User` and the direction becomes certain — somebody
    /// has now said which way round it goes. The id changes with the claim,
    /// because an id derived from the terms would otherwise name something the
    /// row no longer says; the old row is deleted in the same transaction.
    pub fn correct_assertion(
        &self,
        notebook_id: &str,
        owner_user_id: &str,
        assertion_id: &str,
        subject: &str,
        predicate: &str,
        object: &str,
        note: Option<&str>,
    ) -> Result<String> {
        if self.get(notebook_id, owner_user_id)?.is_none() {
            anyhow::bail!("that notebook does not exist");
        }
        let predicate = predicate.trim();
        if predicate.is_empty() {
            anyhow::bail!("a relationship needs a name");
        }
        if subject == object {
            anyhow::bail!("a relationship needs two different terms");
        }

        let mut conn = self.conn.lock().expect("notebook store lock poisoned");
        let tx = conn.transaction()?;

        let existing: Option<Assertion> = tx
            .query_row(
                "SELECT id, notebook_id, document_sha256, subject, subject_label, predicate,
                        object, object_label, provenance, status, direction_certain,
                        extractor, extractor_version, source_revision, stale, note,
                        created_at, updated_at, reviewed_at
                   FROM graph_assertions
                  WHERE notebook_id = ?1 AND id = ?2",
                params![notebook_id, assertion_id],
                row_to_assertion,
            )
            .optional()?;
        let Some(existing) = existing else {
            anyhow::bail!("that relationship is not in this notebook");
        };

        // The labels follow the terms. Looked up from the graph rather than
        // carried from the old row, because the corrected subject may be a
        // different term entirely.
        let subject_label =
            label_for(&tx, notebook_id, subject)?.unwrap_or_else(|| subject.to_string());
        let object_label =
            label_for(&tx, notebook_id, object)?.unwrap_or_else(|| object.to_string());

        let evidence = read_evidence(&tx, assertion_id)?;
        let new = NewAssertion {
            document_sha256: existing.document_sha256.clone(),
            subject: subject.to_string(),
            subject_label,
            predicate: predicate.to_string(),
            object: object.to_string(),
            object_label,
            provenance: AssertionProvenance::User,
            status: AssertionStatus::Accepted,
            direction_certain: true,
            extractor: existing.extractor.clone(),
            extractor_version: existing.extractor_version,
            source_revision: existing.source_revision.clone(),
            evidence,
        };

        // The old row goes before the new one is written, so a correction that
        // happens to land back on the same id does not delete what it just
        // wrote.
        let next_id = assertion_id_of(notebook_id, &new);
        if assertion_id != next_id {
            tx.execute(
                "DELETE FROM graph_assertion_evidence WHERE assertion_id = ?1",
                params![assertion_id],
            )?;
            tx.execute(
                "DELETE FROM graph_assertions WHERE notebook_id = ?1 AND id = ?2",
                params![notebook_id, assertion_id],
            )?;
        }

        let id = write_assertion(&tx, notebook_id, &new)?;
        let now = chrono::Utc::now().to_rfc3339();
        // Forced after the write: `write_assertion` preserves an existing
        // review, and a correction *is* the new review.
        tx.execute(
            "UPDATE graph_assertions
                SET note = ?3, reviewed_at = ?4, updated_at = ?4,
                    provenance = 'user', status = 'accepted', direction_certain = 1
              WHERE notebook_id = ?1 AND id = ?2",
            params![notebook_id, &id, note, &now],
        )?;
        tx.commit()?;
        Ok(id)
    }

    /// Records a claim a person wrote themselves.
    ///
    /// Marked `User` provenance and `Accepted`, with no evidence — which is the
    /// honest description of it. A retrieval that uses it says it is unverified,
    /// because there is no passage to check it against.
    pub fn assert_by_hand(
        &self,
        notebook_id: &str,
        owner_user_id: &str,
        subject: &str,
        predicate: &str,
        object: &str,
        note: Option<&str>,
    ) -> Result<String> {
        if self.get(notebook_id, owner_user_id)?.is_none() {
            anyhow::bail!("that notebook does not exist");
        }
        let predicate = predicate.trim();
        if predicate.is_empty() {
            anyhow::bail!("a relationship needs a name");
        }
        if subject == object {
            anyhow::bail!("a relationship needs two different terms");
        }

        let mut conn = self.conn.lock().expect("notebook store lock poisoned");
        let tx = conn.transaction()?;
        let subject_label =
            label_for(&tx, notebook_id, subject)?.unwrap_or_else(|| subject.to_string());
        let object_label =
            label_for(&tx, notebook_id, object)?.unwrap_or_else(|| object.to_string());
        let new = NewAssertion {
            document_sha256: None,
            subject: subject.to_string(),
            subject_label,
            predicate: predicate.to_string(),
            object: object.to_string(),
            object_label,
            provenance: AssertionProvenance::User,
            status: AssertionStatus::Accepted,
            direction_certain: true,
            extractor: "hand".to_string(),
            extractor_version: 0,
            source_revision: None,
            evidence: Vec::new(),
        };
        let id = write_assertion(&tx, notebook_id, &new)?;
        let now = chrono::Utc::now().to_rfc3339();
        tx.execute(
            "UPDATE graph_assertions SET note = ?3, reviewed_at = ?4, updated_at = ?4
              WHERE notebook_id = ?1 AND id = ?2",
            params![notebook_id, &id, note, &now],
        )?;
        tx.commit()?;
        Ok(id)
    }

    /// Deletes a claim a person wrote. Model proposals are rejected, not
    /// deleted — see [`AssertionStatus`].
    pub fn delete_hand_assertion(
        &self,
        notebook_id: &str,
        owner_user_id: &str,
        assertion_id: &str,
    ) -> Result<()> {
        if self.get(notebook_id, owner_user_id)?.is_none() {
            anyhow::bail!("that notebook does not exist");
        }
        let mut conn = self.conn.lock().expect("notebook store lock poisoned");
        let tx = conn.transaction()?;
        let provenance: Option<String> = tx
            .query_row(
                "SELECT provenance FROM graph_assertions WHERE notebook_id = ?1 AND id = ?2",
                params![notebook_id, assertion_id],
                |row| row.get(0),
            )
            .optional()?;
        match provenance.as_deref() {
            None => anyhow::bail!("that relationship is not in this notebook"),
            Some("user") => {}
            Some(_) => anyhow::bail!(
                "That relationship came from an extraction pass. Reject it rather than deleting \
                 it, so the same claim is not proposed again with nothing to say it was refused."
            ),
        }
        tx.execute(
            "DELETE FROM graph_assertion_evidence WHERE assertion_id = ?1",
            params![assertion_id],
        )?;
        tx.execute(
            "DELETE FROM graph_assertions WHERE notebook_id = ?1 AND id = ?2",
            params![notebook_id, assertion_id],
        )?;
        tx.commit()?;
        Ok(())
    }
}

/// The id the store will give this claim.
fn assertion_id_of(notebook_id: &str, new: &NewAssertion) -> String {
    assertion_id(
        notebook_id,
        new.document_sha256.as_deref(),
        &new.subject,
        &new.predicate,
        &new.object,
    )
}

/// Reads a claim off a row. Shared by every query in this file so the column
/// order is written down once.
fn row_to_assertion(row: &rusqlite::Row<'_>) -> rusqlite::Result<Assertion> {
    Ok(Assertion {
        id: row.get(0)?,
        notebook_id: row.get(1)?,
        document_sha256: row.get(2)?,
        subject: row.get(3)?,
        subject_label: row.get(4)?,
        predicate: row.get(5)?,
        object: row.get(6)?,
        object_label: row.get(7)?,
        provenance: AssertionProvenance::parse(&row.get::<_, String>(8)?),
        status: AssertionStatus::parse(&row.get::<_, String>(9)?),
        direction_certain: row.get::<_, i64>(10)? != 0,
        extractor: row.get(11)?,
        extractor_version: row.get::<_, i64>(12)? as u32,
        source_revision: row.get(13)?,
        stale: row.get::<_, i64>(14)? != 0,
        note: row.get(15)?,
        created_at: row.get(16)?,
        updated_at: row.get(17)?,
        reviewed_at: row.get(18)?,
        evidence: Vec::new(),
    })
}

fn read_evidence(conn: &Connection, assertion_id: &str) -> Result<Vec<AssertionEvidence>> {
    let mut statement = conn.prepare(
        "SELECT chunk_id, document_sha256, page, quote
           FROM graph_assertion_evidence
          WHERE assertion_id = ?1
          ORDER BY document_sha256 ASC, page ASC, chunk_id ASC",
    )?;
    let rows = statement.query_map(params![assertion_id], |row| {
        Ok(AssertionEvidence {
            chunk_id: row.get(0)?,
            document_sha256: row.get(1)?,
            page: row.get::<_, i64>(2)? as u32,
            quote: row.get(3)?,
        })
    })?;
    rows.collect::<rusqlite::Result<Vec<_>>>()
        .context("the relationship's evidence could not be read")
}

fn label_for(conn: &Connection, notebook_id: &str, normalised: &str) -> Result<Option<String>> {
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

/// Inserts or refreshes one claim inside a caller's transaction.
///
/// Kept free-standing so [`NotebookStore::correct_assertion`] and the
/// extraction passes reach the same writer: two writers would be two chances
/// for a review to be dropped on one of the paths.
pub(super) fn write_assertion(
    conn: &Connection,
    notebook_id: &str,
    new: &NewAssertion,
) -> Result<String> {
    let id = assertion_id_of(notebook_id, new);
    let now = chrono::Utc::now().to_rfc3339();

    // What a person already decided about this exact claim. Read before the
    // write so a re-proposal cannot quietly reset it to `proposed`.
    let reviewed: Option<(String, String, Option<String>, Option<String>)> = conn
        .query_row(
            "SELECT status, provenance, note, reviewed_at FROM graph_assertions WHERE id = ?1",
            params![&id],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
        )
        .optional()?;

    let (status, provenance, note, reviewed_at) = match reviewed {
        Some((status, provenance, note, reviewed_at)) if reviewed_at.is_some() => {
            (status, provenance, note, reviewed_at)
        }
        _ => (
            new.status.as_str().to_string(),
            new.provenance.as_str().to_string(),
            None,
            None,
        ),
    };

    conn.execute(
        "INSERT INTO graph_assertions
             (id, notebook_id, document_sha256, subject, subject_label, predicate,
              object, object_label, provenance, status, direction_certain, extractor,
              extractor_version, source_revision, stale, note, created_at, updated_at,
              reviewed_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, 0, ?15,
                 ?16, ?16, ?17)
         ON CONFLICT(id) DO UPDATE SET
             subject_label     = excluded.subject_label,
             object_label      = excluded.object_label,
             provenance        = excluded.provenance,
             status            = excluded.status,
             direction_certain = excluded.direction_certain,
             extractor         = excluded.extractor,
             extractor_version = excluded.extractor_version,
             source_revision   = excluded.source_revision,
             stale             = 0,
             note              = excluded.note,
             updated_at        = excluded.updated_at,
             reviewed_at       = excluded.reviewed_at",
        params![
            &id,
            notebook_id,
            &new.document_sha256,
            &new.subject,
            &new.subject_label,
            &new.predicate,
            &new.object,
            &new.object_label,
            &provenance,
            &status,
            new.direction_certain as i64,
            &new.extractor,
            new.extractor_version as i64,
            &new.source_revision,
            &note,
            &now,
            &reviewed_at,
        ],
    )?;

    conn.execute(
        "DELETE FROM graph_assertion_evidence WHERE assertion_id = ?1",
        params![&id],
    )?;
    for evidence in &new.evidence {
        conn.execute(
            "INSERT OR REPLACE INTO graph_assertion_evidence
                 (assertion_id, chunk_id, document_sha256, page, quote)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                &id,
                &evidence.chunk_id,
                &evidence.document_sha256,
                evidence.page as i64,
                &evidence.quote
            ],
        )?;
    }

    Ok(id)
}
