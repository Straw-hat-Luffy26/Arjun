//! Saved notes and generated reports, owned by a notebook.
//!
//! ## Three kinds of text, never merged
//!
//! A research notebook accumulates three different things that all look like
//! prose on the screen, and confusing them is how a draft becomes a lie:
//!
//! - what a **person** wrote, which is nobody's claim but theirs,
//! - what a **model** produced from retrieved passages, which carries citations
//!   that resolve to real evidence, and
//! - the **passages themselves**, which are what the documents say.
//!
//! [`NoteKind`] records which a note started as, and [`Note::edited_since_saved`]
//! records whether the text has been changed since. Both are stored, because the
//! dangerous case is the quiet one: somebody saves a cited answer, edits a
//! sentence into it, and the note still displays the verification banner it
//! earned before the edit. A note that has been touched says so, and the banner
//! changes with it.
//!
//! The evidence manifest travels with the note rather than being re-derived, so
//! reopening a saved answer a month later shows the passages that answer was
//! actually built on — including the ones whose source has since been removed,
//! which show as unavailable rather than silently disappearing.

use anyhow::{Context, Result};
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::store::NotebookStore;

/// Where a note's text came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum NoteKind {
    /// A person wrote it.
    Written,
    /// An assistant answer, saved with the evidence it cited.
    Answer,
    /// A model-written summary of the notebook's sources.
    Summary,
    /// A model-written comparison across chosen sources or entities.
    Comparison,
}

impl NoteKind {
    pub fn as_str(self) -> &'static str {
        match self {
            NoteKind::Written => "written",
            NoteKind::Answer => "answer",
            NoteKind::Summary => "summary",
            NoteKind::Comparison => "comparison",
        }
    }

    pub fn parse(raw: &str) -> Self {
        match raw {
            "answer" => NoteKind::Answer,
            "summary" => NoteKind::Summary,
            "comparison" => NoteKind::Comparison,
            _ => NoteKind::Written,
        }
    }

    /// Whether a model wrote the text this kind of note starts with.
    pub fn is_model_written(self) -> bool {
        !matches!(self, NoteKind::Written)
    }
}

/// One note or report in a notebook.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Note {
    pub id: String,
    pub notebook_id: String,
    pub title: String,
    pub body: String,
    pub kind: NoteKind,
    /// The turn this was saved from, when it was saved from one.
    pub source_conversation_id: Option<String>,
    pub source_message_id: Option<String>,
    /// The evidence the text was built on, as JSON produced by
    /// [`super::research::EvidenceManifest`]. Kept verbatim so a saved answer
    /// can still be opened at its passages.
    pub evidence_json: Option<String>,
    /// True when the body no longer matches what was saved.
    ///
    /// The point of the field: a cited answer somebody has since typed into is
    /// no longer wholly a cited answer, and the interface must stop presenting
    /// it as one.
    pub edited_since_saved: bool,
    pub created_at: String,
    pub updated_at: String,
}

impl NotebookStore {
    pub(super) fn prepare_notes(conn: &Connection) -> Result<()> {
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS notebook_notes (
                id                     TEXT PRIMARY KEY,
                notebook_id            TEXT NOT NULL,
                owner_user_id          TEXT NOT NULL,
                title                  TEXT NOT NULL,
                body                   TEXT NOT NULL,
                kind                   TEXT NOT NULL,
                source_conversation_id TEXT,
                source_message_id      TEXT,
                evidence_json          TEXT,
                saved_body_sha256      TEXT,
                created_at             TEXT NOT NULL,
                updated_at             TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS notebook_notes_notebook_idx
                ON notebook_notes(notebook_id, owner_user_id);",
        )?;
        Ok(())
    }

    /// Every note in a notebook, most recently changed first.
    pub fn notes(&self, notebook_id: &str, owner_user_id: &str) -> Result<Vec<Note>> {
        if self.get(notebook_id, owner_user_id)?.is_none() {
            return Ok(Vec::new());
        }
        let conn = self.conn.lock().expect("notebook store lock poisoned");
        let mut statement = conn.prepare(
            "SELECT id, notebook_id, title, body, kind, source_conversation_id,
                    source_message_id, evidence_json, saved_body_sha256, created_at, updated_at
               FROM notebook_notes
              WHERE notebook_id = ?1 AND owner_user_id = ?2
              ORDER BY updated_at DESC",
        )?;
        let rows = statement.query_map(params![notebook_id, owner_user_id], row_to_note)?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .context("the notebook's notes could not be read")
    }

    /// One note, or `None` when it is not this person's.
    pub fn note(
        &self,
        notebook_id: &str,
        owner_user_id: &str,
        note_id: &str,
    ) -> Result<Option<Note>> {
        let conn = self.conn.lock().expect("notebook store lock poisoned");
        Ok(conn
            .query_row(
                "SELECT id, notebook_id, title, body, kind, source_conversation_id,
                        source_message_id, evidence_json, saved_body_sha256, created_at, updated_at
                   FROM notebook_notes
                  WHERE notebook_id = ?1 AND owner_user_id = ?2 AND id = ?3",
                params![notebook_id, owner_user_id, note_id],
                row_to_note,
            )
            .optional()?)
    }

    /// Writes a new note.
    ///
    /// `evidence_json` is the manifest the text was built from, or `None` for
    /// something a person typed. The body's hash is recorded alongside it, which
    /// is what lets a later read say whether the text still matches the
    /// evidence it arrived with.
    #[allow(clippy::too_many_arguments)]
    pub fn create_note(
        &self,
        notebook_id: &str,
        owner_user_id: &str,
        title: &str,
        body: &str,
        kind: NoteKind,
        source_conversation_id: Option<&str>,
        source_message_id: Option<&str>,
        evidence_json: Option<&str>,
    ) -> Result<Note> {
        if self.get(notebook_id, owner_user_id)?.is_none() {
            anyhow::bail!("that notebook does not exist");
        }
        let title = title.trim();
        if title.is_empty() {
            anyhow::bail!("a note needs a title");
        }

        let id = format!("note-{}", uuid::Uuid::new_v4());
        let now = chrono::Utc::now().to_rfc3339();
        // Recorded only for text a model produced. A note somebody typed has no
        // verified form to diverge from, so there is nothing to compare against
        // and "edited" would be meaningless on it.
        let digest = kind.is_model_written().then(|| body_digest(body));
        {
            let conn = self.conn.lock().expect("notebook store lock poisoned");
            conn.execute(
                "INSERT INTO notebook_notes
                     (id, notebook_id, owner_user_id, title, body, kind,
                      source_conversation_id, source_message_id, evidence_json,
                      saved_body_sha256, created_at, updated_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?11)",
                params![
                    &id,
                    notebook_id,
                    owner_user_id,
                    title,
                    body,
                    kind.as_str(),
                    source_conversation_id,
                    source_message_id,
                    evidence_json,
                    &digest,
                    &now
                ],
            )?;
        }
        self.note(notebook_id, owner_user_id, &id)?
            .ok_or_else(|| anyhow::anyhow!("the note could not be read back"))
    }

    /// Changes a note's title, its body, or both.
    ///
    /// The saved-body hash is deliberately *not* refreshed. Editing the text of
    /// a cited answer does not re-verify it — no passage was re-read and no
    /// citation was re-resolved — so the note keeps its manifest and starts
    /// reporting that it has been edited since. Silently re-stamping it as
    /// verified is the failure this field exists to prevent.
    pub fn update_note(
        &self,
        notebook_id: &str,
        owner_user_id: &str,
        note_id: &str,
        title: Option<&str>,
        body: Option<&str>,
    ) -> Result<Note> {
        if title.is_none() && body.is_none() {
            return self
                .note(notebook_id, owner_user_id, note_id)?
                .ok_or_else(|| anyhow::anyhow!("that note is not in this notebook"));
        }
        if let Some(title) = title {
            if title.trim().is_empty() {
                anyhow::bail!("a note needs a title");
            }
        }

        let now = chrono::Utc::now().to_rfc3339();
        let changed = {
            let conn = self.conn.lock().expect("notebook store lock poisoned");
            conn.execute(
                "UPDATE notebook_notes
                    SET title = COALESCE(?4, title),
                        body  = COALESCE(?5, body),
                        updated_at = ?6
                  WHERE notebook_id = ?1 AND owner_user_id = ?2 AND id = ?3",
                params![
                    notebook_id,
                    owner_user_id,
                    note_id,
                    title.map(str::trim),
                    body,
                    &now
                ],
            )?
        };
        if changed == 0 {
            anyhow::bail!("that note is not in this notebook");
        }
        self.note(notebook_id, owner_user_id, note_id)?
            .ok_or_else(|| anyhow::anyhow!("the note could not be read back"))
    }

    pub fn delete_note(
        &self,
        notebook_id: &str,
        owner_user_id: &str,
        note_id: &str,
    ) -> Result<()> {
        let conn = self.conn.lock().expect("notebook store lock poisoned");
        let removed = conn.execute(
            "DELETE FROM notebook_notes
              WHERE notebook_id = ?1 AND owner_user_id = ?2 AND id = ?3",
            params![notebook_id, owner_user_id, note_id],
        )?;
        if removed == 0 {
            anyhow::bail!("that note is not in this notebook");
        }
        Ok(())
    }
}

/// The hash a note's text is compared against.
fn body_digest(body: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(body.as_bytes());
    format!("{:x}", hasher.finalize())
}

fn row_to_note(row: &rusqlite::Row<'_>) -> rusqlite::Result<Note> {
    let body: String = row.get(3)?;
    let saved: Option<String> = row.get(8)?;
    Ok(Note {
        id: row.get(0)?,
        notebook_id: row.get(1)?,
        title: row.get(2)?,
        kind: NoteKind::parse(&row.get::<_, String>(4)?),
        source_conversation_id: row.get(5)?,
        source_message_id: row.get(6)?,
        evidence_json: row.get(7)?,
        edited_since_saved: saved
            .as_deref()
            .map(|digest| digest != body_digest(&body))
            .unwrap_or(false),
        created_at: row.get(9)?,
        updated_at: row.get(10)?,
        body,
    })
}
