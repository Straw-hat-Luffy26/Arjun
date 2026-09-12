//! What one research turn actually retrieved, kept after the turn ends.
//!
//! ## Why the answer is not enough
//!
//! An assistant answer carries `[E1]`, `[E2]` markers, and during the run those
//! resolve against [`crate::agent_runtime::retrieval::RunPassages`] — a table in
//! memory, keyed by run id, that is gone the moment the process exits. So a
//! citation was checkable exactly once: while the run that produced it was
//! still alive. Reopen the conversation tomorrow and `[E2]` is three characters
//! of text.
//!
//! A manifest is that table written down. It holds, for each marker, the chunk
//! the passage came from, the document's content address, the page, the heading
//! path, and the *extraction revision* the passage was read at. That last field
//! is what makes a citation honest a month later: the same document re-read at a
//! better OCR stop is a different text, and a manifest that only named the
//! document would quietly resolve an old citation against new words.
//!
//! ## Historical turns do not move
//!
//! A new turn retrieves against whatever the notebook holds now. A turn already
//! answered keeps the references it was built from, including references to
//! sources that have since been removed — which resolve to an explicit
//! "no longer in this notebook" rather than to nothing. Rewriting history to
//! match the present would make every past answer appear to cite the current
//! documents, which is the most convincing possible way to be wrong.

use anyhow::{Context, Result};
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};

use super::store::NotebookStore;

/// How the passages for a turn were found.
///
/// Reported to the person, never guessed. Calling a keyword scan "semantic
/// search" because a vector index was *intended* is the kind of claim this
/// repository has a standing rule against.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum RetrievalMode {
    /// Term-frequency ranking over the notebook's passages. No model involved.
    Keyword,
    /// Vector similarity against passages embedded by a local model.
    Semantic,
    /// Both, fused.
    Hybrid,
}

impl RetrievalMode {
    /// The sentence shown on the screen and given to the model.
    pub fn describe(self) -> &'static str {
        match self {
            RetrievalMode::Keyword => {
                "keyword search over the selected sources — no embedding model is loaded, so \
                 passages were ranked by the words they share with the question"
            }
            RetrievalMode::Semantic => {
                "vector search over the selected sources, using the local embedding model"
            }
            RetrievalMode::Hybrid => {
                "keyword and vector search over the selected sources, fused into one ranking"
            }
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            RetrievalMode::Keyword => "keyword",
            RetrievalMode::Semantic => "semantic",
            RetrievalMode::Hybrid => "hybrid",
        }
    }
}

/// What the person asked the turn to look at.
///
/// Ids only. The frontend names a notebook, some of its sources and — when the
/// question came from the graph — some nodes and claims. Everything those ids
/// stand for is resolved in Rust, against the store, for the signed-in owner.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResearchScope {
    pub notebook_id: String,
    /// The sources the answer may use. Empty means every source in the
    /// notebook — which is a choice the interface makes explicit, not a
    /// fallback that happens when a list fails to arrive.
    #[serde(default)]
    pub source_sha256s: Vec<String>,
    /// Graph nodes the question is focused on, if any.
    #[serde(default)]
    pub node_ids: Vec<String>,
    /// Directed claims the question is focused on, if any.
    #[serde(default)]
    pub assertion_ids: Vec<String>,
}

impl ResearchScope {
    /// Whether the graph narrowed this question.
    pub fn is_graph_focused(&self) -> bool {
        !self.node_ids.is_empty() || !self.assertion_ids.is_empty()
    }
}

/// One passage the turn retrieved, as it can be found again.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EvidenceEntry {
    /// The number the model was told to cite: `[E1]` is marker 1.
    pub marker: u32,
    pub chunk_id: String,
    pub document_sha256: String,
    pub document_name: String,
    pub page: u32,
    #[serde(default)]
    pub section_path: Vec<String>,
    /// The extraction this passage was read from. See the module docs.
    pub source_revision: String,
    /// The passage text as retrieved, so a reader can be shown what to look for
    /// even when the source has since changed.
    pub quote: String,
}

/// Everything needed to reconstruct what an answer was built on.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EvidenceManifest {
    pub run_id: String,
    pub notebook_id: String,
    pub conversation_id: String,
    pub message_id: String,
    pub created_at: String,
    pub scope: ResearchScope,
    pub retrieval_mode: RetrievalMode,
    pub entries: Vec<EvidenceEntry>,
    /// What the turn could not do, in sentences meant for a person: a source
    /// whose extraction is missing, a selection truncated by the context
    /// budget, a graph pass that has not run.
    #[serde(default)]
    pub limitations: Vec<String>,
    /// The newest assertion timestamp in the notebook when the turn ran, when
    /// graph context was used. Lets a later read say the graph has moved on.
    #[serde(default)]
    pub graph_revision: Option<String>,
    /// Sources in scope that contributed no passage to this answer.
    ///
    /// Named rather than counted: "three of your eight sources were not used"
    /// is the difference between an answer that read the collection and one
    /// that read part of it.
    #[serde(default)]
    pub sources_unused: Vec<String>,
}

impl NotebookStore {
    pub(super) fn prepare_research(conn: &Connection) -> Result<()> {
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS notebook_research_turns (
                run_id          TEXT PRIMARY KEY,
                notebook_id     TEXT NOT NULL,
                owner_user_id   TEXT NOT NULL,
                conversation_id TEXT NOT NULL,
                message_id      TEXT NOT NULL,
                created_at      TEXT NOT NULL,
                manifest_json   TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS notebook_research_turns_message_idx
                ON notebook_research_turns(conversation_id, message_id);
            CREATE INDEX IF NOT EXISTS notebook_research_turns_notebook_idx
                ON notebook_research_turns(notebook_id, owner_user_id);

            CREATE TABLE IF NOT EXISTS notebook_conversations (
                conversation_id TEXT PRIMARY KEY,
                notebook_id     TEXT NOT NULL,
                owner_user_id   TEXT NOT NULL,
                created_at      TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS notebook_conversations_notebook_idx
                ON notebook_conversations(notebook_id, owner_user_id);",
        )?;
        Ok(())
    }

    /// Writes one turn's manifest.
    ///
    /// Keyed by run id, so a re-run replaces its own record and cannot overwrite
    /// another turn's.
    pub fn record_research_turn(
        &self,
        owner_user_id: &str,
        manifest: &EvidenceManifest,
    ) -> Result<()> {
        if self.get(&manifest.notebook_id, owner_user_id)?.is_none() {
            anyhow::bail!("that notebook does not exist");
        }
        let json = serde_json::to_string(manifest)
            .context("the evidence manifest could not be serialised")?;
        let conn = self.conn.lock().expect("notebook store lock poisoned");
        conn.execute(
            "INSERT OR REPLACE INTO notebook_research_turns
                 (run_id, notebook_id, owner_user_id, conversation_id, message_id,
                  created_at, manifest_json)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                &manifest.run_id,
                &manifest.notebook_id,
                owner_user_id,
                &manifest.conversation_id,
                &manifest.message_id,
                &manifest.created_at,
                &json
            ],
        )?;
        Ok(())
    }

    /// The manifest for one assistant message, if that turn recorded one.
    ///
    /// Owner-scoped in the `WHERE`, like every other read here: a message id is
    /// guessable and a manifest carries passage text.
    pub fn research_turn(
        &self,
        owner_user_id: &str,
        conversation_id: &str,
        message_id: &str,
    ) -> Result<Option<EvidenceManifest>> {
        let conn = self.conn.lock().expect("notebook store lock poisoned");
        let json: Option<String> = conn
            .query_row(
                "SELECT manifest_json FROM notebook_research_turns
                  WHERE owner_user_id = ?1 AND conversation_id = ?2 AND message_id = ?3",
                params![owner_user_id, conversation_id, message_id],
                |row| row.get(0),
            )
            .optional()?;
        match json {
            None => Ok(None),
            Some(json) => Ok(Some(
                serde_json::from_str(&json)
                    .context("the stored evidence manifest could not be read")?,
            )),
        }
    }

    /// Binds a conversation to the notebook it was started from.
    ///
    /// Without this a notebook's chat is indistinguishable from any other
    /// thread, and reopening the notebook could not restore what was said in
    /// it. The row is owner-scoped so one person's binding cannot name
    /// another's conversation.
    pub fn bind_conversation(
        &self,
        notebook_id: &str,
        owner_user_id: &str,
        conversation_id: &str,
    ) -> Result<()> {
        if self.get(notebook_id, owner_user_id)?.is_none() {
            anyhow::bail!("that notebook does not exist");
        }
        let conn = self.conn.lock().expect("notebook store lock poisoned");
        conn.execute(
            "INSERT OR REPLACE INTO notebook_conversations
                 (conversation_id, notebook_id, owner_user_id, created_at)
             VALUES (?1, ?2, ?3, ?4)",
            params![
                conversation_id,
                notebook_id,
                owner_user_id,
                chrono::Utc::now().to_rfc3339()
            ],
        )?;
        Ok(())
    }

    /// The conversations started from one notebook, newest first.
    pub fn notebook_conversations(
        &self,
        notebook_id: &str,
        owner_user_id: &str,
    ) -> Result<Vec<String>> {
        if self.get(notebook_id, owner_user_id)?.is_none() {
            return Ok(Vec::new());
        }
        let conn = self.conn.lock().expect("notebook store lock poisoned");
        let mut statement = conn.prepare(
            "SELECT conversation_id FROM notebook_conversations
              WHERE notebook_id = ?1 AND owner_user_id = ?2
              ORDER BY created_at DESC",
        )?;
        let rows = statement.query_map(params![notebook_id, owner_user_id], |row| {
            row.get::<_, String>(0)
        })?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .context("the notebook's conversations could not be read")
    }

    /// Which notebook a conversation belongs to, if any.
    pub fn conversation_notebook(
        &self,
        owner_user_id: &str,
        conversation_id: &str,
    ) -> Result<Option<String>> {
        let conn = self.conn.lock().expect("notebook store lock poisoned");
        Ok(conn
            .query_row(
                "SELECT notebook_id FROM notebook_conversations
                  WHERE conversation_id = ?1 AND owner_user_id = ?2",
                params![conversation_id, owner_user_id],
                |row| row.get::<_, String>(0),
            )
            .optional()?)
    }

    /// Forgets a conversation's binding, for when the conversation is deleted.
    pub fn unbind_conversation(&self, owner_user_id: &str, conversation_id: &str) -> Result<()> {
        let conn = self.conn.lock().expect("notebook store lock poisoned");
        conn.execute(
            "DELETE FROM notebook_conversations
              WHERE conversation_id = ?1 AND owner_user_id = ?2",
            params![conversation_id, owner_user_id],
        )?;
        conn.execute(
            "DELETE FROM notebook_research_turns
              WHERE conversation_id = ?1 AND owner_user_id = ?2",
            params![conversation_id, owner_user_id],
        )?;
        Ok(())
    }

    /// The newest change to any claim in this notebook.
    ///
    /// Recorded on a manifest so a later read can tell whether the graph has
    /// moved since the answer was written.
    pub fn graph_revision(&self, notebook_id: &str, owner_user_id: &str) -> Result<Option<String>> {
        if self.get(notebook_id, owner_user_id)?.is_none() {
            return Ok(None);
        }
        let conn = self.conn.lock().expect("notebook store lock poisoned");
        Ok(conn
            .query_row(
                "SELECT MAX(updated_at) FROM graph_assertions WHERE notebook_id = ?1",
                params![notebook_id],
                |row| row.get::<_, Option<String>>(0),
            )
            .optional()?
            .flatten())
    }
}
