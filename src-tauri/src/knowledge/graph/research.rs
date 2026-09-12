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
    /// The legacy source list. **Do not read this directly** — use
    /// [`ResearchScope::selection`].
    ///
    /// Kept only so manifests written before [`SourceSelection`] existed still
    /// deserialise. In those records an empty list meant "every source", which
    /// is exactly the ambiguity that made this field unsafe: a list that failed
    /// to load, a state that had not arrived yet, and a deliberate "all" were
    /// the same three bytes on the wire — and the widest of the three readings
    /// was the one that happened.
    #[serde(default)]
    pub source_sha256s: Vec<String>,
    /// What the person actually chose, said out loud.
    ///
    /// `None` only for a manifest written before this field existed. Every
    /// scope arriving over IPC must carry one, which
    /// [`ResearchScope::require_explicit_selection`] enforces at the boundary.
    #[serde(default)]
    pub sources: Option<SourceSelection>,
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

    /// What this scope selects, resolving a pre-[`SourceSelection`] record.
    ///
    /// A stored manifest with no `sources` field is read the way it was written
    /// — empty meant all — because rewriting history to a stricter reading
    /// would change what a past answer claims to have used. New scopes never
    /// reach this branch: see [`Self::require_explicit_selection`].
    pub fn selection(&self) -> SourceSelection {
        match &self.sources {
            Some(selection) => selection.clone(),
            None if self.source_sha256s.is_empty() => SourceSelection::All,
            None => SourceSelection::subset(self.source_sha256s.clone()),
        }
    }

    /// The selection, refusing a scope that did not state one.
    ///
    /// Called at every boundary where a scope arrives from outside Rust. This
    /// is the enforcement point for the rule that an absent selection is not a
    /// selection: a fetch that failed, a component that had not loaded yet, and
    /// a person who chose every source are three different situations, and only
    /// the last of them may read every source.
    pub fn require_explicit_selection(&self) -> Result<SourceSelection, String> {
        self.sources.clone().ok_or_else(|| {
            "This question did not say which sources it may use. Choose the sources on the \
             notebook chip — \"all\", some of them, or none — and ask again."
                .to_string()
        })
    }
}

/// Which of a notebook's sources one question may read.
///
/// Three cases, named, because the difference between them is the difference
/// between a right answer and a confident wrong one. The type exists so that
/// "no sources are selected" cannot be represented by the same value as "every
/// source is selected", which is what an empty `Vec<String>` did.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", tag = "mode")]
pub enum SourceSelection {
    /// Every source in the notebook, as it stands when the turn is frozen.
    ///
    /// Resolved to a concrete list at freeze time rather than carried as "all",
    /// so a source added after the question was queued does not join a turn
    /// that was scoped before it existed.
    All,
    /// These sources and no others.
    Subset { sha256s: Vec<String> },
    /// Deliberately none — the notebook is attached and no source may be read.
    ///
    /// A real choice, not an error: it is how somebody asks a question of the
    /// model with a notebook on screen and no evidence drawn from it. The turn
    /// refuses to claim notebook grounding rather than quietly using all of it.
    None,
}

impl SourceSelection {
    /// A subset, from a plain list. Convenience for callers holding a `Vec`.
    pub fn subset(sha256s: Vec<String>) -> Self {
        SourceSelection::Subset { sha256s }
    }

    /// The sha list, for a subset. `None` for the other two, which are not
    /// lists and must not be flattened into one.
    pub fn explicit_list(&self) -> Option<&[String]> {
        match self {
            SourceSelection::Subset { sha256s } => Some(sha256s),
            _ => None,
        }
    }

    /// Whether this selection permits reading anything at all.
    pub fn reads_anything(&self) -> bool {
        match self {
            SourceSelection::All => true,
            SourceSelection::Subset { sha256s } => !sha256s.is_empty(),
            SourceSelection::None => false,
        }
    }

    /// The phrase the chip and the answer both use.
    pub fn describe(&self, total: u32) -> String {
        match self {
            SourceSelection::All => format!("all {total} sources"),
            SourceSelection::Subset { sha256s } => {
                format!("{} of {total} sources", sha256s.len())
            }
            SourceSelection::None => "no sources".to_string(),
        }
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

/// Adds a column, ignoring only the error that means it is already there.
///
/// The same shape as `agent_runtime::events::idempotency::add_column_if_missing`
/// and for the same reason: SQLite has no `ADD COLUMN IF NOT EXISTS`, and a
/// blanket `let _ =` swallows a read-only database and a busy one alongside the
/// duplicate this is actually expecting.
fn add_column_if_missing(conn: &Connection, name: &str, decl: &str) -> Result<()> {
    match conn.execute_batch(&format!(
        "ALTER TABLE notebook_conversations ADD COLUMN {name} {decl};"
    )) {
        Ok(()) => Ok(()),
        Err(error) if error.to_string().contains("duplicate column name") => Ok(()),
        Err(error) => {
            Err(error).context("the notebook conversation table could not be migrated")
        }
    }
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

        // The source selection, remembered with the conversation.
        //
        // Added rather than baked into the CREATE above so an existing database
        // gains the column without losing its rows — this table already carries
        // every notebook thread somebody has ever had.
        //
        // Only the "already there" error is swallowed. Discarding every error
        // meant a read-only database, a `SQLITE_BUSY` under a concurrent open
        // or a corrupt schema all left this returning `Ok(())` with the store
        // looking healthy, and the failure surfaced later and elsewhere as
        // "no such column: selection_json" — reported to the person as a broken
        // notebook chip.
        add_column_if_missing(conn, "selection_json", "TEXT")?;
        add_column_if_missing(conn, "updated_at", "TEXT")?;
        Ok(())
    }

    /// Remembers which notebook a conversation is asking, and of what.
    ///
    /// The *preference*, not the scope of any turn. A turn's scope is frozen
    /// onto its manifest when the message is submitted and never moves again;
    /// this is only what the chip should show when the conversation is reopened,
    /// and what the next question will default to.
    ///
    /// Upserts, so switching notebooks mid-conversation replaces the preference
    /// and leaves every manifest exactly where it was.
    pub fn set_conversation_notebook(
        &self,
        notebook_id: &str,
        owner_user_id: &str,
        conversation_id: &str,
        selection: Option<&super::research::SourceSelection>,
    ) -> Result<()> {
        if self.get(notebook_id, owner_user_id)?.is_none() {
            anyhow::bail!("that notebook does not exist");
        }
        let selection_json = match selection {
            Some(selection) => Some(
                serde_json::to_string(selection)
                    .context("the source selection could not be stored")?,
            ),
            None => None,
        };
        let now = chrono::Utc::now().to_rfc3339();
        let conn = self.conn.lock().expect("notebook store lock poisoned");
        let changed = conn.execute(
            "INSERT INTO notebook_conversations
                 (conversation_id, notebook_id, owner_user_id, created_at, selection_json,
                  updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?4)
             ON CONFLICT(conversation_id) DO UPDATE SET
                 notebook_id    = excluded.notebook_id,
                 selection_json = excluded.selection_json,
                 updated_at     = excluded.updated_at
              WHERE notebook_conversations.owner_user_id = ?3",
            params![
                conversation_id,
                notebook_id,
                owner_user_id,
                &now,
                selection_json
            ],
        )?;
        if changed == 0 {
            // SQLite raises nothing when the `WHERE` on the upsert is false — it
            // simply changes no rows. Returning `Ok(())` on that told the screen
            // the notebook had been attached while the row belonged to somebody
            // else and nothing had been written.
            anyhow::bail!(
                "that conversation belongs to a different account, so its notebook was \
                 not changed"
            );
        }
        Ok(())
    }

    /// The notebook and selection a conversation was last left on.
    ///
    /// `Ok(None)` when the conversation has no notebook, and when it belongs to
    /// somebody else — the caller cannot tell those apart, which is the point.
    pub fn conversation_notebook_preference(
        &self,
        owner_user_id: &str,
        conversation_id: &str,
    ) -> Result<Option<(String, Option<super::research::SourceSelection>)>> {
        let conn = self.conn.lock().expect("notebook store lock poisoned");
        let row: Option<(String, Option<String>)> = conn
            .query_row(
                "SELECT notebook_id, selection_json FROM notebook_conversations
                  WHERE conversation_id = ?1 AND owner_user_id = ?2",
                params![conversation_id, owner_user_id],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()?;
        let Some((notebook_id, selection_json)) = row else {
            return Ok(None);
        };
        // A selection this build cannot read is dropped rather than guessed at.
        // The caller then has a notebook with no selection, which the chip shows
        // as "choose sources" — not as "all of them".
        //
        // Logged, because dropping it silently merges "this conversation
        // predates selections" with "this conversation's selection is corrupt on
        // disk", and only the second is a fault worth anybody knowing about.
        let selection = match selection_json.as_deref() {
            None => None,
            Some(json) => match serde_json::from_str(json) {
                Ok(selection) => Some(selection),
                Err(error) => {
                    log::warn!(
                        "[notebook] conversation {conversation_id}: the stored source \
                         selection could not be read ({error}); the chip will ask for the \
                         sources to be chosen again"
                    );
                    None
                }
            },
        };
        Ok(Some((notebook_id, selection)))
    }

    /// Forgets which notebook a conversation is asking, and nothing else.
    ///
    /// Deliberately not [`Self::unbind_conversation`], which also deletes every
    /// evidence manifest recorded against the conversation. Taking the chip off
    /// a thread must not destroy the evidence behind the answers already in it:
    /// those citations were true when they were written and stay resolvable.
    pub fn clear_conversation_notebook(
        &self,
        owner_user_id: &str,
        conversation_id: &str,
    ) -> Result<()> {
        let conn = self.conn.lock().expect("notebook store lock poisoned");
        conn.execute(
            "DELETE FROM notebook_conversations
              WHERE conversation_id = ?1 AND owner_user_id = ?2",
            params![conversation_id, owner_user_id],
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
        // Upsert with the owner guard, not `INSERT OR REPLACE`.
        //
        // `INSERT OR REPLACE` is DELETE-then-INSERT on a primary-key conflict,
        // and this statement carried no owner check — so naming somebody else's
        // conversation id destroyed their binding outright and replaced the
        // owner with the caller. It also rewrote the row without naming
        // `selection_json` or `updated_at`, so a re-bind silently discarded the
        // source selection the conversation was carrying.
        //
        // The selection is deliberately left alone here: this call records
        // *which notebook*, and a caller meaning to change the selection calls
        // `set_conversation_notebook`.
        let conn = self.conn.lock().expect("notebook store lock poisoned");
        let now = chrono::Utc::now().to_rfc3339();
        let changed = conn.execute(
            "INSERT INTO notebook_conversations
                 (conversation_id, notebook_id, owner_user_id, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?4)
             ON CONFLICT(conversation_id) DO UPDATE SET
                 notebook_id = excluded.notebook_id,
                 updated_at  = excluded.updated_at
              WHERE notebook_conversations.owner_user_id = ?3",
            params![conversation_id, notebook_id, owner_user_id, &now],
        )?;
        if changed == 0 {
            anyhow::bail!(
                "that conversation belongs to a different account, so its notebook was \
                 not changed"
            );
        }
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

#[cfg(test)]
mod selection_tests {
    use super::*;

    const SHA_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const SHA_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

    fn scope(sources: Option<SourceSelection>) -> ResearchScope {
        ResearchScope {
            notebook_id: "nb-1".into(),
            source_sha256s: Vec::new(),
            sources,
            node_ids: Vec::new(),
            assertion_ids: Vec::new(),
        }
    }

    /// The property the whole type exists for. Three selections, three values,
    /// none of which can be mistaken for another.
    #[test]
    fn all_a_subset_and_none_are_three_distinct_values() {
        let all = SourceSelection::All;
        let subset = SourceSelection::subset(vec![SHA_A.into()]);
        let none = SourceSelection::None;

        assert_ne!(all, subset);
        assert_ne!(subset, none);
        assert_ne!(all, none);

        assert!(all.reads_anything());
        assert!(subset.reads_anything());
        assert!(!none.reads_anything(), "none must never read anything");

        assert_eq!(all.explicit_list(), None, "'all' is not a list of sources");
        assert_eq!(none.explicit_list(), None, "'none' is not a list of sources");
        assert_eq!(subset.explicit_list().map(<[String]>::len), Some(1));
    }

    /// A scope that did not state a selection is refused, not widened.
    #[test]
    fn a_scope_with_no_selection_is_refused_rather_than_read_as_all() {
        let refusal = scope(None)
            .require_explicit_selection()
            .expect_err("a scope with no selection must be refused");
        assert!(
            refusal.contains("did not say which sources"),
            "the refusal must say what is missing: {refusal}"
        );
    }

    /// An explicitly empty subset is not "everything" either.
    #[test]
    fn an_empty_subset_is_not_the_same_as_all() {
        let empty = SourceSelection::subset(Vec::new());
        assert_ne!(empty, SourceSelection::All);
        assert!(!empty.reads_anything());
    }

    /// Choosing none survives the round trip. If `None` deserialised back as
    /// `All` — which it would with an untagged or defaulted representation —
    /// a question the person scoped to nothing would read the whole notebook.
    #[test]
    fn every_selection_survives_the_json_round_trip() {
        for selection in [
            SourceSelection::All,
            SourceSelection::subset(vec![SHA_A.into(), SHA_B.into()]),
            SourceSelection::None,
        ] {
            let json = serde_json::to_string(&scope(Some(selection.clone())))
                .expect("a scope serialises");
            let back: ResearchScope = serde_json::from_str(&json).expect("and reads back");
            assert_eq!(back.selection(), selection, "{json}");
            assert_eq!(
                back.require_explicit_selection().expect("it stated one"),
                selection
            );
        }
    }

    /// A manifest written before this field existed keeps its original
    /// meaning. Historical turns must not be re-read under a stricter rule —
    /// that would change what a past answer claims to have used.
    #[test]
    fn a_manifest_written_before_this_field_keeps_its_original_meaning() {
        let legacy_all: ResearchScope =
            serde_json::from_str(r#"{"notebookId":"nb-1","sourceSha256s":[]}"#)
                .expect("a legacy scope reads");
        assert_eq!(legacy_all.selection(), SourceSelection::All);

        let legacy_subset: ResearchScope = serde_json::from_str(&format!(
            r#"{{"notebookId":"nb-1","sourceSha256s":["{SHA_A}"]}}"#
        ))
        .expect("a legacy scope reads");
        assert_eq!(
            legacy_subset.selection(),
            SourceSelection::subset(vec![SHA_A.into()])
        );

        // …and it is still refused at the IPC boundary, because a *new*
        // request must state its selection even though an old record need not.
        assert!(legacy_all.require_explicit_selection().is_err());
    }

    #[test]
    fn the_description_names_what_was_chosen() {
        assert_eq!(SourceSelection::All.describe(6), "all 6 sources");
        assert_eq!(
            SourceSelection::subset(vec![SHA_A.into(), SHA_B.into()]).describe(6),
            "2 of 6 sources"
        );
        assert_eq!(SourceSelection::None.describe(6), "no sources");
    }
}
