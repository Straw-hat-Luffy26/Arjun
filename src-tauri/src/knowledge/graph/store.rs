//! Notebooks and their membership.
//!
//! Lives in `sarathi.db` alongside the prose and multimodal indexes, opened the
//! way [`crate::knowledge::multimodal::MultimodalIndex`] opens it: this module
//! creates its own tables in its own `prepare`, because there is no shared
//! migration runner for that database and inventing one here would be a second
//! opinion about a schema five other subsystems already manage themselves.
//!
//! ## The table names are prefixed on purpose
//!
//! `sarathi.db` already contains a `documents` table created *twice*, with two
//! incompatible schemas, by `crate::documents::store` and by
//! [`crate::knowledge::multimodal`] — whichever opens first wins and the other's
//! inserts fail. It is latent today only because one of the two is never
//! constructed outside tests. `projects` and `memory_nodes` are likewise taken,
//! by `crate::memory_engine::persistence`.
//!
//! Every table here is therefore prefixed `notebook_`, and the unprefixed nouns
//! a document library naturally wants — `documents`, `projects` — are avoided.

use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};

/// A named library of documents.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Notebook {
    pub id: String,
    pub name: String,
    /// RFC 3339 UTC, matching `ExtractedDocument::extracted_at`.
    pub created_at: String,
    pub updated_at: String,
    /// Documents currently in the notebook.
    ///
    /// Carried on the summary because every surface that lists notebooks wants
    /// it, and a second round trip per row to fetch it would be the whole cost
    /// of the list.
    pub document_count: u32,
}

/// One document's membership of a notebook.
///
/// The name is stored rather than looked up because the notebook has to be
/// listable when the extraction behind it cannot be read — a document whose JSON
/// is missing should show as a row with a name and a problem, not vanish from a
/// library the person can see they added to.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NotebookDocument {
    pub notebook_id: String,
    pub document_sha256: String,
    pub document_name: String,
    pub added_at: String,
}

/// Longest notebook name accepted.
///
/// Not a security boundary — a title bar that a 4 KB name pushes off the screen
/// is a usability problem, and refusing it here is cheaper than truncating it in
/// three different components.
const MAX_NAME_CHARS: usize = 120;

pub struct NotebookStore {
    /// Shared with [`super::persist`], which owns the graph tables in the same
    /// database and the same connection.
    pub(super) conn: Arc<Mutex<Connection>>,
}

impl NotebookStore {
    pub fn open(app_data_dir: &Path) -> Result<Self> {
        std::fs::create_dir_all(app_data_dir)?;
        let conn = Connection::open(app_data_dir.join("sarathi.db"))
            .context("could not open the notebook store")?;
        Self::prepare(&conn)?;
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    /// Opens an in-memory store, for tests.
    #[cfg(test)]
    pub fn in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()?;
        Self::prepare(&conn)?;
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    fn prepare(conn: &Connection) -> Result<()> {
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS notebooks (
                id              TEXT PRIMARY KEY,
                name            TEXT NOT NULL,
                owner_user_id   TEXT NOT NULL,
                created_at      TEXT NOT NULL,
                updated_at      TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS notebooks_owner_idx
                ON notebooks(owner_user_id);

            CREATE TABLE IF NOT EXISTS notebook_documents (
                notebook_id     TEXT NOT NULL,
                document_sha256 TEXT NOT NULL,
                document_name   TEXT NOT NULL,
                added_at        TEXT NOT NULL,
                PRIMARY KEY (notebook_id, document_sha256)
            );
            CREATE INDEX IF NOT EXISTS notebook_documents_notebook_idx
                ON notebook_documents(notebook_id);",
        )?;
        // The graph tables live beside these, created in the same open so a
        // store handed to a caller is always complete.
        Self::prepare_graph(conn)?;
        Self::prepare_assertions(conn)?;
        Self::prepare_notes(conn)?;
        Self::prepare_research(conn)?;
        // Runs after every table exists, because it moves rows between them.
        Self::migrate_graph(conn)?;
        Ok(())
    }

    /// Creates a notebook owned by this person.
    pub fn create(&self, owner_user_id: &str, name: &str) -> Result<Notebook> {
        let name = name.trim();
        if name.is_empty() {
            anyhow::bail!("a notebook needs a name");
        }
        if name.chars().count() > MAX_NAME_CHARS {
            anyhow::bail!("that name is longer than {MAX_NAME_CHARS} characters");
        }

        let now = chrono::Utc::now().to_rfc3339();
        let id = uuid::Uuid::new_v4().to_string();
        let conn = self.conn.lock().expect("notebook store lock poisoned");
        conn.execute(
            "INSERT INTO notebooks (id, name, owner_user_id, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?4)",
            params![&id, name, owner_user_id, &now],
        )?;

        Ok(Notebook {
            id,
            name: name.to_string(),
            created_at: now.clone(),
            updated_at: now,
            document_count: 0,
        })
    }

    /// Every notebook this person owns, most recently touched first.
    /// Renames a notebook.
    ///
    /// Owner-scoped in the `WHERE`, not checked and then written: a check
    /// followed by a write is two statements a second connection can interleave
    /// with, and the whole point of the clause is that somebody else's notebook
    /// is not reachable from here at all.
    pub fn rename(&self, id: &str, owner_user_id: &str, name: &str) -> Result<Notebook> {
        let name = name.trim();
        if name.is_empty() {
            anyhow::bail!("a notebook needs a name");
        }
        if name.chars().count() > MAX_NAME_CHARS {
            anyhow::bail!("that name is longer than {MAX_NAME_CHARS} characters");
        }

        let now = chrono::Utc::now().to_rfc3339();
        let changed = {
            let conn = self.conn.lock().expect("notebook store lock poisoned");
            conn.execute(
                "UPDATE notebooks SET name = ?1, updated_at = ?2
                  WHERE id = ?3 AND owner_user_id = ?4",
                params![name, &now, id, owner_user_id],
            )?
        };
        if changed == 0 {
            anyhow::bail!("no such notebook");
        }
        self.get(id, owner_user_id)?
            .ok_or_else(|| anyhow::anyhow!("no such notebook"))
    }

    /// Deletes a notebook and everything built over it.
    ///
    /// The graph tables carry `notebook_id` but no foreign key, so nothing
    /// removes their rows on their own. Left behind they are not merely
    /// clutter: ids are uuids and will not be reused, but a later `graph` read
    /// for a *recreated* notebook of the same name would be answered from
    /// whichever rows happened to remain. So the delete is explicit, ordered
    /// children-first, and wrapped in one transaction - a half-deleted notebook
    /// is a graph with no notebook to explain it.
    ///
    /// The documents themselves are untouched. A notebook is a way of grouping
    /// files, not the only place they live, and deleting the grouping must not
    /// delete somebody's source material.
    pub fn delete(&self, id: &str, owner_user_id: &str) -> Result<()> {
        let mut conn = self.conn.lock().expect("notebook store lock poisoned");
        let transaction = conn.transaction()?;

        // Owner-scoped first, so a notebook belonging to somebody else leaves
        // this function before any child row is touched.
        let owned: i64 = transaction.query_row(
            "SELECT COUNT(*) FROM notebooks WHERE id = ?1 AND owner_user_id = ?2",
            params![id, owner_user_id],
            |row| row.get(0),
        )?;
        if owned == 0 {
            anyhow::bail!("no such notebook");
        }

        // The evidence of a claim is keyed by the claim's id, not by the
        // notebook, so it is deleted through its parent before the parent goes.
        transaction.execute(
            "DELETE FROM graph_assertion_evidence
              WHERE assertion_id IN (SELECT id FROM graph_assertions WHERE notebook_id = ?1)",
            params![id],
        )?;

        for table in [
            "graph_evidence",
            "graph_edges",
            "graph_nodes",
            "graph_build_units",
            "graph_assertions",
            "notebook_notes",
            "notebook_research_turns",
            "notebook_documents",
        ] {
            transaction.execute(
                &format!("DELETE FROM {table} WHERE notebook_id = ?1"),
                params![id],
            )?;
        }
        transaction.execute("DELETE FROM notebooks WHERE id = ?1", params![id])?;
        transaction.commit()?;
        Ok(())
    }

    /// Takes one document out of a notebook, and everything derived from it.
    ///
    /// ## What the old version left behind
    ///
    /// It deleted the membership row, the evidence and the build unit — and
    /// left `graph_nodes` and `graph_edges` untouched. So the term a removed
    /// document was the only source of stayed on the canvas, kept its
    /// occurrence count, and was still offered for selection and for export:
    /// a graph asserting something whose only citation had just been taken
    /// away, with the "why do you think that?" panel now empty. Worse, a term
    /// removed from the *documents* went on steering retrieval.
    ///
    /// ## Why deleting this document's rows is enough
    ///
    /// Every graph row is keyed by `(notebook, document, term)` — see
    /// [`super::persist`] — and the notebook's view is the `GROUP BY` over
    /// them. So removing this document's rows removes exactly this document's
    /// contribution: a term two files supported keeps the other file's row and
    /// stays in the graph with its count reduced by what this one gave it, and
    /// a term only this file supported has no rows left and is gone. There is
    /// no orphan sweep to get wrong, because there is no shared row to orphan.
    ///
    /// Other notebooks are untouched: `notebook_id` is in every `WHERE`. The
    /// document itself is untouched too — the extraction stays in the
    /// [`crate::agent_runtime::documents::DocumentStore`], because the same
    /// bytes may be in another notebook or in somebody's chat.
    pub fn remove_document(
        &self,
        notebook_id: &str,
        owner_user_id: &str,
        document_sha256: &str,
    ) -> Result<()> {
        let mut conn = self.conn.lock().expect("notebook store lock poisoned");
        let transaction = conn.transaction()?;

        let owned: i64 = transaction.query_row(
            "SELECT COUNT(*) FROM notebooks WHERE id = ?1 AND owner_user_id = ?2",
            params![notebook_id, owner_user_id],
            |row| row.get(0),
        )?;
        if owned == 0 {
            anyhow::bail!("no such notebook");
        }

        let removed = transaction.execute(
            "DELETE FROM notebook_documents
              WHERE notebook_id = ?1 AND document_sha256 = ?2",
            params![notebook_id, document_sha256],
        )?;
        if removed == 0 {
            anyhow::bail!("that document is not in this notebook");
        }

        super::persist::purge_document_contributions(
            &transaction,
            notebook_id,
            document_sha256,
        )?;

        let now = chrono::Utc::now().to_rfc3339();
        transaction.execute(
            "UPDATE notebooks SET updated_at = ?2 WHERE id = ?1",
            params![notebook_id, &now],
        )?;
        transaction.commit()?;
        Ok(())
    }

    pub fn list(&self, owner_user_id: &str) -> Result<Vec<Notebook>> {
        let conn = self.conn.lock().expect("notebook store lock poisoned");
        let mut statement = conn.prepare(
            "SELECT n.id, n.name, n.created_at, n.updated_at,
                    (SELECT COUNT(*) FROM notebook_documents d
                      WHERE d.notebook_id = n.id) AS document_count
               FROM notebooks n
              WHERE n.owner_user_id = ?1
              ORDER BY n.updated_at DESC",
        )?;
        let rows = statement.query_map(params![owner_user_id], |row| {
            Ok(Notebook {
                id: row.get(0)?,
                name: row.get(1)?,
                created_at: row.get(2)?,
                updated_at: row.get(3)?,
                document_count: row.get::<_, i64>(4)? as u32,
            })
        })?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .context("the notebook list could not be read")
    }

    /// One notebook, or `None` when it does not exist *or* belongs to somebody
    /// else. The caller cannot tell those apart, which is deliberate: a
    /// distinguishable "not yours" confirms the id names something real.
    pub fn get(&self, id: &str, owner_user_id: &str) -> Result<Option<Notebook>> {
        Ok(self
            .list(owner_user_id)?
            .into_iter()
            .find(|notebook| notebook.id == id))
    }

    /// Adds a document to a notebook, if this person owns the notebook.
    ///
    /// The ownership check is the `SELECT … WHERE owner_user_id = ?` feeding the
    /// insert, so a notebook id belonging to somebody else inserts no row rather
    /// than being caught by a separate `if` that a later edit could drop.
    ///
    /// Returns whether a new membership row was written; adding a document twice
    /// is not an error, because re-adding a file the notebook already has is a
    /// reasonable thing to do and failing it would be noise.
    pub fn add_document(
        &self,
        notebook_id: &str,
        owner_user_id: &str,
        document_sha256: &str,
        document_name: &str,
    ) -> Result<bool> {
        let now = chrono::Utc::now().to_rfc3339();
        let conn = self.conn.lock().expect("notebook store lock poisoned");
        let inserted = conn.execute(
            "INSERT OR IGNORE INTO notebook_documents
                    (notebook_id, document_sha256, document_name, added_at)
             SELECT ?1, ?2, ?3, ?4
               FROM notebooks
              WHERE id = ?1 AND owner_user_id = ?5",
            params![
                notebook_id,
                document_sha256,
                document_name,
                &now,
                owner_user_id
            ],
        )?;

        if inserted > 0 {
            conn.execute(
                "UPDATE notebooks SET updated_at = ?2
                  WHERE id = ?1 AND owner_user_id = ?3",
                params![notebook_id, &now, owner_user_id],
            )?;
        }
        Ok(inserted > 0)
    }

    /// The documents in a notebook this person owns.
    ///
    /// An empty vector for a notebook that is not theirs, for the same reason
    /// [`Self::get`] returns `None`.
    pub fn documents(
        &self,
        notebook_id: &str,
        owner_user_id: &str,
    ) -> Result<Vec<NotebookDocument>> {
        let conn = self.conn.lock().expect("notebook store lock poisoned");
        let mut statement = conn.prepare(
            "SELECT d.notebook_id, d.document_sha256, d.document_name, d.added_at
               FROM notebook_documents d
               JOIN notebooks n ON n.id = d.notebook_id
              WHERE d.notebook_id = ?1 AND n.owner_user_id = ?2
              ORDER BY d.added_at ASC, d.document_sha256 ASC",
        )?;
        let rows = statement.query_map(params![notebook_id, owner_user_id], |row| {
            Ok(NotebookDocument {
                notebook_id: row.get(0)?,
                document_sha256: row.get(1)?,
                document_name: row.get(2)?,
                added_at: row.get(3)?,
            })
        })?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .context("the notebook's documents could not be read")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALICE: &str = "user-alice";
    const BOB: &str = "user-bob";

    #[test]
    fn a_new_notebook_starts_empty() {
        let store = NotebookStore::in_memory().unwrap();
        let notebook = store.create(ALICE, "Turbine overhaul").unwrap();
        assert_eq!(notebook.name, "Turbine overhaul");
        assert_eq!(notebook.document_count, 0);
        assert_eq!(notebook.created_at, notebook.updated_at);
    }

    #[test]
    fn a_name_is_trimmed_and_must_not_be_blank() {
        let store = NotebookStore::in_memory().unwrap();
        assert_eq!(store.create(ALICE, "  Spaced  ").unwrap().name, "Spaced");
        assert!(store.create(ALICE, "   ").is_err());
    }

    #[test]
    fn one_person_cannot_see_another_persons_notebooks() {
        let store = NotebookStore::in_memory().unwrap();
        store.create(ALICE, "Alice's contracts").unwrap();

        assert_eq!(store.list(BOB).unwrap().len(), 0);
        assert_eq!(store.list(ALICE).unwrap().len(), 1);
    }

    #[test]
    fn one_person_cannot_read_another_persons_notebook_by_id() {
        let store = NotebookStore::in_memory().unwrap();
        let alices = store.create(ALICE, "Alice's contracts").unwrap();

        assert!(store.get(&alices.id, BOB).unwrap().is_none());
        assert!(store.get(&alices.id, ALICE).unwrap().is_some());
    }

    #[test]
    fn one_person_cannot_add_documents_to_another_persons_notebook() {
        let store = NotebookStore::in_memory().unwrap();
        let alices = store.create(ALICE, "Alice's contracts").unwrap();

        // Bob's write is refused by the SQL, not by a branch above it.
        assert!(!store
            .add_document(&alices.id, BOB, "sha-1", "leak.pdf")
            .unwrap());
        assert_eq!(store.documents(&alices.id, ALICE).unwrap().len(), 0);
    }

    #[test]
    fn one_person_cannot_list_another_persons_documents() {
        let store = NotebookStore::in_memory().unwrap();
        let alices = store.create(ALICE, "Alice's contracts").unwrap();
        store
            .add_document(&alices.id, ALICE, "sha-1", "contract.pdf")
            .unwrap();

        assert_eq!(store.documents(&alices.id, BOB).unwrap().len(), 0);
        assert_eq!(store.documents(&alices.id, ALICE).unwrap().len(), 1);
    }

    #[test]
    fn adding_the_same_document_twice_is_not_an_error_and_does_not_duplicate() {
        let store = NotebookStore::in_memory().unwrap();
        let notebook = store.create(ALICE, "Site A").unwrap();

        assert!(store
            .add_document(&notebook.id, ALICE, "sha-1", "pump.pdf")
            .unwrap());
        assert!(!store
            .add_document(&notebook.id, ALICE, "sha-1", "pump.pdf")
            .unwrap());

        assert_eq!(store.documents(&notebook.id, ALICE).unwrap().len(), 1);
        assert_eq!(store.list(ALICE).unwrap()[0].document_count, 1);
    }

    #[test]
    fn documents_are_listed_and_counted() {
        let store = NotebookStore::in_memory().unwrap();
        let notebook = store.create(ALICE, "Site A").unwrap();
        store
            .add_document(&notebook.id, ALICE, "sha-1", "pump.pdf")
            .unwrap();
        store
            .add_document(&notebook.id, ALICE, "sha-2", "contract.pdf")
            .unwrap();

        let documents = store.documents(&notebook.id, ALICE).unwrap();
        assert_eq!(documents.len(), 2);
        assert_eq!(documents[0].document_sha256, "sha-1");
        assert_eq!(store.list(ALICE).unwrap()[0].document_count, 2);
    }

    #[test]
    fn adding_a_document_to_a_notebook_that_does_not_exist_writes_nothing() {
        let store = NotebookStore::in_memory().unwrap();
        assert!(!store
            .add_document("no-such-notebook", ALICE, "sha-1", "pump.pdf")
            .unwrap());
    }
}

#[cfg(test)]
mod management_tests {
    use super::*;

    const ALICE: &str = "user-alice";
    const BOB: &str = "user-bob";

    fn store() -> NotebookStore {
        NotebookStore::in_memory().expect("store opens")
    }

    #[test]
    fn a_notebook_can_be_renamed() {
        let store = store();
        let made = store.create(ALICE, "Draft").unwrap();

        let renamed = store.rename(&made.id, ALICE, "Supplier Contracts").unwrap();

        assert_eq!(renamed.name, "Supplier Contracts");
        assert_eq!(renamed.id, made.id);
        assert_eq!(store.get(&made.id, ALICE).unwrap().unwrap().name, "Supplier Contracts");
    }

    #[test]
    fn a_blank_rename_is_refused() {
        let store = store();
        let made = store.create(ALICE, "Draft").unwrap();

        assert!(store.rename(&made.id, ALICE, "   ").is_err());
        // And the old name survives the refusal.
        assert_eq!(store.get(&made.id, ALICE).unwrap().unwrap().name, "Draft");
    }

    /// The isolation the whole store exists to keep. Bob must not be able to
    /// rename, delete, or even see Alice's notebook.
    #[test]
    fn another_person_cannot_touch_it() {
        let store = store();
        let made = store.create(ALICE, "Private") .unwrap();

        assert!(store.rename(&made.id, BOB, "Mine now").is_err());
        assert!(store.delete(&made.id, BOB).is_err());
        assert!(store.get(&made.id, BOB).unwrap().is_none());
        // Untouched.
        assert_eq!(store.get(&made.id, ALICE).unwrap().unwrap().name, "Private");
    }

    #[test]
    fn deleting_takes_the_notebook_out_of_the_list() {
        let store = store();
        let kept = store.create(ALICE, "Kept").unwrap();
        let gone = store.create(ALICE, "Gone").unwrap();

        store.delete(&gone.id, ALICE).unwrap();

        let names: Vec<String> = store.list(ALICE).unwrap().into_iter().map(|n| n.name).collect();
        assert_eq!(names, vec!["Kept".to_string()]);
        assert!(store.get(&gone.id, ALICE).unwrap().is_none());
        assert!(store.get(&kept.id, ALICE).unwrap().is_some());
    }

    /// The reason `delete` is a transaction over five tables rather than one
    /// statement. A recreated notebook must not inherit the old one's graph.
    #[test]
    fn deleting_removes_the_graph_built_over_it() {
        let store = store();
        let notebook = store.create(ALICE, "Unit Four").unwrap();
        let sha = "a1".repeat(32);
        store.add_document(&notebook.id, ALICE, &sha, "pump.pdf").unwrap();

        {
            let conn = store.conn.lock().unwrap();
            conn.execute(
                "INSERT INTO graph_build_units
                 (notebook_id, document_sha256, pass, extractor_version, completed_at)
                 VALUES (?1, ?2, 'statistical', 1, '2026-01-01T00:00:00Z')",
                params![&notebook.id, &sha],
            )
            .unwrap();
        }

        store.delete(&notebook.id, ALICE).unwrap();

        let conn = store.conn.lock().unwrap();
        for table in ["graph_build_units", "notebook_documents"] {
            let left: i64 = conn
                .query_row(
                    &format!("SELECT COUNT(*) FROM {table} WHERE notebook_id = ?1"),
                    params![&notebook.id],
                    |row| row.get(0),
                )
                .unwrap();
            assert_eq!(left, 0, "{table} kept rows for a deleted notebook");
        }
    }

    #[test]
    fn a_document_can_be_taken_out_again() {
        let store = store();
        let notebook = store.create(ALICE, "Unit Four").unwrap();
        let kept = "a1".repeat(32);
        let removed = "b2".repeat(32);
        store.add_document(&notebook.id, ALICE, &kept, "kept.pdf").unwrap();
        store.add_document(&notebook.id, ALICE, &removed, "removed.pdf").unwrap();

        store.remove_document(&notebook.id, ALICE, &removed).unwrap();

        let names: Vec<String> = store
            .documents(&notebook.id, ALICE)
            .unwrap()
            .into_iter()
            .map(|d| d.document_name)
            .collect();
        assert_eq!(names, vec!["kept.pdf".to_string()]);
        // The count on the summary follows.
        assert_eq!(store.get(&notebook.id, ALICE).unwrap().unwrap().document_count, 1);
    }

    #[test]
    fn removing_a_document_that_is_not_there_says_so() {
        let store = store();
        let notebook = store.create(ALICE, "Unit Four").unwrap();

        let problem = store
            .remove_document(&notebook.id, ALICE, &"c3".repeat(32))
            .unwrap_err()
            .to_string();

        assert!(problem.contains("not in this notebook"), "{problem}");
    }

    /// Taking a source out must take its evidence with it, or the graph cites a
    /// passage the notebook can no longer show.
    #[test]
    fn removing_a_document_removes_its_evidence() {
        let store = store();
        let notebook = store.create(ALICE, "Unit Four").unwrap();
        let sha = "a1".repeat(32);
        store.add_document(&notebook.id, ALICE, &sha, "pump.pdf").unwrap();

        {
            let conn = store.conn.lock().unwrap();
            conn.execute(
                "INSERT INTO graph_evidence
                 (notebook_id, subject, subject_kind, chunk_id, document_sha256, page, quote)
                 VALUES (?1, 'PV-2201', 'term', 'chunk-1', ?2, 1, 'a quote')",
                params![&notebook.id, &sha],
            )
            .unwrap();
        }

        store.remove_document(&notebook.id, ALICE, &sha).unwrap();

        let conn = store.conn.lock().unwrap();
        let left: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM graph_evidence WHERE notebook_id = ?1",
                params![&notebook.id],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(left, 0, "evidence outlived the document it came from");
    }
}
