//! Everything a conversation has produced, kept where a later turn can reach it.
//!
//! ## The failure this exists to remove
//!
//! One model draws a diagram. A second writes code. A third is asked for "a PDF
//! with that diagram and that code", and cannot get either. Three separate
//! reasons, none of them fixable by telling a model to remember:
//!
//! 1. **Artifacts were remembered per run, in memory.**
//!    [`crate::agent_runtime::artifacts::RunArtifacts`] is a map keyed by run
//!    id, and `forget` drops the entry once the run's report is written. A
//!    later run in the same conversation had nothing to look at, and a restart
//!    left nothing at all.
//! 2. **Run workspaces cannot read each other.** That is deliberate and stays
//!    — see [`crate::agent_runtime::workspace`], where the isolation is the
//!    point. What was missing is an *authorised channel* that does not depend
//!    on one run guessing another's filesystem path.
//! 3. **Most outputs were never files.** A diagram written as a fenced mermaid
//!    block and code written as a fenced python block live only inside an
//!    assistant message. Nothing captured them, so there was nothing to reuse
//!    even in principle.
//!
//! This is the store that fixes the first and third, and gives the second a
//! path that is checked rather than guessed.
//!
//! ## Identity, versions and content
//!
//! An artifact has a **stable id** that survives revision, and an **immutable
//! version** under it. "Use the revised diagram with the original code" is a
//! sentence about two versions of two ids, and it cannot be expressed — let
//! alone frozen into an export — unless both are addressable.
//!
//! Content is stored by hash, beside the index rather than inside it. Two
//! versions that happen to be byte-identical share one blob, and re-recording
//! identical content returns the existing version rather than minting a new
//! one — which is what keeps a retry or a re-delivered stream from filling the
//! list with duplicates of the same diagram.
//!
//! ## Authorisation
//!
//! Every read takes the owner and puts it in the SQL, the rule the rest of this
//! codebase follows. A conversation id is guessable and an artifact carries
//! content, so a check followed by an unscoped query is not good enough. There
//! is no call here that returns an artifact without an owner.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// What an artifact *is*, which decides how it can be reused.
///
/// Deliberately about reuse rather than about file extension. `DiagramSource`
/// and `RenderedImage` are both "a diagram" to a person and two completely
/// different things to a document writer: one can be re-laid-out, the other can
/// only be placed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ArtifactKind {
    /// Source text in a programming language. Reused verbatim.
    Code,
    /// A diagram in a form that can be re-rendered — SVG, Mermaid, a spec.
    DiagramSource,
    /// A rendered picture. What the chat actually showed.
    RenderedImage,
    /// A produced deliverable: `.docx`, `.xlsx`, `.pptx`.
    Document,
    /// A produced PDF.
    Pdf,
    /// Prose, notes — anything textual that is not code.
    Text,
    /// Structured data: JSON, CSV, a table.
    Data,
}

impl ArtifactKind {
    pub fn as_str(self) -> &'static str {
        match self {
            ArtifactKind::Code => "code",
            ArtifactKind::DiagramSource => "diagramSource",
            ArtifactKind::RenderedImage => "renderedImage",
            ArtifactKind::Document => "document",
            ArtifactKind::Pdf => "pdf",
            ArtifactKind::Text => "text",
            ArtifactKind::Data => "data",
        }
    }

    pub fn parse(text: &str) -> Option<Self> {
        Some(match text {
            "code" => ArtifactKind::Code,
            "diagramSource" | "diagram_source" | "diagram" => ArtifactKind::DiagramSource,
            "renderedImage" | "rendered_image" | "image" => ArtifactKind::RenderedImage,
            "document" => ArtifactKind::Document,
            "pdf" => ArtifactKind::Pdf,
            "text" => ArtifactKind::Text,
            "data" => ArtifactKind::Data,
            _ => return None,
        })
    }

    /// Whether this is something a document can embed as a picture.
    pub fn is_visual(self) -> bool {
        matches!(
            self,
            ArtifactKind::DiagramSource | ArtifactKind::RenderedImage
        )
    }
}

/// A pointer to one exact version of one artifact.
///
/// The unit an export freezes. A reference without a version is a reference to
/// "whatever it is now", which is precisely what must not happen once a PDF has
/// started building.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ArtifactRef {
    pub artifact_id: String,
    pub version: u32,
}

impl ArtifactRef {
    pub fn new(artifact_id: impl Into<String>, version: u32) -> Self {
        Self {
            artifact_id: artifact_id.into(),
            version,
        }
    }
}

impl std::fmt::Display for ArtifactRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}@{}", self.artifact_id, self.version)
    }
}

/// Who made it.
///
/// Kept because "which model drew this" is a question the whole feature exists
/// to answer, and because a later turn under a different model must be able to
/// tell.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Producer {
    /// The model that was serving when this was produced, when one was.
    pub model_id: Option<String>,
    /// The tool that wrote it, or the path that captured it.
    pub tool: Option<String>,
    /// A delegated worker, when the producer was one.
    pub agent: Option<String>,
}

/// One immutable version of one artifact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ArtifactRecord {
    pub artifact_id: String,
    pub version: u32,
    pub conversation_id: String,
    /// The assistant message this came out of, when it came out of one.
    pub message_id: Option<String>,
    pub run_id: Option<String>,
    pub producer: Producer,
    pub kind: ArtifactKind,
    pub mime: String,
    pub title: String,
    /// The name it should be written under, where that is meaningful.
    pub filename: Option<String>,
    pub sha256: String,
    pub bytes: u64,
    /// RFC 3339, UTC.
    pub created_at: String,
    /// False while something is still being written. A turn must not freeze an
    /// incomplete artifact into an export.
    pub complete: bool,
    /// The artifact this was derived from — a revision's predecessor, or the
    /// source a rendering came from.
    pub derived_from: Option<ArtifactRef>,
    /// For a [`ArtifactKind::RenderedImage`], the source it renders.
    ///
    /// The editable-source-to-rendered-output relationship, held in one
    /// direction so it cannot disagree with itself.
    pub renders: Option<ArtifactRef>,
    /// For code, the language as it was written on the fence.
    pub language: Option<String>,
    /// What is needed to render this, when something is.
    ///
    /// Named rather than assumed, so a missing renderer is reported as a
    /// missing renderer instead of producing a blank figure.
    #[serde(default)]
    pub render_requires: Vec<String>,
}

impl ArtifactRecord {
    pub fn reference(&self) -> ArtifactRef {
        ArtifactRef::new(self.artifact_id.clone(), self.version)
    }

    /// The one-line form that goes into a turn's inventory.
    ///
    /// Compact on purpose: the inventory names what exists so a model can ask
    /// for it, and injecting every artifact's contents into every prompt is the
    /// thing this design is avoiding.
    pub fn summarise(&self) -> String {
        let language = self
            .language
            .as_deref()
            .map(|l| format!(" [{l}]"))
            .unwrap_or_default();
        let by = self
            .producer
            .model_id
            .as_deref()
            .map(|m| format!(", by {m}"))
            .unwrap_or_default();
        format!(
            "{}@{} — {} \"{}\"{language} ({} bytes{by})",
            self.artifact_id,
            self.version,
            self.kind.as_str(),
            self.title,
            self.bytes
        )
    }
}

/// What is being recorded, before it has a version.
#[derive(Debug, Clone)]
pub struct NewArtifact {
    /// Supply one to add a version to an existing artifact; omit for a new one.
    pub artifact_id: Option<String>,
    pub conversation_id: String,
    pub owner_user_id: String,
    pub message_id: Option<String>,
    pub run_id: Option<String>,
    pub producer: Producer,
    pub kind: ArtifactKind,
    pub mime: String,
    pub title: String,
    pub filename: Option<String>,
    pub complete: bool,
    pub derived_from: Option<ArtifactRef>,
    pub renders: Option<ArtifactRef>,
    pub language: Option<String>,
    pub render_requires: Vec<String>,
    pub content: Vec<u8>,
}

/// The durable, conversation-scoped artifact store.
pub struct ConversationArtifacts {
    conn: Arc<Mutex<Connection>>,
    blobs: PathBuf,
}

impl ConversationArtifacts {
    pub fn open(app_data_dir: &Path) -> Result<Self> {
        std::fs::create_dir_all(app_data_dir)?;
        let blobs = app_data_dir.join("artifacts").join("blobs");
        std::fs::create_dir_all(&blobs)?;
        let conn = Connection::open(app_data_dir.join("sarathi.db"))
            .context("could not open the artifact store")?;
        Self::prepare(&conn)?;
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
            blobs,
        })
    }

    /// An in-memory index with a real blob directory, for tests.
    #[cfg(test)]
    pub fn in_memory(blobs: &Path) -> Result<Self> {
        std::fs::create_dir_all(blobs)?;
        let conn = Connection::open_in_memory()?;
        Self::prepare(&conn)?;
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
            blobs: blobs.to_path_buf(),
        })
    }

    fn prepare(conn: &Connection) -> Result<()> {
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS conversation_artifacts (
                artifact_id           TEXT NOT NULL,
                version               INTEGER NOT NULL,
                conversation_id       TEXT NOT NULL,
                owner_user_id         TEXT NOT NULL,
                message_id            TEXT,
                run_id                TEXT,
                model_id              TEXT,
                tool                  TEXT,
                agent                 TEXT,
                kind                  TEXT NOT NULL,
                mime                  TEXT NOT NULL,
                title                 TEXT NOT NULL,
                filename              TEXT,
                sha256                TEXT NOT NULL,
                bytes                 INTEGER NOT NULL,
                created_at            TEXT NOT NULL,
                complete              INTEGER NOT NULL,
                derived_from_id       TEXT,
                derived_from_version  INTEGER,
                renders_id            TEXT,
                renders_version       INTEGER,
                language              TEXT,
                render_requires       TEXT NOT NULL DEFAULT '',
                PRIMARY KEY (artifact_id, version)
             );
             CREATE INDEX IF NOT EXISTS conversation_artifacts_conversation_idx
                ON conversation_artifacts(conversation_id, owner_user_id);
             CREATE INDEX IF NOT EXISTS conversation_artifacts_hash_idx
                ON conversation_artifacts(conversation_id, owner_user_id, sha256);",
        )?;
        Ok(())
    }

    fn blob_path(&self, sha256: &str) -> Option<PathBuf> {
        // The hash is checked rather than trusted: it reaches this from a tool
        // call, and a model-supplied string used unchecked as a filename is a
        // path traversal with extra steps. The same rule the document store
        // applies to its extractions.
        let looks_like_a_hash = sha256.len() == 64
            && sha256
                .chars()
                .all(|c| c.is_ascii_digit() || ('a'..='f').contains(&c));
        looks_like_a_hash.then(|| self.blobs.join(sha256))
    }

    /// Records content, returning the version it became.
    ///
    /// Content-addressed and idempotent within an artifact: recording bytes
    /// identical to the newest version of the same `artifact_id` returns that
    /// version rather than minting a new one. That is what keeps a retried tool
    /// call, a re-delivered stream, or a model writing the same file twice from
    /// filling the list with indistinguishable duplicates.
    pub fn record(&self, new: NewArtifact) -> Result<ArtifactRecord> {
        let mut hasher = Sha256::new();
        hasher.update(&new.content);
        let sha256 = format!("{:x}", hasher.finalize());
        let bytes = new.content.len() as u64;

        let artifact_id = new
            .artifact_id
            .clone()
            .unwrap_or_else(|| format!("art-{}", &sha256[..16]));

        let conn = self.conn.lock().expect("artifact store lock poisoned");
        let newest: Option<(u32, String)> = conn
            .query_row(
                "SELECT version, sha256 FROM conversation_artifacts
                  WHERE artifact_id = ?1 AND owner_user_id = ?2
                  ORDER BY version DESC LIMIT 1",
                params![&artifact_id, &new.owner_user_id],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()?;

        if let Some((version, existing_hash)) = &newest {
            if existing_hash == &sha256 {
                let version = *version;
                drop(conn);
                return self
                    .get(&new.owner_user_id, &artifact_id, Some(version))?
                    .context("the existing version disappeared between writing and reading");
            }
        }
        let version = newest.map(|(v, _)| v + 1).unwrap_or(1);

        // The blob first. A row pointing at content that was never written is
        // an artifact the list offers and nothing can open; a blob with no row
        // is invisible and harmless.
        let Some(path) = self.blob_path(&sha256) else {
            anyhow::bail!("the artifact content hash is not a sha-256");
        };
        if !path.exists() {
            let temporary = path.with_extension("tmp");
            std::fs::write(&temporary, &new.content)
                .with_context(|| format!("the artifact content could not be written to {path:?}"))?;
            std::fs::rename(&temporary, &path)
                .with_context(|| format!("the artifact content could not be placed at {path:?}"))?;
        }

        let created_at = chrono::Utc::now().to_rfc3339();
        conn.execute(
            "INSERT INTO conversation_artifacts
                (artifact_id, version, conversation_id, owner_user_id, message_id, run_id,
                 model_id, tool, agent, kind, mime, title, filename, sha256, bytes,
                 created_at, complete, derived_from_id, derived_from_version,
                 renders_id, renders_version, language, render_requires)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15,
                     ?16, ?17, ?18, ?19, ?20, ?21, ?22, ?23)",
            params![
                &artifact_id,
                version,
                &new.conversation_id,
                &new.owner_user_id,
                &new.message_id,
                &new.run_id,
                &new.producer.model_id,
                &new.producer.tool,
                &new.producer.agent,
                new.kind.as_str(),
                &new.mime,
                &new.title,
                &new.filename,
                &sha256,
                bytes as i64,
                &created_at,
                i32::from(new.complete),
                new.derived_from.as_ref().map(|r| r.artifact_id.clone()),
                new.derived_from.as_ref().map(|r| r.version),
                new.renders.as_ref().map(|r| r.artifact_id.clone()),
                new.renders.as_ref().map(|r| r.version),
                &new.language,
                new.render_requires.join(","),
            ],
        )?;
        drop(conn);

        self.get(&new.owner_user_id, &artifact_id, Some(version))?
            .context("the artifact could not be read back after writing")
    }

    /// One version, or the newest when none is named.
    pub fn get(
        &self,
        owner_user_id: &str,
        artifact_id: &str,
        version: Option<u32>,
    ) -> Result<Option<ArtifactRecord>> {
        let conn = self.conn.lock().expect("artifact store lock poisoned");
        let found = match version {
            Some(version) => conn
                .query_row(
                    &format!(
                        "SELECT {FIELDS} FROM conversation_artifacts
                          WHERE artifact_id = ?1 AND owner_user_id = ?2 AND version = ?3"
                    ),
                    params![artifact_id, owner_user_id, version],
                    row_to_record,
                )
                .optional()?,
            None => conn
                .query_row(
                    &format!(
                        "SELECT {FIELDS} FROM conversation_artifacts
                          WHERE artifact_id = ?1 AND owner_user_id = ?2
                          ORDER BY version DESC LIMIT 1"
                    ),
                    params![artifact_id, owner_user_id],
                    row_to_record,
                )
                .optional()?,
        };
        Ok(found)
    }

    /// Every artifact in one conversation, newest version of each, newest first.
    ///
    /// Owner-scoped in the SQL. A conversation id from another account returns
    /// nothing rather than an error: the caller must not be able to tell an
    /// empty conversation from somebody else's.
    pub fn list(&self, owner_user_id: &str, conversation_id: &str) -> Result<Vec<ArtifactRecord>> {
        let conn = self.conn.lock().expect("artifact store lock poisoned");
        let mut statement = conn.prepare(&format!(
            "SELECT {FIELDS} FROM conversation_artifacts a
              WHERE a.conversation_id = ?1 AND a.owner_user_id = ?2
                AND a.version = (SELECT MAX(b.version) FROM conversation_artifacts b
                                  WHERE b.artifact_id = a.artifact_id
                                    AND b.owner_user_id = a.owner_user_id)
              ORDER BY a.created_at DESC, a.artifact_id DESC"
        ))?;
        let rows = statement.query_map(params![conversation_id, owner_user_id], row_to_record)?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .context("the conversation's artifacts could not be read")
    }

    /// Every version of one artifact, oldest first.
    pub fn versions(&self, owner_user_id: &str, artifact_id: &str) -> Result<Vec<ArtifactRecord>> {
        let conn = self.conn.lock().expect("artifact store lock poisoned");
        let mut statement = conn.prepare(&format!(
            "SELECT {FIELDS} FROM conversation_artifacts
              WHERE artifact_id = ?1 AND owner_user_id = ?2
              ORDER BY version ASC"
        ))?;
        let rows = statement.query_map(params![artifact_id, owner_user_id], row_to_record)?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .context("the artifact's versions could not be read")
    }

    /// The bytes of one version, for a caller that may see it.
    ///
    /// Resolved through [`Self::get`], so the owner check happens before the
    /// blob is opened. Reading by hash alone would make the content addressable
    /// by anybody who learned a hash.
    pub fn read(
        &self,
        owner_user_id: &str,
        reference: &ArtifactRef,
    ) -> Result<Option<(ArtifactRecord, Vec<u8>)>> {
        let Some(record) =
            self.get(owner_user_id, &reference.artifact_id, Some(reference.version))?
        else {
            return Ok(None);
        };
        let Some(path) = self.blob_path(&record.sha256) else {
            anyhow::bail!("that artifact's content address is not a sha-256");
        };
        let content = std::fs::read(&path).with_context(|| {
            format!(
                "the content of {} is recorded but could not be read from disk",
                record.reference()
            )
        })?;
        Ok(Some((record, content)))
    }

    /// Marks a version finished.
    ///
    /// Separate from recording because something written in pieces is not
    /// reusable until the last piece has landed, and an export that froze it
    /// half-written would be citing a fragment.
    pub fn complete(&self, owner_user_id: &str, reference: &ArtifactRef) -> Result<()> {
        let conn = self.conn.lock().expect("artifact store lock poisoned");
        let changed = conn.execute(
            "UPDATE conversation_artifacts SET complete = 1
              WHERE artifact_id = ?1 AND version = ?2 AND owner_user_id = ?3",
            params![&reference.artifact_id, reference.version, owner_user_id],
        )?;
        if changed == 0 {
            anyhow::bail!("{reference} is not an artifact this account holds");
        }
        Ok(())
    }
}

/// The column list every read shares, so the row decoder cannot drift from it.
const FIELDS: &str = "artifact_id, version, conversation_id, message_id, run_id, model_id, \
                      tool, agent, kind, mime, title, filename, sha256, bytes, created_at, \
                      complete, derived_from_id, derived_from_version, renders_id, \
                      renders_version, language, render_requires";

fn row_to_record(row: &rusqlite::Row<'_>) -> rusqlite::Result<ArtifactRecord> {
    let derived_id: Option<String> = row.get(16)?;
    let derived_version: Option<u32> = row.get(17)?;
    let renders_id: Option<String> = row.get(18)?;
    let renders_version: Option<u32> = row.get(19)?;
    let requires: String = row.get(21)?;
    Ok(ArtifactRecord {
        artifact_id: row.get(0)?,
        version: row.get(1)?,
        conversation_id: row.get(2)?,
        message_id: row.get(3)?,
        run_id: row.get(4)?,
        producer: Producer {
            model_id: row.get(5)?,
            tool: row.get(6)?,
            agent: row.get(7)?,
        },
        // An unknown kind reads as `Text` rather than failing the row. A record
        // written by a later build must not make the whole list unreadable to
        // this one; the worst case is a document offered as plain text.
        kind: ArtifactKind::parse(&row.get::<_, String>(8)?).unwrap_or(ArtifactKind::Text),
        mime: row.get(9)?,
        title: row.get(10)?,
        filename: row.get(11)?,
        sha256: row.get(12)?,
        bytes: row.get::<_, i64>(13)? as u64,
        created_at: row.get(14)?,
        complete: row.get::<_, i32>(15)? != 0,
        derived_from: derived_id
            .zip(derived_version)
            .map(|(id, v)| ArtifactRef::new(id, v)),
        renders: renders_id
            .zip(renders_version)
            .map(|(id, v)| ArtifactRef::new(id, v)),
        language: row.get(20)?,
        render_requires: requires
            .split(',')
            .filter(|token| !token.is_empty())
            .map(str::to_string)
            .collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALICE: &str = "user-alice";
    const BOB: &str = "user-bob";

    fn store() -> (tempfile::TempDir, ConversationArtifacts) {
        let dir = tempfile::tempdir().expect("a temporary directory");
        let store =
            ConversationArtifacts::in_memory(&dir.path().join("blobs")).expect("the store opens");
        (dir, store)
    }

    fn new(conversation: &str, owner: &str, kind: ArtifactKind, content: &str) -> NewArtifact {
        NewArtifact {
            artifact_id: None,
            conversation_id: conversation.into(),
            owner_user_id: owner.into(),
            message_id: Some("a-1".into()),
            run_id: Some("run-1".into()),
            producer: Producer {
                model_id: Some("qwen2.5-coder-7b".into()),
                tool: Some("workspace.write_text".into()),
                agent: None,
            },
            kind,
            mime: "text/plain".into(),
            title: "Thing".into(),
            filename: None,
            complete: true,
            derived_from: None,
            renders: None,
            language: None,
            render_requires: Vec::new(),
            content: content.as_bytes().to_vec(),
        }
    }

    /// The whole point: what one run produced, a later run can read.
    #[test]
    fn an_artifact_recorded_by_one_run_is_readable_by_another() {
        let (_dir, store) = store();

        let mut first = new("conv-1", ALICE, ArtifactKind::Code, "print('hello')");
        first.run_id = Some("run-a".into());
        first.title = "hello.py".into();
        let recorded = store.record(first).expect("recorded");

        // A different run, a different model, the same conversation.
        let (found, content) = store
            .read(ALICE, &recorded.reference())
            .expect("the read answers")
            .expect("it is there");
        assert_eq!(content, b"print('hello')");
        assert_eq!(found.run_id.as_deref(), Some("run-a"));
        assert_eq!(
            found.producer.model_id.as_deref(),
            Some("qwen2.5-coder-7b"),
            "and it still says which model produced it"
        );
    }

    /// Retries, re-delivered streams and a model writing the same file twice
    /// must not fill the list with copies.
    #[test]
    fn recording_identical_content_returns_the_same_version() {
        let (_dir, store) = store();
        let first = store
            .record(new("conv-1", ALICE, ArtifactKind::Code, "x = 1"))
            .expect("recorded");

        let mut again = new("conv-1", ALICE, ArtifactKind::Code, "x = 1");
        again.artifact_id = Some(first.artifact_id.clone());
        let second = store.record(again).expect("recorded again");

        assert_eq!(second.version, first.version, "no new version");
        assert_eq!(store.versions(ALICE, &first.artifact_id).unwrap().len(), 1);
        assert_eq!(store.list(ALICE, "conv-1").unwrap().len(), 1);
    }

    /// A revision is a new version under the same id, and the old one stays.
    #[test]
    fn a_revision_adds_a_version_and_leaves_the_original_readable() {
        let (_dir, store) = store();
        let first = store
            .record(new(
                "conv-1",
                ALICE,
                ArtifactKind::DiagramSource,
                "<svg>a</svg>",
            ))
            .expect("recorded");

        let mut revised = new("conv-1", ALICE, ArtifactKind::DiagramSource, "<svg>b</svg>");
        revised.artifact_id = Some(first.artifact_id.clone());
        let second = store.record(revised).expect("revised");

        assert_eq!(second.version, 2);
        let (_, original) = store
            .read(ALICE, &ArtifactRef::new(&first.artifact_id, 1))
            .unwrap()
            .expect("version 1 is still there");
        assert_eq!(original, b"<svg>a</svg>");

        assert_eq!(
            store
                .get(ALICE, &first.artifact_id, None)
                .unwrap()
                .unwrap()
                .version,
            2,
            "and 'the newest' resolves to the revision"
        );
        let listed = store.list(ALICE, "conv-1").unwrap();
        assert_eq!(listed.len(), 1, "one artifact, not two rows");
        assert_eq!(listed[0].version, 2);
    }

    /// "The revised diagram with the original code" needs both reachable by
    /// exact version at the same time.
    #[test]
    fn an_old_version_and_a_new_one_can_be_selected_together() {
        let (_dir, store) = store();
        let code = store
            .record(new("conv-1", ALICE, ArtifactKind::Code, "def f(): pass"))
            .expect("code");
        let diagram = store
            .record(new(
                "conv-1",
                ALICE,
                ArtifactKind::DiagramSource,
                "<svg>1</svg>",
            ))
            .expect("diagram");
        let mut revised = new("conv-1", ALICE, ArtifactKind::DiagramSource, "<svg>2</svg>");
        revised.artifact_id = Some(diagram.artifact_id.clone());
        let diagram_v2 = store.record(revised).expect("revised diagram");

        let (_, code_bytes) = store.read(ALICE, &code.reference()).unwrap().unwrap();
        let (_, diagram_bytes) = store.read(ALICE, &diagram_v2.reference()).unwrap().unwrap();

        assert_eq!(code_bytes, b"def f(): pass", "the original code");
        assert_eq!(diagram_bytes, b"<svg>2</svg>", "and the revised diagram");
    }

    /// Recording a newer version must not disturb a reference already frozen.
    #[test]
    fn a_frozen_reference_still_reads_its_own_version_after_a_later_edit() {
        let (_dir, store) = store();
        let first = store
            .record(new("conv-1", ALICE, ArtifactKind::Code, "original"))
            .expect("recorded");
        let frozen = first.reference();

        let mut edited = new("conv-1", ALICE, ArtifactKind::Code, "edited");
        edited.artifact_id = Some(first.artifact_id.clone());
        store.record(edited).expect("edited");

        let (_, bytes) = store.read(ALICE, &frozen).unwrap().unwrap();
        assert_eq!(bytes, b"original", "the frozen version did not move");
    }

    #[test]
    fn another_person_cannot_list_or_read_this_conversations_artifacts() {
        let (_dir, store) = store();
        let recorded = store
            .record(new("conv-1", ALICE, ArtifactKind::Code, "secret = 1"))
            .expect("recorded");

        assert!(
            store.list(BOB, "conv-1").unwrap().is_empty(),
            "Bob must not see Alice's artifacts"
        );
        assert!(
            store.read(BOB, &recorded.reference()).unwrap().is_none(),
            "nor read one by its id"
        );
        assert!(store
            .get(BOB, &recorded.artifact_id, None)
            .unwrap()
            .is_none());
        assert!(
            store.versions(BOB, &recorded.artifact_id).unwrap().is_empty(),
            "nor enumerate its versions"
        );
    }

    #[test]
    fn artifacts_from_another_conversation_are_not_listed() {
        let (_dir, store) = store();
        store
            .record(new("conv-1", ALICE, ArtifactKind::Code, "one"))
            .expect("recorded");
        store
            .record(new("conv-2", ALICE, ArtifactKind::Code, "two"))
            .expect("recorded");

        let listed = store.list(ALICE, "conv-1").unwrap();
        assert_eq!(listed.len(), 1);
        let (_, bytes) = store.read(ALICE, &listed[0].reference()).unwrap().unwrap();
        assert_eq!(bytes, b"one");
    }

    /// The durable half. A second handle onto the same directory is what a
    /// restart looks like from here.
    #[test]
    fn artifacts_survive_reopening_the_store() {
        let dir = tempfile::tempdir().expect("a temporary directory");
        let recorded = {
            let store = ConversationArtifacts::open(dir.path()).expect("the store opens");
            store
                .record(new("conv-1", ALICE, ArtifactKind::Code, "durable = True"))
                .expect("recorded")
        };

        let reopened = ConversationArtifacts::open(dir.path()).expect("the store reopens");
        let (found, content) = reopened
            .read(ALICE, &recorded.reference())
            .expect("the read answers")
            .expect("it survived");
        assert_eq!(content, b"durable = True");
        assert_eq!(found.sha256, recorded.sha256, "and the same content hash");
    }

    /// The source-to-rendering relationship, so a document writer can choose
    /// between re-laying-out and placing.
    #[test]
    fn a_rendering_points_back_at_the_source_it_came_from() {
        let (_dir, store) = store();
        let source = store
            .record(new(
                "conv-1",
                ALICE,
                ArtifactKind::DiagramSource,
                "graph TD; A-->B",
            ))
            .expect("source");

        let mut rendered = new("conv-1", ALICE, ArtifactKind::RenderedImage, "<svg/>");
        rendered.renders = Some(source.reference());
        rendered.mime = "image/svg+xml".into();
        let rendered = store.record(rendered).expect("rendered");

        assert_eq!(rendered.renders, Some(source.reference()));
        assert!(rendered.kind.is_visual() && source.kind.is_visual());
    }

    #[test]
    fn an_incomplete_artifact_says_so_until_it_is_finished() {
        let (_dir, store) = store();
        let mut partial = new("conv-1", ALICE, ArtifactKind::Code, "half");
        partial.complete = false;
        let recorded = store.record(partial).expect("recorded");
        assert!(!recorded.complete);

        store
            .complete(ALICE, &recorded.reference())
            .expect("finished");
        assert!(
            store
                .get(ALICE, &recorded.artifact_id, Some(1))
                .unwrap()
                .unwrap()
                .complete
        );

        assert!(
            store.complete(BOB, &recorded.reference()).is_err(),
            "and somebody else cannot mark it finished"
        );
    }

    /// The inventory line goes into a prompt, so it must be short and must
    /// carry the reference a tool call needs.
    #[test]
    fn the_summary_names_the_reference_and_stays_short() {
        let (_dir, store) = store();
        let mut code = new("conv-1", ALICE, ArtifactKind::Code, "print(1)");
        code.title = "pipeline.py".into();
        code.language = Some("python".into());
        let recorded = store.record(code).expect("recorded");

        let line = recorded.summarise();
        assert!(
            line.contains(&format!("{}@1", recorded.artifact_id)),
            "{line}"
        );
        assert!(line.contains("pipeline.py"), "{line}");
        assert!(line.contains("python"), "{line}");
        assert!(
            !line.contains("print(1)"),
            "the content is not in the inventory: {line}"
        );
        assert!(line.len() < 200, "an inventory line stays short: {line}");
    }
}
