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
//!
//! ## Stages, effects and what a version rests on (plan P04)
//!
//! A version's bytes never change once written, and every read re-hashes the
//! blob against the recorded SHA-256 — a blob altered on disk is refused, not
//! served. On top of that immutable content each version carries:
//!
//! - a **stage**: `recorded` (captured from a message, or written before
//!   stages existed), `candidate` (produced by a tool, not yet accepted) or
//!   `final` (published after an accepted validation of *these exact bytes*).
//!   A stage only moves forward; a final version is never demoted.
//! - an **effect key**: the same key presented twice returns the first
//!   registration and writes nothing, so a retried or replayed create — in
//!   another run, after a restart — does not publish a duplicate. The same key
//!   with different bytes is a conflict, never a silent second version.
//! - its **dependencies**: the sources, memory items, artifact versions and
//!   template it was produced from, each at the exact version read, and the
//!   citation marker that named it. Whether they are still current is decided
//!   against their own stores at the moment it matters (`agent_runtime`'s
//!   artifact tools), never assumed from this table.
//! - its **validations** and **renders**, each tied to the SHA-256 they were
//!   run against.

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
    #[serde(default)]
    pub stage: Stage,
    /// The classification label it carries, derived from the run.
    #[serde(default = "internal")]
    pub classification: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub template: Option<TemplateRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub project_id: Option<String>,
    /// When it became final.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub published_at: Option<String>,
    /// Its node in the shared memory graph, once linked.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub graph_item_id: Option<String>,
}

fn internal() -> String {
    "Internal".to_string()
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

/// Where a version stands. Only ever moves forward.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Stage {
    /// Captured from a message, or written before stages existed.
    #[default]
    Recorded,
    /// Produced by a tool and not yet accepted.
    Candidate,
    /// Published after an accepted validation of these exact bytes.
    Final,
}

impl Stage {
    pub const fn as_str(self) -> &'static str {
        match self {
            Stage::Recorded => "recorded",
            Stage::Candidate => "candidate",
            Stage::Final => "final",
        }
    }

    pub fn parse(raw: &str) -> Option<Self> {
        Some(match raw {
            "recorded" => Stage::Recorded,
            "candidate" => Stage::Candidate,
            "final" => Stage::Final,
            _ => return None,
        })
    }
}

/// The template a version was produced from, at the exact definition used.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TemplateRef {
    pub id: String,
    pub version: String,
    pub sha256: String,
}

/// What kind of thing a version rests on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum DependencyKind {
    /// A document in the organisation's index, by content hash.
    Document,
    /// A file attached to this conversation, by content hash.
    Attachment,
    /// A shared-memory item, at a revision.
    MemoryItem,
    /// Another artifact, at a version.
    Artifact,
    /// A calculation the engine ran, by its expression.
    Calculation,
}

impl DependencyKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            DependencyKind::Document => "document",
            DependencyKind::Attachment => "attachment",
            DependencyKind::MemoryItem => "memoryItem",
            DependencyKind::Artifact => "artifact",
            DependencyKind::Calculation => "calculation",
        }
    }

    pub fn parse(raw: &str) -> Option<Self> {
        Some(match raw {
            "document" => DependencyKind::Document,
            "attachment" => DependencyKind::Attachment,
            "memoryItem" => DependencyKind::MemoryItem,
            "artifact" => DependencyKind::Artifact,
            "calculation" => DependencyKind::Calculation,
            _ => return None,
        })
    }
}

/// One thing a version rests on, at the version it was read at.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VersionDependency {
    pub kind: DependencyKind,
    /// A document hash, memory item id, artifact id or expression.
    pub id: String,
    /// The revision or version read, where the kind has one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
    /// Where inside it: a page, a chunk, a cell.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub locator: Option<String>,
    /// The citation marker in the content that names it, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub marker: Option<String>,
    /// A human label: a document's name, an artifact's title.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
}

/// Everything recorded about a version beyond its content.
#[derive(Debug, Clone, Default)]
pub struct VersionMeta {
    pub stage: Stage,
    /// Derived from the run, never from a model's argument.
    pub classification: Option<String>,
    pub template: Option<TemplateRef>,
    pub project_id: Option<String>,
    pub effect_key: Option<String>,
    pub dependencies: Vec<VersionDependency>,
}

/// What recording did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Registration {
    pub record: ArtifactRecord,
    /// Nothing new was written: the same bytes were already the newest
    /// version, or the effect key had already been applied.
    pub duplicate: bool,
    /// The duplicate was recognised by its effect key.
    pub by_effect_key: bool,
    /// The effect key had already been applied and these bytes differ from
    /// the ones it produced — a retried render with new timestamps, typically.
    /// The first registration stands and these bytes were not recorded.
    pub content_differed: bool,
}

/// One validation, as stored.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StoredValidation {
    pub validation_id: i64,
    pub reference: ArtifactRef,
    pub sha256: String,
    pub accepted: bool,
    /// The full report, as JSON.
    pub report: String,
    pub created_at: String,
}

/// One render, as stored.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StoredRender {
    pub render_id: String,
    pub reference: ArtifactRef,
    pub sha256: String,
    pub state: String,
    /// The render outcome, as JSON.
    pub outcome: String,
    pub created_at: String,
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

        // Additive only: a database from before P04 opens, gains these columns
        // with their defaults, and every existing version reads as `recorded`.
        let present: Vec<String> = {
            let mut statement = conn.prepare("PRAGMA table_info(conversation_artifacts)")?;
            let names = statement.query_map([], |row| row.get::<_, String>(1))?;
            names.collect::<rusqlite::Result<Vec<_>>>()?
        };
        for (column, declaration) in [
            ("stage", "TEXT NOT NULL DEFAULT 'recorded'"),
            ("classification", "TEXT NOT NULL DEFAULT 'Internal'"),
            ("template_id", "TEXT"),
            ("template_version", "TEXT"),
            ("template_sha256", "TEXT"),
            ("project_id", "TEXT"),
            ("published_at", "TEXT"),
            ("graph_item_id", "TEXT"),
        ] {
            if !present.iter().any(|name| name == column) {
                conn.execute_batch(&format!(
                    "ALTER TABLE conversation_artifacts ADD COLUMN {column} {declaration};"
                ))?;
            }
        }

        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS conversation_artifact_dependencies (
                artifact_id  TEXT NOT NULL,
                version      INTEGER NOT NULL,
                ordinal      INTEGER NOT NULL,
                kind         TEXT NOT NULL,
                dep_id       TEXT NOT NULL,
                dep_version  TEXT,
                sha256       TEXT,
                locator      TEXT,
                marker       TEXT,
                label        TEXT,
                PRIMARY KEY (artifact_id, version, ordinal)
             );
             CREATE TABLE IF NOT EXISTS conversation_artifact_effects (
                owner_user_id TEXT NOT NULL,
                effect_key    TEXT NOT NULL,
                artifact_id   TEXT NOT NULL,
                version       INTEGER NOT NULL,
                created_at    TEXT NOT NULL,
                PRIMARY KEY (owner_user_id, effect_key)
             );
             CREATE TABLE IF NOT EXISTS conversation_artifact_validations (
                validation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                artifact_id   TEXT NOT NULL,
                version       INTEGER NOT NULL,
                owner_user_id TEXT NOT NULL,
                sha256        TEXT NOT NULL,
                accepted      INTEGER NOT NULL,
                report        TEXT NOT NULL,
                created_at    TEXT NOT NULL
             );
             CREATE INDEX IF NOT EXISTS conversation_artifact_validations_idx
                ON conversation_artifact_validations(artifact_id, version, owner_user_id);
             CREATE TABLE IF NOT EXISTS conversation_artifact_renders (
                render_id     TEXT PRIMARY KEY,
                artifact_id   TEXT NOT NULL,
                version       INTEGER NOT NULL,
                owner_user_id TEXT NOT NULL,
                sha256        TEXT NOT NULL,
                state         TEXT NOT NULL,
                outcome       TEXT NOT NULL,
                created_at    TEXT NOT NULL
             );",
        )?;
        Ok(())
    }

    /// Where rendered pages are kept, one directory per render.
    pub fn renders_dir(&self) -> PathBuf {
        self.blobs
            .parent()
            .map(|artifacts| artifacts.join("renders"))
            .unwrap_or_else(|| self.blobs.join("renders"))
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
        self.record_version(new, VersionMeta::default()).map(|registration| registration.record)
    }

    /// Records content with its stage, template, classification, effect key
    /// and dependencies, in one transaction.
    ///
    /// An effect key already applied for this owner returns the version it
    /// produced and writes nothing — a replay in another run or after a
    /// restart publishes no duplicate. The same key over different bytes is
    /// refused: one key names one effect.
    pub fn record_version(&self, new: NewArtifact, meta: VersionMeta) -> Result<Registration> {
        let sha256 = format!("{:x}", Sha256::digest(&new.content));
        let bytes = new.content.len() as u64;

        // The default id is derived from the owner *and* the content. From
        // the content alone, two accounts recording identical bytes computed
        // the same id: the second collided on the primary key, and the error
        // told them somebody else held those exact bytes.
        let artifact_id = new.artifact_id.clone().unwrap_or_else(|| {
            let mut hasher = Sha256::new();
            hasher.update(new.owner_user_id.as_bytes());
            hasher.update([0u8]);
            hasher.update(sha256.as_bytes());
            format!("art-{}", &format!("{:x}", hasher.finalize())[..16])
        });

        let mut conn = self.conn.lock().expect("artifact store lock poisoned");

        if let Some(key) = meta.effect_key.as_deref() {
            let applied: Option<(String, u32)> = conn
                .query_row(
                    "SELECT artifact_id, version FROM conversation_artifact_effects
                      WHERE owner_user_id = ?1 AND effect_key = ?2",
                    params![&new.owner_user_id, key],
                    |row| Ok((row.get(0)?, row.get(1)?)),
                )
                .optional()?;
            if let Some((id, version)) = applied {
                drop(conn);
                let record = self
                    .get(&new.owner_user_id, &id, Some(version))?
                    .context("the version an effect key produced has disappeared")?;
                // One key names one effect, and the first one stands. A retry
                // that rendered again produces different bytes (a timestamp in
                // the properties, say); it is the same effect, so nothing new
                // is published -- and the caller is told the bytes differed.
                let content_differed = record.sha256 != sha256;
                return Ok(Registration { record, duplicate: true, by_effect_key: true, content_differed });
            }
        }

        let newest: Option<(u32, String)> = conn
            .query_row(
                "SELECT version, sha256 FROM conversation_artifacts
                  WHERE artifact_id = ?1 AND owner_user_id = ?2
                  ORDER BY version DESC LIMIT 1",
                params![&artifact_id, &new.owner_user_id],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()?;

        let created_at = chrono::Utc::now().to_rfc3339();
        if let Some((version, existing_hash)) = &newest {
            if existing_hash == &sha256 {
                let version = *version;
                // The key is remembered against the version it resolved to, so
                // the next presentation of it is recognised directly.
                if let Some(key) = meta.effect_key.as_deref() {
                    conn.execute(
                        "INSERT OR IGNORE INTO conversation_artifact_effects
                            (owner_user_id, effect_key, artifact_id, version, created_at)
                         VALUES (?1, ?2, ?3, ?4, ?5)",
                        params![&new.owner_user_id, key, &artifact_id, version, &created_at],
                    )?;
                }
                drop(conn);
                let record = self
                    .get(&new.owner_user_id, &artifact_id, Some(version))?
                    .context("the existing version disappeared between writing and reading")?;
                return Ok(Registration { record, duplicate: true, by_effect_key: false, content_differed: false });
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

        let transaction = conn.transaction()?;
        transaction.execute(
            "INSERT INTO conversation_artifacts
                (artifact_id, version, conversation_id, owner_user_id, message_id, run_id,
                 model_id, tool, agent, kind, mime, title, filename, sha256, bytes,
                 created_at, complete, derived_from_id, derived_from_version,
                 renders_id, renders_version, language, render_requires,
                 stage, classification, template_id, template_version, template_sha256,
                 project_id)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15,
                     ?16, ?17, ?18, ?19, ?20, ?21, ?22, ?23, ?24, ?25, ?26, ?27, ?28, ?29)",
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
                // A version is never *recorded* straight into `final`: that
                // stage is reached only through `promote`, against an accepted
                // validation of these bytes.
                match meta.stage {
                    Stage::Final => Stage::Candidate.as_str(),
                    other => other.as_str(),
                },
                meta.classification.as_deref().unwrap_or("Internal"),
                meta.template.as_ref().map(|t| t.id.clone()),
                meta.template.as_ref().map(|t| t.version.clone()),
                meta.template.as_ref().map(|t| t.sha256.clone()),
                &meta.project_id,
            ],
        )?;
        for (ordinal, dependency) in meta.dependencies.iter().enumerate() {
            transaction.execute(
                "INSERT INTO conversation_artifact_dependencies
                    (artifact_id, version, ordinal, kind, dep_id, dep_version, sha256, locator,
                     marker, label)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
                params![
                    &artifact_id,
                    version,
                    ordinal as i64,
                    dependency.kind.as_str(),
                    &dependency.id,
                    &dependency.version,
                    &dependency.sha256,
                    &dependency.locator,
                    &dependency.marker,
                    &dependency.label,
                ],
            )?;
        }
        if let Some(key) = meta.effect_key.as_deref() {
            transaction.execute(
                "INSERT INTO conversation_artifact_effects
                    (owner_user_id, effect_key, artifact_id, version, created_at)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                params![&new.owner_user_id, key, &artifact_id, version, &created_at],
            )?;
        }
        transaction.commit()?;
        drop(conn);

        let record = self
            .get(&new.owner_user_id, &artifact_id, Some(version))?
            .context("the artifact could not be read back after writing")?;
        Ok(Registration { record, duplicate: false, by_effect_key: false, content_differed: false })
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
        // Immutable means checked, not assumed: a blob altered on disk is
        // refused rather than served as the version it no longer is.
        let actual = format!("{:x}", Sha256::digest(&content));
        if actual != record.sha256 {
            anyhow::bail!(
                "the stored bytes of {} no longer hash to the recorded {}; the content was altered \
                 outside this store and is not served",
                record.reference(),
                &record.sha256[..12]
            );
        }
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

    /// What a version rests on, in the order it was recorded.
    ///
    /// Owner-scoped: the join refuses another account's version.
    pub fn dependencies(&self, owner_user_id: &str, reference: &ArtifactRef) -> Result<Vec<VersionDependency>> {
        let conn = self.conn.lock().expect("artifact store lock poisoned");
        let mut statement = conn.prepare(
            "SELECT d.kind, d.dep_id, d.dep_version, d.sha256, d.locator, d.marker, d.label
               FROM conversation_artifact_dependencies d
               JOIN conversation_artifacts a
                 ON a.artifact_id = d.artifact_id AND a.version = d.version
              WHERE d.artifact_id = ?1 AND d.version = ?2 AND a.owner_user_id = ?3
              ORDER BY d.ordinal",
        )?;
        let rows = statement.query_map(
            params![&reference.artifact_id, reference.version, owner_user_id],
            |row| {
                let kind: String = row.get(0)?;
                Ok(VersionDependency {
                    kind: DependencyKind::parse(&kind).unwrap_or(DependencyKind::Document),
                    id: row.get(1)?,
                    version: row.get(2)?,
                    sha256: row.get(3)?,
                    locator: row.get(4)?,
                    marker: row.get(5)?,
                    label: row.get(6)?,
                })
            },
        )?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .context("the version's dependencies could not be read")
    }

    /// Keeps a validation of one version's exact bytes. Returns its id.
    pub fn record_validation(
        &self,
        owner_user_id: &str,
        reference: &ArtifactRef,
        sha256: &str,
        accepted: bool,
        report_json: &str,
    ) -> Result<i64> {
        let record = self
            .get(owner_user_id, &reference.artifact_id, Some(reference.version))?
            .with_context(|| format!("{reference} is not an artifact this account holds"))?;
        if record.sha256 != sha256 {
            anyhow::bail!("the validation was of different bytes than {reference} holds");
        }
        let conn = self.conn.lock().expect("artifact store lock poisoned");
        conn.execute(
            "INSERT INTO conversation_artifact_validations
                (artifact_id, version, owner_user_id, sha256, accepted, report, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                &reference.artifact_id,
                reference.version,
                owner_user_id,
                sha256,
                i32::from(accepted),
                report_json,
                chrono::Utc::now().to_rfc3339(),
            ],
        )?;
        Ok(conn.last_insert_rowid())
    }

    /// The newest validation of a version, for this owner.
    pub fn latest_validation(&self, owner_user_id: &str, reference: &ArtifactRef) -> Result<Option<StoredValidation>> {
        let conn = self.conn.lock().expect("artifact store lock poisoned");
        conn.query_row(
            "SELECT validation_id, sha256, accepted, report, created_at
               FROM conversation_artifact_validations
              WHERE artifact_id = ?1 AND version = ?2 AND owner_user_id = ?3
              ORDER BY validation_id DESC LIMIT 1",
            params![&reference.artifact_id, reference.version, owner_user_id],
            |row| {
                Ok(StoredValidation {
                    validation_id: row.get(0)?,
                    reference: reference.clone(),
                    sha256: row.get(1)?,
                    accepted: row.get::<_, i32>(2)? != 0,
                    report: row.get(3)?,
                    created_at: row.get(4)?,
                })
            },
        )
        .optional()
        .context("the version's validations could not be read")
    }

    /// Keeps a render of one version's exact bytes.
    pub fn record_render(
        &self,
        owner_user_id: &str,
        reference: &ArtifactRef,
        render_id: &str,
        sha256: &str,
        state: &str,
        outcome_json: &str,
    ) -> Result<()> {
        let conn = self.conn.lock().expect("artifact store lock poisoned");
        conn.execute(
            "INSERT OR REPLACE INTO conversation_artifact_renders
                (render_id, artifact_id, version, owner_user_id, sha256, state, outcome, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                render_id,
                &reference.artifact_id,
                reference.version,
                owner_user_id,
                sha256,
                state,
                outcome_json,
                chrono::Utc::now().to_rfc3339(),
            ],
        )?;
        Ok(())
    }

    /// One render, for the owner who asked for it.
    pub fn render(&self, owner_user_id: &str, render_id: &str) -> Result<Option<StoredRender>> {
        let conn = self.conn.lock().expect("artifact store lock poisoned");
        conn.query_row(
            "SELECT render_id, artifact_id, version, sha256, state, outcome, created_at
               FROM conversation_artifact_renders
              WHERE render_id = ?1 AND owner_user_id = ?2",
            params![render_id, owner_user_id],
            |row| {
                Ok(StoredRender {
                    render_id: row.get(0)?,
                    reference: ArtifactRef::new(row.get::<_, String>(1)?, row.get(2)?),
                    sha256: row.get(3)?,
                    state: row.get(4)?,
                    outcome: row.get(5)?,
                    created_at: row.get(6)?,
                })
            },
        )
        .optional()
        .context("the render could not be read")
    }

    /// Moves a version to `final`, against an accepted validation of its
    /// exact bytes. Idempotent; a final version is never moved back.
    pub fn promote(&self, owner_user_id: &str, reference: &ArtifactRef, validation_id: i64) -> Result<ArtifactRecord> {
        let record = self
            .get(owner_user_id, &reference.artifact_id, Some(reference.version))?
            .with_context(|| format!("{reference} is not an artifact this account holds"))?;
        if record.stage == Stage::Final {
            return Ok(record);
        }
        {
            let conn = self.conn.lock().expect("artifact store lock poisoned");
            let validation: Option<(String, i32)> = conn
                .query_row(
                    "SELECT sha256, accepted FROM conversation_artifact_validations
                      WHERE validation_id = ?1 AND artifact_id = ?2 AND version = ?3
                        AND owner_user_id = ?4",
                    params![validation_id, &reference.artifact_id, reference.version, owner_user_id],
                    |row| Ok((row.get(0)?, row.get(1)?)),
                )
                .optional()?;
            match validation {
                None => anyhow::bail!("validation {validation_id} is not a validation of {reference}"),
                Some((sha, _)) if sha != record.sha256 => {
                    anyhow::bail!("validation {validation_id} was of different bytes than {reference}")
                }
                Some((_, 0)) => anyhow::bail!("validation {validation_id} did not accept {reference}"),
                Some(_) => {}
            }
            conn.execute(
                "UPDATE conversation_artifacts SET stage = 'final', published_at = ?1
                  WHERE artifact_id = ?2 AND version = ?3 AND owner_user_id = ?4 AND stage != 'final'",
                params![chrono::Utc::now().to_rfc3339(), &reference.artifact_id, reference.version, owner_user_id],
            )?;
        }
        self.get(owner_user_id, &reference.artifact_id, Some(reference.version))?
            .context("the version could not be read back after it was published")
    }

    /// Every render of one version, newest first.
    pub fn renders_for(&self, owner_user_id: &str, reference: &ArtifactRef) -> Result<Vec<StoredRender>> {
        let conn = self.conn.lock().expect("artifact store lock poisoned");
        let mut statement = conn.prepare(
            "SELECT render_id, sha256, state, outcome, created_at FROM conversation_artifact_renders
              WHERE artifact_id = ?1 AND version = ?2 AND owner_user_id = ?3
              ORDER BY created_at DESC, render_id DESC",
        )?;
        let rows = statement.query_map(params![&reference.artifact_id, reference.version, owner_user_id], |row| {
            Ok(StoredRender {
                render_id: row.get(0)?,
                reference: reference.clone(),
                sha256: row.get(1)?,
                state: row.get(2)?,
                outcome: row.get(3)?,
                created_at: row.get(4)?,
            })
        })?;
        rows.collect::<rusqlite::Result<Vec<_>>>().context("the version's renders could not be read")
    }

    /// Moves a `recorded` version to `candidate`. Nothing else moves.
    pub fn mark_candidate(&self, owner_user_id: &str, reference: &ArtifactRef) -> Result<ArtifactRecord> {
        {
            let conn = self.conn.lock().expect("artifact store lock poisoned");
            conn.execute(
                "UPDATE conversation_artifacts SET stage = 'candidate'
                  WHERE artifact_id = ?1 AND version = ?2 AND owner_user_id = ?3 AND stage = 'recorded'",
                params![&reference.artifact_id, reference.version, owner_user_id],
            )?;
        }
        self.get(owner_user_id, &reference.artifact_id, Some(reference.version))?
            .with_context(|| format!("{reference} is not an artifact this account holds"))
    }

    /// The version an effect key already produced, for this owner.
    pub fn effect(&self, owner_user_id: &str, effect_key: &str) -> Result<Option<ArtifactRef>> {
        let conn = self.conn.lock().expect("artifact store lock poisoned");
        conn.query_row(
            "SELECT artifact_id, version FROM conversation_artifact_effects
              WHERE owner_user_id = ?1 AND effect_key = ?2",
            params![owner_user_id, effect_key],
            |row| Ok(ArtifactRef::new(row.get::<_, String>(0)?, row.get(1)?)),
        )
        .optional()
        .context("the effect keys could not be read")
    }

    /// Records that an effect key produced this version. First write wins.
    pub fn remember_effect(&self, owner_user_id: &str, effect_key: &str, reference: &ArtifactRef) -> Result<()> {
        let conn = self.conn.lock().expect("artifact store lock poisoned");
        conn.execute(
            "INSERT OR IGNORE INTO conversation_artifact_effects
                (owner_user_id, effect_key, artifact_id, version, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![owner_user_id, effect_key, &reference.artifact_id, reference.version, chrono::Utc::now().to_rfc3339()],
        )?;
        Ok(())
    }

    /// Records the version's node in the shared memory graph.
    pub fn set_graph_item(&self, owner_user_id: &str, reference: &ArtifactRef, item_id: &str) -> Result<()> {
        let conn = self.conn.lock().expect("artifact store lock poisoned");
        let changed = conn.execute(
            "UPDATE conversation_artifacts SET graph_item_id = ?1
              WHERE artifact_id = ?2 AND version = ?3 AND owner_user_id = ?4",
            params![item_id, &reference.artifact_id, reference.version, owner_user_id],
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
                      renders_version, language, render_requires, stage, classification, \
                      template_id, template_version, template_sha256, project_id, published_at, \
                      graph_item_id";

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
        stage: Stage::parse(&row.get::<_, String>(22)?).unwrap_or_default(),
        classification: row.get(23)?,
        template: match (
            row.get::<_, Option<String>>(24)?,
            row.get::<_, Option<String>>(25)?,
            row.get::<_, Option<String>>(26)?,
        ) {
            (Some(id), Some(version), Some(sha256)) => Some(TemplateRef { id, version, sha256 }),
            _ => None,
        },
        project_id: row.get(27)?,
        published_at: row.get(28)?,
        graph_item_id: row.get(29)?,
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

    // ── P04: stages, effects, dependencies ───────────────────────────────

    fn candidate(effect_key: Option<&str>) -> VersionMeta {
        VersionMeta {
            stage: Stage::Candidate,
            classification: Some("Internal".into()),
            template: Some(TemplateRef { id: "approval_note".into(), version: "1".into(), sha256: "a".repeat(64) }),
            project_id: None,
            effect_key: effect_key.map(str::to_string),
            dependencies: vec![VersionDependency {
                kind: DependencyKind::MemoryItem,
                id: "mem-1".into(),
                version: Some("2".into()),
                sha256: None,
                locator: None,
                marker: Some("[M:mem-1@2]".into()),
                label: None,
            }],
        }
    }

    /// The same effect key, presented again from another run after a restart,
    /// publishes nothing new.
    #[test]
    fn a_repeated_effect_key_does_not_publish_a_duplicate_even_after_a_restart() {
        let dir = tempfile::tempdir().expect("a temporary directory");
        let first = {
            let store = ConversationArtifacts::open(dir.path()).expect("opens");
            let mut produced = new("conv-1", ALICE, ArtifactKind::Document, "note bytes");
            produced.run_id = Some("run-a".into());
            store.record_version(produced, candidate(Some("effect-1"))).expect("recorded")
        };
        assert!(!first.duplicate);
        assert_eq!(first.record.stage, Stage::Candidate);

        let reopened = ConversationArtifacts::open(dir.path()).expect("reopens");
        let mut again = new("conv-1", ALICE, ArtifactKind::Document, "note bytes");
        again.run_id = Some("run-b".into());
        let second = reopened.record_version(again, candidate(Some("effect-1"))).expect("replayed");
        assert!(second.duplicate && second.by_effect_key);
        assert_eq!(second.record.reference(), first.record.reference());
        assert_eq!(second.record.run_id.as_deref(), Some("run-a"), "the first run's record stands");
        assert_eq!(reopened.list(ALICE, "conv-1").unwrap().len(), 1);

        // A retry that rendered different bytes is the same effect: the first
        // registration stands, nothing new is written, and it says so.
        let retried = reopened
            .record_version(new("conv-1", ALICE, ArtifactKind::Document, "other bytes"), candidate(Some("effect-1")))
            .expect("the first registration is returned");
        assert!(retried.duplicate && retried.content_differed);
        assert_eq!(retried.record.reference(), first.record.reference());
        assert_eq!(reopened.versions(ALICE, &first.record.artifact_id).unwrap().len(), 1);

        // Another owner's key space is their own.
        let bobs = reopened
            .record_version(new("conv-9", BOB, ArtifactKind::Document, "note bytes"), candidate(Some("effect-1")))
            .expect("bob records his own");
        assert!(!bobs.duplicate);
    }

    #[test]
    fn a_version_reaches_final_only_through_an_accepted_validation_of_its_own_bytes() {
        let (_dir, store) = store();
        let mut meta = candidate(None);
        meta.stage = Stage::Final;
        let recorded = store.record_version(new("conv-1", ALICE, ArtifactKind::Document, "v1"), meta).unwrap().record;
        assert_eq!(recorded.stage, Stage::Candidate, "nothing is recorded straight into final");

        let rejected = store.record_validation(ALICE, &recorded.reference(), &recorded.sha256, false, "{}").unwrap();
        assert!(store.promote(ALICE, &recorded.reference(), rejected).is_err());
        assert!(store.record_validation(ALICE, &recorded.reference(), &"f".repeat(64), true, "{}").is_err());
        let accepted = store.record_validation(ALICE, &recorded.reference(), &recorded.sha256, true, "{}").unwrap();
        assert!(store.promote(BOB, &recorded.reference(), accepted).is_err(), "not Bob's to publish");

        let published = store.promote(ALICE, &recorded.reference(), accepted).expect("published");
        assert_eq!(published.stage, Stage::Final);
        assert!(published.published_at.is_some());
        let again = store.promote(ALICE, &recorded.reference(), accepted).expect("idempotent");
        assert_eq!(again.published_at, published.published_at);
        assert_eq!(store.latest_validation(ALICE, &recorded.reference()).unwrap().unwrap().validation_id, accepted);
    }

    #[test]
    fn dependencies_are_kept_per_version_and_only_shown_to_the_owner() {
        let (_dir, store) = store();
        let recorded = store.record_version(new("conv-1", ALICE, ArtifactKind::Document, "v1"), candidate(None)).unwrap().record;
        let dependencies = store.dependencies(ALICE, &recorded.reference()).unwrap();
        assert_eq!(dependencies.len(), 1);
        assert_eq!(dependencies[0].marker.as_deref(), Some("[M:mem-1@2]"));
        assert!(store.dependencies(BOB, &recorded.reference()).unwrap().is_empty());
        assert_eq!(recorded.template.as_ref().map(|t| t.id.as_str()), Some("approval_note"));
    }

    /// Exact bytes, checked: a blob changed on disk is refused, not served.
    #[test]
    fn a_blob_altered_on_disk_is_refused_rather_than_served() {
        let (dir, store) = store();
        let recorded = store.record(new("conv-1", ALICE, ArtifactKind::Code, "original")).unwrap();
        std::fs::write(dir.path().join("blobs").join(&recorded.sha256), b"tampered").unwrap();
        let error = store.read(ALICE, &recorded.reference()).expect_err("refused");
        assert!(error.to_string().contains("no longer hash"), "{error}");
    }

    /// A database written before P04 opens, gains the columns, and its rows
    /// read as `recorded`.
    #[test]
    fn a_store_from_before_stages_existed_opens_and_reads() {
        let dir = tempfile::tempdir().expect("a temporary directory");
        {
            let conn = Connection::open(dir.path().join("sarathi.db")).unwrap();
            conn.execute_batch(
                "CREATE TABLE conversation_artifacts (
                    artifact_id TEXT NOT NULL, version INTEGER NOT NULL, conversation_id TEXT NOT NULL,
                    owner_user_id TEXT NOT NULL, message_id TEXT, run_id TEXT, model_id TEXT, tool TEXT,
                    agent TEXT, kind TEXT NOT NULL, mime TEXT NOT NULL, title TEXT NOT NULL, filename TEXT,
                    sha256 TEXT NOT NULL, bytes INTEGER NOT NULL, created_at TEXT NOT NULL,
                    complete INTEGER NOT NULL, derived_from_id TEXT, derived_from_version INTEGER,
                    renders_id TEXT, renders_version INTEGER, language TEXT,
                    render_requires TEXT NOT NULL DEFAULT '', PRIMARY KEY (artifact_id, version));
                 INSERT INTO conversation_artifacts VALUES ('art-old', 1, 'conv-1', 'user-alice', NULL, NULL,
                    NULL, NULL, NULL, 'code', 'text/plain', 'old.py', NULL, 'abc', 3,
                    '2026-01-01T00:00:00Z', 1, NULL, NULL, NULL, NULL, NULL, '');",
            )
            .unwrap();
        }
        let store = ConversationArtifacts::open(dir.path()).expect("the old store opens");
        let old = store.get(ALICE, "art-old", None).unwrap().expect("the old row reads");
        assert_eq!(old.stage, Stage::Recorded);
        assert_eq!(old.classification, "Internal");
        assert!(old.template.is_none());
        // And opening twice does not try to add the columns twice.
        drop(store);
        ConversationArtifacts::open(dir.path()).expect("reopens");
    }
}
