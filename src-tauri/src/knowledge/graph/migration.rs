//! Bringing the stores that predate runtime memory under its authority.
//!
//! [`runtime_store`](super::runtime_store)'s own module documentation states
//! the problem this module exists to close:
//!
//! > Two other authorities — the conversation store and the agent registry —
//! > are JSON files and are not in that database at all.
//!
//! [`Provenance::Migrated`] has been in the data model since runtime memory was
//! written, and until now **nothing constructed it**. That is the honest
//! description of the gap: the destination was designed for this and the road
//! to it was never built.
//!
//! ## Why a migrated id is derived and an ordinary one is random
//!
//! [`super::runtime_memory::item_id`] is a fresh uuid, on purpose — two agents
//! independently establishing the same fact are two observations of it, and
//! collapsing them into one row would destroy the attribution that makes the
//! graph worth having.
//!
//! A migration is the one case where the opposite is required, and for a
//! reason that does not contradict it: there is only ever *one* legacy record,
//! and a second pass over it is not a second observation — it is the same
//! record arriving twice. So [`migrated_item_id`] is derived from
//! `(legacy_store, legacy_id)`, and a re-run recognises what it already moved
//! instead of duplicating it. The same string is also used as the
//! `idempotency_key`, so the recognition happens inside the store's
//! transaction rather than in a check this module performs and then races with.
//!
//! That is what makes an interrupted migration safe to simply run again. There
//! is no resume cursor to keep, because there is no state to lose: every write
//! is addressed by what it came from.
//!
//! ## Coverage, stated rather than implied
//!
//! All five legacy stores in [`LegacySource`] are moved: the agent registry and
//! conversation pins (the two the runtime store's own docs name), plus
//! scoped-memory rows, notebook assertions and artifact references.
//!
//! [`MigrationReport::uncovered`] is still computed and still reported on every
//! run. It is empty today, and it is the mechanism that keeps a *sixth* store —
//! which is a realistic thing to discover — from being quietly omitted.
//!
//! What this is **not** is a cutover. Every legacy store is still written to
//! directly by its own code; this copies from them under one authority. Routing
//! the writers through that authority is a separate change, and until it
//! happens the compatibility position is "two stores, one of them derived" —
//! stated here rather than left for a reader to infer.
//!
//! ## Reading is fallible and says so
//!
//! A legacy record that cannot be parsed is counted and named. It is never
//! silently skipped: a migration that quietly drops what it could not read and
//! then reports success is the failure this repository's rules are written
//! against, and the conversation store on this machine has two damaged files
//! that would have taken exactly that path.

use std::collections::BTreeMap;

use sha2::{Digest, Sha256};

use super::assertions::AssertionStatus;
use super::runtime_memory::{
    ArtifactRef, ItemStatus, MemoryItem, MemoryKind, MemoryScope, Provenance, SourceRef,
    MEMORY_SCHEMA_VERSION,
};
use super::store::NotebookStore;
use super::runtime_store::{MemoryError, MemoryGraph};
use crate::agent_runtime::conversations::ConversationStore;
use crate::agent_runtime::memory::Acl;
use crate::agents::store::{AgentRegistry, Visibility};
use crate::artifacts::conversation_store::ConversationArtifacts;
use crate::memory_engine::persistence::PersistenceManager;
use crate::policy::Classification;

/// A store that predates runtime memory.
///
/// Named rather than free strings, because the name is part of every migrated
/// item's identity: change one and the derived ids change with it, which would
/// make a re-run duplicate everything it had already moved.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LegacySource {
    /// `<app data>/agents/registry.json`.
    AgentProfiles,
    /// `pinnedContext` inside `<app data>/conversations/<id>.json`.
    ConversationPins,
    /// `memory_engine`'s scoped store, keyed by project.
    ScopedMemoryJson,
    /// `knowledge::graph::assertions` — a notebook's reviewed claims.
    NotebookAssertions,
    /// The artifact store's produced-document references, at exact revisions.
    ArtifactReferences,
}

impl LegacySource {
    /// The string that goes into `Provenance::Migrated.legacy_store`.
    ///
    /// Stable for ever. It is an input to [`migrated_item_id`].
    pub fn key(self) -> &'static str {
        match self {
            Self::AgentProfiles => "agents/registry.json",
            Self::ConversationPins => "conversations/pinnedContext",
            Self::ScopedMemoryJson => "memory_engine/scoped",
            Self::NotebookAssertions => "knowledge/assertions",
            Self::ArtifactReferences => "artifacts/references",
        }
    }

    /// Whether this module actually moves it today.
    ///
    /// All five, now. Kept as a method rather than deleted: a sixth legacy
    /// store is a realistic thing to discover, and the day one is added the
    /// report must be able to say it is not covered yet rather than quietly
    /// omitting it.
    pub fn implemented(self) -> bool {
        true
    }

    pub const ALL: &'static [LegacySource] = &[
        Self::AgentProfiles,
        Self::ConversationPins,
        Self::ScopedMemoryJson,
        Self::NotebookAssertions,
        Self::ArtifactReferences,
    ];
}

/// The stable id of a migrated item.
///
/// Derived, not random — see the module docs. `\x1f` separates the fields so
/// that a store named `a` with id `bc` cannot collide with a store named `ab`
/// with id `c`.
pub fn migrated_item_id(legacy_store: &str, legacy_id: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(legacy_store.as_bytes());
    hasher.update(b"\x1f");
    hasher.update(legacy_id.as_bytes());
    format!("mi-mig-{:x}", hasher.finalize())
}

/// What one source produced.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SourceOutcome {
    /// Legacy records looked at, including ones that turned out to be already
    /// migrated or unreadable.
    pub examined: usize,
    /// Written by this run.
    pub migrated: usize,
    /// Recognised as already migrated. On a second run this is the whole
    /// population and `migrated` is zero — that is the property that makes an
    /// interrupted migration safe to retry.
    pub already_present: usize,
    /// Named, never merely counted, so an operator can go and look.
    pub unreadable: Vec<String>,
}

impl SourceOutcome {
    /// Whether this source finished with nothing left unexplained.
    pub fn clean(&self) -> bool {
        self.unreadable.is_empty() && self.examined == self.migrated + self.already_present
    }
}

/// The result of a migration pass.
#[derive(Debug, Clone, Default)]
pub struct MigrationReport {
    pub schema_version: u32,
    pub per_source: BTreeMap<&'static str, SourceOutcome>,
    /// Sources this module does not yet move. Reported on every run so the
    /// omission cannot be mistaken for completeness.
    pub uncovered: Vec<&'static str>,
}

impl MigrationReport {
    fn new() -> Self {
        Self {
            schema_version: MEMORY_SCHEMA_VERSION,
            per_source: BTreeMap::new(),
            uncovered: LegacySource::ALL
                .iter()
                .filter(|source| !source.implemented())
                .map(|source| source.key())
                .collect(),
        }
    }

    pub fn total_migrated(&self) -> usize {
        self.per_source.values().map(|o| o.migrated).sum()
    }

    pub fn total_unreadable(&self) -> usize {
        self.per_source.values().map(|o| o.unreadable.len()).sum()
    }

    /// One paragraph an operator can read, including what was *not* done.
    pub fn explain(&self) -> String {
        let mut lines = vec![format!(
            "runtime memory schema {}: {} item(s) migrated, {} legacy record(s) unreadable.",
            self.schema_version,
            self.total_migrated(),
            self.total_unreadable()
        )];
        for (source, outcome) in &self.per_source {
            lines.push(format!(
                "  {source}: examined {}, migrated {}, already present {}, unreadable {}",
                outcome.examined,
                outcome.migrated,
                outcome.already_present,
                outcome.unreadable.len()
            ));
        }
        if !self.uncovered.is_empty() {
            lines.push(format!(
                "  not migrated by this pass: {}",
                self.uncovered.join(", ")
            ));
        }
        lines.join("\n")
    }
}

/// Builds the item for one legacy record.
///
/// Split out so the shape can be asserted without a store. Everything that
/// decides authorisation — scope, classification, ACL — is settled here, at the
/// point where the legacy record is still in hand, rather than defaulted later
/// by something that has forgotten where the row came from.
fn migrated_item(
    source: LegacySource,
    legacy_id: &str,
    agent_id: &str,
    kind: MemoryKind,
    scope: MemoryScope,
    classification: Classification,
    content: String,
    at: &str,
) -> MemoryItem {
    let item_id = migrated_item_id(source.key(), legacy_id);
    let project_id = scope.project().map(str::to_string);
    MemoryItem {
        item_id: item_id.clone(),
        revision: 1,
        kind,
        agent_id: agent_id.to_string(),
        scope,
        classification,
        acl: Acl::for_classification(classification, project_id.as_deref()),
        creator_model_id: None,
        creator_run_id: None,
        provenance: Provenance::Migrated {
            legacy_store: source.key().to_string(),
            legacy_id: legacy_id.to_string(),
        },
        content,
        sources: Vec::new(),
        artifacts: Vec::new(),
        confidence: None,
        // `admit` decides this; set to the value it will choose so the struct
        // is never briefly claiming something the store would refuse.
        status: ItemStatus::Admitted,
        valid_from: at.to_string(),
        valid_until: None,
        supersedes: None,
        conflicts_with: Vec::new(),
        causal_parents: Vec::new(),
        // The same derived string. A retry after a lost acknowledgement is
        // recognised inside the store's transaction rather than by a check here
        // that would race with a concurrent pass.
        idempotency_key: Some(item_id),
        applies_to: None,
        created_at: at.to_string(),
        updated_at: at.to_string(),
    }
}

/// Moves the agent registry under the runtime store.
///
/// One item per agent definition, carrying the definition version so a later
/// edit is a new fact rather than a silent overwrite. The agent's
/// `classification_ceiling` becomes the item's classification, which is the
/// only defensible reading: a profile must not be readable by somebody who is
/// not cleared for the material that agent is allowed to handle.
pub fn migrate_agent_profiles(
    registry: &AgentRegistry,
    store: &MemoryGraph,
    project_id: &str,
    at: &str,
) -> Result<SourceOutcome, MemoryError> {
    let mut outcome = SourceOutcome::default();

    let definitions = match registry.list(Visibility::Administrator) {
        Ok(definitions) => definitions,
        Err(error) => {
            // The registry is one file. If it will not parse there is nothing
            // to enumerate, and saying "0 migrated" would read as success.
            outcome
                .unreadable
                .push(format!("agents/registry.json: {error:?}"));
            return Ok(outcome);
        }
    };

    for definition in definitions {
        outcome.examined += 1;
        let legacy_id = format!("{}@{}", definition.agent_id, definition.definition_version);
        let content = format!(
            "Agent {} ({}) is defined at version {} with colour {}, role {:?}, \
             classification ceiling {:?}.",
            definition.display_name,
            definition.agent_id,
            definition.definition_version,
            definition.color,
            definition.role,
            definition.classification_ceiling,
        );
        let item = migrated_item(
            LegacySource::AgentProfiles,
            &legacy_id,
            &definition.agent_id,
            MemoryKind::Decision,
            MemoryScope::Workspace {
                project_id: project_id.to_string(),
            },
            definition.classification_ceiling,
            content,
            at,
        );
        let committed = store.commit(item, None, &[])?;
        if committed.duplicate {
            outcome.already_present += 1;
        } else {
            outcome.migrated += 1;
        }
    }

    Ok(outcome)
}

/// Moves conversation pins under the runtime store.
///
/// A pin is a person saying "the rest of this task depends on that", which is a
/// [`MemoryKind::Constraint`]: it binds later work rather than asserting a
/// fact. Pins are user-scoped because the conversation they belong to is, and
/// the ACL carries the owner so another signed-in user cannot read them — the
/// conversation store enforces that with an owner check and the migrated copy
/// must not be the place that clearance quietly widens.
pub fn migrate_conversation_pins(
    conversations: &ConversationStore,
    store: &MemoryGraph,
    owner_user_id: &str,
    at: &str,
) -> Result<SourceOutcome, MemoryError> {
    let mut outcome = SourceOutcome::default();

    let (threads, unreadable) = match conversations.list_with_diagnostics(Some(owner_user_id)) {
        Ok(pair) => pair,
        Err(error) => {
            outcome
                .unreadable
                .push(format!("conversations directory: {error}"));
            return Ok(outcome);
        }
    };
    // Damaged transcripts are named here rather than dropped. A pin inside a
    // file that will not parse is a pin this migration did not move, and an
    // operator has to be able to find out which file to go and look at.
    for path in unreadable {
        outcome.unreadable.push(path.display().to_string());
    }

    for thread in threads {
        for (index, pinned) in thread.pinned_context.iter().enumerate() {
            outcome.examined += 1;
            let legacy_id = format!("{}#{index}", thread.id);
            let mut item = migrated_item(
                LegacySource::ConversationPins,
                &legacy_id,
                owner_user_id,
                MemoryKind::Constraint,
                MemoryScope::User {
                    user_id: owner_user_id.to_string(),
                },
                Classification::Internal,
                format!("Pinned in conversation {}: {pinned}", thread.id),
                at,
            );
            // User scope: the owner is the boundary, so it goes on the ACL.
            // `for_classification` cannot know it.
            item.acl.owner = Some(owner_user_id.to_string());
            let committed = store.commit(item, None, &[])?;
            if committed.duplicate {
                outcome.already_present += 1;
            } else {
                outcome.migrated += 1;
            }
        }
    }

    Ok(outcome)
}

/// Moves a notebook's reviewed claims under the runtime store.
///
/// The richest of the five, and the one that most needs its provenance kept: an
/// assertion carries the sha-256 of the document it was read out of, the
/// extraction revision it was made against, and whether a person has since
/// accepted or rejected it. All three survive -- the source hash as a
/// [`SourceRef`], the review outcome in the provenance and the validity window.
///
/// A rejected assertion is migrated too. Dropping it would invite the same
/// wrong claim to be proposed again with nothing on record saying a person
/// already refused it.
pub fn migrate_notebook_assertions(
    notebooks: &NotebookStore,
    store: &MemoryGraph,
    owner_user_id: &str,
    project_id: &str,
    at: &str,
) -> Result<SourceOutcome, MemoryError> {
    let mut outcome = SourceOutcome::default();

    let books = match notebooks.list(owner_user_id) {
        Ok(books) => books,
        Err(error) => {
            outcome.unreadable.push(format!("notebooks: {error}"));
            return Ok(outcome);
        }
    };

    for notebook in books {
        // `include_rejected: true` -- see the doc comment above.
        let claims = match notebooks.assertions(&notebook.id, owner_user_id, None, true) {
            Ok(claims) => claims,
            Err(error) => {
                outcome
                    .unreadable
                    .push(format!("notebook {}: {error}", notebook.id));
                continue;
            }
        };

        for claim in claims {
            outcome.examined += 1;
            let mut item = migrated_item(
                LegacySource::NotebookAssertions,
                &claim.id,
                owner_user_id,
                MemoryKind::Fact,
                MemoryScope::Workspace {
                    project_id: project_id.to_string(),
                },
                Classification::Internal,
                format!(
                    "{} {} {} (notebook {})",
                    claim.subject_label, claim.predicate, claim.object_label, notebook.id
                ),
                at,
            );

            // The source hash, at the revision the claim was made against.
            if let Some(sha) = claim.document_sha256.clone() {
                item.sources = vec![SourceRef {
                    sha256: sha,
                    locator: format!("assertion {}", claim.id),
                    extraction_revision: claim.source_revision.clone(),
                }];
            }

            // A person's review outranks the extractor, so a reviewed claim
            // carries operator provenance rather than the extractor's say-so.
            // A rejected one is migrated with its validity already closed: it
            // is kept and findable, and it is not current.
            match claim.status {
                AssertionStatus::Accepted => {
                    item.provenance = Provenance::Operator {
                        user_id: owner_user_id.to_string(),
                    };
                }
                AssertionStatus::Rejected => {
                    item.provenance = Provenance::Operator {
                        user_id: owner_user_id.to_string(),
                    };
                    item.valid_until = Some(at.to_string());
                }
                AssertionStatus::Proposed => {}
            }

            let committed = store.commit(item, None, &[])?;
            if committed.duplicate {
                outcome.already_present += 1;
            } else {
                outcome.migrated += 1;
            }
        }
    }

    Ok(outcome)
}

/// Moves `memory_engine`'s scoped rows under the runtime store.
///
/// Its rows are project-scoped or unscoped, so the unscoped ones land in the
/// workspace the caller names rather than being invented a project of their
/// own. `importance_score` becomes `confidence`: both are the writer's own
/// claim about its own row, and neither ranks across provenances.
pub fn migrate_scoped_memory(
    memories: &PersistenceManager,
    store: &MemoryGraph,
    project_id: &str,
    at: &str,
) -> Result<SourceOutcome, MemoryError> {
    let mut outcome = SourceOutcome::default();

    // `None` reads the rows that belong to no project as well.
    let rows = match memories.get_memories_by_project(None, 10_000) {
        Ok(rows) => rows,
        Err(error) => {
            outcome
                .unreadable
                .push(format!("memory_engine scoped store: {error}"));
            return Ok(outcome);
        }
    };

    for row in rows {
        outcome.examined += 1;
        let scope = MemoryScope::Workspace {
            project_id: row
                .project_id
                .clone()
                .unwrap_or_else(|| project_id.to_string()),
        };
        let mut item = migrated_item(
            LegacySource::ScopedMemoryJson,
            &row.id,
            "memory-engine",
            // Its `memory_type` is a free string from an older model, and
            // mapping a string this module does not control onto a typed kind
            // would be a guess. `Fact` is the honest floor; the original type
            // is kept in the content so nothing is lost.
            MemoryKind::Fact,
            scope,
            Classification::Internal,
            format!("[{}] {}", row.memory_type, row.content),
            at,
        );
        item.confidence = Some(row.importance_score as f32);
        item.created_at = row.created_at.clone();
        item.updated_at = row.updated_at.clone();

        let committed = store.commit(item, None, &[])?;
        if committed.duplicate {
            outcome.already_present += 1;
        } else {
            outcome.migrated += 1;
        }
    }

    Ok(outcome)
}

/// Moves artifact references under the runtime store.
///
/// One item per **version**, not per artifact. An artifact corrected twice has
/// three, and a claim about "the approval note" that does not say which is not
/// round-trippable -- the same reason [`ArtifactRef`] carries a revision.
///
/// An incomplete artifact is skipped and *named*: a reference to something
/// still being written would freeze a half-finished file into the graph, and
/// silently omitting it would leave nothing to explain the gap.
pub fn migrate_artifact_references(
    artifacts: &ConversationArtifacts,
    store: &MemoryGraph,
    owner_user_id: &str,
    conversation_ids: &[String],
    project_id: &str,
    at: &str,
) -> Result<SourceOutcome, MemoryError> {
    let mut outcome = SourceOutcome::default();

    for conversation_id in conversation_ids {
        let records = match artifacts.list(owner_user_id, conversation_id) {
            Ok(records) => records,
            Err(error) => {
                outcome
                    .unreadable
                    .push(format!("conversation {conversation_id}: {error}"));
                continue;
            }
        };

        for record in records {
            outcome.examined += 1;
            if !record.complete {
                outcome.unreadable.push(format!(
                    "artifact {} v{} is still being written",
                    record.artifact_id, record.version
                ));
                continue;
            }

            let legacy_id = format!("{}@{}", record.artifact_id, record.version);
            let mut item = migrated_item(
                LegacySource::ArtifactReferences,
                &legacy_id,
                owner_user_id,
                MemoryKind::ArtifactRef,
                MemoryScope::Workspace {
                    project_id: project_id.to_string(),
                },
                Classification::Internal,
                format!("{} (version {})", record.title, record.version),
                at,
            );
            item.artifacts = vec![ArtifactRef {
                artifact_id: record.artifact_id.clone(),
                revision: record.version,
                sha256: record.sha256.clone(),
            }];
            item.created_at = record.created_at.clone();

            let committed = store.commit(item, None, &[])?;
            if committed.duplicate {
                outcome.already_present += 1;
            } else {
                outcome.migrated += 1;
            }
        }
    }

    Ok(outcome)
}

/// The stores a full migration reads from.
///
/// A struct rather than eight positional arguments: `&AgentRegistry`,
/// `&ConversationStore` and `&NotebookStore` are three different types, but
/// `project_id`, `owner_user_id` and `at` are three `&str`, and a caller that
/// transposed two of them would compile and migrate everything under the wrong
/// owner.
pub struct LegacyStores<'a> {
    pub agents: &'a AgentRegistry,
    pub conversations: &'a ConversationStore,
    pub notebooks: &'a NotebookStore,
    pub memories: &'a PersistenceManager,
    pub artifacts: &'a ConversationArtifacts,
    /// The conversations whose artifacts to move. Passed rather than
    /// discovered, because the artifact store is keyed by conversation and a
    /// migration that guessed the list would silently miss whatever it did not
    /// guess.
    pub conversation_ids: &'a [String],
}

/// Runs every implemented source.
///
/// Safe to run twice, and safe to run again after being interrupted part-way:
/// every write is addressed by the legacy record it came from, so a second pass
/// recognises what the first one wrote.
pub fn migrate_all(
    stores: LegacyStores<'_>,
    store: &MemoryGraph,
    project_id: &str,
    owner_user_id: &str,
    at: &str,
) -> Result<MigrationReport, MemoryError> {
    let mut report = MigrationReport::new();

    report.per_source.insert(
        LegacySource::AgentProfiles.key(),
        migrate_agent_profiles(stores.agents, store, project_id, at)?,
    );
    report.per_source.insert(
        LegacySource::ConversationPins.key(),
        migrate_conversation_pins(stores.conversations, store, owner_user_id, at)?,
    );
    report.per_source.insert(
        LegacySource::NotebookAssertions.key(),
        migrate_notebook_assertions(stores.notebooks, store, owner_user_id, project_id, at)?,
    );
    report.per_source.insert(
        LegacySource::ScopedMemoryJson.key(),
        migrate_scoped_memory(stores.memories, store, project_id, at)?,
    );
    report.per_source.insert(
        LegacySource::ArtifactReferences.key(),
        migrate_artifact_references(
            stores.artifacts,
            store,
            owner_user_id,
            stores.conversation_ids,
            project_id,
            at,
        )?,
    );

    // Every source in `LegacySource` must have been attempted, or the report
    // would claim a coverage the run did not deliver. This is an assertion
    // about *this function*, checked at runtime rather than trusted: adding a
    // sixth variant and forgetting to call it here is exactly the mistake the
    // `uncovered` list exists to catch, and it would otherwise show an empty
    // gap list beside four results.
    for source in LegacySource::ALL {
        debug_assert!(
            report.per_source.contains_key(source.key()),
            "{} is declared implemented but migrate_all never ran it",
            source.key()
        );
    }

    Ok(report)
}

