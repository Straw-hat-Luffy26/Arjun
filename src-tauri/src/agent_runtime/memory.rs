//! What the workbench is allowed to remember, and for whom.
//!
//! ## Why memory is a policy problem, not a storage problem
//!
//! Every agent product eventually adds memory, and the usual shape is a single
//! store that everything reads and everything writes. That shape is wrong here
//! for a reason that has nothing to do with retrieval quality: this workbench
//! reads vendor negotiations, unreleased designs and internal correspondence,
//! and the people who may read one of those are not the people who may read the
//! next. A memory that a run for one department writes and a run for another
//! reads has moved confidential material across a boundary somebody agreed to,
//! and it has done so without a refusal, a prompt, or an audit line — because
//! from the store's point of view nothing unusual happened.
//!
//! So the store here is not a cache with a key. Every item carries the
//! classification of what it came from and an access list, and the reader is
//! checked against both. Items are also *scoped*, and a scope is not a
//! namespacing convenience:
//!
//! - [`MemoryScope::Run`] — one task's private, durable state, until retention deletion.
//! - [`MemoryScope::Workspace`] — terminology, templates and stable facts for
//!   one project. Read by every run *on that project* and by no other.
//! - [`MemoryScope::User`] — a person's preferences. Theirs alone.
//!
//! ## The promotion rule
//!
//! The dangerous operation is not writing, it is *promoting*: taking something
//! a run learned from a document and putting it somewhere later runs will read.
//! A run that reads a confidential tender and writes "the unit price is ₹4.2
//! crore" into workspace memory has published it, quietly and permanently.
//!
//! So a promotion out of run scope is refused whenever the value came from a
//! document that is not ordinary internal material, unless a person explicitly
//! approved that specific promotion. Never automatically, never because the
//! model judged it useful, and never because the value "looked like a fact
//! rather than a quote" — a paraphrase of a confidential figure is the
//! confidential figure.
//!
//! ## Why no vector store
//!
//! There is none, here or anywhere else in this crate. Recall is by scope and
//! key over a small local table. An embedding index would be a second copy of
//! the material in a form no reviewer can read, sitting outside the
//! classification checks above, and the volume of memory a workbench accumulates
//! does not need one.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};

use crate::identity::{Role, Session};
use crate::policy::Classification;

/// Who a memory item belongs to.
///
/// Untagged variants would let a workspace item deserialise as a run item on a
/// field-name collision, which is precisely the boundary this type exists to
/// hold, so the tag is explicit.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum MemoryScope {
    /// One task's own state. Never read by another run.
    Run { run_id: String },
    /// One project's shared, approved knowledge.
    Workspace { project_id: String },
    /// One person's preferences.
    User { user_id: String },
}

impl MemoryScope {
    /// True for scopes that outlive the run that wrote them.
    ///
    /// The test the promotion rule turns on: writing into run scope is private
    /// to that task and reversible by ending it, and writing anywhere else is
    /// publication.
    pub fn is_durable(&self) -> bool {
        !matches!(self, MemoryScope::Run { .. })
    }

    /// The project this scope belongs to, if any.
    pub fn project(&self) -> Option<&str> {
        match self {
            MemoryScope::Workspace { project_id } => Some(project_id.as_str()),
            _ => None,
        }
    }

    /// The filename this scope's items are stored under.
    fn file_name(&self) -> String {
        format!("scope-{}.json", crate::agent_runtime::events::digest(&scope_key(self)))
    }

    fn legacy_file_name(&self) -> String {
        match self {
            MemoryScope::Run { run_id } => format!("run-{}.json", sanitise(run_id)),
            MemoryScope::Workspace { project_id } => {
                format!("workspace-{}.json", sanitise(project_id))
            }
            MemoryScope::User { user_id } => format!("user-{}.json", sanitise(user_id)),
        }
    }
}

/// Keeps an identifier to one safe path component.
///
/// Project and user ids come from configuration and from the UI, so neither is
/// a value this process generated. Anything that is not a letter, digit, dash or
/// underscore becomes an underscore, which cannot name a parent directory.
fn sanitise(raw: &str) -> String {
    raw.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// What kind of thing is being remembered.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum MemoryKind {
    /// A run's own goal, stage and next action.
    RunState,
    /// Something the run decided and is bound by.
    Decision,
    /// An approved term and what it means on this project.
    Terminology,
    /// A document or artifact template this project uses.
    Template,
    /// A stable fact about the project — a unit convention, a site name.
    ProjectFact,
    /// How one person likes their work presented.
    Preference,
}

impl MemoryKind {
    /// Which scopes this kind may legitimately live in.
    ///
    /// A `Preference` in workspace scope would be one person's taste imposed on
    /// a project; a `Terminology` in user scope would be a shared definition
    /// only one person sees. Both are the kind of mistake that is invisible
    /// afterwards, so they are refused at the point of writing.
    fn permitted_in(self, scope: &MemoryScope) -> bool {
        match self {
            MemoryKind::RunState | MemoryKind::Decision => !scope.is_durable(),
            MemoryKind::Terminology | MemoryKind::Template | MemoryKind::ProjectFact => {
                matches!(scope, MemoryScope::Workspace { .. })
            }
            MemoryKind::Preference => matches!(scope, MemoryScope::User { .. }),
        }
    }
}

/// Where a remembered value came from.
///
/// The field the promotion rule reads. A value with no traceable origin is
/// treated as though it came from the operator, because that is the only source
/// that could have supplied something the system did not read.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "from", rename_all = "camelCase")]
pub enum MemorySource {
    /// A person typed it.
    Operator { user_id: String },
    /// A run produced it from its own reasoning, not by quoting a document.
    Run { run_id: String },
    /// It came out of an indexed document. Carries what that document was
    /// classified as, which is what decides whether it may be promoted.
    Document {
        document_sha256: String,
        page: u32,
        classification: Classification,
    },
}

impl MemorySource {
    /// The classification the *source* carried, which may exceed the item's own.
    fn source_classification(&self) -> Option<Classification> {
        match self {
            MemorySource::Document { classification, .. } => Some(*classification),
            _ => None,
        }
    }
}

/// Who may read an item.
///
/// Held on the item rather than derived at read time. Deriving it would mean a
/// later change to the clearance table silently widening what has already been
/// stored, and the whole point of writing the list down is that a reviewer can
/// see what it was when the decision was taken.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Acl {
    /// Roles cleared to read this. Empty means nobody, which is the correct
    /// reading of an item whose clearance was never established.
    pub cleared_roles: Vec<Role>,
    /// The project this is confined to. `None` only for user-scope items.
    pub project_id: Option<String>,
    /// The person this belongs to, for user-scope items.
    pub owner: Option<String>,
}

impl Acl {
    /// The list a value of this classification gets by default.
    pub fn for_classification(classification: Classification, project_id: Option<&str>) -> Self {
        Self {
            cleared_roles: classification.cleared_roles().to_vec(),
            project_id: project_id.map(str::to_string),
            owner: None,
        }
    }

    /// Whether this reader, working on this project, may see the item.
    ///
    /// Both halves are required and neither implies the other: holding the role
    /// for a vendor negotiation does not entitle somebody to *another project's*
    /// vendor negotiation, and being on the project does not confer the role.
    pub fn admits(&self, session: &Session, project_id: Option<&str>) -> bool {
        if let Some(owner) = &self.owner {
            if owner != &session.user.id {
                return false;
            }
        }
        if let Some(confined_to) = &self.project_id {
            // A reader who named no project is not thereby cleared for every
            // project. Absence of a project is not a wildcard.
            if project_id != Some(confined_to.as_str()) {
                return false;
            }
        }
        self.cleared_roles
            .iter()
            .any(|role| session.user.roles.contains(role))
    }
}

/// One thing the workbench remembers.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MemoryItem {
    pub id: String,
    pub scope: MemoryScope,
    pub kind: MemoryKind,
    /// What this is about, in a form a later run can ask for by name.
    pub key: String,
    pub value: String,
    /// How sensitive the value is. Every item has one; there is no default that
    /// means "not classified", because that is how unclassified material ends up
    /// being treated as public.
    pub classification: Classification,
    pub acl: Acl,
    pub source: MemorySource,
    /// The decision that permitted this entry to exist here, when one was
    /// needed. Stored so a reviewer can check the approval against the item
    /// rather than against a log entry that may have been written separately.
    #[serde(default)]
    pub approval: Option<ApprovalBinding>,
    /// When this stops being recalled. `None` never expires.
    #[serde(default)]
    pub expires_at: Option<String>,
    /// RFC 3339, UTC.
    pub created_at: String,
    pub updated_at: String,
    /// Prior values with their original provenance and access policy. Entries
    /// are flat (their own history is empty) and filtered independently on read.
    #[serde(default)]
    pub superseded: Vec<MemoryItem>,
}

impl MemoryItem {
    fn readable_by(&self, session: &Session, project_id: Option<&str>) -> bool {
        self.acl.admits(session, project_id)
            && self.classification.cleared_roles().iter().any(|role| session.user.roles.contains(role))
            && self.source.source_classification().is_none_or(|classification|
                classification.cleared_roles().iter().any(|role| session.user.roles.contains(role)))
            && self.approval.as_ref().is_none_or(|binding| binding.policy_version == POLICY_VERSION)
    }
    /// Whether this item has passed its expiry at the given instant.
    ///
    /// An unparseable expiry counts as expired. The alternative — treating a
    /// timestamp nobody can read as "never expires" — turns a corrupted field
    /// into an item that outlives its retention silently.
    pub fn is_expired_at(&self, now: chrono::DateTime<chrono::Utc>) -> bool {
        match &self.expires_at {
            None => false,
            Some(raw) => match chrono::DateTime::parse_from_rfc3339(raw) {
                Ok(at) => at <= now,
                Err(_) => true,
            },
        }
    }
}

/// Why a write was refused.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum MemoryError {
    #[error("Memory {key:?} conflicts with an existing source; an explicit correction is required.")]
    Conflict { key: String },
    #[error(
        "{value_kind:?} cannot be remembered in this scope: it belongs to a different kind of memory"
    )]
    WrongScope { value_kind: MemoryKind },

    #[error(
        "this came from {document_name}, which is classified {classification}. Material of that \
         classification is not promoted into shared memory automatically; a person has to approve \
         this specific entry."
    )]
    PromotionNeedsApproval {
        document_name: String,
        classification: String,
    },

    #[error(
        "{key:?} is a sensitive preference and is not remembered unless the person it belongs to \
         explicitly approves it."
    )]
    SensitivePreference { key: String },

    #[error(
        "the approval {approval_id} does not cover this entry: {field} is not what was approved.          A new approval is needed."
    )]
    ApprovalDoesNotCover { field: String, approval_id: String },

    #[error("{key:?} is not held in this scope, or you are not cleared to see it.")]
    NotFound { key: String },

    #[error("the memory for this scope could not be read or written: {detail}")]
    Storage { detail: String },
}

/// Preference keys that are never stored on a shrug.
///
/// Not a filter over the *value* — a filter over values is a guess, and a wrong
/// guess here stores a credential. These are the keys whose whole purpose is to
/// hold something personal or secret, and storing one requires the person to say
/// so for that entry.
const SENSITIVE_PREFERENCE_KEYS: &[&str] = &[
    "password",
    "passphrase",
    "token",
    "api_key",
    "apikey",
    "secret",
    "credential",
    "pin",
    "salary",
    "compensation",
    "health",
    "medical",
    "home_address",
    "personal_phone",
    "personal_email",
    "national_id",
    "aadhaar",
    "pan",
];

fn is_sensitive_preference(key: &str) -> bool {
    let normalised = key.to_ascii_lowercase().replace(['-', ' '], "_");
    SENSITIVE_PREFERENCE_KEYS
        .iter()
        .any(|sensitive| normalised.contains(sensitive))
}

/// The version of the promotion rules an approval was granted under.
///
/// Stored on every binding. An approval taken under one set of rules is not
/// evidence of consent under a different set: if the rules below change, the
/// version changes with them, and every stored approval becomes re-checkable
/// rather than silently inherited.
pub const POLICY_VERSION: u32 = 1;

/// A person's explicit go-ahead, bound to exactly what they approved.
///
/// ## Why a boolean is not enough
///
/// The obvious shape is `approved: bool`, or an approver's name. Both fail the
/// same way and the failure is silent: a person approves promoting *one* term
/// with *one* value from *one* document, the flag is stored, and then the value
/// changes — a later run rewrites it, a different source is substituted, the
/// target project is changed — and the flag still reads `true`. The record says
/// somebody approved this, and nobody approved this.
///
/// So the binding carries a hash of every input the decision was about. Before
/// the item is stored, [`Self::verify`] recomputes those hashes from the request
/// actually being made and refuses on any mismatch. An approval is thereby good
/// for one key, one value, one source, one classification and one destination —
/// and for nothing else.
///
/// The hashes are of the content, not the content itself: a reviewer can see
/// *that* the approved value is the stored value without the record carrying a
/// second copy of restricted text.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ApprovalBinding {
    /// Identifies this decision in the approval queue and the audit log.
    pub approval_id: String,
    pub approver: String,
    /// RFC 3339, UTC.
    pub at: String,
    pub key_hash: String,
    pub value_hash: String,
    /// Hash of the source the value came from — the document digest for a
    /// document, the run or operator id otherwise.
    pub source_hash: String,
    /// What the source was classified as when the decision was taken.
    pub source_classification: Classification,
    /// The scope this was approved *into*, as [`scope_key`] spells it.
    pub target_scope: String,
    pub target_project: Option<String>,
    pub policy_version: u32,
}

/// What a source hashes to, for binding purposes.
pub fn source_fingerprint(source: &MemorySource) -> String {
    match source {
        MemorySource::Operator { user_id } => crate::agent_runtime::events::digest(&format!(
            "operator:{user_id}"
        )),
        MemorySource::Run { run_id } => {
            crate::agent_runtime::events::digest(&format!("run:{run_id}"))
        }
        // The document digest and the page, because approving a figure on page
        // 12 is not approving the same figure quoted from page 40 of something
        // else.
        MemorySource::Document {
            document_sha256,
            page,
            classification,
        } => crate::agent_runtime::events::digest(&format!(
            "document:{document_sha256}:{page}:{}",
            classification.label()
        )),
    }
}

impl ApprovalBinding {
    /// Builds a binding over exactly what is being asked for.
    ///
    /// Constructed on the Rust side from the approval record and the request,
    /// never from anything a model supplied — which is what makes the hashes
    /// mean something. A model that could choose them could choose them to
    /// match.
    pub fn bind(
        approval_id: impl Into<String>,
        approver: impl Into<String>,
        request: &Remember,
    ) -> Self {
        Self {
            approval_id: approval_id.into(),
            approver: approver.into(),
            at: chrono::Utc::now().to_rfc3339(),
            key_hash: crate::agent_runtime::events::digest(&request.key),
            value_hash: crate::agent_runtime::events::digest(&request.value),
            source_hash: source_fingerprint(&request.source),
            source_classification: request
                .source
                .source_classification()
                .unwrap_or(request.classification),
            target_scope: scope_key(&request.scope),
            target_project: request.scope.project().map(str::to_string),
            policy_version: POLICY_VERSION,
        }
    }

    /// Whether this binding actually authorises the request being made.
    ///
    /// Every field is checked. Returning the *first* mismatch by name matters:
    /// a refusal that says only "not approved" sends somebody to re-approve the
    /// same thing, whereas one that says the value changed sends them to look
    /// at what changed it.
    pub fn verify(&self, request: &Remember) -> Result<(), MemoryError> {
        let mismatch = |field: &str| {
            Err(MemoryError::ApprovalDoesNotCover {
                field: field.to_string(),
                approval_id: self.approval_id.clone(),
            })
        };

        if self.policy_version != POLICY_VERSION {
            return mismatch("the policy version it was granted under");
        }
        if self.key_hash != crate::agent_runtime::events::digest(&request.key) {
            return mismatch("the key");
        }
        if self.value_hash != crate::agent_runtime::events::digest(&request.value) {
            return mismatch("the value");
        }
        if self.source_hash != source_fingerprint(&request.source) {
            return mismatch("the source it came from");
        }
        let classification = request
            .source
            .source_classification()
            .unwrap_or(request.classification);
        if self.source_classification != classification {
            return mismatch("the classification of the source");
        }
        if self.target_scope != scope_key(&request.scope) {
            return mismatch("the scope it was approved into");
        }
        if self.target_project.as_deref() != request.scope.project() {
            return mismatch("the project it was approved for");
        }
        Ok(())
    }
}

/// What a caller asks to have remembered.
#[derive(Debug, Clone)]
pub struct Remember {
    pub scope: MemoryScope,
    pub kind: MemoryKind,
    pub key: String,
    pub value: String,
    pub classification: Classification,
    pub source: MemorySource,
    /// Present only when a person approved this exact entry. See
    /// [`ApprovalBinding`] — a binding that does not verify against this
    /// request is refused, not honoured.
    pub approval: Option<ApprovalBinding>,
    /// When this stops being recalled, RFC 3339 UTC. `None` never expires.
    pub expires_at: Option<String>,
}

/// The store.
///
/// Every scope, including private task memory, is persisted atomically. The
/// promotion rules still apply only when data leaves its originating run.
#[derive(Debug, Default)]
pub struct MemoryStore {
    items: Mutex<HashMap<MemoryScope, Vec<MemoryItem>>>,
    /// Serializes load/mutate/persist/rollback as one operation.
    mutation: Mutex<()>,
    root: Option<PathBuf>,
}

/// Shared handle, as the runtime holds it.
pub type SharedMemory = Arc<MemoryStore>;

impl MemoryStore {
    /// A store with no disk behind it. Used by tests and by a run that has no
    /// data directory yet.
    pub fn in_memory() -> Self {
        Self::default()
    }

    /// Opens the store under the application's data directory.
    ///
    /// Existing files are read lazily, on first access to their scope, rather
    /// than all at start-up: a deployment with two hundred projects should not
    /// pay for two hundred file reads to answer a question about one.
    pub fn open(app_data_dir: &Path) -> Self {
        Self {
            items: Mutex::new(HashMap::new()),
            mutation: Mutex::new(()),
            root: Some(app_data_dir.join("memory")),
        }
    }

    /// Records something, or explains why it will not be.
    ///
    /// The single write path. Every rule this module exists to hold is enforced
    /// here, so there is no second entry point that could be added later without
    /// noticing what it skipped.
    pub fn remember(&self, request: Remember) -> Result<MemoryItem, MemoryError> {
        let _guard = self.mutation.lock().map_err(|_| MemoryError::Storage {
            detail: "The memory writer is unavailable.".into(),
        })?;
        self.load_if_needed(&request.scope)?;
        if !request.kind.permitted_in(&request.scope) {
            return Err(MemoryError::WrongScope {
                value_kind: request.kind,
            });
        }

        // The promotion rule. Checked against where the value *came from*, not
        // against how the caller classified it: a run that read a vendor
        // negotiation and labelled its summary `Internal` is exactly the case
        // this must catch, and trusting `request.classification` here would let
        // one mislabelling defeat the whole mechanism.
        if request.scope.is_durable() && request.approval.is_none() {
            if let Some(source_classification) = request.source.source_classification() {
                if source_classification != Classification::Internal {
                    let document_name = match &request.source {
                        MemorySource::Document {
                            document_sha256, ..
                        } => document_sha256.clone(),
                        _ => "a document".to_string(),
                    };
                    return Err(MemoryError::PromotionNeedsApproval {
                        document_name,
                        classification: source_classification.label().to_string(),
                    });
                }
            }
        }

        if matches!(request.scope, MemoryScope::User { .. })
            && request.approval.is_none()
            && is_sensitive_preference(&request.key)
        {
            return Err(MemoryError::SensitivePreference {
                key: request.key.clone(),
            });
        }

        // An approval only counts for what it was actually given for. Checked
        // *after* the rules above so a binding cannot be used to skip a check it
        // was never examined against, and before anything is written so a
        // refusal leaves no trace of the value it refused.
        if let Some(binding) = &request.approval {
            binding.verify(&request)?;
        }

        let now = chrono::Utc::now().to_rfc3339();
        let mut acl = Acl::for_classification(request.classification, request.scope.project());
        if let MemoryScope::User { user_id } = &request.scope {
            acl.owner = Some(user_id.clone());
        }
        // An item promoted on somebody's approval is not thereby widened: the
        // approval says this entry may exist, not that everyone may read it.

        let mut item = MemoryItem {
            id: format!("{}::{}", scope_key(&request.scope), request.key),
            scope: request.scope.clone(),
            kind: request.kind,
            key: request.key,
            value: request.value,
            classification: request.classification,
            acl,
            source: request.source,
            approval: request.approval.clone(),
            expires_at: request.expires_at.clone(),
            created_at: now.clone(),
            updated_at: now,
            superseded: Vec::new(),
        };

        // The previous value of this key, kept so a failed write can be undone.
        // Without it a store whose disk write failed would keep serving the new
        // value from memory while the file on disk says something else - and the
        // next start would silently revert, which is the shape of defect that
        // gets diagnosed as "the setting does not save sometimes".
        let previous: Option<MemoryItem>;
        {
            let mut table = self.lock()?;
            let bucket = table.entry(request.scope.clone()).or_default();
            match bucket.iter_mut().find(|held| held.key == item.key) {
                // Updated in place rather than appended, so a key means one
                // value and recall does not have to decide between two.
                Some(existing) => {
                    if existing.value == item.value && existing.source == item.source
                        && existing.classification == item.classification && existing.approval == item.approval
                        && existing.expires_at == item.expires_at { return Ok(existing.clone()); }
                    if existing.value != item.value && existing.source != item.source
                        && item.approval.is_none()
                        && !matches!(item.source, MemorySource::Operator { .. }) {
                        return Err(MemoryError::Conflict { key: item.key.clone() });
                    }
                    if existing.superseded.len() >= 64 {
                        return Err(MemoryError::Storage { detail: "Memory history reached its 64-revision limit; review or forget this key before updating.".into() });
                    }
                    previous = Some(existing.clone());
                    item.created_at = existing.created_at.clone();
                    item.superseded = existing.superseded.clone();
                    let mut prior = existing.clone();
                    prior.superseded.clear();
                    item.superseded.push(prior);
                    *existing = item.clone();
                }
                None => {
                    previous = None;
                    bucket.push(item.clone());
                }
            }
        }

        {
            if let Err(error) = self.persist(&request.scope) {
                // Rolled back before the error is returned, so a caller that
                // catches it and carries on is not carrying on with a value this
                // store has already disowned.
                self.rollback(&request.scope, &item.key, previous);
                return Err(error);
            }
        }
        Ok(item)
    }

    /// Puts one key back the way it was after a failed durable write.
    ///
    /// Best-effort by necessity: the lock was reachable a moment ago because the
    /// mutation went through it, and if it is not now there is nothing further
    /// this can do. It never invents an entry - an absent `previous` removes the
    /// key rather than leaving a default in its place.
    fn rollback(&self, scope: &MemoryScope, key: &str, previous: Option<MemoryItem>) {
        let Ok(mut table) = self.items.lock() else {
            return;
        };
        let Some(bucket) = table.get_mut(scope) else {
            return;
        };
        bucket.retain(|held| held.key != key);
        if let Some(restored) = previous {
            bucket.push(restored);
        }
    }

    /// Removes one item, if this reader is entitled to see it.
    ///
    /// Entitlement is checked first and deliberately: somebody who may not read
    /// a key may not delete it either, and a delete that succeeded on an item
    /// the caller could not see would be a way to probe for its existence.
    pub fn forget(
        &self,
        scope: &MemoryScope,
        key: &str,
        session: &Session,
        project_id: Option<&str>,
    ) -> Result<MemoryItem, MemoryError> {
        let _guard = self.mutation.lock().map_err(|_| MemoryError::Storage { detail: "The memory writer is unavailable.".into() })?;
        self.load_if_needed(scope)?;
        let held = self
            .recall_inner(scope, session, project_id).into_iter().find(|item| item.key == key)
            .ok_or_else(|| MemoryError::NotFound {
                key: key.to_string(),
            })?;

        {
            let mut table = self.lock()?;
            if let Some(bucket) = table.get_mut(scope) {
                bucket.retain(|item| item.key != key);
            }
        }

        {
            if let Err(error) = self.persist(scope) {
                // Put back. A delete that did not reach disk has not happened,
                // and reporting it as done would lose the item on the next start
                // for a reason nobody could reconstruct.
                self.rollback(scope, key, Some(held.clone()));
                return Err(error);
            }
        }
        Ok(held)
    }

    /// Drops everything past its expiry in one scope. Returns the keys that went.
    pub fn expire(&self, scope: &MemoryScope) -> Result<Vec<String>, MemoryError> {
        let _guard = self.mutation.lock().map_err(|_| MemoryError::Storage { detail: "The memory writer is unavailable.".into() })?;
        self.load_if_needed(scope)?;
        let now = chrono::Utc::now();
        let dropped: Vec<String>;
        let before: Vec<MemoryItem>;
        {
            let mut table = self.lock()?;
            let bucket = table.entry(scope.clone()).or_default();
            before = bucket.clone();
            dropped = bucket
                .iter()
                .filter(|item| item.is_expired_at(now))
                .map(|item| item.key.clone())
                .collect();
            bucket.retain(|item| !item.is_expired_at(now));
        }
        if dropped.is_empty() {
            return Ok(dropped);
        }
        {
            if let Err(error) = self.persist(scope) {
                if let Ok(mut table) = self.items.lock() {
                    table.insert(scope.clone(), before);
                }
                return Err(error);
            }
        }
        Ok(dropped)
    }

    /// Everything in one scope this reader may see.
    ///
    /// `project_id` is the project the *reader* is working on. Passing `None`
    /// does not widen the result — it narrows it to items confined to no
    /// project, which is what a reader outside every project should get.
    pub fn recall(
        &self,
        scope: &MemoryScope,
        session: &Session,
        project_id: Option<&str>,
    ) -> Vec<MemoryItem> {
        let Ok(_guard) = self.mutation.lock() else { return Vec::new(); };
        self.recall_inner(scope, session, project_id)
    }

    fn recall_inner(&self, scope: &MemoryScope, session: &Session, project_id: Option<&str>) -> Vec<MemoryItem> {
        if self.load_if_needed(scope).is_err() { return Vec::new(); }
        let Ok(table) = self.items.lock() else {
            // A poisoned lock means a previous writer panicked. Returning
            // nothing is the safe reading: an empty recall makes a run do the
            // work again, and a wrong one makes it act on somebody else's facts.
            return Vec::new();
        };
        table
            .get(scope)
            .map(|items| {
                let now = chrono::Utc::now();
                items
                    .iter()
                    // Expiry is applied on the way out rather than by a sweep,
                    // so an item past its retention is never returned even if
                    // nothing has swept since it lapsed.
                    .filter(|item| !item.is_expired_at(now))
                    .filter(|item| item.readable_by(session, project_id))
                    .map(|item| {
                        let mut item = item.clone();
                        item.superseded.retain(|prior| !prior.is_expired_at(now) && prior.readable_by(session, project_id));
                        item
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// One item by key, if this reader may see it.
    pub fn recall_one(
        &self,
        scope: &MemoryScope,
        key: &str,
        session: &Session,
        project_id: Option<&str>,
    ) -> Option<MemoryItem> {
        self.recall(scope, session, project_id)
            .into_iter()
            .find(|item| item.key == key)
    }

    /// Evicts the cache; durable memory remains available for recovery.
    pub fn forget_run(&self, run_id: &str) {
        let Ok(_guard) = self.mutation.lock() else { return; };
        if let Ok(mut table) = self.items.lock() {
            table.remove(&MemoryScope::Run {
                run_id: run_id.to_string(),
            });
        }
    }

    /// Explicit retention deletion. Cache eviction alone must never delete a
    /// paused task's durable memory.
    pub fn delete_run(&self, run_id: &str) -> Result<(), MemoryError> {
        let _guard = self.mutation.lock().map_err(|_| MemoryError::Storage { detail: "The memory writer is unavailable.".into() })?;
        let scope = MemoryScope::Run { run_id: run_id.into() };
        self.load_if_needed(&scope)?;
        if let Some(root) = &self.root {
            for name in [scope.file_name(), scope.legacy_file_name()] {
            match std::fs::remove_file(root.join(name)) {
                Ok(()) => {},
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {},
                Err(error) => return Err(MemoryError::Storage { detail: error.to_string() }),
            }
            }
        }
        self.lock()?.remove(&scope);
        Ok(())
    }

    /// Recovery must distinguish an empty scope from damaged storage.
    pub fn restore_run(&self, run_id: &str) -> Result<(), MemoryError> {
        let _guard = self.mutation.lock().map_err(|_| MemoryError::Storage { detail: "The memory writer is unavailable.".into() })?;
        self.load_if_needed(&MemoryScope::Run { run_id: run_id.into() })
    }

    fn lock(
        &self,
    ) -> Result<std::sync::MutexGuard<'_, HashMap<MemoryScope, Vec<MemoryItem>>>, MemoryError> {
        self.items.lock().map_err(|_| MemoryError::Storage {
            detail: "the memory table was left locked by a failed write".to_string(),
        })
    }

    /// Reads a durable scope's file the first time it is asked for.
    fn load_if_needed(&self, scope: &MemoryScope) -> Result<(), MemoryError> {
        let Some(root) = &self.root else { return Ok(()) };
        {
            let table = self.lock()?;
            if table.contains_key(scope) {
                return Ok(());
            }
        }
        let path = root.join(scope.file_name());
        let body = match std::fs::read(&path) {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => std::fs::read(root.join(scope.legacy_file_name())),
            other => other,
        };
        let loaded: Vec<MemoryItem> = match body {
            Ok(body) => serde_json::from_slice(&body).map_err(|_| MemoryError::Storage { detail: "The saved memory is unreadable; it was not replaced.".into() })?,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Vec::new(),
            Err(error) => return Err(MemoryError::Storage { detail: error.to_string() }),
        };
        if loaded.iter().any(|item| &item.scope != scope || item.superseded.iter().any(|prior| &prior.scope != scope || !prior.superseded.is_empty())) {
            return Err(MemoryError::Storage { detail: "The saved memory belongs to another scope or has invalid history.".into() });
        }
        self.lock()?.entry(scope.clone()).or_insert(loaded);
        Ok(())
    }

    /// Writes one durable scope to disk, atomically.
    fn persist(&self, scope: &MemoryScope) -> Result<(), MemoryError> {
        let Some(root) = &self.root else {
            return Ok(());
        };
        std::fs::create_dir_all(root).map_err(|error| MemoryError::Storage {
            detail: error.to_string(),
        })?;

        let body = {
            let table = self.lock()?;
            let items = table.get(scope).cloned().unwrap_or_default();
            serde_json::to_vec_pretty(&items).map_err(|error| MemoryError::Storage {
                detail: error.to_string(),
            })?
        };

        // Written to a temporary name and renamed into place, so a crash midway
        // leaves the previous file rather than half of a new one.
        let path = root.join(scope.file_name());
        let temporary = path.with_extension("json.writing");
        use std::io::Write;
        let mut file = std::fs::File::create(&temporary).map_err(|error| MemoryError::Storage { detail: error.to_string() })?;
        file.write_all(&body).and_then(|_| file.sync_all()).map_err(|error| MemoryError::Storage { detail: error.to_string() })?;
        drop(file);
        std::fs::rename(&temporary, &path).map_err(|error| MemoryError::Storage {
            detail: error.to_string(),
        })
    }
}

fn scope_key(scope: &MemoryScope) -> String {
    match scope {
        MemoryScope::Run { run_id } => format!("run:{run_id}"),
        MemoryScope::Workspace { project_id } => format!("workspace:{project_id}"),
        MemoryScope::User { user_id } => format!("user:{user_id}"),
    }
}

/// A run's own memory, in the shape the runtime keeps it.
///
/// The Rust mirror of `working-notes.ts`. It exists on this side so a run's
/// state can be persisted with the task record and handed back to a resumed
/// run — the runtime process does not survive a restart, and this does.
///
/// The caps are not repeated here: the runtime enforces them on the way in and
/// on the way out, and a second set of numbers in a second language would be
/// two things to keep in step. What this holds is whatever the runtime last
/// reported, which is by construction already bounded.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RunMemory {
    pub goal: String,
    pub stage: RunStage,
    pub decisions: Vec<RunDecision>,
    /// Citation markers, not passages. The passages are in the evidence table.
    pub evidence_ids: Vec<String>,
    pub calculation_ids: Vec<String>,
    pub artifact_ids: Vec<String>,
    pub open_questions: Vec<String>,
    pub next_action: String,
    /// Side effects that already happened. Read before a resumed run acts.
    pub completed: Vec<CompletedEffect>,
    /// Milestone checkpoints the run has crossed, in order. The
    /// resume path reads the last entry to know which gate the human
    /// approved last; the UI reads the same list to render the
    /// decision history alongside the run.
    ///
    /// ARJUN calls these "evidence-anchored decision points"; the phrase is
    /// ours, not the problem statement's.
    /// The model says "I think we are here" and a human signs off;
    /// that signature is the durable artefact, not the model's text.
    #[serde(default)]
    pub milestones: Vec<MilestoneRecord>,
    /// How many entries the runtime's caps dropped, per list.
    #[serde(default)]
    pub dropped: HashMap<String, u32>,
}

/// A milestone the run crossed, signed off by a person.
///
/// `at` is RFC 3339, UTC. `acknowledged_by` is the human's user id,
/// not their display name; the resume path uses it to attribute the
/// decision. `intent` is the step's user-visible text, copied into
/// the record so a later audit can show what was approved without
/// re-reading the plan.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MilestoneRecord {
    pub checkpoint_id: String,
    pub ordinal: u32,
    pub intent: String,
    pub acknowledged_by: String,
    /// RFC 3339, UTC.
    pub at: String,
    /// `approved` or `rejected`.
    ///
    /// Defaulted to `approved` so records written before a rejection could be
    /// expressed still load as what they were: at that point the only decision
    /// this list could hold was an approval, so reading them that way restates
    /// the truth rather than assuming it.
    #[serde(default = "approved_decision")]
    pub decision: String,
}

/// The decision a `MilestoneRecord` written before the field existed carries.
fn approved_decision() -> String {
    "approved".to_string()
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RunStage {
    pub ordinal: u32,
    pub intent: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RunDecision {
    pub what: String,
    pub because: String,
    /// RFC 3339, UTC.
    pub at: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompletedEffect {
    pub tool: String,
    pub target: String,
    /// RFC 3339, UTC.
    pub at: String,
}

impl RunMemory {
    /// Whether this side effect is already known to have happened.
    ///
    /// What makes a resumption safe rather than merely faster. A run that
    /// resumes without asking this writes the approval note twice.
    pub fn has_done(&self, tool: &str, target: &str) -> bool {
        self.completed
            .iter()
            .any(|effect| effect.tool == tool && effect.target == target)
    }

    /// True when nothing has been recorded, so a resumption has nothing to read.
    pub fn is_empty(&self) -> bool {
        self.goal.is_empty()
            && self.next_action.is_empty()
            && self.stage.ordinal == 0
            && self.decisions.is_empty()
            && self.evidence_ids.is_empty()
            && self.calculation_ids.is_empty()
            && self.artifact_ids.is_empty()
            && self.open_questions.is_empty()
            && self.completed.is_empty()
            && self.milestones.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::User;

    fn session(id: &str, roles: Vec<Role>) -> Session {
        Session::open(User::new(id, id, roles))
    }

    fn workspace(project: &str) -> MemoryScope {
        MemoryScope::Workspace {
            project_id: project.to_string(),
        }
    }

    fn term(project: &str, key: &str, value: &str) -> Remember {
        Remember {
            scope: workspace(project),
            kind: MemoryKind::Terminology,
            key: key.to_string(),
            value: value.to_string(),
            classification: Classification::Internal,
            source: MemorySource::Operator {
                user_id: "kiran".to_string(),
            },
            approval: None,
            expires_at: None,
        }
    }

    #[test]
    fn memory_from_one_project_is_not_visible_from_another() {
        // The boundary this module exists to hold. A run on Project B that can
        // read Project A's terminology has crossed it, and nothing in the audit
        // record would say so.
        let store = MemoryStore::in_memory();
        let reader = session("kiran", vec![Role::Employee]);

        store
            .remember(term("project-a", "hot-tap", "A tap made on a live line."))
            .expect("stored");

        assert_eq!(
            store
                .recall(&workspace("project-a"), &reader, Some("project-a"))
                .len(),
            1
        );
        assert!(store
            .recall(&workspace("project-b"), &reader, Some("project-b"))
            .is_empty());
    }

    #[test]
    fn one_projects_item_is_not_returned_to_a_reader_working_on_another() {
        // The same boundary from the other direction: asking Project A's scope
        // while working on Project B must not succeed either, or the confinement
        // would only be a matter of which key the caller happened to use.
        let store = MemoryStore::in_memory();
        let reader = session("kiran", vec![Role::Employee]);
        store
            .remember(term("project-a", "hot-tap", "…"))
            .expect("stored");

        assert!(store
            .recall(&workspace("project-a"), &reader, Some("project-b"))
            .is_empty());
        // And naming no project is not a wildcard.
        assert!(store
            .recall(&workspace("project-a"), &reader, None)
            .is_empty());
    }

    #[test]
    fn one_runs_memory_is_not_anothers() {
        let store = MemoryStore::in_memory();
        let reader = session("kiran", vec![Role::Employee]);
        store
            .remember(Remember {
                scope: MemoryScope::Run {
                    run_id: "run-1".to_string(),
                },
                kind: MemoryKind::Decision,
                key: "revision".to_string(),
                value: "Use the 2019 revision.".to_string(),
                classification: Classification::Internal,
                source: MemorySource::Run {
                    run_id: "run-1".to_string(),
                },
                approval: None,
                expires_at: None,
            })
            .expect("stored");

        let other = MemoryScope::Run {
            run_id: "run-2".to_string(),
        };
        assert!(store.recall(&other, &reader, None).is_empty());
    }

    #[test]
    fn restricted_document_text_is_not_promoted_into_shared_memory() {
        // The publication failure. A run reads a vendor negotiation, decides the
        // price is a useful fact, and writes it where every later run reads.
        let store = MemoryStore::in_memory();
        let refusal = store
            .remember(Remember {
                scope: workspace("project-a"),
                kind: MemoryKind::ProjectFact,
                key: "unit-price".to_string(),
                value: "The tendered unit price is ₹4.2 crore.".to_string(),
                // Mislabelled as ordinary on purpose: the rule must read the
                // source, not the label the caller chose.
                classification: Classification::Internal,
                source: MemorySource::Document {
                    document_sha256: "tender-2026".to_string(),
                    page: 12,
                    classification: Classification::VendorNegotiation,
                },
                approval: None,
                expires_at: None,
            })
            .expect_err("must be refused");

        assert!(matches!(refusal, MemoryError::PromotionNeedsApproval { .. }));
        // And nothing was stored on the way to being refused.
        let reader = session("kiran", vec![Role::Employee]);
        assert!(store
            .recall(&workspace("project-a"), &reader, Some("project-a"))
            .is_empty());
    }

    /// The promotion a person is asked to approve, as the tests use it.
    fn restricted_promotion() -> Remember {
        Remember {
            scope: workspace("project-a"),
            kind: MemoryKind::ProjectFact,
            key: "unit-price".to_string(),
            value: "The tendered unit price is ₹4.2 crore.".to_string(),
            classification: Classification::VendorNegotiation,
            source: MemorySource::Document {
                document_sha256: "tender-2026".to_string(),
                page: 12,
                classification: Classification::VendorNegotiation,
            },
            approval: None,
            expires_at: None,
        }
    }

    /// The same promotion, carrying a binding a person granted for it.
    fn approved_promotion() -> Remember {
        let mut request = restricted_promotion();
        request.approval = Some(ApprovalBinding::bind("apr-1", "asha", &request));
        request
    }

    #[test]
    fn the_same_promotion_is_allowed_once_a_person_approves_it() {
        // The rule is "not automatically", not "never". A refusal with no way
        // through would make people work around the mechanism entirely.
        let store = MemoryStore::in_memory();
        let stored = store
            .remember(approved_promotion())
            .expect("approved promotion is stored");

        // Approval permits the entry; it does not widen who may read it.
        assert_eq!(stored.classification, Classification::VendorNegotiation);
        // The legacy KnowledgeAdministrator role (kept on the enum for
        // compatibility with the historical test surface) is not a reader
        // of vendor material. Pinned here so a regression that re-enables
        // the role is caught at the ACL level.
        assert!(!stored
            .acl
            .cleared_roles
            .contains(&Role::KnowledgeAdministrator));
        // And the decision is on the item, not only in a log beside it.
        let binding = stored.approval.expect("the binding is stored");
        assert_eq!(binding.approver, "asha");
        assert_eq!(binding.approval_id, "apr-1");
        assert_eq!(binding.policy_version, POLICY_VERSION);
    }

    #[test]
    fn an_approval_does_not_cover_a_value_that_changed_after_it_was_granted() {
        // The failure a boolean cannot catch. A person approves one figure;
        // something rewrites the value; the flag still reads "approved". Every
        // field below is one that, if it drifted, would mean the stored item is
        // not the item anybody agreed to.
        let store = MemoryStore::in_memory();
        let granted = approved_promotion();
        let binding = granted.approval.clone().expect("bound");

        let mut altered_value = granted.clone();
        altered_value.value = "The tendered unit price is ₹9.9 crore.".to_string();

        let mut altered_key = granted.clone();
        altered_key.key = "headline-price".to_string();

        let mut altered_source = granted.clone();
        altered_source.source = MemorySource::Document {
            document_sha256: "some-other-tender".to_string(),
            page: 12,
            classification: Classification::VendorNegotiation,
        };

        let mut altered_classification = granted.clone();
        altered_classification.source = MemorySource::Document {
            document_sha256: "tender-2026".to_string(),
            page: 12,
            // Downgraded, which is exactly how a restricted value would be
            // smuggled past the promotion rule.
            classification: Classification::Internal,
        };

        let mut altered_target = granted.clone();
        altered_target.scope = workspace("project-b");

        let mut stale_policy = granted.clone();
        let mut stale = binding.clone();
        stale.policy_version = POLICY_VERSION + 1;
        stale_policy.approval = Some(stale);

        for (what, request) in [
            ("value", altered_value),
            ("key", altered_key),
            ("source", altered_source),
            ("classification", altered_classification),
            ("target", altered_target),
            ("policy version", stale_policy),
        ] {
            let refusal = store
                .remember(request)
                .expect_err("a changed {what} must invalidate the approval");
            assert!(
                matches!(refusal, MemoryError::ApprovalDoesNotCover { .. }),
                "changing the {what} was not caught: {refusal}"
            );
        }

        // And nothing was written on the way to any of those refusals.
        let reader = session("kiran", vec![Role::Employee]);
        assert!(store
            .recall(&workspace("project-a"), &reader, Some("project-a"))
            .is_empty());
    }

    #[test]
    fn an_approval_binding_survives_a_restart_intact() {
        // A binding that is not durable is a binding that stops being checkable
        // the moment the process ends — and the next start would either trust an
        // unverified item or refuse one a person really did approve.
        let dir = tempfile::tempdir().expect("temp dir");
        {
            let store = MemoryStore::open(dir.path());
            store.remember(approved_promotion()).expect("stored");
        }

        let reopened = MemoryStore::open(dir.path());
        let reader = session("kiran", vec![Role::Employee]);
        let recalled = reopened
            .recall_one(&workspace("project-a"), "unit-price", &reader, Some("project-a"))
            .expect("the promoted item survives");
        let binding = recalled.approval.expect("its binding survives too");

        assert_eq!(binding.approval_id, "apr-1");
        assert_eq!(binding.approver, "asha");
        assert_eq!(binding.source_classification, Classification::VendorNegotiation);
        assert_eq!(binding.target_project.as_deref(), Some("project-a"));
        // Still verifies against the request it was granted for.
        assert!(binding.verify(&approved_promotion()).is_ok());
    }

    #[test]
    fn an_expired_item_stops_being_recalled_even_before_it_is_swept() {
        let store = MemoryStore::in_memory();
        let mut lapsed = term("project-a", "old-term", "…");
        lapsed.expires_at = Some("2020-01-01T00:00:00+00:00".to_string());
        store.remember(lapsed).expect("stored");

        let reader = session("kiran", vec![Role::Employee]);
        assert!(store
            .recall(&workspace("project-a"), &reader, Some("project-a"))
            .is_empty());
    }

    #[test]
    fn an_expiry_nobody_can_parse_is_treated_as_expired() {
        // The safe reading. Treating an unreadable timestamp as "never expires"
        // turns one corrupted field into an item that outlives its retention
        // with nothing to show that it did.
        let store = MemoryStore::in_memory();
        let mut broken = term("project-a", "odd", "…");
        broken.expires_at = Some("whenever".to_string());
        store.remember(broken).expect("stored");

        let reader = session("kiran", vec![Role::Employee]);
        assert!(store
            .recall(&workspace("project-a"), &reader, Some("project-a"))
            .is_empty());
    }

    #[test]
    fn sweeping_expired_items_removes_them_durably() {
        let dir = tempfile::tempdir().expect("temp dir");
        let scope = workspace("project-a");
        {
            let store = MemoryStore::open(dir.path());
            let mut lapsed = term("project-a", "old-term", "…");
            lapsed.expires_at = Some("2020-01-01T00:00:00+00:00".to_string());
            store.remember(lapsed).expect("stored");
            store.remember(term("project-a", "kept", "…")).expect("stored");
            assert_eq!(store.expire(&scope).expect("swept"), vec!["old-term"]);
        }

        let reopened = MemoryStore::open(dir.path());
        let reader = session("kiran", vec![Role::Employee]);
        let held = reopened.recall(&scope, &reader, Some("project-a"));
        assert_eq!(held.len(), 1);
        assert_eq!(held[0].key, "kept");
    }

    #[test]
    fn deleting_needs_the_entitlement_that_reading_needs() {
        // Otherwise a delete that succeeded on an unreadable item would be a way
        // to learn the item exists.
        let store = MemoryStore::in_memory();
        store.remember(term("project-a", "hot-tap", "…")).expect("stored");

        // An outsider with no role for this material — they cannot see it, so
        // they cannot delete it either.
        let outsider = session("ravi", vec![]);
        assert!(matches!(
            store.forget(&workspace("project-a"), "hot-tap", &outsider, Some("project-a")),
            Err(MemoryError::NotFound { .. })
        ));

        let cleared = session("kiran", vec![Role::Employee]);
        assert!(store
            .forget(&workspace("project-a"), "hot-tap", &cleared, Some("project-a"))
            .is_ok());
        assert!(store
            .recall(&workspace("project-a"), &cleared, Some("project-a"))
            .is_empty());
    }

    #[test]
    fn a_failed_durable_write_leaves_no_value_behind_in_memory() {
        // The false success this prevents: the disk write fails, the caller sees
        // an error, and the store carries on serving the value anyway until the
        // next start quietly reverts it.
        let dir = tempfile::tempdir().expect("temp dir");
        let store = MemoryStore::open(dir.path());
        let reader = session("kiran", vec![Role::Employee]);
        store
            .remember(term("project-a", "hot-tap", "first"))
            .expect("the first write lands");

        // A file where the memory directory needs to be. Every later write to
        // this scope fails at `create_dir_all`.
        let blocked = MemoryStore::open(&dir.path().join("blocked"));
        std::fs::write(dir.path().join("blocked").with_extension("tmp"), b"x").ok();
        std::fs::create_dir_all(dir.path().join("blocked")).ok();
        std::fs::write(dir.path().join("blocked").join("memory"), b"not a directory")
            .expect("occupy the memory path with a file");

        let refusal = blocked.remember(term("project-b", "term", "value"));
        assert!(
            matches!(refusal, Err(MemoryError::Storage { .. })),
            "expected a storage failure, got {refusal:?}"
        );
        // Nothing readable from the store whose write failed.
        assert!(blocked
            .recall(&workspace("project-b"), &reader, Some("project-b"))
            .is_empty());

        // And the store that did write is untouched by the other's failure.
        let held = store.recall(&workspace("project-a"), &reader, Some("project-a"));
        assert_eq!(held.len(), 1);
        assert_eq!(held[0].value, "first");
    }

    #[test]
    fn a_failed_overwrite_leaves_the_previous_value_in_place() {
        let dir = tempfile::tempdir().expect("temp dir");
        let store = MemoryStore::open(dir.path());
        let reader = session("kiran", vec![Role::Employee]);
        store
            .remember(term("project-a", "hot-tap", "first"))
            .expect("the first write lands");

        // Make the scope's file unwritable by replacing the directory it lives
        // in with something a rename cannot target.
        let memory_dir = dir.path().join("memory");
        std::fs::remove_dir_all(&memory_dir).ok();
        std::fs::write(&memory_dir, b"not a directory").expect("occupy the path");

        assert!(store
            .remember(term("project-a", "hot-tap", "second"))
            .is_err());

        // The old value, not the new one: the overwrite did not happen.
        let held = store.recall(&workspace("project-a"), &reader, Some("project-a"));
        assert_eq!(held.len(), 1);
        assert_eq!(held[0].value, "first");
    }

    #[test]
    fn ordinary_internal_material_promotes_without_ceremony() {
        let store = MemoryStore::in_memory();
        assert!(store
            .remember(Remember {
                scope: workspace("project-a"),
                kind: MemoryKind::Terminology,
                key: "hot-tap".to_string(),
                value: "A tap made on a live line.".to_string(),
                classification: Classification::Internal,
                source: MemorySource::Document {
                    document_sha256: "sop".to_string(),
                    page: 4,
                    classification: Classification::Internal,
                },
                approval: None,
                expires_at: None,
            })
            .is_ok());
    }

    #[test]
    fn a_run_may_hold_what_it_may_not_publish() {
        // Reading a confidential document into the run's own state is the work.
        // Only promotion out of run scope is publication.
        let store = MemoryStore::in_memory();
        assert!(store
            .remember(Remember {
                scope: MemoryScope::Run {
                    run_id: "run-1".to_string(),
                },
                kind: MemoryKind::Decision,
                key: "price".to_string(),
                value: "The tendered unit price is ₹4.2 crore.".to_string(),
                classification: Classification::VendorNegotiation,
                source: MemorySource::Document {
                    document_sha256: "tender-2026".to_string(),
                    page: 12,
                    classification: Classification::VendorNegotiation,
                },
                approval: None,
                expires_at: None,
            })
            .is_ok());
    }

    #[test]
    fn a_sensitive_preference_is_not_remembered_on_a_shrug() {
        let store = MemoryStore::in_memory();
        for key in ["password", "personal-phone", "API_KEY"] {
            let refusal = store
                .remember(Remember {
                    scope: MemoryScope::User {
                        user_id: "kiran".to_string(),
                    },
                    kind: MemoryKind::Preference,
                    key: key.to_string(),
                    value: "something".to_string(),
                    classification: Classification::Internal,
                    source: MemorySource::Operator {
                        user_id: "kiran".to_string(),
                    },
                    approval: None,
                    expires_at: None,
                })
                .expect_err("must be refused");
            assert!(
                matches!(refusal, MemoryError::SensitivePreference { .. }),
                "{key} was not treated as sensitive"
            );
        }
    }

    #[test]
    fn an_ordinary_preference_is_remembered_without_asking() {
        // The mechanism has to stay usable for what it is actually for.
        let store = MemoryStore::in_memory();
        assert!(store
            .remember(Remember {
                scope: MemoryScope::User {
                    user_id: "kiran".to_string(),
                },
                kind: MemoryKind::Preference,
                key: "units".to_string(),
                value: "Prefers SI units in drafted notes.".to_string(),
                classification: Classification::Internal,
                source: MemorySource::Operator {
                    user_id: "kiran".to_string(),
                },
                approval: None,
                expires_at: None,
            })
            .is_ok());
    }

    #[test]
    fn one_persons_preferences_are_not_anothers() {
        let store = MemoryStore::in_memory();
        store
            .remember(Remember {
                scope: MemoryScope::User {
                    user_id: "kiran".to_string(),
                },
                kind: MemoryKind::Preference,
                key: "units".to_string(),
                value: "SI".to_string(),
                classification: Classification::Internal,
                source: MemorySource::Operator {
                    user_id: "kiran".to_string(),
                },
                approval: None,
                expires_at: None,
            })
            .expect("stored");

        let someone_else = session("asha", vec![Role::Employee]);
        assert!(store
            .recall(
                &MemoryScope::User {
                    user_id: "kiran".to_string()
                },
                &someone_else,
                None
            )
            .is_empty());
    }

    #[test]
    fn every_item_carries_a_classification_and_an_access_list() {
        let store = MemoryStore::in_memory();
        let item = store
            .remember(term("project-a", "hot-tap", "…"))
            .expect("stored");

        assert_eq!(item.classification, Classification::Internal);
        assert!(!item.acl.cleared_roles.is_empty());
        assert_eq!(item.acl.project_id.as_deref(), Some("project-a"));
    }

    #[test]
    fn a_reader_without_the_role_sees_nothing() {
        let store = MemoryStore::in_memory();
        store
            .remember(Remember {
                scope: workspace("project-a"),
                kind: MemoryKind::ProjectFact,
                key: "terms".to_string(),
                value: "Payment is net 60.".to_string(),
                classification: Classification::VendorNegotiation,
                source: MemorySource::Operator {
                    user_id: "kiran".to_string(),
                },
                approval: None,
                expires_at: None,
            })
            .expect("stored");

        // The legacy "auditor" role grants nothing in the active product
        // and is not cleared for any classification; the assertion that
        // this outsider reads nothing catches a regression that re-enables
        // a legacy role.
        let outsider = session("ravi", vec![Role::Auditor]);
        assert!(store
            .recall(&workspace("project-a"), &outsider, Some("project-a"))
            .is_empty());
    }

    #[test]
    fn a_kind_cannot_be_filed_in_a_scope_it_does_not_belong_to() {
        let store = MemoryStore::in_memory();
        let refusal = store
            .remember(Remember {
                scope: workspace("project-a"),
                kind: MemoryKind::Preference,
                key: "units".to_string(),
                value: "SI".to_string(),
                classification: Classification::Internal,
                source: MemorySource::Operator {
                    user_id: "kiran".to_string(),
                },
                approval: None,
                expires_at: None,
            })
            .expect_err("must be refused");

        assert!(matches!(refusal, MemoryError::WrongScope { .. }));
    }

    #[test]
    fn writing_a_key_twice_updates_it_rather_than_storing_two_answers() {
        let store = MemoryStore::in_memory();
        let reader = session("kiran", vec![Role::Employee]);
        store
            .remember(term("project-a", "hot-tap", "first"))
            .expect("stored");
        store
            .remember(term("project-a", "hot-tap", "second"))
            .expect("stored");

        let recalled = store.recall(&workspace("project-a"), &reader, Some("project-a"));
        assert_eq!(recalled.len(), 1);
        assert_eq!(recalled[0].value, "second");
    }

    #[test]
    fn updating_after_restart_preserves_other_keys_and_provenance() {
        let dir = tempfile::tempdir().unwrap();
        let store = MemoryStore::open(dir.path());
        store.remember(term("project-a", "revision", "2019")).unwrap();
        store.remember(term("project-a", "unit", "mm")).unwrap();
        drop(store);
        let store = MemoryStore::open(dir.path());
        let corrected = store.remember(term("project-a", "revision", "2026")).unwrap();
        assert_eq!(corrected.superseded.len(), 1);
        assert_eq!(corrected.superseded[0].value, "2019");
        assert_eq!(store.remember(term("project-a", "revision", "2026")).unwrap(), corrected);
        let reader = session("kiran", vec![Role::Employee]);
        assert_eq!(store.recall(&workspace("project-a"), &reader, Some("project-a")).len(), 2);
        assert!(store.recall(&workspace("project-a"), &session("kiran", vec![]), Some("project-a")).is_empty());
    }

    #[test]
    fn another_source_cannot_silently_replace_a_correction() {
        let store = MemoryStore::in_memory();
        store.remember(term("project-a", "revision", "2026")).unwrap();
        let mut stale = term("project-a", "revision", "2019");
        stale.source = MemorySource::Run { run_id: "other".into() };
        assert!(matches!(store.remember(stale), Err(MemoryError::Conflict { .. })));
    }

    #[test]
    fn corrupt_memory_is_not_overwritten_and_colliding_names_stay_separate() {
        let dir = tempfile::tempdir().unwrap();
        let store = MemoryStore::open(dir.path());
        store.remember(term("project/a", "revision", "2019")).unwrap();
        store.remember(term("project_a", "revision", "2026")).unwrap();
        assert_ne!(workspace("project/a").file_name(), workspace("project_a").file_name());
        let path = dir.path().join("memory").join(workspace("project/a").file_name());
        std::fs::write(&path, b"broken").unwrap();
        let reopened = MemoryStore::open(dir.path());
        assert!(matches!(reopened.remember(term("project/a", "revision", "2027")), Err(MemoryError::Storage { .. })));
        assert_eq!(std::fs::read(path).unwrap(), b"broken");
    }

    #[test]
    fn all_scopes_survive_restart_until_explicit_retention_deletion() {
        let dir = tempfile::tempdir().expect("temp dir");
        let reader = session("kiran", vec![Role::Employee]);

        {
            let store = MemoryStore::open(dir.path());
            store
                .remember(term("project-a", "hot-tap", "A tap made on a live line."))
                .expect("stored");
            store
                .remember(Remember {
                    scope: MemoryScope::Run {
                        run_id: "run-1".to_string(),
                    },
                    kind: MemoryKind::Decision,
                    key: "revision".to_string(),
                    value: "2019".to_string(),
                    classification: Classification::Internal,
                    source: MemorySource::Run {
                        run_id: "run-1".to_string(),
                    },
                    approval: None,
                    expires_at: None,
                })
                .expect("stored");
        }

        let reopened = MemoryStore::open(dir.path());
        assert_eq!(
            reopened
                .recall(&workspace("project-a"), &reader, Some("project-a"))
                .len(),
            1
        );
        assert_eq!(reopened
            .recall(
                &MemoryScope::Run {
                    run_id: "run-1".to_string()
                },
                &reader,
                None
            )
            .len(), 1);
        reopened.delete_run("run-1").unwrap();
        assert!(MemoryStore::open(dir.path()).recall(&MemoryScope::Run { run_id: "run-1".into() }, &reader, None).is_empty());
    }

    #[test]
    fn a_project_id_cannot_name_a_path_outside_the_memory_directory() {
        let scope = workspace("../../etc/passwd");
        assert_eq!(
            Path::new(&scope.file_name()).components().count(),
            1,
            "a project id became more than one path component"
        );
    }

    #[test]
    fn a_resumed_run_can_tell_what_it_already_did() {
        let memory = RunMemory {
            completed: vec![CompletedEffect {
                tool: "create_docx".to_string(),
                target: "approval-note.docx".to_string(),
                at: "2026-08-28T09:15:00+00:00".to_string(),
            }],
            ..RunMemory::default()
        };

        assert!(memory.has_done("create_docx", "approval-note.docx"));
        assert!(!memory.has_done("create_docx", "something-else.docx"));
        assert!(!memory.is_empty());
    }
}
