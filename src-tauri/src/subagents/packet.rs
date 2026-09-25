//! What a parent hands a child, and deliberately what it does not.
//!
//! ## A fresh context, not a transcript
//!
//! Requirement 2 is that a child receives a fresh context rather than the whole
//! parent transcript. That is partly about cost — a 7B model with a 8k window
//! cannot hold twenty turns of parent history and do useful work — and mostly
//! about blast radius.
//!
//! A parent transcript contains every passage the parent retrieved, including
//! ones a narrower child has no business seeing, and every instruction the
//! parent was given, including any a poisoned document managed to insert. Handing
//! that down would make every child as exposed as the parent and defeat the
//! point of a narrower worker.
//!
//! So a packet carries an **objective** in the parent's own words and
//! **references** to inputs. Not contents. A child that needs a passage
//! retrieves it itself, under its own clearance, through the same gateway — and
//! what comes back is what *it* is allowed to see, not what the parent was.
//!
//! ## Why the policy hash travels with it
//!
//! So the result can be checked against the constraints the child was actually
//! given. A result that arrives claiming a tool the packet did not grant is a
//! result from a child that was not running the policy this parent sent, and
//! that is worth being able to detect rather than assume away.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::orchestrator::tools::ToolName;
use crate::policy::Classification;

use super::inherit::EffectivePolicy;
use super::profile::{Limits, SchemaKind};

/// One thing a child is pointed at.
///
/// Every variant is a **reference**. There is deliberately no variant carrying
/// text: a packet that could hold a passage is a packet through which a
/// parent's retrieved material reaches a child that may not be cleared for it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", tag = "kind")]
pub enum InputRef {
    /// A document in the knowledge base, by content hash. The child retrieves
    /// it under its own clearance and may legitimately get less than the parent.
    Document {
        sha256: String,
        /// Absent means the whole document.
        page: Option<u32>,
    },
    /// A file in the parent run's workspace, relative to the run root.
    WorkspaceFile { path: String },
    /// A passage the parent already cited, by its marker. The child resolves it
    /// itself; the marker is a name, not the text.
    Evidence { marker: usize },
    /// An expression to check, which is the one case where the value *is* the
    /// reference — a calculation has nothing behind it to point at.
    Expression { expression: String },
    /// A file this task produced, at an exact revision.
    ///
    /// The counterpart to `Document` for the other direction: a reviewer is
    /// pointed at what the run wrote and re-opens it itself rather than being
    /// handed the bytes. The revision and the hash both, for the same reason
    /// [`crate::knowledge::graph::runtime_memory::ArtifactRef`] carries both — a
    /// claim about "the approval note" that does not say which revision is not
    /// checkable afterwards.
    Artifact {
        artifact_id: String,
        revision: u32,
        sha256: String,
    },
    /// Something another worker published to this task's shared memory.
    ///
    /// How a parent points B at A's result without copying it: the id and the
    /// revision travel, and B reads the item itself through the graph under its
    /// own clearance. See [`super::graph_io`].
    GraphItem { item_id: String, revision: u64 },
    /// What a retrieval job may read, pinned when it was queued (P07).
    ///
    /// The index clock at queue time and the parent's notebook selection. A
    /// job that waits behind a model swap reads the corpus it was queued
    /// against: a document added meanwhile is not read, a version superseded
    /// meanwhile is read and flagged, and a source withdrawn or revoked
    /// meanwhile is not read — authorisation is always checked when the job
    /// runs, never carried from when it was queued.
    RetrievalScope {
        pinned_revision: i64,
        #[serde(default)]
        notebook_id: Option<String>,
        #[serde(default)]
        source_sha256s: Vec<String>,
    },
}

impl InputRef {
    /// How this reads in a trace, without resolving anything.
    pub fn describe(&self) -> String {
        match self {
            InputRef::Document { sha256, page } => match page {
                Some(page) => format!("document {}… page {page}", &sha256[..8.min(sha256.len())]),
                None => format!("document {}…", &sha256[..8.min(sha256.len())]),
            },
            InputRef::WorkspaceFile { path } => format!("workspace file {path}"),
            InputRef::Evidence { marker } => format!("[E{marker}]"),
            InputRef::Expression { expression } => format!("expression {expression:?}"),
            InputRef::Artifact {
                artifact_id,
                revision,
                ..
            } => format!("artifact {artifact_id} revision {revision}"),
            InputRef::RetrievalScope { pinned_revision, notebook_id, source_sha256s } => match notebook_id {
                Some(notebook) => format!(
                    "retrieval scope at index revision {pinned_revision}, notebook {notebook} ({} source(s))",
                    source_sha256s.len()
                ),
                None => format!("retrieval scope at index revision {pinned_revision}"),
            },
            InputRef::GraphItem { item_id, revision } => {
                format!("shared memory item {item_id} revision {revision}")
            }
        }
    }
}

/// The routing a child was given, and the policy it was given it under.
///
/// Recorded rather than enforced here. The definition's model binding is
/// carried so a trace can show whether the model a child was routed to is one
/// its definition allows (`within_eligible`); holding routing *to* that binding
/// is the scheduler's (plan P03), and saying `true` without anything having
/// checked it would be the kind of claim this record exists to prevent.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelPolicy {
    /// The model role the definition asks for.
    pub role: String,
    /// The definition's preference order: default, then fallbacks.
    #[serde(default)]
    pub preferred_model_ids: Vec<String>,
    /// Every model the definition may be routed to. Empty means any model
    /// registered for the role.
    #[serde(default)]
    pub eligible_model_ids: Vec<String>,
    /// Why routing chose what it chose, from `certification::choose`.
    #[serde(default)]
    pub routing_reason: String,
    #[serde(default)]
    pub cheaper_than_parent: bool,
    /// Whether the routed model is inside `eligible_model_ids` (or that set is
    /// empty). Computed, not asserted.
    #[serde(default)]
    pub within_eligible: bool,
}

/// The work order handed to a child.
///
/// ## Where each part of the job contract lives
///
/// Plan P01 names what a job must carry. Every item is a field or is derived
/// from one, and none is carried twice:
///
/// | Contract item | Field |
/// |---|---|
/// | agent | `agent_id` (registry id, never a display name) |
/// | immutable definition version | `definition_version`, `definition_origin`, `instructions_sha256` |
/// | role capability | `capability` (from the output schema) |
/// | task / run / attempt / job | `task_id`, `parent_run_id`, `attempt_id`, [`Self::job_id`] (= `idempotency_key`); `child_id` is this one execution |
/// | model policy | `model_policy`, `model_id` |
/// | skill hashes | `skills` |
/// | effective tools | `allowed_tools` (already the intersection with the parent) |
/// | sharing policy | `shared_with_task`, `classification_ceiling` |
/// | input references | `inputs` (references only, never contents) |
/// | expected output schema | `required_schema`, `deliverable` |
/// | dependency revisions | `requirement` (graph revision) and the revisions inside `inputs` |
/// | cancellation | `deadline`, enforced by the manager's timeout and the worker's `Stopping` |
/// | resource limits | `limits` |
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ChildTaskPacket {
    pub child_id: String,
    pub parent_run_id: String,
    /// The agent this child *is*, from the deployment's registry.
    ///
    /// Stable across everything: the child id belongs to one attempt at one
    /// piece of work, and this belongs to the worker for as long as the
    /// deployment has it. Memory is keyed by it — see [`super::graph_io`] — so
    /// what a retriever learned on Tuesday is still attributed to the retriever
    /// on Wednesday, under a different model if the binding moved.
    ///
    /// Defaulted so an event written before this field existed still parses.
    #[serde(default)]
    pub agent_id: String,
    /// The registry version this child was dispatched under.
    ///
    /// Resolved once, when the child was sent, and never re-read — see
    /// [`super::definitions`]. `None` for a child run from a bundled profile
    /// with no registry in reach, and for an event written before this field
    /// existed; both are honestly "no registry version", and neither is `0`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub definition_version: Option<u64>,
    /// `registry` or `bundled-profile`. Where the definition above came from.
    #[serde(default)]
    pub definition_origin: String,
    /// The worker that performs this child: a capability key derived from the
    /// output schema, never from a display name.
    #[serde(default)]
    pub capability: String,
    /// What the child's own model is told it is for, as pinned at dispatch.
    ///
    /// Carried on the packet so the worker reads the definition the child was
    /// sent under rather than the text it was constructed with at start-up.
    /// Skipped when serialising: the event log records the hash below, and a
    /// trace read by more people than the run should not carry a role body.
    #[serde(default, skip_serializing)]
    pub instructions: String,
    /// The sha-256 of `instructions`, which is what the trace records.
    #[serde(default)]
    pub instructions_sha256: String,
    /// Whether this child's publications are readable by its task's other
    /// agents. Carried from the definition; see
    /// [`super::definitions::ResolvedDefinition::shared_with_task`] for why it
    /// is not yet enforced at publication.
    #[serde(default)]
    pub shared_with_task: bool,
    /// The skills the definition was bound to, each at the exact bytes it was
    /// bound to. Pinned with the rest of the definition, so a skill edited
    /// while this child runs is detectable against this record.
    ///
    /// Carried, not yet loaded: the child loop does not load skills today, and
    /// this records what the definition *named* rather than claiming it was in
    /// the child's context.
    #[serde(default)]
    pub skills: Vec<crate::agents::SkillBinding>,
    /// Which attempt at the parent run sent this child. Empty for a parent
    /// with no checkpoint yet, and for a packet written before this existed.
    #[serde(default)]
    pub attempt_id: String,
    /// The routing this child was given, and the definition's model policy.
    /// `None` for a packet written before this existed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_policy: Option<ModelPolicy>,
    /// The task this child is part of, which is the parent's.
    ///
    /// The key the shared memory scope is built from, and therefore the reason
    /// two workers on one task can see each other's results at all. Distinct
    /// from `parent_run_id` so a task that outlives one run keeps one memory.
    #[serde(default)]
    pub task_id: String,
    /// What the child must hand back, in the parent's words.
    ///
    /// Separate from the objective on purpose. The objective is what to do
    /// ("find the passages about seal wear"); this is what counts as done
    /// ("a citation for each figure the note will quote"). The parent's
    /// completion check reads this, and a child that returned something else
    /// has not finished — see [`crate::agent_runtime::completion`].
    #[serde(default)]
    pub deliverable: String,
    /// The graph position this child's inputs were authorised at.
    ///
    /// How one worker is made to wait for another: the parent hands B the
    /// revision A's result landed at, and B does not start until the shared
    /// memory holds it. `Latest` is the ordinary case, for a worker with no
    /// sibling to wait for.
    #[serde(default = "latest_requirement")]
    pub requirement: super::graph_io::Requirement,
    /// The model this child was actually routed to.
    ///
    /// On the work order rather than on the policy, because it is a decision
    /// about this piece of work rather than a constraint on what the child may
    /// do — see `certification::choose`, which is where it is made, and
    /// `scheduling`, which is what reserves it. `None` means the role needs no
    /// model resident, which for a worker that only runs the calculation engine
    /// is the truth rather than an omission.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_id: Option<String>,
    /// What makes creating the same child twice harmless. Derived by the parent
    /// from the work rather than generated, so two attempts at one piece of
    /// work agree without coordinating. See [`derive_idempotency_key`].
    pub idempotency_key: String,
    pub profile: String,
    /// What the child is for, in the parent's words. Prose, not instructions
    /// copied out of a document.
    pub objective: String,
    pub inputs: Vec<InputRef>,
    /// Exactly what this child may call. Already the intersection.
    pub allowed_tools: Vec<ToolName>,
    pub limits: Limits,
    pub required_schema: SchemaKind,
    /// The most sensitive material this child may touch.
    pub classification_ceiling: Classification,
    /// The digest of the policy this was derived from, so a result can be
    /// checked against the constraints the child was actually sent.
    pub policy_hash: String,
    pub created_at: DateTime<Utc>,
    /// When this child must stop.
    pub deadline: DateTime<Utc>,
}

impl ChildTaskPacket {
    /// Builds the packet from a policy that has already been narrowed.
    ///
    /// Takes the policy rather than the profile, so there is no path by which a
    /// packet is built from a profile's requests instead of from what the child
    /// was actually granted.
    pub fn new(
        child_id: impl Into<String>,
        parent_run_id: impl Into<String>,
        idempotency_key: impl Into<String>,
        objective: impl Into<String>,
        inputs: Vec<InputRef>,
        policy: &EffectivePolicy,
        now: DateTime<Utc>,
    ) -> Self {
        let deadline = now
            + chrono::Duration::try_seconds(policy.limits.max_duration_seconds as i64)
                .unwrap_or_else(|| chrono::Duration::minutes(5));
        let parent_run_id = parent_run_id.into();
        Self {
            child_id: child_id.into(),
            // A task that says nothing else is the run it belongs to. Filled in
            // rather than left empty, because the shared memory scope is built
            // from it and an empty scope key is one every task would share.
            task_id: parent_run_id.clone(),
            parent_run_id,
            agent_id: String::new(),
            definition_version: None,
            definition_origin: String::new(),
            capability: String::new(),
            instructions: String::new(),
            instructions_sha256: String::new(),
            shared_with_task: false,
            skills: Vec::new(),
            attempt_id: String::new(),
            model_policy: None,
            deliverable: String::new(),
            requirement: latest_requirement(),
            model_id: None,
            idempotency_key: idempotency_key.into(),
            profile: policy.profile.clone(),
            objective: objective.into(),
            inputs,
            allowed_tools: policy.tools.clone(),
            limits: policy.limits,
            required_schema: policy.required_schema,
            classification_ceiling: policy.classification_ceiling,
            policy_hash: policy.inherited_hash.clone(),
            created_at: now,
            deadline,
        }
    }

    /// Fills in the identities and the contract a production dispatch carries.
    ///
    /// A builder rather than four more positional arguments, and separate from
    /// [`Self::new`] for a reason worth stating: `new` builds a packet from a
    /// *policy*, which is the part that decides what a child may do, and this
    /// adds the part that decides what it is *for*. Getting the first wrong is a
    /// permissions bug; getting the second wrong is a wasted worker.
    pub fn assigned_to(
        mut self,
        agent_id: impl Into<String>,
        task_id: impl Into<String>,
        deliverable: impl Into<String>,
        requirement: super::graph_io::Requirement,
    ) -> Self {
        self.agent_id = agent_id.into();
        let task_id = task_id.into();
        if !task_id.trim().is_empty() {
            self.task_id = task_id;
        }
        self.deliverable = deliverable.into();
        self.requirement = requirement;
        self
    }

    /// Records the model this child was routed to.
    pub fn routed_to(mut self, model_id: Option<String>) -> Self {
        self.model_id = model_id;
        self
    }

    /// Stamps the definition this child was dispatched under.
    ///
    /// Copied, not referenced: the packet holds the definition as it stood when
    /// the child was sent, so an edit saved while the child runs changes the
    /// registry and not this child. See [`super::definitions`].
    ///
    /// For a registry definition this sets `agent_id` too, because the id a
    /// child's memory is attributed to and the id whose version it pinned must
    /// be the same id — a packet naming one agent and carrying another's version
    /// would be a provenance record that contradicts itself. A bundled profile
    /// has no registry id to insist on, so it leaves whatever
    /// [`Self::assigned_to`] set, which is what every dispatch did before
    /// definitions were resolved.
    pub fn pinned_to(mut self, definition: &super::definitions::ResolvedDefinition) -> Self {
        if definition.origin == super::definitions::DefinitionOrigin::Registry {
            self.agent_id = definition.agent_id.clone();
        }
        self.definition_version = definition.definition_version;
        self.definition_origin = definition.origin.as_str().to_string();
        self.capability = definition.capability.clone();
        self.instructions = definition.instructions().to_string();
        self.instructions_sha256 = definition.instructions_sha256();
        self.shared_with_task = definition.shared_with_task;
        self.skills = definition.skills.clone();
        self
    }

    /// Records which attempt at the parent run sent this child.
    pub fn attempted_in(mut self, attempt_id: impl Into<String>) -> Self {
        self.attempt_id = attempt_id.into();
        self
    }

    /// Records the routing this child was given, under its definition's policy.
    pub fn governed_by(mut self, policy: ModelPolicy) -> Self {
        self.model_policy = Some(policy);
        self
    }

    /// The identity of this piece of *work*, across every attempt at it.
    ///
    /// The idempotency key, named for what it is in the job contract: two
    /// dispatches of the same objective, profile and inputs within one run are
    /// one job, and the ledger returns the first one's answer to the second.
    /// `child_id` is one execution of it.
    pub fn job_id(&self) -> &str {
        &self.idempotency_key
    }

    /// Whether the deadline has passed.
    pub fn is_expired(&self, now: DateTime<Utc>) -> bool {
        now >= self.deadline
    }

    /// One line for the trace. Carries no objective text and no input contents,
    /// because a trace is read by more people than a run is.
    pub fn describe(&self) -> String {
        format!(
            "{} on {} input(s), {} tool(s), {} turn(s) at most",
            self.profile,
            self.inputs.len(),
            self.allowed_tools.len(),
            self.limits.max_turns
        )
    }
}

/// What a packet requires of the shared memory when nothing else is said.
///
/// A free function because `serde(default = ...)` needs one, and it is the
/// honest default: a worker with no sibling to wait for reads whatever is there.
fn latest_requirement() -> super::graph_io::Requirement {
    super::graph_io::Requirement::Latest
}

/// The key two attempts at the same piece of work compute independently.
///
/// Over the parent run, the profile and the work itself — so retrying after an
/// ambiguous failure finds the existing child rather than starting a second
/// one, and two different objectives under one profile are two children.
pub fn derive_idempotency_key(
    parent_run_id: &str,
    profile: &str,
    objective: &str,
    inputs: &[InputRef],
) -> String {
    let mut hasher = Sha256::new();
    let mut field = |value: &str| {
        hasher.update(value.as_bytes());
        hasher.update(b"\x1f");
    };
    field(parent_run_id);
    field(profile);
    field(objective);
    for input in inputs {
        field(&serde_json::to_string(input).unwrap_or_default());
    }
    format!("{:x}", hasher.finalize())
}
