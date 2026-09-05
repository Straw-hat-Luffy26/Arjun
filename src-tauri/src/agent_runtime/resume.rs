//! Deriving the world a resumption is checked against, and taking checkpoints.
//!
//! [`super::events::checkpoint`] owns the *decision* — given a checkpoint and a
//! description of the world, may this run continue. This module owns the two
//! things either side of that decision: computing what the world is now, and
//! writing down what it was.
//!
//! The split is deliberate. A comparison that also gathers its own inputs is a
//! comparison that cannot be tested against inputs it would refuse, and the
//! refusals are the whole point.
//!
//! ## What the hashes cover, and why they are hashes
//!
//! Three facts decide whether a stopped run may carry on, and all three can
//! change while it is not running:
//!
//! - **Policy.** The person's roles, the material's classification, and the
//!   machine's sovereignty mode. A run authorised under one set of these and
//!   continued under another has been authorised by nobody.
//! - **Plan.** What the run is permitted to do and how much of it. Derived
//!   deterministically from the prompt, so re-deriving at resume time gives back
//!   the plan the run was actually held to — and a different answer means the
//!   derivation itself changed under it.
//! - **Workspace.** The directory the run owns. If it is gone or somewhere else,
//!   the files the run was working on cannot be identified, and continuing could
//!   write over something that is not them.
//!
//! They are stored as digests rather than as values because a checkpoint is read
//! by the recovery path before anybody has signed in. A record naming a person's
//! roles and a document's classification in the clear would leak the shape of
//! the work to whoever could read the file.

use std::path::Path;

use serde::{Deserialize, Serialize};

use super::events::{digest, RunState, TaskEventLog, WorldNow};
use crate::identity::Session;
use crate::policy::Classification;

/// One hash over everything that decides whether an action is permitted.
///
/// Combined rather than compared field by field: the question at resume time is
/// *is any of this different*, and three separate comparisons is three places
/// for the fourth to be forgotten when a fourth input is added.
///
/// The roles are sorted before hashing, because role order is an accident of how
/// a directory was written and two identical people should not produce two
/// different hashes.
pub fn policy_hash(
    session: &Session,
    classification: Option<Classification>,
    sovereignty_mode: &str,
) -> String {
    let mut roles: Vec<String> = session
        .user
        .roles
        .iter()
        .map(|role| format!("{role:?}"))
        .collect();
    roles.sort();

    digest(&format!(
        "user={}|roles={}|classification={}|mode={}|department={}",
        session.user.id,
        roles.join(","),
        classification
            .map(|c| c.label().to_string())
            .unwrap_or_else(|| "unclassified".to_string()),
        sovereignty_mode,
        session.user.department.as_deref().unwrap_or_default(),
    ))
}

/// The plan a prompt produces, hashed.
///
/// Re-derived from the prompt rather than read from anywhere, because
/// [`super::planning::derive`] is deterministic over the prompt and that is what
/// makes this checkable: if the derivation changes in a later build, the hash
/// changes, and a run planned under the old rules is refused rather than
/// silently continued under the new ones.
pub fn plan_hash_of(prompt: &str) -> String {
    let derived = super::planning::derive(prompt);
    let mut tools: Vec<&str> = derived
        .budget
        .permitted_tools
        .iter()
        .map(|tool| tool.as_str())
        .collect();
    tools.sort_unstable();

    digest(&format!(
        "steps={}|maxSteps={}|repeat={}|seconds={}|tools={}",
        derived.steps.len(),
        derived.budget.max_steps,
        derived.budget.repeat_limit,
        derived.budget.max_duration.as_secs(),
        tools.join(","),
    ))
}

/// A run's working directory, as an identity rather than a path.
///
/// `None` when the directory is not there. The distinction between "gone" and
/// "moved" is deliberately not drawn: both mean the files the run was working on
/// cannot be identified, and both must refuse.
pub fn workspace_hash_of(root: &Path) -> Option<String> {
    if !root.is_dir() {
        return None;
    }
    // Canonicalised, so a path reached by a different route is recognised as the
    // same directory, and a symlink repointed elsewhere is not.
    let resolved = root.canonicalize().ok()?;
    Some(digest(&resolved.to_string_lossy()))
}

/// Everything the resume check needs, gathered from the live application.
///
/// A struct rather than nine arguments because every caller passes the same
/// nine, and a list that long is one where two of them eventually get swapped.
pub struct ResumeContext<'a> {
    pub session: &'a Session,
    pub prompt: &'a str,
    pub classification: Option<Classification>,
    pub sovereignty_mode: &'a str,
    pub workspace_root: &'a Path,
    pub model_available: bool,
    /// The person the run belongs to, from its own record.
    pub owner: &'a str,
    pub ended: bool,
    pub state: RunState,
}

impl ResumeContext<'_> {
    /// The world as it is now, ready to be compared against a checkpoint.
    pub fn world(&self) -> WorldNow {
        WorldNow {
            policy_hash: policy_hash(self.session, self.classification, self.sovereignty_mode),
            plan_hash: plan_hash_of(self.prompt),
            workspace_hash: workspace_hash_of(self.workspace_root),
            model_available: self.model_available,
            same_operator: self.session.user.id == self.owner,
            ended: self.ended,
            state: self.state,
        }
    }
}

/// A new attempt at the same logical task.
///
/// The task id never changes — that is what makes a resumption a continuation
/// rather than a second run that happens to look similar. The attempt id is new
/// every time, so the trace can show that the same task was picked up again and
/// how many times.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Attempt {
    pub run_id: String,
    pub attempt_id: String,
    /// Why a person said to continue it. Their words, kept short and shown back
    /// on the trace — a resumption nobody can explain later is one nobody can
    /// audit.
    pub operator_intent: String,
    /// The durable event the resumption carries on after.
    pub from_seq: i64,
    /// RFC 3339, UTC.
    pub at: String,
}

/// Longest operator note kept on a resumption.
///
/// Bounded because it reaches a durable event, and an unbounded field on a
/// durable event is a way to put a document into a record meant to hold none.
pub const MAX_INTENT_CHARS: usize = 500;

impl Attempt {
    pub fn new(run_id: &str, operator_intent: &str, from_seq: i64) -> Self {
        let trimmed: String = operator_intent
            .trim()
            .chars()
            .take(MAX_INTENT_CHARS)
            .collect();
        Self {
            run_id: run_id.to_string(),
            attempt_id: uuid::Uuid::new_v4().to_string(),
            operator_intent: trimmed,
            from_seq,
            at: chrono::Utc::now().to_rfc3339(),
        }
    }
}

/// The parts of a checkpoint that do not change during one attempt.
///
/// Established once, when a run starts, from state the deep loop does not carry:
/// the session that authorised it, the prompt it was planned from, the directory
/// it owns. Holding them lets a checkpoint be taken after every tool result
/// without the tool path needing to know any of that.
///
/// Everything here is a hash or an identifier, for the same reason the
/// checkpoint itself is: it is held in memory next to a run and written into a
/// record read before sign-in.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckpointSeed {
    pub attempt_id: String,
    /// The exact lease acquired by this attempt, never refreshed from a newer worker.
    pub lease: super::events::Lease,
    pub objective: String,
    pub conversation_id: String,
    pub message_id: String,
    pub deadline_ms: i64,
    pub plan_hash: String,
    pub policy_hash: String,
    pub workspace_hash: String,
    pub model_id: String,
    pub model_context: Option<super::model_transition::ModelContext>,
}

impl CheckpointSeed {
    /// Builds the checkpoint for a moment, given what has changed since the run
    /// began.
    ///
    /// The caller supplies the state, the sequence, the notes and the unsettled
    /// effects; this supplies the fixed half and seals the result.
    pub fn checkpoint(
        &self,
        run_id: &str,
        state: RunState,
        last_event_seq: i64,
        notes: crate::agent_runtime::memory::RunMemory,
        ledger: Option<crate::agent_runtime::tasks::ContextLedgerRecord>,
        unknown_effects: Vec<String>,
    ) -> super::events::RunCheckpoint {
        super::events::RunCheckpoint::new(
            run_id,
            &self.attempt_id,
            state,
            last_event_seq,
            notes,
            ledger,
            &self.plan_hash,
            &self.policy_hash,
            &self.workspace_hash,
            &self.model_id,
            unknown_effects,
        )
    }
}

/// Takes a checkpoint, and says whether it landed.
///
/// Thin on purpose: everything deciding *what* goes into a checkpoint is the
/// caller's, and everything deciding whether it may be written is the store's.
/// What this adds is the one thing neither should own — that a failure to write
/// a checkpoint is a fact the caller has to handle rather than a warning it may
/// log and move past.
///
/// Returns the error rather than logging it, so a caller doing something
/// recovery-critical can stop. A run that believes it checkpointed and did not
/// is a run that will be offered for resumption from a point that does not
/// exist.
pub fn checkpoint_now(
    events: &TaskEventLog,
    checkpoint: &super::events::RunCheckpoint,
) -> Result<bool, String> {
    events.save_checkpoint(checkpoint)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::{Role, User};

    fn session_of(id: &str, roles: Vec<Role>) -> Session {
        Session::open(User::new(id, id, roles))
    }

    #[test]
    fn the_same_person_and_material_hash_the_same_way_twice() {
        let one = session_of("priya", vec![Role::Employee, Role::Administrator]);
        let two = session_of("priya", vec![Role::Administrator, Role::Employee]);

        // Role order is an accident of how the directory was written. Two
        // identical people must not produce two different hashes, or every
        // resumption would refuse for a reason nobody could find.
        assert_eq!(
            policy_hash(&one, Some(Classification::Internal), "sovereign"),
            policy_hash(&two, Some(Classification::Internal), "sovereign"),
        );
    }

    #[test]
    fn changing_any_part_of_the_policy_changes_the_hash() {
        let base = session_of("priya", vec![Role::Employee]);
        let reference = policy_hash(&base, Some(Classification::Internal), "sovereign");

        // A role added.
        let widened = session_of("priya", vec![Role::Employee, Role::Administrator]);
        assert_ne!(
            policy_hash(&widened, Some(Classification::Internal), "sovereign"),
            reference
        );

        // A different person entirely.
        let other = session_of("ravi", vec![Role::Employee]);
        assert_ne!(
            policy_hash(&other, Some(Classification::Internal), "sovereign"),
            reference
        );

        // The material reclassified.
        assert_ne!(
            policy_hash(&base, Some(Classification::VendorNegotiation), "sovereign"),
            reference
        );

        // The machine put into a different mode.
        assert_ne!(
            policy_hash(&base, Some(Classification::Internal), "provisioning"),
            reference
        );
    }

    #[test]
    fn an_unclassified_run_is_not_hashed_as_though_it_were_internal() {
        // Otherwise classifying a run after the fact would leave the hash
        // unchanged, and the resumption would not notice.
        let session = session_of("priya", vec![Role::Employee]);
        assert_ne!(
            policy_hash(&session, None, "sovereign"),
            policy_hash(&session, Some(Classification::Internal), "sovereign"),
        );
    }

    #[test]
    fn the_same_prompt_plans_to_the_same_hash() {
        let prompt = "Draft an approval note for the seal replacement.";
        assert_eq!(plan_hash_of(prompt), plan_hash_of(prompt));
    }

    #[test]
    fn a_prompt_that_earns_a_different_plan_hashes_differently() {
        // The plan carries the budget and the permitted tools. A prompt that
        // earns the right to write a workbook is a different plan from one that
        // does not, and continuing the first under the second would either
        // widen or narrow what the run may do.
        let reading = plan_hash_of("Summarise the maintenance SOP.");
        let computing = plan_hash_of(
            "Calculate the remaining wall thickness and produce a spreadsheet of the working.",
        );
        assert_ne!(reading, computing);
    }

    #[test]
    fn a_workspace_that_is_gone_has_no_identity() {
        let dir = tempfile::tempdir().expect("temp dir");
        let missing = dir.path().join("never-created");
        assert_eq!(workspace_hash_of(&missing), None);
    }

    #[test]
    fn a_workspace_hashes_the_same_way_through_a_different_route_to_it() {
        // Canonicalised, so `runs/./run-1` and `runs/run-1` are the same
        // directory. Otherwise a path assembled differently by a later build
        // would refuse every resumption on this machine.
        let dir = tempfile::tempdir().expect("temp dir");
        let root = dir.path().join("run-1");
        std::fs::create_dir_all(&root).expect("created");

        let direct = workspace_hash_of(&root).expect("hashed");
        let indirect = workspace_hash_of(&root.join(".")).expect("hashed");
        assert_eq!(direct, indirect);
    }

    #[test]
    fn two_workspaces_do_not_share_an_identity() {
        let dir = tempfile::tempdir().expect("temp dir");
        let one = dir.path().join("run-1");
        let two = dir.path().join("run-2");
        std::fs::create_dir_all(&one).expect("created");
        std::fs::create_dir_all(&two).expect("created");

        assert_ne!(workspace_hash_of(&one), workspace_hash_of(&two));
    }

    #[test]
    fn a_file_where_a_workspace_should_be_has_no_identity() {
        // Not a directory is not a workspace, however much the path matches.
        let dir = tempfile::tempdir().expect("temp dir");
        let impostor = dir.path().join("run-1");
        std::fs::write(&impostor, b"not a directory").expect("written");

        assert_eq!(workspace_hash_of(&impostor), None);
    }

    #[test]
    fn every_attempt_at_one_task_gets_a_new_id_and_keeps_the_task_id() {
        // The property that makes a resumption a continuation: one task, many
        // attempts. A new task id would make the Tasks screen show two rows for
        // one piece of work.
        let first = Attempt::new("run-1", "picking it up after the restart", 12);
        let second = Attempt::new("run-1", "trying again", 18);

        assert_eq!(first.run_id, second.run_id);
        assert_ne!(first.attempt_id, second.attempt_id);
        assert_eq!(second.from_seq, 18);
    }

    #[test]
    fn an_operators_note_is_bounded_before_it_reaches_a_durable_record() {
        // Unbounded, this field is a way to put a document into a record that is
        // meant to hold none.
        let attempt = Attempt::new("run-1", &"x".repeat(10_000), 1);
        assert!(attempt.operator_intent.chars().count() <= MAX_INTENT_CHARS);
    }
}
