//! What each tool does to the world, and what may be done about it going wrong.
//!
//! ## Why the existing answer was not enough
//!
//! `events::idempotency::is_side_effecting` answers one question — does this
//! leave a trace outside the task — and answers it well. It is the right gate
//! for "must an intent be written before this runs".
//!
//! It is the wrong shape for two others that came up as soon as recovery could
//! actually continue a run:
//!
//! - **May this be retried?** A search that timed out should simply be asked
//!   again. A document write that timed out must not be, because the timeout
//!   says nothing about whether the file was written.
//! - **Can anyone tell whether it happened?** For a file the answer is yes, by
//!   opening the path. For code executed in the sandbox the answer is no: the
//!   effects are whatever the code did, and nothing here can enumerate them.
//!
//! Those are different questions with different answers per tool, and a `bool`
//! cannot carry them.
//!
//! ## Why a `const` table beside the enum
//!
//! Every tool must have an entry, and the compiler must be the thing that says
//! so. A `match` over [`ToolName`] with no catch-all arm means adding a tool
//! fails to build until somebody has decided what it does to the world — which
//! is exactly the moment to decide it, and exactly the decision that gets
//! skipped when the default is "assume it is safe".
//!
//! `tool-names.ts` records what a mismatched name table cost the last time this
//! project kept two of them. There is one here, derived from the enum, and the
//! tests assert it agrees with the idempotency ledger.
//!
//! ## Why the defaults lean the way they do
//!
//! An ambiguous case fails closed: not retryable, not reconcilable, needs a
//! person. A wrong "safe to retry" writes a second document; a wrong "not safe
//! to retry" makes somebody click a button. Those costs are not symmetric.

use crate::orchestrator::tools::ToolName;

/// What a tool does to the world.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolClass {
    /// Reads. Running it twice is indistinguishable from running it once.
    ReadOnly,
    /// Changes something that can be put back as it was.
    Reversible,
    /// Leaves a trace outside the task that can be found and inspected.
    SideEffecting,
    /// Leaves a trace that cannot be reliably found, undone, or even
    /// enumerated. The strongest claim, and the one that most restricts what
    /// recovery may do on its own.
    Irreversible,
}

impl ToolClass {
    pub const fn as_str(self) -> &'static str {
        match self {
            ToolClass::ReadOnly => "read_only",
            ToolClass::Reversible => "reversible",
            ToolClass::SideEffecting => "side_effecting",
            ToolClass::Irreversible => "irreversible",
        }
    }

    /// Whether an intent must be written before the call is made.
    ///
    /// Agrees with `idempotency::is_side_effecting` by construction, and is
    /// checked against it in the tests: two tables that disagree about which
    /// calls are dangerous would be worse than either alone.
    pub const fn needs_intent(self) -> bool {
        matches!(self, ToolClass::SideEffecting | ToolClass::Irreversible)
    }
}

/// How an outcome nobody saw can be settled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reconciliation {
    /// Nothing to settle: running it again costs nothing.
    NotNeeded,
    /// Re-open what it would have produced and see whether it is there and
    /// sound. `artifacts::report_for_run` already does exactly this.
    InspectArtifact,
    /// Nobody can say. A person has to look.
    ///
    /// Deliberately available: an honest "cannot tell" is the whole reason
    /// `ToolEffectUnknown` exists, and a reconciliation method that always
    /// claimed an answer would defeat it.
    AskAPerson,
}

/// What may be done when a call does not come back cleanly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RetryPolicy {
    pub max_retries: u8,
    /// Seconds before the first retry. Doubles each time.
    pub backoff_seconds: u8,
    /// Whether the call is naturally idempotent — the same arguments give the
    /// same result and leave the world in the same place.
    pub idempotent: bool,
    /// Whether a retry may happen without anyone being asked.
    ///
    /// Not the same as `idempotent`. A call can be idempotent and still unsafe
    /// to retry automatically when nobody can tell whether the first attempt
    /// landed.
    pub safe_to_retry: bool,
    pub reconciliation: Reconciliation,
    /// Whether a person is asked before the call happens at all.
    pub requires_approval: bool,
}

impl RetryPolicy {
    /// How long to wait before attempt `attempt` (1-based), in seconds.
    pub const fn backoff_for(&self, attempt: u8) -> u64 {
        if attempt == 0 {
            return 0;
        }
        // Doubling, capped so a long backoff cannot outlive the run's deadline.
        let shift = if attempt > 6 { 6 } else { attempt - 1 };
        let seconds = (self.backoff_seconds as u64) << shift;
        if seconds > 300 {
            300
        } else {
            seconds
        }
    }
}

/// A read that can simply be asked again.
const READ: RetryPolicy = RetryPolicy {
    max_retries: 2,
    backoff_seconds: 1,
    idempotent: true,
    safe_to_retry: true,
    reconciliation: Reconciliation::NotNeeded,
    requires_approval: false,
};

/// Something that wrote a file. The file can be looked for, so an unknown
/// outcome is answerable — but not by retrying blindly.
const WROTE_A_FILE: RetryPolicy = RetryPolicy {
    max_retries: 0,
    backoff_seconds: 0,
    idempotent: false,
    safe_to_retry: false,
    reconciliation: Reconciliation::InspectArtifact,
    requires_approval: true,
};

/// Something whose effects cannot be enumerated.
const CANNOT_BE_UNDONE: RetryPolicy = RetryPolicy {
    max_retries: 0,
    backoff_seconds: 0,
    idempotent: false,
    safe_to_retry: false,
    reconciliation: Reconciliation::AskAPerson,
    requires_approval: true,
};

/// What this tool does to the world.
///
/// No catch-all arm, on purpose: a new tool does not compile until somebody has
/// answered this.
pub const fn class_of(tool: ToolName) -> ToolClass {
    match tool {
        ToolName::SearchDocuments
        | ToolName::LoadMoreEvidence
        | ToolName::MediaExtractFindings
        | ToolName::MemoryRecallAuthorized
        | ToolName::ReadScopedFile
        | ToolName::ValidateArtifact
        | ToolName::CapabilitySearch
        | ToolName::SovereigntyGetEvidence
        | ToolName::KnowledgeMultimodalRetrieve
        // Reads back a file the asker attached to this conversation, from a
        // store this machine wrote. It changes nothing and reaches nothing.
        | ToolName::ReadAttachedPages
        | ToolName::SearchAttachedDocuments
        | ToolName::BuildDocumentGraph
        | ToolName::NotebookList
        | ToolName::NotebookSources => ToolClass::ReadOnly,
        // Reversible: a notebook created by mistake can be deleted, a
        // rename can be renamed back, and a source taken out can be put
        // back - the document itself is never touched by any of them.
        ToolName::NotebookCreate
        | ToolName::NotebookRename
        | ToolName::NotebookAddSource
        | ToolName::NotebookRemoveSource => ToolClass::Reversible,
        // Irreversible: the notebook, its membership rows and the whole
        // graph built over it go together, and nothing here can put the
        // graph back - rebuilding it means running the extraction passes
        // again over every document.
        // Writes a file into the run's own workspace, which is a trace
        // outside the task even though nothing outside it changes.
        ToolName::CreateChart
        | ToolName::CreateDiagram
        | ToolName::CreatePdf
        | ToolName::CreateTable => ToolClass::SideEffecting,
        ToolName::NotebookDelete => ToolClass::Irreversible,

        // Deterministic arithmetic recorded in the run's own calculation table.
        // It changes state this process owns and can discard, and it reaches
        // nothing outside the task.
        ToolName::RunCalculation => ToolClass::Reversible,

        // Writes into the project's memory, under an approval granted for that
        // exact fact. Reversible because `memory_forgotten` can take it back
        // out, and the memory store is ARJUN's own.
        ToolName::MemoryPromoteApproved => ToolClass::Reversible,

        // A sub-agent that may only read. Its own tool calls go through this
        // same gateway, so it cannot exceed what a read-only tool may do; what
        // makes it more than `ReadOnly` here is that it consumes budget and
        // leaves a subagent record.
        ToolName::AgentDelegateReadonly => ToolClass::Reversible,

        // Files on disk outside the process. Findable, inspectable, and not to
        // be written twice.
        ToolName::WriteScopedFile
        | ToolName::CreateDocx
        | ToolName::CreateXlsx
        | ToolName::CreatePptx => ToolClass::SideEffecting,

        // Whatever the code did. The sandbox bounds it, and bounding is not the
        // same as being able to list it afterwards.
        ToolName::ExecuteCode => ToolClass::Irreversible,
    }
}

/// What may be done when a call to this tool does not come back cleanly.
///
/// **No production caller.** Nothing retries a failed tool call anywhere in
/// this product: `execute` returns the failure, the loop hands it to the model
/// as an error result, and that is the whole of it. This table is the policy a
/// retry *would* follow, and `class_of` above — which it derives from — is
/// live and is what `only_what_cannot_be_undone_here_asks_a_person` reads.
///
/// Said plainly because "there is a retry policy" and "failed tool calls are
/// retried" are different statements, and only the first is true.
pub const fn retry_policy_of(tool: ToolName) -> RetryPolicy {
    match class_of(tool) {
        ToolClass::ReadOnly => READ,
        // Reversible calls are retryable for the same reason reads are — the
        // state they touch is ARJUN's own and can be put back — but they are
        // not free, so the ceiling is lower.
        ToolClass::Reversible => RetryPolicy {
            max_retries: 1,
            ..READ
        },
        ToolClass::SideEffecting => WROTE_A_FILE,
        ToolClass::Irreversible => CANNOT_BE_UNDONE,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent_runtime::events::idempotency::is_side_effecting;

    /// The invariant that stops this becoming a second, disagreeing table.
    ///
    /// `is_side_effecting` is what actually gates writing an intent before a
    /// call. If these two ever disagree, one of them is wrong about which calls
    /// are dangerous, and the tests would not otherwise say which.
    #[test]
    fn the_two_tables_agree_about_which_calls_are_dangerous() {
        for tool in ToolName::ALL.iter().copied() {
            assert_eq!(
                class_of(tool).needs_intent(),
                is_side_effecting(tool),
                "{} is classified {} here and {} by the idempotency ledger",
                tool.as_str(),
                class_of(tool).as_str(),
                is_side_effecting(tool)
            );
        }
    }

    #[test]
    fn a_tool_that_self_retries_never_also_needs_a_person() {
        for tool in ToolName::ALL.iter().copied() {
            let policy = retry_policy_of(tool);
            // Otherwise the retry silently re-uses a decision that was made for
            // one attempt.
            if policy.safe_to_retry {
                assert!(
                    !policy.requires_approval,
                    "{} may be retried automatically but also needs approval",
                    tool.as_str()
                );
            }
        }
    }

    /// The rule the whole thing exists for: nothing that touches the world
    /// outside the task is retried without a person.
    #[test]
    fn nothing_side_effecting_is_retried_on_its_own() {
        for tool in ToolName::ALL.iter().copied() {
            if class_of(tool).needs_intent() {
                let policy = retry_policy_of(tool);
                assert!(
                    !policy.safe_to_retry,
                    "{} must not self-retry",
                    tool.as_str()
                );
                assert_eq!(policy.max_retries, 0, "{}", tool.as_str());
            }
        }
    }

    /// A file can be looked for; sandboxed code cannot be reasoned about.
    #[test]
    fn only_effects_that_can_be_found_claim_to_be_reconcilable() {
        assert_eq!(
            retry_policy_of(ToolName::CreateDocx).reconciliation,
            Reconciliation::InspectArtifact
        );
        assert_eq!(
            retry_policy_of(ToolName::ExecuteCode).reconciliation,
            Reconciliation::AskAPerson
        );
        assert_eq!(
            retry_policy_of(ToolName::SearchDocuments).reconciliation,
            Reconciliation::NotNeeded
        );
    }

    #[test]
    fn backoff_doubles_and_is_capped() {
        let policy = READ;
        assert_eq!(policy.backoff_for(0), 0);
        assert_eq!(policy.backoff_for(1), 1);
        assert_eq!(policy.backoff_for(2), 2);
        assert_eq!(policy.backoff_for(3), 4);
        assert!(policy.backoff_for(50) <= 300, "backoff must stay bounded");
    }

    #[test]
    fn reads_are_retryable_and_need_nobody() {
        let policy = retry_policy_of(ToolName::SearchDocuments);
        assert!(policy.safe_to_retry);
        assert!(policy.idempotent);
        assert!(!policy.requires_approval);
    }
}
