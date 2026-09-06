//! What the assistant is allowed to ask for.
//!
//! A model in ARJUN cannot open a file, run a command, or reach the network. It
//! can only emit a [`ToolCall`] — a name and some arguments — and something else
//! decides whether that happens. This module is the catalogue of what may be
//! asked for; [`super::gateway`] is what decides.
//!
//! ## Every tool declares its own cost
//!
//! ARJUN design rule 13 asks that each tool carry a schema, permitted inputs, permitted
//! directories, size and time limits, and whether a human has to approve it.
//! Those live on the [`ToolSpec`] rather than in the code that runs the tool,
//! so the limits can be read without reading an implementation — and so a new
//! tool cannot be added without stating them.
//!
//! ## Approval is a property of the tool, not of the moment
//!
//! Whether something needs a human is decided here, once, by what the tool can
//! do — not by how risky a particular call looks. A judgement made per-call
//! would depend on the arguments, and the arguments come from the model.

use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::identity::Permission;

/// How a tool behaves towards the network.
///
/// Declared per tool rather than inferred, because the question Work mode asks
/// is not "did this call reach the internet?" — by then it has — but "may this
/// tool be *offered* at all?". A tool absent from the catalogue cannot be
/// called by a model that has never heard of it, which is a stronger guarantee
/// than one refused well.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum NetworkUse {
    /// Touches nothing outside this process.
    None,
    /// Talks to a local sidecar or inference server over loopback. Permitted in
    /// Work mode: loopback is not egress, and the broker holds that line.
    Loopback,
    /// Reaches a host outside this machine. Never offered in Work mode.
    Outbound,
}

impl NetworkUse {
    /// Whether a tool with this behaviour may appear in the catalogue in `mode`.
    pub const fn permitted_in(self, mode: crate::sovereignty::OperatingMode) -> bool {
        match self {
            NetworkUse::None | NetworkUse::Loopback => true,
            NetworkUse::Outbound => mode.permits_network(),
        }
    }

    pub const fn describe(self) -> &'static str {
        match self {
            NetworkUse::None => "no network access",
            NetworkUse::Loopback => "local sidecars only, no outbound network",
            NetworkUse::Outbound => "reaches hosts outside this machine",
        }
    }
}

/// Who has to say yes, and what that costs the person.
///
/// Separate from the boolean the gateway branches on, because a model choosing
/// between two tools needs to know which one will interrupt somebody — and
/// `needs_approval: true` does not distinguish an interruption from a refusal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ApprovalClass {
    /// Runs immediately. Reading and computing: undone by ignoring the result.
    Automatic,
    /// A person is asked before it happens, and the run waits.
    PersonBeforeEffect,
    /// A person approved a specific value in advance; the call is checked
    /// against that approval rather than prompting afresh.
    PreApprovedValue,
}

impl ApprovalClass {
    pub const fn describe(self) -> &'static str {
        match self {
            ApprovalClass::Automatic => "runs without asking anyone",
            ApprovalClass::PersonBeforeEffect => {
                "a person must approve it before it happens"
            }
            ApprovalClass::PreApprovedValue => {
                "requires an approval granted earlier for this exact value"
            }
        }
    }
}

/// Every tool ARJUN exposes. Nothing outside this list can be requested.
///
/// ## Why the names carry a namespace
///
/// A flat list of verbs — `search_documents`, `validate_artifact`,
/// `execute_code` — reads to a 7B model as unrelated options, and the
/// characteristic failure is reaching for a near neighbour: validating a file it
/// never produced, writing a deliverable with the plain-text tool. A namespace
/// puts the family in the name, so `artifact.verify_docx` is visibly the partner
/// of `artifact.create_approval_note` and visibly not a way to read a file. The
/// gain is disambiguation, not tidiness.
///
/// ## Why the old names still resolve
///
/// Task records, audit lines and event payloads written before this change hold
/// the old spellings, and those records are the evidence a reviewer reads months
/// later. [`ToolName::from_str`] accepts both; [`ToolName::as_str`] emits only
/// the new name, so nothing further is written in the old spelling. The
/// migration is therefore read-side only — the one kind that cannot corrupt a
/// record by failing half way through.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolName {
    /// Search the organisation's own documents.
    SearchDocuments,
    /// Read a named page range of a document already retrieved from.
    ///
    /// The counterpart to search, and the reason a run does not need whole
    /// documents in its context: a passage that stops mid-clause is followed by
    /// a request for the two pages around it, not for the file.
    LoadMoreEvidence,
    /// Read findings from a scanned or image-bearing page range.
    MediaExtractFindings,
    /// Read what this machine remembers for a scope the signed-in person may see.
    MemoryRecallAuthorized,
    /// Copy one of this run's facts into the project's memory, under an approval
    /// a person granted for that exact fact.
    MemoryPromoteApproved,
    /// Read a file inside the task workspace.
    ReadScopedFile,
    /// Write a file inside the task workspace.
    WriteScopedFile,
    /// Arithmetic with units, done deterministically rather than by a model.
    RunCalculation,
    /// Produce a Word document from a template.
    CreateDocx,
    /// Produce a spreadsheet from a template.
    CreateXlsx,
    /// Produce a briefing deck from a template.
    ///
    /// PS 26117 names the deliverables as *"approval notes, PPT/Word/Excel
    /// files, working code, calculations with steps shown"*. PPT is on that
    /// list, and `artifacts::pptx` has been able to write one since it was
    /// added — it simply had no tool through which a run could ask.
    CreatePptx,
    /// Run code in the sandbox.
    ExecuteCode,
    /// Re-open a produced file and check it is sound.
    ValidateArtifact,
    /// List the skills this run could load, as metadata only.
    CapabilitySearch,
    /// Hand a bounded read-only sub-task to a worker agent.
    AgentDelegateReadonly,
    /// Read this machine's own record of what it did and did not send.
    SovereigntyGetEvidence,
    /// Search both the prose index and the multimodal index (image regions
    /// and table rows), returning passages alongside their visual evidence.
    KnowledgeMultimodalRetrieve,
    /// Read a page range of a document attached to *this conversation*.
    ///
    /// The counterpart to the OCR budget. A large attachment enters the prompt
    /// in part or not at all, and before this the rest was simply gone — the
    /// text existed only inside the composed prompt string. Every page is now
    /// stored as it is read, and this is how a run reaches the pages the budget
    /// left out, including the ones it never saw and the ones an earlier turn
    /// in the same conversation attached.
    ReadAttachedPages,
}

impl ToolName {
    pub const ALL: &'static [ToolName] = &[
        ToolName::SearchDocuments,
        ToolName::LoadMoreEvidence,
        ToolName::MediaExtractFindings,
        ToolName::MemoryRecallAuthorized,
        ToolName::MemoryPromoteApproved,
        ToolName::ReadScopedFile,
        ToolName::WriteScopedFile,
        ToolName::RunCalculation,
        ToolName::CreateDocx,
        ToolName::CreateXlsx,
        ToolName::CreatePptx,
        ToolName::ExecuteCode,
        ToolName::ValidateArtifact,
        ToolName::CapabilitySearch,
        ToolName::AgentDelegateReadonly,
        ToolName::SovereigntyGetEvidence,
        ToolName::KnowledgeMultimodalRetrieve,
        ToolName::ReadAttachedPages,
    ];

    /// The wire name a model emits, and the only spelling ever written.
    pub const fn as_str(self) -> &'static str {
        match self {
            ToolName::SearchDocuments => "knowledge.search_authorized",
            ToolName::LoadMoreEvidence => "knowledge.load_evidence_region",
            ToolName::MediaExtractFindings => "media.extract_findings",
            ToolName::MemoryRecallAuthorized => "memory.recall_authorized",
            ToolName::MemoryPromoteApproved => "memory.promote_approved",
            ToolName::ReadScopedFile => "workspace.read_text",
            ToolName::WriteScopedFile => "workspace.write_text",
            ToolName::RunCalculation => "calculation.evaluate_with_units",
            ToolName::CreateDocx => "artifact.create_approval_note",
            ToolName::CreateXlsx => "artifact.create_calculation_workbook",
            ToolName::CreatePptx => "artifact.create_briefing_deck",
            ToolName::ExecuteCode => "sandbox.run_code",
            ToolName::ValidateArtifact => "artifact.verify_docx",
            ToolName::CapabilitySearch => "capability.search",
            ToolName::AgentDelegateReadonly => "agent.delegate_readonly",
            ToolName::SovereigntyGetEvidence => "sovereignty.get_evidence",
            ToolName::KnowledgeMultimodalRetrieve => "knowledge.multimodal_retrieve",
            ToolName::ReadAttachedPages => "document.read_pages",
        }
    }

    /// The pre-namespace spelling, for the tools that had one.
    ///
    /// Read-side only. Records and audit lines written before the rename hold
    /// these, and a reader that could not resolve them would show a months-old
    /// approval as having authorised a tool that does not exist.
    pub const fn legacy_str(self) -> Option<&'static str> {
        match self {
            ToolName::SearchDocuments => Some("search_documents"),
            ToolName::LoadMoreEvidence => Some("load_more_evidence"),
            ToolName::MemoryRecallAuthorized => Some("memory_recall_authorized"),
            ToolName::MemoryPromoteApproved => Some("memory_promote_approved"),
            ToolName::ReadScopedFile => Some("read_scoped_file"),
            ToolName::WriteScopedFile => Some("write_scoped_file"),
            ToolName::RunCalculation => Some("run_calculation"),
            ToolName::CreateDocx => Some("create_docx"),
            ToolName::CreateXlsx => Some("create_xlsx"),
            ToolName::CreatePptx => Some("create_pptx"),
            ToolName::ExecuteCode => Some("execute_code"),
            ToolName::ValidateArtifact => Some("validate_artifact"),
            // Introduced with the namespace. No older spelling exists, and
            // inventing one would create a name no record can contain.
            ToolName::MediaExtractFindings
            | ToolName::CapabilitySearch
            | ToolName::AgentDelegateReadonly
            | ToolName::SovereigntyGetEvidence
            | ToolName::KnowledgeMultimodalRetrieve
            | ToolName::ReadAttachedPages => None,
        }
    }

    /// Resolves a wire name, accepting the current spelling or the legacy one.
    pub fn from_str(raw: &str) -> Option<Self> {
        Self::ALL
            .iter()
            .copied()
            .find(|t| t.as_str() == raw || t.legacy_str() == Some(raw))
    }

    /// Whether this call only reads.
    ///
    /// The property that decides parallelism. A read cannot change what another
    /// read returns, so several may run at once and the operator waits for the
    /// slowest rather than the sum. Anything that writes, produces a file, runs
    /// code or asks a person is sequential: two writes to one path have an
    /// order, and it should not be whichever finished first.
    /// Whether this tool reads from the organisation's own record.
    ///
    /// A plan that permits one of these is a plan for a task the record has to
    /// answer, which is what the verifier needs to know: an answer to such a
    /// task with nothing retrieved behind it came from the model rather than
    /// from a document. See `artifacts::verifier::Grounding`.
    pub const fn is_retrieval(self) -> bool {
        matches!(
            self,
            ToolName::SearchDocuments
                | ToolName::LoadMoreEvidence
                | ToolName::MediaExtractFindings
                | ToolName::KnowledgeMultimodalRetrieve
                | ToolName::ReadAttachedPages
        )
    }

    pub const fn is_read_only(self) -> bool {
        match self {
            ToolName::SearchDocuments
            | ToolName::LoadMoreEvidence
            | ToolName::MediaExtractFindings
            | ToolName::MemoryRecallAuthorized
            | ToolName::ReadScopedFile
            | ToolName::RunCalculation
            | ToolName::ValidateArtifact
            | ToolName::CapabilitySearch
            | ToolName::AgentDelegateReadonly
            | ToolName::SovereigntyGetEvidence
            | ToolName::KnowledgeMultimodalRetrieve
            | ToolName::ReadAttachedPages => true,
            ToolName::MemoryPromoteApproved
            | ToolName::WriteScopedFile
            | ToolName::CreateDocx
            | ToolName::CreateXlsx
            | ToolName::CreatePptx
            | ToolName::ExecuteCode => false,
        }
    }

    /// How the action reads in an approval prompt or an audit line.
    pub const fn describe(self) -> &'static str {
        match self {
            ToolName::SearchDocuments => "search the knowledge base",
            ToolName::LoadMoreEvidence => "read a specific page range of a document",
            ToolName::MediaExtractFindings => "read findings from a scanned page range",
            ToolName::MemoryRecallAuthorized => "read this machine's memory for one scope",
            ToolName::MemoryPromoteApproved => "record an approved fact in the project's memory",
            ToolName::ReadScopedFile => "read a file from the task workspace",
            ToolName::WriteScopedFile => "write a file into the task workspace",
            ToolName::RunCalculation => "run a calculation",
            ToolName::CreateDocx => "produce a Word document",
            ToolName::CreateXlsx => "produce a spreadsheet",
            ToolName::CreatePptx => "produce a briefing deck",
            ToolName::ExecuteCode => "run code in the sandbox",
            ToolName::ValidateArtifact => "check a produced file",
            ToolName::CapabilitySearch => "list the skills this task could load",
            ToolName::AgentDelegateReadonly => "hand a read-only sub-task to a worker",
            ToolName::SovereigntyGetEvidence => "read this machine's own network record",
            ToolName::KnowledgeMultimodalRetrieve => "search text, image regions, and tables together",
            ToolName::ReadAttachedPages => "read pages of a document attached to this conversation",
        }
    }
}

/// One required argument, and what it has to be.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArgumentSpec {
    pub name: &'static str,
    pub kind: ArgumentKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArgumentKind {
    Text,
    /// A path. Always checked against the task's permitted directories.
    Path,
    Integer,
    /// A nested object, validated by the tool itself rather than here.
    Object,
}

/// Everything the gateway needs to know about a tool before allowing it.
#[derive(Debug, Clone)]
pub struct ToolSpec {
    pub name: ToolName,
    /// What the user must hold for this to be permitted at all.
    pub permission: Permission,
    pub arguments: &'static [ArgumentSpec],
    /// Whether a person has to say yes before this runs.
    ///
    /// True for anything that leaves a trace outside the task — a file, a
    /// document, an execution — and false for reading and computing, which can
    /// be undone by ignoring the result.
    pub needs_approval: bool,
    /// Largest input or output this tool may handle.
    pub max_bytes: Option<u64>,
    /// Wall-clock ceiling. Every tool has one: a tool that can hang has a
    /// budget of infinity, and a plan containing it can never be bounded.
    pub timeout: Duration,
    /// Whether the call touches a path that must be inside the workspace.
    pub scoped_to_workspace: bool,
    /// What the tool does to the network. Decides whether it is offered at all.
    pub network: NetworkUse,
    /// Who has to say yes. Richer than `needs_approval`, which it agrees with.
    pub approval_class: ApprovalClass,
    /// Largest response body handed back to the model, in bytes.
    ///
    /// Separate from `max_bytes`, which bounds the *input*. A tool can be given
    /// a small argument and answer with a document: search over a large corpus
    /// is exactly that shape. Truncation happens at this ceiling, deterministically
    /// and with a line saying what was dropped and how to ask for the rest —
    /// silent truncation is how a model comes to cite the half of a table it was
    /// shown as though it were the whole.
    pub max_response_bytes: usize,
}

/// The default ceiling on a tool response.
///
/// About four thousand tokens on English prose: large enough for six passages
/// with their citations, small enough that one call cannot take half of an 8k
/// window. Tools that answer with more than this say so and offer a page.
pub const DEFAULT_MAX_RESPONSE_BYTES: usize = 16 * 1024;

/// What every tool gets unless its own arm says otherwise.
///
/// The safe end of each axis: no network, no path, the ordinary response
/// ceiling, and — for anything not read-only — an approval. A tool added later
/// that forgets to state its network behaviour is therefore offline rather than
/// outbound, and one that forgets its approval class interrupts a person rather
/// than acting unasked. Defaults are where a future omission lands, so they are
/// chosen for what a mistake should cost.
fn defaults(name: ToolName) -> ToolSpec {
    let read_only = name.is_read_only();
    ToolSpec {
        name,
        permission: Permission::UseModel,
        arguments: &[],
        needs_approval: !read_only,
        max_bytes: None,
        timeout: Duration::from_secs(30),
        scoped_to_workspace: false,
        network: NetworkUse::None,
        approval_class: if read_only {
            ApprovalClass::Automatic
        } else {
            ApprovalClass::PersonBeforeEffect
        },
        max_response_bytes: DEFAULT_MAX_RESPONSE_BYTES,
    }
}

/// The catalogue.
///
/// Deliberately a function over a static table rather than data loaded at
/// runtime: a tool the code does not know about cannot be enabled by editing a
/// file, which is the correct trade for the one surface a model can reach.
pub fn spec_for(name: ToolName) -> ToolSpec {
    use ArgumentKind::*;
    use Permission::*;

    match name {
        ToolName::SearchDocuments => ToolSpec {
            permission: SearchKnowledge,
            arguments: &[ArgumentSpec { name: "query", kind: Text }],
            // Reads the index, which is on this machine.
            network: NetworkUse::None,
            ..defaults(name)
        },
        ToolName::LoadMoreEvidence => ToolSpec {
            // The same permission as search, because it reads the same shelf
            // through the same clearance checks. A weaker permission here would
            // be a way to read by page number what may not be read by searching.
            permission: SearchKnowledge,
            arguments: &[
                ArgumentSpec { name: "documentSha256", kind: Text },
                ArgumentSpec { name: "fromPage", kind: Integer },
                ArgumentSpec { name: "toPage", kind: Integer },
            ],
            ..defaults(name)
        },
        ToolName::MediaExtractFindings => ToolSpec {
            // Reads the same shelf as search, by the same clearance. A scanned
            // page is not a lesser document for being a picture.
            permission: SearchKnowledge,
            arguments: &[
                ArgumentSpec { name: "documentSha256", kind: Text },
                ArgumentSpec { name: "fromPage", kind: Integer },
                ArgumentSpec { name: "toPage", kind: Integer },
            ],
            // The OCR and vision engines are Python sidecars this machine talks
            // to over loopback. Loopback is not egress, so this stays available
            // in Work mode — which is the mode a scanned inspection report is
            // actually read in.
            network: NetworkUse::Loopback,
            // A vision pass over a page is slow next to a text read.
            timeout: Duration::from_secs(90),
            ..defaults(name)
        },
        ToolName::KnowledgeMultimodalRetrieve => ToolSpec {
            // Reads the same shelf as `SearchDocuments`, applying the same
            // clearance. The multimodal index carries the same classifications
            // the prose index does, and a row the asker cannot see is not
            // returned.
            permission: SearchKnowledge,
            arguments: &[
                ArgumentSpec { name: "query", kind: Text },
                ArgumentSpec { name: "documentType", kind: Text },
                ArgumentSpec { name: "documentSha256", kind: Text },
                ArgumentSpec { name: "page", kind: Integer },
                ArgumentSpec { name: "maxResults", kind: Integer },
            ],
            // A search over a 200-page P&ID set is not as cheap as a text
            // search. Allow more time before the budget trips.
            timeout: Duration::from_secs(45),
            // A larger response budget: this tool returns prose passages
            // *and* image regions, each with its own citation. The model is
            // expected to use it once and reason about a result, not to
            // page through twenty of them.
            max_response_bytes: 32 * 1024,
            ..defaults(name)
        },
        ToolName::ReadAttachedPages => ToolSpec {
            // `UseModel`, not `SearchKnowledge`, and the difference is the whole
            // reason this is a separate tool from `load_evidence_region`.
            //
            // `SearchKnowledge` is clearance to read the *organisation's* shelf,
            // and the classifications on it are what that permission exists to
            // enforce. This reads nothing from that shelf. It reads back a file
            // the signed-in person attached to their own conversation minutes
            // ago — a document they already hold, part of whose text they were
            // already shown in this same turn. Requiring shelf clearance to
            // re-read your own attachment would refuse a person their own file;
            // granting shelf clearance through this door would be a way to read
            // the shelf by page number.
            //
            // The isolation that matters here is owner and conversation, and it
            // is enforced inside the operation rather than by a permission: see
            // `agent_runtime::documents`, where a document is invisible to
            // anyone who did not attach it and to any conversation it was not
            // attached to.
            permission: UseModel,
            arguments: &[
                ArgumentSpec { name: "documentSha256", kind: Text },
                ArgumentSpec { name: "fromPage", kind: Integer },
                ArgumentSpec { name: "toPage", kind: Integer },
            ],
            // Reads a file this machine already wrote. No model, no sidecar, no
            // socket.
            network: NetworkUse::None,
            timeout: Duration::from_secs(10),
            // The store's own `MAX_READ_BYTES` is the real bound and is applied
            // first; this is the outer guard, sized above it so a read the store
            // considered complete is never truncated a second time by the
            // gateway without anything saying so.
            max_response_bytes: 32 * 1024,
            ..defaults(name)
        },
        ToolName::MemoryRecallAuthorized => ToolSpec {
            // Reading memory is reading whatever the person is already cleared
            // for; the store applies the same clearance the index does.
            permission: UseModel,
            arguments: &[ArgumentSpec { name: "scope", kind: Text }],
            timeout: Duration::from_secs(10),
            ..defaults(name)
        },
        ToolName::MemoryPromoteApproved => ToolSpec {
            // Promotion writes something later runs will read. That is the same
            // kind of act as producing a document, and it is entitled the same
            // way.
            permission: GenerateArtifact,
            arguments: &[
                ArgumentSpec { name: "key", kind: Text },
                ArgumentSpec { name: "approvalId", kind: Text },
            ],
            // The approval is checked inside the operation, against the exact
            // value being promoted — a gateway prompt could only ask about the
            // call, and the call is not what needs approving.
            needs_approval: false,
            approval_class: ApprovalClass::PreApprovedValue,
            timeout: Duration::from_secs(10),
            ..defaults(name)
        },
        ToolName::ReadScopedFile => ToolSpec {
            permission: UseModel,
            arguments: &[ArgumentSpec { name: "path", kind: Path }],
            // Large enough for a long report, small enough that one file cannot
            // fill the context window and push the task's own instructions out.
            max_bytes: Some(8 * 1024 * 1024),
            timeout: Duration::from_secs(15),
            scoped_to_workspace: true,
            ..defaults(name)
        },
        ToolName::WriteScopedFile => ToolSpec {
            permission: GenerateArtifact,
            arguments: &[
                ArgumentSpec { name: "path", kind: Path },
                ArgumentSpec { name: "content", kind: Text },
            ],
            max_bytes: Some(32 * 1024 * 1024),
            scoped_to_workspace: true,
            ..defaults(name)
        },
        ToolName::RunCalculation => ToolSpec {
            permission: UseModel,
            arguments: &[ArgumentSpec { name: "expression", kind: Text }],
            timeout: Duration::from_secs(5),
            ..defaults(name)
        },
        ToolName::CreateDocx | ToolName::CreateXlsx | ToolName::CreatePptx => ToolSpec {
            permission: GenerateArtifact,
            arguments: &[
                ArgumentSpec { name: "path", kind: Path },
                ArgumentSpec { name: "template", kind: Text },
                ArgumentSpec { name: "content", kind: Object },
            ],
            max_bytes: Some(64 * 1024 * 1024),
            timeout: Duration::from_secs(120),
            scoped_to_workspace: true,
            ..defaults(name)
        },
        ToolName::ExecuteCode => ToolSpec {
            permission: ExecuteCode,
            arguments: &[
                ArgumentSpec { name: "language", kind: Text },
                ArgumentSpec { name: "source", kind: Text },
            ],
            max_bytes: Some(1024 * 1024),
            // Short on purpose. A calculation that needs longer than this is
            // not a calculation, and an agent loop that can wait five minutes
            // per step will exhaust a person's patience before its own budget.
            timeout: Duration::from_secs(60),
            ..defaults(name)
        },
        ToolName::ValidateArtifact => ToolSpec {
            permission: GenerateArtifact,
            arguments: &[ArgumentSpec { name: "path", kind: Path }],
            max_bytes: Some(64 * 1024 * 1024),
            timeout: Duration::from_secs(60),
            scoped_to_workspace: true,
            ..defaults(name)
        },
        ToolName::CapabilitySearch => ToolSpec {
            // Reading a description costs nothing and reveals nothing beyond
            // what the person may already see. Entitling it any harder would
            // teach an operator that the gate is noise.
            permission: UseModel,
            arguments: &[ArgumentSpec { name: "query", kind: Text }],
            timeout: Duration::from_secs(5),
            // Cards only, never a skill body. Small by construction, and capped
            // anyway so a machine with many skills installed cannot fill a
            // window with descriptions of skills the run will not load.
            max_response_bytes: 8 * 1024,
            ..defaults(name)
        },
        ToolName::AgentDelegateReadonly => ToolSpec {
            // A child may never exceed its parent, and the parent needed this
            // to search at all. Entitlement is re-derived for the child from the
            // inherited policy; this is the floor, not the whole check.
            permission: SearchKnowledge,
            arguments: &[
                ArgumentSpec { name: "profile", kind: Text },
                ArgumentSpec { name: "task", kind: Text },
            ],
            // Read-only by construction: the child's inherited policy permits no
            // writing tool, so no approval can be needed for what it may do.
            // That is what makes delegation cheap enough to be worth having.
            needs_approval: false,
            // A child runs a whole loop of its own.
            timeout: Duration::from_secs(120),
            ..defaults(name)
        },
        ToolName::SovereigntyGetEvidence => ToolSpec {
            // Reading this machine's own record of what it refused to send. The
            // point of it is to be readable by whoever is being asked to trust
            // the claim, so it is not gated behind the audit entitlement.
            permission: UseModel,
            arguments: &[],
            timeout: Duration::from_secs(5),
            ..defaults(name)
        },
    }
}

/// Room kept for the notice that says a result was cut.
///
/// Reserved rather than added afterwards, so the returned string is at or below
/// the tool's ceiling including its own notice. Appending the notice to an
/// already-full result would push it over the limit the ceiling exists to
/// enforce — a bug that only shows up on the largest results, which are exactly
/// the ones that matter.
const TRUNCATION_NOTICE_BUDGET: usize = 512;

/// Cuts a tool result to the ceiling its spec declares.
///
/// ## Why truncation is stated rather than silent
///
/// A model handed the first half of a table has no way to know it is the first
/// half. It answers from what it was given, confidently, and the answer is
/// wrong in a way nobody can see by reading it. So the cut always carries a
/// sentence saying it happened, how much went, and what to ask for instead.
///
/// ## Why it is deterministic
///
/// The same result cut twice gives the same string, byte for byte. That is what
/// makes a run reproducible from its record: a cut that depended on timing, a
/// hash seed or the terminal width would make two replays of one run disagree
/// about what the model actually read, and the record would no longer be
/// evidence of anything.
///
/// The boundary is found by walking back to the nearest character start, never
/// by slicing at a byte offset — the two differ on any non-ASCII text, and
/// slicing mid-character panics rather than producing a wrong answer.
pub fn truncate_response(tool: ToolName, text: String) -> String {
    let limit = spec_for(tool).max_response_bytes;
    if text.len() <= limit {
        return text;
    }

    let keep = limit.saturating_sub(TRUNCATION_NOTICE_BUDGET);
    // Walk back to a character boundary. At most three steps for UTF-8, so this
    // cannot loop far, and `is_char_boundary(0)` is always true so it ends.
    let mut cut = keep.min(text.len());
    while cut > 0 && !text.is_char_boundary(cut) {
        cut -= 1;
    }

    let dropped = text.len() - cut;
    let kept = &text[..cut];
    format!(
        "{kept}\n\n[This result was {} bytes, above the {} byte limit for {}. \
         The first {} bytes are above; {dropped} bytes were not included. This is the \
         beginning of the result, not the whole of it — do not treat what you can see as \
         complete. Ask for a narrower range or a more specific query to see the rest.]",
        text.len(),
        limit,
        tool.as_str(),
        cut
    )
}

/// Strips a failure message of anything that is not useful to a model.
///
/// ## What is removed, and why each one
///
/// - **Absolute paths** become their final component. A model does not act on
///   `C:\Users\priya\AppData\...\run-8f2\draft.md`; it acts on `draft.md`, which
///   is also the only part it may legitimately name. The prefix is the operator's
///   home directory and the run's internal id, neither of which belongs in a
///   sentence the model may repeat into a document.
/// - **Anything that looks like a stack frame.** A backtrace tells a model
///   nothing it can act on, costs a large slice of the context window, and names
///   internal symbols that end up quoted back in an answer.
///
/// What is deliberately *kept* is the sentence saying what to do next. An error
/// a model cannot recover from costs a step and teaches it nothing, so the
/// sanitising is about removing noise rather than removing detail.
pub fn sanitise_failure(reason: &str) -> String {
    let mut out = String::with_capacity(reason.len());

    for (index, line) in reason.lines().enumerate() {
        let trimmed = line.trim_start();
        // Rust and Python backtrace shapes. Dropped whole: a frame has no
        // recoverable information in it.
        if trimmed.starts_with("at ")
            || trimmed.starts_with("File \"")
            || trimmed.starts_with("Traceback")
            || trimmed.starts_with("stack backtrace")
            || trimmed
                .split_once(':')
                .is_some_and(|(head, _)| head.trim().chars().all(|c| c.is_ascii_digit()) && !head.is_empty())
        {
            continue;
        }
        if index > 0 {
            out.push('\n');
        }
        out.push_str(&shorten_paths(line));
    }

    out.trim_end().to_string()
}

/// Replaces absolute paths in a line with their final component.
///
/// Whitespace-delimited rather than regex-matched: the shapes that matter are a
/// Windows drive letter and a leading slash, both of which are decidable from
/// the first two characters of a token. A pattern clever enough to find a path
/// inside prose would also find one in a sentence that merely mentioned a
/// colon, and mangling a useful message is worse than leaving a path in it.
fn shorten_paths(line: &str) -> String {
    line.split(' ')
        .map(|token| {
            let bare = token.trim_matches(|c| c == '"' || c == '\'' || c == ',' || c == '.');
            let looks_absolute = bare.starts_with('/')
                || bare
                    .as_bytes()
                    .get(1)
                    .is_some_and(|&b| b == b':')
                    && bare.chars().next().is_some_and(|c| c.is_ascii_alphabetic());
            if !looks_absolute {
                return token.to_string();
            }
            match bare.rsplit(['/', '\\']).next() {
                Some(name) if !name.is_empty() => token.replace(bare, name),
                _ => token.to_string(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// A request from the model. Nothing more than a name and some arguments —
/// deliberately inert until the gateway has looked at it.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolCall {
    pub tool: String,
    pub arguments: serde_json::Value,
}

impl ToolCall {
    pub fn new(tool: impl Into<String>, arguments: serde_json::Value) -> Self {
        Self {
            tool: tool.into(),
            arguments,
        }
    }

    /// The argument as a string, if present and of the right shape.
    pub fn text(&self, key: &str) -> Option<&str> {
        self.arguments.get(key).and_then(|v| v.as_str())
    }

    /// The argument as a non-negative whole number.
    ///
    /// A string of digits is accepted as well as a JSON number: local models
    /// emit `"fromPage": "11"` often enough that refusing it would spend a turn
    /// on a formatting quarrel rather than on the work. Anything else — a
    /// negative, a fraction, a word — is absent, because a page number this
    /// could not read is not a page number it should guess at.
    pub fn integer(&self, key: &str) -> Option<u32> {
        let value = self.arguments.get(key)?;
        if let Some(number) = value.as_u64() {
            return u32::try_from(number).ok();
        }
        value.as_str()?.trim().parse::<u32>().ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_tool_round_trips_through_its_wire_name() {
        for tool in ToolName::ALL {
            assert_eq!(ToolName::from_str(tool.as_str()), Some(*tool));
        }
    }

    #[test]
    fn an_unknown_name_is_not_a_tool() {
        assert_eq!(ToolName::from_str("delete_everything"), None);
        assert_eq!(ToolName::from_str(""), None);
    }

    /// A tool without a time limit makes every plan containing it unbounded.
    #[test]
    fn every_tool_has_a_time_limit() {
        for tool in ToolName::ALL {
            let spec = spec_for(*tool);
            assert!(spec.timeout > Duration::ZERO, "{} has no timeout", tool.as_str());
            assert!(
                spec.timeout <= Duration::from_secs(120),
                "{} may hang for too long",
                tool.as_str()
            );
        }
    }

    /// Anything that leaves a trace outside the task needs a person.
    #[test]
    fn tools_that_write_or_execute_all_require_approval() {
        for tool in [
            ToolName::WriteScopedFile,
            ToolName::CreateDocx,
            ToolName::CreateXlsx,
            ToolName::CreatePptx,
            ToolName::ExecuteCode,
        ] {
            assert!(spec_for(tool).needs_approval, "{} should need approval", tool.as_str());
        }
    }

    /// Reading and computing can be undone by ignoring the result, so making a
    /// person confirm them would only train them to click through.
    #[test]
    fn reading_and_computing_do_not_interrupt_anyone() {
        for tool in [
            ToolName::SearchDocuments,
            ToolName::LoadMoreEvidence,
            ToolName::MemoryRecallAuthorized,
            ToolName::ReadScopedFile,
            ToolName::RunCalculation,
            ToolName::ValidateArtifact,
        ] {
            assert!(!spec_for(tool).needs_approval, "{} should not need approval", tool.as_str());
        }
    }

    #[test]
    fn every_tool_that_takes_a_path_is_scoped_to_the_workspace() {
        for tool in ToolName::ALL {
            let spec = spec_for(*tool);
            let takes_path = spec.arguments.iter().any(|a| a.kind == ArgumentKind::Path);
            assert_eq!(
                takes_path,
                spec.scoped_to_workspace,
                "{} disagrees about whether its path is scoped",
                tool.as_str()
            );
        }
    }

    /// A tool that could read or write without limit could exhaust the machine.
    #[test]
    fn every_tool_touching_a_file_has_a_size_ceiling() {
        for tool in ToolName::ALL {
            let spec = spec_for(*tool);
            if spec.scoped_to_workspace {
                assert!(spec.max_bytes.is_some(), "{} has no size limit", tool.as_str());
            }
        }
    }

    // ── Truncation ───────────────────────────────────────────────────────

    /// The same result cut twice is the same string, byte for byte.
    ///
    /// What makes a run reproducible from its record. A cut that varied with
    /// timing or a hash seed would make two replays disagree about what the
    /// model actually read, and the record would stop being evidence.
    #[test]
    fn truncation_gives_the_same_answer_every_time() {
        let long = "x".repeat(200_000);
        let once = truncate_response(ToolName::SearchDocuments, long.clone());
        let twice = truncate_response(ToolName::SearchDocuments, long);
        assert_eq!(once, twice);
    }

    /// The result, notice included, is inside the tool's ceiling.
    ///
    /// Appending the notice after cutting to the limit would push the answer
    /// over it — a bug that only appears on the largest results, which are the
    /// only ones the ceiling exists for.
    #[test]
    fn a_cut_result_including_its_notice_stays_under_the_limit() {
        for tool in ToolName::ALL {
            let limit = spec_for(*tool).max_response_bytes;
            let cut = truncate_response(*tool, "y".repeat(limit * 3));
            assert!(
                cut.len() <= limit,
                "{} returned {} bytes against a {limit} byte limit",
                tool.as_str(),
                cut.len()
            );
        }
    }

    /// A result inside the limit is returned untouched.
    #[test]
    fn a_short_result_is_not_touched() {
        let text = "6 passage(s) found.".to_string();
        assert_eq!(
            truncate_response(ToolName::SearchDocuments, text.clone()),
            text
        );
    }

    /// The cut says it happened, in words that stop a model treating the part
    /// it can see as the whole.
    #[test]
    fn a_cut_result_says_it_is_incomplete() {
        let cut = truncate_response(ToolName::SearchDocuments, "z".repeat(100_000));
        assert!(cut.contains("not the whole of it"));
        assert!(cut.contains("bytes were not included"));
    }

    /// Cutting multi-byte text lands on a character boundary.
    ///
    /// Slicing UTF-8 at an arbitrary byte panics. The material this reads is
    /// Devanagari and Latin both, so the case is routine rather than exotic.
    #[test]
    fn cutting_multibyte_text_does_not_panic_or_corrupt() {
        let cut = truncate_response(ToolName::SearchDocuments, "पंप की सील ".repeat(20_000));
        // Round-tripping proves every byte kept is part of a whole character.
        assert_eq!(
            String::from_utf8(cut.clone().into_bytes()).expect("still valid UTF-8"),
            cut
        );
    }

    // ── Failure messages ─────────────────────────────────────────────────

    /// A path in a failure becomes the name the model may actually use.
    #[test]
    fn a_failure_names_the_file_rather_than_the_operators_home_directory() {
        let sanitised =
            sanitise_failure("C:\\Users\\priya\\AppData\\Local\\arjun\\runs\\r-8f2\\draft.md does not exist.");
        assert!(sanitised.contains("draft.md"));
        assert!(!sanitised.contains("priya"));
        assert!(!sanitised.contains("AppData"));
    }

    #[test]
    fn a_posix_path_is_shortened_the_same_way() {
        let sanitised = sanitise_failure("/home/priya/runs/r-8f2/working.xlsx could not be read.");
        assert!(sanitised.contains("working.xlsx"));
        assert!(!sanitised.contains("/home/priya"));
    }

    /// A backtrace is dropped whole. It names internal symbols, costs a large
    /// slice of the window, and tells a model nothing it can act on.
    #[test]
    fn a_backtrace_does_not_reach_the_model() {
        let sanitised = sanitise_failure(
            "the workbook could not be written\n\
             stack backtrace:\n\
             0: sarathi::artifacts::render\n\
             at /build/src/artifacts/mod.rs:88\n\
             1: sarathi::orchestrator::runner::write",
        );
        assert!(sanitised.contains("the workbook could not be written"));
        assert!(!sanitised.contains("backtrace"));
        assert!(!sanitised.contains("sarathi::artifacts"));
    }

    /// The recoverable sentence survives. An error a model cannot act on costs
    /// a step and teaches it nothing, so sanitising removes noise, not detail.
    #[test]
    fn the_sentence_telling_the_model_what_to_do_is_kept() {
        let sanitised = sanitise_failure(
            "that page range could not be read: no such document. \
             Search first, then use the documentSha256 from a passage you retrieved.",
        );
        assert!(sanitised.contains("Search first"));
        assert!(sanitised.contains("documentSha256"));
    }

    /// Ordinary prose with a colon in it is not mistaken for a stack frame.
    #[test]
    fn a_message_with_a_colon_survives_intact() {
        let sanitised = sanitise_failure("Refused: that content is above the size limit.");
        assert_eq!(sanitised, "Refused: that content is above the size limit.");
    }

    // ── Metadata ─────────────────────────────────────────────────────────

    /// Nothing in the shipped catalogue reaches outside this machine.
    ///
    /// The claim the product is built on. A tool added later that needs the
    /// network has to change this test, which is the point of asserting it.
    #[test]
    fn no_shipped_tool_reaches_outside_this_machine() {
        for tool in ToolName::ALL {
            assert_ne!(
                spec_for(*tool).network,
                NetworkUse::Outbound,
                "{} would reach outside the machine",
                tool.as_str()
            );
        }
    }

    /// Work mode offers every tool that ships; Provisioning is where the
    /// catalogue would narrow, and it does not narrow today.
    #[test]
    fn every_shipped_tool_is_offered_in_work_mode() {
        for tool in ToolName::ALL {
            assert!(
                spec_for(*tool).network.permitted_in(crate::sovereignty::OperatingMode::Work),
                "{} is not offered in Work mode",
                tool.as_str()
            );
        }
    }

    /// A read-only tool never needs an approval, and anything that needs one is
    /// not read-only. The two are different questions with the same answer, and
    /// a tool where they disagreed would either interrupt a person for a read or
    /// let a write through unasked.
    #[test]
    fn being_read_only_and_needing_nobody_are_the_same_set() {
        for tool in ToolName::ALL {
            let spec = spec_for(*tool);
            if tool.is_read_only() {
                assert!(
                    !spec.needs_approval,
                    "{} reads but interrupts a person",
                    tool.as_str()
                );
                assert_eq!(spec.approval_class, ApprovalClass::Automatic);
            } else {
                assert!(
                    spec.approval_class != ApprovalClass::Automatic,
                    "{} causes an effect without anyone approving it",
                    tool.as_str()
                );
            }
        }
    }

    /// Every tool bounds what it can hand back.
    #[test]
    fn every_tool_has_a_response_ceiling_it_can_actually_fill() {
        for tool in ToolName::ALL {
            let limit = spec_for(*tool).max_response_bytes;
            assert!(
                limit > TRUNCATION_NOTICE_BUDGET,
                "{}'s ceiling leaves no room for its own truncation notice",
                tool.as_str()
            );
        }
    }

    #[test]
    fn arguments_are_read_out_of_a_call_safely() {
        let call = ToolCall::new("search_documents", serde_json::json!({ "query": "wall thickness" }));
        assert_eq!(call.text("query"), Some("wall thickness"));
        assert_eq!(call.text("missing"), None);

        // A wrong-typed argument reads as absent rather than panicking.
        let wrong = ToolCall::new("search_documents", serde_json::json!({ "query": 42 }));
        assert_eq!(wrong.text("query"), None);
    }
}
