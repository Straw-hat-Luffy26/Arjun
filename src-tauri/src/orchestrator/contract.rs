//! One registration contract per tool.
//!
//! ## What was scattered, and what this gathers
//!
//! Everything the system knows about a tool was true somewhere, and nowhere in
//! one place:
//!
//! | Fact | Authority |
//! |---|---|
//! | wire name, legacy name, aliases | [`ToolName`] |
//! | arguments, permission, approval, network, limits | [`spec_for`] |
//! | what it does to the world | [`class_of`] |
//! | whether an intent is written first | [`is_side_effecting`] |
//! | what may be done when it fails | [`retry_policy_of`] |
//! | which code answers it | a `match` in `agent_runtime::execute`, and another in `LocalToolRunner::run` |
//! | what must exist before it can work | nowhere |
//!
//! Each of those stays the authority for its own fact -- a second copy of an
//! approval class would be a second opinion that eventually disagrees. What
//! this module adds is the missing rows (route, prerequisite, output kind,
//! cancellation) and a single [`ToolContract`] that reads all of them, so
//! "what is `agent.delegate_readonly`?" has one answer that a test, the
//! catalogue and the runtime's conformance check can all ask for.
//!
//! ## Why the missing rows are exhaustive `match`es
//!
//! For the same reason [`class_of`] is: a tool added later does not compile
//! until somebody has said which code answers it and what it needs. A default
//! would be the one place a new tool could slip through without anyone
//! deciding.
//!
//! ## The published copy
//!
//! [`published_contract`] renders the contract as JSON, and a test holds
//! `agent-runtime/src/tool-contract.json` byte-equal to it. The runtime's
//! conformance test reads that file and checks the TypeScript catalogue
//! against it -- arguments, read/write mode, aliases and effect class. Before
//! this, the two sides were compared through lists typed by hand on the
//! TypeScript side, and those lists had drifted exactly where it mattered: the
//! gateway required arguments the model's schema did not offer, so four tools
//! were in every catalogue and could not be called. Generated, never typed --
//! the rule the fixture manifest and the SBOM already follow.

use serde::Serialize;
use serde_json::{json, Value};

use super::tools::{spec_for, ApprovalClass, ArgumentKind, ArgumentSpec, NetworkUse, ToolName};
use crate::agent_runtime::events::idempotency::is_side_effecting;
use crate::agent_runtime::tool_policy::{class_of, retry_policy_of, Reconciliation, ToolClass};
use crate::identity::Permission;

/// Bumped when the published JSON changes shape (not when a tool changes).
pub const CONTRACT_VERSION: u32 = 1;

/// Which code answers a call once the gateway has allowed it.
///
/// Both are reached from one place, `agent_runtime::execute`, which is the
/// only production dispatcher: the runtime's `tool.execute` request lands
/// there, redeems its grant, re-asks the gateway and then routes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum Route {
    /// An arm of `agent_runtime::execute` itself, which holds what a per-call
    /// runner cannot: the signed-in owner, the conversation, and the run's
    /// accumulated evidence, calculations and artifacts.
    AgentPath,
    /// `LocalToolRunner::run`, built per call by `agent_runtime::runner_for`
    /// with the run's inherited policy, workspace, model registry and task.
    Runner,
}

/// Something that has to exist before a tool can do its work.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum Prerequisite {
    /// A workspace root for the run. The gateway refuses a path-taking call
    /// when the run has none.
    WorkspaceRoot,
    /// A worker registered for at least one declared subagent role. Without
    /// one the tool is withheld from the catalogue rather than offered and
    /// refused.
    SubagentWorker,
    /// The model registry and the parent run's model, so a child can be
    /// routed to a real model rather than an asserted one.
    ModelRegistry,
    /// A container runtime the sandbox policy accepts.
    ContainerSandbox,
    /// The image-region and table index.
    MultimodalIndex,
    /// At least one calculation already run in this task, for the calculation
    /// workbook form.
    RunCalculations,
    /// The pinned layout engine and rasteriser (`artifacts::render`). Unmet
    /// is reported as an unavailable rung, never as a pass.
    PageRenderer,
    /// The run is attached to a conversation, whose artifact store the tool
    /// reads.
    ConversationArtifacts,
}

/// Where a prerequisite is checked.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum CheckedAt {
    /// Before the model is shown the tool. Unmet means not offered.
    Catalogue,
    /// At `ToolGateway::decide`. Unmet means refused before anything runs.
    Gateway,
    /// Inside the handler, which refuses and says what is missing.
    Handler,
}

impl Prerequisite {
    pub const fn checked_at(self) -> CheckedAt {
        match self {
            Prerequisite::WorkspaceRoot => CheckedAt::Gateway,
            Prerequisite::SubagentWorker => CheckedAt::Catalogue,
            Prerequisite::ModelRegistry
            | Prerequisite::ContainerSandbox
            | Prerequisite::MultimodalIndex
            | Prerequisite::RunCalculations
            | Prerequisite::PageRenderer
            | Prerequisite::ConversationArtifacts => CheckedAt::Handler,
        }
    }
}

/// What a successful call hands back.
///
/// The runtime's `tool-names.ts` classifies tools the same way (evidence,
/// calculation, artifact, code execution) to decide what its working notes
/// record; the conformance test holds the two to one answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum OutputKind {
    /// Passages numbered `[En]` against the run's evidence table.
    Evidence,
    /// A deterministic result with its working, recorded for the workbook.
    Calculation,
    /// A file written into the run's workspace and registered as an artifact.
    Artifact,
    /// The outcome of running a program.
    Execution,
    /// A typed [`crate::subagents::ChildResult`], rendered for the parent.
    ChildResult,
    /// Prose, a listing or a structure. Bounded, cited where it cites.
    Text,
}

/// What stopping a call actually stops.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum Cancellation {
    /// The runtime stops *waiting* -- at the tool's timeout or at Stop -- and
    /// tells the model the outcome is not to be relied on. The Rust handler is
    /// not interrupted. Harmless for a call that only reads or whose change is
    /// the run's own to undo.
    AbandonWait,
    /// The same, for a call with an effect: the intent written before it ran
    /// stays `pending`, and is promoted to `unknown` rather than retried, so an
    /// abandoned write can never be quietly repeated.
    AbandonWaitWithIntent,
    /// Bounded in Rust as well: the work is stopped at its own deadline -- a
    /// child at `limits.max_duration_seconds`, a program when its container is
    /// killed.
    RustDeadline,
}

/// Everything registered about one tool.
#[derive(Debug, Clone)]
pub struct ToolContract {
    pub tool: ToolName,
    /// Every other spelling that resolves to it.
    pub aliases: Vec<&'static str>,
    pub required: &'static [ArgumentSpec],
    pub optional: &'static [ArgumentSpec],
    /// The approver's sentence.
    pub summary: &'static str,
    /// What the model is told its arguments must carry, where names do not say.
    pub argument_notes: Option<&'static str>,
    pub permission: Permission,
    pub effect: ToolClass,
    pub read_only: bool,
    pub approval: ApprovalClass,
    pub network: NetworkUse,
    pub timeout_seconds: u64,
    pub max_input_bytes: Option<u64>,
    /// Output validation: a result above this is cut, deterministically and
    /// with a notice (`tools::truncate_response`), and a failure is sanitised
    /// (`tools::sanitise_failure`) -- on fresh and replayed results alike.
    pub max_response_bytes: usize,
    pub output: OutputKind,
    pub cancellation: Cancellation,
    /// Whether an intent is written to the idempotency ledger before it runs.
    pub intent_before_effect: bool,
    pub safe_to_retry: bool,
    pub max_retries: u8,
    pub reconciliation: Reconciliation,
    pub prerequisites: &'static [Prerequisite],
    pub route: Route,
    /// The function that does the work, for a person tracing a call.
    pub handler: &'static str,
}

/// Which code answers `tool`, and the function that does it.
///
/// Consulted by `agent_runtime::execute` before its fallback, so a tool
/// registered here for the agent path that reached the runner is an error with
/// a name rather than a quiet "served elsewhere" refusal.
pub const fn route_of(tool: ToolName) -> (Route, &'static str) {
    use Route::*;
    match tool {
        ToolName::CreateDocx => (AgentPath, "agent_runtime::artifacts::create_docx_with_evidence"),
        ToolName::CreateXlsx => (AgentPath, "agent_runtime::artifacts::create_xlsx"),
        ToolName::CreatePptx => (AgentPath, "agent_runtime::artifacts::create_pptx"),
        ToolName::SearchDocuments => (
            AgentPath,
            "LocalToolRunner::search_hits, then agent_runtime::retrieval::record",
        ),
        ToolName::LoadMoreEvidence => (
            AgentPath,
            "LocalToolRunner::region_hits, then agent_runtime::retrieval::record_region",
        ),
        ToolName::MemoryRecallAuthorized => (AgentPath, "agent_runtime::memory_api::recall_authorized"),
        ToolName::MemoryPromoteApproved => (AgentPath, "agent_runtime::memory_api::promote_approved"),
        ToolName::CapabilitySearch => (AgentPath, "agent_runtime::capability_search"),
        ToolName::ValidateArtifact => (AgentPath, "agent_runtime::validate"),
        ToolName::ReadAttachedPages => (AgentPath, "agent_runtime::read_attached_pages"),
        ToolName::SearchAttachedDocuments => (AgentPath, "agent_runtime::search_attached_documents"),
        ToolName::BuildDocumentGraph => (AgentPath, "agent_runtime::build_document_graph"),
        ToolName::NotebookList => (AgentPath, "agent_runtime::notebook_list"),
        ToolName::NotebookCreate => (AgentPath, "agent_runtime::notebook_create"),
        ToolName::NotebookRename => (AgentPath, "agent_runtime::notebook_rename"),
        ToolName::NotebookDelete => (AgentPath, "agent_runtime::notebook_delete"),
        ToolName::NotebookSources => (AgentPath, "agent_runtime::notebook_sources"),
        ToolName::NotebookAddSource => (AgentPath, "agent_runtime::notebook_add_source"),
        ToolName::NotebookRemoveSource => (AgentPath, "agent_runtime::notebook_remove_source"),
        ToolName::ArtifactList => (AgentPath, "agent_runtime::artifact_list"),
        ToolName::ArtifactRead => (AgentPath, "agent_runtime::artifact_read"),
        ToolName::CreateChart => (AgentPath, "agent_runtime::create_chart"),
        ToolName::CreateDiagram => (AgentPath, "agent_runtime::create_diagram"),
        ToolName::CreatePdf => (AgentPath, "agent_runtime::create_pdf"),
        ToolName::CreateTable => (AgentPath, "agent_runtime::create_table"),
        ToolName::ArtifactManifest => (AgentPath, "agent_runtime::artifact_tools::manifest"),
        ToolName::ArtifactReadVersion => (AgentPath, "agent_runtime::artifact_tools::read_version"),
        ToolName::ArtifactReadRegion => (AgentPath, "agent_runtime::artifact_tools::read_region"),
        ToolName::ArtifactListTemplates => (AgentPath, "agent_runtime::artifact_tools::list_templates"),
        ToolName::ArtifactValidate => (AgentPath, "agent_runtime::artifact_tools::validate_version"),
        ToolName::ArtifactRender => (AgentPath, "agent_runtime::artifact_tools::render_version"),
        ToolName::ArtifactDiff => (AgentPath, "agent_runtime::artifact_tools::diff"),
        ToolName::ArtifactResolveEvidence => (AgentPath, "agent_runtime::artifact_tools::resolve_evidence"),
        ToolName::ArtifactRegisterVersion => (AgentPath, "agent_runtime::artifact_tools::register_version"),
        ToolName::ArtifactEdit => (AgentPath, "agent_runtime::artifact_tools::edit_version"),
        ToolName::MediaExtractFindings => (Runner, "LocalToolRunner::extract_findings"),
        ToolName::KnowledgeMultimodalRetrieve => (Runner, "LocalToolRunner::multimodal_retrieve"),
        ToolName::ReadScopedFile => (Runner, "LocalToolRunner::read"),
        ToolName::WriteScopedFile => (Runner, "LocalToolRunner::write"),
        ToolName::RunCalculation => (
            Runner,
            "LocalToolRunner::calculate, then the run's calculation table",
        ),
        ToolName::ExecuteCode => (Runner, "LocalToolRunner::execute_code"),
        ToolName::SovereigntyGetEvidence => (Runner, "LocalToolRunner::sovereignty_evidence"),
        ToolName::AgentDelegateReadonly => (
            Runner,
            "LocalToolRunner::delegate_to_subagent, then SubagentManager::spawn",
        ),
    }
}

/// What must exist before `tool` can work.
pub const fn prerequisites_of(tool: ToolName) -> &'static [Prerequisite] {
    use Prerequisite::*;
    match tool {
        ToolName::ReadScopedFile
        | ToolName::WriteScopedFile
        | ToolName::CreateDocx
        | ToolName::CreatePptx
        | ToolName::ValidateArtifact => &[WorkspaceRoot],
        // The workbook form writes the calculations this task already ran, and
        // refuses rather than write an empty one (`argument_guidance`).
        ToolName::CreateXlsx => &[WorkspaceRoot, RunCalculations],
        ToolName::AgentDelegateReadonly => &[SubagentWorker, ModelRegistry],
        ToolName::ExecuteCode => &[ContainerSandbox],
        ToolName::KnowledgeMultimodalRetrieve => &[MultimodalIndex],
        ToolName::ArtifactValidate | ToolName::ArtifactRender => &[ConversationArtifacts, PageRenderer],
        ToolName::ArtifactManifest
        | ToolName::ArtifactReadVersion
        | ToolName::ArtifactReadRegion
        | ToolName::ArtifactDiff
        | ToolName::ArtifactResolveEvidence
        | ToolName::ArtifactRegisterVersion
        | ToolName::ArtifactEdit => &[ConversationArtifacts],
        // `media.extract_findings` reads text the ingest pipeline already
        // extracted. It calls no OCR engine at request time -- a page nothing
        // read comes back named as unread -- so it has no engine to require.
        ToolName::MediaExtractFindings
        | ToolName::SearchDocuments
        | ToolName::LoadMoreEvidence
        | ToolName::MemoryRecallAuthorized
        | ToolName::MemoryPromoteApproved
        | ToolName::RunCalculation
        | ToolName::CapabilitySearch
        | ToolName::SovereigntyGetEvidence
        | ToolName::ReadAttachedPages
        | ToolName::SearchAttachedDocuments
        | ToolName::BuildDocumentGraph
        | ToolName::NotebookList
        | ToolName::NotebookCreate
        | ToolName::NotebookRename
        | ToolName::NotebookDelete
        | ToolName::NotebookSources
        | ToolName::NotebookAddSource
        | ToolName::NotebookRemoveSource
        | ToolName::CreateChart
        | ToolName::CreateDiagram
        | ToolName::CreatePdf
        | ToolName::CreateTable
        | ToolName::ArtifactList
        | ToolName::ArtifactRead
        | ToolName::ArtifactListTemplates => &[],
    }
}

/// What a successful call to `tool` hands back.
pub const fn output_of(tool: ToolName) -> OutputKind {
    use OutputKind::*;
    match tool {
        ToolName::SearchDocuments
        | ToolName::LoadMoreEvidence
        | ToolName::MediaExtractFindings
        | ToolName::KnowledgeMultimodalRetrieve => Evidence,
        ToolName::RunCalculation => Calculation,
        ToolName::WriteScopedFile
        | ToolName::CreateDocx
        | ToolName::CreateXlsx
        | ToolName::CreatePptx
        | ToolName::CreateChart
        | ToolName::CreateDiagram
        | ToolName::CreatePdf
        | ToolName::CreateTable
        | ToolName::ArtifactEdit => Artifact,
        ToolName::ExecuteCode => Execution,
        ToolName::AgentDelegateReadonly => ChildResult,
        ToolName::MemoryRecallAuthorized
        | ToolName::MemoryPromoteApproved
        | ToolName::ReadScopedFile
        | ToolName::ValidateArtifact
        | ToolName::CapabilitySearch
        | ToolName::SovereigntyGetEvidence
        | ToolName::ReadAttachedPages
        | ToolName::SearchAttachedDocuments
        | ToolName::BuildDocumentGraph
        | ToolName::NotebookList
        | ToolName::NotebookCreate
        | ToolName::NotebookRename
        | ToolName::NotebookDelete
        | ToolName::NotebookSources
        | ToolName::NotebookAddSource
        | ToolName::NotebookRemoveSource
        | ToolName::ArtifactList
        | ToolName::ArtifactRead
        | ToolName::ArtifactManifest
        | ToolName::ArtifactReadVersion
        | ToolName::ArtifactReadRegion
        | ToolName::ArtifactListTemplates
        | ToolName::ArtifactValidate
        | ToolName::ArtifactRender
        | ToolName::ArtifactDiff
        | ToolName::ArtifactResolveEvidence
        // Publishing changes a version's stage and writes no file.
        | ToolName::ArtifactRegisterVersion => Text,
    }
}

/// What stopping a call to `tool` stops.
pub const fn cancellation_of(tool: ToolName) -> Cancellation {
    match tool {
        // Both are bounded where they run, not only where they are waited on.
        ToolName::AgentDelegateReadonly | ToolName::ExecuteCode => Cancellation::RustDeadline,
        _ if is_side_effecting(tool) => Cancellation::AbandonWaitWithIntent,
        _ => Cancellation::AbandonWait,
    }
}

/// The whole contract for one tool.
pub fn contract_for(tool: ToolName) -> ToolContract {
    let spec = spec_for(tool);
    let retry = retry_policy_of(tool);
    let (route, handler) = route_of(tool);
    ToolContract {
        tool,
        aliases: tool
            .accepted_spellings()
            .into_iter()
            .filter(|spelling| *spelling != tool.as_str())
            .collect(),
        required: spec.arguments,
        optional: spec.optional_arguments,
        summary: tool.describe(),
        argument_notes: tool.argument_guidance(),
        permission: spec.permission,
        effect: class_of(tool),
        read_only: tool.is_read_only(),
        approval: spec.approval_class,
        network: spec.network,
        timeout_seconds: spec.timeout.as_secs(),
        max_input_bytes: spec.max_bytes,
        max_response_bytes: spec.max_response_bytes,
        output: output_of(tool),
        cancellation: cancellation_of(tool),
        intent_before_effect: is_side_effecting(tool),
        safe_to_retry: retry.safe_to_retry,
        max_retries: retry.max_retries,
        reconciliation: retry.reconciliation,
        prerequisites: prerequisites_of(tool),
        route,
        handler,
    }
}

const fn kind_name(kind: ArgumentKind) -> &'static str {
    match kind {
        ArgumentKind::Text => "text",
        ArgumentKind::Path => "path",
        ArgumentKind::Integer => "integer",
        ArgumentKind::Object => "object",
        ArgumentKind::List => "list",
    }
}

const fn reconciliation_name(reconciliation: Reconciliation) -> &'static str {
    match reconciliation {
        Reconciliation::NotNeeded => "notNeeded",
        Reconciliation::InspectArtifact => "inspectArtifact",
        Reconciliation::AskAPerson => "askAPerson",
    }
}

fn arguments_json(arguments: &[ArgumentSpec]) -> Vec<Value> {
    arguments
        .iter()
        .map(|argument| json!({ "name": argument.name, "kind": kind_name(argument.kind) }))
        .collect()
}

/// The contract as the runtime's conformance test reads it.
///
/// Sorted by wire name and free of anything that varies between runs, so the
/// same build always renders the same bytes.
pub fn published_contract() -> Value {
    let mut tools: Vec<ToolName> = ToolName::ALL.to_vec();
    tools.sort_by_key(|tool| tool.as_str());
    json!({
        "contractVersion": CONTRACT_VERSION,
        "generatedBy": "src-tauri/src/orchestrator/contract.rs -- regenerate with \
                        ARJUN_WRITE_TOOL_CONTRACT=1 cargo test --lib orchestrator::contract",
        "tools": tools.into_iter().map(|tool| {
            let contract = contract_for(tool);
            json!({
                "name": tool.as_str(),
                "aliases": contract.aliases,
                "required": arguments_json(contract.required),
                "optional": arguments_json(contract.optional),
                "readOnly": contract.read_only,
                "effect": contract.effect.as_str(),
                "sideEffecting": contract.intent_before_effect,
                "approvalClass": contract.approval,
                "permission": contract.permission,
                "network": contract.network,
                "timeoutSeconds": contract.timeout_seconds,
                "maxResponseBytes": contract.max_response_bytes,
                "output": contract.output,
                "cancellation": contract.cancellation,
                "retry": {
                    "safeToRetry": contract.safe_to_retry,
                    "maxRetries": contract.max_retries,
                    "reconciliation": reconciliation_name(contract.reconciliation),
                },
                "prerequisites": contract.prerequisites.iter().map(|prerequisite| json!({
                    "requires": prerequisite,
                    "checkedAt": prerequisite.checked_at(),
                })).collect::<Vec<_>>(),
                "route": contract.route,
                "handler": contract.handler,
            })
        }).collect::<Vec<_>>(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn published_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("src-tauri has a parent")
            .join("agent-runtime")
            .join("src")
            .join("tool-contract.json")
    }

    /// The runtime's copy is this build's contract, byte for byte.
    ///
    /// Regenerated, never edited: `ARJUN_WRITE_TOOL_CONTRACT=1` rewrites it,
    /// and the diff is the review. A contract that changed here without the
    /// file changing is exactly the drift that left four tools uncallable.
    #[test]
    fn the_published_contract_is_current() {
        let rendered = format!(
            "{}\n",
            serde_json::to_string_pretty(&published_contract()).expect("renders")
        );
        let path = published_path();
        if std::env::var_os("ARJUN_WRITE_TOOL_CONTRACT").is_some() {
            std::fs::write(&path, &rendered).expect("writes the published contract");
        }
        let held = std::fs::read_to_string(&path)
            .unwrap_or_default()
            .replace("\r\n", "\n");
        assert!(
            held == rendered,
            "agent-runtime/src/tool-contract.json is not this build's tool contract. \
             Regenerate it with `ARJUN_WRITE_TOOL_CONTRACT=1 cargo test --manifest-path \
             src-tauri/Cargo.toml --lib orchestrator::contract` and review the diff."
        );
    }

    /// Read-only, effect class and approval tell one story.
    #[test]
    fn a_contract_never_contradicts_itself() {
        for tool in ToolName::ALL.iter().copied() {
            let contract = contract_for(tool);
            assert_eq!(
                contract.read_only,
                contract.effect == ToolClass::ReadOnly
                    // Delegation is read-only in what the child may do and
                    // `Reversible` in what it leaves behind: a subagent record
                    // and budget spent. Both are true, and each governs a
                    // different question.
                    || tool == ToolName::AgentDelegateReadonly
                    // Deterministic arithmetic reads nothing outside the task
                    // and writes only the run's own calculation table.
                    || tool == ToolName::RunCalculation,
                "{} is read-only in one table and {} in another",
                tool.as_str(),
                contract.effect.as_str()
            );
            if contract.intent_before_effect {
                assert!(!contract.safe_to_retry, "{} retries an effect", tool.as_str());
                assert_ne!(contract.cancellation, Cancellation::AbandonWait, "{}", tool.as_str());
            }
            assert!(contract.timeout_seconds > 0, "{} has no timeout", tool.as_str());
            // A required argument is not also optional.
            for argument in contract.required {
                assert!(
                    !contract.optional.iter().any(|other| other.name == argument.name),
                    "{}'s {:?} is both required and optional",
                    tool.as_str(),
                    argument.name
                );
            }
        }
    }

    /// The gateway-checked prerequisite is the one the gateway checks.
    #[test]
    fn a_workspace_prerequisite_is_exactly_a_workspace_scoped_tool() {
        for tool in ToolName::ALL.iter().copied() {
            assert_eq!(
                prerequisites_of(tool).contains(&Prerequisite::WorkspaceRoot),
                spec_for(tool).scoped_to_workspace,
                "{}",
                tool.as_str()
            );
        }
    }

    /// Every alias a contract lists resolves back to that tool, and no two
    /// tools claim one spelling.
    #[test]
    fn every_listed_alias_resolves_to_its_own_tool_and_to_no_other() {
        let mut seen = std::collections::BTreeMap::new();
        for tool in ToolName::ALL.iter().copied() {
            for spelling in tool.accepted_spellings() {
                assert_eq!(ToolName::from_str(spelling), Some(tool), "{spelling}");
                if let Some(other) = seen.insert(spelling, tool) {
                    panic!("{spelling} is claimed by {} and {}", other.as_str(), tool.as_str());
                }
            }
        }
    }

    /// The published document names every tool the gateway knows, once.
    #[test]
    fn the_published_contract_covers_every_tool() {
        let published = published_contract();
        let names: Vec<&str> = published["tools"]
            .as_array()
            .expect("a list")
            .iter()
            .map(|tool| tool["name"].as_str().expect("a name"))
            .collect();
        assert_eq!(names.len(), ToolName::ALL.len());
        for tool in ToolName::ALL {
            assert!(names.contains(&tool.as_str()), "{} is not published", tool.as_str());
        }
    }
}
