//! Exactly what a turn was built from, written down before the model sees it.
//!
//! ## The failure this exists to remove
//!
//! A run could be resumed, and the resumption had no idea what the original
//! attempt had been *given*. `agent_resume_run` rebuilt its request with
//! `conversation_id: None`, `message_id: None`, `attachments: []` and
//! `research: None`, so the continuation of a turn that had answered from five
//! selected drawings answered from nothing, in a conversation of its own, under
//! whatever model routing happened to pick that day.
//!
//! None of that was recoverable from the event history either. The history says
//! a search happened and what it returned; it does not say which five documents
//! the person had selected, at which revision, or which notebook's graph was
//! frozen for the turn. So there was no record anywhere a resumption could
//! reconstruct the original context from — only a record of what that context
//! had produced.
//!
//! A manifest is that record. It is written into the checkpoint, so it is
//! sealed by the same hash and read on the same path.
//!
//! ## What is in it, and what is deliberately not
//!
//! Identifiers, hashes, counts. No passage, no page text, no answer, no
//! document body — the same rule [`super::events::checkpoint`] holds, and for
//! the same reason: this is read by the recovery path before anybody has
//! authenticated, so it must contain nothing that authentication protects.
//!
//! A document is named by its sha256, which *is* its version: the store is
//! content-addressed, so the same hash is the same bytes and a different hash
//! is a different document. Nothing else is needed to pin a version.
//!
//! ## Why the research binding points at a manifest rather than copying it
//!
//! `knowledge::graph::research::EvidenceManifest` already records what a
//! notebook turn retrieved — entries, limitations, graph revision, the sources
//! that contributed nothing — and `record_research_turn` stores it keyed by
//! `run_id`. Copying that here would make two records of one fact that could
//! disagree. So this holds the key and the frozen *selection*, which is the
//! part a resumption needs before it is allowed to retrieve anything again.
//!
//! ## What this is not, yet
//!
//! Phase 6 extends this with graph retrieval: the authorised graph revision, the
//! selected memory revisions, the token accounting and the final-projection
//! record. [`MANIFEST_VERSION`] exists so that extension is a version bump
//! rather than a silent change of meaning.

use serde::{Deserialize, Serialize};

use super::chat_memory_bus::OmittedPin;
use super::events::model::digest;
use crate::knowledge::graph::SourceSelection;

/// The layout of a stored manifest.
///
/// Bumped when a field changes meaning. A manifest written under a version this
/// build does not understand is refused rather than partly read — resuming a
/// run from a description of its context you can only half parse is exactly the
/// case where being wrong is silent.
pub const MANIFEST_VERSION: u32 = 3;

/// Versions this build can still read.
///
/// Version 1 is a Phase 2 manifest: conversation, documents and notebook scope,
/// with no graph binding, no budget record and no retrieval record. It is read
/// as exactly that — a turn compiled before the graph existed — rather than
/// refused, because refusing would make every run checkpointed by the previous
/// build unresumable for a reason that has nothing to do with whether it is
/// safe to resume.
///
/// Version 2 recorded a graph revision that was *supplied* to the compiler and
/// read no rows at it — see [`GraphBinding::graph_revision`]. Version 3 records
/// the cursor the rows were actually read at, and the scopes they came from. A
/// version 2 manifest is still read, as what it is.
pub const READABLE_MANIFEST_VERSIONS: &[u32] = &[1, 2, 3];

/// One document the turn was given, pinned to the bytes it was given.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DocumentBinding {
    /// The content address. This *is* the version.
    pub sha256: String,
    /// What it was called on screen. Carried for the sentence a refusal writes,
    /// never for lookup — the hash is the identity.
    pub name: String,
    pub pages: u32,
}

/// The notebook scope a turn was frozen to.
///
/// Every field here is the *selection*, not the result. A resumption re-runs
/// `notebook_retrieval::resolve` against this, which re-checks ownership and
/// membership as they are now — so a source that has since been removed or
/// revoked produces a refusal rather than a quietly smaller answer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResearchBinding {
    pub notebook_id: String,
    /// The run whose `notebook_research_turns` row holds the evidence manifest.
    ///
    /// The run id, because that is the key `record_research_turn` writes under
    /// and `run_id` survives a resumption unchanged. Named separately from the
    /// manifest's own `run_id` so it stays right if the two ever diverge.
    pub manifest_run_id: String,
    /// Exactly what the person chose, as the type that keeps the three cases
    /// apart.
    ///
    /// A `Vec<String>` was the obvious field here and it is the wrong one: an
    /// empty list cannot distinguish "every source" from "deliberately none",
    /// and that ambiguity is the one `SourceSelection` exists to remove. A
    /// resumption that flattened them would answer a question scoped to no
    /// evidence using the whole notebook.
    ///
    /// `All` is already resolved to a concrete list when the turn is frozen, so
    /// what is stored here pins the sources the original turn actually read —
    /// a source added to the notebook since does not join the continuation.
    pub selection: SourceSelection,
    pub node_ids: Vec<String>,
    pub assertion_ids: Vec<String>,
    /// The notebook's graph revision when the turn ran, when it had one.
    pub graph_revision: Option<String>,
}

/// What the conversation contributed, and what it could not.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HistoryBinding {
    pub carried: u32,
    pub dropped: u32,
    pub tokens: u32,
    /// The pins in force, as stored. See [`super::pins`].
    #[serde(default)]
    pub pinned: Vec<String>,
    /// Protected content the turn could not carry, and why.
    #[serde(default)]
    pub omitted_pins: Vec<OmittedPin>,
}

/// The graph revision a turn was authorised against, and what it selected.
///
/// The revision is a *cursor*, not a timestamp: `agent_memory_log.revision` is
/// monotonic, so "compiled at revision 412" is a position another reader can
/// resolve to exactly the same set. A `MAX(updated_at)` would not be — two
/// commits in the same second are indistinguishable, and a resumption asking
/// what had been established by then would get a different answer each time.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GraphBinding {
    /// The changefeed position the selected rows were read at.
    ///
    /// Read in the same transaction as the rows (version 3 onward), so it
    /// names exactly the last change folded into them. A version 2 manifest
    /// holds whatever the caller supplied instead, which is plan §3's finding 1.
    pub graph_revision: i64,
    /// The position the caller asked to compile at, when it asked for one and
    /// the graph had moved since.
    ///
    /// Recorded rather than pretended: this store serves current rows, not
    /// historical ones, so a compilation asked for "revision 412" that read at
    /// 415 says so instead of labelling 415's rows as 412's.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub requested_revision: Option<i64>,
    /// Exactly which memory items, at which revision of each.
    ///
    /// The revision matters: an item corrected after this turn was compiled is
    /// a different claim, and a manifest naming only the item id would describe
    /// a context the model never saw.
    pub selected: Vec<SelectedItem>,
}

/// One memory item a turn carried, pinned to its revision.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SelectedItem {
    pub item_id: String,
    pub revision: u64,
    /// Why this one was chosen, in a word: `mandatory`, `neighbour`, `recall`,
    /// `evidence`. Recorded so a person reading the manifest can tell what the
    /// compiler was doing, not only what it produced.
    pub reason: String,
    /// Which scope it came from: `task`, `project`, `procedure` or
    /// `preference`. Absent on a version 2 manifest, which read the task only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scope: Option<String>,
    /// Its precedence when it and something else disagree: 0 is the task's own
    /// state and the operator's corrections, and every later scope yields to
    /// every earlier one. See `context_compiler::ScopeRole`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub precedence: Option<u8>,
}

/// How the window was divided, and by what counting.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BudgetRecord {
    /// The served window. Not the trained one.
    pub window: u32,
    /// Held back for the tool schemas the runtime will attach.
    pub reserved_tool_schemas: u32,
    /// Held back for the model's own output.
    pub reserved_output: u32,
    /// Held back for chat-template framing — the tokens a template spends on
    /// role markers and separators, which are real and are nobody's content.
    pub reserved_framing: u32,
    /// Held back for what the counting may have missed. Zero on a manifest
    /// written before the reserve existed, and then not serialised, so the
    /// seal on those manifests still verifies.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub reserved_safety: u32,
    /// Where `window` came from: `server` when the server said what it was
    /// started with, `registryDeclared` when it did not.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub window_source: Option<String>,
    /// What was left for context after the three reserves.
    pub available: u32,
    /// What the compiler actually spent.
    pub spent: u32,
    /// How the counting was done: `tokenizer` when the model's own tokenizer
    /// answered, `estimate` when nothing better was available.
    ///
    /// Named rather than assumed, because an estimate presented as a count is
    /// how a turn overruns a window it was told it fitted.
    pub counted_by: String,
    /// What the server said it actually received, once it has said.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub server_reported_tokens: Option<u32>,
}

fn is_zero(value: &u32) -> bool {
    *value == 0
}

/// What one scope contributed to a round, in counts.
///
/// Counts and a role, never the scope's key: the key names a person or a
/// project, and this record is read by the recovery path before anybody has
/// signed in.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ScopeRecord {
    /// `task`, `project`, `procedure` or `preference`.
    pub role: String,
    /// Its precedence; lower wins. See [`SelectedItem::precedence`].
    pub precedence: u8,
    /// Items the reader was authorised to see in it.
    pub read: u32,
    /// Items carried into the round.
    pub selected: u32,
    /// Items left out, for any reason the omissions list names.
    pub omitted: u32,
}

/// Why something was left out.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Omission {
    /// What it was: an item id, a source hash, a pin.
    pub what: String,
    /// One of `budget`, `deduplicated`, `unauthorised`, `revoked`, `expired`,
    /// `notEstablished` (a proposal outside the task's own scope),
    /// `notApplicable` (a procedure written for another role), or
    /// `mandatoryOverflow`.
    pub reason: String,
    /// The sentence a person reads.
    pub detail: String,
}

/// How retrieval was done, and whether it was what it looks like.
///
/// ## Why degradation is a field rather than a log line
///
/// Because "semantic retrieval found nothing relevant" and "there is no
/// embedding model on this machine so only keywords were tried" produce the
/// same empty result and mean completely different things. A manifest that
/// could not tell them apart would let a lexical-only deployment be reported as
/// a semantic one for its whole life.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RetrievalRecord {
    /// `lexical` or `hybrid`. Never `semantic` unless vectors were searched.
    pub mode: String,
    /// The embedding model, when one was used.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding_model_id: Option<String>,
    /// Its dimension. Recorded so an index built under a different one is
    /// detectable rather than silently mismatched.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding_dimension: Option<u32>,
    /// The index build this searched.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub index_version: Option<String>,
    /// True when the search was narrower than the deployment intends — no
    /// embedder, an incompatible index, a vector search that failed. Surfaced,
    /// never quietly absorbed.
    pub degraded: bool,
    /// Why, when it is.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub degraded_because: Option<String>,
}

/// What the model was actually sent, after the runtime's own transformations.
///
/// ## Why this is separate from the manifest and linked to it
///
/// The manifest describes what the compiler *selected*. Between that and the
/// wire, the Node runtime compacts, prunes stale tool results, applies the chat
/// template and serialises. A manifest alone therefore describes an intention,
/// and the question an auditor asks is what the model received.
///
/// So this records the other end: the hash of the serialised request, its token
/// cost, and the manifest it claims to be a projection of. The two together are
/// what makes "the model saw exactly this" checkable rather than asserted.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FinalProjection {
    /// The manifest this is a projection of.
    pub manifest_hash: String,
    pub run_id: String,
    pub attempt_id: String,
    /// Which model round within the attempt. A turn makes several, and each is
    /// its own outgoing input.
    pub call_index: u32,
    /// SHA-256 of the serialised request body as it went to the server.
    pub input_hash: String,
    pub input_tokens: u32,
    /// Content hashes present in the outgoing input that the manifest does not
    /// account for.
    ///
    /// Empty is the required state. A non-empty list means the runtime
    /// introduced content the compiler did not authorise, which is the failure
    /// this record exists to make visible.
    #[serde(default)]
    pub unaccounted: Vec<String>,
    /// RFC 3339, UTC.
    pub created_at: String,
}

impl FinalProjection {
    /// Whether the outgoing input contained only content the manifest accounts
    /// for.
    pub fn is_faithful(&self) -> bool {
        self.unaccounted.is_empty()
    }
}

/// Everything a turn was built from.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ContextManifest {
    pub manifest_version: u32,
    /// The logical task. Stable across every attempt.
    pub run_id: String,
    /// The attempt that built this context.
    pub attempt_id: String,
    /// The transcript this turn belongs to. A resumption continues *this*
    /// conversation and *this* cell; it does not open a new one.
    pub conversation_id: String,
    pub message_id: String,
    /// The model the context was compiled for. A different window is a
    /// different projection, so this is part of what the manifest describes.
    pub model_id: String,
    /// The window the server was actually started with, not the trained one.
    pub served_window: u32,
    pub documents: Vec<DocumentBinding>,
    pub research: Option<ResearchBinding>,
    pub history: HistoryBinding,
    /// What the memory graph contributed. `None` on a version 1 manifest — a
    /// turn compiled before the graph existed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub graph: Option<GraphBinding>,
    /// How the window was divided and by what counting.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub budget: Option<BudgetRecord>,
    /// How retrieval was done, and whether it was narrower than intended.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retrieval: Option<RetrievalRecord>,
    /// What was left out, and why. Empty is the ordinary case.
    #[serde(default)]
    pub omissions: Vec<Omission>,
    /// SHA-256 of each block of content this turn carried, in order.
    ///
    /// What makes a final-projection check possible: the outgoing request can
    /// be searched for content whose hash is not in this list, and content the
    /// compiler did not authorise is exactly what that finds.
    #[serde(default)]
    pub content_hashes: Vec<String>,
    /// What each scope contributed. Empty on a manifest compiled before scopes
    /// were composed, and then left out of the seal so those still verify.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub scopes: Vec<ScopeRecord>,
    /// RFC 3339, UTC.
    pub created_at: String,
    /// Over every field above.
    pub manifest_hash: String,
}

impl ContextManifest {
    /// Builds a manifest and seals it with its own hash.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        run_id: impl Into<String>,
        attempt_id: impl Into<String>,
        conversation_id: impl Into<String>,
        message_id: impl Into<String>,
        model_id: impl Into<String>,
        served_window: u32,
        documents: Vec<DocumentBinding>,
        research: Option<ResearchBinding>,
        history: HistoryBinding,
    ) -> Self {
        let mut manifest = Self {
            manifest_version: MANIFEST_VERSION,
            run_id: run_id.into(),
            attempt_id: attempt_id.into(),
            conversation_id: conversation_id.into(),
            message_id: message_id.into(),
            model_id: model_id.into(),
            served_window,
            documents,
            research,
            history,
            // Filled in by the compiler, which is the only thing that knows
            // them. A manifest built without it describes a turn that carried
            // no graph context — which is a true statement about a turn
            // compiled by the Phase 2 path, and must not be confused with a
            // graph that was consulted and found nothing.
            graph: None,
            budget: None,
            retrieval: None,
            omissions: Vec::new(),
            content_hashes: Vec::new(),
            scopes: Vec::new(),
            created_at: chrono::Utc::now().to_rfc3339(),
            manifest_hash: String::new(),
        };
        manifest.manifest_hash = manifest.compute_hash();
        manifest
    }

    /// Adds what the compiler established, and re-seals.
    ///
    /// Separate from [`Self::new`] because the two are known at different
    /// moments: the conversation, documents and scope are settled when the turn
    /// is composed, and the graph selection, budget and retrieval mode are
    /// settled when the context is compiled against them. Re-sealing rather
    /// than mutating in place is what keeps `manifest_hash` a statement about
    /// the whole record.
    pub fn with_compilation(
        mut self,
        graph: Option<GraphBinding>,
        budget: Option<BudgetRecord>,
        retrieval: Option<RetrievalRecord>,
        omissions: Vec<Omission>,
        content_hashes: Vec<String>,
    ) -> Self {
        self.graph = graph;
        self.budget = budget;
        self.retrieval = retrieval;
        self.omissions = omissions;
        self.content_hashes = content_hashes;
        self.manifest_hash = self.compute_hash();
        self
    }

    /// Adds the per-scope record, and re-seals.
    pub fn with_scopes(mut self, scopes: Vec<ScopeRecord>) -> Self {
        self.scopes = scopes;
        self.manifest_hash = self.compute_hash();
        self
    }

    /// The hash of everything except the hash itself.
    ///
    /// Built from a canonical string rather than from serialised JSON, for the
    /// reason [`super::events::checkpoint::RunCheckpoint::compute_hash`] gives:
    /// field order is a property of the serialiser and this has to be stable
    /// across builds of it.
    pub fn compute_hash(&self) -> String {
        let documents = self
            .documents
            .iter()
            .map(|document| format!("{}:{}", document.sha256, document.pages))
            .collect::<Vec<_>>()
            .join(",");
        let research = self
            .research
            .as_ref()
            .map(|binding| {
                format!(
                    "{}|{}|{}|{}|{}|{}",
                    binding.notebook_id,
                    binding.manifest_run_id,
                    serde_json::to_string(&binding.selection).unwrap_or_default(),
                    binding.node_ids.join(","),
                    binding.assertion_ids.join(","),
                    binding.graph_revision.as_deref().unwrap_or(""),
                )
            })
            .unwrap_or_default();
        let history = format!(
            "{}|{}|{}|{}|{}",
            self.history.carried,
            self.history.dropped,
            self.history.tokens,
            self.history.pinned.join(","),
            self.history
                .omitted_pins
                .iter()
                .map(|omitted| format!("{}:{}", omitted.kind, omitted.pin))
                .collect::<Vec<_>>()
                .join(","),
        );
        // Serialised for the hash rather than field-by-field, because these are
        // optional records whose shape will grow: a canonical string per field
        // would have to be extended in step with every addition, and the field
        // somebody forgets is the field an editor can change undetected. The
        // *outer* format string stays canonical, which is what the field-order
        // argument above is actually about.
        let mut compiled = format!(
            "{}|{}|{}|{}|{}",
            serde_json::to_string(&self.graph).unwrap_or_default(),
            serde_json::to_string(&self.budget).unwrap_or_default(),
            serde_json::to_string(&self.retrieval).unwrap_or_default(),
            serde_json::to_string(&self.omissions).unwrap_or_default(),
            self.content_hashes.join(","),
        );
        // Only when present, so a manifest sealed before scopes existed still
        // verifies — the rule `subagents::result` follows for its new lists.
        if !self.scopes.is_empty() {
            compiled.push('|');
            compiled.push_str(&serde_json::to_string(&self.scopes).unwrap_or_default());
        }
        digest(&format!(
            "v{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}",
            self.manifest_version,
            self.run_id,
            self.attempt_id,
            self.conversation_id,
            self.message_id,
            self.model_id,
            self.served_window,
            digest(&documents),
            // Present-but-empty and absent hash differently, because a scope of
            // no sources and no scope at all are different things.
            digest(&format!(
                "{}|{research}",
                if self.research.is_some() { "some" } else { "none" }
            )),
            digest(&history),
            digest(&compiled),
            self.created_at,
        ))
    }

    /// Whether the stored hash still matches the stored body.
    pub fn is_intact(&self) -> bool {
        self.manifest_hash == self.compute_hash()
    }

    /// Whether this build understands the layout it was written under.
    pub fn is_known_version(&self) -> bool {
        READABLE_MANIFEST_VERSIONS.contains(&self.manifest_version)
    }

    /// Whether this manifest describes a turn compiled from the memory graph.
    ///
    /// False for a version 1 manifest, which is not a fault: it is a turn from
    /// before the compiler existed. Named so a caller can say "this run was
    /// built without graph context" rather than reporting an empty binding as
    /// though the graph had been consulted and found nothing.
    pub fn has_graph_binding(&self) -> bool {
        self.graph.is_some()
    }

    /// The documents this turn was given, as content addresses.
    pub fn document_hashes(&self) -> Vec<&str> {
        self.documents
            .iter()
            .map(|document| document.sha256.as_str())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn history() -> HistoryBinding {
        HistoryBinding {
            carried: 6,
            dropped: 2,
            tokens: 1_840,
            pinned: vec!["msg:u1".into()],
            omitted_pins: Vec::new(),
        }
    }

    fn manifest() -> ContextManifest {
        ContextManifest::new(
            "run-1",
            "attempt-1",
            "conv-1",
            "a-run-1",
            "model-a",
            32_768,
            vec![DocumentBinding {
                sha256: "ab12".into(),
                name: "vessel.pdf".into(),
                pages: 12,
            }],
            Some(ResearchBinding {
                notebook_id: "nb-1".into(),
                manifest_run_id: "run-1".into(),
                selection: SourceSelection::subset(vec!["ab12".into()]),
                node_ids: Vec::new(),
                assertion_ids: Vec::new(),
                graph_revision: Some("2026-09-16T11:02:03Z".into()),
            }),
            history(),
        )
    }

    #[test]
    fn a_fresh_manifest_is_intact_and_current() {
        let manifest = manifest();
        assert!(manifest.is_intact());
        assert!(manifest.is_known_version());
        assert_eq!(manifest.manifest_version, MANIFEST_VERSION);
    }

    #[test]
    fn it_round_trips_through_json() {
        let manifest = manifest();
        let text = serde_json::to_string(&manifest).expect("serialises");
        let back: ContextManifest = serde_json::from_str(&text).expect("parses");
        assert_eq!(manifest, back);
        assert!(back.is_intact());
    }

    /// Every field is in the hash. A field left out is a field an editor could
    /// change undetected, and this record is what a resumption trusts.
    #[test]
    fn editing_any_field_breaks_the_seal() {
        let mut edited = manifest();
        edited.conversation_id = "somebody-elses-conversation".into();
        assert!(!edited.is_intact());

        let mut swapped = manifest();
        swapped.documents[0].sha256 = "ffff".into();
        assert!(!swapped.is_intact());

        let mut widened = manifest();
        widened.research.as_mut().expect("has research").selection =
            SourceSelection::subset(vec!["ab12".into(), "cd34".into()]);
        assert!(
            !widened.is_intact(),
            "a source added to a frozen selection must not pass the seal"
        );

        let mut rebound = manifest();
        rebound.model_id = "model-b".into();
        assert!(!rebound.is_intact());

        let mut unpinned = manifest();
        unpinned.history.pinned.clear();
        assert!(!unpinned.is_intact());
    }

    /// A selection of nothing is a real answer and must not read as "all".
    /// The distinction the type exists for, checked at the manifest level:
    /// "the notebook is attached and no source may be read" and "there is no
    /// notebook" are different turns, and so are "none" and "all".
    #[test]
    fn an_empty_selection_is_distinct_from_no_research_at_all() {
        let empty = ContextManifest::new(
            "run-1",
            "attempt-1",
            "conv-1",
            "a-run-1",
            "model-a",
            32_768,
            Vec::new(),
            Some(ResearchBinding {
                notebook_id: "nb-1".into(),
                manifest_run_id: "run-1".into(),
                selection: SourceSelection::None,
                node_ids: Vec::new(),
                assertion_ids: Vec::new(),
                graph_revision: None,
            }),
            history(),
        );
        let none = ContextManifest::new(
            "run-1",
            "attempt-1",
            "conv-1",
            "a-run-1",
            "model-a",
            32_768,
            Vec::new(),
            None,
            history(),
        );
        assert!(empty.research.is_some());
        assert!(none.research.is_none());
        assert_ne!(
            empty.manifest_hash, none.manifest_hash,
            "a selection of none must not hash the same as no notebook at all"
        );

        // And "none" must not hash the same as "all", which is the reading a
        // resumption would take if the selection were flattened to a list.
        let mut all = empty.clone();
        all.research.as_mut().expect("has research").selection = SourceSelection::All;
        assert_ne!(
            all.compute_hash(),
            empty.manifest_hash,
            "a selection of none must not hash the same as every source"
        );
    }

    /// A manifest sealed under version 2 — no scopes, no safety reserve, no
    /// requested revision — must still verify after this build reads and
    /// re-serialises it. The new fields are left out of both the JSON and the
    /// seal when they are empty for exactly this reason.
    #[test]
    fn a_version_two_manifest_with_a_budget_still_verifies_after_the_fields_grew() {
        let mut old = manifest();
        old.manifest_version = 2;
        old.graph = Some(GraphBinding {
            graph_revision: 7,
            requested_revision: None,
            selected: vec![SelectedItem {
                item_id: "mi-1".into(),
                revision: 1,
                reason: "mandatory".into(),
                scope: None,
                precedence: None,
            }],
        });
        old.budget = Some(BudgetRecord {
            window: 8_192,
            reserved_tool_schemas: 1_000,
            reserved_output: 1_024,
            reserved_framing: 128,
            reserved_safety: 0,
            window_source: None,
            available: 6_040,
            spent: 12,
            counted_by: "estimate".into(),
            server_reported_tokens: None,
        });
        old.manifest_hash = old.compute_hash();
        let text = serde_json::to_string(&old).expect("serialises");
        assert!(!text.contains("reservedSafety"), "{text}");
        assert!(!text.contains("scopes"), "{text}");
        assert!(!text.contains("requestedRevision"), "{text}");
        let back: ContextManifest = serde_json::from_str(&text).expect("parses");
        assert!(back.is_intact(), "a version 2 manifest no longer verifies");
        assert!(back.is_known_version());
    }

    #[test]
    fn a_manifest_from_a_newer_build_is_not_treated_as_current() {
        let mut future = manifest();
        future.manifest_version = MANIFEST_VERSION + 1;
        future.manifest_hash = future.compute_hash();
        assert!(future.is_intact(), "the seal is over the version too");
        assert!(!future.is_known_version());
    }

    #[test]
    fn document_hashes_are_the_versions() {
        let manifest = manifest();
        assert_eq!(manifest.document_hashes(), vec!["ab12"]);
    }
}
