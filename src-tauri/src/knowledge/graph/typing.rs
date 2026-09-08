//! The expensive pass: asking a local model to type terms and name relations.
//!
//! [`super::statistical`] produces terms and co-occurrences — true, cheap, and
//! untyped. This pass turns "Northern Valve Company and PV-2201 appear together
//! in three passages" into "supplier — supplies → equipment", which is what the
//! feature is actually for.
//!
//! ## Why every claim must cite and quote
//!
//! Models fabricate relationships. A study of six thousand LLM-extracted triples
//! found 2.4% anomalous and 0.65% with no grounding in the source at all. On a
//! graph a person will use to decide something, an invented `supplies` edge is
//! worse than no edge: it is indistinguishable from a real one and it is exactly
//! the sort of confident-looking artefact this repository has rules against.
//!
//! The literature's usual mitigation is two-model consensus, which doubles the
//! most expensive step. This module takes the cheaper and stricter route: every
//! proposed node and edge must name a chunk it came from **and quote a span from
//! it**, and [`verify`] drops anything whose quote is not actually in the chunk
//! it cited. That converts "trust the model" into a mechanical check that runs
//! in microseconds and cannot be argued with.
//!
//! Four gates, and a proposal must pass all of them:
//!
//! 1. The cited chunk must be one that was actually sent in this request.
//! 2. The quote must appear in that chunk.
//! 3. The type must be one of [`NODE_TYPES`] — the model may not invent a
//!    taxonomy.
//! 4. The term must be one [`super::statistical`] already found. This pass
//!    *types and labels*; it does not get to add entities. A term the cheap pass
//!    never saw has no provenance, and a node without provenance is the thing
//!    the whole design exists to prevent.
//!
//! ## What happens when the model is not there
//!
//! It fails, loudly, and the graph stays statistical. There is deliberately no
//! heuristic fallback that assigns types by keyword — that is precisely the
//! `MockProvider` defect documented at `memory_engine::extractor`, where a
//! substitute nobody chose wrote string-matched guesses into storage in the same
//! shape as real model output, and nothing downstream could tell them apart.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use super::statistical::normalise_phrase;

/// Bumping this invalidates stored typing units and forces a re-run.
///
/// Raised to 2 when `approval` and `project` joined [`NODE_TYPES`]. A document
/// typed under version 1 was typed against a vocabulary that could not express
/// either, so leaving those units marked complete would freeze the gap in place
/// for exactly the documents already in a notebook.
pub const TYPING_VERSION: u32 = 2;

/// The closed set of types a term may be given.
///
/// Closed because an open vocabulary produces "valve", "Valve", "valve
/// (equipment)" and "mechanical component" for the same thing across four
/// documents, and a graph cannot be filtered by a type nobody can enumerate.
pub const NODE_TYPES: &[&str] = &[
    "supplier",
    "equipment",
    "department",
    "contract",
    "risk",
    "person",
    "site",
    "document",
    // Added after an audit against the feature list found both missing. Their
    // absence was not neutral: the vocabulary is closed and enforced, so a
    // model correctly typing "the Kochi revamp" as a project had that entry
    // discarded as an unknown type, and the term ended up untyped rather than
    // typed differently. An approval note and the project it belongs to are
    // the two things this product is most often asked about, which is why they
    // earn a place in a list kept deliberately short.
    "approval",
    "project",
];

/// Shortest quote accepted as evidence.
///
/// A model that quotes "the" would satisfy a substring check trivially while
/// evidencing nothing. Deliberately low, because the rule that does the real
/// work is [`quote_carries_context`], not this floor: a short quote that
/// genuinely supports a claim ("PV-2201 failed") should not be thrown away for
/// being short.
const MIN_QUOTE_CHARS: usize = 8;

/// Whether a quote says anything beyond the label it is supporting.
///
/// Found by running the pass against a real model. Asked to type "PV-2201", it
/// cites the passage and quotes `"PV-2201"` — which is true, already known from
/// the statistical pass, and evidence of nothing. The claim under test is not
/// *that the term appears*; it is that the term is **equipment**, and a quote
/// that merely repeats the term supports that no better than silence.
///
/// So a quote must contain the label *and* something else. That is the whole
/// rule: it is what separates "the passage says PV-2201 is a control valve"
/// from "the passage says PV-2201".
fn quote_carries_context(quote: &str, label: &str) -> bool {
    let quote = comparable(quote);
    let label = comparable(label);
    quote != label && quote.len() > label.len()
}

/// What the model is asked to return. Field names match the schema it is given.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct Proposal {
    #[serde(default)]
    pub nodes: Vec<ProposedNode>,
    #[serde(default)]
    pub edges: Vec<ProposedEdge>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ProposedNode {
    pub label: String,
    #[serde(rename = "type")]
    pub node_type: String,
    pub evidence_chunk_id: String,
    pub quote: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ProposedEdge {
    pub source: String,
    pub target: String,
    pub relation: String,
    pub evidence_chunk_id: String,
    pub quote: String,
}

/// A node the gates accepted.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TypedNode {
    pub normalised: String,
    pub node_type: String,
    pub chunk_id: String,
    pub quote: String,
}

/// An edge the gates accepted.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TypedEdge {
    pub source: String,
    pub target: String,
    pub relation: String,
    pub chunk_id: String,
    pub quote: String,
}

/// What the model offered and what survived.
///
/// Reported, not logged. A pass that kept nine of two hundred proposals is
/// telling you something — a wrong model, a bad prompt, a document of scanned
/// noise — and a screen that showed only the nine would look like a small graph
/// rather than a failing extraction.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TypingStats {
    pub proposed_nodes: u32,
    pub proposed_edges: u32,
    pub kept_nodes: u32,
    pub kept_edges: u32,
    /// Rejected for naming a type outside [`NODE_TYPES`].
    pub dropped_unknown_type: u32,
    /// Rejected for naming a term the statistical pass never found.
    pub dropped_unknown_term: u32,
    /// Rejected for citing a chunk that was not in the request.
    pub dropped_uncited: u32,
    /// Rejected because the quote is not in the chunk it cited. The headline
    /// number: this is fabrication, caught.
    pub dropped_misquoted: u32,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Verdict {
    pub nodes: Vec<TypedNode>,
    pub edges: Vec<TypedEdge>,
    pub stats: TypingStats,
}

/// Collapses whitespace and case for comparison.
///
/// Case is folded because a model reproducing a real sentence with different
/// capitalisation has not fabricated anything, and rejecting it would discard
/// good evidence. Whitespace is collapsed because line breaks in the source
/// rarely survive a round trip through a prompt. Neither loosening lets an
/// invented sentence through: a fabricated quote does not match at any case or
/// spacing.
fn comparable(text: &str) -> String {
    text.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

/// Applies the four gates to what the model returned.
///
/// `known` maps a normalised term to its label — the terms
/// [`super::statistical`] found. `chunks` maps chunk id to the text that was
/// actually sent. Anything referring outside those two is dropped and counted.
pub fn verify(
    proposal: &Proposal,
    known: &BTreeMap<String, String>,
    chunks: &BTreeMap<String, String>,
) -> Verdict {
    let allowed: BTreeSet<&str> = NODE_TYPES.iter().copied().collect();
    let comparable_chunks: BTreeMap<String, String> = chunks
        .iter()
        .map(|(id, text)| (id.clone(), comparable(text)))
        .collect();

    let mut stats = TypingStats {
        proposed_nodes: proposal.nodes.len() as u32,
        proposed_edges: proposal.edges.len() as u32,
        ..Default::default()
    };

    // A quote is checked against the chunk the model *cited*, never against
    // every chunk. Searching them all would let a model attach a real sentence
    // from page 90 to a claim about page 3 and still pass.
    fn cites(
        comparable_chunks: &BTreeMap<String, String>,
        chunk_id: &str,
        quote: &str,
        stats: &mut TypingStats,
    ) -> bool {
        let Some(haystack) = comparable_chunks.get(chunk_id) else {
            stats.dropped_uncited += 1;
            return false;
        };
        let needle = comparable(quote);
        if needle.chars().count() < MIN_QUOTE_CHARS || !haystack.contains(&needle) {
            stats.dropped_misquoted += 1;
            return false;
        }
        true
    }

    let mut nodes: Vec<TypedNode> = Vec::new();
    let mut typed: BTreeSet<String> = BTreeSet::new();
    for proposed in &proposal.nodes {
        let node_type = proposed.node_type.trim().to_lowercase();
        if !allowed.contains(node_type.as_str()) {
            stats.dropped_unknown_type += 1;
            continue;
        }
        let normalised = normalise_phrase(&proposed.label);
        if !known.contains_key(&normalised) {
            stats.dropped_unknown_term += 1;
            continue;
        }
        if !cites(
            &comparable_chunks,
            &proposed.evidence_chunk_id,
            &proposed.quote,
            &mut stats,
        ) {
            continue;
        }
        // A quote that only repeats the label evidences the term's existence,
        // which was never in doubt — not its type, which is the claim.
        if !quote_carries_context(&proposed.quote, &proposed.label) {
            stats.dropped_misquoted += 1;
            continue;
        }
        // First type wins. A model that names one term twice with two types has
        // contradicted itself, and picking the later one silently would make the
        // result depend on array order.
        if !typed.insert(normalised.clone()) {
            continue;
        }
        nodes.push(TypedNode {
            normalised,
            node_type,
            chunk_id: proposed.evidence_chunk_id.clone(),
            quote: proposed.quote.trim().to_string(),
        });
    }
    stats.kept_nodes = nodes.len() as u32;

    let mut edges: Vec<TypedEdge> = Vec::new();
    let mut seen: BTreeSet<(String, String)> = BTreeSet::new();
    for proposed in &proposal.edges {
        let source = normalise_phrase(&proposed.source);
        let target = normalise_phrase(&proposed.target);
        if source == target || !known.contains_key(&source) || !known.contains_key(&target) {
            stats.dropped_unknown_term += 1;
            continue;
        }
        let relation = proposed.relation.trim().to_lowercase();
        if relation.is_empty() {
            stats.dropped_unknown_type += 1;
            continue;
        }
        if !cites(
            &comparable_chunks,
            &proposed.evidence_chunk_id,
            &proposed.quote,
            &mut stats,
        ) {
            continue;
        }
        // Stored the way the statistical pass stores an edge: ends ordered, so
        // one relationship is one row however the model happened to phrase it.
        let (a, b) = if source < target {
            (source, target)
        } else {
            (target, source)
        };
        if !seen.insert((a.clone(), b.clone())) {
            continue;
        }
        edges.push(TypedEdge {
            source: a,
            target: b,
            relation,
            chunk_id: proposed.evidence_chunk_id.clone(),
            quote: proposed.quote.trim().to_string(),
        });
    }
    stats.kept_edges = edges.len() as u32;

    Verdict {
        nodes,
        edges,
        stats,
    }
}

/// The JSON schema the server is asked to constrain generation to.
///
/// llama.cpp compiles this into a grammar and will only sample tokens that keep
/// the output on a valid path, so a malformed reply is not a case to handle. The
/// schema is not shown to the model, though — it constrains, it does not
/// instruct — which is why [`build_prompt`] spells out the same shape in words.
pub fn json_schema() -> serde_json::Value {
    let node_types: Vec<&str> = NODE_TYPES.to_vec();
    serde_json::json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["nodes", "edges"],
        "properties": {
            "nodes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": false,
                    "required": ["label", "type", "evidence_chunk_id", "quote"],
                    "properties": {
                        "label": { "type": "string" },
                        "type": { "type": "string", "enum": node_types },
                        "evidence_chunk_id": { "type": "string" },
                        "quote": { "type": "string" }
                    }
                }
            },
            "edges": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": false,
                    "required": ["source", "target", "relation", "evidence_chunk_id", "quote"],
                    "properties": {
                        "source": { "type": "string" },
                        "target": { "type": "string" },
                        "relation": { "type": "string" },
                        "evidence_chunk_id": { "type": "string" },
                        "quote": { "type": "string" }
                    }
                }
            }
        }
    })
}

/// Builds the instruction for one document.
///
/// The terms are given, not asked for: the model's job is to classify what the
/// cheap pass already found and to name relations between them. Asking it to
/// find entities as well would invite exactly the additions [`verify`] then
/// throws away, and would spend tokens producing them.
pub fn build_prompt(document_name: &str, terms: &[String], chunks: &[(String, String)]) -> String {
    let mut out = String::new();
    out.push_str(
        "You are labelling an index built from an organisation's own documents.\n\n\
         Below are passages, each with an id, and a list of terms already found in \
         them. For each term you can classify, give its type. Where two terms have \
         a relationship the passages actually state, give that relationship.\n\n\
         Rules:\n\
         - Use only the terms listed, for both ends of a relationship. Do not add \
         new ones. An entry naming anything not in the list is discarded.\n\
         - Every entry must quote, word for word, a span from the passage whose id \
         you cite. An entry whose quote is not in that passage is discarded.\n\
         - Quote a whole clause or sentence that shows *why* — not the term on its \
         own. \"PV-2201\" is not evidence that PV-2201 is equipment; \"the supply \
         of control valves to Unit Four\" is. An entry quoting only the term is \
         discarded.\n\
         - If a passage does not say what a term is, leave it out. Omitting is \
         correct; guessing is not.\n\
         - A relationship must be stated in the passage, not merely implied by two \
         terms appearing together.\n\n",
    );
    out.push_str(&format!("Types: {}\n\n", NODE_TYPES.join(", ")));
    out.push_str(&format!("Document: {document_name}\n\nTerms:\n"));
    for term in terms {
        out.push_str(&format!("- {term}\n"));
    }
    out.push_str("\nPassages:\n");
    for (id, text) in chunks {
        out.push_str(&format!("\n[{id}]\n{text}\n"));
    }
    out
}

/// How long one document's typing request may take.
///
/// Generous, because this is a quantised model on a workstation reading a few
/// thousand tokens under a grammar constraint, and minutes is normal. Bounded
/// all the same: a request that never returns would hold the pass open forever
/// and there would be nothing to report.
const REQUEST_TIMEOUT_SECONDS: u64 = 300;

/// Room for the reply. A document with sixty terms needs a long JSON array.
const MAX_REPLY_TOKENS: u32 = 2048;

/// Asks a local model to type one document's terms.
///
/// The impure half, kept apart from [`verify`] so the gates stay testable
/// without a server. Everything this returns is a *proposal*: nothing reaches
/// storage until `verify` has checked it.
///
/// Fails rather than degrading. There is no keyword fallback, because a
/// substitute nobody chose writing guesses into the same rows as model output is
/// the defect documented at `memory_engine::extractor` — undetectable afterwards
/// and indistinguishable from the real thing.
pub async fn request_typing(
    base_url: &str,
    model_id: &str,
    prompt: &str,
) -> Result<Proposal, String> {
    // Loopback only, checked the same way the rest of the serving layer checks
    // it. This pass sends passages from the organisation's own documents, and
    // an endpoint that is not local is an exfiltration path.
    crate::serving::probe::check_loopback(base_url)
        .map_err(|outcome| outcome.explain(base_url))?;

    // Built by the serving layer, not here. It is the only place that
    // constructs a client for a local model server, and the `.no_proxy()` it
    // applies is what stops an inherited `HTTP_PROXY` from turning this request
    // — which carries passages from the organisation's own documents — into one
    // that leaves the machine.
    let client = crate::serving::probe::loopback_client(std::time::Duration::from_secs(
        REQUEST_TIMEOUT_SECONDS,
    ))
    .map_err(|error| format!("the HTTP client could not be built: {error}"))?;

    let body = serde_json::json!({
        "model": model_id,
        "messages": [{ "role": "user", "content": prompt }],
        // Zero, so a second run over an unchanged document produces the same
        // types. A graph that reclassified its own terms on every rebuild would
        // be impossible to review.
        "temperature": 0,
        "max_tokens": MAX_REPLY_TOKENS,
        "response_format": {
            "type": "json_schema",
            "json_schema": { "name": "graph_typing", "strict": true, "schema": json_schema() }
        }
    });

    let url = format!("{}/chat/completions", base_url.trim_end_matches('/'));
    let response = client
        .post(&url)
        .json(&body)
        .send()
        .await
        .map_err(|error| format!("{url} could not be reached: {error}"))?;

    let status = response.status();
    if !status.is_success() {
        let detail = response.text().await.unwrap_or_default();
        let detail: String = detail.chars().take(300).collect();
        return Err(format!("{url} answered {status}: {detail}"));
    }

    #[derive(Deserialize)]
    struct Message {
        content: Option<String>,
    }
    #[derive(Deserialize)]
    struct Choice {
        message: Message,
    }
    #[derive(Deserialize)]
    struct Reply {
        choices: Vec<Choice>,
    }

    let reply: Reply = response
        .json()
        .await
        .map_err(|error| format!("{url} answered in an unexpected shape: {error}"))?;
    let content = reply
        .choices
        .first()
        .and_then(|choice| choice.message.content.clone())
        .ok_or_else(|| format!("{url} returned no content"))?;

    // The schema constrains the sampler, so malformed JSON should be
    // impossible — but "should be impossible" is not a reason to unwrap. A
    // server without schema support silently ignores the field and returns
    // prose, and that must surface as a clear failure rather than a panic.
    serde_json::from_str::<Proposal>(&content).map_err(|error| {
        format!(
            "the model's reply was not the expected JSON ({error}). The server may not support \
             constrained output."
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn known() -> BTreeMap<String, String> {
        [
            ("northern valve company", "Northern Valve Company"),
            ("pv-2201", "PV-2201"),
            ("unit four", "Unit Four"),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
    }

    fn chunks() -> BTreeMap<String, String> {
        [(
            "c1",
            "Northern Valve Company supplied PV-2201 to Unit Four in March.",
        )]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
    }

    fn node(label: &str, node_type: &str, chunk: &str, quote: &str) -> ProposedNode {
        ProposedNode {
            label: label.into(),
            node_type: node_type.into(),
            evidence_chunk_id: chunk.into(),
            quote: quote.into(),
        }
    }

    fn edge(source: &str, target: &str, relation: &str, chunk: &str, quote: &str) -> ProposedEdge {
        ProposedEdge {
            source: source.into(),
            target: target.into(),
            relation: relation.into(),
            evidence_chunk_id: chunk.into(),
            quote: quote.into(),
        }
    }

    #[test]
    fn a_grounded_node_is_kept() {
        let proposal = Proposal {
            nodes: vec![node(
                "Northern Valve Company",
                "supplier",
                "c1",
                "Northern Valve Company supplied PV-2201",
            )],
            edges: vec![],
        };
        let verdict = verify(&proposal, &known(), &chunks());

        assert_eq!(verdict.nodes.len(), 1);
        assert_eq!(verdict.nodes[0].normalised, "northern valve company");
        assert_eq!(verdict.nodes[0].node_type, "supplier");
        assert_eq!(verdict.stats.kept_nodes, 1);
    }

    // ── The gate that matters most ──────────────────────────────────────────

    #[test]
    fn a_node_whose_quote_is_not_in_the_cited_chunk_is_dropped() {
        let proposal = Proposal {
            nodes: vec![node(
                "Northern Valve Company",
                "supplier",
                "c1",
                "Northern Valve Company holds the maintenance contract",
            )],
            edges: vec![],
        };
        let verdict = verify(&proposal, &known(), &chunks());

        assert!(verdict.nodes.is_empty(), "a fabricated quote must not pass");
        assert_eq!(verdict.stats.dropped_misquoted, 1);
    }

    #[test]
    fn an_edge_whose_quote_is_not_in_the_cited_chunk_is_dropped() {
        let proposal = Proposal {
            nodes: vec![],
            edges: vec![edge(
                "Northern Valve Company",
                "PV-2201",
                "maintains",
                "c1",
                "Northern Valve Company maintains PV-2201 quarterly",
            )],
        };
        let verdict = verify(&proposal, &known(), &chunks());

        assert!(verdict.edges.is_empty());
        assert_eq!(verdict.stats.dropped_misquoted, 1);
    }

    #[test]
    fn a_real_quote_from_a_chunk_that_was_not_cited_does_not_rescue_a_claim() {
        // The sentence exists, but in c1, and the model cited c9. Searching
        // every chunk would let a model attach page 90's sentence to a claim
        // about page 3.
        let proposal = Proposal {
            nodes: vec![node(
                "PV-2201",
                "equipment",
                "c9",
                "Northern Valve Company supplied PV-2201",
            )],
            edges: vec![],
        };
        let verdict = verify(&proposal, &known(), &chunks());

        assert!(verdict.nodes.is_empty());
        assert_eq!(verdict.stats.dropped_uncited, 1);
    }

    #[test]
    fn a_trivially_short_quote_is_not_evidence() {
        let proposal = Proposal {
            nodes: vec![node("PV-2201", "equipment", "c1", "to")],
            edges: vec![],
        };
        let verdict = verify(&proposal, &known(), &chunks());

        assert!(verdict.nodes.is_empty());
        assert_eq!(verdict.stats.dropped_misquoted, 1);
    }

    // ── What a real model actually returned ─────────────────────────────────
    //
    // Nemotron3-Nano-4B, asked to type this exact passage through llama-server
    // with the schema below, quoted the bare term for two of three nodes and
    // invented both edge targets. These pin that behaviour.

    #[test]
    fn a_quote_that_only_repeats_the_label_is_not_evidence_of_its_type() {
        // The observed failure. "PV-2201" is in the passage, and the statistical
        // pass already established that. It says nothing about *equipment*.
        let proposal = Proposal {
            nodes: vec![node("PV-2201", "equipment", "c1", "PV-2201")],
            edges: vec![],
        };
        let verdict = verify(&proposal, &known(), &chunks());

        assert!(verdict.nodes.is_empty());
        assert_eq!(verdict.stats.dropped_misquoted, 1);
    }

    #[test]
    fn a_long_label_quoted_bare_is_still_not_evidence() {
        // "Northern Valve Company" is 22 characters, so any length floor would
        // wave it through. It is still only the label.
        let proposal = Proposal {
            nodes: vec![node(
                "Northern Valve Company",
                "supplier",
                "c1",
                "Northern Valve Company",
            )],
            edges: vec![],
        };
        let verdict = verify(&proposal, &known(), &chunks());

        assert!(verdict.nodes.is_empty());
        assert_eq!(verdict.stats.dropped_misquoted, 1);
    }

    #[test]
    fn a_quote_carrying_context_around_the_label_is_kept() {
        let proposal = Proposal {
            nodes: vec![node(
                "PV-2201",
                "equipment",
                "c1",
                "supplied PV-2201 to Unit Four",
            )],
            edges: vec![],
        };
        let verdict = verify(&proposal, &known(), &chunks());

        assert_eq!(verdict.nodes.len(), 1);
        assert_eq!(verdict.nodes[0].node_type, "equipment");
    }

    /// The contract passage the observed run was given, verbatim.
    fn contract_chunk() -> BTreeMap<String, String> {
        [(
            "c1",
            "Supply Agreement\n\nThis agreement is between the operator and Northern \
             Valve Company for the supply of control valves to Unit Four. Northern \
             Valve Company shall hold spares for PV-2201 and equivalent assemblies \
             for the term of this agreement.",
        )]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
    }

    #[test]
    fn the_reply_the_improved_prompt_produced_passes_every_gate() {
        // Verbatim from Nemotron3-Nano-4B through llama-server, after the prompt
        // was changed to demand a clause rather than the bare term. Pinned
        // because it is the evidence that the gate is strict without being
        // unusable: five claims offered, five kept, with real relations.
        let proposal = Proposal {
            nodes: vec![
                node(
                    "Northern Valve Company",
                    "supplier",
                    "c1",
                    "Northern Valve Company shall hold spares for PV-2201 and equivalent \
                     assemblies for the term of this agreement.",
                ),
                node("PV-2201", "equipment", "c1", "spares for PV-2201"),
                node(
                    "Unit Four",
                    "site",
                    "c1",
                    "the supply of control valves to Unit Four",
                ),
            ],
            edges: vec![
                edge(
                    "Northern Valve Company",
                    "Unit Four",
                    "supplies",
                    "c1",
                    "the supply of control valves to Unit Four",
                ),
                edge(
                    "Northern Valve Company",
                    "PV-2201",
                    "provides spares for",
                    "c1",
                    "spares for PV-2201",
                ),
            ],
        };
        let verdict = verify(&proposal, &known(), &contract_chunk());

        assert_eq!(verdict.stats.kept_nodes, 3, "all three types are evidenced");
        assert_eq!(verdict.stats.kept_edges, 2, "both relations are evidenced");
        assert_eq!(verdict.stats.dropped_misquoted, 0);
        assert_eq!(verdict.stats.dropped_unknown_term, 0);

        let valve = verdict
            .nodes
            .iter()
            .find(|n| n.normalised == "pv-2201")
            .unwrap();
        assert_eq!(valve.node_type, "equipment");
        assert!(verdict.edges.iter().any(|e| e.relation == "supplies"));
    }

    #[test]
    fn the_edge_targets_a_real_model_invented_are_both_rejected() {
        // Verbatim from the run: the model was given three terms and named
        // "operator" and "supply of control valves" as edge ends anyway.
        let proposal = Proposal {
            nodes: vec![],
            edges: vec![
                edge(
                    "Northern Valve Company",
                    "operator",
                    "operator and Northern Valve Company",
                    "c1",
                    "Northern Valve Company supplied PV-2201",
                ),
                edge(
                    "Unit Four",
                    "supply of control valves",
                    "supply of control valves to Unit Four",
                    "c1",
                    "Northern Valve Company supplied PV-2201",
                ),
            ],
        };
        let verdict = verify(&proposal, &known(), &chunks());

        assert!(verdict.edges.is_empty(), "invented ends must not reach the graph");
        assert_eq!(verdict.stats.dropped_unknown_term, 2);
    }

    #[test]
    fn a_quote_differing_only_in_case_and_spacing_is_accepted() {
        let proposal = Proposal {
            nodes: vec![node(
                "PV-2201",
                "equipment",
                "c1",
                "northern valve   company SUPPLIED pv-2201",
            )],
            edges: vec![],
        };
        let verdict = verify(&proposal, &known(), &chunks());

        assert_eq!(
            verdict.nodes.len(),
            1,
            "a real sentence must not be rejected"
        );
        assert_eq!(verdict.stats.dropped_misquoted, 0);
    }

    // ── The other three gates ───────────────────────────────────────────────

    #[test]
    fn an_invented_type_is_dropped() {
        let proposal = Proposal {
            nodes: vec![node(
                "PV-2201",
                "mechanical component",
                "c1",
                "Northern Valve Company supplied PV-2201",
            )],
            edges: vec![],
        };
        let verdict = verify(&proposal, &known(), &chunks());

        assert!(verdict.nodes.is_empty());
        assert_eq!(verdict.stats.dropped_unknown_type, 1);
    }

    #[test]
    fn a_term_the_statistical_pass_never_found_is_dropped() {
        // This pass types and labels. It does not get to add entities: a node
        // the cheap pass never saw has no provenance behind it.
        let proposal = Proposal {
            nodes: vec![node(
                "Southern Gasket Ltd",
                "supplier",
                "c1",
                "Northern Valve Company supplied PV-2201",
            )],
            edges: vec![],
        };
        let verdict = verify(&proposal, &known(), &chunks());

        assert!(verdict.nodes.is_empty());
        assert_eq!(verdict.stats.dropped_unknown_term, 1);
    }

    #[test]
    fn an_edge_naming_an_unknown_term_is_dropped() {
        let proposal = Proposal {
            nodes: vec![],
            edges: vec![edge(
                "Northern Valve Company",
                "Southern Gasket Ltd",
                "partners with",
                "c1",
                "Northern Valve Company supplied PV-2201",
            )],
        };
        let verdict = verify(&proposal, &known(), &chunks());

        assert!(verdict.edges.is_empty());
        assert_eq!(verdict.stats.dropped_unknown_term, 1);
    }

    // ── Shape and bookkeeping ───────────────────────────────────────────────

    #[test]
    fn a_grounded_edge_is_kept_with_its_ends_ordered() {
        let proposal = Proposal {
            nodes: vec![],
            edges: vec![edge(
                "PV-2201",
                "Northern Valve Company",
                "SUPPLIED BY",
                "c1",
                "Northern Valve Company supplied PV-2201",
            )],
        };
        let verdict = verify(&proposal, &known(), &chunks());

        assert_eq!(verdict.edges.len(), 1);
        // Ordered, so one relationship is one row however it was phrased.
        assert!(verdict.edges[0].source < verdict.edges[0].target);
        assert_eq!(verdict.edges[0].relation, "supplied by");
    }

    #[test]
    fn a_term_typed_twice_keeps_the_first_type_not_the_last() {
        let proposal = Proposal {
            nodes: vec![
                node(
                    "PV-2201",
                    "equipment",
                    "c1",
                    "Northern Valve Company supplied PV-2201",
                ),
                node(
                    "PV-2201",
                    "contract",
                    "c1",
                    "Northern Valve Company supplied PV-2201",
                ),
            ],
            edges: vec![],
        };
        let verdict = verify(&proposal, &known(), &chunks());

        assert_eq!(verdict.nodes.len(), 1);
        assert_eq!(verdict.nodes[0].node_type, "equipment");
    }

    #[test]
    fn a_self_edge_is_dropped() {
        let proposal = Proposal {
            nodes: vec![],
            edges: vec![edge(
                "PV-2201",
                "PV-2201",
                "relates to",
                "c1",
                "Northern Valve Company supplied PV-2201",
            )],
        };
        assert!(verify(&proposal, &known(), &chunks()).edges.is_empty());
    }

    #[test]
    fn the_counts_add_up_so_a_screen_can_be_honest_about_them() {
        let proposal = Proposal {
            nodes: vec![
                node("PV-2201", "equipment", "c1", "supplied PV-2201 to Unit Four"),
                node("PV-2201", "nonsense", "c1", "supplied PV-2201 to Unit Four"),
                node(
                    "Ghost Ltd",
                    "supplier",
                    "c1",
                    "supplied PV-2201 to Unit Four",
                ),
                node("Unit Four", "site", "c1", "a sentence that is not present"),
            ],
            edges: vec![],
        };
        let verdict = verify(&proposal, &known(), &chunks());

        assert_eq!(verdict.stats.proposed_nodes, 4);
        assert_eq!(verdict.stats.kept_nodes, 1);
        assert_eq!(verdict.stats.dropped_unknown_type, 1);
        assert_eq!(verdict.stats.dropped_unknown_term, 1);
        assert_eq!(verdict.stats.dropped_misquoted, 1);
    }

    #[test]
    fn an_empty_proposal_produces_an_empty_verdict() {
        let verdict = verify(&Proposal::default(), &known(), &chunks());
        assert_eq!(verdict, Verdict::default());
    }

    #[test]
    fn the_schema_closes_the_type_vocabulary_and_forbids_extra_fields() {
        let schema = json_schema();
        let node = &schema["properties"]["nodes"]["items"];
        assert_eq!(node["additionalProperties"], serde_json::json!(false));
        let types = node["properties"]["type"]["enum"].as_array().unwrap();
        assert_eq!(types.len(), NODE_TYPES.len());
        assert!(types.iter().any(|t| t == "supplier"));
    }

    #[test]
    fn the_prompt_lists_the_terms_and_the_passage_ids_it_expects_cited() {
        let prompt = build_prompt(
            "pump.pdf",
            &["PV-2201".to_string()],
            &[("c1".to_string(), "PV-2201 was replaced.".to_string())],
        );
        assert!(prompt.contains("pump.pdf"));
        assert!(prompt.contains("- PV-2201"));
        assert!(prompt.contains("[c1]"));
        // The schema constrains but does not instruct, so the prompt has to say
        // the quoting rule in words or the model never learns it.
        assert!(prompt.contains("word for word"));
    }
}
