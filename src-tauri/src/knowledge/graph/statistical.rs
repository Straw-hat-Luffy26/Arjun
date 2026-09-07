//! The cheap pass: a browsable graph in seconds, with no model involved.
//!
//! ## Why this exists at all
//!
//! The obvious way to build an entity graph is to ask a language model to read
//! every passage and name the entities and relations in it. That works, and it
//! is the single most expensive thing in this problem space: Microsoft's
//! GraphRAG spends roughly three quarters of its token budget before anyone asks
//! a question, and indexing one corpus once cost them tens of thousands of
//! dollars in API calls. Their own answer — LazyGraphRAG — cuts that to a
//! thousandth by extracting noun phrases and their co-occurrences at index time
//! and deferring the model to query time.
//!
//! On an air-gapped workstation there is no bill, so the cost arrives as
//! wall-clock time on a quantised 7B model, which is worse: somebody is watching
//! it. So this pass runs first, over text that has already been read and cut,
//! and produces a real graph immediately. [`super::typing`] then improves it.
//!
//! ## What it does and does not claim
//!
//! Nodes here are **terms**, not typed entities. An edge means "these two terms
//! appear in the same passage", and its weight is the number of passages in
//! which that is true — a count, not a similarity, not a confidence, not a
//! normalised score. Nothing here infers that a supplier supplies anything.
//!
//! That restraint is the point. A graph that draws a plausible-looking
//! `supplies` edge it cannot support is the failure this repository's rules
//! exist to prevent, and an honest "appears with, in 4 passages" is both true
//! and useful.
//!
//! ## Determinism
//!
//! The same chunks produce the same nodes, the same edges and the same ids,
//! every time. Every collection is ordered before it leaves this module, ties
//! are broken lexicographically rather than by hash order, and nothing consults
//! a clock or a random number. A graph that redrew itself differently on each
//! build would be impossible to review.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::knowledge::chunking::Chunk;

/// Bumping this invalidates every stored build unit, forcing a rebuild.
///
/// It belongs to the *algorithm*, not to the release: change a threshold or a
/// scanner and the graph a previous build produced is no longer the graph this
/// code would produce, and leaving the two mixed in one table would make the
/// result unreproducible.
pub const EXTRACTOR_VERSION: u32 = 1;

/// A term seen in only one passage is not yet evidence of anything.
///
/// This is the single biggest noise filter over OCR'd text, where a mis-read
/// word appears once and never again.
const MIN_OCCURRENCES: u32 = 2;

/// A term in more than this share of a document's passages says nothing about
/// any particular passage.
///
/// "Pump" in a pump manual is in every chunk; it is the subject of the document,
/// not a distinguishing entity within it. Kept as a fraction rather than a count
/// because a 12-page memo and a 400-page standard have very different totals.
const MAX_DOCUMENT_FREQUENCY: f64 = 0.4;

/// Longest title-case run treated as one term.
const MAX_PHRASE_TOKENS: usize = 4;

/// Most terms considered from a single passage.
///
/// Edges are drawn between every surviving pair in a chunk, which is quadratic.
/// A pathological chunk — a parts table read as prose — could otherwise produce
/// hundreds of thousands of pairs from one passage and dominate the whole graph.
/// Terms are taken in order of first appearance so the cap is deterministic.
const MAX_TERMS_PER_CHUNK: usize = 48;

/// Words that are never a term on their own and never start or end a phrase.
///
/// Deliberately short. A long stopword list starts deleting real equipment names
/// ("Control", "Service", "Main"), and the frequency filter above removes the
/// common words this list would otherwise have to enumerate.
const STOPWORDS: &[&str] = &[
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "has", "have", "he",
    "her", "his", "if", "in", "is", "it", "its", "of", "on", "or", "she", "such", "than", "that",
    "the", "their", "then", "there", "these", "they", "this", "to", "was", "were", "which", "will",
    "with", "would", "shall", "may", "must", "not", "all", "any", "each", "when", "where", "while",
];

fn is_stopword(token: &str) -> bool {
    STOPWORDS.contains(&token)
}

/// One term the pass kept, with the passages it came from.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DraftNode {
    /// Casefolded join key. Two spellings of one term share this.
    pub normalised: String,
    /// The surface form to show — the spelling seen most often.
    pub label: String,
    /// Passages this term appeared in.
    pub occurrences: u32,
    /// Those passages, sorted. This is the provenance: every node can say
    /// exactly where it came from, and nothing here exists without it.
    pub chunk_ids: Vec<String>,
}

/// Two terms that share passages, and how many.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DraftEdge {
    /// Lexicographically the smaller of the pair. The edge is undirected, and
    /// ordering the ends is what stops one pair being stored as two edges.
    pub source: String,
    pub target: String,
    /// Passages containing both. A count of passages, nothing more.
    pub weight: u32,
    pub chunk_ids: Vec<String>,
}

/// What the pass looked at and what it discarded.
///
/// Reported rather than logged because the screen has to be able to say "912
/// candidates, 87 kept" — a graph with eleven nodes is either a small document
/// or a broken scanner, and the person looking at it cannot tell which without
/// these numbers.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExtractionStats {
    pub chunks: u32,
    pub candidates_found: u32,
    pub kept: u32,
    /// Dropped for appearing in fewer than [`MIN_OCCURRENCES`] passages.
    pub dropped_rare: u32,
    /// Dropped for appearing in more than [`MAX_DOCUMENT_FREQUENCY`] of them.
    pub dropped_generic: u32,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GraphDraft {
    pub nodes: Vec<DraftNode>,
    pub edges: Vec<DraftEdge>,
    pub stats: ExtractionStats,
}

/// Normalises one token: casefolded, stripped of surrounding punctuation.
///
/// Interior punctuation is kept, because it is load-bearing in exactly the terms
/// that matter most — `PV-2201` must not become `pv2201` and collide with a
/// different tag.
fn normalise_token(token: &str) -> String {
    token
        .trim_matches(|c: char| !c.is_alphanumeric())
        .to_lowercase()
}

/// Normalises a whole phrase to the join key this module produces.
///
/// `pub` because the typing pass has to map a label the model wrote back to a
/// term this pass found, and it must do that with *this* function rather than a
/// second one that agrees today. If the two ever disagreed, the typing pass
/// would silently fail to match terms that are plainly the same, and the graph
/// would look untypeable for no visible reason.
pub fn normalise_phrase(text: &str) -> String {
    text.split_whitespace()
        .map(normalise_token)
        .filter(|token| !token.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

/// Whether a token is a plant tag: `PV-2201`, `PT2201`, `FIC-101A`.
///
/// Matched by shape rather than by a dictionary, because the whole value of
/// these is that nobody has to enumerate them in advance. This is the
/// highest-signal term in a refinery document and the reason the pass is worth
/// running on scanned drawings at all — it is the same identity
/// [`crate::knowledge::multimodal::RegionKind::Symbol`] carries.
fn is_tag(token: &str) -> bool {
    let cleaned: &str = token.trim_matches(|c: char| !c.is_alphanumeric());
    let bytes = cleaned.as_bytes();
    if bytes.len() < 3 {
        return false;
    }

    let mut i = 0;
    let mut letters = 0;
    while i < bytes.len() && bytes[i].is_ascii_uppercase() {
        letters += 1;
        i += 1;
    }
    if !(2..=4).contains(&letters) {
        return false;
    }

    if i < bytes.len() && bytes[i] == b'-' {
        i += 1;
    }

    let mut digits = 0;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        digits += 1;
        i += 1;
    }
    if !(2..=5).contains(&digits) {
        return false;
    }

    // An optional single trailing letter — the `A` in `FIC-101A`.
    if i < bytes.len() && bytes[i].is_ascii_uppercase() {
        i += 1;
    }
    i == bytes.len()
}

/// Whether a token could begin or continue a title-case phrase.
fn is_title_case(token: &str) -> bool {
    let cleaned = token.trim_matches(|c: char| !c.is_alphanumeric());
    let mut chars = cleaned.chars();
    match chars.next() {
        Some(first) if first.is_uppercase() => cleaned.chars().count() > 1,
        _ => false,
    }
}

/// Splits text into tokens paired with whether each starts a sentence.
///
/// Sentence position matters for one rule only: a lone title-case word at the
/// start of a sentence is usually just a capitalised ordinary word, and treating
/// every one of them as an entity fills the graph with "However" and "Ensure".
fn tokens_with_position(text: &str) -> Vec<(String, bool)> {
    let mut out = Vec::new();
    let mut starts_sentence = true;
    for raw in text.split_whitespace() {
        let token = raw.to_string();
        let ends_sentence = raw.ends_with('.') || raw.ends_with('!') || raw.ends_with('?');
        if !token.trim_matches(|c: char| !c.is_alphanumeric()).is_empty() {
            out.push((token, starts_sentence));
            starts_sentence = ends_sentence;
        }
    }
    out
}

/// Every candidate term in one passage, in order of first appearance.
///
/// Returns `(normalised, surface)` pairs. Duplicates within the passage are
/// removed here, because the unit of counting is the passage: a term named five
/// times in one paragraph is one piece of evidence, not five.
fn candidates_in(text: &str, section_path: &[String]) -> Vec<(String, String)> {
    let mut seen: BTreeSet<String> = BTreeSet::new();
    let mut out: Vec<(String, String)> = Vec::new();

    fn push(
        normalised: String,
        surface: String,
        seen: &mut BTreeSet<String>,
        out: &mut Vec<(String, String)>,
    ) {
        if normalised.is_empty() || seen.contains(&normalised) {
            return;
        }
        seen.insert(normalised.clone());
        out.push((normalised, surface));
    }

    // The headings above this passage. Already curated by the chunker, so they
    // need none of the heuristics the body text does.
    for heading in section_path {
        let trimmed = heading.trim();
        if trimmed.is_empty() || trimmed.chars().count() < 3 {
            continue;
        }
        let normalised = trimmed
            .split_whitespace()
            .map(normalise_token)
            .filter(|t| !t.is_empty())
            .collect::<Vec<_>>()
            .join(" ");
        push(normalised, trimmed.to_string(), &mut seen, &mut out);
    }

    let tokens = tokens_with_position(text);

    // Tags first, and independently of case runs — a tag inside a sentence is
    // still a tag, and it must not be swallowed into a longer title-case phrase.
    for (token, _) in &tokens {
        if is_tag(token) {
            let surface = token
                .trim_matches(|c: char| !c.is_alphanumeric())
                .to_string();
            push(normalise_token(token), surface, &mut seen, &mut out);
        }
    }

    // Title-case runs.
    //
    // A run is split at any stopword *inside* it, not merely trimmed at its
    // ends. "Inspect The Pump Station" is one unbroken run of capitalised
    // tokens, and end-trimming alone would keep it whole — burying the entity
    // that matters inside a phrase nothing else will ever match. Splitting
    // yields "Inspect" (dropped, a lone sentence-opener) and "Pump Station".
    let mut i = 0;
    while i < tokens.len() {
        if !is_title_case(&tokens[i].0) || is_tag(&tokens[i].0) {
            i += 1;
            continue;
        }

        // The maximal run first; the length cap is applied to each segment
        // afterwards, so a long company name is not cut mid-phrase by where the
        // run happened to begin.
        let mut end = i;
        while end < tokens.len() && is_title_case(&tokens[end].0) && !is_tag(&tokens[end].0) {
            end += 1;
        }

        let mut segment_start = i;
        for boundary in i..=end {
            let breaks = boundary == end || is_stopword(&normalise_token(&tokens[boundary].0));
            if !breaks {
                continue;
            }

            let stop = boundary.min(segment_start + MAX_PHRASE_TOKENS);
            if stop > segment_start {
                // A single capitalised word opening a sentence is not evidence
                // of an entity. A phrase of two or more is, wherever it sits.
                let sentence_initial_single =
                    stop - segment_start == 1 && tokens[segment_start].1;
                if !sentence_initial_single {
                    let surface: Vec<String> = tokens[segment_start..stop]
                        .iter()
                        .map(|(t, _)| t.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
                        .collect();
                    let normalised: Vec<String> = tokens[segment_start..stop]
                        .iter()
                        .map(|(t, _)| normalise_token(t))
                        .collect();
                    if !normalised.iter().any(|t| t.is_empty()) {
                        push(normalised.join(" "), surface.join(" "), &mut seen, &mut out);
                    }
                }
            }
            segment_start = boundary + 1;
        }
        i = end.max(i + 1);
    }

    out.truncate(MAX_TERMS_PER_CHUNK);
    out
}

/// Builds a draft graph from one document's passages.
///
/// Every number in [`ExtractionStats`] is counted from the work actually done.
pub fn extract(chunks: &[Chunk]) -> GraphDraft {
    if chunks.is_empty() {
        return GraphDraft::default();
    }

    // normalised -> (surface -> count), and the chunks it appeared in.
    let mut surfaces: BTreeMap<String, BTreeMap<String, u32>> = BTreeMap::new();
    let mut appearances: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    // Per chunk, the terms found in it — kept so edges can be drawn without a
    // second pass over the text.
    let mut per_chunk: Vec<(String, Vec<String>)> = Vec::with_capacity(chunks.len());

    for chunk in chunks {
        let found = candidates_in(&chunk.text, &chunk.section_path);
        let mut normalised_here = Vec::with_capacity(found.len());
        for (normalised, surface) in found {
            *surfaces
                .entry(normalised.clone())
                .or_default()
                .entry(surface)
                .or_insert(0) += 1;
            appearances
                .entry(normalised.clone())
                .or_default()
                .insert(chunk.id.clone());
            normalised_here.push(normalised);
        }
        per_chunk.push((chunk.id.clone(), normalised_here));
    }

    let total_chunks = chunks.len() as u32;
    let candidates_found = appearances.len() as u32;
    let generic_ceiling = (f64::from(total_chunks) * MAX_DOCUMENT_FREQUENCY).ceil() as u32;

    let mut dropped_rare = 0u32;
    let mut dropped_generic = 0u32;
    let mut nodes: Vec<DraftNode> = Vec::new();
    let mut kept: BTreeSet<String> = BTreeSet::new();

    for (normalised, chunk_ids) in &appearances {
        let occurrences = chunk_ids.len() as u32;
        if occurrences < MIN_OCCURRENCES {
            dropped_rare += 1;
            continue;
        }
        // Only meaningful once there are enough passages for a share to mean
        // something. In a three-chunk document every real term is "in 40% of
        // it", and applying the ceiling there would empty the graph.
        if total_chunks >= 5 && occurrences > generic_ceiling {
            dropped_generic += 1;
            continue;
        }

        // The most frequent spelling wins; ties go to the lexicographically
        // smaller one, so the label does not depend on iteration order.
        let label = surfaces
            .get(normalised)
            .and_then(|forms| {
                forms
                    .iter()
                    .max_by(|a, b| a.1.cmp(b.1).then_with(|| b.0.cmp(a.0)))
                    .map(|(surface, _)| surface.clone())
            })
            .unwrap_or_else(|| normalised.clone());

        kept.insert(normalised.clone());
        nodes.push(DraftNode {
            normalised: normalised.clone(),
            label,
            occurrences,
            chunk_ids: chunk_ids.iter().cloned().collect(),
        });
    }

    // Edges: every surviving pair sharing a passage.
    let mut pairs: BTreeMap<(String, String), BTreeSet<String>> = BTreeMap::new();
    for (chunk_id, terms) in &per_chunk {
        let present: Vec<&String> = terms.iter().filter(|t| kept.contains(*t)).collect();
        for (i, left) in present.iter().enumerate() {
            for right in present.iter().skip(i + 1) {
                if left == right {
                    continue;
                }
                let key = if left < right {
                    ((*left).clone(), (*right).clone())
                } else {
                    ((*right).clone(), (*left).clone())
                };
                pairs.entry(key).or_default().insert(chunk_id.clone());
            }
        }
    }

    let edges: Vec<DraftEdge> = pairs
        .into_iter()
        .map(|((source, target), chunk_ids)| DraftEdge {
            source,
            target,
            weight: chunk_ids.len() as u32,
            chunk_ids: chunk_ids.into_iter().collect(),
        })
        .collect();

    // Heaviest first, then alphabetically — a stable order the UI can rely on.
    nodes.sort_by(|a, b| {
        b.occurrences
            .cmp(&a.occurrences)
            .then_with(|| a.normalised.cmp(&b.normalised))
    });

    GraphDraft {
        stats: ExtractionStats {
            chunks: total_chunks,
            candidates_found,
            kept: nodes.len() as u32,
            dropped_rare,
            dropped_generic,
        },
        nodes,
        edges,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge::chunking::ChunkKind;

    fn chunk(id: &str, text: &str) -> Chunk {
        Chunk {
            id: id.to_string(),
            document_sha256: "sha-doc".to_string(),
            ordinal: 0,
            text: text.to_string(),
            page: 1,
            section_path: Vec::new(),
            kind: ChunkKind::Prose,
            char_count: text.chars().count() as u32,
        }
    }

    #[test]
    fn recognises_plant_tags_by_shape() {
        assert!(is_tag("PV-2201"));
        assert!(is_tag("PT2201"));
        assert!(is_tag("FIC-101A"));
        assert!(is_tag("(PV-2201)"));

        assert!(!is_tag("PUMP"));
        assert!(!is_tag("2201"));
        assert!(!is_tag("A-1"));
        assert!(!is_tag("Pressure"));
    }

    #[test]
    fn a_tag_seen_in_two_passages_becomes_a_node() {
        let draft = extract(&[
            chunk("c1", "Valve PV-2201 was replaced during the outage."),
            chunk("c2", "Inspect PV-2201 before restart."),
        ]);

        let node = draft
            .nodes
            .iter()
            .find(|n| n.normalised == "pv-2201")
            .expect("the tag should be a node");
        assert_eq!(node.label, "PV-2201");
        assert_eq!(node.occurrences, 2);
        assert_eq!(node.chunk_ids, vec!["c1", "c2"]);
    }

    #[test]
    fn a_term_seen_once_is_dropped_and_counted() {
        let draft = extract(&[chunk("c1", "Valve PV-2201 was replaced.")]);

        assert!(draft.nodes.is_empty());
        assert!(draft.stats.dropped_rare > 0);
        assert_eq!(draft.stats.kept, 0);
    }

    #[test]
    fn an_edge_weight_is_the_number_of_shared_passages() {
        let draft = extract(&[
            chunk("c1", "Northern Valve Company supplied PV-2201 to the plant."),
            chunk("c2", "Northern Valve Company serviced PV-2201 last spring."),
            chunk("c3", "Northern Valve Company also supplies gaskets."),
        ]);

        let edge = draft
            .edges
            .iter()
            .find(|e| e.source == "northern valve company" && e.target == "pv-2201")
            .expect("the supplier and the tag share passages");
        // Two passages contain both, not three.
        assert_eq!(edge.weight, 2);
        assert_eq!(edge.chunk_ids, vec!["c1", "c2"]);
    }

    #[test]
    fn every_node_and_edge_carries_its_provenance() {
        let draft = extract(&[
            chunk("c1", "Northern Valve Company supplied PV-2201."),
            chunk("c2", "Northern Valve Company supplied PV-2201 again."),
        ]);

        assert!(!draft.nodes.is_empty());
        for node in &draft.nodes {
            assert!(!node.chunk_ids.is_empty(), "{} has no source", node.label);
        }
        for edge in &draft.edges {
            assert!(!edge.chunk_ids.is_empty());
            assert_eq!(edge.weight as usize, edge.chunk_ids.len());
        }
    }

    #[test]
    fn a_term_in_almost_every_passage_is_dropped_as_generic() {
        // Six passages, "Pump Station" in all of them: it is the subject of the
        // document, not an entity within it.
        let chunks: Vec<Chunk> = (0..6)
            .map(|i| {
                chunk(
                    &format!("c{i}"),
                    "The Pump Station reading was logged by Duty Engineer today.",
                )
            })
            .collect();
        let draft = extract(&chunks);

        assert!(
            !draft.nodes.iter().any(|n| n.normalised == "pump station"),
            "a term in every passage should be dropped"
        );
        assert!(draft.stats.dropped_generic > 0);
    }

    #[test]
    fn a_lone_capitalised_word_opening_a_sentence_is_not_an_entity() {
        let draft = extract(&[
            chunk("c1", "However the reading was low. However it recovered."),
            chunk("c2", "However the reading was low again."),
        ]);

        assert!(!draft.nodes.iter().any(|n| n.normalised == "however"));
    }

    #[test]
    fn stopwords_are_trimmed_from_the_ends_of_a_phrase() {
        let draft = extract(&[
            chunk("c1", "Inspect The Pump Station and log it."),
            chunk("c2", "Inspect The Pump Station once more."),
        ]);

        assert!(draft.nodes.iter().any(|n| n.normalised == "pump station"));
        assert!(!draft.nodes.iter().any(|n| n.normalised.starts_with("the ")));
    }

    #[test]
    fn the_label_is_the_most_frequent_spelling() {
        let draft = extract(&[
            chunk("c1", "PV-2201 and pv-2201 were logged."),
            chunk("c2", "PV-2201 was logged."),
            chunk("c3", "PV-2201 was logged again."),
        ]);

        let node = draft
            .nodes
            .iter()
            .find(|n| n.normalised == "pv-2201")
            .unwrap();
        assert_eq!(node.label, "PV-2201");
    }

    #[test]
    fn extraction_is_deterministic() {
        let chunks = vec![
            chunk("c1", "Northern Valve Company supplied PV-2201 to Unit Four."),
            chunk(
                "c2",
                "Unit Four reported PV-2201 drift; Northern Valve Company attended.",
            ),
            chunk("c3", "Northern Valve Company replaced PV-2201 in Unit Four."),
        ];

        let first = extract(&chunks);
        let second = extract(&chunks);
        assert_eq!(first, second);
    }

    #[test]
    fn an_empty_document_produces_an_empty_graph_not_a_guess() {
        let draft = extract(&[]);
        assert_eq!(draft, GraphDraft::default());
        assert_eq!(draft.stats.candidates_found, 0);
    }

    #[test]
    fn headings_contribute_terms() {
        let mut with_heading = chunk("c1", "The reading was nominal.");
        with_heading.section_path = vec!["Compressor Overhaul".to_string()];
        let mut second = chunk("c2", "The reading was high.");
        second.section_path = vec!["Compressor Overhaul".to_string()];

        let draft = extract(&[with_heading, second]);
        assert!(draft
            .nodes
            .iter()
            .any(|n| n.normalised == "compressor overhaul"));
    }

    #[test]
    fn one_pair_produces_one_edge_not_two() {
        let draft = extract(&[
            chunk("c1", "Northern Valve Company supplied PV-2201."),
            chunk("c2", "Northern Valve Company supplied PV-2201 again."),
        ]);

        let between: Vec<&DraftEdge> = draft
            .edges
            .iter()
            .filter(|e| {
                (e.source == "northern valve company" && e.target == "pv-2201")
                    || (e.source == "pv-2201" && e.target == "northern valve company")
            })
            .collect();
        assert_eq!(between.len(), 1);
        assert!(between[0].source < between[0].target);
    }
}
