//! Combining two ways of finding a passage into one ranking.
//!
//! Keyword search and vector search fail differently, which is exactly why the
//! plan calls for both. Keyword search is unbeatable on an exact token — `PV-2201`,
//! `9.0 mm`, a drawing number — and blind to a question phrased in different
//! words. Vector search is the reverse. A refinery asks both kinds of question,
//! often in the same sentence.
//!
//! ## The order of operations is the security property
//!
//! 1. **Authorise.** Both halves read through the index's one authorisation
//!    clause ([`KnowledgeIndex::search_lexical`], [`KnowledgeIndex::search_dense`]),
//!    so a passage the reader may not see is never fetched, never scored and
//!    never counted.
//! 2. **Score** within that set: bm25 for the keyword half, cosine for the dense
//!    half, each in its own units.
//! 3. **Fuse** by rank, never by score — see [`reciprocal_rank_fusion`].
//! 4. **Deduplicate**: one passage found by both halves is one hit; two passages
//!    with identical text (the same SOP copied into two folders) are one hit
//!    naming both.
//! 5. **Cut** to the limit, and say what was not covered.
//!
//! Reranking ([`LexicalProximityReranker`]) and graph expansion
//! ([`crate::knowledge::graph::runtime_store::MemoryGraph::neighbourhood`]) work
//! on what this returned, so they cannot reach past step 1 either.
//!
//! ## Reciprocal rank fusion
//!
//! BM25 relevance and cosine similarity cannot be added, averaged, or sensibly
//! weighted without tuning that would not survive a change of corpus. So only
//! the *ordering* is used — `1 / (k + rank)`, summed across searches. A passage
//! both searches rank highly beats one only a single search loved; with one
//! search available, the fused order *is* that search's order.
//!
//! ## Scores are labelled for what they are
//!
//! Every hit carries each score it has under its own name ([`HitScores`]) and a
//! sentence saying what that number is for. A fused score is an ordering
//! device, not a confidence, and presenting it as one is how a caller comes to
//! threshold on a number that means nothing outside one query.

use std::collections::{BTreeMap, BTreeSet};

use anyhow::Result;
use serde::{Deserialize, Serialize};

use super::embedding::{embed_query, Embedder};
use super::index::{
    DenseHit, EmbeddingCoverage, KnowledgeIndex, LexicalHit, LexicalPass, SearchScope,
};
use super::SearchResult;
use crate::identity::Session;

/// Damping constant in the fusion formula.
///
/// 60 is the value from the original published comparison and has been the
/// default everywhere since. With `k = 60`, rank 1 scores 1/61 and rank 2 scores
/// 1/62, so a passage needs agreement across searches to rise, not just one
/// enthusiastic vote.
const RRF_K: f64 = 60.0;

/// Candidates each half contributes before fusion.
pub const CANDIDATES_PER_HALF: usize = 24;

/// Most hits one search returns. Matches what `knowledge.search_authorized`
/// clamps to, plus two: a hybrid search is asked for when a literal one was
/// not enough.
pub const MAX_RESULTS: usize = 8;

/// Longest excerpt a hit carries.
///
/// Short on purpose: the excerpt is for deciding whether to read the passage,
/// and a model that needs more asks `knowledge.load_evidence_region` for the
/// page. Eight full passages would be most of a small window.
pub const EXCERPT_CHARS: usize = 480;

/// How many candidates the reranker will look at.
pub const RERANK_LIMIT: usize = 24;

/// Fuses several rankings of the same items into one.
///
/// Each input is an ordered list of ids, best first. Returns ids with their
/// fused score, best first. Ties keep the order of the first ranking that
/// contained them, so the output is deterministic rather than depending on hash
/// iteration order — a citation that moves between identical runs is a bug
/// report waiting to happen.
pub fn reciprocal_rank_fusion(rankings: &[Vec<String>]) -> Vec<(String, f64)> {
    let mut order: Vec<String> = Vec::new();
    let mut scores: BTreeMap<String, f64> = BTreeMap::new();

    for ranking in rankings {
        for (position, id) in ranking.iter().enumerate() {
            let contribution = 1.0 / (RRF_K + (position + 1) as f64);
            let entry = scores.entry(id.clone()).or_insert_with(|| {
                order.push(id.clone());
                0.0
            });
            *entry += contribution;
        }
    }

    let mut fused: Vec<(usize, String, f64)> = order
        .into_iter()
        .enumerate()
        .map(|(first_seen, id)| {
            let score = scores[&id];
            (first_seen, id, score)
        })
        .collect();
    fused.sort_by(|a, b| b.2.total_cmp(&a.2).then(a.0.cmp(&b.0)));
    fused.into_iter().map(|(_, id, score)| (id, score)).collect()
}

/// Which halves found a hit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum HitMethod {
    Lexical,
    Dense,
    LexicalAndDense,
}

impl HitMethod {
    pub fn label(self) -> &'static str {
        match self {
            HitMethod::Lexical => "keyword",
            HitMethod::Dense => "semantic",
            HitMethod::LexicalAndDense => "keyword+semantic",
        }
    }
}

/// Every score a hit has, each under its own name.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HitScores {
    /// FTS5 bm25 for this query: lower is better, comparable only within one
    /// pass of one query.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub keyword_bm25: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub keyword_pass: Option<LexicalPass>,
    /// Cosine similarity of the closest window to the query, in this space.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub semantic_cosine: Option<f64>,
    /// Reciprocal-rank fusion. For ordering only.
    pub fused_rank: f64,
    /// The reranker's score, when one ran. For ordering only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rerank: Option<f64>,
}

impl HitScores {
    /// One line naming each score and what it is for.
    pub fn describe(&self) -> String {
        let mut parts = Vec::new();
        if let Some(bm25) = self.keyword_bm25 {
            let pass = match self.keyword_pass {
                Some(LexicalPass::AnyTerm) => "some terms",
                _ => "all terms",
            };
            parts.push(format!("keyword bm25 {bm25:.2} ({pass}; lower is better, this query only)"));
        }
        if let Some(cosine) = self.semantic_cosine {
            parts.push(format!("semantic cosine {cosine:.3} (closeness to the question)"));
        }
        parts.push(format!("fused rank {:.4} (ordering only)", self.fused_rank));
        if let Some(rerank) = self.rerank {
            parts.push(format!("rerank {rerank:.2} (ordering only)"));
        }
        parts.join("; ")
    }
}

/// Where a hit's source stands in its version history.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HitSource {
    pub collection_id: String,
    pub relative_path: String,
    pub version: u32,
    pub status: String,
    /// True when this search was pinned and the version has been superseded
    /// since: read because the job was queued against it, and flagged.
    pub superseded_since_pin: bool,
}

/// One passage the search found.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EvidenceHit {
    /// Resolves back to exactly this passage through
    /// [`KnowledgeIndex::passage`], under the reader's own clearance.
    pub handle: String,
    pub passage: SearchResult,
    pub excerpt: String,
    pub method: HitMethod,
    pub scores: HitScores,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<HitSource>,
    /// Other passages with identical text, merged into this one.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub duplicates: Vec<String>,
}

impl EvidenceHit {
    pub fn handle_for(chunk_id: &str) -> String {
        format!("ev:{chunk_id}")
    }
}

/// What the search covered, and what it could not.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Coverage {
    /// `hybrid` when the dense half contributed, `lexical` otherwise.
    pub mode: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub degraded_because: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub space_key: Option<String>,
    /// Over the passages this reader may see in this scope, never the index.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedding: Option<EmbeddingCoverage>,
    pub keyword_candidates: usize,
    pub semantic_candidates: usize,
    /// Nothing the reader may see matched. A finding, not a failure.
    pub no_answer: bool,
    /// Sentences naming what this result does not cover.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub partial: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pinned_revision: Option<i64>,
}

/// A hybrid search's answer.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HybridResponse {
    pub query: String,
    pub hits: Vec<EvidenceHit>,
    pub coverage: Coverage,
}

/// One hybrid search.
#[derive(Debug, Clone, Default)]
pub struct HybridRequest {
    pub query: String,
    pub limit: usize,
    pub scope: SearchScope,
}

/// Runs both halves over one index.
pub struct HybridRetriever<'a> {
    pub index: &'a KnowledgeIndex,
    /// A *qualified* embedder, or `None` with the reason in `dense_unavailable`.
    pub embedder: Option<&'a dyn Embedder>,
    pub dense_unavailable: Option<String>,
    /// The semantic floor measured when this embedder was qualified — see
    /// `QualificationRecord::dense_floor`. A semantic candidate below it casts
    /// no vote; if the keyword half found it too, it stays as a keyword hit.
    pub dense_floor: Option<f64>,
}

impl<'a> HybridRetriever<'a> {
    pub fn lexical_only(index: &'a KnowledgeIndex, why: impl Into<String>) -> Self {
        Self {
            index,
            embedder: None,
            dense_unavailable: Some(why.into()),
            dense_floor: None,
        }
    }

    pub async fn search(&self, session: &Session, request: &HybridRequest) -> Result<HybridResponse> {
        let limit = request.limit.clamp(1, MAX_RESULTS);
        let query = request.query.trim().to_string();
        let mut coverage = Coverage {
            pinned_revision: request.scope.pinned_revision,
            ..Default::default()
        };
        if query.is_empty() {
            coverage.mode = "lexical".into();
            coverage.no_answer = true;
            coverage.partial.push("the question was empty, so nothing was searched".into());
            return Ok(HybridResponse {
                query,
                hits: Vec::new(),
                coverage,
            });
        }

        // 1–2. Both halves, each already inside the reader's clearance.
        let lexical: Vec<LexicalHit> =
            self.index
                .search_lexical(session, &query, CANDIDATES_PER_HALF, &request.scope)?;
        coverage.keyword_candidates = lexical.len();

        let mut dense: Vec<DenseHit> = Vec::new();
        match self.embedder {
            Some(embedder) => {
                let space = embedder.identity().space_key();
                coverage.space_key = Some(space.clone());
                let covered = self.index.embedding_coverage(session, &space, &request.scope)?;
                coverage.embedding = Some(covered);
                match embed_query(embedder, &query).await {
                    Ok(vector) => {
                        dense = self.index.search_dense(
                            session,
                            &space,
                            &vector,
                            CANDIDATES_PER_HALF,
                            &request.scope,
                        )?;
                        if let Some(floor) = self.dense_floor {
                            // Strictly. A passage the keyword half also found
                            // keeps its keyword vote and its keyword label; it
                            // does not get a semantic vote, or a "semantic"
                            // label, for a similarity the model's own
                            // qualification says means "not about this".
                            let before = dense.len();
                            dense.retain(|hit| hit.similarity >= floor);
                            let dropped = before - dense.len();
                            if dropped > 0 && dense.is_empty() && lexical.is_empty() {
                                coverage.partial.push(format!(
                                    "no passage was closer to the question than this model's \
                                     qualified floor ({floor:.3})"
                                ));
                            }
                        }
                        coverage.mode = "hybrid".into();
                        if covered.embedded < covered.total {
                            coverage.partial.push(format!(
                                "semantic search covers {} of the {} passages you may read here; \
                                 the rest were found by keyword only{}",
                                covered.embedded,
                                covered.total,
                                if covered.failed > 0 {
                                    format!(" ({} could not be embedded)", covered.failed)
                                } else {
                                    String::new()
                                }
                            ));
                        }
                    }
                    Err(error) => {
                        coverage.mode = "lexical".into();
                        coverage.degraded_because = Some(format!(
                            "the question could not be embedded ({error}), so this search ran \
                             by keyword only"
                        ));
                    }
                }
            }
            None => {
                coverage.mode = "lexical".into();
                coverage.degraded_because = Some(
                    self.dense_unavailable
                        .clone()
                        .unwrap_or_else(|| "no qualified embedding model".into()),
                );
            }
        }
        coverage.semantic_candidates = dense.len();

        // 3. Fuse by rank.
        let lexical_ids: Vec<String> = lexical.iter().map(|hit| hit.result.chunk_id.clone()).collect();
        let dense_ids: Vec<String> = dense.iter().map(|hit| hit.result.chunk_id.clone()).collect();
        let fused = reciprocal_rank_fusion(&[lexical_ids, dense_ids]);

        let by_lexical: BTreeMap<&str, &LexicalHit> =
            lexical.iter().map(|hit| (hit.result.chunk_id.as_str(), hit)).collect();
        let by_dense: BTreeMap<&str, &DenseHit> =
            dense.iter().map(|hit| (hit.result.chunk_id.as_str(), hit)).collect();

        let pinned = match request.scope.pinned_revision {
            Some(revision) => Some(self.index.resolve_pin(revision)?),
            None => None,
        };

        // 4–5. Deduplicate, cut, annotate.
        let mut hits: Vec<EvidenceHit> = Vec::new();
        let mut seen_text: BTreeMap<String, usize> = BTreeMap::new();
        for (chunk_id, fused_score) in fused {
            let from_lexical = by_lexical.get(chunk_id.as_str());
            let from_dense = by_dense.get(chunk_id.as_str());
            let passage = match (from_lexical, from_dense) {
                (Some(hit), _) => hit.result.clone(),
                (None, Some(hit)) => hit.result.clone(),
                (None, None) => continue,
            };
            let fingerprint = normalised(&passage.text);
            if let Some(&kept) = seen_text.get(&fingerprint) {
                hits[kept].duplicates.push(passage.citation());
                continue;
            }
            if hits.len() >= limit {
                continue;
            }
            let method = match (from_lexical.is_some(), from_dense.is_some()) {
                (true, true) => HitMethod::LexicalAndDense,
                (false, true) => HitMethod::Dense,
                _ => HitMethod::Lexical,
            };
            let scores = HitScores {
                keyword_bm25: from_lexical.map(|hit| hit.bm25),
                keyword_pass: from_lexical.map(|hit| hit.pass),
                semantic_cosine: from_dense.map(|hit| hit.similarity),
                fused_rank: fused_score,
                rerank: None,
            };
            let source = self.source_of(session, &passage.document_sha256, pinned.as_ref())?;
            seen_text.insert(fingerprint, hits.len());
            hits.push(EvidenceHit {
                handle: EvidenceHit::handle_for(&passage.chunk_id),
                excerpt: excerpt(&passage.text, &query, EXCERPT_CHARS),
                passage,
                method,
                scores,
                source,
                duplicates: Vec::new(),
            });
        }

        if hits.is_empty() {
            coverage.no_answer = true;
        } else if hits
            .iter()
            .all(|hit| hit.method == HitMethod::Lexical && hit.scores.keyword_pass == Some(LexicalPass::AnyTerm))
        {
            coverage.partial.push(
                "no passage matched the whole question; these matched some of its terms".into(),
            );
        }
        let flagged = hits
            .iter()
            .filter(|hit| hit.source.as_ref().is_some_and(|source| source.superseded_since_pin))
            .count();
        if flagged > 0 {
            coverage.partial.push(format!(
                "{flagged} passage(s) come from a version superseded after this job was queued"
            ));
        }

        Ok(HybridResponse { query, hits, coverage })
    }

    /// The version-history row for this passage's document, if it has one.
    fn source_of(
        &self,
        session: &Session,
        sha: &str,
        pinned: Option<&super::index::PinnedCorpus>,
    ) -> Result<Option<HitSource>> {
        let Some(history) = self.index.source_versions(session, sha)? else {
            return Ok(None);
        };
        // The newest version that holds these bytes.
        let row = history
            .iter()
            .filter(|version| version.document_sha256 == sha)
            .max_by(|a, b| {
                (a.status == "current")
                    .cmp(&(b.status == "current"))
                    .then(a.version.cmp(&b.version))
            });
        Ok(row.map(|version| HitSource {
            collection_id: version.collection_id.clone(),
            relative_path: version.relative_path.clone(),
            version: version.version,
            status: version.status.clone(),
            superseded_since_pin: pinned.is_some_and(|corpus| corpus.superseded_since.contains(sha)),
        }))
    }
}

/// Lowercased, whitespace-collapsed text: identical passages compare equal.
fn normalised(text: &str) -> String {
    text.split_whitespace()
        .map(|word| word.to_lowercase())
        .collect::<Vec<_>>()
        .join(" ")
}

/// The significant terms of a question, lowercased.
pub fn significant_terms(query: &str) -> Vec<String> {
    const COMMON: &[&str] = &[
        "a", "an", "the", "is", "are", "was", "were", "be", "of", "to", "in", "on", "for", "and",
        "or", "what", "which", "who", "how", "when", "where", "why", "does", "do", "did", "with",
        "by", "at", "from", "as", "it", "this", "that", "should", "must", "can", "i", "we", "you",
    ];
    let mut seen = BTreeSet::new();
    query
        .split_whitespace()
        .map(|token| token.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase())
        .filter(|token| token.chars().count() >= 2 && !COMMON.contains(&token.as_str()))
        .filter(|token| seen.insert(token.clone()))
        .collect()
}

/// A short excerpt of a passage, centred on the first significant term it
/// contains, cut at word boundaries.
pub fn excerpt(text: &str, query: &str, max_chars: usize) -> String {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() {
        return String::new();
    }
    let joined = words.join(" ");
    if joined.chars().count() <= max_chars {
        return joined;
    }
    let terms = significant_terms(query);
    let anchor = words
        .iter()
        .position(|word| {
            let lower = word.to_lowercase();
            terms.iter().any(|term| lower.contains(term.as_str()))
        })
        .unwrap_or(0);
    // Start a few words before the anchor so the sentence has its subject.
    let mut start = anchor.saturating_sub(8);
    let mut out: Vec<&str> = Vec::new();
    let mut length = 0usize;
    while start < words.len() {
        let next = words[start].chars().count() + usize::from(!out.is_empty());
        if length + next > max_chars.saturating_sub(2) {
            break;
        }
        length += next;
        out.push(words[start]);
        start += 1;
    }
    let first = anchor.saturating_sub(8);
    let mut excerpt = out.join(" ");
    if first > 0 {
        excerpt = format!("…{excerpt}");
    }
    if start < words.len() {
        excerpt.push('…');
    }
    excerpt
}

/// A reranker that runs on this machine, over a bounded candidate set.
pub trait Reranker: Send + Sync {
    /// What this reranker is, for the record: a caller must be able to tell a
    /// lexical reorder from a model's judgement.
    fn describe(&self) -> &'static str;
    /// One score per passage, same order. Higher is better. For ordering only.
    fn score(&self, query: &str, passages: &[&SearchResult]) -> Vec<f64>;
}

/// Reorders by how completely and how closely a passage contains the
/// question's significant terms.
///
/// No model. Deterministic, bounded to [`RERANK_LIMIT`] passages, and labelled
/// as exactly what it is — it breaks ties that fusion leaves and promotes a
/// passage that holds the whole phrase over one that repeats one word. It does
/// not understand paraphrase; the dense half is what does that.
pub struct LexicalProximityReranker;

impl Reranker for LexicalProximityReranker {
    fn describe(&self) -> &'static str {
        "lexical-proximity reranker (deterministic, no model): term coverage, exact phrase, \
         and how close the terms sit"
    }

    fn score(&self, query: &str, passages: &[&SearchResult]) -> Vec<f64> {
        let terms = significant_terms(query);
        let phrase = normalised(query);
        passages
            .iter()
            .map(|passage| {
                if terms.is_empty() {
                    return 0.0;
                }
                let text = normalised(&passage.text);
                let tokens: Vec<String> = text
                    .split(' ')
                    .map(|token| token.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
                    .collect();
                let positions: Vec<Vec<usize>> = terms
                    .iter()
                    .map(|term| {
                        tokens
                            .iter()
                            .enumerate()
                            .filter(|(_, token)| token.contains(term.as_str()))
                            .map(|(index, _)| index)
                            .collect()
                    })
                    .collect();
                let present = positions.iter().filter(|p| !p.is_empty()).count();
                let coverage = present as f64 / terms.len() as f64;
                let exact = if phrase.chars().count() >= 4 && text.contains(&phrase) { 1.0 } else { 0.0 };
                // The shortest span holding one occurrence of every present term.
                let proximity = if present >= 2 {
                    let found: Vec<&Vec<usize>> = positions.iter().filter(|p| !p.is_empty()).collect();
                    let mut best = usize::MAX;
                    for &anchor in found[0] {
                        let mut low = anchor;
                        let mut high = anchor;
                        for others in &found[1..] {
                            let nearest = others
                                .iter()
                                .min_by_key(|&&position| position.abs_diff(anchor))
                                .copied()
                                .unwrap_or(anchor);
                            low = low.min(nearest);
                            high = high.max(nearest);
                        }
                        best = best.min(high - low + 1);
                    }
                    present as f64 / best.max(present) as f64
                } else {
                    0.0
                };
                2.0 * coverage + exact + proximity
            })
            .collect()
    }
}

/// Reranks a hybrid response in place, over at most [`RERANK_LIMIT`] hits.
///
/// Ties keep the fused order, so reranking never makes the output less
/// deterministic than it was.
pub fn rerank(response: &mut HybridResponse, reranker: &dyn Reranker) {
    let bounded = response.hits.len().min(RERANK_LIMIT);
    let passages: Vec<&SearchResult> = response.hits[..bounded].iter().map(|hit| &hit.passage).collect();
    let scores = reranker.score(&response.query, &passages);
    let mut indexed: Vec<(usize, f64, EvidenceHit)> = response
        .hits
        .drain(..bounded)
        .zip(scores)
        .enumerate()
        .map(|(position, (mut hit, score))| {
            hit.scores.rerank = Some(score);
            (position, score, hit)
        })
        .collect();
    indexed.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    let mut reordered: Vec<EvidenceHit> = indexed.into_iter().map(|(_, _, hit)| hit).collect();
    reordered.append(&mut response.hits);
    response.hits = reordered;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::{Role, User};
    use crate::knowledge::embedding::{EmbedError, EmbeddingIdentity, EmbeddingProfile, MULTILINGUAL_E5_SMALL};
    use crate::knowledge::{Chunk, ChunkKind};
    use crate::policy::Classification;

    fn ids(pairs: &[(String, f64)]) -> Vec<&str> {
        pairs.iter().map(|(id, _)| id.as_str()).collect()
    }

    fn ranking(ids: &[&str]) -> Vec<String> {
        ids.iter().map(|id| id.to_string()).collect()
    }

    #[test]
    fn one_ranking_fuses_to_itself() {
        let fused = reciprocal_rank_fusion(&[ranking(&["a", "b", "c"])]);
        assert_eq!(ids(&fused), vec!["a", "b", "c"]);
    }

    #[test]
    fn a_passage_both_searches_like_beats_one_only_a_single_search_loved() {
        let fused = reciprocal_rank_fusion(&[ranking(&["solo", "both"]), ranking(&["x", "both"])]);
        assert_eq!(fused[0].0, "both");
    }

    #[test]
    fn identical_input_produces_identical_output() {
        let input = [ranking(&["a", "b", "c", "d"]), ranking(&["d", "c", "b", "a"])];
        assert_eq!(reciprocal_rank_fusion(&input), reciprocal_rank_fusion(&input));
    }

    #[test]
    fn fusing_nothing_yields_nothing() {
        assert!(reciprocal_rank_fusion(&[]).is_empty());
        assert!(reciprocal_rank_fusion(&[Vec::new(), Vec::new()]).is_empty());
    }

    fn chunk(id: &str, sha: &str, ordinal: u32, text: &str) -> Chunk {
        Chunk {
            id: id.into(),
            document_sha256: sha.into(),
            ordinal,
            char_count: text.len() as u32,
            text: text.into(),
            page: 1,
            section_path: Vec::new(),
            kind: ChunkKind::Prose,
        }
    }

    fn session() -> Session {
        Session::open(User::new("kiran", "Kiran", vec![Role::Employee]))
    }

    /// A transport double for the dense half: vectors chosen per text, so the
    /// test can say which passage is "about" which question. Everything the
    /// test checks is the production code in front of it.
    struct Keyed {
        identity: EmbeddingIdentity,
        axes: Vec<(&'static str, usize)>,
        fail: bool,
    }

    #[async_trait::async_trait]
    impl Embedder for Keyed {
        fn identity(&self) -> &EmbeddingIdentity {
            &self.identity
        }
        fn profile(&self) -> &'static EmbeddingProfile {
            &MULTILINGUAL_E5_SMALL
        }
        async fn embed_raw(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>, EmbedError> {
            if self.fail {
                return Err(EmbedError::Failed("connection refused".into()));
            }
            Ok(inputs
                .iter()
                .map(|input| {
                    let mut v = vec![0.01f32; 384];
                    for (marker, axis) in &self.axes {
                        if input.contains(marker) {
                            v[*axis] += 1.0;
                        }
                    }
                    v
                })
                .collect())
        }
    }

    async fn embed_all(index: &KnowledgeIndex, embedder: &Keyed) {
        index.register_space(embedder.identity()).unwrap();
        let space = embedder.identity().space_key();
        for pending in index.passages_needing_embeddings(&space, 100).unwrap() {
            let vectors = crate::knowledge::embedding::embed_passage(embedder, &pending.body)
                .await
                .unwrap();
            index.store_embeddings(&space, &pending.chunk_id, &pending.body, &vectors).unwrap();
        }
    }

    fn corpus() -> (tempfile::TempDir, KnowledgeIndex) {
        let dir = tempfile::tempdir().unwrap();
        let index = KnowledgeIndex::open(dir.path()).unwrap();
        index
            .index_document(
                "Pump manual",
                Classification::Internal,
                &[
                    chunk("npsh", "pm", 0, "NPSH margin shall exceed 1.5 m at rated flow."),
                    chunk("seal", "pm", 1, "Replace the mechanical seal every 8000 hours."),
                ],
            )
            .unwrap();
        (dir, index)
    }

    fn keyed(fail: bool) -> Keyed {
        Keyed {
            identity: EmbeddingIdentity::new("e5", &MULTILINGUAL_E5_SMALL, &"ab".repeat(32)),
            // The question about cavitation and the NPSH passage share a
            // direction; nothing in their words overlaps.
            axes: vec![("NPSH", 7), ("cavitat", 7), ("seal", 9)],
            fail,
        }
    }

    #[tokio::test]
    async fn without_an_embedder_it_searches_by_keyword_and_says_why() {
        let (_dir, index) = corpus();
        let retriever = HybridRetriever::lexical_only(&index, "no qualified embedding model is installed");
        let response = retriever
            .search(&session(), &HybridRequest { query: "mechanical seal".into(), limit: 5, ..Default::default() })
            .await
            .unwrap();
        assert_eq!(response.coverage.mode, "lexical");
        assert_eq!(
            response.coverage.degraded_because.as_deref(),
            Some("no qualified embedding model is installed")
        );
        assert_eq!(response.hits[0].passage.chunk_id, "seal");
        assert_eq!(response.hits[0].method, HitMethod::Lexical);
    }

    #[tokio::test]
    async fn a_paraphrase_is_found_by_the_dense_half_and_labelled_so() {
        let (_dir, index) = corpus();
        let embedder = keyed(false);
        embed_all(&index, &embedder).await;
        let retriever = HybridRetriever { index: &index, embedder: Some(&embedder), dense_unavailable: None, dense_floor: None };
        let question = "Why would the charge pump start cavitating?";

        let lexical = HybridRetriever::lexical_only(&index, "off")
            .search(&session(), &HybridRequest { query: question.into(), limit: 5, ..Default::default() })
            .await
            .unwrap();
        assert!(
            lexical.hits.iter().all(|hit| hit.passage.chunk_id != "npsh"),
            "keyword search found the paraphrase, so this test proves nothing"
        );

        let hybrid = retriever
            .search(&session(), &HybridRequest { query: question.into(), limit: 5, ..Default::default() })
            .await
            .unwrap();
        assert_eq!(hybrid.coverage.mode, "hybrid");
        assert_eq!(hybrid.hits[0].passage.chunk_id, "npsh");
        assert_eq!(hybrid.hits[0].method, HitMethod::Dense);
        assert!(hybrid.hits[0].scores.semantic_cosine.is_some());
        assert!(hybrid.hits[0].scores.keyword_bm25.is_none());
    }

    #[tokio::test]
    async fn an_embedder_that_cannot_answer_degrades_to_keyword_and_says_so() {
        let (_dir, index) = corpus();
        let good = keyed(false);
        embed_all(&index, &good).await;
        let broken = keyed(true);
        let retriever = HybridRetriever { index: &index, embedder: Some(&broken), dense_unavailable: None, dense_floor: None };
        let response = retriever
            .search(&session(), &HybridRequest { query: "seal".into(), limit: 5, ..Default::default() })
            .await
            .unwrap();
        assert_eq!(response.coverage.mode, "lexical");
        assert!(response.coverage.degraded_because.unwrap().contains("could not be embedded"));
        assert_eq!(response.hits[0].passage.chunk_id, "seal");
    }

    #[tokio::test]
    async fn half_an_embedded_index_says_it_is_half() {
        let (_dir, index) = corpus();
        let embedder = keyed(false);
        index.register_space(embedder.identity()).unwrap();
        let space = embedder.identity().space_key();
        let first = index.passages_needing_embeddings(&space, 1).unwrap().remove(0);
        let vectors = crate::knowledge::embedding::embed_passage(&embedder, &first.body).await.unwrap();
        index.store_embeddings(&space, &first.chunk_id, &first.body, &vectors).unwrap();
        let retriever = HybridRetriever { index: &index, embedder: Some(&embedder), dense_unavailable: None, dense_floor: None };
        let response = retriever
            .search(&session(), &HybridRequest { query: "seal".into(), limit: 5, ..Default::default() })
            .await
            .unwrap();
        assert!(response.coverage.partial.iter().any(|line| line.contains("1 of the 2")), "{:?}", response.coverage.partial);
    }

    #[tokio::test]
    async fn no_answer_is_said_rather_than_filled() {
        let (_dir, index) = corpus();
        let response = HybridRetriever::lexical_only(&index, "off")
            .search(&session(), &HybridRequest { query: "flare stack purge".into(), limit: 5, ..Default::default() })
            .await
            .unwrap();
        assert!(response.hits.is_empty());
        assert!(response.coverage.no_answer);
    }

    #[tokio::test]
    async fn identical_text_in_two_documents_is_one_hit_naming_both() {
        let dir = tempfile::tempdir().unwrap();
        let index = KnowledgeIndex::open(dir.path()).unwrap();
        for (name, sha) in [("SOP copy A", "a"), ("SOP copy B", "b")] {
            index
                .index_document(name, Classification::Internal, &[chunk(&format!("{sha}0"), sha, 0, "Purge the drum with nitrogen.")])
                .unwrap();
        }
        let response = HybridRetriever::lexical_only(&index, "off")
            .search(&session(), &HybridRequest { query: "purge drum".into(), limit: 5, ..Default::default() })
            .await
            .unwrap();
        assert_eq!(response.hits.len(), 1);
        assert_eq!(response.hits[0].duplicates.len(), 1);
    }

    #[test]
    fn an_excerpt_is_short_centred_and_cut_at_words() {
        let filler = "word ".repeat(200);
        let text = format!("{filler}the design pressure is 14 bar gauge {filler}");
        let cut = excerpt(&text, "design pressure", 120);
        assert!(cut.chars().count() <= 120, "{}", cut.chars().count());
        assert!(cut.contains("design pressure"));
        assert!(cut.starts_with('…') && cut.ends_with('…'));
    }

    #[test]
    fn the_reranker_prefers_the_whole_phrase_and_is_deterministic() {
        let make = |id: &str, text: &str| SearchResult {
            chunk_id: id.into(),
            document_sha256: "d".into(),
            document_name: "D".into(),
            text: text.into(),
            page: 1,
            section_path: Vec::new(),
            classification: Classification::Internal,
            score: 0.0,
            retrieval: crate::knowledge::Retrieval::Keyword,
        };
        let far = make("far", "The pump is large. Much later the text mentions a seal.");
        let near = make("near", "Replace the pump seal.");
        let scores = LexicalProximityReranker.score("pump seal", &[&far, &near]);
        assert!(scores[1] > scores[0], "{scores:?}");
        assert_eq!(scores, LexicalProximityReranker.score("pump seal", &[&far, &near]));
    }
}
