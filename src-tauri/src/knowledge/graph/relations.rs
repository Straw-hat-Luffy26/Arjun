//! Naming the edges, with Babelscape/rebel-large.
//!
//! [`super::statistical`] finds that two things are mentioned together.
//! [`super::typing`] says what each of them *is*. Neither says what the link
//! between them means, and an edge labelled only by how many passages it
//! appeared in is a line on a picture, not a fact anybody can act on. This pass
//! puts a name on it: *manufacturer*, *owned by*, *operator*.
//!
//! # Why a separate model, and a separate process
//!
//! REBEL is a 400M-parameter BART sequence-to-sequence model that emits
//! `(subject, relation, object)` triplets directly. llama.cpp cannot load it —
//! it is not a GGUF decoder-only model — so none of [`crate::serving`] applies:
//! no VRAM planning, no admission, no llama-server. It runs in a Python sidecar
//! over stdin/stdout, the same transport [`crate::memory_engine`] uses, on CPU.
//!
//! That last part is a feature. [`super::typing`] has to borrow whichever
//! llama-server endpoint is already warm and refuses when there is none, because
//! admitting a model to VRAM can evict the one the user is chatting with. This
//! pass evicts nothing and waits for nothing.
//!
//! # What was measured before this was written
//!
//! REBEL was trained on Wikipedia abstracts against Wikidata properties, and it
//! shows. Run over four passages on this machine it returned, verbatim:
//!
//! - Wikipedia prose — six triplets, all correct.
//! - *"Pump PV-2201 was supplied by Acme Pumps Ltd under contract CT-4471…"* —
//!   one triplet, `(Acme Pumps Ltd, headquarters location, Pune)`. It missed the
//!   supplier link, the contract and the division entirely.
//! - *"…vessel V-101 shall be fabricated from SA-516 Grade 70 plate."* —
//!   `(SA-516, instance of, Grade 70)`, which is a material name cut in half.
//! - *"PT-2201 | 0-25 barg | Rosemount 3051 | Loop 2201"* —
//!   `(PT-2201, located in the administrative territorial entity, Rosemount)`.
//!   Rosemount is the instrument manufacturer. There is no territory.
//!
//! That last one is why [`RELATION_ALLOWLIST`] exists. It passes every gate the
//! typing pass would apply — both entities really are in the passage, and they
//! really do co-occur as an edge — so the only thing standing between it and the
//! user's graph is a judgement about which relations mean anything here. The
//! allowlist is that judgement, written down, with the measurement above as its
//! justification rather than a guess about what a model might do.
//!
//! # The gates
//!
//! Same discipline as [`super::typing::verify`], for the same reason: a model
//! that returns a plausible link nobody can trace is worse than a model that
//! returns nothing, because the graph then looks richer than the documents
//! support. Five gates, every rejection counted and reported:
//!
//! 1. the triplet cites a chunk that was actually sent,
//! 2. subject and object both occur in that chunk,
//! 3. both resolve to terms the statistical pass already found,
//! 4. that edge already exists for this document,
//! 5. the relation is one this domain has a use for.
//!
//! Gate 3 and gate 4 together mean this pass can only ever *name* an edge. It
//! cannot add a node, and it cannot add an edge. [`super::persist`] enforces the
//! same rule at the storage layer by issuing only `UPDATE`s, the way
//! `store_document_typing` does.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

/// Bumped when a change would produce different relations from the same input.
///
/// Recorded per document in `graph_build_units`, so a notebook part-way through
/// a pass finishes where it stopped, and a version bump re-runs everything
/// rather than leaving a notebook half-labelled by two different extractors.
pub const RELATION_VERSION: u32 = 1;

/// The relations worth keeping.
///
/// REBEL's vocabulary is Wikidata properties — a few hundred of them, most
/// about people, places and creative works. This is the subset that says
/// something about equipment, suppliers, sites and documents. Everything else
/// is dropped and counted as `dropped_offdomain`.
///
/// An allowlist rather than a blocklist because the failure it guards against is
/// a *confident geographic assertion about an instrument tag*, and there is no
/// way to enumerate those. Naming what belongs is finite; naming what does not
/// is not.
///
/// It is deliberately short. A relation admitted here is one a reader will act
/// on, and the cost of admitting a bad one is a wrong fact in an approval note.
/// Widening it is a decision to make against measured output, not in advance.
pub const RELATION_ALLOWLIST: &[&str] = &[
    // Who made or supplied a thing.
    "manufacturer",
    "developer",
    "brand",
    "distributed by",
    "product or material produced",
    "supplier",
    // Who owns, runs or answers for it.
    "owned by",
    "operator",
    "maintained by",
    "parent organization",
    "subsidiary",
    "employer",
    "member of",
    // How things are built out of other things.
    "part of",
    "has part",
    "has parts",
    "made from material",
    "material used",
    "uses",
    "powered by",
    // Documents and the things they govern.
    "author",
    "publisher",
    "editor",
    // Where an *organisation* sits. Kept because a supplier's location is a
    // real procurement fact; `located in the administrative territorial entity`
    // is not on this list, because that is the one REBEL reaches for when it
    // mistakes an instrument tag for a town.
    "headquarters location",
    "industry",
];

/// One relation as the sidecar returned it. Unverified.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Triplet {
    pub chunk_id: String,
    pub subject: String,
    pub relation: String,
    pub object: String,
}

/// One relation that survived every gate, addressed the way the graph stores it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedRelation {
    /// Normalised source term, as `graph_edges.source` holds it.
    pub source: String,
    /// Normalised target term, as `graph_edges.target` holds it.
    pub target: String,
    pub relation: String,
    /// The passage this was read out of, so the claim stays traceable.
    pub chunk_id: String,
}

/// What the gates rejected, and why.
///
/// Reported rather than logged. A pass that kept four relations out of ninety is
/// saying the model is wrong for these documents, and a screen showing only the
/// four reads as a sparse graph instead of a failed extraction.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RelationStats {
    pub proposed: u32,
    pub kept: u32,
    /// Cited a passage that was not in the request.
    pub dropped_uncited: u32,
    /// Named something the cited passage does not contain. Fabrication, caught.
    pub dropped_misquoted: u32,
    /// Named something the statistical pass never found.
    pub dropped_unknown_term: u32,
    /// Named a pair that is not an edge of this document's graph.
    pub dropped_unknown_edge: u32,
    /// A real relation, in a vocabulary this domain has no use for.
    pub dropped_offdomain: u32,
}

/// The verified result of one document's extraction.
#[derive(Debug, Clone, Default)]
pub struct Verdict {
    pub relations: Vec<VerifiedRelation>,
    pub stats: RelationStats,
}

/// Collapses whitespace and case for comparison.
///
/// The same fold [`super::typing`] applies to quotes, so a passage that counts
/// as containing a quote there counts as containing an entity here.
fn comparable(text: &str) -> String {
    text.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

/// Whether `haystack` contains `needle` as a whole run of words.
///
/// Substring matching alone would let `pv-220` match `pv-2201`, which is a
/// different piece of equipment. Both sides are already normalised, so splitting
/// on spaces is enough to compare word runs.
fn contains_words(haystack: &str, needle: &str) -> bool {
    if needle.is_empty() || haystack.len() < needle.len() {
        return false;
    }
    let hay: Vec<&str> = haystack.split(' ').collect();
    let need: Vec<&str> = needle.split(' ').collect();
    if need.is_empty() || need.len() > hay.len() {
        return false;
    }
    hay.windows(need.len()).any(|window| window == need.as_slice())
}

/// Maps an entity string the model returned onto a term the graph already holds.
///
/// Exact match on the normalised form first. Failing that, one containment
/// match — but *only* one.
///
/// The loosening is not generosity, it is a measured behaviour: asked twice
/// about the same sentence, REBEL returned `Acme Pumps Ltd` once and
/// `Acme Pumps` the other time. It clips entity strings. Requiring an exact
/// match would drop the second as a fabrication when the model had in fact read
/// the passage correctly.
///
/// Ambiguity is refused rather than guessed. If `pump` is contained in both
/// `pump pv-2201` and `pump pv-2202`, nothing in the triplet says which was
/// meant, and picking one would invent a link between two real things — the
/// most expensive kind of wrong answer this pass can produce.
fn resolve_term(entity: &str, known: &BTreeMap<String, String>) -> Option<String> {
    let needle = super::statistical::normalise_phrase(entity);
    if needle.is_empty() {
        return None;
    }
    if known.contains_key(&needle) {
        return Some(needle);
    }

    let mut matches = known.keys().filter(|term| {
        // Word-run containment in either direction: the model clips a term
        // short, or pads it with a word the extractor dropped.
        contains_words(term, &needle) || contains_words(&needle, term)
    });
    let first = matches.next()?;
    if matches.next().is_some() {
        // More than one candidate. No basis to choose, so choose nothing.
        return None;
    }
    Some(first.clone())
}

/// Applies the five gates to what the sidecar returned.
///
/// `known` maps a normalised term to its label — what
/// [`super::persist::NotebookStore::document_terms`] returned for this document.
/// `chunks` maps chunk id to the text actually sent. `edges` holds the
/// `(source, target)` pairs this document's graph already has.
///
/// Pure, and separated from the sidecar for the same reason
/// [`super::typing::verify`] is separated from its HTTP call: the gates are the
/// part worth testing, and they should be testable without a 1.6 GB model.
pub fn verify(
    triplets: &[Triplet],
    known: &BTreeMap<String, String>,
    chunks: &BTreeMap<String, String>,
    edges: &BTreeSet<(String, String)>,
) -> Verdict {
    let mut verdict = Verdict {
        stats: RelationStats {
            proposed: triplets.len() as u32,
            ..Default::default()
        },
        ..Default::default()
    };

    // One relation per edge. The first passage to support an edge names it, and
    // a later disagreement does not silently overwrite it — an edge whose label
    // changes depending on read order is not reproducible.
    let mut claimed: BTreeSet<(String, String)> = BTreeSet::new();

    for triplet in triplets {
        // 1. Did this passage go out in the request?
        let Some(chunk_text) = chunks.get(&triplet.chunk_id) else {
            verdict.stats.dropped_uncited += 1;
            continue;
        };

        // 2. Are both entities actually in it?
        let haystack = comparable(chunk_text);
        if !haystack.contains(&comparable(&triplet.subject))
            || !haystack.contains(&comparable(&triplet.object))
        {
            verdict.stats.dropped_misquoted += 1;
            continue;
        }

        // 5. Is this relation one the domain can use? Checked before the term
        //    lookups because it is the cheapest of the three, and because an
        //    off-domain relation between two perfectly real terms is the common
        //    case — see the module docs.
        let relation = triplet.relation.trim().to_lowercase();
        if !RELATION_ALLOWLIST.contains(&relation.as_str()) {
            verdict.stats.dropped_offdomain += 1;
            continue;
        }

        // 3. Do both resolve to terms the cheap pass already found?
        let (Some(source), Some(target)) = (
            resolve_term(&triplet.subject, known),
            resolve_term(&triplet.object, known),
        ) else {
            verdict.stats.dropped_unknown_term += 1;
            continue;
        };
        if source == target {
            // A self-edge. The graph has no row for it and it says nothing.
            verdict.stats.dropped_unknown_edge += 1;
            continue;
        }

        // 4. Is this an edge the document's graph actually has? Checked both
        //    ways round: co-occurrence is symmetric and `graph_edges` stores one
        //    row per unordered pair, so the stored order is what gets written
        //    back regardless of which way the model named it.
        let forward = (source.clone(), target.clone());
        let backward = (target, source);
        let stored = if edges.contains(&forward) {
            forward
        } else if edges.contains(&backward) {
            backward
        } else {
            verdict.stats.dropped_unknown_edge += 1;
            continue;
        };

        if !claimed.insert(stored.clone()) {
            // Already named by an earlier passage. Not a rejection.
            continue;
        }

        verdict.stats.kept += 1;
        verdict.relations.push(VerifiedRelation {
            source: stored.0,
            target: stored.1,
            relation,
            chunk_id: triplet.chunk_id.clone(),
        });
    }

    verdict
}

/// A running graph sidecar, and the pipes to talk to it.
///
/// Long-lived on purpose. Loading REBEL costs several seconds and 1.6 GB of
/// reads, so a pass over forty documents that spawned per document would spend
/// most of its time loading. One process serves the whole pass and is killed
/// when this value drops.
///
/// The impure half of this module, kept beside the gates the way
/// [`super::typing::request_typing`] sits beside `verify` — one file, one
/// subject, with the part worth testing separable from the part that needs a
/// model on disk.
pub struct RelationSidecar {
    child: std::process::Child,
    stdin: std::process::ChildStdin,
    reader: std::io::BufReader<std::process::ChildStdout>,
    request_id: u64,
}

impl RelationSidecar {
    /// Spawns the sidecar against the REBEL directory in the model library.
    ///
    /// `model_dir` is passed through the environment rather than as an argument
    /// because it is a path on a machine that may have spaces in it, and an
    /// environment variable cannot be re-split by a shell that is not there.
    pub fn spawn(model_dir: &std::path::Path) -> anyhow::Result<Self> {
        use std::process::Stdio;

        if !model_dir.is_dir() {
            anyhow::bail!(
                "The relation model is not installed. Expected Babelscape/rebel-large at {}.",
                model_dir.display()
            );
        }

        let script = crate::deployment::require_path("graph-sidecar")
            .map_err(|problem| anyhow::anyhow!(problem))?;
        let script_dir = script
            .parent()
            .map(std::path::Path::to_path_buf)
            .unwrap_or_default();

        // Hidden, or the GUI build pops a console window when it spawns a
        // console application.
        let mut command =
            crate::system_analyzer::process_utils::create_hidden_command(
                crate::deployment::program("python"),
            );
        command.arg(&script);
        command.env("PYTHONPATH", &script_dir);
        command.env("ARJUN_REBEL_DIR", model_dir);
        command.stdin(Stdio::piped());
        command.stdout(Stdio::piped());
        // Inherited, not piped. transformers writes a weight-loading progress
        // bar to stderr; piping it without a reader would fill the pipe buffer
        // and deadlock the child part-way through its first load.
        command.stderr(Stdio::inherit());

        let mut child = command
            .spawn()
            .map_err(|problem| anyhow::anyhow!("the graph sidecar could not start: {problem}"))?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| anyhow::anyhow!("the graph sidecar has no stdin"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| anyhow::anyhow!("the graph sidecar has no stdout"))?;

        Ok(Self {
            child,
            stdin,
            reader: std::io::BufReader::new(stdout),
            request_id: 0,
        })
    }

    /// Asks the sidecar for the triplets in a batch of passages.
    ///
    /// Returns them exactly as the model produced them. Nothing here judges
    /// whether a triplet is supportable — that is [`verify`], which needs the
    /// document's terms and edges and so cannot run at this layer.
    pub fn extract(&mut self, chunks: &[(String, String)]) -> anyhow::Result<Vec<Triplet>> {
        use std::io::{BufRead, Write};

        let payload: Vec<serde_json::Value> = chunks
            .iter()
            .map(|(id, text)| serde_json::json!({ "id": id, "text": text }))
            .collect();

        self.request_id += 1;
        let mut line = serde_json::to_string(&serde_json::json!({
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": "graph.extract_triplets",
            "params": { "chunks": payload },
        }))?;
        line.push('\n');

        self.stdin.write_all(line.as_bytes())?;
        self.stdin.flush()?;

        let mut response = String::new();
        self.reader.read_line(&mut response)?;
        if response.trim().is_empty() {
            anyhow::bail!("the graph sidecar stopped without answering");
        }

        let value: serde_json::Value = serde_json::from_str(&response)?;
        if let Some(error) = value.get("error") {
            anyhow::bail!(
                "the graph sidecar failed: {}",
                error
                    .get("message")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("no message")
            );
        }

        let triplets = value
            .get("result")
            .and_then(|result| result.get("triplets"))
            .cloned()
            .unwrap_or_else(|| serde_json::json!([]));
        serde_json::from_value(triplets)
            .map_err(|problem| anyhow::anyhow!("the graph sidecar returned unreadable triplets: {problem}"))
    }
}

impl Drop for RelationSidecar {
    fn drop(&mut self) {
        // The sidecar holds 1.6 GB resident. Leaving one behind per pass would
        // exhaust a 16 GB machine in a morning.
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Where the relation model lives inside the model library.
///
/// Mirrors the layout the downloader writes and the registry records, so the
/// two cannot drift apart silently.
pub fn model_directory(models_dir: &std::path::Path) -> std::path::PathBuf {
    models_dir
        .join("local")
        .join("Babelscape_rebel-large")
        .join("base")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verbatim from Babelscape/rebel-large on this machine, for the tag-row
    /// passage in the module docs. Pinned because a gate tested against an
    /// invented string passes until it meets the model.
    const REAL_WIKI_RELATION: &str = "located in the administrative territorial entity";

    fn known() -> BTreeMap<String, String> {
        BTreeMap::from([
            ("pump pv-2201".to_string(), "Pump PV-2201".to_string()),
            ("acme pumps ltd".to_string(), "Acme Pumps Ltd".to_string()),
            ("pt-2201".to_string(), "PT-2201".to_string()),
            ("rosemount".to_string(), "Rosemount".to_string()),
        ])
    }

    fn chunks() -> BTreeMap<String, String> {
        BTreeMap::from([
            (
                "c-0007".to_string(),
                "Pump PV-2201 was supplied by Acme Pumps Ltd under contract CT-4471."
                    .to_string(),
            ),
            (
                "c-0044".to_string(),
                "PT-2201 | 0-25 barg | Rosemount 3051 | Loop 2201".to_string(),
            ),
        ])
    }

    fn edges() -> BTreeSet<(String, String)> {
        BTreeSet::from([
            ("acme pumps ltd".to_string(), "pump pv-2201".to_string()),
            ("pt-2201".to_string(), "rosemount".to_string()),
        ])
    }

    fn triplet(chunk: &str, subject: &str, relation: &str, object: &str) -> Triplet {
        Triplet {
            chunk_id: chunk.to_string(),
            subject: subject.to_string(),
            relation: relation.to_string(),
            object: object.to_string(),
        }
    }

    #[test]
    fn a_good_relation_is_kept_and_addressed_by_stored_term() {
        let proposals = vec![triplet(
            "c-0007",
            "Acme Pumps Ltd",
            "manufacturer",
            "Pump PV-2201",
        )];
        let verdict = verify(&proposals, &known(), &chunks(), &edges());

        assert_eq!(verdict.stats.kept, 1);
        assert_eq!(
            verdict.relations,
            vec![VerifiedRelation {
                source: "acme pumps ltd".into(),
                target: "pump pv-2201".into(),
                relation: "manufacturer".into(),
                chunk_id: "c-0007".into(),
            }]
        );
    }

    /// The measurement that justifies the allowlist. Both entities really are in
    /// the passage and they really are an edge, so gates 1-4 all pass. Only the
    /// vocabulary check stops it.
    #[test]
    fn the_instrument_tag_geography_error_is_refused() {
        let proposals = vec![triplet("c-0044", "PT-2201", REAL_WIKI_RELATION, "Rosemount")];
        let verdict = verify(&proposals, &known(), &chunks(), &edges());

        assert_eq!(verdict.stats.kept, 0);
        assert_eq!(verdict.stats.dropped_offdomain, 1);
        assert!(verdict.relations.is_empty());
    }

    /// REBEL clips entity strings: the same sentence yielded `Acme Pumps Ltd`
    /// once and `Acme Pumps` the next time. The clipped form must still land on
    /// the term the statistical pass found.
    #[test]
    fn a_clipped_entity_still_resolves() {
        let proposals = vec![triplet(
            "c-0007",
            "Acme Pumps",
            "manufacturer",
            "Pump PV-2201",
        )];
        let verdict = verify(&proposals, &known(), &chunks(), &edges());

        assert_eq!(verdict.stats.kept, 1);
        assert_eq!(verdict.relations[0].source, "acme pumps ltd");
    }

    #[test]
    fn an_ambiguous_clip_is_refused_rather_than_guessed() {
        let mut terms = known();
        terms.insert("pump pv-2202".to_string(), "Pump PV-2202".to_string());
        let mut passages = chunks();
        passages.insert(
            "c-0009".to_string(),
            "Pump PV-2201 and Pump PV-2202 were supplied by Acme Pumps Ltd.".to_string(),
        );

        let proposals = vec![triplet("c-0009", "Acme Pumps Ltd", "manufacturer", "Pump")];
        let verdict = verify(&proposals, &terms, &passages, &edges());

        assert_eq!(verdict.stats.kept, 0);
        assert_eq!(verdict.stats.dropped_unknown_term, 1);
    }

    #[test]
    fn a_relation_between_things_not_in_the_passage_is_a_fabrication() {
        let proposals = vec![triplet(
            "c-0007",
            "Acme Pumps Ltd",
            "manufacturer",
            "Vessel V-101",
        )];
        let verdict = verify(&proposals, &known(), &chunks(), &edges());

        assert_eq!(verdict.stats.dropped_misquoted, 1);
        assert_eq!(verdict.stats.kept, 0);
    }

    #[test]
    fn a_passage_that_was_never_sent_is_refused() {
        let proposals = vec![triplet(
            "c-9999",
            "Acme Pumps Ltd",
            "manufacturer",
            "Pump PV-2201",
        )];
        let verdict = verify(&proposals, &known(), &chunks(), &edges());

        assert_eq!(verdict.stats.dropped_uncited, 1);
    }

    #[test]
    fn a_pair_that_is_not_already_an_edge_cannot_become_one() {
        let mut passages = chunks();
        passages.insert(
            "c-0011".to_string(),
            "Acme Pumps Ltd and PT-2201 both appear here.".to_string(),
        );
        let proposals = vec![triplet("c-0011", "Acme Pumps Ltd", "manufacturer", "PT-2201")];
        let verdict = verify(&proposals, &known(), &passages, &edges());

        assert_eq!(verdict.stats.kept, 0);
        assert_eq!(verdict.stats.dropped_unknown_edge, 1);
    }

    /// The edge is stored under one unordered pair. A triplet naming it the
    /// other way round must still find it.
    #[test]
    fn an_edge_is_found_whichever_way_the_triplet_names_it() {
        let proposals = vec![triplet(
            "c-0007",
            "Pump PV-2201",
            "manufacturer",
            "Acme Pumps Ltd",
        )];
        let verdict = verify(&proposals, &known(), &chunks(), &edges());

        assert_eq!(verdict.stats.kept, 1);
        // Written under the pair the graph already holds, not the model's order.
        assert_eq!(verdict.relations[0].source, "acme pumps ltd");
        assert_eq!(verdict.relations[0].target, "pump pv-2201");
    }

    #[test]
    fn one_edge_takes_one_relation_however_many_passages_mention_it() {
        let proposals = vec![
            triplet("c-0007", "Acme Pumps Ltd", "manufacturer", "Pump PV-2201"),
            triplet("c-0007", "Acme Pumps Ltd", "owned by", "Pump PV-2201"),
        ];
        let verdict = verify(&proposals, &known(), &chunks(), &edges());

        assert_eq!(verdict.relations.len(), 1);
        assert_eq!(verdict.relations[0].relation, "manufacturer");
        // The second is not a rejection — nothing was wrong with it.
        assert_eq!(verdict.stats.dropped_offdomain, 0);
        assert_eq!(verdict.stats.dropped_unknown_edge, 0);
    }

    #[test]
    fn word_boundaries_are_respected_when_resolving() {
        // `pv-220` must not resolve to `pv-2201`: different equipment.
        assert!(!contains_words("pump pv-2201", "pv-220"));
        assert!(contains_words("pump pv-2201", "pv-2201"));
        assert!(contains_words("acme pumps ltd", "acme pumps"));
    }

    #[test]
    fn every_allowlisted_relation_is_lowercase_and_unique() {
        // The gate lowercases before comparing, so an uppercase entry here would
        // be unreachable and would look like a working rule.
        let mut seen = BTreeSet::new();
        for relation in RELATION_ALLOWLIST {
            assert_eq!(*relation, relation.to_lowercase(), "{relation}");
            assert!(seen.insert(*relation), "duplicate: {relation}");
        }
    }
}
