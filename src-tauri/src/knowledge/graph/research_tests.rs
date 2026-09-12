//! The notebook research journey, over the real stores.
//!
//! Every test here is a regression for a specific way this feature was wrong,
//! or could quietly become wrong again. They run against a real
//! [`DocumentStore`] writing JSON to a real directory and a real
//! [`NotebookStore`] writing to a real SQLite file, because the bugs below live
//! in the seams between those two and are invisible to a unit test holding
//! fixtures.
//!
//! The collection is deliberately shaped to contain the awkward cases:
//!
//! - **A repeated entity.** Northern Valve Company is named in three documents,
//!   so removing one source must not remove the entity.
//! - **A directed relationship whose subject sorts after its object.**
//!   `pv-2201` > `northern valve company`, which is the case the old
//!   sort-the-ends rule inverted.
//! - **Conflicting statements.** Two documents disagree about who maintains
//!   PV-2201, so the graph must carry both rather than pick one.
//! - **A question nothing supports**, and **a source that is removed.**

use std::collections::{BTreeMap, BTreeSet};

use crate::agent_runtime::documents::{DocumentStore, NewExtraction, Sighting};
use crate::knowledge::graph::assertions::{AssertionProvenance, AssertionStatus, NewAssertion};
use crate::knowledge::graph::research::{ResearchScope, RetrievalMode};
use crate::knowledge::graph::statistical::{extract, EXTRACTOR_VERSION};
use crate::knowledge::graph::{NoteKind, NotebookStore};
use crate::knowledge::notebook_retrieval;

const ALICE: &str = "user-alice";
const BOB: &str = "user-bob";

const LOG_SHA: &str = "c1c2c3d4e5f60718293a4b5c6d7e8f90a1b2c3d4e5f60718293a4b5c6d7e8f90";
const CONTRACT_SHA: &str = "d2c3d4e5f60718293a4b5c6d7e8f90a1b2c3d4e5f60718293a4b5c6d7e8f90a1";
const MEMO_SHA: &str = "e3c3d4e5f60718293a4b5c6d7e8f90a1b2c3d4e5f60718293a4b5c6d7e8f90a2";

struct Journey {
    documents: DocumentStore,
    notebooks: NotebookStore,
    notebook_id: String,
    /// A second notebook belonging to the same person, holding one of the same
    /// documents. Every destructive test checks this one is untouched.
    other_notebook_id: String,
}

fn temp_dir(tag: &str) -> std::path::PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let mut dir = std::env::temp_dir();
    dir.push(format!(
        "arjun-research-{tag}-{}-{}",
        std::process::id(),
        nanos
    ));
    std::fs::create_dir_all(&dir).expect("temp dir");
    dir
}

/// One page is one passage here, so a term has to appear on two pages to clear
/// [`super::statistical`]'s two-passage floor. Each fixture is three pages for
/// that reason, and stays under five so the generic-term ceiling — which only
/// applies once there are enough passages for a share to mean anything — does
/// not come into it.
fn maintenance_log() -> BTreeMap<u32, String> {
    let mut pages = BTreeMap::new();
    pages.insert(
        1,
        "Maintenance log for Unit Four.\n\n\
         Valve PV-2201 was replaced during the March outage. Northern Valve \
         Company attended site and completed the work within the shift."
            .to_string(),
    );
    pages.insert(
        2,
        "Northern Valve Company supplied the replacement cartridge for PV-2201. \
         The gasket on PV-2201 is torqued to 90 Nm."
            .to_string(),
    );
    pages.insert(
        3,
        "PV-2201 is maintained by the Unit Four mechanical crew. Northern Valve \
         Company will return in October for the scheduled inspection."
            .to_string(),
    );
    pages
}

fn supply_contract() -> BTreeMap<u32, String> {
    let mut pages = BTreeMap::new();
    pages.insert(
        1,
        "Supply agreement CT-4471.\n\n\
         Northern Valve Company shall hold spares for PV-2201 for the term of \
         this agreement."
            .to_string(),
    );
    pages.insert(
        2,
        "Agreement CT-4471 covers the Unit Four control valves supplied by \
         Northern Valve Company."
            .to_string(),
    );
    pages.insert(
        3,
        "Under CT-4471, Northern Valve Company is the sole approved supplier for \
         PV-2201."
            .to_string(),
    );
    pages
}

/// Disagrees with the maintenance log about who maintains PV-2201.
fn handover_memo() -> BTreeMap<u32, String> {
    let mut pages = BTreeMap::new();
    pages.insert(
        1,
        "Handover memo for Unit Four.\n\n\
         From 1 April, PV-2201 is maintained by Northern Valve Company under the \
         extended service agreement."
            .to_string(),
    );
    pages.insert(
        2,
        "This replaces the earlier arrangement in which PV-2201 was maintained by \
         the Unit Four mechanical crew."
            .to_string(),
    );
    pages.insert(
        3,
        "Northern Valve Company confirmed the extended service agreement covering \
         PV-2201."
            .to_string(),
    );
    pages
}

fn record(
    store: &DocumentStore,
    notebook_id: &str,
    sha: &str,
    name: &str,
    pages: BTreeMap<u32, String>,
) {
    store
        .record(NewExtraction {
            sha256: sha.to_string(),
            name: name.to_string(),
            kind: "pdf-text".to_string(),
            pages: pages.len() as u32,
            truncated: false,
            page_text: pages,
            sighting: Sighting {
                owner_user_id: ALICE.to_string(),
                conversation_id: format!("notebook:{notebook_id}"),
                message_id: format!("notebook-add:{sha}"),
                run_id: format!("notebook:{notebook_id}"),
                at: chrono::Utc::now().to_rfc3339(),
                ocr_model_id: None,
                ocr_detent: None,
            },
        })
        .expect("the extraction should store");
}

fn journey(tag: &str) -> Journey {
    let dir = temp_dir(tag);
    let documents = DocumentStore::open(&dir).expect("document store");
    let notebooks = NotebookStore::open(&dir).expect("notebook store");

    let notebook = notebooks.create(ALICE, "Unit Four").expect("notebook");
    let other = notebooks
        .create(ALICE, "Procurement")
        .expect("second notebook");

    for (sha, name, pages) in [
        (LOG_SHA, "Maintenance log.pdf", maintenance_log()),
        (CONTRACT_SHA, "Supply agreement.pdf", supply_contract()),
        (MEMO_SHA, "Handover memo.pdf", handover_memo()),
    ] {
        record(&documents, &notebook.id, sha, name, pages);
        notebooks
            .add_document(&notebook.id, ALICE, sha, name)
            .expect("membership");
    }

    // The contract is in both notebooks. Removing it from one must leave the
    // other's copy, its graph rows and its membership alone.
    record(
        &documents,
        &other.id,
        CONTRACT_SHA,
        "Supply agreement.pdf",
        supply_contract(),
    );
    notebooks
        .add_document(&other.id, ALICE, CONTRACT_SHA, "Supply agreement.pdf")
        .expect("membership in the second notebook");

    Journey {
        documents,
        notebooks,
        notebook_id: notebook.id,
        other_notebook_id: other.id,
    }
}

/// Runs the statistical pass over every source of a notebook, the way
/// `commands::notebook::notebook_build_graph` does.
fn build(j: &Journey, notebook_id: &str) {
    for member in j.notebooks.documents(notebook_id, ALICE).unwrap() {
        let document = j
            .documents
            .get(
                &member.document_sha256,
                ALICE,
                Some(&format!("notebook:{notebook_id}")),
            )
            .unwrap()
            .expect("a notebook document must be readable by its owner");
        let pages: BTreeMap<String, u32> = document
            .chunks
            .iter()
            .map(|chunk| (chunk.id.clone(), chunk.page))
            .collect();
        let page_of = |chunk_id: &str| pages.get(chunk_id).copied().unwrap_or(0);
        let draft = extract(&document.chunks);
        j.notebooks
            .store_document_graph(
                notebook_id,
                ALICE,
                &member.document_sha256,
                &page_of,
                &draft,
                EXTRACTOR_VERSION,
                &document.extracted_at,
            )
            .expect("the graph should store");
    }
}

fn revisions(j: &Journey, notebook_id: &str) -> BTreeMap<String, String> {
    j.notebooks
        .documents(notebook_id, ALICE)
        .unwrap()
        .into_iter()
        .filter_map(|member| {
            j.documents
                .get(
                    &member.document_sha256,
                    ALICE,
                    Some(&format!("notebook:{notebook_id}")),
                )
                .ok()
                .flatten()
                .map(|document| (member.document_sha256, document.extracted_at))
        })
        .collect()
}

fn labels(j: &Journey, notebook_id: &str) -> BTreeSet<String> {
    j.notebooks
        .graph(notebook_id, ALICE, None, None, 1, 1, false)
        .unwrap()
        .nodes
        .into_iter()
        .map(|node| node.label.to_lowercase())
        .collect()
}

/// The normalised term behind a label, so a test can name a node the way the
/// store does.
fn term_for(j: &Journey, notebook_id: &str, label: &str) -> String {
    j.notebooks
        .graph(notebook_id, ALICE, None, None, 1, 1, false)
        .unwrap()
        .nodes
        .into_iter()
        .find(|node| node.label.eq_ignore_ascii_case(label))
        .map(|node| node.label.to_lowercase())
        .unwrap_or_else(|| panic!("no node labelled {label}"))
}

/// A node whose only source is the supply contract.
fn contract_only_node(j: &Journey) -> Option<crate::knowledge::graph::GraphNode> {
    j.notebooks
        .graph(&j.notebook_id, ALICE, None, None, 1, 1, false)
        .unwrap()
        .nodes
        .into_iter()
        .find(|node| node.label.to_lowercase().contains("ct-4471"))
}

// ── 1A. Source removal ───────────────────────────────────────────────────

#[test]
fn removing_a_source_removes_the_nodes_only_it_supported() {
    let j = journey("remove-nodes");
    build(&j, &j.notebook_id);

    let before = labels(&j, &j.notebook_id);
    assert!(
        before.iter().any(|label| label.contains("ct-4471")),
        "the contract number should be in the graph before removal: {before:?}"
    );

    j.notebooks
        .remove_document(&j.notebook_id, ALICE, CONTRACT_SHA)
        .expect("removal");

    let after = labels(&j, &j.notebook_id);
    assert!(
        !after.iter().any(|label| label.contains("ct-4471")),
        "the contract number came only from the removed source and must be gone: {after:?}"
    );
}

#[test]
fn an_entity_another_source_still_supports_survives_removal() {
    let j = journey("remove-shared");
    build(&j, &j.notebook_id);

    j.notebooks
        .remove_document(&j.notebook_id, ALICE, CONTRACT_SHA)
        .expect("removal");

    let after = labels(&j, &j.notebook_id);
    assert!(
        after.iter().any(|label| label.contains("northern valve")),
        "the supplier is named in the maintenance log too and must survive: {after:?}"
    );
    assert!(
        after.iter().any(|label| label.contains("pv-2201")),
        "the valve is named in three documents: {after:?}"
    );
}

#[test]
fn removing_a_source_takes_its_evidence_and_its_build_unit_with_it() {
    let j = journey("remove-evidence");
    build(&j, &j.notebook_id);

    let node = contract_only_node(&j).expect("the contract number should be a node");
    assert!(
        !j.notebooks
            .node_evidence(&j.notebook_id, ALICE, &node.id)
            .unwrap()
            .is_empty(),
        "it should have evidence before removal"
    );

    j.notebooks
        .remove_document(&j.notebook_id, ALICE, CONTRACT_SHA)
        .expect("removal");

    assert!(
        j.notebooks
            .node_evidence(&j.notebook_id, ALICE, &node.id)
            .unwrap()
            .is_empty(),
        "its evidence must go with it"
    );
    assert!(
        !j.notebooks
            .completed_units(
                &j.notebook_id,
                "statistical",
                EXTRACTOR_VERSION,
                "statistical",
                &revisions(&j, &j.notebook_id)
            )
            .unwrap()
            .contains(CONTRACT_SHA),
        "a removed source must not still count as built"
    );
}

#[test]
fn removing_a_source_from_one_notebook_leaves_the_other_alone() {
    let j = journey("remove-isolation");
    build(&j, &j.notebook_id);
    build(&j, &j.other_notebook_id);

    let other_before = labels(&j, &j.other_notebook_id);
    assert!(!other_before.is_empty(), "the second notebook has a graph");

    j.notebooks
        .remove_document(&j.notebook_id, ALICE, CONTRACT_SHA)
        .expect("removal");

    assert_eq!(
        labels(&j, &j.other_notebook_id),
        other_before,
        "the same document in another notebook is a different set of rows"
    );
    assert!(
        j.notebooks
            .documents(&j.other_notebook_id, ALICE)
            .unwrap()
            .iter()
            .any(|member| member.document_sha256 == CONTRACT_SHA),
        "its membership of the other notebook stands"
    );
    // And the file itself is untouched: a notebook groups documents, it does
    // not own them.
    assert!(j
        .documents
        .get(CONTRACT_SHA, ALICE, None)
        .unwrap()
        .is_some());
}

#[test]
fn a_removed_source_is_no_longer_eligible_for_retrieval() {
    let j = journey("remove-retrieval");
    build(&j, &j.notebook_id);

    let scope = ResearchScope {
        notebook_id: j.notebook_id.clone(),
        ..Default::default()
    };
    let resolved = notebook_retrieval::resolve(&j.notebooks, ALICE, &scope).unwrap();
    let found = notebook_retrieval::retrieve(&j.documents, ALICE, &resolved, "spares agreement");
    assert!(
        found
            .passages
            .iter()
            .any(|hit| hit.document_sha256 == CONTRACT_SHA),
        "the contract answers this question while it is in the notebook"
    );

    j.notebooks
        .remove_document(&j.notebook_id, ALICE, CONTRACT_SHA)
        .expect("removal");

    let resolved = notebook_retrieval::resolve(&j.notebooks, ALICE, &scope).unwrap();
    let found = notebook_retrieval::retrieve(&j.documents, ALICE, &resolved, "spares agreement");
    assert!(
        !found
            .passages
            .iter()
            .any(|hit| hit.document_sha256 == CONTRACT_SHA),
        "a removed source must not be retrievable"
    );
}

// ── 1B. Direction ────────────────────────────────────────────────────────

#[test]
fn a_directed_claim_survives_storage_when_its_subject_sorts_last() {
    let j = journey("direction");
    build(&j, &j.notebook_id);

    // "pv-2201" sorts after "northern valve company", so this is exactly the
    // case the old sorted-pair storage inverted.
    let subject = term_for(&j, &j.notebook_id, "PV-2201");
    let object = term_for(&j, &j.notebook_id, "Northern Valve Company");
    assert!(
        subject > object,
        "the fixture must exercise the reversing case: {subject} vs {object}"
    );

    let id = j
        .notebooks
        .upsert_assertion(
            &j.notebook_id,
            ALICE,
            &NewAssertion {
                document_sha256: Some(LOG_SHA.to_string()),
                subject: subject.clone(),
                subject_label: "PV-2201".to_string(),
                predicate: "supplied by".to_string(),
                object: object.clone(),
                object_label: "Northern Valve Company".to_string(),
                provenance: AssertionProvenance::Model,
                status: AssertionStatus::Proposed,
                direction_certain: true,
                extractor: "rebel".to_string(),
                extractor_version: 1,
                source_revision: None,
                evidence: Vec::new(),
            },
        )
        .expect("the claim should store");

    let read_back = j
        .notebooks
        .assertion(&j.notebook_id, ALICE, &id)
        .unwrap()
        .expect("the claim should read back");
    assert_eq!(read_back.subject, subject, "the subject must not move");
    assert_eq!(read_back.object, object, "the object must not move");

    // And through the graph view the canvas and the exports read.
    let view = j
        .notebooks
        .graph(&j.notebook_id, ALICE, None, None, 1, 1, false)
        .unwrap();
    let claim = view
        .edges
        .iter()
        .flat_map(|edge| edge.assertions.iter())
        .find(|assertion| assertion.id == id)
        .expect("the claim should reach the view");
    assert_eq!(claim.subject_label, "PV-2201");
    assert_eq!(claim.object_label, "Northern Valve Company");
}

#[test]
fn two_claims_about_one_pair_are_both_kept_and_marked_contested() {
    let j = journey("contested");
    build(&j, &j.notebook_id);

    let valve = term_for(&j, &j.notebook_id, "PV-2201");
    let supplier = term_for(&j, &j.notebook_id, "Northern Valve Company");

    for (document, predicate) in [(LOG_SHA, "maintained by"), (MEMO_SHA, "operator")] {
        j.notebooks
            .upsert_assertion(
                &j.notebook_id,
                ALICE,
                &NewAssertion {
                    document_sha256: Some(document.to_string()),
                    subject: valve.clone(),
                    subject_label: "PV-2201".to_string(),
                    predicate: predicate.to_string(),
                    object: supplier.clone(),
                    object_label: "Northern Valve Company".to_string(),
                    provenance: AssertionProvenance::Model,
                    status: AssertionStatus::Proposed,
                    direction_certain: true,
                    extractor: "rebel".to_string(),
                    extractor_version: 1,
                    source_revision: None,
                    evidence: Vec::new(),
                },
            )
            .expect("the claim should store");
    }

    let all = j
        .notebooks
        .assertions(&j.notebook_id, ALICE, None, false)
        .unwrap();
    assert_eq!(
        all.len(),
        2,
        "two documents disagreeing must produce two rows, not one winner"
    );

    let view = j
        .notebooks
        .graph(&j.notebook_id, ALICE, None, None, 1, 1, false)
        .unwrap();
    let edge = view
        .edges
        .iter()
        .find(|edge| !edge.assertions.is_empty())
        .expect("the pair should carry claims");
    assert!(edge.contested, "the disagreement must be visible on the edge");
    assert_eq!(
        edge.relation, None,
        "a contested link gets no single label, rather than an arbitrary winner"
    );
}

#[test]
fn a_reviewed_claim_settles_the_label_and_clears_the_conflict() {
    let j = journey("review-settles");
    build(&j, &j.notebook_id);

    let valve = term_for(&j, &j.notebook_id, "PV-2201");
    let supplier = term_for(&j, &j.notebook_id, "Northern Valve Company");
    let mut ids = Vec::new();
    for (document, predicate) in [(LOG_SHA, "maintained by"), (MEMO_SHA, "operator")] {
        ids.push(
            j.notebooks
                .upsert_assertion(
                    &j.notebook_id,
                    ALICE,
                    &NewAssertion {
                        document_sha256: Some(document.to_string()),
                        subject: valve.clone(),
                        subject_label: "PV-2201".to_string(),
                        predicate: predicate.to_string(),
                        object: supplier.clone(),
                        object_label: "Northern Valve Company".to_string(),
                        provenance: AssertionProvenance::Model,
                        status: AssertionStatus::Proposed,
                        direction_certain: true,
                        extractor: "rebel".to_string(),
                        extractor_version: 1,
                        source_revision: None,
                        evidence: Vec::new(),
                    },
                )
                .unwrap(),
        );
    }

    j.notebooks
        .review_assertion(
            &j.notebook_id,
            ALICE,
            &ids[0],
            AssertionStatus::Accepted,
            Some("the log is the newer record"),
        )
        .expect("review");

    let view = j
        .notebooks
        .graph(&j.notebook_id, ALICE, None, None, 1, 1, false)
        .unwrap();
    let edge = view
        .edges
        .iter()
        .find(|edge| !edge.assertions.is_empty())
        .expect("the pair should carry claims");
    assert!(!edge.contested, "a person has settled it");
    assert_eq!(edge.relation.as_deref(), Some("maintained by"));

    // The other reading is still inspectable, because a disagreement that is
    // deleted cannot be re-examined.
    assert_eq!(
        j.notebooks
            .assertions(&j.notebook_id, ALICE, None, true)
            .unwrap()
            .len(),
        2
    );
}

// ── 1C. Rebuild consistency ──────────────────────────────────────────────

#[test]
fn a_rebuild_permits_the_enrichment_it_invalidated() {
    let j = journey("rebuild-enrichment");
    build(&j, &j.notebook_id);

    let unit = || {
        j.notebooks
            .completed_units(
                &j.notebook_id,
                "relations",
                crate::knowledge::graph::relations::RELATION_VERSION,
                "rebel",
                &[(LOG_SHA.to_string(), "2026-01-01T00:00:00Z".to_string())]
                    .into_iter()
                    .collect(),
            )
            .unwrap()
            .contains(LOG_SHA)
    };

    // The relation pass has run over this document.
    j.notebooks
        .store_document_relations(
            &j.notebook_id,
            ALICE,
            LOG_SHA,
            &crate::knowledge::graph::relations::Verdict::default(),
            crate::knowledge::graph::relations::RELATION_VERSION,
            "2026-01-01T00:00:00Z",
        )
        .expect("relations store");
    assert!(unit(), "the relation pass records its unit");

    // Now rebuild the statistical pass over the same document, which erases
    // whatever the relation pass had added to it.
    build(&j, &j.notebook_id);

    assert!(
        !unit(),
        "a rebuild that erased the enrichment must not leave it marked done — that is the \
         state in which the work can never be redone"
    );
}

#[test]
fn a_reviewed_claim_survives_a_rebuild() {
    let j = journey("rebuild-review");
    build(&j, &j.notebook_id);

    let valve = term_for(&j, &j.notebook_id, "PV-2201");
    let supplier = term_for(&j, &j.notebook_id, "Northern Valve Company");
    let id = j
        .notebooks
        .assert_by_hand(
            &j.notebook_id,
            ALICE,
            &valve,
            "supplied by",
            &supplier,
            Some("confirmed against the purchase order"),
        )
        .expect("a person's claim");

    build(&j, &j.notebook_id);

    let after = j
        .notebooks
        .assertion(&j.notebook_id, ALICE, &id)
        .unwrap()
        .expect("a person's reviewed claim must survive a rebuild");
    assert_eq!(after.provenance, AssertionProvenance::User);
    assert_eq!(after.status, AssertionStatus::Accepted);
    assert!(!after.stale, "its terms are still in the graph");
}

#[test]
fn a_reviewed_claim_goes_stale_when_its_evidence_leaves() {
    let j = journey("stale");
    build(&j, &j.notebook_id);

    let contract_term = contract_only_node(&j)
        .expect("the contract number should be a node")
        .label
        .to_lowercase();
    let supplier = term_for(&j, &j.notebook_id, "Northern Valve Company");

    let id = j
        .notebooks
        .assert_by_hand(
            &j.notebook_id,
            ALICE,
            &contract_term,
            "supplier",
            &supplier,
            None,
        )
        .expect("a person's claim");
    assert!(!j
        .notebooks
        .assertion(&j.notebook_id, ALICE, &id)
        .unwrap()
        .unwrap()
        .stale);

    j.notebooks
        .remove_document(&j.notebook_id, ALICE, CONTRACT_SHA)
        .expect("removal");

    let after = j
        .notebooks
        .assertion(&j.notebook_id, ALICE, &id)
        .unwrap()
        .expect("the claim is kept, not deleted");
    assert!(
        after.stale,
        "the term it is about has no source left, so it must be flagged rather than shown as \
         current"
    );
}

#[test]
fn a_re_read_source_invalidates_the_unit_built_from_the_old_text() {
    let j = journey("revision");
    build(&j, &j.notebook_id);

    let current = revisions(&j, &j.notebook_id);
    assert!(
        j.notebooks
            .completed_units(
                &j.notebook_id,
                "statistical",
                EXTRACTOR_VERSION,
                "statistical",
                &current
            )
            .unwrap()
            .contains(LOG_SHA),
        "built at the revision on disk, so it is done"
    );

    // The same document, read again — a different extraction time.
    let mut moved = current.clone();
    moved.insert(LOG_SHA.to_string(), "2099-01-01T00:00:00Z".to_string());
    assert!(
        !j.notebooks
            .completed_units(
                &j.notebook_id,
                "statistical",
                EXTRACTOR_VERSION,
                "statistical",
                &moved
            )
            .unwrap()
            .contains(LOG_SHA),
        "a source re-read since the pass ran is work again, not a completed unit"
    );
}

#[test]
fn a_different_extractor_does_not_inherit_another_models_completion() {
    let j = journey("extractor-identity");
    build(&j, &j.notebook_id);
    let current = revisions(&j, &j.notebook_id);

    assert!(
        !j.notebooks
            .completed_units(
                &j.notebook_id,
                "statistical",
                EXTRACTOR_VERSION,
                "llm:some-other-model",
                &current
            )
            .unwrap()
            .contains(LOG_SHA),
        "a unit recorded by one extractor must not mark another's work done"
    );
}

// ── 2. Scope, authorisation and retrieval ────────────────────────────────

#[test]
fn another_persons_notebook_cannot_be_scoped_to() {
    let j = journey("auth-notebook");
    build(&j, &j.notebook_id);

    let scope = ResearchScope {
        notebook_id: j.notebook_id.clone(),
        ..Default::default()
    };
    assert!(
        notebook_retrieval::resolve(&j.notebooks, BOB, &scope).is_err(),
        "resolving must refuse before any source is loaded"
    );
}

#[test]
fn a_source_outside_the_notebook_is_refused_rather_than_dropped() {
    let j = journey("auth-source");
    build(&j, &j.notebook_id);

    // A real document of this person's, still in their other notebook, after
    // being removed from the one being asked about.
    j.notebooks
        .remove_document(&j.notebook_id, ALICE, CONTRACT_SHA)
        .unwrap();

    let scope = ResearchScope {
        notebook_id: j.notebook_id.clone(),
        source_sha256s: vec![LOG_SHA.to_string(), CONTRACT_SHA.to_string()],
        ..Default::default()
    };
    assert!(
        notebook_retrieval::resolve(&j.notebooks, ALICE, &scope).is_err(),
        "silently answering from one of the two selected sources would be a wrong answer that \
         looks right"
    );
}

#[test]
fn a_question_retrieves_only_from_the_selected_sources() {
    let j = journey("selected-sources");
    build(&j, &j.notebook_id);

    let scope = ResearchScope {
        notebook_id: j.notebook_id.clone(),
        source_sha256s: vec![LOG_SHA.to_string()],
        ..Default::default()
    };
    let resolved = notebook_retrieval::resolve(&j.notebooks, ALICE, &scope).unwrap();
    let found =
        notebook_retrieval::retrieve(&j.documents, ALICE, &resolved, "who maintains PV-2201");

    assert!(!found.passages.is_empty(), "the log answers this");
    assert!(
        found
            .passages
            .iter()
            .all(|hit| hit.document_sha256 == LOG_SHA),
        "nothing outside the selection may be retrieved"
    );
}

#[test]
fn a_graph_selection_reaches_the_original_passages() {
    let j = journey("graph-focus");
    build(&j, &j.notebook_id);

    let view = j
        .notebooks
        .graph(&j.notebook_id, ALICE, None, None, 1, 1, false)
        .unwrap();
    let supplier = view
        .nodes
        .iter()
        .find(|node| node.label.to_lowercase().contains("northern valve"))
        .expect("the supplier is a node");

    let scope = ResearchScope {
        notebook_id: j.notebook_id.clone(),
        node_ids: vec![supplier.id.clone()],
        ..Default::default()
    };
    let resolved = notebook_retrieval::resolve(&j.notebooks, ALICE, &scope).unwrap();
    assert!(resolved.is_graph_focused());
    assert!(
        resolved
            .focus_labels
            .iter()
            .any(|label| label.to_lowercase().contains("northern valve")),
        "the node id must resolve to the term it stands for"
    );

    let found =
        notebook_retrieval::retrieve(&j.documents, ALICE, &resolved, "who supplied the valve");
    assert!(
        found
            .passages
            .iter()
            .any(|hit| hit.text.contains("Northern Valve Company")),
        "a graph selection must pull back the source passages, not the graph's own summary"
    );
}

#[test]
fn a_notebook_question_works_before_any_graph_is_built() {
    let j = journey("no-graph");
    // Deliberately no build().

    let scope = ResearchScope {
        notebook_id: j.notebook_id.clone(),
        ..Default::default()
    };
    let resolved = notebook_retrieval::resolve(&j.notebooks, ALICE, &scope)
        .expect("a notebook with no graph is still answerable");
    let found = notebook_retrieval::retrieve(
        &j.documents,
        ALICE,
        &resolved,
        "what torque is specified for the gasket",
    );
    assert!(
        found.passages.iter().any(|hit| hit.text.contains("90 Nm")),
        "ordinary retrieval must not depend on the graph passes having run"
    );
}

#[test]
fn an_unsupported_question_returns_nothing_and_says_so() {
    let j = journey("unsupported");
    build(&j, &j.notebook_id);

    let scope = ResearchScope {
        notebook_id: j.notebook_id.clone(),
        ..Default::default()
    };
    let resolved = notebook_retrieval::resolve(&j.notebooks, ALICE, &scope).unwrap();
    let found = notebook_retrieval::retrieve(
        &j.documents,
        ALICE,
        &resolved,
        "helicopter landing pad wind loading certification",
    );

    assert!(
        found.passages.is_empty(),
        "nothing in these documents says this"
    );
    let told = notebook_retrieval::describe_scope(&resolved, &found);
    assert!(
        told.contains("Nothing in the selected sources matched"),
        "the turn must be told the sources are silent, not left to invent: {told}"
    );
    assert_eq!(
        found.sources_unused.len(),
        3,
        "all three contributed nothing"
    );
}

#[test]
fn the_retrieval_mode_is_reported_as_what_actually_ran() {
    let j = journey("mode");
    build(&j, &j.notebook_id);

    let scope = ResearchScope {
        notebook_id: j.notebook_id.clone(),
        ..Default::default()
    };
    let resolved = notebook_retrieval::resolve(&j.notebooks, ALICE, &scope).unwrap();
    let found = notebook_retrieval::retrieve(&j.documents, ALICE, &resolved, "valve");

    assert_eq!(
        found.mode,
        RetrievalMode::Keyword,
        "no embedding model is loaded here, and the mode must say so"
    );
    assert!(
        notebook_retrieval::describe_scope(&resolved, &found).contains("keyword search"),
        "the model is told how its passages were found"
    );
}

#[test]
fn partial_extraction_is_named_rather_than_hidden() {
    let j = journey("partial");
    build(&j, &j.notebook_id);

    // A source in the notebook whose extraction is not on disk.
    let missing = "f4c3d4e5f60718293a4b5c6d7e8f90a1b2c3d4e5f60718293a4b5c6d7e8f90a3";
    j.notebooks
        .add_document(&j.notebook_id, ALICE, missing, "Unread scan.pdf")
        .unwrap();

    let scope = ResearchScope {
        notebook_id: j.notebook_id.clone(),
        ..Default::default()
    };
    let resolved = notebook_retrieval::resolve(&j.notebooks, ALICE, &scope).unwrap();
    let found = notebook_retrieval::retrieve(&j.documents, ALICE, &resolved, "valve");

    assert!(
        found
            .limitations
            .iter()
            .any(|line| line.contains("Unread scan.pdf")),
        "a source that could not be read must be named: {:?}",
        found.limitations
    );
}

// ── 3. The evidence manifest ─────────────────────────────────────────────

#[test]
fn a_manifest_records_what_the_answer_could_cite() {
    let j = journey("manifest");
    build(&j, &j.notebook_id);

    let scope = ResearchScope {
        notebook_id: j.notebook_id.clone(),
        source_sha256s: vec![LOG_SHA.to_string()],
        ..Default::default()
    };
    let resolved = notebook_retrieval::resolve(&j.notebooks, ALICE, &scope).unwrap();
    let found = notebook_retrieval::retrieve(&j.documents, ALICE, &resolved, "gasket torque");
    let manifest = notebook_retrieval::manifest(
        "run-1", "conv-1", "msg-1", &scope, &resolved, &found, None,
    );

    assert!(!manifest.entries.is_empty());
    assert_eq!(
        manifest.entries[0].marker, 1,
        "markers are 1-based, like [E1]"
    );
    assert!(
        !manifest.entries[0].source_revision.is_empty(),
        "the extraction revision is what makes a citation honest later"
    );

    j.notebooks
        .record_research_turn(ALICE, &manifest)
        .expect("the manifest should store");

    let read_back = j
        .notebooks
        .research_turn(ALICE, "conv-1", "msg-1")
        .unwrap()
        .expect("it must survive the round trip");
    assert_eq!(read_back.entries.len(), manifest.entries.len());
    assert_eq!(read_back.entries[0].chunk_id, manifest.entries[0].chunk_id);
    assert_eq!(read_back.retrieval_mode, RetrievalMode::Keyword);

    // And it is this person's.
    assert!(j
        .notebooks
        .research_turn(BOB, "conv-1", "msg-1")
        .unwrap()
        .is_none());
}

#[test]
fn a_historical_manifest_does_not_move_when_the_notebook_does() {
    let j = journey("manifest-history");
    build(&j, &j.notebook_id);

    let scope = ResearchScope {
        notebook_id: j.notebook_id.clone(),
        source_sha256s: vec![CONTRACT_SHA.to_string()],
        ..Default::default()
    };
    let resolved = notebook_retrieval::resolve(&j.notebooks, ALICE, &scope).unwrap();
    let found = notebook_retrieval::retrieve(&j.documents, ALICE, &resolved, "spares");
    let manifest = notebook_retrieval::manifest(
        "run-2", "conv-2", "msg-2", &scope, &resolved, &found, None,
    );
    j.notebooks.record_research_turn(ALICE, &manifest).unwrap();

    j.notebooks
        .remove_document(&j.notebook_id, ALICE, CONTRACT_SHA)
        .unwrap();

    let read_back = j
        .notebooks
        .research_turn(ALICE, "conv-2", "msg-2")
        .unwrap()
        .expect("the record of what an answer used outlives the source");
    assert_eq!(read_back.entries.len(), manifest.entries.len());
    assert_eq!(
        read_back.entries[0].document_sha256, CONTRACT_SHA,
        "the citation still names what it cited, so the interface can say the source is gone"
    );
}

// ── 5. Notes ─────────────────────────────────────────────────────────────

#[test]
fn editing_a_saved_answer_stops_it_reading_as_verified() {
    let j = journey("notes");
    let note = j
        .notebooks
        .create_note(
            &j.notebook_id,
            ALICE,
            "What the log says",
            "PV-2201 was replaced in March [E1].",
            NoteKind::Answer,
            Some("conv-3"),
            Some("msg-3"),
            Some("{\"entries\":[]}"),
        )
        .expect("the note should save");
    assert!(!note.edited_since_saved);

    let edited = j
        .notebooks
        .update_note(
            &j.notebook_id,
            ALICE,
            &note.id,
            None,
            Some("PV-2201 was replaced in March [E1]. It will fail again by June."),
        )
        .expect("the edit should save");
    assert!(
        edited.edited_since_saved,
        "new text in a cited answer is not a source-verified claim and must not read as one"
    );
    assert!(
        edited.evidence_json.is_some(),
        "the evidence it did have is still shown"
    );
}

#[test]
fn a_written_note_is_never_reported_as_edited_away_from_evidence() {
    let j = journey("notes-written");
    let note = j
        .notebooks
        .create_note(
            &j.notebook_id,
            ALICE,
            "My working theory",
            "The valve is probably the cause.",
            NoteKind::Written,
            None,
            None,
            None,
        )
        .unwrap();
    let edited = j
        .notebooks
        .update_note(
            &j.notebook_id,
            ALICE,
            &note.id,
            None,
            Some("The valve is definitely the cause."),
        )
        .unwrap();
    assert!(
        !edited.edited_since_saved,
        "a person's own note had no verified form to diverge from"
    );
}

#[test]
fn notes_and_threads_belong_to_their_notebook_and_their_owner() {
    let j = journey("notes-isolation");
    j.notebooks
        .create_note(
            &j.notebook_id,
            ALICE,
            "Only mine",
            "body",
            NoteKind::Written,
            None,
            None,
            None,
        )
        .unwrap();

    assert_eq!(j.notebooks.notes(&j.notebook_id, ALICE).unwrap().len(), 1);
    assert!(j.notebooks.notes(&j.notebook_id, BOB).unwrap().is_empty());
    assert!(j
        .notebooks
        .notes(&j.other_notebook_id, ALICE)
        .unwrap()
        .is_empty());

    j.notebooks
        .bind_conversation(&j.notebook_id, ALICE, "conv-9")
        .unwrap();
    assert_eq!(
        j.notebooks
            .notebook_conversations(&j.notebook_id, ALICE)
            .unwrap(),
        vec!["conv-9".to_string()]
    );
    assert!(j
        .notebooks
        .notebook_conversations(&j.other_notebook_id, ALICE)
        .unwrap()
        .is_empty());
    assert!(j
        .notebooks
        .notebook_conversations(&j.notebook_id, BOB)
        .unwrap()
        .is_empty());
    assert_eq!(
        j.notebooks
            .conversation_notebook(ALICE, "conv-9")
            .unwrap()
            .as_deref(),
        Some(j.notebook_id.as_str())
    );
}

// ── 6. Review persistence ────────────────────────────────────────────────

#[test]
fn a_correction_changes_the_direction_and_marks_it_the_persons_own() {
    let j = journey("correction");
    build(&j, &j.notebook_id);

    let valve = term_for(&j, &j.notebook_id, "PV-2201");
    let supplier = term_for(&j, &j.notebook_id, "Northern Valve Company");
    // The model gets it backwards: the company is not supplied by the valve.
    let wrong = j
        .notebooks
        .upsert_assertion(
            &j.notebook_id,
            ALICE,
            &NewAssertion {
                document_sha256: Some(LOG_SHA.to_string()),
                subject: supplier.clone(),
                subject_label: "Northern Valve Company".to_string(),
                predicate: "supplied by".to_string(),
                object: valve.clone(),
                object_label: "PV-2201".to_string(),
                provenance: AssertionProvenance::Model,
                status: AssertionStatus::Proposed,
                direction_certain: true,
                extractor: "rebel".to_string(),
                extractor_version: 1,
                source_revision: None,
                evidence: Vec::new(),
            },
        )
        .unwrap();

    let corrected_id = j
        .notebooks
        .correct_assertion(
            &j.notebook_id,
            ALICE,
            &wrong,
            &valve,
            "supplied by",
            &supplier,
            Some("the other way round"),
        )
        .expect("the correction should save");

    let corrected = j
        .notebooks
        .assertion(&j.notebook_id, ALICE, &corrected_id)
        .unwrap()
        .unwrap();
    assert_eq!(corrected.subject, valve);
    assert_eq!(corrected.object, supplier);
    assert_eq!(corrected.provenance, AssertionProvenance::User);
    assert_eq!(corrected.status, AssertionStatus::Accepted);
    assert!(corrected.direction_certain);

    // The wrong row is gone, not left beside the right one.
    assert!(j
        .notebooks
        .assertion(&j.notebook_id, ALICE, &wrong)
        .unwrap()
        .is_none());
}

#[test]
fn a_model_proposal_cannot_be_deleted_only_rejected() {
    let j = journey("no-delete");
    build(&j, &j.notebook_id);

    let valve = term_for(&j, &j.notebook_id, "PV-2201");
    let supplier = term_for(&j, &j.notebook_id, "Northern Valve Company");
    let id = j
        .notebooks
        .upsert_assertion(
            &j.notebook_id,
            ALICE,
            &NewAssertion {
                document_sha256: Some(LOG_SHA.to_string()),
                subject: valve,
                subject_label: "PV-2201".to_string(),
                predicate: "supplied by".to_string(),
                object: supplier,
                object_label: "Northern Valve Company".to_string(),
                provenance: AssertionProvenance::Model,
                status: AssertionStatus::Proposed,
                direction_certain: true,
                extractor: "rebel".to_string(),
                extractor_version: 1,
                source_revision: None,
                evidence: Vec::new(),
            },
        )
        .unwrap();

    assert!(j
        .notebooks
        .delete_hand_assertion(&j.notebook_id, ALICE, &id)
        .is_err());
    assert!(j
        .notebooks
        .assertion(&j.notebook_id, ALICE, &id)
        .unwrap()
        .is_some());
}

#[test]
fn one_person_cannot_review_anothers_relationships() {
    let j = journey("review-auth");
    build(&j, &j.notebook_id);

    let valve = term_for(&j, &j.notebook_id, "PV-2201");
    let supplier = term_for(&j, &j.notebook_id, "Northern Valve Company");
    let id = j
        .notebooks
        .assert_by_hand(&j.notebook_id, ALICE, &valve, "supplied by", &supplier, None)
        .unwrap();

    assert!(j
        .notebooks
        .review_assertion(&j.notebook_id, BOB, &id, AssertionStatus::Rejected, None)
        .is_err());
    assert!(j
        .notebooks
        .assertions(&j.notebook_id, BOB, None, true)
        .unwrap()
        .is_empty());
}

#[test]
fn a_rejected_claim_is_not_offered_as_evidence() {
    let j = journey("rejected-evidence");
    build(&j, &j.notebook_id);

    let valve = term_for(&j, &j.notebook_id, "PV-2201");
    let supplier = term_for(&j, &j.notebook_id, "Northern Valve Company");
    let id = j
        .notebooks
        .assert_by_hand(&j.notebook_id, ALICE, &valve, "supplied by", &supplier, None)
        .unwrap();
    j.notebooks
        .review_assertion(
            &j.notebook_id,
            ALICE,
            &id,
            AssertionStatus::Rejected,
            Some("wrong supplier"),
        )
        .unwrap();

    let scope = ResearchScope {
        notebook_id: j.notebook_id.clone(),
        assertion_ids: vec![id],
        ..Default::default()
    };
    let resolved = notebook_retrieval::resolve(&j.notebooks, ALICE, &scope).unwrap();
    assert!(
        resolved.focus_assertions.is_empty(),
        "a claim a person refused must not steer an answer"
    );
    assert!(
        resolved
            .limitations
            .iter()
            .any(|line| line.contains("rejected on review")),
        "and the turn is told why: {:?}",
        resolved.limitations
    );
}

#[test]
fn a_persons_own_claim_is_described_to_the_model_as_unverified() {
    let j = journey("hand-claim-described");
    build(&j, &j.notebook_id);

    let valve = term_for(&j, &j.notebook_id, "PV-2201");
    let supplier = term_for(&j, &j.notebook_id, "Northern Valve Company");
    let id = j
        .notebooks
        .assert_by_hand(&j.notebook_id, ALICE, &valve, "supplied by", &supplier, None)
        .unwrap();

    let scope = ResearchScope {
        notebook_id: j.notebook_id.clone(),
        assertion_ids: vec![id],
        ..Default::default()
    };
    let resolved = notebook_retrieval::resolve(&j.notebooks, ALICE, &scope).unwrap();
    let found = notebook_retrieval::retrieve(&j.documents, ALICE, &resolved, "who supplied it");
    let told = notebook_retrieval::describe_scope(&resolved, &found);

    assert!(
        told.contains("It is a claim about the sources, not a source itself"),
        "an unverified assertion must not be handed over as if it were evidence: {told}"
    );
}

