//! The whole notebook journey, over the real stores.
//!
//! Every other test in this feature exercises one piece against fixtures. This
//! one walks the path a person actually takes — create a notebook, put two
//! documents in it, build the graph, find the supplier they share, ask what it
//! came from, select it, render it for a chat — using the real
//! [`DocumentStore`] writing real JSON to a real directory and the real
//! [`NotebookStore`] writing to a real SQLite file.
//!
//! It exists because the pieces can each be correct while the seams are not.
//! The two that would not show up anywhere else:
//!
//! - A notebook's documents are recorded under the synthetic conversation id
//!   `notebook:<id>`, and the build reads them back with `conversation_id:
//!   None`. If those two ever disagree the graph is silently empty, every unit
//!   test still passes, and the screen says the documents contained nothing.
//! - Chunk ids are minted by the chunker, stored as graph provenance, and
//!   looked up again when rendering citations. Three modules have to agree on
//!   one string.

use std::collections::BTreeMap;

use crate::agent_runtime::documents::{DocumentStore, NewExtraction, Sighting};
use crate::knowledge::graph::statistical::{extract, EXTRACTOR_VERSION};
use crate::knowledge::graph::{render, NotebookStore};

const ALICE: &str = "user-alice";
const BOB: &str = "user-bob";

/// Content addresses for the two fixtures.
///
/// Real 64-character hex, because `DocumentStore::record` refuses "a document id
/// that is not a sha-256 hash" — a check worth having, and one only a test
/// going through the real store would ever meet.
const LOG_SHA: &str = "a1b2c3d4e5f60718293a4b5c6d7e8f90a1b2c3d4e5f60718293a4b5c6d7e8f90";
const CONTRACT_SHA: &str = "b2c3d4e5f60718293a4b5c6d7e8f90a1b2c3d4e5f60718293a4b5c6d7e8f90a1";

/// The extraction revision the relation pass records against these fixtures.
const REVISION: &str = "2026-01-01T00:00:00Z";

/// What `completed_units` compares a recorded unit against here.
///
/// A unit whose recorded revision does not match what the caller says is on
/// disk is treated as out of date — the safe direction — so these tests name
/// the revision they wrote.
fn revisions() -> BTreeMap<String, String> {
    [
        (LOG_SHA.to_string(), REVISION.to_string()),
        (CONTRACT_SHA.to_string(), REVISION.to_string()),
    ]
    .into_iter()
    .collect()
}

/// A maintenance log and a contract that name the same supplier.
///
/// The point of the whole feature is that this connection is invisible when the
/// two files are read separately.
fn pump_log() -> BTreeMap<u32, String> {
    let mut pages = BTreeMap::new();
    pages.insert(
        1,
        "Maintenance Report — Unit Four\n\n\
         Valve PV-2201 was replaced during the March outage. Northern Valve Company \
         attended site and completed the work within the shift.\n\n\
         Pressure transmitter PT-2201 was recalibrated at the same time. Northern \
         Valve Company supplied the replacement cartridge for PV-2201."
            .to_string(),
    );
    pages.insert(
        2,
        "Follow-up — Unit Four\n\n\
         PV-2201 has held setpoint since the replacement. Northern Valve Company \
         will return in October for the scheduled inspection of PV-2201."
            .to_string(),
    );
    pages
}

fn supply_contract() -> BTreeMap<u32, String> {
    let mut pages = BTreeMap::new();
    pages.insert(
        1,
        "Supply Agreement\n\n\
         This agreement is between the operator and Northern Valve Company for the \
         supply of control valves to Unit Four.\n\n\
         Northern Valve Company shall hold spares for PV-2201 and equivalent \
         assemblies for the term of this agreement."
            .to_string(),
    );
    pages.insert(
        2,
        "Schedule B\n\n\
         Northern Valve Company is the sole approved supplier for Unit Four control \
         valves. Escalation contact is the Maintenance Department."
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
            // Exactly what `commands::notebook::notebook_add_documents` writes.
            // If this drifts from the command, the build below reads nothing.
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
        .expect("the extraction should be stored");
}

/// Builds the statistical graph the way `notebook_build_graph` does.
fn build_graph(documents: &DocumentStore, notebooks: &NotebookStore, notebook_id: &str) {
    for member in notebooks.documents(notebook_id, ALICE).unwrap() {
        let document = documents
            .get(&member.document_sha256, ALICE, None)
            .expect("the store should answer")
            .expect("a notebook document must be readable by its owner");

        let pages: BTreeMap<String, u32> = document
            .chunks
            .iter()
            .map(|chunk| (chunk.id.clone(), chunk.page))
            .collect();
        let page_of = |chunk_id: &str| pages.get(chunk_id).copied().unwrap_or(0);

        let draft = extract(&document.chunks);
        notebooks
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

struct Journey {
    documents: DocumentStore,
    notebooks: NotebookStore,
    notebook_id: String,
    _dir: tempfile::TempDir,
}

fn journey() -> Journey {
    let dir = tempfile::tempdir().expect("a temp dir");
    let documents = DocumentStore::open(dir.path()).expect("the document store should open");
    let notebooks = NotebookStore::open(dir.path()).expect("the notebook store should open");

    let notebook = notebooks.create(ALICE, "Unit Four").expect("create");
    record(&documents, &notebook.id, LOG_SHA, "pump.pdf", pump_log());
    record(
        &documents,
        &notebook.id,
        CONTRACT_SHA,
        "contract.pdf",
        supply_contract(),
    );
    notebooks
        .add_document(&notebook.id, ALICE, LOG_SHA, "pump.pdf")
        .unwrap();
    notebooks
        .add_document(&notebook.id, ALICE, CONTRACT_SHA, "contract.pdf")
        .unwrap();

    Journey {
        documents,
        notebooks,
        notebook_id: notebook.id,
        _dir: dir,
    }
}

/// The relation pass, end to end, over the real stores.
///
/// Everything except the model itself is real here: the real `DocumentStore`,
/// the real SQLite `NotebookStore`, the real statistical pass, the real five
/// gates, the real `UPDATE`-only persistence, the real graph read and the real
/// Mermaid renderer. The one thing standing in for a live process is the list
/// of triplets, and those are not invented either - they are the shape
/// Babelscape/rebel-large actually returned on this machine, including the
/// clipped entity name and the off-domain geographic claim.
///
/// It exists because every piece of this can be correct while the seams are
/// not, and the seams are where this feature can fail silently:
///
/// - `verify` addresses relations by *normalised* term, and `graph_edges`
///   stores `(source, target)` in the order the statistical pass minted them.
///   If those two disagree, every gate passes, the `UPDATE` matches no row, and
///   the graph comes back with every edge still unnamed. No unit test would
///   notice: the verdict would be full and the write would be a no-op.
/// - The renderer reads labels, not normalised terms. A relation stored under
///   the right key and rendered against the wrong one draws an unlabelled
///   graph.
#[test]
fn a_relation_survives_from_the_model_to_the_drawn_diagram() {
    use super::relations::{self, Triplet};

    let j = journey();
    build_graph(&j.documents, &j.notebooks, &j.notebook_id);

    // What the statistical pass actually found, and what it is willing to have
    // named. The relation pass is handed exactly this and nothing more.
    let terms: BTreeMap<String, String> = j
        .notebooks
        .document_terms(&j.notebook_id, ALICE, LOG_SHA)
        .unwrap()
        .into_iter()
        .collect();
    let edges = j
        .notebooks
        .document_edges(&j.notebook_id, ALICE, LOG_SHA)
        .unwrap();
    assert!(
        !edges.is_empty(),
        "the statistical pass found no edges, so there is nothing to name"
    );

    // The passages, read back the way the command reads them.
    let document = j.documents.get(LOG_SHA, ALICE, None).unwrap().unwrap();
    let chunks: BTreeMap<String, String> = document
        .chunks
        .iter()
        .map(|chunk| (chunk.id.clone(), chunk.text.clone()))
        .collect();
    let first_chunk = document.chunks[0].id.clone();

    let proposals = vec![
        // The good one. Note the clipped subject: REBEL returned "Acme Pumps"
        // for "Acme Pumps Ltd" on this machine, so the resolver has to close
        // that gap or a correct reading is discarded as a fabrication.
        Triplet {
            chunk_id: first_chunk.clone(),
            subject: "Northern Valve".to_string(),
            relation: "manufacturer".to_string(),
            object: "PV-2201".to_string(),
        },
        // The measured failure: both entities are really in the passage and
        // really are an edge, so only the allowlist stops it.
        Triplet {
            chunk_id: first_chunk.clone(),
            subject: "PV-2201".to_string(),
            relation: "located in the administrative territorial entity".to_string(),
            object: "Northern Valve Company".to_string(),
        },
    ];

    let verdict = relations::verify(&proposals, &terms, &chunks, &edges);
    assert_eq!(verdict.stats.kept, 1, "{:?}", verdict.stats);
    assert_eq!(verdict.stats.dropped_offdomain, 1, "{:?}", verdict.stats);

    j.notebooks
        .store_document_relations(
            &j.notebook_id,
            ALICE,
            LOG_SHA,
            &verdict,
            relations::RELATION_VERSION,
                REVISION,
        )
        .expect("relations store");

    // Read back through the ordinary graph path - the one the screen and the
    // chat tool both use. This is the assertion that catches a verdict written
    // under a key nothing reads.
    let view = j
        .notebooks
        .graph(&j.notebook_id, ALICE, None, None, 1, 1, false)
        .unwrap();
    let named: Vec<&str> = view
        .edges
        .iter()
        .filter_map(|edge| edge.relation.as_deref())
        .collect();
    assert_eq!(
        named,
        vec!["manufacturer"],
        "the relation did not survive the write and read back"
    );

    // And finally drawn, because a relation that reaches the database but not
    // the diagram has not reached the person who asked.
    let label_of = |id: &str| {
        view.nodes
            .iter()
            .find(|node| node.id == id)
            .map(|node| node.label.clone())
    };
    let nodes: Vec<super::render::RenderNode> = view
        .nodes
        .iter()
        .map(|node| (node.label.clone(), node.node_type.clone(), node.occurrences))
        .collect();
    let render_edges: Vec<super::render::RenderEdge> = view
        .edges
        .iter()
        .filter_map(|edge| {
            Some((
                label_of(&edge.source)?,
                label_of(&edge.target)?,
                edge.weight,
                edge.relation.clone(),
            ))
        })
        .collect();

    let diagram = super::render::render_mermaid(&nodes, &render_edges);
    assert!(
        diagram.contains("manufacturer"),
        "the named relation is missing from the diagram: {diagram}"
    );
    assert!(
        diagram.contains("-->"),
        "a named relation must draw a solid arrow: {diagram}"
    );

    // Every link is drawn as what it is. This fixture's single edge got named,
    // so there is nothing dashed left to check - which is exactly why the
    // assertion is written as an invariant over the edges rather than as a
    // search for a string. A test that looked for "together in" here would pass
    // only by accident of how many edges the fixture happens to produce.
    let unnamed = render_edges
        .iter()
        .filter(|(.., relation)| relation.is_none())
        .count();
    assert_eq!(
        diagram.matches("-.->").count(),
        unnamed,
        "every unnamed link must be drawn dashed, and no named one may be: {diagram}"
    );
    assert_eq!(
        diagram.matches("-->").count() - diagram.matches("-.->").count(),
        render_edges.len() - unnamed,
        "every named relation must be drawn as a solid arrow: {diagram}"
    );
}

/// Running the pass twice must not change the graph.
///
/// `graph_build_units` is what makes an interrupted pass resumable, and the
/// storage is `UPDATE`-only, so a second run over the same document has to be a
/// no-op rather than a second opinion. An edge whose label depends on how many
/// times the pass ran is not reproducible, and reproducibility is the claim this
/// repository makes about its evidence.
#[test]
fn naming_the_same_document_twice_produces_the_same_graph() {
    use super::relations::{self, Triplet};

    let j = journey();
    build_graph(&j.documents, &j.notebooks, &j.notebook_id);

    let terms: BTreeMap<String, String> = j
        .notebooks
        .document_terms(&j.notebook_id, ALICE, LOG_SHA)
        .unwrap()
        .into_iter()
        .collect();
    let edges = j
        .notebooks
        .document_edges(&j.notebook_id, ALICE, LOG_SHA)
        .unwrap();
    let document = j.documents.get(LOG_SHA, ALICE, None).unwrap().unwrap();
    let chunks: BTreeMap<String, String> = document
        .chunks
        .iter()
        .map(|chunk| (chunk.id.clone(), chunk.text.clone()))
        .collect();

    let proposals = vec![Triplet {
        chunk_id: document.chunks[0].id.clone(),
        subject: "Northern Valve Company".to_string(),
        relation: "manufacturer".to_string(),
        object: "PV-2201".to_string(),
    }];

    let run = || {
        let verdict = relations::verify(&proposals, &terms, &chunks, &edges);
        j.notebooks
            .store_document_relations(
                &j.notebook_id,
                ALICE,
                LOG_SHA,
                &verdict,
                relations::RELATION_VERSION,
                REVISION,
            )
            .expect("relations store");
        j.notebooks
            .graph(&j.notebook_id, ALICE, None, None, 1, 1, false)
            .unwrap()
    };

    let first = run();
    let second = run();

    assert_eq!(first.total_edges, second.total_edges);
    assert_eq!(first.total_terms, second.total_terms);
    let relations_of = |view: &super::GraphView| -> Vec<Option<String>> {
        let mut all: Vec<Option<String>> =
            view.edges.iter().map(|edge| edge.relation.clone()).collect();
        all.sort();
        all
    };
    assert_eq!(relations_of(&first), relations_of(&second));

    // And the unit is recorded, so the command skips this document next time
    // rather than paying for the model again to reach the same answer.
    let done = j
        .notebooks
        .completed_units(&j.notebook_id, "relations", relations::RELATION_VERSION, "rebel", &revisions())
        .unwrap();
    assert!(done.contains(LOG_SHA));
}

#[test]
fn a_notebook_finds_the_supplier_two_documents_share() {
    let j = journey();
    build_graph(&j.documents, &j.notebooks, &j.notebook_id);

    let view = j
        .notebooks
        .graph(&j.notebook_id, ALICE, None, None, 1, 1, false)
        .unwrap();
    assert!(view.total_nodes > 0, "the build produced no graph at all");

    let supplier = view
        .nodes
        .iter()
        .find(|node| node.label.contains("Northern Valve"))
        .expect("the supplier named in both documents should be a node");

    // The point of the feature: this node's evidence spans *both* files, which
    // is the fact neither document states on its own.
    let evidence = j
        .notebooks
        .node_evidence(&j.notebook_id, ALICE, &supplier.id)
        .unwrap();
    let sources: std::collections::BTreeSet<&str> = evidence
        .iter()
        .map(|row| row.document_sha256.as_str())
        .collect();
    assert!(
        sources.contains(LOG_SHA) && sources.contains(CONTRACT_SHA),
        "the supplier should be cited from both documents, got {sources:?}"
    );
}

#[test]
fn the_shared_supplier_is_linked_to_the_equipment_both_documents_mention() {
    let j = journey();
    build_graph(&j.documents, &j.notebooks, &j.notebook_id);

    let view = j
        .notebooks
        .graph(&j.notebook_id, ALICE, None, None, 1, 1, false)
        .unwrap();
    let supplier = view
        .nodes
        .iter()
        .find(|node| node.label.contains("Northern Valve"))
        .expect("supplier");
    let valve = view
        .nodes
        .iter()
        .find(|node| node.label == "PV-2201")
        .expect("the tag should be a node");

    assert!(
        view.edges.iter().any(|edge| {
            (edge.source == supplier.id && edge.target == valve.id)
                || (edge.source == valve.id && edge.target == supplier.id)
        }),
        "the supplier and the valve share passages and should be linked"
    );
}

#[test]
fn every_chunk_the_graph_cites_can_be_resolved_back_to_a_page() {
    // Three modules mint, store and look up the same chunk id. If they drift,
    // citations silently become "page 0".
    let j = journey();
    build_graph(&j.documents, &j.notebooks, &j.notebook_id);

    let view = j
        .notebooks
        .graph(&j.notebook_id, ALICE, None, None, 1, 1, false)
        .unwrap();
    let mut checked = 0;
    for node in &view.nodes {
        for row in j
            .notebooks
            .node_evidence(&j.notebook_id, ALICE, &node.id)
            .unwrap()
        {
            assert!(
                row.page > 0,
                "{} cites {} with no page — the chunk id did not resolve",
                node.label,
                row.chunk_id
            );
            checked += 1;
        }
    }
    assert!(checked > 0, "nothing was cited at all");
}

#[test]
fn the_local_view_narrows_to_the_supplier_without_losing_the_total() {
    let j = journey();
    build_graph(&j.documents, &j.notebooks, &j.notebook_id);

    let whole = j
        .notebooks
        .graph(&j.notebook_id, ALICE, None, None, 1, 1, false)
        .unwrap();
    let supplier = whole
        .nodes
        .iter()
        .find(|node| node.label.contains("Northern Valve"))
        .expect("supplier");

    let local = j
        .notebooks
        .graph(&j.notebook_id, ALICE, None, Some(&supplier.id), 1, 1, false)
        .unwrap();
    assert!(local.nodes.iter().any(|node| node.id == supplier.id));
    assert!(local.nodes.len() <= whole.nodes.len());
    // "showing N of M" stays truthful in the narrowed view.
    assert_eq!(local.total_nodes, whole.total_nodes);
}

#[test]
fn a_selection_renders_into_text_a_chat_turn_can_carry() {
    let j = journey();
    build_graph(&j.documents, &j.notebooks, &j.notebook_id);

    let view = j
        .notebooks
        .graph(&j.notebook_id, ALICE, None, None, 1, 1, false)
        .unwrap();
    let chosen: Vec<_> = view
        .nodes
        .iter()
        .filter(|node| node.label.contains("Northern Valve") || node.label == "PV-2201")
        .collect();
    assert_eq!(chosen.len(), 2, "both terms should be in the graph");

    let names: BTreeMap<String, String> = [
        (LOG_SHA.to_string(), "pump.pdf".to_string()),
        (CONTRACT_SHA.to_string(), "contract.pdf".to_string()),
    ]
    .into_iter()
    .collect();

    let ids: std::collections::BTreeSet<&String> = chosen.iter().map(|node| &node.id).collect();
    let nodes: Vec<render::RenderNode> = chosen
        .iter()
        .map(|node| (node.label.clone(), node.node_type.clone(), node.occurrences))
        .collect();
    let edges: Vec<render::RenderEdge> = view
        .edges
        .iter()
        .filter(|edge| ids.contains(&edge.source) && ids.contains(&edge.target))
        .map(|edge| {
            let label = |id: &str| {
                chosen
                    .iter()
                    .find(|node| node.id == id)
                    .map(|node| node.label.clone())
                    .unwrap_or_default()
            };
            (
                label(&edge.source),
                label(&edge.target),
                edge.weight,
                edge.relation.clone(),
            )
        })
        .collect();
    let citations: Vec<render::RenderCitation> = chosen
        .iter()
        .map(|node| {
            (
                node.label.clone(),
                j.notebooks
                    .node_evidence(&j.notebook_id, ALICE, &node.id)
                    .unwrap()
                    .into_iter()
                    .map(|row| {
                        (
                            names.get(&row.document_sha256).cloned().unwrap_or_default(),
                            row.page,
                        )
                    })
                    .collect(),
            )
        })
        .collect();

    let markdown = render::render_markdown("Unit Four", &nodes, &edges, &citations);

    assert!(markdown.contains("Northern Valve"));
    assert!(markdown.contains("PV-2201"));
    // Both files are named, which is the whole reason to import this into a chat.
    assert!(markdown.contains("pump.pdf"));
    assert!(markdown.contains("contract.pdf"));
    // Untyped until the model pass runs, and it says so rather than guessing.
    assert!(markdown.contains("type not determined"));
    assert!(markdown.contains("co-occurrence, not a stated"));
}

#[test]
fn rebuilding_the_whole_notebook_changes_nothing() {
    let j = journey();
    build_graph(&j.documents, &j.notebooks, &j.notebook_id);
    let first = j
        .notebooks
        .graph(&j.notebook_id, ALICE, None, None, 1, 1, false)
        .unwrap();

    build_graph(&j.documents, &j.notebooks, &j.notebook_id);
    let second = j
        .notebooks
        .graph(&j.notebook_id, ALICE, None, None, 1, 1, false)
        .unwrap();

    assert_eq!(
        first, second,
        "a rebuild must be idempotent, counts included"
    );
}

#[test]
fn a_second_person_sees_none_of_it() {
    let j = journey();
    build_graph(&j.documents, &j.notebooks, &j.notebook_id);

    assert!(j.notebooks.list(BOB).unwrap().is_empty());
    assert!(j
        .notebooks
        .documents(&j.notebook_id, BOB)
        .unwrap()
        .is_empty());
    assert_eq!(
        j.notebooks
            .graph(&j.notebook_id, BOB, None, None, 1, 1, false)
            .unwrap()
            .total_nodes,
        0
    );
    // And the documents themselves are invisible, not merely the graph over them.
    assert!(j.documents.get(LOG_SHA, BOB, None).unwrap().is_none());
}
