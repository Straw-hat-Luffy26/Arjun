//! P04, on the production path: every call here goes `authorize` → `execute`
//! exactly as a model's tool call does, against a real artifact store, a real
//! event log and a real memory graph that resolves receipts against it.
//!
//! | Property the plan names | Test |
//! |---|---|
//! | exact bytes survive another run, another model and a restart | [`a_produced_note_survives_another_run_another_model_and_a_restart`] |
//! | scoped reads refuse other users' artifacts | [`another_person_cannot_reach_an_artifact_through_any_of_the_tools`] |
//! | corrupted and wrong-type files fail | [`a_corrupted_or_mistyped_file_fails_validation_and_is_never_published`] |
//! | changed evidence marks the output stale | [`a_correction_to_what_a_note_cites_makes_the_note_stale_and_unpublishable`] |
//! | targeted edits preserve unrelated content | [`an_edit_through_the_tool_changes_one_unit_and_copies_every_other_part`] |
//! | the same effect key does not publish twice | [`the_same_effect_key_publishes_once_across_runs`] |
//! | a real render, acceptance and publication — or an honest unavailable | [`a_sound_note_is_rendered_accepted_and_published_or_reported_unavailable`] |

use std::io::Read;
use std::sync::Arc;

use serde_json::{json, Value};

use super::*;
use crate::artifacts::conversation_store::{ArtifactKind, ConversationArtifacts, NewArtifact, Producer, Stage, VersionMeta};
use crate::identity::{Role, Session, User};
use crate::knowledge::graph::runtime_memory::{ItemStatus, MemoryItem, MemoryKind, Provenance};
use crate::knowledge::graph::runtime_store::MemoryGraph;

const OWNER: &str = "priya";
const CONVERSATION: &str = "c-p04";
/// A later turn in the same conversation, served by a different model.
const LATER: &str = "r-later";

fn session_of(id: &str) -> Arc<std::sync::RwLock<Option<Session>>> {
    Arc::new(std::sync::RwLock::new(Some(Session::open(User::new(id, "A Person", vec![Role::Employee])))))
}

fn reviewer() -> Session {
    Session::open(User::new("ravi", "Ravi Menon", vec![Role::Administrator]))
}

/// Everything `deps_with` builds, with a memory graph that resolves receipts
/// against the same event log, a store of the caller's choosing and a session
/// of the caller's choosing.
fn rebuilt(
    base: &Arc<RuntimeDeps>,
    graph: Option<Arc<MemoryGraph>>,
    store: Arc<ConversationArtifacts>,
    session: Arc<std::sync::RwLock<Option<Session>>>,
) -> Arc<RuntimeDeps> {
    Arc::new(RuntimeDeps {
        memory_graph: graph,
        conversation_artifacts: store,
        registry: base.registry.clone(),
        index: base.index.clone(),
        session,
        workspaces: base.workspaces.clone(),
        approvals: base.approvals.clone(),
        calculations: base.calculations.clone(),
        passages: base.passages.clone(),
        produced: base.produced.clone(),
        calls: base.calls.clone(),
        plans: base.plans.clone(),
        events: base.events.clone(),
        skills: base.skills.clone(),
        hooks: base.hooks.clone(),
        memory: base.memory.clone(),
        checkpoints: base.checkpoints.clone(),
        emit: base.emit.clone(),
        emit_durable: base.emit_durable.clone(),
        subagents: base.subagents.clone(),
        multimodal: base.multimodal.clone(),
        audit_health: base.audit_health.clone(),
        documents: base.documents.clone(),
        run_to_conversation: base.run_to_conversation.clone(),
        notebooks: base.notebooks.clone(),
    })
}

struct Harness {
    deps: Arc<RuntimeDeps>,
    base: Arc<RuntimeDeps>,
    graph: Arc<MemoryGraph>,
    dir: tempfile::TempDir,
}

impl Harness {
    fn new() -> Self {
        let (base, dir) = super::tests::deps_with(session_of(OWNER));
        let graph = Arc::new(MemoryGraph::in_memory().expect("a graph").with_receipts(base.events.clone()));
        let deps = rebuilt(&base, Some(graph.clone()), base.conversation_artifacts.clone(), base.session.clone());
        deps.run_to_conversation.bind("r", CONVERSATION);
        add_run(&deps, &dir, LATER, CONVERSATION);
        Harness { deps, base, graph, dir }
    }

    /// The same deps, as somebody else.
    fn as_user(&self, user: &str, run: &str, conversation: &str) -> Arc<RuntimeDeps> {
        let deps = rebuilt(&self.base, Some(self.graph.clone()), self.deps.conversation_artifacts.clone(), session_of(user));
        add_run(&deps, &self.dir, run, conversation);
        deps
    }
}

fn add_run(deps: &Arc<RuntimeDeps>, dir: &tempfile::TempDir, run: &str, conversation: &str) {
    deps.plans.lock().unwrap().insert(
        run.to_string(),
        crate::orchestrator::plan::PlanRun::new(
            run,
            vec!["work on the note".to_string()],
            crate::orchestrator::plan::Budget::standard(ToolName::ALL.to_vec()),
        ),
    );
    deps.workspaces
        .lock()
        .unwrap()
        .insert(run.to_string(), workspace::Workspace::create(dir.path(), run).expect("workspace"));
    deps.run_to_conversation.bind(run, conversation);
}

/// Authorise, then spend the grant — a read-only or automatic call.
async fn call(deps: &Arc<RuntimeDeps>, run: &str, tool: &str, args: Value) -> Result<String, String> {
    call_with(deps, json!({ "runId": run, "toolCallId": format!("tc-{tool}-{}", uuid::Uuid::new_v4()), "tool": tool, "args": args })).await
}

async fn call_with(deps: &Arc<RuntimeDeps>, request: Value) -> Result<String, String> {
    let allow = authorize(request.clone(), deps).await.map_err(|e| format!("authorise: {}", e.message))?;
    spend(deps, request, allow).await
}

/// Authorise a call a person must approve, by being that person.
async fn approved(deps: &Arc<RuntimeDeps>, run: &str, tool: &str, args: Value) -> Result<String, String> {
    let request = json!({ "runId": run, "toolCallId": format!("tc-{tool}-{}", uuid::Uuid::new_v4()), "tool": tool, "args": args });
    let queue = deps.approvals.clone();
    let waiting = tokio::spawn({
        let deps = deps.clone();
        let request = request.clone();
        async move { authorize(request, &deps).await }
    });
    let item = loop {
        if let Some(item) = queue.pending().first().cloned() {
            break item;
        }
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    };
    queue.decide(&reviewer(), &item.request.id, true, None).expect("approved");
    let allow = waiting.await.expect("finished").map_err(|e| format!("authorise: {}", e.message))?;
    spend(deps, request, allow).await
}

async fn spend(deps: &Arc<RuntimeDeps>, request: Value, allow: Value) -> Result<String, String> {
    let grant = allow.get("grant").and_then(Value::as_str).ok_or_else(|| format!("refused: {allow}"))?.to_string();
    let mut spent = request;
    spent["grant"] = json!(grant);
    let result = execute(spent, deps).await.map_err(|e| format!("execute: {}", e.message))?;
    Ok(result["text"].as_str().unwrap_or_default().to_string())
}

/// A shared-memory fact a note can cite.
fn fact(graph: &MemoryGraph) -> MemoryItem {
    let mut item = crate::knowledge::graph::runtime_memory::tests::item(
        MemoryKind::Fact,
        Provenance::Operator { user_id: "ravi".into() },
    );
    item.content = "The minimum allowable wall thickness is 9.0 mm while pitting is recorded.".into();
    let committed = graph.commit(item.clone(), None, &[]).expect("the fact is committed");
    assert_eq!(committed.status, ItemStatus::Admitted);
    item
}

fn note_args(cites: &str) -> Value {
    json!({
        "path": "note.docx",
        "title": "Shell thickness at point C",
        "sections": [
            {"heading": "Findings", "level": 1, "blocks": [
                {"kind": "paragraph", "text": format!("Point C measured 8.2 mm against the governing minimum {cites}.")},
                {"kind": "table", "header": ["Point", "Measured mm"], "rows": [["C", "8.2"], ["D", "9.4"]],
                 "caption": "Readings from the March survey"}
            ]},
            {"heading": "Recommendation", "level": 1, "blocks": [
                {"kind": "paragraph", "text": "Assess point C before the unit returns to service."}
            ]}
        ]
    })
}

fn only_artifact(deps: &Arc<RuntimeDeps>) -> crate::artifacts::conversation_store::ArtifactRecord {
    let listed = deps.conversation_artifacts.list(OWNER, CONVERSATION).expect("lists");
    assert_eq!(listed.len(), 1, "{listed:#?}");
    listed.into_iter().next().unwrap()
}

fn raw_entries(bytes: &[u8]) -> std::collections::BTreeMap<String, Vec<u8>> {
    let mut archive = zip::ZipArchive::new(std::io::Cursor::new(bytes)).unwrap();
    let mut out = std::collections::BTreeMap::new();
    for index in 0..archive.len() {
        let mut entry = archive.by_index_raw(index).unwrap();
        let mut raw = Vec::new();
        entry.read_to_end(&mut raw).unwrap();
        out.insert(entry.name().to_string(), raw);
    }
    out
}

#[tokio::test]
async fn a_produced_note_survives_another_run_another_model_and_a_restart() {
    let h = Harness::new();
    let cited = fact(&h.graph);
    let written = call_with(
        &h.deps,
        json!({ "runId": "r", "toolCallId": "tc-note", "tool": "artifact.create_approval_note",
                "model": "Spark-X2.5-4B-Q8_0", "args": note_args(&format!("[M:{}@1]", cited.item_id)) }),
    )
    .await
    .expect("the note is produced");
    assert!(written.contains("section"), "{written}");

    // Registered as a candidate, bound to what it cites, on the template it
    // used, linked into the memory graph on its own receipt.
    let record = only_artifact(&h.deps);
    assert_eq!(record.stage, Stage::Candidate);
    assert_eq!(record.template.as_ref().map(|t| t.id.as_str()), Some("document_model"));
    let dependencies = h.deps.conversation_artifacts.dependencies(OWNER, &record.reference()).unwrap();
    assert_eq!(dependencies.len(), 1, "{dependencies:?}");
    assert_eq!(dependencies[0].id, cited.item_id);
    assert_eq!(dependencies[0].version.as_deref(), Some("1"));
    let node = record.graph_item_id.clone().expect("linked into the graph");
    let session = Session::open(User::new(OWNER, "A Person", vec![Role::Employee]));
    let history = h.graph.versions_of(&session, &node, None).expect("the node reads");
    assert_eq!(history.last().unwrap().item.status, ItemStatus::Admitted, "admitted on a verified receipt");
    assert_eq!(history.last().unwrap().item.depends_on[0].item_id, cited.item_id);

    // The bytes on disk in the producing run's workspace are the stored bytes.
    let on_disk = std::fs::read(h.deps.root_for("r").unwrap().join("note.docx")).unwrap();
    let stored_hash = {
        use sha2::Digest;
        format!("{:x}", sha2::Sha256::digest(&on_disk))
    };
    assert_eq!(stored_hash, record.sha256, "the store holds the exact bytes the run wrote");

    // Another run, another model, reads that exact version back.
    let manifest = call(&h.deps, LATER, "artifact.manifest", json!({ "artifact": record.reference().to_string() }))
        .await
        .expect("the manifest reads");
    assert!(manifest.contains(&record.sha256), "{manifest}");
    assert!(manifest.contains("\"current\": true"), "{manifest}");
    assert!(manifest.contains("\"stage\": \"candidate\""), "{manifest}");
    let read = call(&h.deps, LATER, "artifact.read_version", json!({ "artifact": record.reference().to_string() }))
        .await
        .expect("the version reads");
    assert!(read.contains("Point C measured 8.2 mm"), "{read}");
    assert!(read.contains("[M:"), "the citation is shown with the unit: {read}");
    let region = call(&h.deps, LATER, "artifact.read_region", json!({ "artifact": record.reference().to_string(), "region": "section:Recommendation" }))
        .await
        .expect("the region reads");
    assert!(region.contains("Assess point C") && !region.contains("8.2 mm"), "{region}");

    // And a restart: a second handle onto the same directory.
    let reopened = ConversationArtifacts::open(h.dir.path()).expect("the store reopens");
    let (again, bytes) = reopened.read(OWNER, &record.reference()).unwrap().expect("it survived");
    assert_eq!(bytes, on_disk, "byte for byte");
    assert_eq!(again.stage, Stage::Candidate);
    assert_eq!(reopened.dependencies(OWNER, &record.reference()).unwrap(), dependencies);
}

#[tokio::test]
async fn another_person_cannot_reach_an_artifact_through_any_of_the_tools() {
    let h = Harness::new();
    call(&h.deps, "r", "artifact.create_approval_note", note_args("")).await.expect("produced");
    let record = only_artifact(&h.deps);
    let reference = record.reference().to_string();

    // Bob, in his own conversation and then in Alice's conversation id.
    for (run, conversation) in [("r-bob", "c-bob"), ("r-bob-guess", CONVERSATION)] {
        let bob = h.as_user("bob", run, conversation);
        for (tool, args) in [
            ("artifact.manifest", json!({ "artifact": reference })),
            ("artifact.read_version", json!({ "artifact": reference })),
            ("artifact.read_region", json!({ "artifact": reference, "region": "p:1" })),
            ("artifact.validate", json!({ "artifact": reference, "render": "no" })),
            ("artifact.render", json!({ "artifact": reference })),
            ("artifact.diff", json!({ "artifact": record.artifact_id, "against": reference })),
            ("artifact.resolve_evidence", json!({ "artifact": reference })),
            ("artifact.edit", json!({ "artifact": reference, "edits": [{ "locator": "p:1", "replace": "x" }] })),
            ("artifact.read", json!({ "artifact": reference })),
        ] {
            let refused = call(&bob, run, tool, args).await.expect_err(tool);
            assert!(
                refused.contains("is not an artifact this conversation has produced"),
                "{tool} answered Bob with something other than the not-found wording: {refused}"
            );
            assert!(!refused.contains("8.2"), "{tool} leaked content: {refused}");
        }
    }
    // Nothing Bob did touched Alice's record.
    assert_eq!(h.deps.conversation_artifacts.versions(OWNER, &record.artifact_id).unwrap().len(), 1);
}

fn record_candidate(deps: &Arc<RuntimeDeps>, name: &str, mime: &str, bytes: Vec<u8>) -> String {
    deps.conversation_artifacts
        .record_version(
            NewArtifact {
                artifact_id: None,
                conversation_id: CONVERSATION.into(),
                owner_user_id: OWNER.into(),
                message_id: None,
                run_id: Some("r".into()),
                producer: Producer { model_id: None, tool: Some("artifact.create_approval_note".into()), agent: None },
                kind: ArtifactKind::Document,
                mime: mime.into(),
                title: name.into(),
                filename: Some(name.into()),
                complete: true,
                derived_from: None,
                renders: None,
                language: None,
                render_requires: Vec::new(),
                content: bytes,
            },
            VersionMeta { stage: Stage::Candidate, ..Default::default() },
        )
        .unwrap()
        .record
        .reference()
        .to_string()
}

#[tokio::test]
async fn a_corrupted_or_mistyped_file_fails_validation_and_is_never_published() {
    let h = Harness::new();
    const DOCX: &str = "application/vnd.openxmlformats-officedocument.wordprocessingml.document";
    let good = crate::artifacts::content::tests_support::sample_docx(h.dir.path());
    let mut truncated = good.clone();
    truncated.truncate(good.len() / 2);
    let pdf = crate::artifacts::pdf::render(&crate::artifacts::pdf::PdfSpec {
        title: "Note".into(),
        classification: "Internal".into(),
        blocks: vec![crate::artifacts::pdf::Block::Paragraph("A page of text that is plainly a PDF.".into())],
    })
    .unwrap();

    for (name, bytes, expected) in [
        ("broken.docx", truncated, "does not reopen"),
        ("note.docx", pdf, "claims to be a Word document"),
    ] {
        let reference = record_candidate(&h.deps, name, DOCX, bytes);
        let report = call(&h.deps, "r", "artifact.validate", json!({ "artifact": reference, "render": "no" }))
            .await
            .expect("validation reports rather than refusing");
        assert!(report.contains("format reopened: failed"), "{name}: {report}");
        assert!(report.contains(expected), "{name}: {report}");
        assert!(report.contains("accepted: failed"), "{name}: {report}");
        assert!(report.contains("content checked: not run"), "{name}: {report}");

        let refused = approved(&h.deps, "r", "artifact.register_version", json!({ "artifact": reference, "stage": "final" }))
            .await
            .expect_err("never published");
        assert!(refused.contains("did not accept"), "{name}: {refused}");
        let (id, version) = reference.split_once('@').unwrap();
        let record = h.deps.conversation_artifacts.get(OWNER, id, version.parse().ok()).unwrap().unwrap();
        assert_eq!(record.stage, Stage::Candidate);
    }
}

#[tokio::test]
async fn a_correction_to_what_a_note_cites_makes_the_note_stale_and_unpublishable() {
    let h = Harness::new();
    let cited = fact(&h.graph);
    call(&h.deps, "r", "artifact.create_approval_note", note_args(&format!("[M:{}@1]", cited.item_id)))
        .await
        .expect("produced");
    let record = only_artifact(&h.deps);
    let reference = record.reference().to_string();
    let before = call(&h.deps, "r", "artifact.resolve_evidence", json!({ "artifact": reference })).await.unwrap();
    assert!(before.contains("Every citation is bound and everything it rests on is current"), "{before}");

    // A person corrects the fact the note rests on.
    let mut correction = crate::knowledge::graph::runtime_memory::tests::item(
        MemoryKind::Correction,
        Provenance::Operator { user_id: "ravi".into() },
    );
    correction.content = "The minimum is 8.0 mm under Revision D.".into();
    h.graph.correct(correction, &cited.item_id).expect("corrected");

    // The graph marked the note's node stale by itself (P02's propagation)...
    let session = Session::open(User::new(OWNER, "A Person", vec![Role::Employee]));
    let node = h.graph.versions_of(&session, record.graph_item_id.as_deref().unwrap(), None).unwrap();
    assert_eq!(node.last().unwrap().item.status, ItemStatus::Stale, "the correction did not reach the artifact");

    // ...and every tool says so.
    let manifest = call(&h.deps, "r", "artifact.manifest", json!({ "artifact": reference })).await.unwrap();
    assert!(manifest.contains("\"current\": false"), "{manifest}");
    assert!(manifest.contains("moved from revision 1 to 2"), "{manifest}");
    let after = call(&h.deps, "r", "artifact.resolve_evidence", json!({ "artifact": reference })).await.unwrap();
    assert!(after.contains("stale"), "{after}");
    let report = call(&h.deps, "r", "artifact.validate", json!({ "artifact": reference, "render": "no" })).await.unwrap();
    assert!(report.contains("stale:"), "{report}");
    let refused = approved(&h.deps, "r", "artifact.register_version", json!({ "artifact": reference, "stage": "final" }))
        .await
        .expect_err("a stale note is not published");
    assert!(refused.contains("was not published") && refused.contains("moved from revision 1 to 2"), "{refused}");
}

#[tokio::test]
async fn an_edit_through_the_tool_changes_one_unit_and_copies_every_other_part() {
    let h = Harness::new();
    let cited = fact(&h.graph);
    call(&h.deps, "r", "artifact.create_approval_note", note_args(&format!("[M:{}@1]", cited.item_id)))
        .await
        .expect("produced");
    let v1 = only_artifact(&h.deps);
    let read = call(&h.deps, LATER, "artifact.read_version", json!({ "artifact": v1.reference().to_string() })).await.unwrap();
    let locator = read
        .lines()
        .find(|line| line.contains("Point C measured 8.2 mm"))
        .and_then(|line| line.split_whitespace().next())
        .expect("the finding's locator")
        .to_string();

    let edited = call(
        &h.deps,
        LATER,
        "artifact.edit",
        json!({ "artifact": v1.reference().to_string(), "edits": [{ "locator": locator, "find": "8.2 mm", "replace": "8.4 mm" }] }),
    )
    .await
    .expect("edited");
    assert!(edited.contains("@2"), "{edited}");
    let v2 = only_artifact(&h.deps);
    assert_eq!(v2.version, 2);
    assert_eq!(v2.stage, Stage::Candidate);
    assert_eq!(v2.derived_from, Some(v1.reference()));
    assert_eq!(v2.template, v1.template, "the template is carried");
    assert_eq!(
        h.deps.conversation_artifacts.dependencies(OWNER, &v2.reference()).unwrap(),
        h.deps.conversation_artifacts.dependencies(OWNER, &v1.reference()).unwrap(),
        "the citation's binding is carried, not re-guessed"
    );
    assert!(v2.graph_item_id.is_some(), "the new version is in the graph");

    let (_, before) = h.deps.conversation_artifacts.read(OWNER, &v1.reference()).unwrap().unwrap();
    let (_, after) = h.deps.conversation_artifacts.read(OWNER, &v2.reference()).unwrap().unwrap();
    let (a, b) = (raw_entries(&before), raw_entries(&after));
    assert_eq!(a.keys().collect::<Vec<_>>(), b.keys().collect::<Vec<_>>());
    for (name, raw) in &a {
        if name != "word/document.xml" {
            assert_eq!(raw, &b[name], "{name} changed");
        }
    }

    let diff = call(&h.deps, LATER, "artifact.diff", json!({ "artifact": v2.artifact_id })).await.unwrap();
    assert!(diff.contains("1 changed, 0 added, 0 removed"), "{diff}");
    assert!(diff.contains(&format!("~ {locator}")), "{diff}");

    // Editing the version that has been superseded is refused, naming the new one.
    let stale_edit = call(
        &h.deps,
        LATER,
        "artifact.edit",
        json!({ "artifact": v1.reference().to_string(), "edits": [{ "locator": locator, "find": "8.2 mm", "replace": "8.3 mm" }] }),
    )
    .await
    .expect_err("an edit of a superseded version is refused");
    assert!(stale_edit.contains("moved on to version 2"), "{stale_edit}");
}

#[tokio::test]
async fn the_same_effect_key_publishes_once_across_runs() {
    let h = Harness::new();
    // The same create, with the same effect key, in two runs of the
    // conversation -- a replay after a lost acknowledgement or a restart. The
    // retry renders different bytes, so content-addressing alone would mint a
    // second version: only the effect key can make it one publication.
    for (run, wording) in [("r", ""), (LATER, "(as re-rendered on the retry)")] {
        call_with(
            &h.deps,
            json!({ "runId": run, "toolCallId": format!("tc-{run}"), "tool": "artifact.create_approval_note",
                    "idempotencyKey": "note-for-unit-four", "args": note_args(wording) }),
        )
        .await
        .expect("each run's call succeeds");
    }
    let record = only_artifact(&h.deps);
    assert_eq!(h.deps.conversation_artifacts.versions(OWNER, &record.artifact_id).unwrap().len(), 1, "one version, not two");
    assert_eq!(record.run_id.as_deref(), Some("r"), "the first registration stands");
    let session = Session::open(User::new(OWNER, "A Person", vec![Role::Employee]));
    let nodes: Vec<_> = ["r", LATER]
        .iter()
        .flat_map(|run| {
            h.graph
                .snapshot(&session, &crate::knowledge::graph::runtime_memory::MemoryScope::Task { task_id: run.to_string() }, None)
                .unwrap()
        })
        .filter(|item| item.kind == MemoryKind::ArtifactRef)
        .collect();
    assert_eq!(nodes.len(), 1, "one graph node for one publication: {nodes:#?}");
}

#[tokio::test]
async fn a_sound_note_is_rendered_accepted_and_published_or_reported_unavailable() {
    let h = Harness::new();
    let cited = fact(&h.graph);
    call(&h.deps, "r", "artifact.create_approval_note", note_args(&format!("[M:{}@1]", cited.item_id)))
        .await
        .expect("produced");
    let record = only_artifact(&h.deps);
    let reference = record.reference().to_string();
    let report = call(&h.deps, "r", "artifact.validate", json!({ "artifact": reference })).await.expect("validated");
    eprintln!("P04-VALIDATION {report}");

    let inventory = crate::artifacts::render::inventory();
    if inventory.office.usable() && inventory.rasteriser.usable() {
        assert!(report.contains("render checked: passed"), "{report}");
        assert!(report.contains("accepted: passed"), "{report}");
        assert!(report.contains("render:rnd-"), "page handles are returned: {report}");

        let published = approved(&h.deps, "r", "artifact.register_version", json!({ "artifact": reference, "stage": "final", "effectKey": "publish-note" }))
            .await
            .expect("published");
        assert!(published.contains("as final"), "{published}");
        // The same call in the same run is absorbed by the run's own effect
        // ledger and replays the first answer; from another run, it is the
        // registry's effect key that recognises it.
        let replayed = approved(&h.deps, "r", "artifact.register_version", json!({ "artifact": reference, "stage": "final", "effectKey": "publish-note" }))
            .await
            .expect("replayed");
        assert_eq!(replayed, published, "a replay answers exactly as the first call did");
        let again = approved(&h.deps, LATER, "artifact.register_version", json!({ "artifact": reference, "stage": "final", "effectKey": "publish-note" }))
            .await
            .expect("repeat answered");
        assert!(again.contains("nothing was repeated"), "{again}");
        let manifest = call(&h.deps, LATER, "artifact.manifest", json!({ "artifact": reference })).await.unwrap();
        assert!(manifest.contains("\"stage\": \"final\""), "{manifest}");
        assert!(manifest.contains("render:rnd-"), "{manifest}");
        assert!(manifest.contains("LibreOffice"), "the renderer and its version are recorded: {manifest}");
    } else {
        assert!(report.contains("render checked: unavailable"), "{report}");
        assert!(report.contains("accepted: unavailable"), "{report}");
        let refused = approved(&h.deps, "r", "artifact.register_version", json!({ "artifact": reference, "stage": "final" }))
            .await
            .expect_err("not published without the render rung");
        assert!(refused.contains("did not accept"), "{refused}");
    }
}

#[tokio::test]
async fn templates_are_listed_with_their_versions_and_hashes() {
    let h = Harness::new();
    let listed = call(&h.deps, "r", "artifact.list_templates", json!({ "format": "pptx" })).await.unwrap();
    assert!(listed.contains("briefing_deck@1"), "{listed}");
    assert!(!listed.contains("approval_note@"), "narrowed to the format: {listed}");
    assert!(listed.contains(&crate::artifacts::templates::find("briefing_deck").unwrap().sha256), "{listed}");
}

/// A plan offers the artifact family only for work on a deliverable, and last.
#[test]
fn the_artifact_family_is_offered_for_deliverable_work_only_and_last() {
    let question = planning::derive("what is the design pressure of PV-2201?");
    assert!(!question.budget.permits(ToolName::ArtifactValidate));
    for prompt in ["write an approval note on the shell thickness", "review the approval note and publish the final version"] {
        let derived = planning::derive(prompt);
        let permitted = &derived.budget.permitted_tools;
        assert!(derived.budget.permits(ToolName::ArtifactValidate), "{prompt}");
        let first = permitted.iter().position(|t| *t == ToolName::ArtifactManifest).expect("offered");
        for tool in &permitted[first..] {
            assert!(tool.as_str().starts_with("artifact.") && !tool.is_artifact_creation() || *tool == ToolName::ArtifactEdit, "{prompt}: {tool:?} after the family");
        }
    }
}

/// The template form — the one the runtime's catalogue offers a model — is
/// checked against what the template promises, and its citations are bound.
#[tokio::test]
async fn an_approval_note_from_the_template_is_checked_against_its_template() {
    let h = Harness::new();
    let cited = fact(&h.graph);
    call(
        &h.deps,
        "r",
        "artifact.create_approval_note",
        json!({
            "path": "approval.docx",
            "template": "approval_note",
            "content": {
                "title": "Shell thickness at point C",
                "recipient": "Head of Maintenance, Unit Four",
                "subject": "Point C below the governing minimum",
                "findings": format!("Point C measured 8.2 mm against the 9.0 mm minimum [M:{}@1].", cited.item_id),
                "recommendation": "Assess point C before the unit returns to service.",
                "references": "Maintenance SOP Revision D, section 3.1.",
                "assumptions": "The March survey readings are representative."
            }
        }),
    )
    .await
    .expect("the note is produced from the template");
    let record = only_artifact(&h.deps);
    assert_eq!(record.template.as_ref().map(|t| t.id.as_str()), Some("approval_note"));
    let report = call(&h.deps, "r", "artifact.validate", json!({ "artifact": record.reference().to_string(), "render": "no" }))
        .await
        .unwrap();
    assert!(report.contains("content checked: passed"), "{report}");
    assert!(report.contains("everything the approval_note template promises"), "{report}");
    assert!(report.contains("render checked: not run"), "{report}");
    assert!(report.contains("accepted: failed"), "not accepted without its render rung: {report}");
}
