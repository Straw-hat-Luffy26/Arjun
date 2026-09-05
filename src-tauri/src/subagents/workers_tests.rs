
use super::*;
use crate::{
    agent_runtime::{events::TaskEventLog, workspace::Workspace},
    identity::{Role, User},
    knowledge::{Chunk, ChunkKind},
    policy::Classification,
    subagents::{certification::Decision, load_profiles, InheritedPolicy},
};
fn setup() -> (tempfile::TempDir, Resources, InheritedPolicy) {
    let dir = tempfile::tempdir().unwrap();
    let resources = Resources {
        index: Arc::new(KnowledgeIndex::open(dir.path()).unwrap()),
        events: Arc::new(TaskEventLog::open(dir.path()).unwrap()),
        session: Arc::default(),
        workspaces: Arc::default(),
        passages: Arc::default(),
        calculations: Arc::default(),
        produced: Arc::default(),
    };
    let session = Session::open(User::new("operator", "Operator", vec![Role::Employee]));
    *resources.session.write().unwrap() = Some(session.clone());
    use crate::agent_runtime::events::{EventDraft, TaskEventType};
    resources
        .events
        .record(EventDraft::new(
            "parent-00017",
            TaskEventType::RunCreated,
            "operator",
        ))
        .unwrap();
    resources
        .events
        .record(
            EventDraft::new("parent-00017", TaskEventType::RunClassified, "operator")
                .with(serde_json::json!({"classification":"Internal"})),
        )
        .unwrap();
    let workspace = Workspace::create(dir.path(), "parent-00017").unwrap();
    let inherited = InheritedPolicy::of_run(
        &session,
        Classification::Internal,
        workspace.root().to_owned(),
        ToolName::ALL,
    );
    resources
        .workspaces
        .lock()
        .unwrap()
        .insert("parent-00017".into(), workspace);
    resources
        .index
        .index_document(
            "Pump register",
            Classification::Internal,
            &[Chunk {
                id: "chunk-00017".into(),
                document_sha256: "document-00017".into(),
                ordinal: 0,
                char_count: 22,
                text: "PUMP-A17 pressure 3 bar".into(),
                page: 1,
                section_path: Vec::new(),
                kind: ChunkKind::Prose,
            }],
        )
        .unwrap();
    resources.calculations.lock().unwrap().insert(
        "parent-00017".into(),
        vec![calculation::evaluate("3 bar * 2").unwrap()],
    );
    let path = inherited.workspace_root.join("report-00017.txt");
    std::fs::write(&path, "PUMP-A17 pressure 3 bar").unwrap();
    artifacts::remember(
        &resources.produced,
        "parent-00017",
        artifacts::produced_from(
            &path,
            Some(&inherited.workspace_root),
            artifacts::Kind::Text,
            None,
        ),
    );
    (dir, resources, inherited)
}
fn manager(dir: &Path, resources: Resources) -> SubagentManager {
    let profiles = load_profiles(
        &PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("agents"),
    )
    .profiles;
    register(
        SubagentManager::new(profiles, Arc::new(TaskEventLog::open(dir).unwrap())),
        resources,
    )
}
fn decision() -> Decision {
    Decision {
        model_id: "deterministic-local-services-v1".into(),
        role: crate::registry::ModelRole::Reasoning,
        cheaper_than_parent: false,
        reason: "local services; no inference".into(),
        tier: None,
        score: None,
    }
}

#[tokio::test]
async fn real_workers_retrieve_extract_recompute_and_reopen_with_durable_parent_results() {
    let (dir, resources, inherited) = setup();
    let manager = manager(dir.path(), resources.clone());
    for profile in PROFILES {
        assert!(manager.has_worker(profile));
    }
    assert!(!manager.has_worker("code-worker"));
    let retrieval = manager
        .spawn(
            "knowledge-retriever",
            &inherited,
            "PUMP-A17",
            Vec::new(),
            decision(),
        )
        .await
        .unwrap();
    assert!(retrieval.result().is_complete(), "{:?}", retrieval.result());
    assert_eq!(retrieval.result().findings[0].evidence[0].marker, Some(1));
    assert!(retrieval.result().findings[0].statement.contains("3 bar"));
    for profile in &PROFILES[1..] {
        let inputs = manager.handoff_inputs(profile, &inherited).unwrap();
        assert!(!inputs.is_empty(), "{profile} needs real references");
        let child = manager
            .spawn(
                profile,
                &inherited,
                "Check the authorized inputs",
                inputs,
                decision(),
            )
            .await
            .unwrap();
        assert!(child.result().is_complete(), "{:?}", child.result());
        assert!(!child.result().findings.is_empty());
    }
    let log = TaskEventLog::open(dir.path()).unwrap();
    let saved = log.children_for_run("parent-00017").unwrap();
    assert_eq!(saved.len(), 4);
    assert_eq!(saved[0].evidence[0].chunk_id, "chunk-00017");
    assert!(log.children_for_run("another-parent").unwrap().is_empty());
    drop(manager);
    resources.passages.lock().unwrap().clear(); // crash before parent receipt
    let reopened = self::manager(dir.path(), resources.clone());
    let retry = reopened
        .spawn(
            "knowledge-retriever",
            &inherited,
            "PUMP-A17",
            Vec::new(),
            decision(),
        )
        .await
        .unwrap();
    assert!(retry.is_reused());
    assert_eq!(retry.result(), retrieval.result());
    assert_eq!(
        retrieval::for_run(&resources.passages, "parent-00017")[0].chunk_id,
        "chunk-00017"
    );
    let mut parent =
        crate::agent_runtime::tasks::tests::record("parent-00017", "2026-09-05T00:00:00Z");
    parent.children = log.children_for_run("parent-00017").unwrap();
    crate::agent_runtime::tasks::save(dir.path(), &parent).unwrap();
    assert_eq!(
        crate::agent_runtime::tasks::load(dir.path(), "parent-00017", Some(&parent.user_id))
            .unwrap()
            .children
            .len(),
        4
    );
    resources.index.supersede("document-00017").unwrap();
    assert!(
        reopened
            .spawn(
                "knowledge-retriever",
                &inherited,
                "PUMP-A17",
                Vec::new(),
                decision()
            )
            .await
            .is_err(),
        "a saved result cannot bypass source revocation"
    );
}

#[tokio::test]
async fn specialists_refuse_foreign_references_changed_identity_and_bad_artifacts() {
    let (dir, resources, inherited) = setup();
    let manager = manager(dir.path(), resources.clone());
    for (profile, reference) in [
        ("document-extractor", InputRef::Evidence { marker: 999 }),
        (
            "artifact-reviewer",
            InputRef::WorkspaceFile {
                path: "../foreign.txt".into(),
            },
        ),
        (
            "calculation-checker",
            InputRef::Expression {
                expression: "999 kg * 2".into(),
            },
        ),
    ] {
        assert!(manager
            .spawn(profile, &inherited, "Check", vec![reference], decision())
            .await
            .is_err());
    }
    let mut limited = inherited.clone();
    limited.permitted_tools = vec![ToolName::SearchDocuments];
    assert!(manager
        .handoff_inputs("artifact-reviewer", &limited)
        .is_err());
    std::fs::write(inherited.workspace_root.join("report-00017.txt"), "").unwrap();
    let inputs = manager
        .handoff_inputs("artifact-reviewer", &inherited)
        .unwrap();
    let failed = manager
        .spawn(
            "artifact-reviewer",
            &inherited,
            "Check empty file",
            inputs,
            decision(),
        )
        .await
        .unwrap();
    assert!(!failed.result().is_complete());
    assert_eq!(
        TaskEventLog::open(dir.path())
            .unwrap()
            .children_for_run("parent-00017")
            .unwrap()[0]
            .result
            .status,
        super::super::ChildStatus::Failed
    );
    *resources.session.write().unwrap() = Some(Session::open(User::new(
        "other",
        "Other",
        vec![Role::Employee],
    )));
    assert!(manager
        .handoff_inputs("knowledge-retriever", &inherited)
        .is_err());
}
