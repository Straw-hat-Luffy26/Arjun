//! What a bounded worker has to be unable to do.
//!
//! Most of these are negative: a child cannot reach another run's workspace,
//! cannot read above its ceiling, cannot call a tool it was not given, cannot
//! spawn a child of its own. They are written against the real narrowing rather
//! than against a mock, because the property being checked is that there is no
//! path through this code to a wider policy — and a mock would only prove there
//! is no path through the mock.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;

use super::*;
use super::manager::Dispatch;
use crate::agent_runtime::events::TaskEventLog;
use crate::identity::{Role, Session, User};
use crate::orchestrator::tools::ToolName;
use crate::policy::Classification;
use crate::subagents::certification::Decision;
use crate::subagents::profile::{ceiling, Isolation, MemoryScope, SchemaKind, WritePolicy};
use crate::registry::ModelRole;

const RUN: &str = "run-1";

fn user() -> Session {
    Session::open(User::new("priya", "Priya Sharma", vec![Role::Employee]))
}

fn knowledge_admin() -> Session {
    // The legacy `KnowledgeAdministrator` role is kept on the enum for
    // compatibility. Tests use it as a fixture for "a person who is not
    // a work-role user" — the active product's two roles are
    // Administrator and Employee, both of which hold the work
    // permissions, so a legacy variant is what stands in for "not
    // cleared for this material".
    Session::open(User::new(
        "kiran",
        "Kiran Das",
        vec![Role::KnowledgeAdministrator],
    ))
}

/// The parent policy most tests start from.
fn parent(session: &Session, tools: &[ToolName], root: &Path) -> InheritedPolicy {
    InheritedPolicy::of_run(session, Classification::Internal, root.join(RUN), tools)
}

/// The profiles actually shipped.
fn shipped() -> LoadedProfiles {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("src-tauri has a parent")
        .join("agents");
    load_profiles(&dir)
}

fn profile(name: &str) -> AgentProfile {
    shipped()
        .get(name)
        .unwrap_or_else(|| panic!("{name} is not a shipped profile"))
        .clone()
}

fn model() -> Decision {
    Decision {
        model_id: "qwen2.5-7b".to_string(),
        role: ModelRole::Reasoning,
        cheaper_than_parent: false,
        reason: "the run's own model".to_string(),
        tier: None,
        score: None,
    }
}

fn events() -> Arc<TaskEventLog> {
    Arc::new(TaskEventLog::in_memory().expect("an event log"))
}

/// A worker that reports whatever it is told to, so the manager's behaviour can
/// be driven without a model.
struct Fake {
    profile: String,
    behaviour: Behaviour,
}

enum Behaviour {
    Succeed,
    Fail(String),
    /// Sleeps past its deadline.
    Hang,
    /// Returns a result for a different schema than it was asked for.
    WrongShape,
    /// Tries to call a tool, and reports whether the policy permitted it.
    ReportTools,
}

#[async_trait]
impl ChildWorker for Fake {
    fn profile(&self) -> &str {
        &self.profile
    }

    async fn run(
        &self,
        packet: &ChildTaskPacket,
        policy: &EffectivePolicy,
    ) -> Result<ChildResult, String> {
        match &self.behaviour {
            Behaviour::Succeed => Ok(ChildResult::completed(
                &packet.child_id,
                &self.profile,
                packet.required_schema,
                vec![Finding {
                    statement: "something was found".to_string(),
                    evidence: Vec::new(),
                }],
                0.9,
                Vec::new(),
                1,
            )),
            Behaviour::Fail(why) => Err(why.clone()),
            Behaviour::Hang => {
                tokio::time::sleep(std::time::Duration::from_secs(30)).await;
                unreachable!("the manager's deadline should have fired")
            }
            Behaviour::WrongShape => Ok(ChildResult::completed(
                &packet.child_id,
                &self.profile,
                // Not what the packet asked for.
                SchemaKind::Code,
                Vec::new(),
                1.0,
                Vec::new(),
                1,
            )),
            Behaviour::ReportTools => {
                let statement = policy
                    .tools
                    .iter()
                    .map(|tool| tool.as_str())
                    .collect::<Vec<_>>()
                    .join(",");
                Ok(ChildResult::completed(
                    &packet.child_id,
                    &self.profile,
                    packet.required_schema,
                    vec![Finding {
                        statement,
                        evidence: Vec::new(),
                    }],
                    1.0,
                    Vec::new(),
                    1,
                ))
            }
        }
    }
}

fn manager_with(name: &str, behaviour: Behaviour) -> SubagentManager {
    SubagentManager::new(shipped().profiles, events()).with_worker(Arc::new(Fake {
        profile: name.to_string(),
        behaviour,
    }))
}

// == The shipped profiles compile =========================================

#[test]
fn every_shipped_profile_compiles() {
    let loaded = shipped();
    assert!(
        loaded.rejected.is_empty(),
        "rejected: {:?}",
        loaded
            .rejected
            .iter()
            .map(|r| format!("{}: {}", r.file, r.error.explain()))
            .collect::<Vec<_>>()
    );
    let names: Vec<&str> = loaded.profiles.iter().map(|p| p.name.as_str()).collect();
    assert_eq!(
        names,
        vec![
            "artifact-reviewer",
            "calculation-checker",
            "code-worker",
            "document-extractor",
            "knowledge-retriever",
        ]
    );
}

#[test]
fn no_shipped_profile_permits_a_child_of_its_own() {
    for profile in shipped().profiles {
        assert_eq!(profile.limits.max_children, 0, "{}", profile.name);
    }
}

#[test]
fn no_shipped_profile_declares_network() {
    for profile in shipped().profiles {
        assert!(!profile.network_permitted, "{}", profile.name);
    }
}

#[test]
fn every_read_only_profile_can_write_nowhere() {
    // A read-only worker that could write would be neither, and several of them
    // run at once.
    for profile in shipped().profiles {
        if profile.isolation == Isolation::ReadOnly {
            assert_eq!(profile.write_policy, WritePolicy::None, "{}", profile.name);
            assert_eq!(profile.memory_scope, MemoryScope::None, "{}", profile.name);
        }
    }
}

#[test]
fn a_profile_asking_for_more_than_the_hard_ceiling_is_refused() {
    let source = format!(
        "---\nname: greedy\ndescription: d\nversion: 1.0.0\nmodel-role: reasoning\n\
         allowed-tools:\n  - search_documents\nlimits:\n  max-turns: {}\n  \
         max-output-tokens: 512\n  max-children: 0\n  max-duration-seconds: 30\n\
         isolation: read-only\nmemory-scope: none\nnetwork: none\nwrite-policy: none\n\
         classification-ceiling: internal\nrequired-schema: retrieval\n---\nBody.\n",
        ceiling::MAX_TURNS + 1
    );
    match profile::compile(&source, "greedy", "sha") {
        Err(ProfileError::AboveCeiling { field, .. }) => assert_eq!(field, "limits.max-turns"),
        other => panic!("expected a ceiling refusal, got {other:?}"),
    }
}

#[test]
fn a_profile_asking_for_children_is_refused_rather_than_clamped() {
    // Silently giving them zero would hide that the author asked for something
    // the model does not support.
    let source = "---\nname: breeder\ndescription: d\nversion: 1.0.0\nmodel-role: reasoning\n\
                  allowed-tools:\n  - search_documents\nlimits:\n  max-turns: 2\n  \
                  max-output-tokens: 512\n  max-children: 3\n  max-duration-seconds: 30\n\
                  isolation: read-only\nmemory-scope: none\nnetwork: none\nwrite-policy: none\n\
                  classification-ceiling: internal\nrequired-schema: retrieval\n---\nBody.\n";
    assert!(matches!(
        profile::compile(source, "breeder", "sha"),
        Err(ProfileError::AboveCeiling { .. })
    ));
}

#[test]
fn a_profile_that_both_allows_and_denies_a_tool_is_refused() {
    // Either reading is a guess about what the author meant.
    let source = "---\nname: confused\ndescription: d\nversion: 1.0.0\nmodel-role: reasoning\n\
                  allowed-tools:\n  - search_documents\ndisallowed-tools:\n  - search_documents\n\
                  limits:\n  max-turns: 2\n  max-output-tokens: 512\n  max-children: 0\n  \
                  max-duration-seconds: 30\nisolation: read-only\nmemory-scope: none\n\
                  network: none\nwrite-policy: none\nclassification-ceiling: internal\n\
                  required-schema: retrieval\n---\nBody.\n";
    assert!(matches!(
        profile::compile(source, "confused", "sha"),
        Err(ProfileError::Contradiction { .. })
    ));
}

#[test]
fn a_profile_asking_for_the_network_is_refused_rather_than_downgraded() {
    let source = "---\nname: chatty\ndescription: d\nversion: 1.0.0\nmodel-role: reasoning\n\
                  allowed-tools:\n  - search_documents\nlimits:\n  max-turns: 2\n  \
                  max-output-tokens: 512\n  max-children: 0\n  max-duration-seconds: 30\n\
                  isolation: read-only\nmemory-scope: none\nnetwork: loopback\n\
                  write-policy: none\nclassification-ceiling: internal\n\
                  required-schema: retrieval\n---\nBody.\n";
    match profile::compile(source, "chatty", "sha") {
        Err(ProfileError::InvalidField { field, .. }) => assert_eq!(field, "network"),
        other => panic!("expected a refusal, got {other:?}"),
    }
}

// == 1. A child cannot access another run's workspace =====================

#[test]
fn a_child_writes_only_inside_its_own_directory() {
    let dir = tempfile::tempdir().expect("temp dir");
    let session = user();
    let inherited = parent(
        &session,
        &[ToolName::WriteScopedFile, ToolName::ExecuteCode, ToolName::ReadScopedFile,
          ToolName::ValidateArtifact],
        dir.path(),
    );
    let policy = inherited
        .narrow_for(&profile("code-worker"), "child-1")
        .expect("narrowed");

    let own = policy.write_root.clone().expect("a write root");
    assert!(policy.may_write(&own.join("total.py")));

    // Its parent's own directory is not its to write in.
    assert!(!policy.may_write(&dir.path().join(RUN).join("deliverable.docx")));
    // Another run's workspace is not reachable at all.
    assert!(!policy.may_write(&dir.path().join("run-2").join("note.txt")));
    // Nor by climbing out of its own.
    assert!(!policy.may_write(&own.join("..").join("..").join("run-2").join("x")));
    assert!(!policy.may_write(Path::new("/etc/passwd")));
}

#[test]
fn a_read_only_child_may_write_nowhere_at_all() {
    let dir = tempfile::tempdir().expect("temp dir");
    let session = user();
    let inherited = parent(&session, &[ToolName::SearchDocuments], dir.path());
    let policy = inherited
        .narrow_for(&profile("knowledge-retriever"), "child-1")
        .expect("narrowed");

    assert!(policy.write_root.is_none());
    assert!(!policy.may_write(&dir.path().join(RUN).join("anything")));
}

#[test]
fn two_children_of_one_run_do_not_share_a_directory() {
    let dir = tempfile::tempdir().expect("temp dir");
    let session = user();
    let inherited = parent(
        &session,
        &[ToolName::WriteScopedFile, ToolName::ExecuteCode],
        dir.path(),
    );
    let first = inherited
        .narrow_for(&profile("code-worker"), "child-1")
        .expect("narrowed");
    let second = inherited
        .narrow_for(&profile("code-worker"), "child-2")
        .expect("narrowed");

    let a = first.write_root.clone().expect("a root");
    let b = second.write_root.clone().expect("a root");
    assert_ne!(a, b);
    assert!(!first.may_write(&b.join("x")));
    assert!(!second.may_write(&a.join("x")));
}

// == 2. A child cannot read a higher classification =======================

#[test]
fn a_child_cannot_handle_material_above_its_ceiling() {
    let dir = tempfile::tempdir().expect("temp dir");
    let session = user();
    let inherited = parent(&session, &[ToolName::SearchDocuments], dir.path());
    // The retriever's ceiling is `internal`, and so is the run's.
    let policy = inherited
        .narrow_for(&profile("knowledge-retriever"), "child-1")
        .expect("narrowed");

    assert!(policy.may_handle(Classification::Internal));
    assert!(policy.may_handle(Classification::ProcessDiagram));
    // Everything in the more restricted tier is above it.
    for above in [
        Classification::Financial,
        Classification::VendorNegotiation,
        Classification::UnreleasedDesign,
        Classification::InternalCorrespondence,
        Classification::BusinessStrategy,
    ] {
        assert!(!policy.may_handle(above), "{above:?} was permitted");
    }
}

#[test]
fn a_profile_cannot_raise_the_ceiling_the_run_set() {
    let dir = tempfile::tempdir().expect("temp dir");
    let session = user();
    // The run is restricted to `internal`. The extractor's profile declares
    // `processDiagram`, which is the same tier, so it is permitted.
    let restricted = InheritedPolicy::of_run(
        &session,
        Classification::Internal,
        dir.path().join(RUN),
        &[ToolName::SearchDocuments, ToolName::ReadScopedFile],
    );
    let policy = restricted
        .narrow_for(&profile("document-extractor"), "child-1")
        .expect("narrowed");
    assert_eq!(policy.classification_ceiling.sensitivity(), 0);

    // A run whose ceiling is higher does not lower the profile's either — the
    // result is the lower of the two, whichever way round they are.
    let permissive = InheritedPolicy::of_run(
        &session,
        Classification::Financial,
        dir.path().join(RUN),
        &[ToolName::SearchDocuments, ToolName::ReadScopedFile],
    );
    let policy = permissive
        .narrow_for(&profile("document-extractor"), "child-1")
        .expect("narrowed");
    assert!(!policy.may_handle(Classification::Financial));
}

#[test]
fn a_child_is_refused_when_the_person_is_not_cleared_for_its_material() {
    let dir = tempfile::tempdir().expect("temp dir");
    // A knowledge administrator curates manuals and is not cleared for
    // commercially sensitive material.
    let session = knowledge_admin();
    let inherited = InheritedPolicy::of_run(
        &session,
        Classification::Financial,
        dir.path().join(RUN),
        &[ToolName::SearchDocuments],
    );
    // A profile whose ceiling is that material.
    let mut restricted = profile("knowledge-retriever");
    restricted.classification_ceiling = Classification::Financial;

    match inherited.narrow_for(&restricted, "child-1") {
        Err(InheritRefusal::NotCleared { .. }) => {}
        other => panic!("expected a clearance refusal, got {other:?}"),
    }
}

// == 3. A child cannot call a disallowed tool =============================

#[tokio::test]
async fn a_child_receives_only_the_tools_the_parent_and_profile_agree_on() {
    let dir = tempfile::tempdir().expect("temp dir");
    let session = user();
    // The parent holds everything.
    let inherited = parent(&session, ToolName::ALL, dir.path());

    let manager = manager_with("knowledge-retriever", Behaviour::ReportTools);
    let spawned = manager
        .spawn(
            "knowledge-retriever",
            &inherited,
            "find the seal wear passages",
            vec![],
            model(),
            &Dispatch::default(),
        )
        .await
        .expect("spawned");

    // The retriever (1.1.0) declares its six read tools, so that is all it
    // gets — even though the parent holds every tool there is.
    let reported: std::collections::BTreeSet<&str> =
        spawned.result().findings[0].statement.split(',').collect();
    let declared: std::collections::BTreeSet<&str> = [
        "knowledge.search_authorized",
        "knowledge.hybrid_search",
        "knowledge.load_evidence_region",
        "knowledge.source_version",
        "knowledge.rerank",
        "memory.neighbours",
    ]
    .into_iter()
    .collect();
    assert_eq!(reported, declared);
    assert!(reported.len() < ToolName::ALL.len());
}

#[test]
fn a_child_never_receives_a_tool_the_parent_does_not_hold() {
    let dir = tempfile::tempdir().expect("temp dir");
    let session = user();
    // The parent holds one tool. The code worker's profile asks for four.
    let inherited = parent(&session, &[ToolName::ReadScopedFile], dir.path());
    let policy = inherited
        .narrow_for(&profile("code-worker"), "child-1")
        .expect("narrowed");

    assert_eq!(policy.tools, vec![ToolName::ReadScopedFile]);
    assert!(!policy.may_call(ToolName::ExecuteCode));
    assert!(!policy.may_call(ToolName::WriteScopedFile));
    // What it asked for and did not get is recorded, so the trace can say why
    // it did less than its profile describes.
    assert!(policy.refused_tools.contains(&ToolName::ExecuteCode));
}

#[test]
fn a_profiles_denylist_wins_over_the_parents_grant() {
    let dir = tempfile::tempdir().expect("temp dir");
    let session = user();
    // The parent holds `create_docx`; the code worker's profile denies it.
    let inherited = parent(
        &session,
        &[ToolName::CreateDocx, ToolName::WriteScopedFile],
        dir.path(),
    );
    let policy = inherited
        .narrow_for(&profile("code-worker"), "child-1")
        .expect("narrowed");

    assert!(!policy.may_call(ToolName::CreateDocx));
    assert!(policy.may_call(ToolName::WriteScopedFile));
}

#[test]
fn a_child_with_nothing_in_common_with_its_parent_is_refused_rather_than_started() {
    let dir = tempfile::tempdir().expect("temp dir");
    let session = user();
    let inherited = parent(&session, &[ToolName::CreateXlsx], dir.path());

    match inherited.narrow_for(&profile("knowledge-retriever"), "child-1") {
        Err(InheritRefusal::NoToolsInCommon { .. }) => {}
        other => panic!("expected a refusal, got {other:?}"),
    }
}

// == 4. A child cannot spawn a grandchild =================================

#[test]
fn a_child_cannot_start_a_child_of_its_own() {
    let dir = tempfile::tempdir().expect("temp dir");
    let session = user();
    let inherited = parent(&session, &[ToolName::SearchDocuments], dir.path());
    let child = inherited
        .narrow_for(&profile("knowledge-retriever"), "child-1")
        .expect("narrowed");

    assert_eq!(child.inherited.depth, 1);
    assert_eq!(child.limits.max_children, 0);

    // The child's own inherited view, used as a parent, refuses.
    match child
        .inherited
        .narrow_for(&profile("knowledge-retriever"), "grandchild")
    {
        Err(InheritRefusal::TooDeep { depth, ceiling }) => {
            assert_eq!(depth, 2);
            assert_eq!(ceiling, ceiling::MAX_DEPTH);
        }
        other => panic!("expected a depth refusal, got {other:?}"),
    }
}

// == 5. A child cannot change network or sandbox policy ===================

#[test]
fn no_child_is_ever_given_the_network() {
    let dir = tempfile::tempdir().expect("temp dir");
    let session = user();
    let inherited = parent(&session, ToolName::ALL, dir.path());

    for profile in shipped().profiles {
        let Ok(policy) = inherited.narrow_for(&profile, "child-1") else {
            continue;
        };
        assert!(!policy.inherited.network_permitted, "{}", profile.name);
    }
}

#[test]
fn a_child_cannot_drop_the_approval_requirement_it_inherited() {
    // The failure this prevents: a subagent used as the way round an approval
    // the parent owed.
    let dir = tempfile::tempdir().expect("temp dir");
    let session = user();
    let inherited = parent(
        &session,
        &[ToolName::WriteScopedFile, ToolName::ExecuteCode],
        dir.path(),
    );
    assert!(inherited.approval_required);

    let policy = inherited
        .narrow_for(&profile("code-worker"), "child-1")
        .expect("narrowed");
    assert!(policy.inherited.approval_required);
}

#[test]
fn the_policy_hash_changes_when_any_constraint_does() {
    let dir = tempfile::tempdir().expect("temp dir");
    let session = user();
    let base = parent(&session, &[ToolName::SearchDocuments], dir.path());
    let baseline = base.hash();

    let mut wider = base.clone();
    wider.permitted_tools.push(ToolName::ExecuteCode);
    assert_ne!(baseline, wider.hash());

    let mut higher = base.clone();
    higher.classification_ceiling = Classification::Financial;
    assert_ne!(baseline, higher.hash());

    let mut networked = base.clone();
    networked.network_permitted = true;
    assert_ne!(baseline, networked.hash());

    let mut unapproved = base.clone();
    unapproved.approval_required = false;
    assert_ne!(baseline, unapproved.hash());

    // And is stable for the same policy, whatever order the tools are in.
    let mut reordered = base.clone();
    reordered.permitted_tools.reverse();
    assert_eq!(baseline, reordered.hash());
}

// == 6. A child timeout is recovered and recorded =========================

#[tokio::test]
async fn a_child_that_hangs_is_stopped_and_reported_as_timed_out() {
    let dir = tempfile::tempdir().expect("temp dir");
    let session = user();
    let inherited = parent(&session, &[ToolName::SearchDocuments], dir.path());

    // A one-second budget rather than the profile's sixty, so the test costs a
    // second. The deadline being enforced is the manager's, whatever it is.
    let mut profiles = shipped().profiles;
    for profile in &mut profiles {
        profile.limits.max_duration_seconds = 1;
    }
    let manager = SubagentManager::new(profiles, events()).with_worker(Arc::new(Fake {
        profile: "knowledge-retriever".to_string(),
        behaviour: Behaviour::Hang,
    }));

    let spawned = manager
        .spawn("knowledge-retriever", &inherited, "find something", vec![], model(), &Dispatch::default())
        .await
        .expect("spawned");

    let result = spawned.result();
    assert_eq!(result.status, ChildStatus::TimedOut);
    // The part that matters: it is not success.
    assert!(!result.is_complete());
    assert!(result.detail.as_ref().unwrap().contains("limit"));
    // And a timed-out child does not claim confidence in work it did not do.
    assert_eq!(result.confidence, 0.0);
}

#[tokio::test]
async fn a_worker_that_fails_becomes_a_failure_rather_than_an_error() {
    // A parent always gets a typed result, so it cannot handle only the happy
    // path by accident.
    let dir = tempfile::tempdir().expect("temp dir");
    let session = user();
    let inherited = parent(&session, &[ToolName::SearchDocuments], dir.path());

    let manager = manager_with(
        "knowledge-retriever",
        Behaviour::Fail("the index could not be opened".to_string()),
    );
    let spawned = manager
        .spawn("knowledge-retriever", &inherited, "find something", vec![], model(), &Dispatch::default())
        .await
        .expect("spawned");

    assert_eq!(spawned.result().status, ChildStatus::Failed);
    assert!(!spawned.result().is_complete());
    assert!(spawned.result().detail.as_ref().unwrap().contains("index"));
}

#[tokio::test]
async fn a_worker_answering_a_different_question_is_a_failure() {
    let dir = tempfile::tempdir().expect("temp dir");
    let session = user();
    let inherited = parent(&session, &[ToolName::SearchDocuments], dir.path());

    let manager = manager_with("knowledge-retriever", Behaviour::WrongShape);
    let spawned = manager
        .spawn("knowledge-retriever", &inherited, "find something", vec![], model(), &Dispatch::default())
        .await
        .expect("spawned");

    assert_eq!(spawned.result().status, ChildStatus::Failed);
    assert!(spawned.result().findings.is_empty());
}

#[tokio::test]
async fn a_role_with_no_worker_says_nothing_ran() {
    // Distinct from a failure: the role is correctly declared and this build
    // cannot perform it.
    let dir = tempfile::tempdir().expect("temp dir");
    let session = user();
    let inherited = parent(&session, ToolName::ALL, dir.path());
    let manager = SubagentManager::new(shipped().profiles, events());

    let spawned = manager
        .spawn("artifact-reviewer", &inherited, "check the note", vec![], model(), &Dispatch::default())
        .await
        .expect("spawned");

    assert_eq!(spawned.result().status, ChildStatus::Refused);
    let detail = spawned.result().detail.as_ref().unwrap();
    assert!(detail.contains("no worker"), "{detail}");
    assert!(detail.contains("no result exists"), "{detail}");
}

#[test]
fn a_status_that_is_not_completed_can_never_read_as_success() {
    for status in [
        ChildStatus::Failed,
        ChildStatus::TimedOut,
        ChildStatus::Cancelled,
        ChildStatus::Refused,
    ] {
        assert!(!status.is_complete(), "{status:?}");
        let result = ChildResult::ended(
            "c", "p", status, SchemaKind::Retrieval, Vec::new(), "why", 0,
        );
        assert!(!result.is_complete());
    }
    assert!(ChildStatus::Completed.is_complete());
}

#[test]
fn partial_findings_are_kept_and_the_status_still_says_incomplete() {
    // Two passages found before a timeout are two real passages. The status is
    // what stops them being read as a completed search.
    let result = ChildResult::ended(
        "c",
        "knowledge-retriever",
        ChildStatus::TimedOut,
        SchemaKind::Retrieval,
        vec![Finding {
            statement: "one passage".to_string(),
            evidence: Vec::new(),
        }],
        "ran out of time",
        2,
    );
    assert!(!result.is_complete());
    assert!(result.describe().contains("partial finding"));
}

// == 7. Parallel read-only workers do not race ============================

#[tokio::test]
async fn several_read_only_children_run_at_once_and_agree() {
    let dir = tempfile::tempdir().expect("temp dir");
    let session = user();
    let inherited = parent(&session, &[ToolName::SearchDocuments], dir.path());
    let manager = Arc::new(manager_with("knowledge-retriever", Behaviour::ReportTools));

    // Eight at once, each on a different objective so they are eight children
    // rather than one reused.
    let mut handles = Vec::new();
    for index in 0..8 {
        let manager = Arc::clone(&manager);
        let inherited = inherited.clone();
        handles.push(tokio::spawn(async move {
            manager
                .spawn(
                    "knowledge-retriever",
                    &inherited,
                    &format!("objective {index}"),
                    vec![],
                    model(),
                    &Dispatch::default(),
                )
                .await
        }));
    }

    let mut ids = std::collections::BTreeSet::new();
    for handle in handles {
        let spawned = handle.await.expect("joined").expect("spawned");
        let result = spawned.result();
        assert_eq!(result.status, ChildStatus::Completed);
        // Every one saw the same narrowed tool list. A race in the narrowing
        // would show up as one of them reporting something else.
        assert_eq!(result.findings[0].statement, "knowledge.search_authorized");
        assert!(ids.insert(result.child_id.clone()), "a child id repeated");
    }
    assert_eq!(ids.len(), 8);
}

#[test]
fn only_read_only_workers_are_concurrent() {
    for profile in shipped().profiles {
        let concurrent = profile.isolation.is_concurrent();
        assert_eq!(
            concurrent,
            profile.isolation == Isolation::ReadOnly,
            "{}",
            profile.name
        );
    }
    // The code worker writes and needs approval, so it runs alone.
    assert!(!profile("code-worker").isolation.is_concurrent());
}

// == 8. Duplicate creation returns the existing child =====================

#[tokio::test]
async fn the_same_work_asked_for_twice_produces_one_child() {
    let dir = tempfile::tempdir().expect("temp dir");
    let session = user();
    let inherited = parent(&session, &[ToolName::SearchDocuments], dir.path());
    let manager = manager_with("knowledge-retriever", Behaviour::Succeed);

    let first = manager
        .spawn("knowledge-retriever", &inherited, "find the seal wear", vec![], model(), &Dispatch::default())
        .await
        .expect("spawned");
    let second = manager
        .spawn("knowledge-retriever", &inherited, "find the seal wear", vec![], model(), &Dispatch::default())
        .await
        .expect("spawned");

    assert!(!first.is_reused());
    assert!(second.is_reused());
    assert_eq!(first.result().child_id, second.result().child_id);
    assert_eq!(first.result().result_hash, second.result().result_hash);
}

#[tokio::test]
async fn different_work_under_one_profile_produces_different_children() {
    let dir = tempfile::tempdir().expect("temp dir");
    let session = user();
    let inherited = parent(&session, &[ToolName::SearchDocuments], dir.path());
    let manager = manager_with("knowledge-retriever", Behaviour::Succeed);

    let first = manager
        .spawn("knowledge-retriever", &inherited, "seal wear", vec![], model(), &Dispatch::default())
        .await
        .expect("spawned");
    let second = manager
        .spawn("knowledge-retriever", &inherited, "wall thickness", vec![], model(), &Dispatch::default())
        .await
        .expect("spawned");

    assert!(!second.is_reused());
    assert_ne!(first.result().child_id, second.result().child_id);
}

#[test]
fn the_idempotency_key_is_derived_from_the_work_rather_than_generated() {
    let inputs = vec![InputRef::Evidence { marker: 1 }];
    let a = derive_idempotency_key(RUN, "knowledge-retriever", "seal wear", &inputs);
    let b = derive_idempotency_key(RUN, "knowledge-retriever", "seal wear", &inputs);
    assert_eq!(a, b);

    // Different in every dimension that makes it different work.
    assert_ne!(a, derive_idempotency_key("run-2", "knowledge-retriever", "seal wear", &inputs));
    assert_ne!(a, derive_idempotency_key(RUN, "document-extractor", "seal wear", &inputs));
    assert_ne!(a, derive_idempotency_key(RUN, "knowledge-retriever", "wall thickness", &inputs));
    assert_ne!(
        a,
        derive_idempotency_key(RUN, "knowledge-retriever", "seal wear", &[InputRef::Evidence { marker: 2 }])
    );
}

// == The packet carries references, not contents ==========================

#[test]
fn a_packet_carries_no_document_text() {
    let dir = tempfile::tempdir().expect("temp dir");
    let session = user();
    let inherited = parent(&session, &[ToolName::SearchDocuments], dir.path());
    let policy = inherited
        .narrow_for(&profile("knowledge-retriever"), "child-1")
        .expect("narrowed");

    let packet = ChildTaskPacket::new(
        "child-1",
        RUN,
        "key",
        "find the passages about seal wear",
        vec![
            InputRef::Document {
                sha256: "abc123".repeat(8),
                page: Some(4),
            },
            InputRef::Evidence { marker: 2 },
        ],
        &policy,
        chrono::Utc::now(),
    );

    let written = serde_json::to_string(&packet).expect("serialised");
    // The objective is the parent's own words and travels. Nothing else does:
    // there is no variant of `InputRef` that can hold a passage.
    assert!(written.contains("find the passages about seal wear"));
    assert!(written.contains("abc123"));
    assert!(written.contains(&policy.inherited_hash));
    // The trace line carries no objective at all — a trace is read by more
    // people than a run is.
    assert!(!packet.describe().contains("seal wear"));
}

// == Requirement 9: a cheaper model needs certification ===================

#[test]
fn a_role_where_a_small_model_fails_visibly_never_gets_a_cheaper_one() {
    let decision = certification::choose(ModelRole::Reasoning, "qwen2.5-7b", &[]);
    assert!(!decision.cheaper_than_parent);
    assert_eq!(decision.model_id, "qwen2.5-7b");
    assert!(decision.reason.contains("too easy to miss"), "{}", decision.reason);
}

#[test]
fn an_uncertified_model_is_not_used_however_cheap_it_is() {
    let decision = certification::choose(ModelRole::Embedding, "qwen2.5-7b", &[]);
    assert!(!decision.cheaper_than_parent);
    assert_eq!(decision.model_id, "qwen2.5-7b");
}

#[test]
fn certification_below_the_floor_refuses_the_cheaper_model() {
    use crate::subagents::certification::{is_reliable_for, Refused};
    assert_eq!(
        is_reliable_for(None, ModelRole::Embedding),
        Err(Refused::Uncertified)
    );
}

#[test]
fn the_reason_a_model_was_chosen_names_what_it_rested_on() {
    // So an operator reading a run manifest can see the decision rested on a
    // certification pack rather than on evidence from this site.
    let decision = certification::choose(ModelRole::Embedding, "qwen2.5-7b", &[]);
    assert!(!decision.reason.is_empty());
    assert!(decision.reason.starts_with("Using the run's own model"));
}

// == The events the parent records ========================================

#[tokio::test]
async fn a_parent_records_the_child_manifest_and_the_result_hash() {
    let dir = tempfile::tempdir().expect("temp dir");
    let session = user();
    let inherited = parent(&session, &[ToolName::SearchDocuments], dir.path());

    let log = events();
    let manager = SubagentManager::new(shipped().profiles, Arc::clone(&log))
        .with_worker(Arc::new(Fake {
            profile: "knowledge-retriever".to_string(),
            behaviour: Behaviour::Succeed,
        }));

    let spawned = manager
        .spawn("knowledge-retriever", &inherited, "find something", vec![], model(), &Dispatch::default())
        .await
        .expect("spawned");

    let page = log.events_since(RUN, 0).expect("readable");
    let started = page
        .events
        .iter()
        .find(|e| e.event_type == crate::agent_runtime::events::TaskEventType::SubagentStarted)
        .expect("a start event");
    let stopped = page
        .events
        .iter()
        .find(|e| e.event_type == crate::agent_runtime::events::TaskEventType::SubagentStopped)
        .expect("a stop event");

    // The manifest: what the child was permitted, not what it asked for.
    let manifest = &started.payload["manifest"];
    assert_eq!(manifest["allowedTools"][0], "knowledge.search_authorized");
    assert_eq!(manifest["networkPermitted"], false);
    assert_eq!(manifest["maxChildren"], 0);
    assert_eq!(manifest["depth"], 1);
    assert_eq!(started.payload["policyHash"], serde_json::json!(inherited.hash()));
    assert_eq!(started.payload["model"]["modelId"], "qwen2.5-7b");

    assert_eq!(stopped.payload["status"], "completed");
    assert_eq!(stopped.payload["complete"], true);
    assert_eq!(
        stopped.payload["resultHash"],
        serde_json::json!(spawned.result().result_hash)
    );
}

#[tokio::test]
async fn a_timed_out_child_is_recorded_as_incomplete() {
    let dir = tempfile::tempdir().expect("temp dir");
    let session = user();
    let inherited = parent(&session, &[ToolName::SearchDocuments], dir.path());

    let log = events();
    let manager = SubagentManager::new(shipped().profiles, Arc::clone(&log))
        .with_worker(Arc::new(Fake {
            profile: "knowledge-retriever".to_string(),
            behaviour: Behaviour::Fail("nope".to_string()),
        }));

    manager
        .spawn("knowledge-retriever", &inherited, "find something", vec![], model(), &Dispatch::default())
        .await
        .expect("spawned");

    let snapshot = log.snapshot(RUN).expect("readable").expect("a snapshot");
    assert_eq!(snapshot.subagents_started, 1);
    assert_eq!(snapshot.subagents_finished, 1);
    // Counted apart from the total, because a fan-out where workers failed is
    // a different run from one where they all finished.
    assert_eq!(snapshot.subagents_incomplete, 1);
}

// == The result hash =======================================================

#[test]
fn the_result_hash_covers_the_status_as_well_as_the_findings() {
    // A record that kept the findings and changed the status must not match,
    // because that is exactly the alteration requirement 8 is about.
    let findings = vec![Finding {
        statement: "one".to_string(),
        evidence: Vec::new(),
    }];
    let completed = ChildResult::completed(
        "c", "p", SchemaKind::Retrieval, findings.clone(), 1.0, Vec::new(), 1,
    );
    let timed_out = ChildResult::ended(
        "c", "p", ChildStatus::TimedOut, SchemaKind::Retrieval, findings, "late", 1,
    );
    assert_ne!(completed.result_hash, timed_out.result_hash);
}

// == P07: an embedding model is never a conversational child =============

fn certified(model_id: &str) -> crate::model_recommendation::certified_catalog::PackageCertification {
    let scores: serde_json::Map<String, serde_json::Value> = [
        "instructionFollowing", "reasoningQuality", "hallucinationRate", "codingAbility",
        "mathematicalReasoning", "jsonReliability", "toolCallingAccuracy",
        "memoryEngineCompatibility", "contextWindowRetention", "responseStability",
        "chatTemplateCorrectness", "bosEosStopTokenCompliance", "reasoningTagLeakageFilter",
        "streamingParserStability", "runtimeProcessStability", "restartStatePersistence",
    ]
    .iter()
    .map(|key| (key.to_string(), serde_json::json!(99.0)))
    .collect();
    serde_json::from_value(serde_json::json!({
        "packageId": model_id, "modelId": model_id, "modelName": model_id, "quantLabel": "Q8_0",
        "backend": "llama.cpp", "tier": "Certified", "confidenceScore": 99.0,
        "runtimeProfileId": "p", "numericScores": scores,
        "provenance": {"createdBy": "t", "certifiedBy": "t", "generatedWith": "t",
                       "runnerVersion": "t", "profileHash": "t", "signature": "t",
                       "generatedAt": "2026-09-25T00:00:00Z"},
        "quirksAndNotes": ""
    }))
    .expect("a certification record")
}

/// A certified, cheaper embedding model scoring 99 on everything is still not
/// chosen as a child's model — for the embedding role or any other.
#[test]
fn a_certified_embedding_model_is_never_a_childs_conversational_model() {
    let embedder = crate::registry::tests::entry("embed-small", 0.6, vec![ModelRole::Embedding]);
    let ocr = crate::registry::tests::entry("ocr-small", 1.0, vec![ModelRole::DocumentOcr]);
    let embed_cert = certified("embed-small");
    let ocr_cert = certified("ocr-small");
    for role in [ModelRole::Embedding, ModelRole::DocumentOcr, ModelRole::Reasoning] {
        let decision = certification::choose(
            role,
            "model-parent",
            &[(&embedder, Some(&embed_cert)), (&ocr, Some(&ocr_cert))],
        );
        assert_ne!(decision.model_id, "embed-small", "{role:?}: {}", decision.reason);
    }
    // The cheaper path still works where it is meant to.
    let ocr_decision = certification::choose(
        ModelRole::DocumentOcr,
        "model-parent",
        &[(&embedder, Some(&embed_cert)), (&ocr, Some(&ocr_cert))],
    );
    assert_eq!(ocr_decision.model_id, "ocr-small");
}
