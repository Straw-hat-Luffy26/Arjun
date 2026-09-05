//! What the runtime's two questions come to, without a child process.
//!
//! `tool.authorize` and `tool.execute` are the whole security surface, so they
//! are exercised directly here — the cross-language plumbing has its own test in
//! `tests/agent_runtime.rs`, and mixing the two would make a policy failure look
//! like a transport one.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use serde_json::{json, Value};

use super::*;
use crate::identity::{Role, Session, User};

fn signed_in_user() -> Arc<std::sync::RwLock<Option<Session>>> {
    Arc::new(std::sync::RwLock::new(Some(Session::open(User::new(
        "priya",
        "Priya Sharma",
        vec![Role::Employee],
    )))))
}

#[path = "context_live_tests.rs"]
mod context_live_tests;

/// Deps plus the directory they live in.
///
/// The directory is returned rather than dropped because it holds both the
/// knowledge index and the run's workspace; letting it fall out of scope deletes
/// them under the test.
fn deps_with(
    session: Arc<std::sync::RwLock<Option<Session>>>,
) -> (Arc<RuntimeDeps>, tempfile::TempDir) {
    let dir = tempfile::tempdir().expect("temp dir");
    let workspaces = Arc::new(Mutex::new(HashMap::new()));
    workspaces.lock().expect("fresh lock").insert(
        "r".to_string(),
        workspace::Workspace::create(dir.path(), "r").expect("workspace"),
    );
    let deps = Arc::new(RuntimeDeps {
        index: Arc::new(KnowledgeIndex::open(dir.path()).expect("index opens")),
        session,
        workspaces,
        approvals: Arc::new(ApprovalQueue::new()),
        calculations: Arc::default(),
        passages: Arc::default(),
        produced: Arc::default(),
        calls: Arc::default(),
        // A plan permitting every tool, so these tests exercise the gateway
        // rather than the budget -- which has its own tests below.
        //
        // Registered rather than absent: a run with no plan is now refused
        // outright, which is the whole of `missing_plan` further down. Leaving
        // the table empty here would make every test in this file assert that
        // refusal instead of the thing it is about.
        plans: {
            let table: Arc<Mutex<HashMap<String, crate::orchestrator::plan::PlanRun>>> =
                Arc::default();
            {
                let mut plans = table.lock().expect("fresh lock");
                for run_id in ["r", "planned-without-workspace"] {
                    plans.insert(
                        run_id.to_string(),
                        crate::orchestrator::plan::PlanRun::new(
                            run_id,
                            vec!["do the work".to_string()],
                            crate::orchestrator::plan::Budget::standard(ToolName::ALL.to_vec()),
                        ),
                    );
                }
            }
            // `planned-without-workspace` is a real run with a real budget and
            // no workspace, which is what isolates the path check from the plan
            // check: without it, a test aimed at "no workspace" would be
            // satisfied by the refusal for "no plan".
            table
        },
        // In memory: these tests are about the gateway, and a durable history
        // is checked where it belongs, in `events::tests`.
        events: Arc::new(
            crate::agent_runtime::events::TaskEventLog::in_memory().expect("an event log"),
        ),
        // An empty skills directory: these tests are about the gateway, and
        // the skill system is checked where it belongs, in `skills::tests`.
        skills: Arc::new(crate::skills::SkillRegistry::open(dir.path().join("__no_skills__"))),
        // On disk under the temp dir, so the durability and isolation these
        // tests assert are the real ones rather than a per-test map.
        // The deployment's real checks, so these tests exercise the same
        // refusal path production does rather than an empty registry.
        hooks: Arc::new(crate::hooks::HookRegistry::with_builtin_policy()),
        memory: Arc::new(crate::agent_runtime::memory::MemoryStore::open(dir.path())),
        checkpoints: Arc::default(),
        emit: Arc::new(|_| {}),
        emit_durable: Arc::new(|_| {}),
        // A manager with no profiles: these tests are about the gateway, and
        // the subagent system has its own tests in `subagents::tests`. What
        // matters here is that one is *present*, so a delegation refused for
        // want of a manager cannot be mistaken for a policy decision.
        subagents: Arc::new(crate::subagents::SubagentManager::new(
            Vec::new(),
            Arc::new(
                crate::agent_runtime::events::TaskEventLog::in_memory().expect("an event log"),
            ),
        )),
        multimodal: Arc::new(
            crate::knowledge::MultimodalIndex::open(dir.path()).expect("a multimodal index"),
        ),
        // Durable by default: these tests are about the gateway, and a
        // degraded installation has its own tests in `audit_health`.
        audit_health: Arc::new(crate::agent_runtime::audit_health::AuditHealth::durable()),
    });
    (deps, dir)
}

fn search(query: &str) -> Value {
    json!({
        "runId": "r",
        "toolCallId": "tc",
        "tool": "search_documents",
        "args": { "query": query }
    })
}

#[tokio::test]
async fn a_call_with_no_one_signed_in_is_refused() {
    let (deps, _dir) = deps_with(Arc::new(std::sync::RwLock::new(None)));
    let error = authorize(search("x"), &deps).await.unwrap_err();
    assert_eq!(error.code, code::REFUSED);
    assert!(error.message.contains("No one is signed in"));
}

#[tokio::test]
async fn a_malformed_call_names_the_field_it_is_missing() {
    let (deps, _dir) = deps_with(signed_in_user());
    let error = authorize(json!({ "runId": "r" }), &deps).await.unwrap_err();
    assert_eq!(error.code, code::BAD_PARAMS);
    assert!(error.message.contains("toolCallId"));
}

#[tokio::test]
async fn an_unknown_tool_is_refused_with_the_list_of_real_ones() {
    let (deps, _dir) = deps_with(signed_in_user());
    let verdict = authorize(
        json!({ "runId": "r", "toolCallId": "tc", "tool": "rm_rf", "args": {} }),
        &deps,
    )
    .await
    .unwrap();
    assert_eq!(verdict["outcome"], "refuse");
    assert!(verdict["reason"]
        .as_str()
        .unwrap()
        .contains("knowledge.search_authorized"));
}

#[tokio::test]
async fn an_allowed_call_comes_back_with_a_grant() {
    let (deps, _dir) = deps_with(signed_in_user());
    let verdict = authorize(search("x"), &deps).await.unwrap();
    assert_eq!(verdict["outcome"], "allow");
    assert!(!verdict["grant"].as_str().unwrap().is_empty());
}

#[tokio::test]
async fn execution_without_a_grant_is_refused_even_though_the_call_is_permitted() {
    // The whole point: `search_documents` would be allowed if asked for
    // properly. Skipping the asking is what gets refused.
    let (deps, _dir) = deps_with(signed_in_user());
    let error = execute(search("x"), &deps).await.unwrap_err();
    assert_eq!(error.code, code::REFUSED);
    assert!(error.message.contains("no authorisation grant"));
}

#[tokio::test]
async fn execution_with_an_invented_grant_is_refused() {
    let (deps, _dir) = deps_with(signed_in_user());
    let mut call = search("x");
    call["grant"] = json!("made-up");
    let error = execute(call, &deps).await.unwrap_err();
    assert_eq!(error.code, code::REFUSED);
}

#[tokio::test]
async fn a_grant_earned_for_one_query_does_not_execute_another() {
    let (deps, _dir) = deps_with(signed_in_user());
    let allow = authorize(search("pump curve"), &deps).await.unwrap();

    let mut swapped = search("salary list");
    swapped["grant"] = allow["grant"].clone();
    let error = execute(swapped, &deps).await.unwrap_err();

    assert_eq!(error.code, code::REFUSED);
    assert!(error.message.contains("arguments"));
}

#[tokio::test]
async fn an_authorised_search_runs_and_says_it_found_nothing_rather_than_staying_silent() {
    let (deps, _dir) = deps_with(signed_in_user());
    let allow = authorize(search("wall thickness"), &deps).await.unwrap();

    let mut call = search("wall thickness");
    call["grant"] = allow["grant"].clone();
    let result = execute(call, &deps).await.unwrap();

    // The index is empty, and the honest answer is to say so — PS Part C.
    let text = result["text"].as_str().unwrap();
    assert!(text.contains("No passages matched"));
    assert!(text.contains("do not assert it"));
}

#[tokio::test]
async fn the_same_grant_cannot_execute_twice() {
    let (deps, _dir) = deps_with(signed_in_user());
    let allow = authorize(search("x"), &deps).await.unwrap();

    let mut call = search("x");
    call["grant"] = allow["grant"].clone();

    assert!(execute(call.clone(), &deps).await.is_ok());
    assert!(execute(call, &deps).await.is_err());
}

/// A calculation is kept, so the workbook can show working rather than recall.
#[tokio::test]
async fn a_calculation_is_recorded_for_the_workbook() {
    let (deps, _dir) = deps_with(signed_in_user());
    let call = json!({
        "runId": "r",
        "toolCallId": "tc",
        "tool": "run_calculation",
        "args": { "expression": "2 m * 3 m" }
    });
    let allow = authorize(call.clone(), &deps).await.unwrap();

    let mut with_grant = call;
    with_grant["grant"] = allow["grant"].clone();
    execute(with_grant, &deps).await.expect("the calculation runs");

    let table = deps.calculations.lock().expect("fresh lock");
    assert_eq!(table.get("r").map(Vec::len), Some(1));
}

#[tokio::test]
async fn a_write_inside_the_runs_own_directory_is_put_to_a_person() {
    let (deps, _dir) = deps_with(signed_in_user());
    let queue = deps.approvals.clone();

    let waiting = tokio::spawn({
        let deps = deps.clone();
        async move {
            authorize(
                json!({
                    "runId": "r",
                    "toolCallId": "tc",
                    "tool": "write_scoped_file",
                    "args": { "path": "note.txt", "content": "hello" }
                }),
                &deps,
            )
            .await
        }
    });

    // It reaches the approvals queue rather than being refused outright, which
    // is what "needs approval" has to mean once there is somewhere to ask.
    let item = loop {
        if let Some(item) = queue.pending().first().cloned() {
            break item;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    };
    assert_eq!(item.request.tool, "workspace.write_text");
    assert!(item.request.arguments.iter().any(|a| a.contains("note.txt")));

    waiting.abort();
}

/// The same side effect, asked for twice, happens once.
///
/// Exercised through the real `authorize`/`execute` pair rather than against
/// the store, because the thing being checked is that `execute` consults the
/// record *before* it reaches the tool. A test at the store level would pass
/// with the consultation deleted.
#[tokio::test]
async fn a_side_effecting_call_made_twice_is_performed_once() {
    let (deps, dir) = deps_with(signed_in_user());
    let reviewer = Session::open(User::new("ravi", "Ravi Menon", vec![Role::Administrator]));

    let write = |tool_call_id: &str| {
        json!({
            "runId": "r",
            "toolCallId": tool_call_id,
            "tool": "write_scoped_file",
            "args": { "path": "note.txt", "content": "the seal is worn" }
        })
    };

    // A write is put to a person, so each attempt has to be approved before it
    // can be executed at all.
    let approve_next = |deps: Arc<RuntimeDeps>, call: Value| {
        let reviewer = reviewer.clone();
        async move {
            let queue = deps.approvals.clone();
            let waiting = tokio::spawn({
                let deps = deps.clone();
                async move { authorize(call, &deps).await }
            });
            let item = loop {
                if let Some(item) = queue.pending().first().cloned() {
                    break item;
                }
                tokio::time::sleep(std::time::Duration::from_millis(5)).await;
            };
            queue
                .decide_durable(&reviewer, &item.request.id, true, None, |_| {
                    if deps.events.resolve_approval(&item.request.id, events::ApprovalStatus::Approved, &reviewer.user.id, None, chrono::Utc::now())? { Ok(()) }
                    else { Err("Approval did not commit".into()) }
                })
                .expect("the reviewer approves");
            waiting.await.expect("the task finished").expect("authorised")
        }
    };

    let first = approve_next(deps.clone(), write("tc-1")).await;
    let mut call = write("tc-1");
    call["grant"] = first["grant"].clone();
    let done = execute(call, &deps).await.expect("the write runs");
    assert!(done["details"]["replayed"].is_null());

    let path = dir.path().join("runs").join("r").join("note.txt");
    let written = std::fs::read_to_string(&path).expect("the file exists");
    // Changed on disk after the fact, so a genuine second write would be
    // visible as the file going back to what the tool would have put there.
    std::fs::write(&path, "edited after the run").expect("overwritten");

    // The same call again — a new tool-call id and a new grant, which is
    // exactly what a loop replaying an unacknowledged call produces.
    let second = approve_next(deps.clone(), write("tc-2")).await;
    let mut again = write("tc-2");
    again["grant"] = second["grant"].clone();
    let replayed = execute(again, &deps).await.expect("the replay answers");

    assert_eq!(replayed["details"]["replayed"], json!(true));
    assert_eq!(replayed["text"], done["text"]);
    // The file was not written a second time.
    assert_eq!(
        std::fs::read_to_string(&path).expect("still there"),
        "edited after the run"
    );
    assert_ne!(written, "edited after the run");
}

/// A write interrupted mid-flight is not silently attempted again.
///
/// Two independent things stop the retry, and this exercises both in the order
/// they actually apply:
///
/// 1. **The run is over.** Recovery ends every run it finds without an ending,
///    so the loop that was carrying it gets no further authorisations at all.
///    This is what prevents the repeat in practice.
/// 2. **The effect is unaccountable.** Even presented with a grant, the same
///    key is refused, because nobody can say whether the first attempt took.
///    This is the belt to that braces — it holds if anything ever resumed a
///    degraded run.
#[tokio::test]
async fn an_interrupted_write_is_refused_rather_than_repeated() {
    let (deps, dir) = deps_with(signed_in_user());

    let args = json!({ "path": "note.txt", "content": "the seal is worn" });
    let key = crate::agent_runtime::events::derive_key("r", "write_scoped_file", &args);
    let fingerprint = crate::agent_runtime::events::args_fingerprint(&args);

    // A run that was under way, and a write whose intent reached the disk and
    // whose outcome did not. Exactly what a process killed mid-write leaves.
    deps.events
        .record(
            crate::agent_runtime::events::EventDraft::new(
                "r",
                crate::agent_runtime::events::TaskEventType::RunCreated,
                "priya",
            )
            .with(json!({ "promptShown": "draft a note" })),
        )
        .expect("created");
    deps.events
        .begin_effect("r", &key, "write_scoped_file", &fingerprint, "note.txt");

    // The next start finds both.
    deps.events
        .recover_interrupted(crate::agent_runtime::events::SYSTEM_ACTOR)
        .expect("recovery ran");

    // 1. The run is ended, so nothing new is authorised — the loop cannot even
    //    get as far as asking a person to approve the write again.
    let verdict = authorize(
        json!({
            "runId": "r",
            "toolCallId": "tc-2",
            "tool": "write_scoped_file",
            "args": args,
        }),
        &deps,
    )
    .await
    .expect("a verdict");
    assert_eq!(verdict["outcome"], "refuse");
    assert!(verdict["reason"].as_str().unwrap().contains("has ended"));

    // 2. And the effect itself is unaccountable, so even a call that somehow
    //    arrived with a grant would be refused rather than performed.
    match deps
        .events
        .begin_effect("r", &key, "write_scoped_file", &fingerprint, "note.txt")
    {
        crate::agent_runtime::events::EffectLookup::Unknown(recorded) => {
            let refusal = recorded.unknown_refusal();
            assert!(refusal.contains("note.txt"), "{refusal}");
            assert!(
                refusal.contains("may or may not"),
                "the refusal must not claim to know: {refusal}"
            );
            assert!(refusal.contains("not been attempted again"), "{refusal}");
        }
        other => panic!("an interrupted write must not be repeatable: {other:?}"),
    }

    // Nothing was written. A retry that produced the file would be exactly the
    // double-write this exists to prevent.
    assert!(!dir.path().join("runs").join("r").join("note.txt").exists());
}

/// A cancellation stops the run at a boundary, not mid-tool.
#[tokio::test]
async fn no_new_tool_call_is_authorised_once_the_run_has_ended() {
    let (deps, _dir) = deps_with(signed_in_user());

    // Ordinary calls are fine while the run is live.
    let before = authorize(search("wall thickness"), &deps).await.unwrap();
    assert_eq!(before["outcome"], "allow");

    // Somebody presses stop. This is the record of it, which is what the
    // gateway consults — not anything in the child process's memory.
    deps.events
        .record(
            crate::agent_runtime::events::EventDraft::new(
                "r",
                crate::agent_runtime::events::TaskEventType::RunCancelled,
                "priya",
            )
            .with(json!({ "failure": "Stopped, because somebody stopped it." })),
        )
        .expect("cancelled");

    let after = authorize(search("wall thickness"), &deps).await.unwrap();
    assert_eq!(after["outcome"], "refuse");
    let reason = after["reason"].as_str().unwrap();
    assert!(reason.contains("has ended"), "{reason}");
    // Told what to do about it, so the model reports rather than retries.
    assert!(reason.contains("Stop and report"), "{reason}");
}

/// A repeated search is not collapsed, and deliberately.
#[tokio::test]
async fn a_read_only_call_made_twice_is_performed_twice() {
    let (deps, _dir) = deps_with(signed_in_user());

    let run_once = |id: &str| {
        let deps = deps.clone();
        let call = json!({
            "runId": "r",
            "toolCallId": id,
            "tool": "search_documents",
            "args": { "query": "wall thickness" }
        });
        async move {
            let allow = authorize(call.clone(), &deps).await.unwrap();
            let mut with_grant = call;
            with_grant["grant"] = allow["grant"].clone();
            execute(with_grant, &deps).await.expect("the search runs")
        }
    };

    let first = run_once("tc-1").await;
    let second = run_once("tc-2").await;
    // Neither is a replay: collapsing repeated searches would hide a model
    // going in circles from the repeat limit that exists to catch it.
    assert!(first["details"]["replayed"].is_null());
    assert!(second["details"]["replayed"].is_null());
}

#[tokio::test]
async fn a_write_outside_the_runs_directory_is_refused_without_troubling_anybody() {
    let (deps, _dir) = deps_with(signed_in_user());
    let verdict = authorize(
        json!({
            "runId": "r",
            "toolCallId": "tc",
            "tool": "write_scoped_file",
            "args": { "path": "../../elsewhere.txt", "content": "hello" }
        }),
        &deps,
    )
    .await
    .unwrap();

    // Refused by the gateway, so no approval request was ever raised — an
    // approver should not be asked to judge something already impossible.
    assert_eq!(verdict["outcome"], "refuse");
    assert!(deps.approvals.pending().is_empty());
}

#[tokio::test]
async fn a_run_with_no_workspace_cannot_touch_a_path_at_all() {
    // A real run, with a real budget, that has no workspace: every path it
    // could name is under no permitted root, so there is nothing it may read.
    let (deps, _dir) = deps_with(signed_in_user());
    let verdict = authorize(
        json!({
            "runId": "planned-without-workspace",
            "toolCallId": "tc",
            "tool": "read_scoped_file",
            "args": { "path": "note.txt" }
        }),
        &deps,
    )
    .await
    .unwrap();
    assert_eq!(verdict["outcome"], "refuse");
}

/// The bug this pins: the gateway compares a path against the permitted roots,
/// so a bare `"note.txt"` is under no root and is refused. Every relative path
/// the model was told to use would have failed.
#[tokio::test]
async fn a_relative_path_is_anchored_to_the_runs_workspace_rather_than_refused() {
    let (deps, _dir) = deps_with(signed_in_user());
    let verdict = authorize(
        json!({
            "runId": "r",
            "toolCallId": "tc",
            "tool": "read_scoped_file",
            "args": { "path": "note.txt" }
        }),
        &deps,
    )
    .await
    .unwrap();

    assert_eq!(verdict["outcome"], "allow", "{verdict}");
    let resolved = verdict["resolvedPath"].as_str().expect("a resolved path");
    assert!(resolved.ends_with("note.txt"), "{resolved}");
    // Anchored under the run's own directory, not somewhere shared.
    assert!(resolved.contains("runs"), "{resolved}");
}

/// Anchoring makes relative paths *expressible*, not permitted. The containment
/// decision stays exactly where it was.
#[tokio::test]
async fn a_relative_path_that_climbs_out_is_still_refused_after_anchoring() {
    let (deps, _dir) = deps_with(signed_in_user());
    for escape in [
        "../../etc/passwd",
        r"..\..\windows\system32\config\sam",
        "sub/../../../outside.txt",
    ] {
        let verdict = authorize(
            json!({
                "runId": "r",
                "toolCallId": "tc",
                "tool": "read_scoped_file",
                "args": { "path": escape }
            }),
            &deps,
        )
        .await
        .unwrap();
        assert_eq!(verdict["outcome"], "refuse", "{escape} was not refused");
    }
}

/// A fully absolute path is left as written, so the gateway judges exactly what
/// the model asked for.
///
/// "Absolute" is platform-specific and the difference matters here: on Windows
/// `/etc/passwd` is *rooted but not absolute* — it has no drive — so it is
/// anchored rather than passed through. That is the safe direction (anchoring
/// can only narrow where a call may reach, never widen it), and the containment
/// check still runs either way. Asserting one platform's answer on both would
/// pin a behaviour that does not exist.
#[test]
fn a_fully_absolute_path_is_left_alone_so_it_is_judged_as_written() {
    let dir = tempfile::tempdir().expect("temp dir");
    let root = dir.path().to_path_buf();

    // Built from the platform's own idea of an absolute path.
    let elsewhere = if cfg!(windows) {
        std::path::PathBuf::from(r"C:\Windows\System32\config\sam")
    } else {
        std::path::PathBuf::from("/etc/passwd")
    };
    assert!(elsewhere.is_absolute(), "the fixture must be absolute");

    let call = anchor_path(
        ToolCall::new(
            "read_scoped_file",
            json!({ "path": elsewhere.display().to_string() }),
        ),
        &[root.clone()],
    );

    assert_eq!(call.text("path"), Some(elsewhere.display().to_string().as_str()));
    assert!(!std::path::Path::new(call.text("path").unwrap()).starts_with(&root));
}

/// A rooted path is passed through, not anchored.
///
/// `Path::join` *replaces* the root when its argument has one, so anchoring
/// `/etc/passwd` onto a Windows workspace yields `C:/etc/passwd` — outside the
/// workspace. The gateway refuses that either way, but anchoring must not
/// manufacture a path that relies on a later check to be safe.
#[tokio::test]
async fn a_rooted_path_is_passed_through_and_then_refused() {
    let dir = tempfile::tempdir().expect("temp dir");
    let root = dir.path().to_path_buf();

    let call = anchor_path(
        ToolCall::new("read_scoped_file", json!({ "path": "/etc/passwd" })),
        &[root.clone()],
    );
    assert_eq!(call.text("path"), Some("/etc/passwd"), "it must not be anchored");

    // And the gateway refuses it, which is where the decision belongs.
    let (deps, _dir) = deps_with(signed_in_user());
    let verdict = authorize(
        json!({
            "runId": "r",
            "toolCallId": "tc",
            "tool": "read_scoped_file",
            "args": { "path": "/etc/passwd" }
        }),
        &deps,
    )
    .await
    .unwrap();
    assert_eq!(verdict["outcome"], "refuse", "{verdict}");
}

#[test]
fn a_call_with_no_path_argument_passes_through_untouched() {
    let dir = tempfile::tempdir().expect("temp dir");
    let call = anchor_path(
        ToolCall::new("run_calculation", json!({ "expression": "2 m * 3 m" })),
        &[dir.path().to_path_buf()],
    );
    assert_eq!(call.text("expression"), Some("2 m * 3 m"));
    assert!(call.text("path").is_none());
}

#[test]
fn an_approvers_view_of_a_long_argument_is_truncated_rather_than_endless() {
    // A write's content can be a whole document. An approval screen that makes
    // somebody scroll past 30 KB to find the path is one where they stop reading
    // and start clicking yes.
    let rendered = render_arguments(&json!({ "content": "x".repeat(5_000), "path": "a.txt" }));
    let content = rendered
        .iter()
        .find(|a| a.starts_with("content"))
        .expect("content is rendered");
    assert!(content.contains("(5000 characters)"), "{content}");
    assert!(content.len() < 400, "{content}");
    assert!(rendered.iter().any(|a| a == "path = a.txt"));
}

#[tokio::test]
async fn unknown_methods_are_named_rather_than_silently_ignored() {
    let (deps, _dir) = deps_with(signed_in_user());
    let error = handle("tool.please", json!({}), &deps).await.unwrap_err();
    assert_eq!(error.code, code::UNKNOWN_METHOD);
}

#[test]
fn a_missing_bundle_is_reported_with_the_path_and_the_fix() {
    // Matched rather than unwrapped: the success arm holds an `Arc<Self>`, and
    // `unwrap_err` would demand `Debug` on a live child process handle.
    let (deps, _dir) = deps_with(signed_in_user());
    let outcome = AgentRuntime::spawn(
        deps,
        Arc::new(|_| {}),
        PathBuf::from("/nonexistent/runtime.mjs"),
    );

    let Err(error) = outcome else {
        panic!("a missing bundle must not start a runtime");
    };
    assert!(matches!(error, RuntimeError::BundleMissing(_)));
    assert!(error.to_string().contains("npm run build"));
}

#[test]
fn the_catalogue_is_exactly_the_tools_the_gateway_knows() {
    let mut names = catalogue();
    names.sort_unstable();
    assert_eq!(
        names,
        vec![
            "agent.delegate_readonly",
            "artifact.create_approval_note",
            "artifact.create_briefing_deck",
            "artifact.create_calculation_workbook",
            "artifact.verify_docx",
            "calculation.evaluate_with_units",
            "capability.search",
            "knowledge.load_evidence_region",
            "knowledge.multimodal_retrieve",
            "knowledge.search_authorized",
            "media.extract_findings",
            "memory.promote_approved",
            "memory.recall_authorized",
            "sandbox.run_code",
            "sovereignty.get_evidence",
            "workspace.read_text",
            "workspace.write_text",
        ]
    );
}

/// The names in a record written before the rename still resolve to a tool.
///
/// The failure this prevents is quiet and late: an approval recorded months ago
/// says `create_docx`, and a reader that cannot resolve it shows the reviewer an
/// approval for a tool that appears not to exist — which reads as a corrupted
/// record rather than an old one.
#[test]
fn a_record_written_before_the_rename_still_names_a_real_tool() {
    use crate::orchestrator::tools::ToolName;

    for (legacy, expected) in [
        ("search_documents", ToolName::SearchDocuments),
        ("load_more_evidence", ToolName::LoadMoreEvidence),
        ("memory_recall_authorized", ToolName::MemoryRecallAuthorized),
        ("memory_promote_approved", ToolName::MemoryPromoteApproved),
        ("read_scoped_file", ToolName::ReadScopedFile),
        ("write_scoped_file", ToolName::WriteScopedFile),
        ("run_calculation", ToolName::RunCalculation),
        ("create_docx", ToolName::CreateDocx),
        ("create_xlsx", ToolName::CreateXlsx),
        ("execute_code", ToolName::ExecuteCode),
        ("validate_artifact", ToolName::ValidateArtifact),
    ] {
        assert_eq!(
            ToolName::from_str(legacy),
            Some(expected),
            "{legacy} no longer resolves"
        );
    }
}

/// Reading an old name must not make the system start writing it again.
///
/// A migration that accepted both spellings *and* emitted whichever it was
/// given would leave a record where the same tool appears under two names, and
/// no later reader could count calls to it without knowing both.
#[test]
fn resolving_a_legacy_name_still_writes_the_current_one() {
    use crate::orchestrator::tools::ToolName;

    let resolved = ToolName::from_str("create_docx").expect("legacy name resolves");
    assert_eq!(resolved.as_str(), "artifact.create_approval_note");
}

/// Deps with a plan registered, so the budget actually applies.
fn deps_with_plan(prompt: &str) -> (Arc<RuntimeDeps>, tempfile::TempDir) {
    let (deps, dir) = deps_with(signed_in_user());
    deps.plans
        .lock()
        .expect("fresh lock")
        .insert("r".to_string(), planning::plan_for("r", prompt));
    (deps, dir)
}

#[tokio::test]
async fn a_tool_outside_the_plan_is_refused_without_stopping_the_run() {
    // "summarise the report" plans no sandbox work, so execute_code is out. The
    // refusal has to leave the run able to carry on: one wrong guess by the
    // planner must not cost the whole task.
    let (deps, _dir) = deps_with_plan("summarise the inspection report");

    let refused = authorize(
        json!({
            "runId": "r",
            "toolCallId": "tc-1",
            "tool": "execute_code",
            "args": { "language": "python", "source": "print(1)" }
        }),
        &deps,
    )
    .await
    .unwrap();
    assert_eq!(refused["outcome"], "refuse");
    let reason = refused["reason"].as_str().unwrap();
    assert!(reason.contains("planned to use"), "{reason}");
    // It names what it *may* use, so the model can route around it.
    assert!(reason.contains("knowledge.search_authorized"), "{reason}");

    // And the run is still alive.
    let allowed = authorize(search("seal wear"), &deps).await.unwrap();
    assert_eq!(allowed["outcome"], "allow");
}

#[tokio::test]
async fn running_out_of_steps_stops_the_run_and_says_so() {
    let (deps, _dir) = deps_with_plan("what does the SOP say about seal wear?");
    let allowed = {
        let plans = deps.plans.lock().expect("fresh lock");
        plans.get("r").expect("a plan").budget.max_steps
    };

    // Spend the budget. Steps are counted on execution, so each one is a full
    // authorise-and-execute cycle — and each query differs, or the loop
    // detector stops the run first and this would be testing that instead.
    for i in 0..allowed {
        let mut call = search(&format!("seal wear question {i}"));
        call["toolCallId"] = json!(format!("tc-{i}"));
        let verdict = authorize(call.clone(), &deps).await.unwrap();
        assert_eq!(verdict["outcome"], "allow", "step {i} of {allowed}");
        call["grant"] = verdict["grant"].clone();
        let _ = execute(call, &deps).await;
    }

    let refused = authorize(search("one more thing"), &deps).await.unwrap();
    assert_eq!(refused["outcome"], "refuse");
    let reason = refused["reason"].as_str().unwrap();
    assert!(reason.contains("permitted steps"), "{reason}");
    // PS Part C: the incomplete plan is shown, not hidden.
    assert!(reason.contains("what was completed"), "{reason}");
}

#[tokio::test]
async fn the_same_call_over_and_over_is_stopped_as_going_in_circles() {
    // PS Part C: "Agent loop repeats → Stop at the step/time/tool budget."
    // Repeating one search is the shape that failure actually takes, and it
    // stops well before the step budget because it is making no progress.
    let (deps, _dir) = deps_with_plan("what does the SOP say about seal wear?");

    let mut outcomes = Vec::new();
    for i in 0..6 {
        let mut call = search("the identical question");
        call["toolCallId"] = json!(format!("tc-{i}"));
        let verdict = authorize(call.clone(), &deps).await.unwrap();
        outcomes.push(verdict["outcome"].as_str().unwrap().to_string());
        if verdict["outcome"] == "allow" {
            call["grant"] = verdict["grant"].clone();
            let _ = execute(call, &deps).await;
        }
    }

    let refusal = authorize(search("the identical question"), &deps)
        .await
        .unwrap();
    assert_eq!(refusal["outcome"], "refuse");
    let reason = refusal["reason"].as_str().unwrap();
    assert!(reason.contains("going in circles"), "{reason}");
    // Stopped short of the step budget, which is the point of detecting it.
    assert!(
        outcomes.iter().filter(|o| *o == "allow").count() < 12,
        "{outcomes:?}"
    );
}

#[tokio::test]
async fn a_run_with_no_plan_is_not_blocked_by_one() {
    // The health check and the runtime's own probes belong to no run. Refusing
    // every call for a run the plan table never heard of would break those
    // rather than enforce anything.
    let (deps, _dir) = deps_with(signed_in_user());
    let verdict = authorize(search("x"), &deps).await.unwrap();
    assert_eq!(verdict["outcome"], "allow");
}

#[tokio::test]
async fn a_search_that_finds_nothing_records_no_evidence_to_cite() {
    // The index is empty here, which is the case that matters most: a run that
    // retrieved nothing must end up with nothing citable, so the verifier
    // catches an answer that cites [E1] anyway.
    let (deps, _dir) = deps_with(signed_in_user());

    let allow = authorize(search("wall thickness"), &deps).await.unwrap();
    let mut call = search("wall thickness");
    call["grant"] = allow["grant"].clone();
    let result = execute(call, &deps).await.unwrap();

    assert!(result["text"].as_str().unwrap().contains("No passages matched"));
    assert!(retrieval::for_run(&deps.passages, "r").is_empty());
}

#[tokio::test]
async fn a_produced_file_is_remembered_so_it_can_be_re_opened() {
    let (deps, _dir) = deps_with(signed_in_user());
    let root = deps.root_for("r").expect("the run has a workspace");

    // Written directly: the point under test is the registry, and going through
    // write_scoped_file would need an approver on the other end.
    let path = root.join("draft.txt");
    std::fs::write(&path, b"some text").expect("wrote the draft");
    artifacts::remember(
        &deps.produced,
        "r",
        artifacts::produced_from(&path, Some(&root), artifacts::Kind::Text, None),
    );

    let reports = artifacts::report_for_run(&deps.produced, "r");
    assert_eq!(reports.len(), 1);
    assert_eq!(reports[0].name, "draft.txt");
    assert!(reports[0].sound);
}

#[tokio::test]
async fn a_produced_file_that_vanished_is_reported_as_missing_rather_than_sound() {
    // The failure this catches is a run that says it produced a deliverable and
    // a Tasks screen that agrees, over a file nobody can open.
    let (deps, _dir) = deps_with(signed_in_user());
    let root = deps.root_for("r").expect("the run has a workspace");
    let path = root.join("gone.docx");

    artifacts::remember(
        &deps.produced,
        "r",
        artifacts::produced_from(&path, Some(&root), artifacts::Kind::Document, Some("approval_note".into())),
    );

    let reports = artifacts::report_for_run(&deps.produced, "r");
    assert!(!reports[0].sound);
    assert!(reports[0].problems.iter().any(|p| p.contains("does not exist")));
}

/// The gateway when this installation cannot record what it does.
///
/// The rule these pin: **a read may still happen; an effect may not.** The
/// distinction is the one the product rests on. A search that is not written
/// down costs a line in a trace. A document written to disk that is not written
/// down is an artefact with no provenance — the thing an engineer would be
/// asked to sign, and the thing nobody could then stand behind.
///
/// Before this existed, a task event log that failed to open was replaced by an
/// in-memory one at start-up, and the application came up looking entirely
/// normal: it ran tasks, wrote files, and kept a history that evaporated when
/// the process exited.
mod durability {
    use super::*;
    use crate::agent_runtime::audit_health::AuditHealth;

    /// The same deps as everything else here, with a broken audit store.
    fn degraded_deps() -> (Arc<RuntimeDeps>, tempfile::TempDir) {
        let (deps, dir) = deps_with(signed_in_user());
        // The health record is behind an `Arc` inside the deps, so the deps do
        // not have to be rebuilt to degrade it -- which is also how it works in
        // production: one health record, shared, flipped by whatever fails.
        deps.audit_health
            .writes_failed("The task event log could not be opened: the disk is read-only.");
        (deps, dir)
    }

    fn write_call(text: &str) -> Value {
        json!({
            "runId": "r",
            "toolCallId": "tc",
            "tool": "workspace.write_text",
            "args": { "path": "note.txt", "content": text }
        })
    }

    #[tokio::test]
    async fn a_write_is_refused_while_the_record_cannot_be_written() {
        let (deps, _dir) = degraded_deps();
        let verdict = authorize(write_call("the seal is worn"), &deps)
            .await
            .expect("a verdict, not a transport error");
        assert_eq!(verdict["outcome"], "refuse");
        let reason = verdict["reason"].as_str().expect("a reason");
        // The model is told what is wrong, not merely that it may not.
        assert!(reason.contains("cannot record"), "{reason}");
        assert!(reason.contains("read-only"), "{reason}");
    }

    #[tokio::test]
    async fn a_read_is_still_allowed_while_the_record_cannot_be_written() {
        // Refusing reads too would leave a degraded installation unable even to
        // explain itself, and nothing a read does needs recording beyond the
        // event that names it.
        let (deps, _dir) = degraded_deps();
        let verdict = authorize(search("seal specification"), &deps)
            .await
            .expect("a verdict");
        assert_eq!(
            verdict["outcome"], "allow",
            "a read has no effect to account for"
        );
    }

    #[test]
    fn a_healthy_installation_refuses_no_write_on_durability_grounds() {
        // The control. Without it the test above would pass just as well
        // against a gateway that refused every write for some other reason.
        //
        // The guard is driven directly rather than through `authorize`,
        // because a write that gets *past* it needs a person to approve it and
        // `authorize` would then wait for a decision that never comes. What is
        // under test here is the guard, and this asks it exactly.
        let (deps, _dir) = deps_with(signed_in_user());
        let call = read_call(&write_call("the seal is worn")).expect("a well-formed call");
        assert_eq!(
            durability_refusal(&call, &deps),
            None,
            "a healthy installation must not refuse a write on durability grounds"
        );
    }

    #[test]
    fn a_write_is_refused_when_storage_breaks_part_way_through_a_run() {
        // The case a start-up check alone would miss: the run was already in
        // flight when the disk filled. It is also the worst one, because this
        // is the run that would otherwise write a document nobody can account
        // for.
        let (deps, _dir) = deps_with(signed_in_user());
        let call = read_call(&write_call("first")).expect("a well-formed call");
        assert_eq!(durability_refusal(&call, &deps), None);

        deps.audit_health
            .writes_failed("There is no space left on the device.");

        let refusal = durability_refusal(&call, &deps).expect("a refusal");
        assert!(refusal.contains("no space left"), "{refusal}");
        assert!(refusal.contains("workspace.write_text"), "{refusal}");
    }

    #[test]
    fn a_read_is_never_refused_on_durability_grounds() {
        // The other half of the rule, asked of the guard directly: no matter
        // how broken the record is, a read has no effect to account for.
        let (deps, _dir) = deps_with(signed_in_user());
        deps.audit_health.writes_failed("The disk is full.");
        let call = read_call(&search("seal specification")).expect("a well-formed call");
        assert_eq!(durability_refusal(&call, &deps), None);
    }
}

/// What actually breaks, and what it does to the installation's health.
///
/// The two stores that carry a run's record, driven into the two failures that
/// happen in the field: a database that will not open, and a disk that will not
/// take a write.
mod durability_failures {
    use crate::agent_runtime::audit_health::{AuditHealth, AuditState};
    use crate::agent_runtime::events::TaskEventLog;
    use crate::agent_runtime::tasks;

    /// A path that cannot be a directory, because it is a file.
    ///
    /// Portable across platforms in a way that permission bits are not: every
    /// filesystem this ships on refuses to create a directory where a regular
    /// file already is.
    fn blocked_path(tag: &str) -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join(tag);
        std::fs::write(&path, b"not a directory").expect("write the blocking file");
        (dir, path)
    }

    #[test]
    fn a_task_event_log_that_cannot_be_opened_reports_it_rather_than_substituting_one() {
        let (_dir, path) = blocked_path("data");
        let Err(error) = TaskEventLog::open(&path) else {
            panic!("the log must not open where a regular file already is");
        };

        // The store reports the failure. What start-up does with it is the
        // point: it may substitute an in-memory log so the window opens, but it
        // must mark the installation degraded rather than carry on as normal.
        let health = AuditHealth::degraded_at_startup(format!(
            "The task event log could not be opened: {error}"
        ));
        assert!(!health.is_durable());
        let refusal = health.refusal().expect("a reason to refuse runs");
        assert!(refusal.contains("will not run tasks"), "{refusal}");

        // And the read-only half of the bargain: an in-memory log still opens,
        // so past screens and settings work.
        assert!(
            TaskEventLog::in_memory().is_ok(),
            "the desktop must still open read-only"
        );
    }

    #[test]
    fn a_task_record_that_cannot_be_saved_degrades_the_installation() {
        let (_dir, path) = blocked_path("appdata");
        let record = tasks::tests::record("run-1", "2026-08-27T10:00:42+00:00");

        let error = tasks::save(&path, &record).expect_err("the record must not save here");

        // The behaviour under test: this failure is not merely logged. It flips
        // the installation, and the *next* run is refused with the reason.
        let health = AuditHealth::durable();
        assert!(health.is_durable());
        health.writes_failed(error.clone());
        assert!(!health.is_durable());

        let AuditState::Degraded {
            because,
            at_startup,
        } = health.state()
        else {
            panic!("a failed save must degrade the installation");
        };
        assert_eq!(because, error);
        assert!(
            !at_startup,
            "this one broke during the session, and the remedy differs"
        );
        assert!(
            health.refusal().expect("a reason").contains("Restart"),
            "a mid-session failure tells the person to restart once it is fixed"
        );
    }

    #[test]
    fn a_record_saves_and_reads_back_when_the_disk_is_working() {
        // The control for the test above: the same record, the same call, a
        // directory that works.
        let dir = tempfile::tempdir().expect("temp dir");
        let record = tasks::tests::record("run-1", "2026-08-27T10:00:42+00:00");
        tasks::save(dir.path(), &record).expect("the record saves");
        let health = AuditHealth::durable();
        assert!(health.is_durable());
        assert_eq!(health.refusal(), None);
    }
}

/// A run id this side never issued, and one whose run has ended.
///
/// ## The hole this closes
///
/// The plan is the authority for what a run may do, so a run id with no plan
/// has no authority at all. It used to have a good deal:
///
///   - `tool.catalogue` answered with every read-only tool in the product,
///     justified by "the runtime's health probe belongs to no run" — a probe
///     that does not ask for a catalogue. `health` is its own RPC, served in
///     `agent-runtime/src/main.ts` and answered without touching a tool.
///   - `plan_refusal` returned `None` for a missing plan, because its table
///     lookup was an early `?`. The caller read that as "no objection", so the
///     call went on to the gateway, which grants on the signed-in person's
///     permissions alone.
///
/// Put together: any string in the `runId` field could catalogue the read-only
/// tools and then authorise a search of the organisation's own documents,
/// outside every budget, with no step counted and no plan to hold it to.
///
/// The expired case is the same hole reached honestly. A plan is registered
/// when a run starts and released when it ends, so a call arriving a moment
/// after the run finished finds no plan — and used to be treated exactly like
/// a health probe.
mod missing_plan {
    use super::*;

    /// A run id this side has never issued.
    const INVENTED: &str = "run-that-never-existed";

    fn call_for(run_id: &str, tool: &str) -> Value {
        json!({
            "runId": run_id,
            "toolCallId": "tc",
            "tool": tool,
            "args": { "query": "seal specification" }
        })
    }

    /// Ends the harness's run the way finishing it does: the plan is released.
    fn release_plan(deps: &Arc<RuntimeDeps>, run_id: &str) {
        deps.plans
            .lock()
            .expect("plan table")
            .remove(run_id)
            .expect("the harness registered this plan");
    }

    /// A declared role with nothing behind it is not offered to the model.
    ///
    /// The shipped build registers profiles and no workers, so every
    /// delegation was refused with "the role is declared but this build has no
    /// worker for it" — after the model had spent a turn from a budget fixed
    /// before it started. The tool is withheld instead.
    #[tokio::test]
    async fn the_delegate_tool_is_withheld_when_no_worker_can_perform_a_role() {
        // A prompt whose plan permits delegation; the deps' manager has no
        // workers, exactly like the shipped application.
        let (deps, _dir) = deps_with_plan("Summarise the inspection report and draft a note");

        let planned = deps.plans.lock().expect("plan table")["r"]
            .budget
            .permitted_tools
            .clone();
        assert!(
            planned.contains(&ToolName::AgentDelegateReadonly),
            "this test is only meaningful while the plan still permits delegation"
        );

        let catalogue = tool_catalogue(json!({ "runId": "r" }), &deps).expect("a catalogue");
        let names: Vec<&str> = catalogue["tools"]
            .as_array()
            .expect("tools array")
            .iter()
            .filter_map(|tool| tool["name"].as_str())
            .collect();

        assert!(
            !names.contains(&ToolName::AgentDelegateReadonly.as_str()),
            "a tool that can only refuse was offered anyway: {names:?}"
        );
        // The rest of the catalogue is untouched — this withholds one tool, it
        // does not narrow the run.
        assert!(names.contains(&ToolName::SearchDocuments.as_str()));
    }

    #[tokio::test]
    async fn an_invented_run_id_cannot_catalogue_any_tool() {
        let (deps, _dir) = deps_with(signed_in_user());
        let error = tool_catalogue(json!({ "runId": INVENTED }), &deps)
            .expect_err("an unknown run must not be given a catalogue");
        assert_eq!(error.code, code::REFUSED);
        assert!(
            error.message.contains("no plan registered"),
            "{}",
            error.message
        );
    }

    #[tokio::test]
    async fn an_invented_run_id_cannot_authorise_any_tool() {
        let (deps, _dir) = deps_with(signed_in_user());
        // Read-only, and one the signed-in person plainly holds the permission
        // for. It is refused on the run, not on the person.
        let error = authorize(call_for(INVENTED, "search_documents"), &deps)
            .await
            .expect_err("an unknown run must not be authorised");
        assert_eq!(error.code, code::REFUSED);
        assert!(
            error.message.contains("no plan registered"),
            "{}",
            error.message
        );
    }

    #[tokio::test]
    async fn an_invented_run_id_cannot_execute_any_tool() {
        let (deps, _dir) = deps_with(signed_in_user());
        let error = execute(
            json!({
                "runId": INVENTED,
                "toolCallId": "tc",
                "tool": "search_documents",
                "args": { "query": "seal specification" },
                "grant": "forged-grant",
            }),
            &deps,
        )
        .await
        .expect_err("an unknown run must not execute");
        assert_eq!(error.code, code::REFUSED);
        // Refused on the run before the grant is even weighed, so a forged or
        // replayed grant never gets as far as the ledger.
        assert!(
            error.message.contains("no plan registered"),
            "{}",
            error.message
        );
    }

    #[tokio::test]
    async fn an_invented_run_id_cannot_read_the_skill_catalogue_either() {
        let (deps, _dir) = deps_with(signed_in_user());
        let error = capability_search(json!({ "runId": INVENTED, "query": "" }), &deps)
            .expect_err("an unknown run must not be told what this deployment can do");
        assert_eq!(error.code, code::REFUSED);
    }

    #[tokio::test]
    async fn an_expired_run_id_cannot_catalogue_authorise_or_execute() {
        // The same hole, reached honestly: a plan is registered when the run
        // starts and released when it ends, so a call arriving a moment after
        // the run finished finds none.
        let (deps, _dir) = deps_with(signed_in_user());

        // While it is alive, both work.
        assert!(tool_catalogue(json!({ "runId": "r" }), &deps).is_ok());
        let verdict = authorize(call_for("r", "search_documents"), &deps)
            .await
            .expect("a live run is authorised");
        assert_eq!(verdict["outcome"], "allow");

        release_plan(&deps, "r");

        assert!(
            tool_catalogue(json!({ "runId": "r" }), &deps).is_err(),
            "a run that has ended must not be given a catalogue"
        );
        assert!(
            authorize(call_for("r", "search_documents"), &deps)
                .await
                .is_err(),
            "a run that has ended must not authorise anything further"
        );
        assert!(
            execute(
                json!({
                    "runId": "r",
                    "toolCallId": "tc",
                    "tool": "search_documents",
                    "args": { "query": "seal specification" },
                    "grant": "any-grant",
                }),
                &deps,
            )
            .await
            .is_err(),
            "a grant issued before the run ended must not execute after it"
        );
    }

    #[tokio::test]
    async fn the_refusal_says_nothing_about_which_runs_do_exist() {
        // A caller probing ids learns only that this one is not one of them.
        let (deps, _dir) = deps_with(signed_in_user());
        let error = authorize(call_for(INVENTED, "search_documents"), &deps)
            .await
            .expect_err("refused");
        assert!(
            !error.message.contains("planned-without-workspace"),
            "{}",
            error.message
        );
        // It does name the id it was given, which is the caller's own string
        // and helps whoever is reading a trace.
        assert!(error.message.contains(INVENTED), "{}", error.message);
    }

    #[tokio::test]
    async fn a_registered_run_is_still_served_normally() {
        // The control. Without it, everything above would pass just as well
        // against a gateway that refused every call.
        let (deps, _dir) = deps_with(signed_in_user());
        let catalogue = tool_catalogue(json!({ "runId": "r" }), &deps).expect("a catalogue");
        assert!(
            !catalogue["tools"].as_array().expect("tools").is_empty(),
            "a live run must still be offered its planned tools"
        );
        let verdict = authorize(call_for("r", "search_documents"), &deps)
            .await
            .expect("a verdict");
        assert_eq!(verdict["outcome"], "allow");
    }
}

/// The budget when the gateway is asked by several calls at once.
///
/// The unit tests in `orchestrator::plan` drive the reservation directly. These
/// drive it the way the runtime does: through `authorize`, concurrently, on a
/// plan shared behind the same mutex production uses. Between them they cover
/// the two halves of the defect — the accounting, and the fact that the
/// accounting is reached under contention at all.
mod budget_contention {
    use super::*;

    /// A run whose plan has exactly `max_steps` to spend.
    fn deps_with_budget(max_steps: u32) -> (Arc<RuntimeDeps>, tempfile::TempDir) {
        let (deps, dir) = deps_with(signed_in_user());
        deps.plans.lock().expect("plan table").insert(
            "r".to_string(),
            crate::orchestrator::plan::PlanRun::new(
                "r",
                vec!["do the work".to_string()],
                crate::orchestrator::plan::Budget {
                    max_steps,
                    max_duration: std::time::Duration::from_secs(600),
                    permitted_tools: ToolName::ALL.to_vec(),
                    // High, so loop detection cannot be what refuses these.
                    repeat_limit: 100,
                },
            ),
        );
        (deps, dir)
    }

    fn search_call(tool_call_id: &str, query: &str) -> Value {
        json!({
            "runId": "r",
            "toolCallId": tool_call_id,
            "tool": "search_documents",
            "args": { "query": query }
        })
    }

    fn steps_committed(deps: &Arc<RuntimeDeps>) -> u32 {
        deps.plans
            .lock()
            .expect("plan table")
            .get("r")
            .expect("the plan")
            .steps_committed()
    }

    #[tokio::test]
    async fn four_concurrent_calls_against_one_free_slot_admit_exactly_one() {
        // The shape the runtime actually produces: `toolExecution: "parallel"`
        // puts every read-only call in a turn through `beforeToolCall` at once.
        // Before the slot was reserved during authorisation, all four read the
        // same unchanged step count and all four came back with a grant.
        let (deps, _dir) = deps_with_budget(1);

        let verdicts = futures_util::future::join_all((0..4).map(|i| {
            let deps = Arc::clone(&deps);
            async move {
                authorize(search_call(&format!("tc-{i}"), &format!("query {i}")), &deps)
                    .await
                    .expect("a verdict")
            }
        }))
        .await;

        let allowed = verdicts
            .iter()
            .filter(|verdict| verdict["outcome"] == "allow")
            .count();
        assert_eq!(
            allowed, 1,
            "exactly one call may hold the last slot; the others must be refused"
        );
        assert_eq!(steps_committed(&deps), 1);
    }

    #[tokio::test]
    async fn the_calls_that_lose_the_race_are_told_why_without_the_run_ending() {
        let (deps, _dir) = deps_with_budget(1);
        let verdicts = futures_util::future::join_all((0..3).map(|i| {
            let deps = Arc::clone(&deps);
            async move {
                authorize(search_call(&format!("tc-{i}"), &format!("query {i}")), &deps)
                    .await
                    .expect("a verdict")
            }
        }))
        .await;

        let refusals: Vec<&str> = verdicts
            .iter()
            .filter(|verdict| verdict["outcome"] == "refuse")
            .filter_map(|verdict| verdict["reason"].as_str())
            .collect();
        assert_eq!(refusals.len(), 2);
        for reason in &refusals {
            assert!(
                reason.contains("already under way"),
                "a call refused for want of a free slot should say so: {reason}"
            );
        }

        // Contention is a queue, not a budget spent. The run is still live.
        let stopped = deps
            .plans
            .lock()
            .expect("plan table")
            .get("r")
            .expect("the plan")
            .stopped()
            .cloned();
        assert!(
            stopped.is_none(),
            "losing a race for a slot must not end the run: {stopped:?}"
        );
    }

    #[tokio::test]
    async fn a_batch_larger_than_the_budget_never_commits_more_than_the_budget() {
        // Eight calls, three steps. The exact figure matters: this is the
        // assertion that would have read 8 before the reservation existed.
        let (deps, _dir) = deps_with_budget(3);
        let verdicts = futures_util::future::join_all((0..8).map(|i| {
            let deps = Arc::clone(&deps);
            async move {
                authorize(search_call(&format!("tc-{i}"), &format!("query {i}")), &deps)
                    .await
                    .expect("a verdict")
            }
        }))
        .await;

        let allowed = verdicts
            .iter()
            .filter(|verdict| verdict["outcome"] == "allow")
            .count();
        assert_eq!(allowed, 3);
        assert_eq!(steps_committed(&deps), 3);
    }

    #[tokio::test]
    async fn a_call_the_gateway_refuses_gives_its_slot_back_to_the_next_one() {
        // The plan admits before the gateway decides, so a call the gateway
        // then refuses took a slot it never used. If it kept it, a run with one
        // step left that tried one forbidden path would have nothing left for
        // the legitimate call the model tries next.
        let (deps, _dir) = deps_with_budget(1);

        // Refused by the gateway: a path outside the run's workspace.
        let refused = authorize(
            json!({
                "runId": "r",
                "toolCallId": "tc-outside",
                "tool": "read_scoped_file",
                "args": { "path": "/etc/passwd" }
            }),
            &deps,
        )
        .await
        .expect("a verdict");
        assert_eq!(refused["outcome"], "refuse");
        assert_eq!(
            steps_committed(&deps),
            0,
            "a call the gateway refused must not hold a slot"
        );

        // And the slot is genuinely there for the next call.
        let allowed = authorize(search_call("tc-next", "seal specification"), &deps)
            .await
            .expect("a verdict");
        assert_eq!(allowed["outcome"], "allow");
    }

    #[tokio::test]
    async fn a_run_with_room_still_authorises_normally() {
        // The control. Everything above would pass just as well against a
        // gateway that refused every second call.
        let (deps, _dir) = deps_with_budget(12);
        let verdicts = futures_util::future::join_all((0..4).map(|i| {
            let deps = Arc::clone(&deps);
            async move {
                authorize(search_call(&format!("tc-{i}"), &format!("query {i}")), &deps)
                    .await
                    .expect("a verdict")
            }
        }))
        .await;
        assert!(verdicts.iter().all(|verdict| verdict["outcome"] == "allow"));
        assert_eq!(steps_committed(&deps), 4);
    }
}

/// `capability.search`, called the way a model calls it.
///
/// ## The defect
///
/// The tool was in the catalogue and in the plan, and calling it produced an
/// error. `execute` fell through to `LocalToolRunner`, whose branch for it
/// reads "served on the agent path, not by this runner" — so a model asking
/// what skills were available was told the tool existed somewhere else.
///
/// Meanwhile a perfectly good handler existed and was reachable only over the
/// `capability.search` *RPC*, which a model cannot call. Two paths, one of them
/// dead.
///
/// These drive the whole sequence the runtime drives — authorise, take the
/// grant, execute with it — so a regression in any of the three shows up here
/// rather than as a model quietly losing a capability.
mod capability_search_execution {
    use super::*;

    fn call(query: &str) -> Value {
        json!({
            "runId": "r",
            "toolCallId": "tc-cap",
            "tool": "capability.search",
            "args": { "query": query }
        })
    }

    /// Authorises, then executes with the grant it was given.
    async fn authorize_then_execute(
        deps: &Arc<RuntimeDeps>,
        query: &str,
    ) -> Result<Value, WireError> {
        let verdict = authorize(call(query), deps).await.expect("a verdict");
        assert_eq!(
            verdict["outcome"], "allow",
            "capability.search is read-only and in the plan: {verdict:?}"
        );
        let grant = verdict["grant"].as_str().expect("a grant").to_string();

        let mut params = call(query);
        params["grant"] = Value::String(grant);
        execute(params, deps).await
    }

    #[tokio::test]
    async fn a_model_can_call_it_end_to_end() {
        let (deps, _dir) = deps_with(signed_in_user());
        let result = authorize_then_execute(&deps, "approval note")
            .await
            .expect("capability.search must execute, not report itself unavailable");
        let text = result["text"].as_str().expect("prose for the model");
        assert!(
            !text.contains("not by this runner"),
            "the tool reported itself unavailable: {text}"
        );
    }

    #[tokio::test]
    async fn an_empty_registry_says_so_rather_than_failing() {
        // The harness installs no skills. "Nothing matched" is an answer; an
        // error is not, and a model told the call failed retries it.
        let (deps, _dir) = deps_with(signed_in_user());
        let result = authorize_then_execute(&deps, "anything").await.expect("executes");
        let text = result["text"].as_str().expect("prose");
        assert!(text.contains("No installed skill matches"), "{text}");
    }

    #[tokio::test]
    async fn the_answer_is_metadata_only() {
        // The split the whole skill design rests on: cards here, instructions
        // only through a separate deliberate step.
        let (deps, _dir) = deps_with(signed_in_user());
        let result = authorize_then_execute(&deps, "").await.expect("executes");
        let text = result["text"].as_str().expect("prose").to_lowercase();
        assert!(
            text.contains("nothing was loaded") || text.contains("description only"),
            "the answer should say it is metadata: {text}"
        );
    }

    #[tokio::test]
    async fn a_run_with_no_plan_cannot_search_capabilities() {
        // Fail-closed, the same as every other method acting under a run's
        // authority. Asserted through the handler because a call with no plan
        // never gets a grant to execute with.
        let (deps, _dir) = deps_with(signed_in_user());
        let error = capability_search(json!({ "runId": "invented", "query": "" }), &deps)
            .expect_err("an unknown run must not be told what this deployment can do");
        assert_eq!(error.code, code::REFUSED);
    }

    #[tokio::test]
    async fn executing_without_a_grant_is_refused() {
        // The grant is what proves the gateway said yes. The tool being
        // read-only does not make it exempt.
        let (deps, _dir) = deps_with(signed_in_user());
        let error = execute(call("anything"), &deps)
            .await
            .expect_err("no grant was presented");
        assert_eq!(error.code, code::REFUSED);
    }
}

/// The dependencies the agent path used to build its runner without.
///
/// ## The defect
///
/// `execute` constructed the runner with `LocalToolRunner::new(index, session)`,
/// which leaves `multimodal`, `subagents`, `inherited` and `run_workspace` all
/// `None`. `LocalToolRunner` has fields for every one of them and the
/// application had built the values — they were simply never handed over.
///
/// Three tools in the catalogue were therefore unreachable. A model that
/// planned to use one, was granted it by the gateway, and called it, got back a
/// sentence saying the tool was available somewhere else.
mod runtime_wiring {
    use super::*;

    #[test]
    fn the_runner_the_agent_path_builds_carries_the_run_dependencies() {
        // The defect, stated as the thing that was `None`. `LocalToolRunner`
        // has a field for each of these and the application had built every
        // value; the agent path simply never handed them over.
        let (deps, _dir) = deps_with(signed_in_user());
        let session = deps.session().expect("signed in");
        let workspace = deps.root_for("r");
        let inherited = inherited_policy_for(&deps, &session, "r", workspace.as_deref());
        let runner = runner_for(&deps, &session, inherited.as_ref(), workspace.as_deref());

        assert!(runner.subagents.is_some(), "no subagent manager");
        assert!(runner.multimodal.is_some(), "no multimodal index");
        assert!(runner.inherited.is_some(), "no inherited policy");
        assert!(runner.run_workspace.is_some(), "no run workspace");
    }

    #[test]
    fn a_run_with_a_workspace_and_a_plan_yields_a_policy_a_child_can_inherit() {
        let (deps, _dir) = deps_with(signed_in_user());
        let session = deps.session().expect("signed in");
        let workspace = deps.root_for("r").expect("the harness gives run r a workspace");

        let inherited = inherited_policy_for(&deps, &session, "r", Some(&workspace))
            .expect("a run with a plan and a workspace has a policy to pass on");

        // Every field narrows a child and none widens it.
        assert_eq!(inherited.depth, 0);
        assert!(!inherited.network_permitted, "a child may not reach the network");
        assert!(inherited.approval_required, "a child may not skip approval");
        assert_eq!(inherited.workspace_root, workspace);
        assert_eq!(inherited.user_id, session.user.id);
    }

    #[test]
    fn a_child_is_never_given_a_tool_its_parent_does_not_hold() {
        // The property that makes delegation safe: the inherited tool list is
        // the parent's own, so there is no path by which a worker acquires
        // something the run was not planned for.
        let (deps, _dir) = deps_with(signed_in_user());
        let session = deps.session().expect("signed in");
        let workspace = deps.root_for("r").expect("a workspace");

        let permitted = registered_plan_tools(&deps, "r").expect("a plan");
        let inherited = inherited_policy_for(&deps, &session, "r", Some(&workspace))
            .expect("a policy");

        for tool in &inherited.permitted_tools {
            assert!(
                permitted.contains(tool),
                "{} reached a child without the parent holding it",
                tool.as_str()
            );
        }
    }

    #[test]
    fn a_run_with_no_plan_has_no_policy_to_pass_on() {
        // Fail closed, consistently with everything else that acts under a
        // run's authority. A worker started under an unknown run would be a
        // worker bounded by nothing.
        let (deps, _dir) = deps_with(signed_in_user());
        let session = deps.session().expect("signed in");
        let workspace = deps.root_for("r").expect("a workspace");
        assert!(
            inherited_policy_for(&deps, &session, "invented-run", Some(&workspace)).is_none()
        );
    }

    #[test]
    fn a_run_with_no_workspace_has_no_policy_to_pass_on() {
        // A child inherits its parent's workspace as its root. Without one
        // there is nothing to confine it to, and an unconfined worker is worse
        // than no worker.
        let (deps, _dir) = deps_with(signed_in_user());
        let session = deps.session().expect("signed in");
        assert!(inherited_policy_for(&deps, &session, "r", None).is_none());
    }

    #[tokio::test]
    async fn delegation_no_longer_reports_itself_unavailable() {
        // The symptom. Whatever the manager decides about a profile it does
        // not have, it must not be "subagents are not available on this
        // machine" — that sentence meant the wiring, not the request.
        let (deps, _dir) = deps_with(signed_in_user());
        let session = deps.session().expect("signed in");
        let workspace = deps.root_for("r");
        let inherited = inherited_policy_for(&deps, &session, "r", workspace.as_deref());
        let runner = runner_for(&deps, &session, inherited.as_ref(), workspace.as_deref());

        let call = crate::orchestrator::tools::ToolCall::new(
            "agent.delegate_readonly",
            json!({ "profile": "knowledge-retriever", "task": "find the seal specification" }),
        );
        let result = runner
            .run(ToolName::AgentDelegateReadonly, &call, None)
            .await;

        if let Err(reason) = &result {
            assert!(
                !reason.contains("not available on this machine"),
                "the manager was never handed to the runner: {reason}"
            );
            assert!(
                !reason.contains("not wired into this run"),
                "the inherited policy was never handed to the runner: {reason}"
            );
        }
    }

    #[tokio::test]
    async fn multimodal_retrieval_no_longer_reports_itself_unavailable() {
        // Same shape: the index exists, so a search that finds nothing must
        // say it found nothing rather than that the tool has no index.
        let (deps, _dir) = deps_with(signed_in_user());
        let session = deps.session().expect("signed in");
        let runner = runner_for(&deps, &session, None, None);

        let call = crate::orchestrator::tools::ToolCall::new(
            "knowledge.multimodal_retrieve",
            json!({ "query": "pump P-101 general arrangement" }),
        );
        let result = runner
            .run(ToolName::KnowledgeMultimodalRetrieve, &call, None)
            .await;

        if let Err(reason) = &result {
            assert!(
                !reason.to_lowercase().contains("not available"),
                "the multimodal index was never handed to the runner: {reason}"
            );
        }
    }
}
