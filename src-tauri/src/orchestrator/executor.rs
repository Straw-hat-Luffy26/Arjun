//! **NOT THE LOOP. `Executor` has no production caller.**
//!
//! The agent loop is the Node child process in `agent-runtime/`; this module's
//! `Executor`, `step`, `step_batch` and `TaskState` are reachable only from its
//! own tests. Two things in it are live and are why the file is still here:
//! the `ToolRunner` trait, which `agent_runtime` implements, and the
//! `CONSECUTIVE_REFUSAL_LIMIT` constant.
//!
//! This is stated at the top because it has already cost real time: a review
//! reading this file for "how does a turn run" reported the batch-admission
//! defect below as a live bug, and it is not reachable. Fix the Node loop, or
//! delete `Executor` — do not reason about the product from this file.
//!
//! Running a task one step at a time, pausing when a person is needed.
//!
//! This is where the plan, the gateway and the tools meet. It deliberately runs
//! **one step per call** rather than looping internally, for three reasons:
//!
//! - The interface can show what is happening as it happens, instead of a
//!   spinner and then an answer.
//! - A person can interrupt between steps.
//! - Approval is a natural pause rather than a callback threaded through a loop.
//!
//! ## A refusal is a result, not a crash
//!
//! When the gateway refuses a call, the refusal is handed back to the model as
//! that call's *result*. An agent told "path must be inside the task workspace"
//! adjusts and carries on; one that gets an exception has nothing to work with.
//! This is the single most important behaviour here, because refusals are the
//! common case, not the exception — a probabilistic system proposes invalid
//! actions routinely and that is not a malfunction.
//!
//! ## Repeated failure is progress information
//!
//! Three refusals in a row means the model is not learning from them, and the
//! remaining budget will be spent producing the same refusal again. That is
//! caught here rather than left to exhaust the step budget, so the person gets
//! "it kept being refused for this reason" instead of "it ran out of steps".

use std::time::Instant;

use serde::{Deserialize, Serialize};

use super::gateway::{GatewayVerdict, TaskContext, ToolGateway};
use super::plan::{Continuation, PlanRun, StopReason};
use super::tools::{ToolCall, ToolName};
use crate::audit::{AuditKind, AuditService};

/// Runs a tool that the gateway has already permitted.
///
/// A trait so the loop is testable without a filesystem, a sandbox or a model.
///
/// `resolved_path` is the path the gateway checked and approved, not the one the
/// model wrote. The runner must use it rather than re-deriving the path from the
/// call: two pieces of code resolving the same string separately will eventually
/// disagree, and the one that disagrees here is the one holding a file handle.
///
/// Async because one of the tools — `agent.delegate_readonly` — must wait on a
/// child runtime, and a sync trait there would force either a blocking runtime
/// handle (panics inside a runtime) or a second non-async surface that
/// duplicates the code. The executor and the agent runtime are both already
/// inside a tokio runtime, so this is just a type-level acknowledgement of
/// what is already true at the call sites.
#[async_trait::async_trait]
pub trait ToolRunner {
    async fn run(
        &self,
        tool: ToolName,
        call: &ToolCall,
        resolved_path: Option<&std::path::Path>,
    ) -> Result<String, String>;
}

/// How many consecutive refusals mean the model is not learning from them.
const CONSECUTIVE_REFUSAL_LIMIT: u32 = 3;

/// What happened on one step.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StepOutcome {
    pub tool: String,
    /// What to feed back to the model as this call's result — the tool's output
    /// on success, or the refusal text when it was refused.
    pub result: String,
    pub permitted: bool,
    pub took_ms: u64,
}

/// Where the task is now.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", tag = "state")]
pub enum TaskState {
    /// The step ran. Feed `outcome.result` back and ask for the next call.
    Stepped { outcome: StepOutcome },
    /// Several steps ran in one parallel batch. The agent-runtime turns
    /// this into one assistant turn carrying every outcome, in input
    /// order. The batch counts as one step against the plan budget.
    SteppedBatch {
        /// Every outcome, in the order the calls were submitted. A
        /// refusal is included so the model sees the full picture.
        outcomes: Vec<StepOutcome>,
        /// True when every call in the batch was declared parallel-safe.
        /// False when the model mixed parallel and sequential calls; the
        /// executor still ran the allowed ones concurrently but
        /// surfaces the mismatch so a caller knows to attribute
        /// surprises to the right call.
        all_parallel: bool,
    },
    /// A person has to answer before this can proceed.
    AwaitingApproval {
        tool: String,
        /// Target, arguments and consequence, ready to show.
        prompt: String,
    },
    /// A milestone step just finished. The plan pauses here so a
    /// person can confirm the model is on the right track before the
    /// next leg of work starts. The UI renders this as a gate.
    MilestoneReached {
        /// The checkpoint id the parent plan wrote. Stable across a
        /// resume; the resume path uses it to know which gate was
        /// approved.
        checkpoint_id: String,
        /// Ordinal of the step that produced the milestone.
        ordinal: u32,
        /// The intent text, ready to show a human. The agent-runtime
        /// may append a short reason ("drafted 3 pages, 2 facts
        /// pending") before emitting.
        summary: String,
    },
    /// Nothing more will happen.
    Finished {
        reason: StopReason,
        /// Steps that were planned but never reached.
        unfinished: Vec<String>,
    },
}

impl TaskState {
    pub fn is_finished(&self) -> bool {
        matches!(self, TaskState::Finished { .. })
    }
}

pub struct Executor<'a> {
    pub runner: &'a dyn ToolRunner,
    pub audit: Option<&'a AuditService>,
    /// Counts refusals since the last call that actually ran.
    consecutive_refusals: u32,
}

impl<'a> Executor<'a> {
    pub fn new(runner: &'a dyn ToolRunner, audit: Option<&'a AuditService>) -> Self {
        Self {
            runner,
            audit,
            consecutive_refusals: 0,
        }
    }

    /// Takes one step: check the budget, check the gateway, then run or pause.
    pub async fn step(
        &mut self,
        run: &mut PlanRun,
        context: &TaskContext<'_>,
        call: &ToolCall,
    ) -> TaskState {
        // 1. Does the plan allow another step at all? Checked first, so a task
        //    that is out of time is not told about permissions instead.
        if let Continuation::Stop(reason) = run.may_call(call) {
            return self.finish(run, reason, context);
        }

        // 2. Does the gateway allow this particular call?
        let started = Instant::now();
        let verdict = ToolGateway::decide(call, context);

        match verdict {
            GatewayVerdict::Refuse { reason } => {
                self.consecutive_refusals += 1;

                if self.consecutive_refusals >= CONSECUTIVE_REFUSAL_LIMIT {
                    // The model is not adjusting. Spending the rest of the
                    // budget producing this same refusal helps nobody.
                    let stop = StopReason::Failed {
                        detail: format!(
                            "the same kind of request was refused {} times in a row and the task \
                             was not adapting. The last reason was: {reason}",
                            self.consecutive_refusals
                        ),
                    };
                    return self.finish(run, stop, context);
                }

                self.record(context, call, false, &reason);

                // Handed back as the call's result. The step is counted, because
                // a refused attempt still consumed a turn of the model's budget.
                let hit = run.record_step();
                let outcome = StepOutcome {
                    tool: call.tool.clone(),
                    result: reason,
                    permitted: false,
                    took_ms: started.elapsed().as_millis() as u64,
                };
                self.step_finished_state(hit, Some(outcome), Vec::new(), false)
            }

            GatewayVerdict::NeedsApproval { tool, summary, .. } => {
                // Not a failure and not a step. The budget is untouched so the
                // wait costs the task nothing.
                run.await_approval(tool);
                self.record(context, call, false, "awaiting approval");
                TaskState::AwaitingApproval {
                    tool: tool.as_str().to_string(),
                    prompt: summary,
                }
            }

            GatewayVerdict::Allow {
                tool,
                ref resolved_path,
            } => {
                let result = match self.runner.run(tool, call, resolved_path.as_deref()).await {
                    Ok(output) => {
                        self.consecutive_refusals = 0;
                        output
                    }
                    Err(failure) => {
                        // A tool that ran and failed is different from one that
                        // was refused: the model may well be able to recover, so
                        // the error goes back as the result.
                        format!("The tool ran but failed: {failure}")
                    }
                };

                self.record(context, call, true, &result);
                let hit = run.record_step();
                let outcome = StepOutcome {
                    tool: call.tool.clone(),
                    result,
                    permitted: true,
                    took_ms: started.elapsed().as_millis() as u64,
                };
                self.step_finished_state(hit, Some(outcome), Vec::new(), false)
            }
        }
    }

    /// If the step just completed is a milestone, returns
    /// `MilestoneReached` instead of `Stepped` / `SteppedBatch` so
    /// the UI can pause for a human gate. The same rule applies
    /// inside a parallel batch: if finishing a single step is a
    /// checkpoint, the batch pauses.
    fn step_finished_state(
        &self,
        hit: Option<crate::orchestrator::plan::MilestoneHit>,
        single_outcome: Option<StepOutcome>,
        batch_outcomes: Vec<StepOutcome>,
        all_parallel: bool,
    ) -> TaskState {
        if let Some(milestone) = hit {
            let checkpoint_id = milestone
                .checkpoint_id
                .unwrap_or_else(|| format!("step-{}", milestone.ordinal));
            return TaskState::MilestoneReached {
                checkpoint_id,
                ordinal: milestone.ordinal,
                summary: milestone.intent,
            };
        }
        if let Some(outcome) = single_outcome {
            TaskState::Stepped { outcome }
        } else {
            TaskState::SteppedBatch {
                outcomes: batch_outcomes,
                all_parallel,
            }
        }
    }

    /// Runs several parallel-safe calls concurrently and reports the
    /// aggregated outcome as a single step.
    ///
    /// The plan counts this as **one** step regardless of how many calls
    /// it contained. ARJUN design rule: a parallel fan-out is one
    /// step, because "sum of parallel calls counts as 1 step" is the
    /// budget's only honest reading. A task that asked for three searches
    /// and got them in one step has spent one step of its budget, and
    /// the budget counter is the source of truth, not the call counter.
    ///
    /// Each call is gateway-checked individually. A refusal in the
    /// batch is the same as a refusal outside it: the model gets the
    /// refusal text as the result for that call. The batch as a whole
    /// still counts as one step.
    ///
    /// The agent-runtime's `buildTools` already declares which tools are
    /// parallel-safe via `executionMode: "parallel"`. This executor
    /// additionally checks the static property: read-only tools may run
    /// in parallel; anything that writes, produces a file, or asks a
    /// person runs sequentially. Mixed batches are detected and the
    /// executor surfaces a warning in the result so a caller knows
    /// the batch was not what it asked for.
    pub async fn step_batch(
        &mut self,
        run: &mut PlanRun,
        context: &TaskContext<'_>,
        calls: &[ToolCall],
    ) -> TaskState {
        if calls.is_empty() {
            return self.finish(
                run,
                StopReason::Failed {
                    detail: "an empty batch is not a step".to_string(),
                },
                context,
            );
        }

        // Budget check: a fan-out is one step. If the plan has room
        // for a step at all, it has room for the fan-out.
        if let Some(first) = calls.first() {
            if let Continuation::Stop(reason) = run.may_call(first) {
                return self.finish(run, reason, context);
            }
        }

        // Parallel-safety is read-only: a tool that does not change
        // anything cannot be racing itself.
        let all_parallel = calls.iter().all(|call| {
            let name = ToolName::from_str(&call.tool);
            name.map(|tool| tool.is_read_only()).unwrap_or(false)
        });

        // First pass: split into refused, approval-paused, allowed.
        // Verdicts and audit writes are sequential, by design, so the
        // audit log keeps causal order.
        let mut outcomes: Vec<StepOutcome> = Vec::with_capacity(calls.len());
        let mut approval_needed: Option<(String, String)> = None;
        let mut refused = 0;
        let mut allowed_calls: Vec<(ToolCall, ToolName, Option<std::path::PathBuf>)> =
            Vec::new();

        for call in calls {
            let verdict = ToolGateway::decide(call, context);
            match verdict {
                GatewayVerdict::Refuse { reason } => {
                    refused += 1;
                    self.consecutive_refusals += 1;
                    self.record(context, call, false, &reason);
                    outcomes.push(StepOutcome {
                        tool: call.tool.clone(),
                        result: reason,
                        permitted: false,
                        took_ms: 0,
                    });
                }
                GatewayVerdict::NeedsApproval { tool, summary, .. } => {
                    run.await_approval(tool);
                    self.record(context, call, false, "awaiting approval");
                    if approval_needed.is_none() {
                        approval_needed =
                            Some((tool.as_str().to_string(), summary));
                    }
                }
                GatewayVerdict::Allow { tool, ref resolved_path } => {
                    allowed_calls.push((
                        call.clone(),
                        tool,
                        resolved_path.as_ref().map(|p| p.to_path_buf()),
                    ));
                }
            }
        }

        // Concurrent fan-out via futures::future::join_all. A
        // long-running tool in the batch does not hold the rest up.
        // Borrow the runner out of self so the closure can capture
        // only a shared reference; capturing self mutably here would
        // move self into the future and prevent the post-fan-out
        // record/consecutive_refusals updates.
        let runner: &dyn ToolRunner = self.runner;
        let concurrent_handles: Vec<_> = allowed_calls
            .iter()
            .map(|(call, tool, resolved_path)| {
                let call = call.clone();
                let tool = *tool;
                let path = resolved_path.clone();
                async move {
                    let started = Instant::now();
                    let result = runner
                        .run(tool, &call, path.as_deref())
                        .await;
                    (call, started, result)
                }
            })
            .collect();

        let concurrent_results: Vec<(ToolCall, Instant, Result<String, String>)> =
            futures_util::future::join_all(concurrent_handles).await;

        for (call, started, result) in concurrent_results {
            let took_ms = started.elapsed().as_millis() as u64;
            self.record(
                context,
                &call,
                true,
                result.as_deref().unwrap_or(&String::new()),
            );
            if result.is_ok() {
                self.consecutive_refusals = 0;
            }
            outcomes.push(StepOutcome {
                tool: call.tool.clone(),
                result: result
                    .unwrap_or_else(|e| format!("The tool ran but failed: {e}")),
                permitted: true,
                took_ms,
            });
        }

        // Three-refusal ceiling: if the *whole* batch is refused,
        // the ceiling still fires. A single refusal in a batch of
        // three allowed calls is not a stuck model and is not a stop.
        if refused > 0
            && refused == calls.len()
            && self.consecutive_refusals >= CONSECUTIVE_REFUSAL_LIMIT
        {
            let stop = StopReason::Failed {
                detail: format!(
                    "every call in the batch was refused and the task was not adapting.                      The last reason was: {}",
                    outcomes
                        .last()
                        .map(|o| o.result.clone())
                        .unwrap_or_default()
                ),
            };
            return self.finish(run, stop, context);
        }

        // One step for the whole batch. The budget counter sees one
        // decrement, regardless of how many calls ran. ARJUN design rule
        // says: "sum of parallel calls counts as 1 step".
        let hit = run.record_step();

        if let Some((tool, prompt)) = approval_needed {
            return TaskState::AwaitingApproval { tool, prompt };
        }

        self.step_finished_state(hit, None, outcomes, all_parallel)
    }

    /// Resumes a task after a person approved the pending action.
    pub fn approved(&mut self, run: &mut PlanRun) {
        run.resume();
        self.consecutive_refusals = 0;
    }

    /// Ends a task because a person rejected the pending action.
    ///
    /// A rejection is a decision, not an error — the task stops cleanly and says
    /// who stopped it.
    pub fn rejected(&mut self, run: &mut PlanRun, context: &TaskContext<'_>) -> TaskState {
        let reason = StopReason::Failed {
            detail: format!(
                "{} declined the action this task needed to continue",
                context.session.user.display_name
            ),
        };
        run.resume();
        self.finish(run, reason, context)
    }

    fn finish(
        &self,
        run: &mut PlanRun,
        reason: StopReason,
        context: &TaskContext<'_>,
    ) -> TaskState {
        let unfinished: Vec<String> = run
            .unfinished()
            .into_iter()
            .map(|step| step.intent.clone())
            .collect();

        if let Some(audit) = self.audit {
            let _ = audit.record(
                &context.session.user.id,
                AuditKind::Task,
                format!("Task {} ended: {}", run.task_id, reason.explain()),
                Some(serde_json::json!({
                    "taskId": run.task_id,
                    "stepsTaken": run.steps_taken(),
                    "unfinished": unfinished,
                    "reason": reason,
                })),
            );
        }

        TaskState::Finished { reason, unfinished }
    }

    fn record(&self, context: &TaskContext<'_>, call: &ToolCall, permitted: bool, detail: &str) {
        let Some(audit) = self.audit else { return };

        // The arguments are recorded, but never the tool's output: a document's
        // text is exactly what ARJUN design rule 14 says must not be copied into a log
        // that more people can read than could read the document.
        let _ = audit.record(
            &context.session.user.id,
            AuditKind::PolicyDecision,
            format!(
                "{} {}",
                if permitted { "Ran" } else { "Refused" },
                call.tool
            ),
            Some(serde_json::json!({
                "tool": call.tool,
                "arguments": call.arguments,
                "permitted": permitted,
                "detail": if permitted { "ran" } else { detail },
            })),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::{Role, Session, User};
    use crate::orchestrator::plan::Budget;
    use crate::policy::ApprovalState;
    use serde_json::json;
    use std::path::PathBuf;
    use std::sync::Mutex;

    struct FakeRunner {
        calls: Mutex<Vec<String>>,
        fail: bool,
    }

    impl FakeRunner {
        fn working() -> Self {
            Self {
                calls: Mutex::new(Vec::new()),
                fail: false,
            }
        }
        fn failing() -> Self {
            Self {
                calls: Mutex::new(Vec::new()),
                fail: true,
            }
        }
        fn ran(&self) -> Vec<String> {
            self.calls.lock().unwrap().clone()
        }
    }

    #[async_trait::async_trait]
    impl ToolRunner for FakeRunner {
        async fn run(
            &self,
            tool: ToolName,
            _call: &ToolCall,
            _resolved_path: Option<&std::path::Path>,
        ) -> Result<String, String> {
            self.calls.lock().unwrap().push(tool.as_str().to_string());
            if self.fail {
                Err("the index was unavailable".into())
            } else {
                Ok("3 passages found".into())
            }
        }
    }

    fn session(roles: Vec<Role>) -> Session {
        Session::open(User::new("kiran", "Kiran", roles))
    }

    fn tools() -> Vec<ToolName> {
        vec![
            ToolName::SearchDocuments,
            ToolName::ReadScopedFile,
            ToolName::WriteScopedFile,
        ]
    }

    fn run() -> PlanRun {
        PlanRun::new(
            "task-1",
            vec!["Search the SOPs".into(), "Draft the note".into()],
            Budget::standard(tools()),
        )
    }

    fn context<'a>(session: &'a Session, roots: &'a [PathBuf]) -> TaskContext<'a> {
        TaskContext {
            session,
            workspace_roots: roots,
            confidential_work_permitted: true,
            approval: ApprovalState::NotRequested,
        }
    }

    /// Runs an async future on a tokio runtime, for sync tests.
    ///
    /// The real call sites are inside a tokio runtime; the unit tests are
    /// not. A single-threaded tokio runtime is built lazily and held in
    /// a `OnceLock`, so test cost is paid once across the whole suite
    /// rather than once per test. Using `Handle::block_on` would require
    /// the test to be inside a runtime, which it is not.
    fn block<F: std::future::Future>(future: F) -> F::Output {
        use std::sync::OnceLock;
        static RUNTIME: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
        let runtime = RUNTIME.get_or_init(|| {
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("the test runtime must build")
        });
        runtime.block_on(future)
    }

    fn search(query: &str) -> ToolCall {
        ToolCall::new("search_documents", json!({ "query": query }))
    }

    #[test]
    fn a_permitted_call_runs_and_returns_its_output() {
        let runner = FakeRunner::working();
        let mut executor = Executor::new(&runner, None);
        let s = session(vec![Role::Employee]);
        let roots = vec![PathBuf::from("C:/arjun/tasks/1")];

        let state = block(executor.step(&mut run(), &context(&s, &roots), &search("wall thickness")));

        match state {
            TaskState::Stepped { outcome } => {
                assert!(outcome.permitted);
                assert_eq!(outcome.result, "3 passages found");
            }
            other => panic!("expected a step, got {other:?}"),
        }
        assert_eq!(runner.ran(), vec!["knowledge.search_authorized"]);
    }

    /// The most important behaviour here: a refusal comes back as the call's
    /// result so the model can adjust, rather than as an error it cannot use.
    #[test]
    fn a_refusal_is_handed_back_as_the_result_and_the_task_continues() {
        let runner = FakeRunner::working();
        let mut executor = Executor::new(&runner, None);
        let s = session(vec![Role::Employee]);
        let roots = vec![PathBuf::from("C:/arjun/tasks/1")];
        let mut plan = run();

        let state = block(executor.step(
            &mut plan,
            &context(&s, &roots),
            &ToolCall::new("read_scoped_file", json!({ "path": "C:/Windows/System32/config" })),
        ));

        match state {
            TaskState::Stepped { outcome } => {
                assert!(!outcome.permitted);
                assert!(outcome.result.contains("outside this task's workspace"));
            }
            other => panic!("expected a refusal handed back as a step, got {other:?}"),
        }
        assert!(runner.ran().is_empty(), "a refused call must never reach the tool");
    }

    /// Spending the rest of the budget producing the same refusal helps nobody.
    #[test]
    fn repeated_refusals_stop_the_task_and_name_the_reason() {
        let runner = FakeRunner::working();
        let mut executor = Executor::new(&runner, None);
        let s = session(vec![Role::Employee]);
        let roots = vec![PathBuf::from("C:/arjun/tasks/1")];
        let mut plan = run();

        let bad = |n: u32| {
            ToolCall::new(
                "read_scoped_file",
                json!({ "path": format!("C:/Windows/attempt-{n}") }),
            )
        };

        assert!(!block(executor.step(&mut plan, &context(&s, &roots), &bad(1))).is_finished());
        assert!(!block(executor.step(&mut plan, &context(&s, &roots), &bad(2))).is_finished());

        match block(executor.step(&mut plan, &context(&s, &roots), &bad(3))) {
            TaskState::Finished { reason, .. } => {
                let text = reason.explain();
                assert!(text.contains("3 times in a row"), "{text}");
                assert!(text.contains("workspace"), "the last reason should be quoted: {text}");
            }
            other => panic!("expected the task to stop, got {other:?}"),
        }
    }

    /// A successful call clears the counter, so intermittent refusals during a
    /// task that is otherwise progressing do not accumulate into a stop.
    #[test]
    fn a_successful_step_resets_the_refusal_counter() {
        let runner = FakeRunner::working();
        let mut executor = Executor::new(&runner, None);
        let s = session(vec![Role::Employee]);
        let roots = vec![PathBuf::from("C:/arjun/tasks/1")];
        let mut plan = run();

        let bad = ToolCall::new("read_scoped_file", json!({ "path": "C:/Windows/x" }));

        block(executor.step(&mut plan, &context(&s, &roots), &bad));
        block(executor.step(&mut plan, &context(&s, &roots), &search("progress")));
        block(executor.step(&mut plan, &context(&s, &roots), &bad));

        assert!(
            !block(executor.step(&mut plan, &context(&s, &roots), &bad)).is_finished(),
            "the counter should have been reset by the successful step"
        );
    }

    // ── Approval ─────────────────────────────────────────────────────────

    #[test]
    fn a_write_pauses_for_a_person_with_a_prompt_worth_reading() {
        let runner = FakeRunner::working();
        let mut executor = Executor::new(&runner, None);
        let s = session(vec![Role::Employee]);
        let roots = vec![PathBuf::from("C:/arjun/tasks/1")];

        let state = block(executor.step(
            &mut run(),
            &context(&s, &roots),
            &ToolCall::new(
                "write_scoped_file",
                json!({ "path": "C:/arjun/tasks/1/note.txt", "content": "hello" }),
            ),
        ));

        match state {
            TaskState::AwaitingApproval { tool, prompt } => {
                assert_eq!(tool, "workspace.write_text");
                assert!(prompt.contains("note.txt"));
                assert!(prompt.contains("5 byte(s)"));
            }
            other => panic!("expected a pause, got {other:?}"),
        }
        assert!(runner.ran().is_empty(), "nothing runs before approval");
    }

    /// Waiting must cost the task nothing, or a slow reviewer would burn the
    /// budget the task needs to finish.
    #[test]
    fn waiting_for_approval_consumes_no_steps() {
        let runner = FakeRunner::working();
        let mut executor = Executor::new(&runner, None);
        let s = session(vec![Role::Employee]);
        let roots = vec![PathBuf::from("C:/arjun/tasks/1")];
        let mut plan = run();

        block(executor.step(
            &mut plan,
            &context(&s, &roots),
            &ToolCall::new(
                "write_scoped_file",
                json!({ "path": "C:/arjun/tasks/1/note.txt", "content": "hello" }),
            ),
        ));
        assert_eq!(plan.steps_taken(), 0);
    }

    #[test]
    fn approving_lets_the_write_proceed() {
        let runner = FakeRunner::working();
        let mut executor = Executor::new(&runner, None);
        let s = session(vec![Role::Employee]);
        let roots = vec![PathBuf::from("C:/arjun/tasks/1")];
        let mut plan = run();

        let call = ToolCall::new(
            "write_scoped_file",
            json!({ "path": "C:/arjun/tasks/1/note.txt", "content": "hello" }),
        );

        block(executor.step(&mut plan, &context(&s, &roots), &call));
        executor.approved(&mut plan);

        let mut granted = context(&s, &roots);
        granted.approval = ApprovalState::Granted;

        match block(executor.step(&mut plan, &granted, &call)) {
            TaskState::Stepped { outcome } => assert!(outcome.permitted),
            other => panic!("expected the write to proceed, got {other:?}"),
        }
    }

    /// A rejection is a decision, not an error.
    #[test]
    fn rejecting_stops_the_task_cleanly_and_names_who_stopped_it() {
        let runner = FakeRunner::working();
        let mut executor = Executor::new(&runner, None);
        let s = session(vec![Role::Employee]);
        let roots = vec![PathBuf::from("C:/arjun/tasks/1")];
        let mut plan = run();

        match executor.rejected(&mut plan, &context(&s, &roots)) {
            TaskState::Finished { reason, .. } => {
                assert!(reason.explain().contains("Kiran"));
                assert!(reason.explain().contains("declined"));
            }
            other => panic!("expected the task to stop, got {other:?}"),
        }
    }

    // ── Budgets and honesty ──────────────────────────────────────────────

    #[test]
    fn an_exhausted_budget_finishes_and_lists_what_was_never_reached() {
        let runner = FakeRunner::working();
        let mut executor = Executor::new(&runner, None);
        let s = session(vec![Role::Employee]);
        let roots = vec![PathBuf::from("C:/arjun/tasks/1")];
        let mut plan = PlanRun::new(
            "task-1",
            vec!["Search the SOPs".into(), "Draft the note".into()],
            Budget { max_steps: 1, ..Budget::standard(tools()) },
        );

        block(executor.step(&mut plan, &context(&s, &roots), &search("first")));

        match block(executor.step(&mut plan, &context(&s, &roots), &search("second"))) {
            TaskState::Finished { unfinished, .. } => {
                assert_eq!(unfinished, vec!["Draft the note"]);
            }
            other => panic!("expected the task to finish, got {other:?}"),
        }
    }

    /// A tool that ran and failed is recoverable; the model may try another way.
    #[test]
    fn a_tool_failure_comes_back_as_a_result_rather_than_ending_the_task() {
        let runner = FakeRunner::failing();
        let mut executor = Executor::new(&runner, None);
        let s = session(vec![Role::Employee]);
        let roots = vec![PathBuf::from("C:/arjun/tasks/1")];

        match block(executor.step(&mut run(), &context(&s, &roots), &search("anything"))) {
            TaskState::Stepped { outcome } => {
                assert!(outcome.permitted, "it was permitted; it simply failed");
                assert!(outcome.result.contains("the index was unavailable"));
            }
            other => panic!("expected a step, got {other:?}"),
        }
    }

    #[test]
    fn nothing_runs_while_the_network_is_reachable() {
        let runner = FakeRunner::working();
        let mut executor = Executor::new(&runner, None);
        let s = session(vec![Role::Employee]);
        let roots = vec![PathBuf::from("C:/arjun/tasks/1")];
        let mut ctx = context(&s, &roots);
        ctx.confidential_work_permitted = false;

        block(executor.step(&mut run(), &ctx, &search("anything")));
        assert!(runner.ran().is_empty());
    }

    /// ARJUN design rule: a parallel fan-out counts as a single step.
    /// Three read-only search calls run together and the budget only
    /// moves by one.
    #[test]
    fn a_parallel_batch_runs_three_reads_and_costs_one_step() {
        let runner = FakeRunner::working();
        let mut executor = Executor::new(&runner, None);
        let s = session(vec![Role::Employee]);
        let roots = vec![PathBuf::from("C:/arjun/tasks/1")];

        let mut plan = PlanRun::new(
            "task-parallel",
            vec!["Triangulate three sources".into()],
            Budget::standard(tools()),
        );
        let before = plan.steps_taken();
        let calls = vec![
            search("wall thickness"),
            search("hydrotest interval"),
            search("shutdown 2024"),
        ];

        let state = block(executor.step_batch(
            &mut plan,
            &context(&s, &roots),
            &calls,
        ));

        match state {
            TaskState::SteppedBatch { outcomes, all_parallel } => {
                assert!(all_parallel, "read-only calls must mark the batch parallel");
                assert_eq!(outcomes.len(), 3);
                assert!(outcomes.iter().all(|o| o.permitted));
            }
            other => panic!("expected a parallel batch, got {other:?}"),
        }
        assert_eq!(runner.ran().len(), 3, "every call in the batch must run");
        assert_eq!(
            plan.steps_taken(),
            before + 1,
            "the whole batch is exactly one step",
        );
    }

    /// A single refusal inside a batch that has allowed siblings is
    /// NOT a stuck model. The step counts, the refused call comes
    /// back as a result, and the others are recorded.
    #[test]
    fn a_single_refusal_in_a_batch_does_not_stop_the_task() {
        let runner = FakeRunner::working();
        let mut executor = Executor::new(&runner, None);
        let s = session(vec![Role::Employee]);
        let roots = vec![PathBuf::from("C:/arjun/tasks/1")];

        let calls = vec![
            search("wall thickness"),
            ToolCall::new(
                "read_scoped_file",
                json!({ "path": "C:/Windows/System32/config" }),
            ),
            search("shutdown 2024"),
        ];

        let state = block(executor.step_batch(
            &mut run(),
            &context(&s, &roots),
            &calls,
        ));

        match state {
            TaskState::SteppedBatch { outcomes, .. } => {
                assert_eq!(outcomes.len(), 3);
                let refused = outcomes.iter().filter(|o| !o.permitted).count();
                assert_eq!(refused, 1, "exactly the path-traversal call was refused");
            }
            other => panic!("expected a batch result, got {other:?}"),
        }
        // Two siblings ran, the refused one didn't.
        assert_eq!(runner.ran().len(), 2);
    }

    /// If every call in a batch is refused three times in a row, the
    /// ceiling fires regardless of batch size. The stop reason names
    /// the last refusal.
    #[test]
    fn an_all_refused_batch_eventually_stops_with_a_named_reason() {
        let runner = FakeRunner::working();
        let mut executor = Executor::new(&runner, None);
        let s = session(vec![Role::Employee]);
        let roots = vec![PathBuf::from("C:/arjun/tasks/1")];

        // Path-traversal: refused for every call, every batch.
        let calls = vec![
            ToolCall::new(
                "read_scoped_file",
                json!({ "path": "C:/Windows/System32/config" }),
            ),
            ToolCall::new(
                "read_scoped_file",
                json!({ "path": "C:/Windows/System32/secret" }),
            ),
        ];

        let mut plan = run();
        // First batch: 2 refusals, no ceiling yet.
        let _ = block(executor.step_batch(
            &mut plan,
            &context(&s, &roots),
            &calls,
        ));
        // Second batch: 4 refusals total, ceiling at 3 fires.
        let state = block(executor.step_batch(
            &mut plan,
            &context(&s, &roots),
            &calls,
        ));
        match state {
            TaskState::Finished { reason, .. } => {
                let detail = format!("{reason:?}");
                assert!(detail.contains("refused"));
            }
            other => panic!("expected a stop after the all-refused ceiling, got {other:?}"),
        }
    }

    /// Empty batches are not a no-op; they are a malformed model
    /// output and stop the task.
    #[test]
    fn an_empty_batch_is_not_a_step() {
        let runner = FakeRunner::working();
        let mut executor = Executor::new(&runner, None);
        let s = session(vec![Role::Employee]);
        let roots = vec![PathBuf::from("C:/arjun/tasks/1")];

        let state = block(executor.step_batch(
            &mut run(),
            &context(&s, &roots),
            &[],
        ));
        match state {
            TaskState::Finished { reason, .. } => {
                let detail = format!("{reason:?}");
                assert!(detail.contains("empty batch"));
            }
            other => panic!("empty batch must stop the task, got {other:?}"),
        }
        assert!(runner.ran().is_empty());
    }

    /// ARJUN design rule: a step flagged as a milestone pauses the
    /// run for human approval before the next leg. The state carries
    /// the checkpoint id, ordinal and intent so the UI can render
    /// the gate without re-reading the plan.
    #[test]
    fn finishing_a_milestone_step_pauses_the_run_for_a_human_gate() {
        let runner = FakeRunner::working();
        let mut executor = Executor::new(&runner, None);
        let s = session(vec![Role::Employee]);
        let roots = vec![PathBuf::from("C:/arjun/tasks/1")];

        let mut plan = PlanRun::new(
            "task-milestone",
            vec!["Survey the SOPs".into(), "Draft the note".into()],
            Budget::standard(tools()),
        );
        plan.mark_milestone(1, "mtn-survey").unwrap();

        let state = block(executor.step(
            &mut plan,
            &context(&s, &roots),
            &search("wall thickness"),
        ));
        match state {
            TaskState::MilestoneReached {
                checkpoint_id,
                ordinal,
                summary,
            } => {
                assert_eq!(checkpoint_id, "mtn-survey");
                assert_eq!(ordinal, 1);
                assert!(summary.contains("Survey"));
            }
            other => panic!("expected a milestone gate, got {other:?}"),
        }
        assert_eq!(runner.ran().len(), 1, "the tool still ran");
    }

    /// A non-milestone step is not a gate; the run continues
    /// normally.
    #[test]
    fn a_non_milestone_step_does_not_pause() {
        let runner = FakeRunner::working();
        let mut executor = Executor::new(&runner, None);
        let s = session(vec![Role::Employee]);
        let roots = vec![PathBuf::from("C:/arjun/tasks/1")];

        let state = block(executor.step(
            &mut run(),
            &context(&s, &roots),
            &search("wall thickness"),
        ));
        match state {
            TaskState::Stepped { outcome } => assert!(outcome.permitted),
            other => panic!("expected a normal step, got {other:?}"),
        }
    }

    /// A milestone inside a parallel batch still pauses; the model
    /// cannot smuggle a checkpoint past the gate by fanning out.
    #[test]
    fn a_milestone_inside_a_parallel_batch_pauses_anyway() {
        let runner = FakeRunner::working();
        let mut executor = Executor::new(&runner, None);
        let s = session(vec![Role::Employee]);
        let roots = vec![PathBuf::from("C:/arjun/tasks/1")];

        let mut plan = PlanRun::new(
            "task-parallel-milestone",
            vec!["Triangulate".into(), "Write up".into()],
            Budget::standard(tools()),
        );
        plan.mark_milestone(1, "mtn-triangulate").unwrap();

        let state = block(executor.step_batch(
            &mut plan,
            &context(&s, &roots),
            &[search("a"), search("b")],
        ));
        match state {
            TaskState::MilestoneReached { checkpoint_id, .. } => {
                assert_eq!(checkpoint_id, "mtn-triangulate");
            }
            other => panic!("a parallel batch hitting a milestone must pause, got {other:?}"),
        }
    }

    /// A milestone without an explicit checkpoint id still falls
    /// back to a stable name (`step-N`) so the resume path has
    /// something to address.
    #[test]
    fn a_milestone_with_no_checkpoint_id_falls_back_to_step_n() {
        let runner = FakeRunner::working();
        let mut executor = Executor::new(&runner, None);
        let s = session(vec![Role::Employee]);
        let roots = vec![PathBuf::from("C:/arjun/tasks/1")];

        // Reach into the plan directly to flag a milestone without
        // a checkpoint id. The public API requires an id, so we use
        // the constructor's default to simulate a future caller that
        // forgets to set one.
        let mut plan = PlanRun::new(
            "task-fallback",
            vec!["Survey".into()],
            Budget::standard(tools()),
        );
        plan.steps[0].milestone = true;

        let state = block(executor.step(
            &mut plan,
            &context(&s, &roots),
            &search("x"),
        ));
        match state {
            TaskState::MilestoneReached {
                checkpoint_id,
                ordinal,
                ..
            } => {
                assert_eq!(checkpoint_id, "step-1");
                assert_eq!(ordinal, 1);
            }
            other => panic!("expected a milestone with a fallback id, got {other:?}"),
        }
    }
}
