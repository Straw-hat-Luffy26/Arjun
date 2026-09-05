# Context-memory implementation status — 5 September 2026

Audited checkout: `fa1cbbd334c15a164b48249068150c52c4194769` plus the existing local, uncommitted recovery changes. This is a source and test assessment, not verification of the installed desktop application. No application code was changed during this assessment.

The memory and small-context foundations are substantially implemented. Seamless transfer between different models and working specialist agents is still unfinished. There is no measured completion percentage or real-model retention score yet; counting modules would overstate delivery of the user's actual goal.

## Audit findings and follow-up status

1. **Different-model continuation is explicitly refused.** `src-tauri/src/commands/agent.rs:1182` rejects resume when the newly routed model differs from the checkpoint model. `agent-runtime/src/run.ts:260` constructs one model for the run. A controlled transition must rebuild a provider-valid context for the destination window while retaining the original objective, corrections, evidence, pending operations, approvals, and lifetime limits.
2. **Specialist-agent execution is not enabled in the desktop app.** `src-tauri/src/lib.rs:589` explains that no child workers are registered, and delegation is withheld. Packet, profile, policy, and manager infrastructure exists. The delegation caller at `src-tauri/src/orchestrator/runner.rs:891` also supplies an empty input-reference list. Register actual workers and wire authorized task references and result integration before claiming agent handoff works.
3. **Deterministic recovery acceptance is now complete (5 September follow-up).** The original audit's failure is resolved. The journey now uses the production `agent_runtime::task_driver::TaskDriver`, which the desktop start/resume command also calls. A real indexed search supplies evidence before compaction and two Node-worker restarts; the restored approval permits exactly one write, followed by read-back. Plan settlement uses successful tool receipts and the actual answer-verification report. A citation resolves against the restored passage, and the shared publication path saves the verdict before the terminal event. Companion tests reject an invented citation after recovery and completion without the planned search. The planner also no longer interprets permission wording ("after approval") as a request for a Word document. This remains a deterministic backend test: Tauri IPC, the router, authenticated resume dispatch, UI and native inference are not exercised. See [recovery acceptance](agent-runtime-recovery.md#deterministic-recovery-acceptance).
4. **Recovery is not yet an unattended application workflow.** Startup restores pending approvals and records interrupted runs (`src-tauri/src/lib.rs:501`). Manual resume IPC exists (`src-tauri/src/commands/agent.rs:3301`), but frontend callers for `agent_resume_run` and `agent_run_resumability` were not found. A durable pause API, Tasks-screen recovery controls, and authenticated restart dispatch remain to be completed.
5. **Memory lifecycle and failure recovery need more work.** The scoped memory store persists workspace/user items, while its separate run scope is transient (`src-tauri/src/agent_runtime/memory.rs:78,646,816`). Durable checkpoint notes and raw history already exist, but they do not automatically make every run-memory item durable. Same-key memory updates replace the old value (`memory.rs:629`); conflict/supersession history is not implemented there. Tool-operation replay and bounded retries for eligible reads exist; broader retry/backoff and ambiguous-effect reconciliation coverage remains incomplete. Failed summarization currently preserves state and refuses oversized input rather than providing an automatic recovery projection.
6. **Real small-context performance remains unverified.** Tests with scripted model responses cannot establish recall fidelity, successful tool use, or completion quality on installed local models. No native model-switch journey or measured long-task retention benchmark was run in this audit.

## What is already built

| Capability | Current evidence and boundary |
|---|---|
| Structured working notes | Goal, current stage, decisions, evidence/calculation/artifact references, questions, next action, and completed effects; bounded fields in `agent-runtime/src/working-notes.ts:67`. Tool observations update notes in the live loop. |
| Small-context input budgeting | `agent-runtime/src/context-budget.ts:14` caps estimated input at the smaller of 70% of the window or the remaining capacity after output and safety reserves. System/tool overhead is included by the compactor. These are estimates, not exact tokenizer counts. |
| Compaction and bounded summarizer requests | `agent-runtime/src/compaction.ts:517` reduces saved tool outputs, preserves tool-call/result pairing, and carries structured state with summaries. `agent-runtime/src/bounded-summary.ts:38` also bounds the summarizer's own input. Oversized requests are refused when compression cannot make them fit. |
| Exact underlying history and retrieval | `src-tauri/src/agent_runtime/events/context.rs:182` persists versioned context commits and append-only raw messages. Large saved tool results can be represented by previews and retrieved by transcript sequence through the authorized memory boundary. |
| Acknowledged live checkpoints | `agent-runtime/src/run.ts:313,385,555` saves context at model/tool/compaction boundaries. `src-tauri/src/agent_runtime/context_api.rs:11` carries the objective, conversation identity, deadline, plan progress, evidence, calculations, artifacts, and tool history. |
| Safe replay of completed operations | `src-tauri/src/agent_runtime/events/operations.rs:114` reuses durable receipts and rejects uncertain operations; `context_live_tests.rs:30` passes a real file-writer/reopened-store test without another write or step. This is evidence for the tested boundary, not a universal exactly-once guarantee. |
| Execution ownership and approval durability | Fenced writes, persisted exact-call approvals, expiry checks, and persistence-failure refusal are covered by the Rust agent-runtime suite. |
| Completion gating | The desktop command now calls `src-tauri/src/agent_runtime/task_driver.rs` for execution, evidence-based plan settlement, answer checking, fenced completion verification and outcome enforcement. The deterministic recovery journey passes through this shared driver and publication path. |
| Context visibility | `src/components/run/runAdopt.ts:269` loads and updates context ledgers and compaction records. This does not provide the missing pause/resume workflow. |
| Scoped long-term memory | Run/workspace/user scopes, classification/ACL checks, and approval-bound promotion exist. Durable fact-conflict history and complete run-memory restoration remain partial. |

## Checks run in the original audit (before the follow-up)

| Check | Result |
|---|---|
| `npm run runtime:typecheck` | Passed |
| Six focused context test files | 81 passed; included in the full runtime count below |
| `npm run runtime:test` | 2,089 passed across 126 files, including vendored OpenClaw tests |
| `cargo test --offline --manifest-path src-tauri/Cargo.toml --lib agent_runtime:: --no-default-features` | 428 passed; 1,367 other library tests filtered out |
| `npm run test:ui` | 295 passed across 20 files |
| `cargo test --offline --manifest-path src-tauri/Cargo.toml --test agent_runtime compaction_worker_restart --no-default-features -- --nocapture` | Failed at final completion verification; 0 passed, 1 failed, 10 filtered out |
| Actual local LLM, GPU model transition, full desktop restart and recovery UI | Not exercised |

The passing counts establish behavior within those suites. They are not all context tests, and they are not an end-to-end release certification.

## Recovery follow-up checks — 5 September 2026

These checks ran after the driver extraction and planner fix. Rust commands
used `--offline --no-default-features --target x86_64-pc-windows-msvc` in a
configured MSVC developer shell.

| Check | Result |
|---|---|
| `npm run runtime:build` | Passed; real Node bundle rebuilt |
| `npm run runtime:typecheck` | Passed |
| Rust library filter `agent_runtime::` | 429 passed, including the new planner regression |
| Rust library filter `commands::agent::` | 30 passed |
| `--test agent_runtime durable_journey -- --nocapture` | 3 passed, none skipped; successful recovery plus two completion refusals |
| `git diff --check` | Passed |

The original audit's full runtime/UI counts above were not rerun for this
backend-only change. Native inference, Tauri IPC and installed-app recovery
remain outside this follow-up's evidence.

## Recommended completion order

1. **Done in the follow-up:** finish the deterministic recovery journey through genuine plan and answer verification, including a test through the production task driver.
2. Implement an explicit model-transition checkpoint and destination-context builder; test transitions to smaller and larger windows with exact identifiers, corrections, pending tool batches, and approvals.
3. Register real specialist workers; populate authorized handoff references and persist their results into the parent task.
4. Complete durable task-memory lifecycle, provenance/conflict handling, and policy-aware restoration.
5. Connect startup recovery, bounded retry/reconciliation, pause/resume IPC, and frontend recovery controls.
6. Evaluate real local models on long multi-tool tasks spanning repeated compaction and model changes. Measure successful completion, retained facts/corrections, overflow refusals, duplicate effects, and recovery failures before making reliability claims.

## Research basis

The project design cites Anthropic's engineering guidance on [context engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents), [long-running agent harnesses](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents), and [Managed Agents architecture](https://www.anthropic.com/engineering/managed-agents). These support external durable state, compact working context, explicit progress, and verification. They do not guarantee lossless summaries or successful reasoning by every small model.

The existing architecture and recovery documents still contain early-stage status entries that predate several local implementation changes. Use the live-code findings and dated test results above for this assessment.
