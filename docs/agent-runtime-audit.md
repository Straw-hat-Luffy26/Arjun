# Agent runtime audit — durable long-running tasks on a small context window

## Current audit: 2026-09-04, commit `fa1cbbd`

This section supersedes conflicting claims in the historical audit below. It
was checked against the current source, not inferred from earlier test counts.
Scope: the attachment's audit-first response; application implementation is
not part of this pass. Only this document is changed.

**Verdict: high risk; the requested reliability contract is not yet met.**
The repository is suitable for incremental implementation, but its Rust runtime
currently fails to compile. Most necessary components exist; several are only
partly connected, and some live failure paths permit execution without durable
records. Automatic resumption must not be enabled until those paths are fixed.

Audit method: trace entry points through authority checks, storage writes,
model context, tool execution, recovery and final status. Findings below name
the failing layer and the source evidence. Static findings describe reachable
code paths; they are not claims that a real crash simulation passed.

### Severity-ranked findings

1. **Critical — an intent-storage failure permits the side effect.**
   `events/store.rs:659` returns `EffectLookup::Fresh` on a poisoned lock or a
   database insertion error. `agent_runtime/mod.rs:1517` treats `Fresh` as
   permission to proceed. `store.rs:684` also suppresses settlement failures.
   Thus the intended intent-before-effect boundary is not enforced when storage
   fails. Return a typed error and stop before execution; an unrecordable result
   after execution must block further progress as ambiguous. Layer: persistence
   and tool execution. Confidence: 1.00, source-confirmed.

2. **Critical — the completion verifier does not gate completion.**
   `commands/agent.rs:1749` selects the terminal event from the model-loop
   outcome before verification. The verifier runs at `:1917`, but a failed
   report is only logged at `:1982`; the preselected ending is written at
   `:2083`. `agent_runtime/tasks.rs:428` independently computes readiness without
   checking the new verification record. A normally stopped model can therefore
   produce a `RunCompleted` event despite an unknown effect or failed criterion.
   Make the durable verifier verdict control terminal state and all UI readiness
   projections in the same committed transition. Layer: completion and answer
   presentation. Confidence: 1.00, source-confirmed.

3. **Critical — leases do not fence execution or state writes.**
   `commands/agent.rs:105` assigns one owner per process, and `:1313` acquires a
   claim. `events/lease.rs:148` allows the same owner to reclaim a live run, so
   two requests in one process are not excluded by the lease. The token is used
   on release, not on event/checkpoint/effect writes. `renew_claim` has no live
   caller; a lease expires after 300 seconds. Release deletes the row, allowing
   token numbering to restart on a later acquisition. Use unique attempt owners,
   monotonic retained fences, a heartbeat, and transactional fence checks at
   every advance and effect boundary. Layer: concurrency/persistence.
   Confidence: 1.00, source-confirmed.

4. **High — Rust baseline does not compile.** `events/store.rs:533`, `:544`,
   `:569`, `:579` and `:589` reference `MAX_RECOVERY_ATTEMPTS` without importing
   it from the parent module. The constant exists at `events/mod.rs:48`.
   The focused Rust test command fails with five E0425 errors before any test
   runs. Fix this first and establish a fresh baseline. Layer: build/runtime
   integration. Confidence: 1.00, compiler-confirmed.

5. **High — live checkpoints contain empty progress notes.** The only live
   tool-result checkpoint caller, `recording.rs:227`, supplies
   `RunMemory::default()`. `:128` supplies no context ledger. TypeScript updates
   its own notes after receiving the tool result (`tools.ts:257`), but those
   updates are not acknowledged into the checkpoint; final notes arrive only
   when `run.start` returns. Checkpoint reads also turn ledger-query failures
   into empty state (`recording.rs:110`). Persist the structured result,
   derived progress and checkpoint before releasing the next model step.
   Layer: checkpointing and distillation. Confidence: 1.00, source-confirmed.

6. **High — manual resumption is a fresh execution path under an old id.**
   `agent_resume_run` now really calls `drive_run`, so the historical claim that
   it never executes is obsolete. However, it drops attachments, scenario
   instructions and conversation/message identity (`commands/agent.rs:3313`).
   The shared driver reroutes the model, emits opening events (`:1401`), derives
   a new plan (`:1459`), resets the duration allowance (`:1513`) and generates an
   attempt id different from the `RunResumed` attempt (`:1487`, `:3268`). It does
   not restore completed step/budget state. There is also a concrete default
   classification mismatch: the opening records `Internal` when classification
   was absent (`:1412`), but the checkpoint hashes the original `None`
   (`:1490`), and resume hashes the recorded `Internal` (`:3420`). Such runs
   refuse with `PolicyChanged`. Build a distinct restore path with durable input
   references and one attempt identity; do not replay creation or reset budgets.
   Layer: task/session reconstruction. Confidence: 0.99, source-confirmed.

7. **High — approval persistence is not durable suspension/resumption.**
   Requests are now saved and pending requests restored at startup
   (`approval.rs:99`, `lib.rs:466`), correcting the old audit. But request-write
   errors only warn (`approval.rs:111`); decision writes follow the in-memory
   decision and also only warn (`commands/approvals.rs:45`, `:84`). Each call
   creates a new UUID, stores display arguments rather than the complete
   normalized call, and has no stored expiry despite a live wait timeout.
   `DurableApproval::authorises` is used only in tests. Restored approvals have
   no waiter; resolving one does not restart its run. Persist exact call-bound
   approvals and decisions first, suspend at a checkpoint, and consume an
   unexpired decision once through the Rust authorization boundary. Preserve
   requester identity and authorization scope on restore. Layer: human control
   and persistence. Confidence: 1.00, source-confirmed.

8. **High — recovery does not resume work, and its attempt limit cannot count
   repeated recoveries correctly.** Startup now marks suitable runs
   `RecoveryStarted`, rather than degrading every interrupted run
   (`events/store.rs:535`), but `lib.rs:519` only invokes the assessment and logs
   it. There is no dispatcher and no frontend caller for `agent_resume_run`
   (`src-tauri/ipc-manifest.json:80`). Every recovery-start event uses the same
   idempotency component `"recovery"`; duplicate events are ignored, preventing
   the counter from representing subsequent attempts. Startup also strands all
   pending effects without checking live leases. Add per-attempt event identity,
   lease-aware discovery and authenticated bounded dispatch. Layer: lifecycle
   and observability. Confidence: 1.00, source-confirmed.

9. **High — resumability checks stale unknown-effect state.**
   `commands/agent.rs:3380` loads a checkpoint; `checkpoint.rs:293` consults its
   stored unknown-effect list, not a fresh effect-ledger query. A later unknown
   effect can therefore be missed by that assessment. Conversely, startup makes
   an ambiguous run terminal, and reconciliation does not reopen it; the terminal
   check then prevents resumption even after a reviewer settles the effect.
   Reconcile and validate current pending/unknown intents inside the acquired
   lease transaction, then follow explicit reviewed-resume transitions without
   rewriting historical endings. Layer: recovery and idempotency.
   Confidence: 0.99, source-confirmed.

10. **High — there is no complete durable model/tool transcript.**
    `run.ts:590` retains the agent message history in process memory.
    `recording.rs:203` hashes/redacts tool-result content in the event log, while
    `events/idempotency.rs:220` caps replay text at 400 characters. Conversation
    JSON persists the visible chat, not the provider-valid raw tool exchanges.
    Compaction events persist counts, not the summary. A digest is useful audit
    evidence but cannot reconstruct lost content. Add an access-controlled
    transcript/body store with references and digests in the event log; do not
    widen access by copying confidential bodies into broadly readable events.
    Layer: canonical history and context reconstruction.
    Confidence: 0.99, source-confirmed.

11. **High — context measurement is not a hard request budget.**
    `compaction.ts:150` reserves about 20% and triggers at roughly 80%, rather
    than the requested configurable 65–75% safety threshold. The trigger at
    `:524` counts projected messages, while system instructions and tool schemas
    are accounted for separately. If nothing can be cut (`:539`) or summarizing
    fails (`:575`), the oversized projection is returned unchanged. No final
    whole-request admission check prevents an overflow. Keep pair-aware cuts,
    but budget all sections plus output/recovery reserves and fail or retry
    boundedly before making an over-budget provider call. Layer: context and
    hidden summarization failure. Confidence: 1.00, source-confirmed.

12. **High — retry and reconciliation policy exists only as metadata.**
    `tool_policy.rs` declares classes, backoff, retryability and reconciliation,
    but the live executor does not consume those policies. Unknown effects are
    settled manually, not by the declared artifact inspector. The deterministic
    key is based on run, tool and raw arguments (`idempotency.rs:82`), not a
    stable logical operation: distinct steps with identical arguments collapse,
    while alternate argument representations can differ. Failed calls replay
    their cached failure indefinitely. Add logical operation ids, normalized
    argument digests, durable retry counters and per-tool reconciliation adapters;
    retain fail-closed review for opaque executions. Layer: tool lifecycle.
    Confidence: 0.99, source-confirmed.

13. **Medium — task memory is volatile and conflicting values replace history.**
    `MemoryScope::Run` is explicitly non-durable (`memory.rs:78`); user/workspace
    scopes are persisted. Same-key values are replaced (`:632`) without a
    supersession/conflict history. Existing items already have `updated_at`,
    provenance, ACL, classification and optional approval, so these must be
    extended rather than reinvented. Add durable task facts, source-event links,
    confidence and explicit supersession/conflict records; retain the existing
    approved-promotion boundary for shared memory. Layer: long-term memory.
    Confidence: 1.00, source-confirmed.

14. **Medium — lifecycle vocabulary exceeds live behavior.** Compacting,
    recovering, paused and external-wait states now exist. However,
    `ModelRequested`, `CompactionStarted`, `WaitStarted` and `RunPaused` have no
    live producers in the Rust runtime. There is no durable pause/wait command or
    validated control-action union for update-plan/wait/complete. No test found
    drives the requested compaction → process restart → approval → final
    verification journey through the real controller. Wire events at actual
    boundaries, add typed actions and implement the deterministic crash harness.
    Layer: control protocol, tests and observability.
    Confidence: 0.99, source-confirmed.

Source paths above are relative to `src-tauri/src/agent_runtime/` unless prefixed
with `commands/` or `lib.rs` (relative to `src-tauri/src/`), or named TypeScript
files (relative to `agent-runtime/src/`). Source line numbers refer to `fa1cbbd`.

### Current execution and storage map

```text
React chat / Tasks
  -> Tauri commands::agent::{agent_start_run, drive_run}
     -> identity / permission / sovereignty checks
     -> attachments, deterministic planning, Rust model routing and serving
     -> Node child over stdio JSON-RPC: run.start
        -> vendored OpenClaw Agent + local model adapter
        -> tool.authorize -> Rust policy / approvals / single-use grant
        -> tool.execute   -> Rust tool executor / effect ledger
     -> durable event notifications + UI progress
     -> grounding and artifact checks / completion report / terminal record

Rust persistence
  +-- SQLite task_events: append-only triggers, per-run ordered events
  +-- SQLite task_snapshots: rebuildable UI projection
  +-- SQLite run_checkpoints: latest hash-checked checkpoint per run
  +-- SQLite task_tool_effects: pending / succeeded / failed / unknown
  +-- SQLite run_approvals and run_leases
  +-- Conversation JSON: owner-scoped visible messages
  +-- Task JSON: final answer, evidence, artifacts and notes at run end
  +-- Memory files: durable user/workspace scope; task scope remains volatile
```

The app is Tauri 2 / Rust, React 19 / TypeScript / Vite, a bundled Node runtime,
local llama-compatible serving, and Python document/memory sidecars. The
existing npm lockfiles, `rusqlite`, `serde`, `chrono`, `sha2`, `tokio` and vendored
agent packages provide the necessary building blocks. No new framework,
database server or dependency is justified by this audit.

Entrypoints are `src/main.tsx`, `src-tauri/src/{main,lib}.rs` and
`agent-runtime/src/main.ts`. User task commands are Tauri IPC, not REST job
routes. A separate optional loopback Axum gateway exposes `/v1/models`,
`/v1/chat/completions` and `/v1/messages` with SSE; it is not a durable task
worker. Its local-host/origin checks are separate from Tauri session permissions.
The existing inference scheduler is also not a crash-recovery dispatcher.

Failures currently have typed loop outcomes, run deadlines, cancellation,
provider transport helpers and tool-call repair. These do not provide a durable
per-step recovery policy. Existing compaction preserves call/result pairs and
keeps bounded notes, but context is still a projection of in-process history.
Tauri's build configuration bundles the Node runtime; offline deployment scripts
are already present. Recovery should run inside the Rust authority boundary,
not as a new daemon with wider privileges.

### State and migration approach

Keep `run_id` as the logical task id and `attempt_id` as the execution attempt;
introducing another task identifier is unnecessary unless the product needs
multiple independent runs under one task. Preserve conversation/message and
operator identity across attempts.

There is already a transactional `PRAGMA user_version` migration runner with
version 1 creating `run_approvals`. Leases and effect/checkpoint tables also have
existing preparation code. Append new migrations; do not edit a shipped
migration or delete old records. Proposed additions, not applied in this pass:

- Durable execution record: objective, criteria, phase, stable step ids,
  completed/remaining work, next action, attempt identity, state version,
  original deadline, consumed budget, per-step retry and recovery counters,
  current checkpoint/event reference and active fence.
- ACL-scoped request/transcript bodies: original input references, model
  messages, tool call/result pairs, summaries and content digests. Keep raw
  sensitive content out of the general event stream.
- Logical-operation ledger: normalized arguments or protected argument
  references, retry state, request/business identifier, response digest and
  reconciliation evidence. Pending and unknown are separate blocking states.
- Approval requester/scope/operation binding, expiry, consumed-decision marker;
  task-memory provenance/conflict records; completion evidence records.

Snapshots remain caches. A transactional event/state/checkpoint boundary with
fence and expected-version checks becomes the authority to advance. Run- and
event-level schema versions are not substitutes for optimistic state versions.

### Ordered implementation plan and proposed files

1. **Restore the build and fail closed at durable boundaries.** Fix the import;
   make intent creation, result settlement, approval decisions, checkpointing
   and completion persistence return actionable failures. Wire verifier outcome
   to terminal state and UI readiness. Files:
   `src-tauri/src/agent_runtime/{completion,recording,tasks}.rs`,
   `src-tauri/src/agent_runtime/events/{store,idempotency}.rs`,
   `src-tauri/src/commands/{agent,approvals}.rs` and runtime tests.

2. **Make one attempt the exclusive writer.** Add retained monotonic fencing,
   unique owners, heartbeat and compare-and-swap state. Acquire before restoring
   or appending resume events. Refuse stale writes and side effects. Files:
   `src-tauri/src/agent_runtime/events/{lease,store,migrations,checkpoint}.rs`,
   `src-tauri/src/agent_runtime/mod.rs`, `src-tauri/src/commands/agent.rs`.

3. **Persist enough state to restore a task.** Add execution and protected
   transcript stores; restore original identity, effective classification,
   policy, model constraints, attachments, plan progress and budget. Commit
   tool results and structured notes before the next model call. Files:
   `src-tauri/src/agent_runtime/{resume,recording,protocol,memory}.rs`,
   `src-tauri/src/agent_runtime/events/{checkpoint,model,projection,store,migrations}.rs`,
   `src-tauri/src/commands/agent.rs`, `agent-runtime/src/{run,main,protocol,tools}.ts`.
   New focused modules: `agent_runtime/execution.rs` and
   `agent_runtime/transcript.rs` under `src-tauri/src/`.

4. **Project a bounded context and commit compaction separately.** Introduce
   a whole-request budget (default safety fraction 0.70), deterministic trimming
   with pair preservation, typed next actions, durable summary versions and
   task-memory flushes. Failed compaction must not dispatch an oversized call.
   Files: `agent-runtime/src/{run,compaction,context-ledger,working-notes}.ts`,
   new `agent-runtime/src/context-builder.ts`, the transcript/memory modules,
   and context tests. Shared memory promotion remains explicitly approved.

5. **Drive bounded recovery and durable approvals.** Discover candidates only
   after lease checks; reconcile effects; authenticate the original operator;
   resume the same safe boundary. Persist waits and decisions; consume approval
   only for the unchanged call. Wire per-tool retry/reconciliation policy.
   Files: `src-tauri/src/agent_runtime/{resume,approval,tool_policy,mod}.rs`,
   `src-tauri/src/agent_runtime/events/{approvals,store,machine,model}.rs`,
   `src-tauri/src/orchestrator/approvals.rs`, `src-tauri/src/commands/{agent,approvals}.rs`,
   `src-tauri/src/{lib.rs,config/defaults.rs}`, `src-tauri/ipc-manifest.json`,
   `src/services/{agent.service,approvals.service}.ts`, and `src/pages/Tasks.tsx`.
   Add pause/resume/inspect controls without conflating pause with cancellation.

6. **Prove recovery and document deployment.** Add
   `src-tauri/tests/recovery_e2e.rs` using temporary stores, a deterministic
   provider, controlled clock and crash injection. Exercise actual controller
   paths, not just data structures. Add
   `docs/agent-runtime-architecture.md` and `docs/agent-runtime-recovery.md`
   with the implemented component diagram, supported failure boundaries,
   migration/backup procedure and verified task-control examples.

Proposed configuration (none added yet): context safety fraction and reserves,
`max_recovery_attempts`, `lease_duration`, `heartbeat_interval`,
`max_step_retries`, `max_total_iterations`, `max_run_duration`, approval expiry,
and protected-transcript retention. Validate relationships (heartbeat shorter
than lease; usable prompt budget after reserves), persist the effective run
policy, and do not reset lifetime limits on resumption.

### Recovery acceptance gates

- Database reopen tests for execution state, append-only events, checkpoint
  lookup, protected transcript and approvals; transactional migration rollback.
- Two workers, same-process duplicate resume, expiry, heartbeat loss and stale
  fences: exactly one can advance or authorize an effect.
- Crash before intent, after intent but before dispatch, after dispatch but
  before settlement, after settlement but before checkpoint, and after
  checkpoint: no lost acknowledged progress or blindly duplicated operation.
- Provider timeout, network timeout, context overflow, failed summarization and
  repeated recovery: bounded retries, durable reasons and review when ambiguous.
- Approval restart/rejection/expiry/changed arguments: no restored grant for a
  different call; resolving the original request resumes the proper task once.
- Oversized output and repeated compactions: hard whole-request budget, valid
  tool pairs, exact identifiers and retrievable unabridged history.
- False completion, missing/invalid required artifacts, required tests failing,
  unknown or pending intents, pending approvals/waits, incomplete work and
  evidence-read failures: none may yield success.
- Deterministic full journey: multistep task → large tool output → compaction →
  process restart → checkpoint restore → approval suspension → approval →
  artifact checks and independent completion verification.

### Checks executed in this audit

| Check | Current result |
|---|---|
| `npm --prefix agent-runtime test -- src/compaction.test.ts src/context-management.test.ts src/context-reconciliation.test.ts src/note-taking.test.ts src/run.test.ts` | 96 passed, 5 files |
| `npm run runtime:typecheck` | Passed |
| `npm run test:ui` | 294 passed, 20 files, including the command's TypeScript check |
| `npm run check:ipc` | Passed: 125 commands |
| `cargo test --offline --manifest-path src-tauri/Cargo.toml --lib agent_runtime::` | Compilation failed: five E0425 errors; no Rust tests executed |
| `git diff --check` | Passed for the audit-only change |

The Rust warnings are separate from the five blocking errors. The full
repository verification, sidecar suite, real-model run and crash/recovery E2E
simulation were not run. Historical test totals below are not current evidence.
No dependencies, configuration, database or generated bundles were changed.

### Assumptions, authority and handoff

- Preserve the Rust/Node authority split, local/offline deployment and existing
  permission checks. The model may propose an action; it may not authorize it.
- Startup may discover recoverable work before sign-in, but must not execute it
  as a privileged system user. Authenticated continuation is the safe default;
  unattended operation would need a separately authorized service identity.
- Arbitrary code and unsupported external effects cannot be promised exactly
  once. Unknown results require real reconciliation evidence or human review.
- Full history requires protected durable bodies, not disclosure in audit logs.
  Retention and deletion must remain compatible with the user's access rights.
- Existing task IPC includes `agent_start_run`, `agent_abort_run`,
  `agent_run_resumability`, `agent_resume_run`, `agent_task_snapshot`,
  `agent_task_events`, `agent_unknown_effects` and `agent_reconcile_effect`.
  These are not shell CLI commands. A durable pause API and frontend recovery
  flow are still proposed, so there are no verified pause/resume CLI examples
  to publish yet.

No manual migration is needed for this audit-only change. Before deploying a
future migration, back up the existing SQLite database consistently together
with associated protected state, verify an upgrade from a real v1 fixture, and
rebuild the Tauri/Node bundle. Do not claim the requested durability statement
until the acceptance gates above pass.

---

## Historical baseline and implementation notes

Retained for provenance. Statements and test results below describe earlier
stages and must not override the current audit above.

Phase 1 deliverable. This document records what ARJUN's agent runtime does
today, measured against the requirement that a long agentic task survive
context compaction, process restart, interruption, network failure and model
failure. It proposes an implementation approach and states its assumptions.

No production code was changed to produce this audit.

**Headline finding.** ARJUN is not a greenfield case. Seven of the nine
required capabilities are already built, tested and in the tree — a durable
append-only event log on SQLite, an explicit run state machine, world-hashed
checkpoints, an intent-before-effect idempotency ledger, pair-aware
compaction, a sectioned context ledger, and capped working notes. The gap is
narrower and more specific than "build a durable runtime", and it is
concentrated in one place: **the decision layer for resumption is complete and
the execution layer that would act on it is absent.** A run that is
interrupted is recorded faithfully, assessed correctly, and then never
continued.

---

## 1. System shape

ARJUN is a Tauri 2 desktop application. Three processes matter here.

| Process | Language | Role |
|---|---|---|
| Core | Rust (`src-tauri/`) | Owns policy, persistence, tools, the audit chain. Spawns and supervises the runtime. |
| Agent runtime | Node/TypeScript (`agent-runtime/`) | The agent loop. Vendors OpenClaw `agent-core`. Bundled by esbuild to one `.mjs`, spawned as a child, speaks JSON-RPC over stdio. Never opens a socket. |
| Sidecars | Python (`sidecars/`) | Document extraction, OCR, memory provider. |

The split is by authority, and `agent_runtime/mod.rs` states it plainly: *the
runtime may request; only the core decides.* The child process does not hold
the permissions, the workspace boundary or the sovereignty invariant, so
nothing inside it can widen what a run may do. This is the single most
important existing property and every proposal below preserves it.

Entry points: `src-tauri/src/main.rs` → `lib.rs` (Tauri builder, 125 IPC
commands, state registration), `agent-runtime/src/main.ts` (JSON-RPC
dispatch), `src/main.tsx` (React 19 UI).

### Test and verification baseline (measured 2026-09-04)

| Suite | Command | Result |
|---|---|---|
| Rust unit | `npm run test:rust` | 1712 passed, 3 ignored |
| Rust integration | `npm run test:integration` | agent_runtime + two_runtimes, 5 passed |
| Rust baseline | `npm run test:baseline` | pass |
| Agent runtime | `npm run runtime:test` | 2077 passed, 124 files |
| Frontend | `npm run test:ui` | 294 passed, 20 files |
| Python sidecar | `npm run test:sidecar` | 107 passed |

Gates: `check:targets`, `check:ipc`, `check:lint-budget`, `check:egress`,
`check:no-lora`, `check:reachable`, `check:bundle`, `check:deployment`,
`check:offline`, `check:generated`, `check:whitespace`. All pass except
`check:generated`, which fails for an unrelated pre-existing reason recorded
in §6 (R11).

---

## 2. Current execution flow

`commands::agent::agent_start_run` is the spine. In order:

1. Session and permission check (`require_permission(Permission::UseModel)`).
2. Attachments read through the OCR model; pages rendered.
3. Plan derived deterministically from the prompt (`agent_runtime::planning`).
4. Hardware probed, model routed (`registry::router`) — **the core routes, the
   runtime never picks a model**.
5. `serving::admission::admit` budgets VRAM and, if needed, releases another
   server; `llama-server` is started and awaited.
6. `run.start` JSON-RPC to the Node child with prompt, system prompt, routed
   model, deadline, prior `notes`, and `preserved` state.
7. The loop turns. Every tool call comes back as `tool.authorize` (may this
   happen) and then `tool.please` (do it). Both are answered by the core.
8. Lifecycle events stream back and are recorded via
   `agent_runtime::recording` into the event log, and published to the UI.
9. On ending, a typed `RunOutcome` is read from the loop's own report — not
   inferred from whether the JSON-RPC call resolved. This distinction is
   already correct and was hard-won; `outcome.rs` documents the bug it fixes.
10. Answer verified for grounding, artifacts re-opened from disk, task record
    written as one JSON file.

Progress is published through `agent_runtime::stages` with a contract worth
keeping: *a stage is emitted only when the work it names is actually starting.*
No timers, no interpolated percentages.

---

## 3. Current persistence behaviour

Four stores, with different shapes for different jobs. This separation is
sound and should be preserved.

### 3.1 Durable event log — `agent_runtime/events/store.rs`

SQLite, beside the audit database. This is the piece that most of the
requirement already rests on.

- **Append-only by trigger.** `task_events_is_append_only_update` and
  `..._delete` reject rewrites at the database, not in application code.
- **Atomic and ordered.** Each append is one `BEGIN IMMEDIATE` transaction
  that reads the run's tail and writes the next `seq`, under
  `UNIQUE (run_id, seq)`. Two racing writers cannot take the same number.
- **Redacted on the way in.** `model::redact` runs on every payload.
  Document text becomes a hash and a length. Nothing carries a message, a
  reasoning trace or a partial completion — deliberate, under ARJUN design
  rule 14.
- **42 event types**, `RunCreated` through `RunInterrupted`.
- **Snapshots** (`task_snapshots`) are an explicitly-labelled cache with a
  `seq` on them, foldable forward and rebuildable from events. There is no
  state a snapshot can hold that the events cannot correct.
- **Checkpoints** (`run_checkpoints`), one row per run, body is a serialised
  `RunCheckpoint`.

Schema is created with `CREATE TABLE IF NOT EXISTS`. There is **no
`user_version` and no migration runner** — see risk R7.

### 3.2 Task record — `agent_runtime/tasks.rs`

One JSON file per run, written **once, when the run ends**. Holds the answer,
evidence, calculations, plan, approvals and artifacts. Correct shape for a
document nobody appends to. Its end-of-run timing is the root of gap R3.

### 3.3 Conversations — `agent_runtime/conversations.rs`

One JSON file per conversation, write-to-`.tmp`-fsync-rename. Schema-versioned
with a v1→v2 in-place migration. This is the user-visible transcript.

### 3.4 Memory — `agent_runtime/memory.rs` + `memory_engine/`

`MemoryStore` with `MemoryItem { id, scope, kind, key, value, classification,
acl, source, approval, expires_at, created_at }`, ACL-checked and
classification-aware. `memory_engine/` adds providers (mock, Python sidecar),
retriever, injector, ranking.

---

## 4. Current failure, retry and recovery behaviour

This is where the gap lives, so it is worth being exact.

### 4.1 What happens on restart today

`lib.rs:461` calls `task_events.recover_interrupted(SYSTEM_ACTOR)` at startup.
It does two things:

1. Every still-`pending` side effect is promoted to `unknown` and a
   `ToolEffectUnknown` event is written — *"in flight when the process went
   away; nobody can say whether it took effect"*. This is correct and is the
   hard half of the problem, already solved.
2. Every run with events but no terminal event gets a `RunDegraded` event:
   *"Interrupted: the application closed while this was still running.
   Somebody needs to look at this before it is relied on."*

So **restart terminates interrupted runs rather than continuing them.** The
run is closed off honestly and a human is asked to look. Nothing resumes.

### 4.2 The resumption machinery that exists and is not driven

A complete, tested decision layer sits unused:

- `events/checkpoint.rs` — `RunCheckpoint` carrying `run_id`, `attempt_id`,
  `state`, `last_event_seq`, `notes: RunMemory`, `ledger`, `plan_hash`,
  `policy_hash`, `workspace_hash`, `model_id`, `unknown_effects`, `at`,
  `schema_version`, `checkpoint_hash`. `resumable_against(&WorldNow)`
  re-derives all three hashes from the world *as it is now* and refuses on any
  disagreement. `NotResumable` enumerates the refusals.
- `resume.rs` — `policy_hash`, `plan_hash_of`, `workspace_hash_of`,
  `ResumeContext::world()`, `Attempt::new`, `checkpoint_now`.
- `commands/agent.rs` — `agent_run_resumability` (read-only assessment) and
  `agent_resume_run`, both registered in `lib.rs`.

**`agent_resume_run` does not resume.** It assesses, constructs an `Attempt`,
writes a `RunResumed` event, writes an audit line, and returns the `Attempt`.
It never spawns the runtime and never re-drives the loop. The IPC manifest is
candid about it:

> `"consumer": "admin"`, *"Resumes an interrupted run. Reachable from the
> Tasks screen's recovery flow, which is not yet built; the command is the half
> that exists and is tested."*

### 4.3 The seeding defect

`agent_start_run` seeds a resumed run's notes at `commands/agent.rs:1410`:

```rust
let resumed_notes = tasks::load(&app_data_dir(&app)?, &run_id, ...)
    .ok()
    .and_then(|previous| previous.working_notes)
```

`tasks::load` reads `{run_id}.json` — **the record written when a run ends**.
An interrupted run never wrote one. So the notes are empty for exactly the
runs that most need them. The in-code comment reasons carefully about why it
does not reconstruct notes from the event history (the history records only
that compactions happened), and that reasoning is right about the *history* —
but it overlooks `run_checkpoints`, where `store.checkpoint(run_id)` returns a
`RunCheckpoint` whose `notes: RunMemory` field is precisely the honestly
recorded, non-reconstructed set of notes taken at the last safe point.

This is a small, contained fix with a large effect, and it is the first thing
to change.

### 4.4 Retry and network failure

- Model/provider transport: OpenClaw's retry package is vendored
  (`vendor/openclaw/packages/retry`). Provider quirks handled in
  `providers.ts`.
- Malformed tool calls from small quantised models are recovered by
  `repair.ts` (promotes plain-text tool calls into real ones) rather than by
  constraining the sampler.
- There is **no per-tool retry policy** and **no `retry_counts`** anywhere in
  durable state.

### 4.5 Approvals

`orchestrator/approvals.rs`:

```rust
pub struct ApprovalQueue { items: Mutex<Vec<ApprovalItem>> }
```

In-memory only. No file, no table. `approval.rs` implements waiting by making
`tool.authorize` *take longer* — the run is held inside a live JSON-RPC call
while a quarter-second poll waits for a decision. From the loop's point of
view a slow authorisation is indistinguishable from slow anything else, which
is an elegant property; but it means:

- an approval pending at process death is **lost**, and
- there is no durable `WAITING_FOR_APPROVAL` suspension — the waiting state
  lives in a blocked call stack, not on disk.

---

## 5. Current context-window handling

Strong, and closer to the requirement than any other area.

- **`compaction.ts`** — `RunCompactor` over OpenClaw's compaction.
  `settingsForWindow` reserves 20% of the window and keeps the most recent
  40%. `alignCutToPairs` and `pairingIsIntact` guarantee an assistant tool
  call is never separated from its tool result — Phase 4's pairing rule is
  already enforced and tested. `pruneStaleToolResults` trims oversized old
  results. Image blocks are counted as a flat token cost so rendered PDF pages
  are not read as ~0 tokens.
- **`context-ledger.ts`** — per-section accounting across `system`, `skill`,
  `toolSchema`, `evidence`, `notes`, `transcript`, `compaction`, `reserve`,
  with `occupied`, `committed`, `window`, `headroom`. `headroom` is signed on
  purpose: a negative value means the next turn does not fit, and clamping
  would report that as "exactly full".
- **`context-entities.ts`** — itemised layer beneath sections, with the
  invariant that entities of a section sum to that section's total, checked by
  `reconcileSections`. `evictionPlan` and `pressure` are derived, not invented.
- **`working-notes.ts`** — fixed-shape, hard-capped record; identifiers
  (`E3`, `C1`, `approval-note.docx`) never content. Overflow drops the oldest
  and increments a visible counter.
- **`note-taking.ts`** — notes derived from what tools actually returned,
  *not* written by the model. The rationale is exactly the requirement's: a
  model that has just produced a document does not think to note it, then the
  process dies and it produces the document again.

**What is missing:** the context is assembled from the loop's in-memory
transcript. It is not rebuilt from durable state. `RunRequest.notes` and
`preserved` exist as the seam through which durable state *could* be
projected, and `run.note` refreshes them — but the projection is one-way and
only at start.

---

## 6. Existing risks

| # | Risk | Severity |
|---|---|---|
| R1 | Interrupted runs are terminated as `RunDegraded`, never continued. The product cannot do the thing this work is for. | High |
| R2 | Approvals are in-memory; a pending approval does not survive restart, and there is no durable waiting state. | High |
| R3 | Resume seeds notes from the end-of-run record, which by construction does not exist for interrupted runs (§4.3). | High |
| R4 | No lease or fencing. Nothing prevents two workers advancing one run. Single-user desktop makes this low-likelihood today, but `agent_resume_run` plus a live run is already two writers. | Medium |
| R5 | No `state_version`; no optimistic concurrency on snapshot or checkpoint writes. | Medium |
| R6 | Tool taxonomy is binary (`is_side_effecting -> bool`) over five hardcoded tools. No reversibility/irreversibility distinction, no per-tool retry or reconciliation policy. | Medium |
| R7 | Event schema created with `CREATE TABLE IF NOT EXISTS`, no `user_version`, no migration runner. Adding a column later has no defined path. | Medium |
| R8 | `is_ready()` gates completion on failure, grounding verification, artifact soundness and unfinished plan steps — but **not** on unknown tool intents or pending approvals. | Medium |
| R9 | `VerificationStarted` has no corresponding outcome event; the verification *result* never enters the event log. | Low |
| R10 | Reconciliation has no provider-query step: `reconcile_effect` records a human's conclusion, nothing automatically asks "did this file get written?" | Low |
| R11 | Pre-existing, unrelated: the agent-runtime bundle is not byte-reproducible (esbuild 0.28.2 emits `"use strict"` non-deterministically in `__esm` wrappers), so `check:generated` cannot reliably pass. Not caused by and not addressed by this work. | Low |

---

## 7. Gap analysis against the requirement

Legend: **✅** present and adequate · **◐** partial · **❌** absent

### Statuses (Phase 2)

`events/machine.rs` has 16 states: `Created`, `Classified`, `Routed`,
`Planned`, `Running`, `AwaitingApproval`, `ExecutingTool`,
`ToolResultRecorded`, `Verifying`, `Completed`, `Cancelled`, `Failed`,
`StoppedByBudget`, `StoppedByLength`, `StoppedByPolicy`, `DegradedNeedsHuman`.

| Required | Present | Note |
|---|---|---|
| `QUEUED` | ◐ | `Created`/`Classified`/`Routed`/`Planned` cover the pre-run phases more finely. |
| `RUNNING` | ✅ | |
| `WAITING_FOR_APPROVAL` | ◐ | State exists; not durable (R2). |
| `WAITING_FOR_EXTERNAL_EVENT` | ❌ | |
| `COMPACTING` | ❌ | `ContextCompacted` event exists; no state. |
| `RECOVERING` | ❌ | |
| `PAUSED` | ❌ | |
| `SUCCEEDED` | ✅ | `Completed` |
| `FAILED` | ✅ | |
| `NEEDS_REVIEW` | ✅ | `DegradedNeedsHuman` |
| `CANCELLED` | ✅ | |

### Events (Phase 2)

Present: `TASK_CREATED`, `RUN_STARTED`, `TOOL_INTENT_CREATED`
(`ToolEffectPending`), `TOOL_STARTED` (`ToolAuthorized`), `TOOL_RESULT`,
`TOOL_FAILED`, `TOOL_RECONCILIATION`, `CHECKPOINT_CREATED`,
`COMPACTION_COMPLETED` (`ContextCompacted`), `MEMORY_FLUSHED`
(`MemoryPromoted`), `APPROVAL_REQUESTED`, `APPROVAL_RESOLVED`, `TASK_FAILED`,
`TASK_CANCELLED`.

Absent: `COMPACTION_STARTED`, `WAIT_STARTED`, `WAIT_COMPLETED`,
`RECOVERY_STARTED`, `RECOVERY_FAILED`, `COMPLETION_VERIFIED`.

Deliberately absent and **should stay absent**: `USER_MESSAGE`,
`ASSISTANT_MESSAGE`, `MODEL_RESPONSE` as *content*. ARJUN design rule 14
forbids copying confidential content into a record more people can read.
Message bodies live in the conversation store under ACL; the event log carries
hashes and lengths. I propose satisfying the requirement's intent with
`MODEL_REQUESTED`/`MODEL_RESPONDED` envelope events carrying token counts and
digests, never text. This is a deviation from the letter of the spec and is
called out as assumption A3.

### Checkpoint fields (Phase 2)

`RunCheckpoint` + its embedded `RunMemory { goal, stage, decisions,
evidence_ids, calculation_ids, artifact_ids, open_questions, next_action,
completed, milestones }`.

| Required | Present |
|---|---|
| `task_id` / `run_id` | ✅ (`run_id`; ARJUN has no separate task id) |
| `phase` | ✅ (`RunMemory.stage`) |
| `current_step` | ❌ no stable step identifier |
| `next_action` | ✅ |
| `completed_items` | ✅ (`completed: Vec<CompletedEffect>`) |
| `remaining_items` | ◐ (`open_questions` is adjacent, not the same) |
| `pending_approvals` | ❌ |
| `pending_tool_intents` | ◐ (`unknown_effects` only, not `pending`) |
| `retry_counts` | ❌ |
| `state_version` | ❌ (`schema_version` is a different thing) |
| `last_event_id` | ✅ (`last_event_seq`) |
| `created_at` | ✅ (`at`) |

### Capability summary

| Capability | Status |
|---|---|
| 1. Durable task state | ✅ event log + snapshots + state machine |
| 2. Canonical history | ✅ (redacted by design) |
| 3. Compaction & prompt projection | ◐ compaction excellent; projection not rebuilt from durable state |
| 4. Long-term memory | ◐ store exists; lacks confidence / supersedes / source_event_id / updated_at |
| 5. Execution checkpoints | ✅ decision layer; ❌ not acted on |
| 6. Tool intent/result tracking | ✅ |
| 7. Idempotency & reconciliation | ✅ intent-before-effect; ◐ no provider query, binary taxonomy |
| 8. Human approval state | ❌ in-memory |
| 9. Completion verification | ◐ grounding + artifact re-opening; not criterion-by-criterion, does not gate on unknown intents or pending approvals |

---

## 8. Proposed implementation approach

Ordered so each step is independently valuable and independently testable, and
so the highest-risk item is not last. No new framework and no new dependency —
`rusqlite`, `serde`, `chrono` and the vendored OpenClaw packages already
provide everything needed.

**Step 1 — Seed resumption from the checkpoint (fixes R3).**
Change `agent_start_run` to prefer `events.checkpoint(&run_id)`'s
`notes: RunMemory` and fall back to `tasks::load`. Small, contained, unblocks
every later step. Test: resume an interrupted run with no task record and
assert the notes arrive.

**Step 2 — Durable approvals (fixes R2).**
Add an `approvals` table (`approval_id`, `run_id`, `tool`, `args_fingerprint`,
`arguments`, `reason`, `status`, `allowed_decisions`, `created_at`,
`expires_at`, `resolved_at`, `resolved_by`, `resolution`). Keep
`ApprovalQueue` as the in-memory read cache, rehydrated at startup.
Re-validate on resume: an approval whose `args_fingerprint` no longer matches,
or whose `expires_at` has passed, is not valid — reuse
`idempotency::args_fingerprint` so "same arguments" means the same thing in
both subsystems.

**Step 3 — Lease and fencing (fixes R4, R5).**
Add `lease_owner`, `lease_expires_at`, `fence_token` and `state_version` to
`task_snapshots`. Acquire on start and on resume; renew on heartbeat; release
on ending. Snapshot writes become conditional on `state_version`.

**Step 4 — New states and events (fills the Phase 2 tables).**
`Compacting`, `Recovering`, `Paused`, `WaitingForExternalEvent` in
`machine.rs` with their transitions; `CompactionStarted`, `WaitStarted`,
`WaitCompleted`, `RecoveryStarted`, `RecoveryFailed`, `CompletionVerified`,
`ModelRequested`, `ModelResponded` in `model.rs`. Requires the migration
runner (Step 7) because `RunState::from_str` and `TaskEventType::from_str`
must keep reading old rows.

**Step 5 — Drive the resumption (fixes R1).**
The substantial one. `agent_resume_run` gains an execution half: after
recording `RunResumed`, rebuild the projected context from durable state
(checkpoint notes + reconciled intents + recent event-derived pairs), then
call the same `run.start` path with `notes` and `preserved` populated. Startup
recovery changes from "mark degraded" to "assess, and where `Resumable`, mark
`Recovering` and re-drive; where not, keep today's honest `RunDegraded`".
Bounded by `max_recovery_attempts`.

**Step 6 — Completion verifier (fixes R8, R9).**
Extend `is_ready()` to fail closed on any `unknown_effect` and any unresolved
approval. Add a `CompletionVerification { criterion_id, status, evidence,
verified_at, verifier_version }` record, written as a `CompletionVerified`
event. Keep the existing grounding verifier and artifact re-opening as two of
the criteria rather than replacing them.

**Step 7 — Migration runner (fixes R7).**
`PRAGMA user_version`, a small ordered list of migrations, applied in one
transaction at open. Needed by Steps 2, 3 and 4.

**Step 8 — Tool policy table (fixes R6, R10).**
Widen the binary `is_side_effecting` into `ToolClass { ReadOnly, Reversible,
SideEffecting, Irreversible }` plus `RetryPolicy { max_retries, backoff,
idempotent, reconciliation_method, safe_to_retry, requires_approval }`, as a
`const` table beside `ToolName` so it cannot drift from the catalogue. A
`reconciliation_method` for file-producing tools can genuinely answer "did this
happen?" by re-opening the path — `artifacts::check` already does exactly that.

**Step 9 — Observability and the end-to-end recovery test.**
The deterministic multi-step task from Phase 12: large tool output → forced
compaction → simulated restart → resume from checkpoint → approval → resume →
artifact and criteria verification.

### Files expected to change

Modified: `agent_runtime/events/{model,machine,store,checkpoint,projection,idempotency}.rs`,
`agent_runtime/{resume,approval,tasks,recording}.rs`,
`orchestrator/approvals.rs`, `commands/agent.rs`, `lib.rs`,
`src-tauri/ipc-manifest.json`, `agent-runtime/src/{run,main}.ts`.

New: `agent_runtime/events/migrations.rs`, `agent_runtime/lease.rs`,
`agent_runtime/completion.rs`, `agent_runtime/tool_policy.rs`,
`agent_runtime/events/approvals.rs`, plus tests alongside each and
`src-tauri/tests/recovery_e2e.rs`.

Docs: `docs/agent-runtime-architecture.md`, `docs/agent-runtime-recovery.md`.

---

## 9. Assumptions

- **A1.** `run_id` is the task id. ARJUN has no separate task/run split; a
  resumption is a new `attempt_id` under the same `run_id`, which
  `RunCheckpoint` already models. I will not introduce a second identifier.
- **A2.** SQLite beside the audit database is the right store. No new database
  and no new dependency.
- **A3.** Message *content* stays out of the event log. `MODEL_REQUESTED` /
  `MODEL_RESPONDED` will carry token counts and digests only. This deviates
  from the literal event list and is required by ARJUN design rule 14 and by
  the fact that `checkpoint.rs` is read during recovery *before anyone has
  authenticated*.
- **A4.** Single-user desktop deployment. Leases are for crash-and-restart and
  for the resume-versus-live-run race, not for a worker pool.
- **A5.** "Provider confirmation" for reconciliation means the local
  filesystem or the local inference server. ARJUN is air-gapped by design;
  there is no remote provider to query, and the egress gate would refuse one.
- **A6.** The existing authority split is invariant. The Node runtime will not
  be given policy, approval or persistence authority as part of this work.
- **A7.** No existing behaviour is removed. `RunDegraded` on unresumable runs
  stays exactly as it is; resumption is added alongside it.

---

## 10. Recommended scope decision

Steps 1–3, 6 and 7 are contained and low-risk, and together deliver durable
approvals, correct resume seeding, safe concurrency and a completion verifier
that fails closed. Step 5 is where the real behaviour change lives and is the
largest single piece.

I recommend implementing Steps 1, 2, 3, 6 and 7 as one reviewable change, then
Step 5 with its end-to-end recovery test as a second — because re-driving a
run touches the live execution path and deserves to be reviewed without four
other subsystems moving underneath it.

---

## 11. Implementation progress

Updated as steps land. Every entry below is verified by the test counts in the
table at the end, not by inspection.

| Step | Status | What changed |
|---|---|---|
| 1 — Seed resumption from the checkpoint (R3) | **Done** | `commands/agent.rs`: `notes_to_resume_from` extracted and given the record-then-checkpoint precedence. An interrupted run now resumes with the notes its last checkpoint recorded instead of nothing. |
| 7 — Migration runner (R7) | **Done** | `events/migrations.rs`: `PRAGMA user_version`, ordered list, one transaction. Wired into `store.rs::from_connection` after the baseline batch. |
| 2 — Durable approvals (R2) | **Storage done; wiring pending** | `events/approvals.rs`: `run_approvals` table, `DurableApproval`, `authorises()` enforcing status, unchanged-arguments and expiry. Store API on `TaskEventLog`. **Not yet wired** into `ApprovalQueue` or the `tool.authorize` path — see below. |
| 3 — Lease and fencing (R4, R5) | Not started | |
| 4 — New states and events | Not started | |
| 5 — Drive the resumption (R1) | Not started | The headline capability. |
| 6 — Completion verifier (R8, R9) | Not started | |
| 8 — Tool policy table (R6, R10) | Not started | |
| 9 — Observability + end-to-end recovery test | Not started | |

### What Step 2 does and does not yet do

The storage layer is complete and tested: a request can be written before
anybody is asked, a decision recorded once and only once, pending requests
listed after a restart, and an approval refuses to authorise a call whose
arguments changed or whose window closed.

Nothing calls it yet. `ApprovalQueue` is still the only thing the live
`tool.authorize` path consults, so **approvals still do not survive a restart
in the running product.** That wiring touches the live authorisation path and
is deliberately held back as its own change — the storage half is additive and
provably correct on its own, and reviewing it separately from a change to how
authorisation behaves is the point of splitting them.

### Correction to §7

The audit's checkpoint table lists `pending_approvals` as absent from
`RunCheckpoint`. That remains true. It cannot be filled until Step 2 is wired,
because until then there is no durable pending-approval set to put in it.

### Verification after each step

| Suite | Baseline | After Steps 1, 7, 2-storage |
|---|---|---|
| `test:rust` | 1712 passed | **1731 passed**, 0 failed |
| `test:integration` | 15 passed | **15 passed** |
| `test:ui` | 294 passed | 294 passed |
| `check:lint-budget` | 46 / ceiling 46 | **46 / ceiling 46** |
| `check:targets`, `check:ipc`, `check:egress`, `check:no-lora`, `check:whitespace` | pass | **pass** |

No test was removed, no type loosened, and the warning ceiling was not raised.

---

## 12. Correction to §4.3, and what it changed

While implementing Step 5 a further defect turned up that the audit had missed,
and it makes §4.3 an understatement rather than an error.

### The finding

`agent_start_run` minted its run id unconditionally:

```rust
let run_id = uuid::Uuid::new_v4().to_string();
```

and `StartRunRequest` had no field by which a caller could name a run — a
deliberate decision, documented on `correlation_id`: *"The caller does not get
to name the run."* The reasoning is sound; a caller that could name a run could
write events into somebody else's.

The consequence was not intended. Every lookup keyed on `run_id` earlier in that
function was therefore keyed on **an id that had just been invented**, so:

- `tasks::load(app_data_dir, &run_id, ...)` could never find a record, and
- after Step 1, `events.checkpoint(&run_id)` could never find a checkpoint.

`resumed_notes` was therefore **always `None` in the shipping product**, and the
whole carried-over-notes path — including `RunRequest.notes` on the TypeScript
side, documented as *"Sent when a run is resumed after the process went away"* —
was unreachable. So was every line of the comment explaining which source it
preferred and why.

§4.3 described this as reading the wrong source. It was worse than that: there
was no way to express "continue run X" at all, so the seeding never happened
from either source. **Step 1 was correct but inert until Step 5 landed.**

### What was done about it

`agent_start_run` is now a thin wrapper over an internal `drive_run`, which
takes `existing_run_id: Option<String>`. A fresh run passes `None` and mints its
own id exactly as before; `agent_resume_run` passes `Some(run_id)`.

The invariant that motivated the original design is kept intact. `drive_run` is
not a command and `StartRunRequest` still has no run-id field, so no caller can
name a fresh run. The only way to supply an id is `agent_resume_run`, which
first puts the run through `assess_resumability` — permission, ownership, and
the checkpoint's policy, plan and workspace hashes re-derived against the world
as it is now.

`agent_resume_run` also no longer returns after recording its intent. It reads
the prompt and classification back off the run's own snapshot — never from the
caller, since the plan is derived from the prompt and a different prompt is a
different plan than the one whose hash was just checked — and drives the run
through the same path a fresh run takes.

### Step 5 status

| | |
|---|---|
| Resumption executes | **Done.** `agent_resume_run` drives the run under its original id. |
| Shared execution path | **Done.** One `drive_run`; starting and continuing differ only in which id the work is recorded under. |
| Notes seeded from durable state | **Done, and now reachable.** Step 1's precedence is live. |
| Startup auto-recovery | **Not done.** `recover_interrupted` still marks interrupted runs `RunDegraded`; nothing calls `agent_resume_run` automatically. Resumption is operator-initiated. |
| Bounded by `max_recovery_attempts` | **Not done.** No recovery counter or ceiling yet. |
| UI recovery flow | **Not done.** No frontend caller; the manifest records this. |

So a stopped run can now genuinely be continued, and is not yet continued
*automatically*. The remaining half of Step 5 is startup policy — deciding which
interrupted runs to offer or resume, under a bounded attempt count — and that
needs Step 3's lease to be safe, because auto-resume plus a live run is exactly
the two-writer case §6 R4 describes.

### Verification after Step 5

| Suite | Result |
|---|---|
| `test:rust` | 1731 passed, 0 failed |
| `check:lint-budget` | 46 / ceiling 46 |
| `check:ipc` | 125 commands |
| `check:targets` | pass |

The `agent_resume_run` manifest note was rewritten, because the old one said the
command was "the half that exists" and that is no longer what it is.
