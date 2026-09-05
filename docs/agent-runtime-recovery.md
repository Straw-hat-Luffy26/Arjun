# Agent runtime recovery

Implementation is in progress. This is the recovery contract and verification
guide, not a claim that unattended recovery is ready to deploy.

## Safe recovery sequence

1. Discover nonterminal runs; leave a run alone while another valid lease owns it.
2. Reauthenticate the original operator and check current policy/classification.
3. Acquire a unique attempt lease and fence before writing resume events.
4. Load structured task state, original inputs and the latest valid checkpoint.
5. Inspect every pending or unknown tool intent against the authoritative ledger.
6. Reconcile with observable evidence. Never infer success from a file name alone
   or retry opaque side effects merely because their result is missing.
7. Restore plan progress, lifetime budget, conversation identity and approvals.
8. Build an admitted bounded context; continue the next incomplete logical step.
9. Commit results and checkpoints; renew the lease while work is active.
10. Complete only after independent verification. Preserve limitations and partial
    failures in the final record.

The Node process is replaceable; losing it must not erase task history. Startup
discovery does not authorize a system user to act on another person's behalf.
An unsupported or ambiguous effect stops in `NEEDS_REVIEW` (the existing
`DegradedNeedsHuman` state), rather than being silently repeated.

## Crash boundaries to verify

| Interruption point | Required recovery behavior |
|---|---|
| Before durable intent | The tool has not been authorized to run |
| Intent committed, dispatch unconfirmed | Reconcile before retry; distinguish undispatched intent where provable |
| Dispatch returned, result not durable | Outcome remains ambiguous; no blind replay |
| Result durable, checkpoint not committed | Fold the result into state without executing the operation again |
| Checkpoint committed | Resume the next logical action |
| Compaction started, no summary committed | Restore prior checkpoint and rebuild/retry within limits |
| Summary committed | Use the successor projection; old raw history stays retrievable |
| Approval pending | Restore the same request and remain suspended |
| Approval decided, not consumed | Check expiry and exact arguments, then consume once |
| Lease expired/replaced | Old worker may not advance or perform a new effect |
| Completion verification not committed | Task is not successful yet |

## Current control interfaces

These are Tauri IPC operations, not shell commands:

- `agent_start_run`: start a task with the existing `StartRunRequest`.
- `agent_abort_run`: cancel a run; cancellation is not a resumable pause.
- `agent_task_snapshot` / `agent_task_events`: inspect durable progress.
- `agent_run_resumability`: assess an interrupted run.
- `agent_resume_run`: existing manual resume path, being hardened.
- `agent_unknown_effects` / `agent_reconcile_effect`: reviewer inspection and
  recording of an ambiguous action's outcome.

A durable pause API and Tasks-screen recovery controls are still pending.
Verified start/pause/resume examples will be added when those paths are wired.

## Validation commands

From the repository root:

```powershell
cargo test --offline --manifest-path src-tauri/Cargo.toml --lib durability_tests
cargo test --offline --manifest-path src-tauri/Cargo.toml --lib agent_runtime::
npm run runtime:typecheck
npm run runtime:test
npm run test:ui
npm run check:ipc
git diff --check
```

### Deterministic recovery acceptance

`tests/agent_runtime_durable_journey.rs`, included by the `agent_runtime` test
target, exercises the production `agent_runtime::task_driver::TaskDriver` used
by the desktop start/resume command. Build the actual Node bundle first:

```powershell
npm run runtime:build
cargo test --offline --manifest-path src-tauri/Cargo.toml --test agent_runtime durable_journey --no-default-features -- --nocapture
```

On Windows, use the configured MSVC developer shell, with CMake and libclang
available, and add `--target x86_64-pc-windows-msvc` if Cargo defaults to GNU.

The journey uses an isolated SQLite database/workspace and a loopback scripted
provider. It searches a real indexed passage, forces compaction in a 4k window,
kills the Node worker twice (once during inference and once awaiting approval),
restores the same approval, writes the file once, and reads it back. The shared
driver settles the production planner's steps from successful tool receipts,
resolves the answer's citation against restored evidence, reopens the artifact,
records fenced completion verification, and publishes the task record and
terminal event through the same publication function as the desktop command.
Raw history, the single write receipt, approval identity, step budget, saved
verification and event ordering are asserted.

Two rejection cases prevent a permissive gate from passing this acceptance
test: an invented citation after the same recovery journey, and a model that
reports completion without performing the required search. Both must remain
unsuccessful in the saved task and durable terminal state.

The live checkpoint is a recovery input. Its execution step flags are not a
final completion verdict; the final task plan is settled independently from
receipts and verification. No test marks checkpoint steps done or supplies a
hardcoded successful grounding result.

This covers the production execution/completion driver, not Tauri IPC, model
routing, authenticated resume dispatch, recovery UI, or native inference. The
three deterministic tests do not measure real-model retention or answer quality.

## Deployment and migration

No production database migration has been added in the initial implementation
stage. Later schema additions must use the existing transactional migration
runner. Before deploying them, make a consistent backup of SQLite and the
associated protected body/memory files, and test an upgrade from a v1 fixture.
Never copy an active WAL database without a SQLite-consistent backup strategy.

Rebuild the bundled Node runtime together with Tauri when protocol changes land.
Do not bypass the network broker, relax permissions, install new services, or
enable unattended execution to make a recovery test pass.
