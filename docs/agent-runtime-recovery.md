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

The planned `recovery_e2e` integration suite must exercise the real task driver
with a deterministic provider and isolated temporary storage. Do not inject
crashes against an operator's real work or treat mocks as proof of native-model
performance. Record which suites actually ran and which gates remain unmet.

## Deployment and migration

No production database migration has been added in the initial implementation
stage. Later schema additions must use the existing transactional migration
runner. Before deploying them, make a consistent backup of SQLite and the
associated protected body/memory files, and test an upgrade from a v1 fixture.
Never copy an active WAL database without a SQLite-consistent backup strategy.

Rebuild the bundled Node runtime together with Tauri when protocol changes land.
Do not bypass the network broker, relax permissions, install new services, or
enable unattended execution to make a recovery test pass.
