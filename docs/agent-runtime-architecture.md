# Durable small-context runtime

Status: implementation in progress. The acceptance gates below are the definition
of completion; the existence of a module or a passing isolated test is not enough.
See [the current audit](agent-runtime-audit.md) for the starting defects.

## Research basis

Anthropic's [context engineering article](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
describes compact working context, just-in-time retrieval, compaction and notes
kept outside the context window. ARJUN will retain the exact underlying records
and build a bounded projection before each model call. A summary is an index
into durable evidence, not permission to discard that evidence.

Anthropic's [long-running agent harness work](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)
uses explicit requirements, incremental work, handoff artifacts and end-to-end
verification to address lost progress and premature completion. ARJUN applies
that principle to structured task criteria and checkpoints, rather than asking
the model to decide whether its own output is complete.

Anthropic's [Managed Agents architecture](https://www.anthropic.com/engineering/managed-agents)
separates durable sessions, replaceable model harnesses and tool execution.
ARJUN already has the Rust/Node authority split; we are preserving it. The Node
process may be replaced without replacing the logical task, and the model
context can be rebuilt without replaying side effects. This does not introduce
Anthropic's hosted service or any network dependency into the local product.

These are engineering references, not a guarantee that any model can finish any
task. Their general principles require independent validation on ARJUN's local
models, tools, privacy rules and failure boundaries.

## Component ownership

```text
Task Controller (Rust; authenticated operator)
    |
    +-- Durable Task State      SQLite; versioned state + attempt + lifetime budget
    +-- Append-only Event Log   SQLite; ordered metadata and protected-body references
    +-- Checkpoint Store        SQLite; structured safe-boundary state
    +-- Memory Store            task facts + approved user/project facts
    +-- Tool Intent Ledger      logical operation ids, results, reconciliation
    +-- Context Builder         bounded projection; exact pending state and recent pairs
    +-- Model Adapter           replaceable Node/OpenClaw + local serving endpoint
    +-- Tool Executor           Rust authorization and call-bound grants
    +-- Completion Verifier     criteria, artifacts, tests, pending work/effects
```

The diagram is the target architecture, not a claim that every path is wired.
SQLite remains the storage technology, npm the JS dependency manager, and the
current local serving and tool interfaces remain in place.

## Invariants

1. The model context is disposable. An acknowledged safe step is reconstructible
   from durable state even if Node or the desktop process disappears.
2. Only the current fenced attempt may advance a task. Lease acquisition is not
   enough: effects, state, checkpoint and terminal writes check the same fence.
3. No durable intent means no side effect. A dispatched action whose result
   cannot be recorded blocks continuation until reconciled.
4. A logical operation id survives retries and restart. It is distinct from a
   transport request id and from the hash of arguments; different intended
   operations may legitimately have identical arguments.
5. An approval binds the exact normalized call, owner/scope, and operation. Its
   decision is durable before the executor can observe it; expiration and
   consumed status are checked again at execution.
6. Every model request accounts for instructions, tools, state, history, memory
   and an output/error reserve. Pair-aware reduction must not create orphaned
   tool calls or results. If mandatory content cannot fit, the controller reports
   an explicit limitation instead of discarding the objective or sending an
   oversized request.
7. Compaction and checkpointing are separate. A checkpoint acknowledges
   execution; compaction creates a new prompt projection. Neither grants tools
   or satisfies completion criteria.
8. Only independent verification can produce task success. Missing evidence,
   unreadable stores, unknown effects, pending approvals, waits and unfinished
   required steps prevent success.
9. Authentication and classification apply to restored bodies as well as live
   inputs. Tool output is data, never a new system instruction. Shared memory
   promotion requires the existing approval/ACL boundary.
10. Recovery attempts, total iterations, retries and run duration are bounded
    across attempts, not restarted at each context reset.

## Durable records and protocol

Keep `run_id` as the logical task identity and `attempt_id` as the worker attempt.
Persist the conversation/message identity, effective classification, original
objective and acceptance criteria, input references, phase, current step,
completed/remaining work, pending waits/approvals/operations, next action, state
version, lifetime counters, deadline, fence and latest event/checkpoint references.

Persist provider-valid messages and unabridged tool data in an access-controlled
body store. The metadata event stream carries references, lengths and digests;
it must not become a less-protected copy of confidential input. Persist versioned
compaction summaries alongside their covered event range. Keep precise ids,
paths and business identifiers outside lossy model summaries.

Before changing the Rust/Node recovery protocol, add one canonical JSON-schema
artifact and common serialized fixtures. Rust and Node must verify the same
version, required fields, absent-versus-empty semantics and error cases. Never
interpret an empty response as an empty task or permission to proceed.

The intended next-action envelope is a discriminated union: tool call, plan
update, approval request, external wait, completion claim, or needs-review.
The existing native tool-call format can remain for tool actions; control
actions must be validated before affecting state. Completion is a claim submitted
to the verifier, not a terminal instruction.

## Context projection and compaction

Build each prompt from the objective and criteria, structured progress, immutable
constraints, exact pending state, relevant scoped memory, a compact summary and
recent complete tool pairs. Retrieve older bodies by stable reference only when
needed. Preserve subsequent user corrections as authoritative task input.

The target default input safety fraction is 0.70 of the effective provider
window, additionally constrained by the required response and recovery reserve.
Reduce redundant metadata and oversized tool bodies before summarizing older
history. The final admission check covers the whole serialized request, not
only its transcript. Estimates must be labeled as estimates and reconciled with
measured usage; no fabricated token values are allowed.

Compaction starts from a committed checkpoint, flushes task-local facts with
provenance, persists a started event, creates and stores the successor summary,
then persists completion. Resume can use the preceding checkpoint if the
summary operation fails. Model/provider failure invokes a bounded retry or a
deterministic projection from already durable state; it must not erase progress.

## Implementation acceptance ledger

Each row requires production wiring and tests, not merely definitions.

| Requirement | Current implementation state / required proof |
|---|---|
| Fail-closed intent/result persistence | In progress; storage fault injection and actual executor rejection |
| Versioned execution state and protected raw transcript | Pending; reopen and reconstruct an interrupted run |
| Single fenced writer with heartbeat | Partial existing lease; same-process and cross-process stale-worker tests pending |
| Structured checkpoint after safe boundaries | Existing store; live notes/result acknowledgement wiring pending |
| Hard bounded context and pair-aware projection | Existing compactor; full-request admission and durable reconstruction pending |
| Separate durable compaction | Pending; failure at each compaction boundary and raw-history retrieval |
| Scoped durable memory and conflict provenance | Partial; durable task scope and conflict history pending |
| Stable operation ids and reconciliation adapters | Partial; normalized calls, real reconciliation and per-tool retry wiring pending |
| Restart/model/network recovery with lifetime limits | Partial; authenticated dispatcher and failure simulations pending |
| Durable approval suspension and exact-call resume | Partial storage; atomic decisions and live resumption pending |
| Independent completion controls terminal status | Existing verifier; production terminal gating pending |
| Pause/resume/inspect UI and IPC | Partial; pause and frontend resume workflow pending |
| Model/compaction/wait/recovery observability | Partial event vocabulary; real boundary producers and metrics pending |
| Deterministic full recovery journey | Pending; multistep + large output + compaction + restart + approval + verified artifacts |
| Real local-model/native deployment verification | Pending; unit/mock tests are not a substitute |

No new dependency is planned. Append migrations to the existing `user_version`
runner and verify old databases remain readable. Do not edit already shipped
migrations or make unrelated framework changes.
