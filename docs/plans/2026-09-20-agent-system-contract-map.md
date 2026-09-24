# Agent system — tool, job and result contract map

Delivered by plan prompt **P01**. The machine-readable authority is
[`agent-runtime/src/tool-contract.json`](../../agent-runtime/src/tool-contract.json),
generated from [`src-tauri/src/orchestrator/contract.rs`](../../src-tauri/src/orchestrator/contract.rs)
and held byte-equal to it by `orchestrator::contract::tests::the_published_contract_is_current`.
This page is the human reading of that file at the P01 working tree; where the
two disagree, the JSON is right and this page is stale.

Regenerate the JSON after changing any tool:

```bash
ARJUN_WRITE_TOOL_CONTRACT=1 cargo test --manifest-path src-tauri/Cargo.toml --lib orchestrator::contract
```

---

## 1. The one production path a tool call takes

There is exactly one production dispatcher. `orchestrator::executor` and
`orchestrator::grammar` compile and are tested, and have no production caller.

| Step | Where | What is decided |
|---|---|---|
| 1. Offer | `agent_runtime::mod` `tool.catalogue` → `agent-runtime/src/tools.ts` `buildTools` | Plan permits it; mode permits its network use; its **catalogue-checked prerequisites** are met (today: `agent.delegate_readonly` needs a registered worker). Only names with a TS definition in `catalogue.ts` are bound. |
| 2. Schema | `catalogue.ts` (TypeBox, `additionalProperties: false`) | The model fills a closed schema. |
| 3. Authorise | runtime `beforeToolCall` → RPC `tool.authorize` → `agent_runtime::authorize` | Plan budget reserved; `ToolGateway::decide`. |
| 4. Gateway | `orchestrator::gateway::ToolGateway::decide` | (a) name or alias resolves; (b) arguments are an object, **every name declared**, every required one present, every kind right; (c) path inside the run's workspace (the gateway-checked prerequisite); (d) input size; (e) `PolicyGateway`: mode, permission, approval. |
| 5. Execute | runtime `hostTool.execute` (waits at most the catalogue's `timeoutSeconds`, aborts on Stop) → RPC `tool.execute` → `agent_runtime::execute` | Grant redeemed (single use, bound to the call); plan re-checked; gateway **asked again**. |
| 6. Intent | `events::idempotency` via `begin_effect` | For side-effecting tools only: an intent is on disk before the handler runs; a settled key replays, an in-flight or unknown one refuses. |
| 7. Route | `agent_runtime::execute` `match` | An agent-path arm, or the runner fallback via `runner_for` → `LocalToolRunner::run`. A tool the contract routes to the agent path that reaches the fallback now fails with its registered handler named. |
| 8. Output | `tools::truncate_response`, `tools::sanitise_failure` | Deterministic cut with a notice at `maxResponseBytes`; failure text stripped of paths and stack frames — on fresh and replayed results alike. |
| 9. Settle | `settle_effect`, `record_call`, `record_step` | The effect is settled, the call recorded, the step counted. |

IPC: no tool has its own Tauri command, and P01 added none. The model reaches
tools only through the runtime's stdio RPC (`tool.catalogue`, `tool.authorize`,
`tool.execute`). `check:ipc` passes at 172 commands, unchanged.

### The model round around those steps (P03)

Every model call, including the compactor's own summary call, is one *round*,
bracketed by three more stdio RPCs (`agent_runtime::rounds`, runtime
`context-refresh.ts` `RoundBoundary`):

| Step | Where | What is decided |
|---|---|---|
| R1. Open | runtime `transformContext` (before compaction) → RPC `context.refresh` | The run's lease is taken from the one `ModelScheduler` (capacity one, whole process); its endpoint is re-bound (a restarted server is reported `cache: cold`); the four memory scopes are compiled at one cursor against the **served** window; the manifest is recorded (`context_compiled`). Mandatory state that cannot fit, a context that cannot be compiled, a load failure, or an unobtainable lease → `round_refused`, the card is given back, and the model is not called. |
| R2. Count | provider `onPayload` → RPC `context.count` | The exact outgoing body is rendered by the served model's template and counted by its tokenizer; images are charged at the cost the server reported, or recorded unmeasured; the fit is judged against window − reply − safety; the projection is checked; `model_requested` is recorded. Overflow → `round_refused`. |
| R3. Settle | the stream's `result()` → RPC `context.settle` | Lease released; the count reconciled against the server's reported usage; `model_responded` recorded. |

`tool.authorize` (step 3) also releases the round: a tool call means the model
finished generating, so no tool — including a delegation that needs the card
for a child — ever waits on a lease its own run holds.

---

## 2. Every tool

Route **A** = an arm of `agent_runtime::execute`; **R** = `LocalToolRunner::run`
through the fallback. Prerequisites are *checked at* **cat**alogue, **gw**
(gateway) or **h**andler. Cancellation: **wait** = the runtime stops waiting and
says the outcome is unknown, the handler is not interrupted; **wait+intent** =
the same, with the pending intent promoted to `unknown` rather than retried;
**deadline** = also stopped in Rust.

| Tool | Accepted spellings | Required | Optional | Mode / effect | Approval | Route → handler | Prerequisites | Output | Cancel | Timeout s |
|---|---|---|---|---|---|---|---|---|---|---|
| `agent.delegate_readonly` | — | profile, task | documents[], files[], expressions[], artifacts[], deliverable, after_revision | read / reversible | automatic | R → `delegate_to_subagent` → `SubagentManager::spawn` | worker@cat, model registry@h | child result | deadline | 120 |
| `artifact.create_approval_note` | create_docx | path | template, content, sections, title, classification | write / side-effecting | automatic | A → `artifacts::create_docx_with_evidence` | workspace@gw | artifact | wait+intent | 120 |
| `artifact.create_briefing_deck` | create_pptx | path, content | — | write / side-effecting | automatic | A → `artifacts::create_pptx` | workspace@gw | artifact | wait+intent | 120 |
| `artifact.create_calculation_workbook` | create_xlsx | path | sheets, title, classification | write / side-effecting | automatic | A → `artifacts::create_xlsx` | workspace@gw, calculations@h | artifact | wait+intent | 120 |
| `artifact.create_chart` | create_chart | title, kind, categories, series, valueLabel | — | write / side-effecting | automatic | A → `create_chart` | — | artifact | wait+intent | 120 |
| `artifact.create_diagram` | create_diagram, create_flowchart, artifact.create_flowchart | title, direction, blocks, connections | — | write / side-effecting | automatic | A → `create_diagram` | — | artifact | wait+intent | 120 |
| `artifact.create_pdf` | create_pdf | title, classification, body | — | write / side-effecting | automatic | A → `create_pdf` | — | artifact | wait+intent | 120 |
| `artifact.create_table` | create_table | title, header, rows, classification | — | write / side-effecting | automatic | A → `create_table` | — | artifact | wait+intent | 120 |
| `artifact.list` | — | — | kind | read / read-only | automatic | A → `artifact_list` | — | text | wait | 30 |
| `artifact.read` | — | artifact | version | read / read-only | automatic | A → `artifact_read` | — | text | wait | 30 |
| `artifact.verify_docx` | validate_artifact | path | — | read / read-only | automatic | A → `validate` | workspace@gw | text | wait | 60 |
| `calculation.evaluate_with_units` | run_calculation | expression | — | read / reversible | automatic | R → `calculate`, then the run's calculation table | — | calculation | wait | 5 |
| `capability.search` | — | query | — | read / read-only | automatic | A → `capability_search` | — | text | wait | 5 |
| `document.read_pages` | — | documentSha256, fromPage | toPage | read / read-only | automatic | A → `read_attached_pages` | — | text | wait | 10 |
| `document.search` | — | query | — | read / read-only | automatic | A → `search_attached_documents` | — | text | wait | 10 |
| `knowledge.build_graph` | — | — | notebook, documentSha256, focus | read / read-only | automatic | A → `build_document_graph` | — | text | wait | 15 |
| `knowledge.load_evidence_region` | load_more_evidence | documentSha256, fromPage | toPage | read / read-only | automatic | A → `region_hits` + `retrieval::record_region` | — | evidence | wait | 30 |
| `knowledge.multimodal_retrieve` | — | query | documentType, documentSha256, maxResults | read / read-only | automatic | R → `multimodal_retrieve` | multimodal index@h | evidence | wait | 45 |
| `knowledge.search_authorized` | search_documents | query | detail, maxResults | read / read-only | automatic | A → `search_hits` + `retrieval::record` | — | evidence | wait | 30 |
| `media.extract_findings` | — | documentSha256, fromPage | toPage | read / read-only | automatic | R → `extract_findings` | — | evidence | wait | 90 |
| `memory.promote_approved` | memory_promote_approved | key, approvalId | — | write / reversible | pre-approved value | A → `memory_api::promote_approved` | — | text | wait | 10 |
| `memory.recall_authorized` | memory_recall_authorized | scope | — | read / read-only | automatic | A → `memory_api::recall_authorized` | — | text | wait | 10 |
| `notebook.add_source` | — | document | notebook | write / reversible | automatic | A → `notebook_add_source` | — | text | wait | 10 |
| `notebook.create` | — | name | — | write / reversible | automatic | A → `notebook_create` | — | text | wait | 10 |
| `notebook.delete` | — | — | notebook | write / irreversible | person | A → `notebook_delete` | — | text | wait+intent | 10 |
| `notebook.list` | — | — | — | read / read-only | automatic | A → `notebook_list` | — | text | wait | 10 |
| `notebook.list_sources` | — | — | notebook | read / read-only | automatic | A → `notebook_sources` | — | text | wait | 10 |
| `notebook.remove_source` | — | document | notebook | write / reversible | automatic | A → `notebook_remove_source` | — | text | wait | 10 |
| `notebook.rename` | — | name | notebook | write / reversible | automatic | A → `notebook_rename` | — | text | wait | 10 |
| `sandbox.run_code` | execute_code | language, source | — | write / irreversible | person | R → `execute_code` | container sandbox@h | execution | deadline | 60 |
| `sovereignty.get_evidence` | — | — | — | read / read-only | automatic | R → `sovereignty_evidence` | — | text | wait | 5 |
| `workspace.read_text` | read_scoped_file | path | fromLine, maxLines | read / read-only | automatic | R → `read` | workspace@gw | text | wait | 15 |
| `workspace.write_text` | write_scoped_file | path, content | — | write / side-effecting | person | R → `write` | workspace@gw | artifact | wait+intent | 30 |

Every tool has `network: none` except `media.extract_findings` (`loopback`);
none is `outbound`. Permissions and response ceilings are in the JSON.

---

## 3. What P01 found on this path, and fixed

The gateway and the model's schema had drifted. Each of these was a call the
runtime's own schema allowed and the gateway refused — on the one production
path — while the unit tests passed because they drove `LocalToolRunner`
directly and never met the gateway.

| Tool | Defect | Fix |
|---|---|---|
| `agent.delegate_readonly` | Six arguments *required* that the schema never offered; four typed `Object` while the runner reads arrays. **Every delegation was refused.** | `profile`, `task` required; the rest optional; a new `List` kind; the schema offers them. |
| `knowledge.multimodal_retrieve` | `page` required — nothing reads it and the schema has no such field. **Every call was refused.** | Three filters optional; `page` dropped. |
| `knowledge.load_evidence_region`, `media.extract_findings`, `document.read_pages` | `toPage` required; schema optional; handlers default it. | Optional. |
| 6 notebook/graph tools | `notebook` required for the grammar's sake; schema optional; `resolve_notebook` takes an `Option`. | Optional; the grammar now expresses optional members. |
| `artifact.list`, `artifact.read` | In Rust's catalogue and served, **never in the TS catalogue** — `buildTools` skipped them, so no model was offered the cross-turn artifact channel. | Defined in `catalogue.ts`, named in `tool-names.ts`. |
| `knowledge.search_authorized` | `page` offered as "fetch the next batch"; no path reads it — page 2 came back as page 1. | Removed from the schema and the description. |
| `notebook.delete` | Side-effecting in Rust, not in the TS `SIDE_EFFECTING` set. | Added. |
| any tool | An undeclared argument was silently ignored at the gateway. | Refused, naming the accepted arguments. |

Delegation-specific, found tracing `delegate_to_subagent`:

| Defect | Fix |
|---|---|
| "Every role but the retriever needs inputs" compared the literal `"knowledge-retriever"`, so a **clone** of the retriever was refused. | Keyed on the resolved capability. |
| An artifact reference without a revision was given **revision 1** — a plausible number nobody supplied. | Refused; the revision may be `revision` or `id@N` (as `artifact.list` shows it). |

---

## 4. The job contract (`subagents::packet::ChildTaskPacket`)

| Contract item | Field | Source |
|---|---|---|
| agent | `agent_id` | registry id; never a display name |
| immutable definition version | `definition_version`, `definition_origin`, `instructions_sha256` | resolved once per dispatch (`subagents::definitions`) |
| role capability | `capability` | `capability_for(output_schema)` |
| task / run / attempt / job | `task_id`, `parent_run_id`, `attempt_id`, `job_id()` (= `idempotency_key`); `child_id` is one execution | attempt from the parent's checkpoint seed |
| model policy | `model_policy`, `model_id` | definition binding + `certification::choose` decision; `within_eligible` computed; the eligible set is **enforced at lease admission** (`SchedulingRefusal::OutsideEligible`, P03) |
| skill hashes | `skills` | definition's `SkillBinding`s (name, version, sha-256) |
| effective tools | `allowed_tools` | `InheritedPolicy::narrow_for` ∩ parent, then `enforce_mode` |
| sharing policy | `shared_with_task`, `classification_ceiling` | definition + narrowing |
| input references | `inputs` | references only — `InputRef` has no variant carrying text |
| expected output schema | `required_schema`, `deliverable` | definition + dispatch |
| dependency revisions | `requirement`, revisions inside `inputs` | `after_revision`, artifact/graph-item revisions |
| cancellation | `deadline` | manager timeout + worker `Stopping` |
| resource limits | `limits` | narrowed |

Every field added is `#[serde(default)]`: a packet recorded before it existed
reads (`a_packet_recorded_before_definitions_were_pinned_still_reads`).

**Dispatch keys.** A dispatch names an `ag-` id or a bundled role key (matched
only against the row imported from that role's file). A display name is never
a key. The worker is found by capability. The Agents screen's `workerAvailable`
/ `ready`, the test-run command and `agent_dependents` now use the same rules
(`commands::agents::{worker_key, has_worker_for, trace_keys}`).

## 5. The result contract (`subagents::result::ChildResult`)

Status: `completed`, `failed`, `timed_out`, `cancelled`, `refused`, **`partial`**,
**`blocked`** (the last two are not complete and name what is missing).
Fields: `findings` (each with evidence refs), `artifacts` (id, version,
sha-256), `receipts` (run, tool, event sequence — **a receipt with no event is
refused** by `with_receipt`), `validation` (check id, passed/failed/blocked),
`missing`, `published` (memory item ids), `confidence`, `uncertainty`,
`result_hash`. New lists enter the hash only when non-empty, so every hash
recorded before them still verifies.

Schemas: `extraction`, `retrieval`, `calculation`, `review`, `code` (each with a
worker) and **`document`, `deck`, `workbook` — registered, with no worker**:
dispatch refuses them as "registered and cannot yet be run", the Agents screen
labels them "(no worker yet)" and never reports such an agent ready.

## 6. Writer delegation

`subagents::manager::DelegationMode::{ReadOnly, Writer}`. `agent.delegate_readonly`
always sends `ReadOnly`: a role declared as writing is refused, and any write
tool an edited read-only role holds is withheld and recorded as refused.
`Writer` never widens (narrowing runs first), leaves every effect to the
gateway's approval, runs in the exclusive lane whenever the child holds any
write tool, and is reachable only through `Dispatch::writing()` — which no
model-facing tool calls in this build (P05/P11).

## 7. Known deviations

Opened by P01; a row a later phase closed says which phase and how.

| Where | What | Owner |
|---|---|---|
| `knowledge.search_authorized` on the agent path | `detail: "citations"` is honoured by `LocalToolRunner::search` but not by `retrieval::record`, which the agent path uses — full passages come back. | P07 |
| `artifact.create_pdf`, `artifact.create_table` (and the optional `classification` on the OOXML writers) | The classification label is taken from the model's arguments; §11.1 requires it be derived from the run. | P04 |
| `media.extract_findings` | Declares `loopback` and 90 s, but reads text the ingest pipeline already extracted; no OCR runs at call time. | P06 |
| every tool but delegation and the sandbox | `ToolSpec::timeout` bounds the runtime's *wait*; the Rust handler is not interrupted. **Still open after P03**, deliberately: interrupting a side-effecting handler mid-effect leaves an effect nobody can account for (why `authorize` lets a running tool finish). P03 made sure no handler can hold the model lease while it runs (the round is released at `tool.authorize`), which was the part that could deadlock. A read-only-only deadline remains to do. | P04/P05 |
| `model_policy` | ~~The definition's eligible set is recorded (`within_eligible`) and not enforced by routing.~~ **Closed in P03:** the worker's lease request carries the eligible ids and the scheduler refuses any other model (`a_model_outside_the_definitions_eligible_set_is_refused`). | — |
| in-process generations of the external-tool gateway (`ai_engine::scheduler`) | Not taken through the P03 lease. Off by default and serves external clients, not agent work; loading a server model still unloads the in-process one through `admission::admit`. | P14 |
| `skills` on the packet | Pinned and traced; the child loop does not load skills. | P05 |
| `shared_with_task` | Carried on the packet; not enforced at publication (plan §3 finding 8). | P02 |
| receipts | `ReceiptRef` refuses sequence 0; workers still publish no receipt (§3 finding 2). | P02 |
| `SubagentManager::recall` | A replayed result is rebuilt with schema `retrieval` whatever the original was. | P02/P05 |
| idempotency key | Derived from the dispatch key as given, so the same agent named by role key and by `ag-` id yields two keys. | P05 |
| `artifact.create_flowchart` | Resolved by Rust, unknown to the runtime's alias table. | — |
| `src/services/toolNames.ts` | The UI label table lacks 12 tools; they display their wire name. | P14 |
