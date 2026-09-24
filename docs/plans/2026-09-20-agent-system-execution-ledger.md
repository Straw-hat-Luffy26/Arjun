# Agent system — execution ledger

The running record for [`2026-09-20-agent-system-build-plan.md`](2026-09-20-agent-system-build-plan.md),
prompts P00–P16. One ledger, updated at the end of each phase.

| Fact | Value |
|---|---|
| Plan baseline SHA | `4164fc26bbdc86eeed61fcf6513f0b97f2cc7e0e` (the plan's own header) |
| HEAD when P00 started | `b9ff7fbcbab3c610dffd541b674ec04706710561`, working tree clean |
| Branch | `main` |
| Host | Windows 11 Home Single Language 10.0.26200 |
| CPU / RAM | AMD Ryzen 7 250 (8C/16T) · 25 025 695 744 B (≈23.3 GiB) |
| GPU | NVIDIA GeForce RTX 5060 **Laptop** GPU, 8151 MiB, driver 595.95, CUDA 13.2, cc 12.0 |
| node / npm | v24.12.0 / 11.6.2 |
| cargo / rustc | 1.93.1 (083ac5135) / 1.93.1 (01f6ddf75) |
| Python | 3.11.9 |
| llama-server | 0.4.1-dev, build **10970**, commit `bfdc32183`, Clang 20.1.8 |
| App data identity | `com.arjun.workbench` (`src-tauri/tauri.conf.json`) |

> **This machine is not established to be the target machine.** The plan targets
> "RTX 5060, 8 GB VRAM". This host reports an RTX 5060 **Laptop** GPU — a
> different SKU with a different power envelope and memory bandwidth. Every
> figure below that was measured here says so; none of them is presented as a
> target-machine measurement. Only the hardware owner can close that gap.

---

## Phase status

| Phase | Scope | Status |
|---|---|---|
| P00 | Baseline, inventory, executable acceptance fixtures | **Complete** — see below |
| P01 | Agent definitions, jobs, tool/result contracts | **Complete** (portable code and deterministic tests; no native gate) — see below |
| P02 | Shared memory correctness and authority | In progress — not closed; P03 fixed finding 1 on the compile path only (see P03 below) |
| P03 | Context assembly, model scheduling and durable handoff | **Complete** for portable code and deterministic contract tests; **native gate blocked** (no GPU, server or models on the machine it was built on) — see below |
| P04–P16 | — | Not started |

---

# P00 — Reproducible baseline and acceptance harness

## Implemented

| File | What it is |
|---|---|
| `scripts/qualification-inventory.mjs` | The machine-readable qualification inventory. Every value is `{value, source}` with `source ∈ measured \| declared \| unknown`; an unknown carries `unknown_because`. Parses GGUF headers for architecture, trained context and chat-template hash. `--hash` computes model sha-256s. |
| `scripts/fixture-manifest.mjs` | Builds and verifies `fixtures/agent-system/v1/manifest.json`. `--check` fails if a fixture changed without its hash being updated. |
| `scripts/agent-baseline.mjs` | The harness. Five-state report: executed / passed / failed / blocked / skipped. |
| `src-tauri/tests/agent_baseline.rs` | Four cases driven through the **production task driver** — `AgentRuntime::spawn` starting the real Node bundle over the real stdio protocol against real stores. |
| `fixtures/agent-system/v1/` | The fixture pack: 19 files, plus 3 reused in place. |
| `package.json` | Six entry points: `inventory`, `inventory:hash`, `fixtures:manifest`, `fixtures:check`, `baseline:agents`, `test:baseline:agents`. |
| `docs/plans/2026-09-20-agent-system-build-plan.md` | The plan itself, checked in so every `docs/plans/…` reference in P00–P16 resolves. |

Nothing existing was replaced. No registry row, model file or agent profile was
written. No model was downloaded.

## Contract decisions taken in this phase

1. **`{value, source}` everywhere in the inventory.** A bare `null` cannot be
   told apart from a field that was never asked about. The three sources are
   `measured` (a command ran here), `declared` (a file on this machine asserts
   it, unchecked) and `unknown` (with `unknown_because` naming what would
   establish it).
2. **Hashes are generated, never typed.** `fixture-manifest.mjs --check` is the
   mechanism; the same reasoning the repo already applies to the SBOM through
   `check:generated`.
3. **The grading key lives in the checkout, not the pack's source tree.** A run's
   workspace is `<app data>/runs/<run_id>` (`agent_runtime/workspace.rs:53`), so
   `fixtures/agent-system/v1/expected/` is unreachable from any run. The harness
   asserts this before grading rather than trusting the layout.
4. **Plain substrings, not regular expressions, in the fixture check files.**
   They are read by Node, Rust and Python; one escaping convention per language
   is three chances to get it wrong.
5. **A blocked case is never a pass, and a group that selected nothing is
   reported as failed.** Both are enforced in `agent-baseline.mjs`, and the
   hidden tests exit non-zero on zero selected cases.

## Revalidation of the nine §3 findings at HEAD `b9ff7fb`

Each was re-read in current source. Line numbers below are **current**, and
differ from the plan's where the file has moved.

| # | Finding | Classification | Current location | Note |
|---|---|---|---|---|
| 1 | Graph cursor accuracy | **Present** | `agent_runtime/context_compiler.rs:304` calls `self.graph.snapshot(…)`; `:471` records `graph_revision: scope.graph_revision`. `knowledge/graph/runtime_store.rs:946` is `snapshot_at`. | Unchanged. `snapshot_at` has exactly one production caller, `commands/memory_graph.rs:73`, and context compilation is not it. The manifest labels rows with a revision the read did not come from. |
| 2 | Real receipt provenance | **Present** | `subagents/worker.rs:391` publishes `event_seq: 0`; `knowledge/graph/runtime_memory.rs:507` admits only when `event_seq > 0`. | Unchanged (the plan cited `:388`/`:504`). The system is at least consistent: every worker finding is kept as a **proposal**, not admitted. The gap is that nothing carries the real event identity. |
| 3 | Semantic retrieval wiring | **Present, and narrowed** | `knowledge/embedding.rs:64` declares `LocalEmbedder`; re-exported at `knowledge/mod.rs:35`; **no production caller**. `context_compiler.rs:552` reports `mode: "lexical"`. | The code is honest — it refuses to call itself semantic. **New fact:** two embedding models *are* installed and registered on this machine (`Qwen3-Embedding-0.6B-Q8_0`, `nomic-embed-text-v2-moe.Q8_0`). So this is a code gap, not a weights gap. |
| 4 | Memory authority migration | **Present, and documented as such** | `knowledge/graph/migration.rs`, module header lines 44–48. | Its own words: *"What this is **not** is a cutover. Every legacy store is still written to directly by its own code."* Five legacy sources are covered and `uncovered` is computed on every run. The remaining work is routing writers, not writing the copier. |
| 5 | Executable new roles | **Present** | `subagents/worker.rs:143` lists five performable profiles; `subagents/profile.rs:162` `SchemaKind` has five variants (extraction/retrieval/calculation/review/code). | Unchanged. `agents/` holds five role files; orchestrator, document-author, presentation-author and spreadsheet-author have no file and no schema. |
| 6 | Visual identity | **Present** | `agents/mod.rs:82` — `AGENT_PALETTE: [&str; 8]`. | Unchanged. Nine roles, eight colours. `agents/store.rs:734` assigns `AGENT_PALETTE[held.len() % 8]`, so the ninth agent silently reuses the first's colour. |
| 7 | Model/projector discovery | **Present, with a concrete instance** | `registry/discovery.rs:151` and `registry/scan.rs:124` both require a filename starting `mmproj-`. | **New fact:** the installed Unlimited-OCR directory holds *two* matching files, `mmproj-Unlimited-OCR-F16.gguf` and `mmproj-ref-q8_0.gguf`. `discovery.rs` sorts by filename length and takes the shortest, which is `mmproj-ref-q8_0.gguf` — the wrong one. The OCR registry rows dodge this by pinning `projector` explicitly; auto-discovery of a new package does not. |
| 8 | Memory sharing policy wiring | **Present** | `agents/mod.rs:223` stores `shared_with_task`; `agents/mod.rs:502` (`to_profile`) carries `memory_scope` and **not** the flag; `subagents/profile.rs` `AgentProfile` has no field for it. `src/pages/Agents.tsx:671` renders the control. | Unchanged. The setting is visible, editable and reaches nothing. `agents/store.rs:750` hardcodes `shared_with_task: false` on import. |
| 9 | Registry versions reaching children | **Present** | `subagents/packet.rs:121` has `agent_id` and no `definition_version`. `subagents/manager.rs:209` snapshots profiles at construction; `subagents/worker.rs:169` holds instructions at construction. | Unchanged. A registry edit cannot reach the next child, because neither the manager's map nor the worker's string is re-read. |
| §13 | `store.rs` loses the role body | **Present** | `agents/store.rs:705` and `:741` both assign `agent.instructions = profile.description`. | Confirmed. `AgentProfile` carries a separate `instructions` field which both import paths ignore, so an imported agent is instructed by its one-line description. |

**Nine of nine present. None already fixed. None requires further evidence to
classify** — each was confirmed by reading current source, not by inference.

Two findings were *narrowed* by measurement rather than reclassified: #3 (the
weights exist, the wiring does not) and #7 (a real wrong-projector case exists
in the installed OCR directory).

## Qualification inventory

`evidence/agent-system/P00/qualification-inventory.json` — 15 registered models,
15 present on this machine, **15 with a sha-256** (3 declared by the registry,
12 computed here).

Architecture and trained context come from each file's own GGUF header, so they
are `measured`:

| Model | Arch (header) | Trained ctx | Quantization | sha-256 |
|---|---|---|---|---|
| Spark-X2.5-4B-Q8_0 | `spark2_5` | 1 048 576 | Q8_0 | measured |
| Qwen3.5-9B-Q4_K_S | `qwen35` | 262 144 | Q4_K_S | measured |
| gemma-4-E4B-it-qat-UD-Q4_K_XL | `gemma4` | 131 072 | registry says **`unknown`** | measured |
| gemma-4-12b-it-UD-Q4_K_XL | `gemma4` | 262 144 | registry says **`unknown`** | measured |
| NVIDIA-Nemotron3-Nano-4B-Q4_K_M | `nemotron_h` | 1 048 576 | **Q4_K_M** | measured |
| unlimited-ocr-q6-k / q4-k-m | `deepseek2-ocr` | 32 768 | Q6_K / Q4_K_M | declared by registry |
| Qwen3-Embedding-0.6B-Q8_0 | `qwen3` | 32 768 | Q8_0 | measured |
| nomic-embed-text-v2-moe.Q8_0 | `nomic-bert-moe` | 512 | Q8_0 | measured |
| Qwen3.6-35B-A3B-UD-Q4_K_M | `qwen35moe` | 262 144 | Q4_K_M | measured |
| gemma-3-12b-it, Qwen2.5-VL-3B, 2× mtp-gemma-4-12b, rebel-large | — | — | — | measured |

Questions the inventory settles that the plan left open:

- **The Nemotron variant question (§4) is answered.** It is
  `NVIDIA-Nemotron3-Nano-4B`, architecture `nemotron_h` — not the
  Llama-3.1-Nemotron variant. It is installed at **Q4_K_M**, not the Q8 the
  plan's "8-bit around 4B" policy calls for. Changing that is a decision, not a
  fix, and nothing here changed it.
- **Unlimited-OCR's architecture is `deepseek2-ocr`**, both tiers, sharing one
  chat template (`sha256 177c96be…`). The registry pins the **patched**
  projector, `mmproj-Unlimited-OCR-F16-patched.gguf`.
- **Two Gemma rows record `quantization: "unknown"`** for files whose names end
  `UD-Q4_K_XL`. The inventory records the registry's word and the filename's
  separately rather than reconciling them silently.
- **No registry row carries `minLlamaBuild`.** `registry/mod.rs:205` documents
  that Spark's `spark2_5` needs b10828 and the field is absent everywhere, so
  the launch gate cannot fire. The installed build is 10970, which satisfies it
  — by luck, not by check.
- **Container daemon: not reachable.** Docker CLI is installed at
  `C:\Program Files\Docker\Docker\resources\bin\docker.exe`; the daemon is not
  running, so its image list is `unknown`, not empty.
- **Renderers: none.** No `soffice`, no `libreoffice`. Python has PIL 12.3.0,
  PyMuPDF 1.28.0, python-docx 1.2.0, python-pptx 1.0.2, reportlab 5.0.0 —
  **openpyxl is absent**, which is what an independent workbook reopen needs.
- **A second, legacy app-data identity exists.** `com.sarathi.app` holds a
  partial mirror of the model tree and a stale `registry.json`. Several Rust
  tests still hard-code paths into it (`ai_engine/runtime.rs:1387`, `:1513`,
  `:1554`, `:1599`, `:1646`). Recorded, not touched.

## Measured on this machine (not on the target machine)

`evidence/agent-system/P00/served-context-observations.json`.

Real `llama-server` b10970 serving Spark-X2.5-4B Q8_0, driven by the production
task driver:

| Requested ctx | Observed served ctx | VRAM used at rest | Free | Load | Run outcome |
|---|---|---|---|---|---|
| 4096 | 4096 | 4813 MiB / 8151 | 3087 MiB | ~6 s | **failed** — 8852-token request exceeds 4096 |
| 16384 | 16384 | 5266 MiB / 8151 | 2634 MiB | ~4 s | **passed** — 12 characters generated |

Three findings came out of it:

- **P00-OBS-1 — the tool schemas cost 8584 tokens before anything else.** From
  the driver's own `context_ledger` event: `sections.toolSchema = 8584`. A
  one-word question costs 8852 input tokens. **A 4096-token served context
  cannot complete a single turn of this runtime, whatever the model.** Whether
  role-scoped tool loading (plan §4) reduces this is not measured.
- **P00-OBS-2 — the ledger reports `window: 0` and `headroom: 0`.** The runtime
  is not told the served window, so its budget accounting bounds nothing. Not in
  the plan's §3 list; it belongs to P03.
- **P00-OBS-3 — Spark Q8 at 16 k leaves 2634 MiB free** with the desktop
  resident. Consistent with the plan's one-heavy-slot policy. No second model,
  projector or OCR service was measured alongside it.

## Fixture pack

`fixtures/agent-system/v1/`, version 1.0.0, 19 files + 3 reused in place, every
one hashed in `manifest.json`. Nonconfidential throughout: every tag, reading,
name and certificate number is invented (PS-K).

| Input | What it is |
|---|---|
| 3 scanned pages | Rendered by `sources/scan/make-scans.py` — rotation, noise, bleed-through, blur. Deterministic (seed 26117): re-running gives byte-identical files. Page 3 carries a signature degraded past legibility **on purpose**. |
| SOP Revision C | Reused at `src-tauri/tests/fixtures/maintenance-sop.md`. Minimum allowable 9.0 mm. |
| SOP Revision D | `sources/sop/maintenance-sop-rev-d.md`. Reduces the minimum to 8.0 mm — and §3.1 suspends that reduction wherever pitting is recorded. |
| 6 calculation cases | Unit-aware: mm vs inch, percentage-of-what, 90 days vs three months, a corrosion rate over a non-integer interval, and one with no answer at all. |
| 3 briefs | Word approval note, four-slide deck, recalculating workbook. |
| Coding task | Two functions, graded by 11 hidden tests. |
| 8 failure cases | Each with a *correct* failure: blocked, partial, refused, or completed-with-the-conflict-surfaced. |

**The conflict is the point.** Pitting is recorded adjacent to point C, no
pitting assessment exists, so Revision D §3.1 is engaged and **9.0 mm governs**.
An agent that cites the newest revision answers 8.0 mm and concludes the vessel
is fine. Answering 9.0 mm *for the wrong reason* also fails: the number and the
reasoning are both graded.

**The grading key is held out and the holding-out is enforced.**
`expected/` sits in the checkout; a run's workspace is `<app data>/runs/<run_id>`;
`agent-baseline.mjs` asserts no `expected/` path resolves inside a workspace root
before grading anything.

**The coding fixture grades in both directions, and both were run.** The
reference implementation passes 11 of 11; the deliberately wrong one fails 8 of
11. A suite nobody has seen pass is not a grading key, and a suite that passes a
broken program grades nothing.

## Baseline result

`evidence/agent-system/P00/baseline.json`. Exact command:

```bash
ARJUN_BASELINE_MODEL_URL=http://127.0.0.1:8080/v1 ARJUN_BASELINE_MODEL_ID=Spark-X2.5-4B-Q8_0 node scripts/agent-baseline.mjs
```

| | |
|---|---|
| cases | 39 |
| **executed** | **9** |
| passed | 9 |
| failed | 0 |
| blocked | 30 |
| skipped | 0 |

**Nine executed out of thirty-nine is the honest reading of this build**, not a
score. Thirty cases are blocked because the agents they grade do not exist yet;
each names the phase that delivers it.

The nine that ran:

| Case | Kind | Result |
|---|---|---|
| `heldout-01-key-outside-every-workspace` | precondition | passed |
| `fixture-01-manifest-current` | precondition | passed |
| `fixture-02-inputs-match-their-hashes` | precondition | passed — 17 inputs |
| `transport-01-health` | **deterministic-transport** | passed |
| `transport-02-unknown-method-refused` | **deterministic-transport** | passed |
| `driver-01-no-model-server-fails-honestly` | real-driver | passed |
| `model-01-real-generation` | **real-model** | passed |
| `coding-01-reference-passes-the-hidden-tests` | **deterministic-fixture** | passed — 11 selected |
| `coding-02-wrong-implementation-is-reported-as-failed` | **deterministic-fixture** | passed — 8 of 11 failed |

Two of the nine are marked `deterministic-transport` and two
`deterministic-fixture`, as §11.1 requires. The other five reach real
infrastructure.

`driver-01` is the one worth reading. `run.start` was pointed at a loopback
address nothing serves. The driver returned `outcome: {kind: "failed", detail:
"Connection error."}`, `stopReason: "error"`, empty text, with a real 7-event
trace. It did not invent an answer.

### A defect found in this harness, and fixed

`model-01-real-generation` first reported **passed** for a run whose own outcome
was `failed: request (8852 tokens) exceeds the available context size (4096)`.
The case checked only that `run.start` returned `Ok` — that the driver answered,
not that the run worked. That is precisely the fabricated pass the plan forbids:
a green case on top of a failed run, with a real trace underneath it.

Fixed at `src-tauri/tests/agent_baseline.rs` — the case now reads
`outcome.kind`, requires non-empty text, and reports `failed` with the reason
otherwise. Re-run at 4096 it correctly reports **failed**; re-run at 16384 it
reports **passed** with 12 characters generated. Both runs are recorded.

## Repository gates run

| Gate | Result |
|---|---|
| `node scripts/check-ipc.mjs` | pass — 172 commands |
| `node scripts/check-reachable.mjs` | pass — 196 modules |
| `node scripts/check-egress.mjs` | pass — one chokepoint, no unapproved hosts |
| `node scripts/check-no-lora.mjs` | pass — 588 product files |
| `cargo check --all-targets` | pass — no errors; pre-existing warnings only |

Raw output: `evidence/agent-system/P00/log_repo-gates.txt`.

## Unverified or blocked

| What | Why | Runnable command |
|---|---|---|
| Sandbox / code execution | The container daemon is not running | Start Docker Desktop, then `npm run baseline:agents` |
| Independent workbook reopen | `openpyxl` is not installed | `python -m pip install openpyxl` |
| Rendered page / slide review | No `soffice` on PATH | Install LibreOffice and put `soffice` on PATH |
| Served context for the other 14 models | Only Spark was served | `llama-server -m <weights> --port 8080 -c <n>`, then `npm run baseline:agents` with `ARJUN_BASELINE_MODEL_URL` set |
| Peak VRAM during generation | Only at-rest VRAM was sampled | Sample `nvidia-smi` during a generation |
| **Every figure on the target machine** | This machine is not established to be it | Confirm the target hardware, then re-run `npm run inventory:hash` and `npm run baseline:agents` there |
| OCR page accuracy | No OCR page was run | Blocked on P06 |
| 30 fixture cases | Their agents do not exist | Blocked on P02, P04, P06–P09, P11–P13 |

## PS 26117 evidence matrix after P00

IDs are this plan's tracking labels (§12.1), not official steps.

| ID | P00's contribution | Evidence |
|---|---|---|
| PS-A | No network path added. Egress gate re-run and clean; the model server runs on loopback; the inventory reaches nothing off-machine. | `log_repo-gates.txt` |
| PS-B | 15 open-weight models registered and inventoried with architecture, quantization and hash. Routing itself is untouched — P01/P03/P05. | `qualification-inventory.json` |
| PS-C | Not addressed. The multi-step agent loop is P01–P05. | — |
| PS-D | Sandbox availability measured and **blocked**: CLI present, daemon down. Honest refusal is a defined failure case (`fail-04`). | inventory `containers`; `failures/failure-cases.json` |
| PS-E | Scanned-page fixtures for the printed and degraded classes; the P&ID excerpt reused for the drawing class. Handwriting and photographs are **not covered** — a gap this pack does not close. | `manifest.json` |
| PS-F | Objective reopen-based checks defined for docx/pptx/xlsx before any authoring agent exists. All blocked on P09/P12/P13. | `expected/artifact-checks.json` |
| PS-G | Two-revision SOP corpus with a real conflict, plus missing-source, revoked-source and denied-source failure cases. | `sources/sop/`, `failures/` |
| PS-H | **Measured on this machine**: served context, VRAM at two context sizes, load time, and the 8584-token schema floor. Explicitly not a target-machine measurement. | `served-context-observations.json` |
| PS-I | Not addressed. End-to-end routing is P05–P10. | — |
| PS-J | Failure cases defined for all three demonstrations. None exercisable yet. | `failures/failure-cases.json` |
| PS-K | **Satisfied for the fixture pack.** Every fixture is synthetic or already in the repo; nothing came from a plant, vendor or customer; the manifest records provenance and licence. | `manifest.json` |

## P00 close-out

- **Implemented** — inventory collector, fixture-manifest generator, baseline
  harness, Rust driver target, 19-file fixture pack, 6 package entry points,
  the plan checked in, this ledger.
- **Wired** — all six entry points run from `package.json`; the Rust target
  compiles under `cargo check --all-targets` and runs under
  `cargo test --test agent_baseline`; the harness invokes the real production
  driver and the real hidden tests.
- **Tested** — 9 cases executed, 9 passed, 0 failed. Real-model generation
  confirmed through the production driver against Spark-X2.5-4B Q8_0 at a 16384
  served context. One harness defect found and fixed during the phase.
- **Unverified or blocked** — 30 fixture cases (agents not built), the sandbox
  (daemon down), renderers (absent), 14 models' served context (not served), and
  every target-machine figure (machine not established).
- **Remaining** — nothing in P00.

**Exact next step:** P01 — make agent definitions and tools executable
contracts. Begin at `src-tauri/src/subagents/packet.rs:121`, which carries
`agent_id` and no `definition_version`; that single omission is finding 9 and is
the dependency for proving an administrator edit reaches the next child.

---

# P01 — Agent definitions, jobs, tool and result contracts

Worked across two sessions on 2026-09-20 and 2026-09-24, on `main` at HEAD
`b9ff7fb`. Nothing is committed: the working tree carries P00 and P01 together.
The first session's P01 work was found uncommitted and not recorded here; it
was re-read, kept, tested and is recorded below with the second session's.

**Contract map:** [`2026-09-20-agent-system-contract-map.md`](2026-09-20-agent-system-contract-map.md)
— every tool's real routing path, the job and result contracts, and the known
deviations with their owning phase. Machine-readable: `agent-runtime/src/tool-contract.json`.

## Traced before changing

Registration: `lib.rs` loads `agents/*.md` → `SubagentManager` +
`AgentRegistry::import_bundled`. Dispatch: runtime `tool.execute` →
`agent_runtime::execute` → fallback → `runner_for` → `LocalToolRunner::delegate_to_subagent`
→ `SubagentManager::spawn` → worker. Result: `ChildResult` → `settle` →
rendered text for the parent. Tools: TS `catalogue.ts` → `tool.catalogue` →
`buildTools` → `tool.authorize` → `ToolGateway::decide` → `tool.execute`.
`orchestrator::executor` and `orchestrator::grammar` have **no production
caller**; the agent path is the only dispatcher.

## Implemented

| Area | Change | Files |
|---|---|---|
| Definitions at dispatch (finding 9) | `DefinitionSource::resolve` reads the registry once per dispatch; the copy is pinned on the packet; keys are `ag-` ids or bundled role keys, never display names; the worker is found by capability (`capability_for(output_schema)`); a disabled agent is refused and not replaced by its bundled profile. | `subagents/definitions.rs` (new), `manager.rs`, `lib.rs`, `runner.rs` |
| §13 import defect | Imports store the profile body, not its description; rows written wrongly by the old import are repaired once, with a version bump. | `agents/store.rs` |
| Job packet | Adds definition version/origin, capability, instructions + hash (hash only is serialised), skills, attempt id, model policy (with a computed `within_eligible`), sharing policy; `job_id()`; a field-by-field contract table on the type. | `subagents/packet.rs`, `definitions.rs`, `manager.rs`, `agent_runtime/mod.rs` |
| Result contract | `partial`, `blocked`; artifact versions, receipts, validation checks, `missing`; new lists enter the hash only when non-empty; `with_receipt` refuses a receipt with no event. | `subagents/result.rs` |
| New role schemas | `document`, `deck`, `workbook` registered; `capability_for` gives them no worker; dispatch and the test-run command refuse them as "registered and cannot yet be run"; the Agents screen labels them "(no worker yet)". | `subagents/profile.rs`, `definitions.rs`, `src/services/agentRegistry.service.ts`, `src/pages/Agents.tsx` |
| Tool contract | `orchestrator::contract`: one `ToolContract` per tool gathering the existing authorities and adding route, prerequisite (+ where checked), output kind and cancellation, all as exhaustive matches; published as JSON and checked from both languages. | `orchestrator/contract.rs` (new), `agent-runtime/src/tool-contract.json` (generated) |
| Gateway | Arguments must be an object; **undeclared arguments refused**; `List` kind; 12 tools' specs corrected to match the handlers and the model's schema (§3 of the contract map). | `orchestrator/gateway.rs`, `tools.rs`, `grammar.rs` |
| Catalogue | Offered prerequisites asked of the contract; the fallback refuses, by name, a tool the contract routes to the agent path. | `agent_runtime/mod.rs` |
| TS mirror | `artifact.list` / `artifact.read` defined and named (never offered before); delegation inputs offered; search `page` removed (nothing read it); `notebook.delete` side-effecting. | `agent-runtime/src/{catalogue,tool-names}.ts` |
| Writer delegation | `DelegationMode::{ReadOnly, Writer}`: read-only refuses writing roles and withholds write tools; writer never widens and takes the exclusive lane whenever it holds a write tool; no model-facing tool can request it. | `subagents/manager.rs` |
| Name dependence removed | The runner's empty-input rule (was the literal `"knowledge-retriever"`); the Agents screen's `workerAvailable`/`ready` (was profile name, else display name); `agent_test_run` (dispatched a clone by display name — never resolvable); `agent_dependents` (matched only the bundled name — registry children were invisible). | `orchestrator/runner.rs`, `commands/agents.rs`, `commands/agent_admin.rs` |
| Plausible-number fallback | An artifact reference with no revision was given revision 1. Now refused; `art-7@4` or `revision` accepted. | `orchestrator/runner.rs` |

No IPC command was added (172, unchanged). `AgentView`'s JSON shape is
unchanged; only its values are now correct for clones and for schemas with no worker.

## Contract decisions

1. **The contract gathers; it does not duplicate.** `spec_for`, `class_of`,
   `is_side_effecting`, `retry_policy_of` and the `ToolName` spellings stay the
   authority for their own facts. `ToolContract` reads them and adds the four
   facts that had no home.
2. **Generated, never typed — across languages.** `tool-contract.json` is
   rendered by Rust and checked byte-for-byte by a Rust test; the runtime's
   conformance test checks the TS catalogue against it (names, required and
   optional arguments, kinds, read-only, side-effecting, output kind, aliases).
   The hand-typed lists on the TS side had drifted exactly where it mattered.
3. **The gateway's schema is closed**, like the model's. Required and optional
   are the whole input schema. The grammar learned optional members rather than
   the specs pretending they were required.
4. **A capability is decided by output schema**, and is the one key tied to
   code. Everything else (id, display name, role key) may change or multiply.
5. **Registered is not executable.** A schema with no worker has a contract, is
   savable, and is refused at dispatch, at test-run and in readiness.
6. **Recorded is not enforced, and says so.** The packet carries the model
   policy and the skills; routing does not yet hold the eligible set and the
   child loop does not load skills. Both are named as such (P03, P05).
7. **Writer semantics exist before any writer tool.** `Writer` is a manager
   contract with a single constructor that no model-facing tool calls.

## Tests

All deterministic. Every test below drives the production `SubagentManager`
against a real `AgentRegistry` in a temporary directory, or the production
gateway, or the production `authorize` → `execute` path — no mocked handler.

| Property the plan names | Test |
|---|---|
| admin edits affect the next child | `subagents::definitions_tests::an_administrators_edit_reaches_the_next_child` |
| active jobs keep their definition | `…::a_running_child_keeps_the_definition_it_was_sent_under` (edit saved while the worker is held mid-run) |
| clone / rename | `…::a_clone_is_performed_by_its_capability_and_answers_to_its_own_id`, `…::renaming_an_agent_changes_nothing_about_how_it_is_dispatched`, `commands::agents::tests::the_worker_is_the_capability_of_the_output_and_not_a_name` |
| unsupported capabilities stay blocked | `…::a_schema_with_no_worker_is_registered_and_refused`, `…::a_disabled_agent_is_refused_and_not_replaced_by_its_bundled_profile`, `…::the_agents_screen_offers_exactly_the_schemas_a_worker_produces` |
| malformed tool arguments refused | `orchestrator::gateway::tests::{an_undeclared_argument_is_refused_and_the_real_ones_are_named, a_tool_with_no_arguments_refuses_any, a_payload_that_is_not_an_object_is_refused, a_list_argument_given_as_anything_else_is_refused}`, `agent_runtime::tests::tool_contract_path::an_undeclared_argument_is_refused_on_the_production_path`, `orchestrator::runner::tests::an_artifact_reference_without_a_revision_is_refused_not_defaulted` |
| child never exceeds parent | `…::an_edit_cannot_widen_a_child_beyond_its_parent`, `…::no_shipped_role_is_ever_granted_what_its_parent_does_not_hold` (every shipped role × five parent grants), `…::a_read_only_dispatch_withholds_a_write_tool_the_parent_does_hold`, `…::a_role_declared_as_writing_is_refused_by_a_read_only_dispatch`, `…::two_writers_never_share_the_reader_lane` |
| older saved records | `…::a_packet_recorded_before_definitions_were_pinned_still_reads` (strips every added field), `…::a_result_recorded_before_the_contract_grew_reads_and_keeps_its_hash`, `…::the_new_schemas_survive_a_round_trip_through_the_registry_file`, `…::a_row_the_earlier_import_wrote_wrongly_is_repaired_exactly_once` |
| canonical aliases | `orchestrator::gateway::tests::every_accepted_spelling_is_decided_as_the_tool_itself`, `orchestrator::contract::tests::every_listed_alias_resolves_to_its_own_tool_and_to_no_other`, runtime `catalogue.conformance.test.ts` "resolves every alias this runtime knows…" |
| the real path | `agent_runtime::tests::tool_contract_path::{a_delegation_the_schema_allows_reaches_the_delegation_handler, a_page_read_without_its_optional_end_page_is_not_refused_by_the_gateway}`, `orchestrator::runner::tests::the_runner_serves_exactly_what_the_contract_routes_to_it` |
| the job contract | `…::a_packet_carries_the_whole_job_contract`, `…::a_model_outside_the_eligible_set_is_recorded_as_outside_it`, `…::a_receipt_with_no_event_behind_it_is_refused` |

**Seen failing.** Three regression tests were run against the fix reverted —
`two_writers_never_share_the_reader_lane`, `an_undeclared_argument_is_refused_…`,
`a_tool_with_no_arguments_refuses_any` — and all three failed; restored, all
pass. A test nobody has seen fail grades nothing.

## Checks run

Raw output: `evidence/agent-system/P01/log_checks.txt`.

| Check | Result |
|---|---|
| `cargo test --lib` (full, before the grammar fix) | 2675 passed, **1 failed** — `grammar::a_tool_that_takes_arguments_can_be_sent_them`, which the optional-argument change broke. Fixed by teaching the grammar optional members. |
| `cargo test --lib -- orchestrator:: subagents:: commands::agents commands::agent_admin agent_runtime::tool_policy agent_runtime::tests` | 421 passed, 0 failed (every P01 test named above confirmed in the run) |
| `cargo test --lib -- agent_runtime::tests::tool_contract_path` | 3 passed |
| `cargo test --lib --no-fail-fast` (full, after every fix) | **2679 passed, 0 failed**, 2 ignored |
| `npm run check:targets` | pass |
| `npm run runtime:typecheck`, `npm run runtime:test` | pass; 131 files, 2291 tests |
| `npx tsc --noEmit -p tsconfig.json`, `npm run test:ui` | pass; 43 files, 618 tests |
| `npm run runtime:build`, `npm run check:bundle:self` | pass; bundle sha-256 `7eafd6f0…` |
| `check-ipc`, `check-reachable`, `check-egress`, `check-no-lora` | pass — 172 commands, 196 frontend modules, one chokepoint, 591 files |
| `node scripts/agent-baseline.mjs --out evidence/agent-system/P01/baseline.json` | 39 cases: 8 executed, 8 passed, 0 failed, 31 blocked. P00 executed 9: `model-01-real-generation` is **blocked** here because no model server was running — not a regression, and not a pass. |

## Measured

**P01-OBS-1 — the tool-schema floor rose by 455 estimated tokens**, from 8584
(P00) to **9039**, per the driver's own `context_ledger` in `driver-01`. That is
the cost of actually offering `artifact.list`, `artifact.read` and the
delegation inputs, net of removing search's `page`. P00-OBS-1 stands and is
slightly worse: a 4096-token served context cannot complete one turn. Role-scoped
tool loading (P03) is the remedy; nothing here measured it.

## Unverified or blocked

| What | Why | Runnable command |
|---|---|---|
| Real-model generation through the new catalogue | No model server was running this session | `llama-server -m <Spark-X2.5-4B-Q8_0.gguf> --port 8080 -c 16384`, then `ARJUN_BASELINE_MODEL_URL=http://127.0.0.1:8080/v1 ARJUN_BASELINE_MODEL_ID=Spark-X2.5-4B-Q8_0 node scripts/agent-baseline.mjs` |
| A model choosing `artifact.list` / delegation unprompted | Needs a real model on the full catalogue | as above, with a prompt that refers to an earlier artifact |
| The installed desktop app | Not rebuilt or redeployed; the installed copy predates P01 (see the deploy memory: the app is a manual copy) | `npm run tauri build`, then replace the installed binary |

## P01 close-out

- **Implemented** — definitions resolved and pinned at dispatch; the full job
  packet; the extended result contract with partial/blocked; three writer role
  schemas registered and not executable; one tool contract, published and
  checked from both languages; a closed gateway schema; writer delegation
  semantics; name-keyed dispatch replaced by capability in five places.
- **Wired** — through the production path: runtime schema → `tool.catalogue` →
  `tool.authorize`/`ToolGateway` → `tool.execute` → route → handler; the manager
  used by `lib.rs` reads the registry the Agents screen writes.
- **Tested** — the table above; mutation-checked; full lib, runtime and UI
  suites green.
- **Unverified or blocked** — real-model behaviour on the enlarged catalogue;
  the installed app.
- **Remaining** — nothing in P01's scope. Deviations found and deliberately left
  to their owning phases are listed in §7 of the contract map.

**Exact next step:** P02 — begin at `subagents/worker.rs:391`, which publishes
`event_seq: 0`, and at `knowledge/graph/runtime_memory.rs:507`, which admits only
`event_seq > 0`; the receipt must name the child's own successful tool event.

---

# P03 — Context assembly, model scheduling and durable handoff

Worked on 2026-09-24 on branch `claude/affectionate-cori-dsvo6e`, starting at
HEAD `9dd01ec`, in a cloud Linux container (Ubuntu 24.04, 4 vCPU, 15 GiB RAM,
**no GPU, no `llama-server`, no model weights**). Nothing here is a measurement
on the target machine, and nothing claims to be.

**P02 was not closed when P03 started** (status above: in progress, no P02
evidence directory). P03 depends on it. What P03 did that overlaps P02, and what
it deliberately left there:

| P02 item | State after P03 |
|---|---|
| Finding 1 — cursor labelled onto rows not read at it | **Fixed on the compile path.** `MemoryGraph::snapshot_scopes` reads every scope and the cursor in one transaction; the manifest's `graph_revision` is that cursor and a requested revision that differs is recorded separately as `requested_revision`. Historical (old-revision) reads are still not implemented — P02. |
| Finding 2 — worker receipts published with `event_seq: 0` | Untouched — P02. |
| Authority migration, outbox reconciliation, `shared_with_task` enforcement | Untouched — P02. |

## Traced before changing

- **Scheduling.** `subagents::scheduling::ModelScheduler` served subagent workers
  only: a per-model semaphore of **two** request slots plus a card mutex taken
  only when residency said `Serialise`. The parent run (`commands::agent::drive_run`)
  held no lease at all; OCR (`commands::ocr`, two sites), the model handoff (two
  sites), the administrator's orchestrator swap and the in-process
  `prepare_model_for` each called `serving::admission::admit` on their own.
  Consequences found: two generations on one warm model at once; a child's
  admission could stop the parent's server while the parent awaited that child,
  and the parent's next round then went to a dead base URL; a child waiting on a
  card its parent held was a deadlock with nothing to break it; the swap command
  stopped every server, including one mid-generation.
- **Context.** `context.refresh` compiled task scope only, read the cursor and
  the rows in two separate reads, was sent `reservedToolSchemas/Output/Framing: 0`
  by the runtime, budgeted against the window the run was *started* with, logged
  a mandatory overflow and carried on, and persisted no manifest. There was no
  exact count of the outgoing request anywhere — only chars/4 with a drift factor.
- **Continuity.** `model_handoff` already implements drain → settle → checkpoint
  → validate → load → recompile → commit with rollback and crash reconciliation.
  What was missing was the per-round path: a server restarted under a running
  run, and a load that fails mid-run or at the start of a turn.
- **Second scheduler check.** `ai_engine::scheduler::GenerationScheduler` is the
  in-process engine's job queue for the external-tool gateway; it is not an
  admission service and was not duplicated. No new scheduler was created: the
  existing `ModelScheduler` *became* the one service.

## Implemented

| Area | Change | Files |
|---|---|---|
| One lease/admission service | `ModelScheduler` is now a single book with capacity `ONE_HEAVY_CALL = 1` for the whole process. Classes `parent`, `child`, `ocr`, `vision`, `background`. Bounded queue (`MAX_WAITING = 16`, refused beyond), per-request wait deadline and cancellation token, FIFO for interactive work with background ageing into the same tier after 30 s (starvation guard), per-holder hold limit expired only when somebody waits (leak guard), re-entrant per owner (a worker's guard and its child's own rounds are one holder). Eligible-model set from the job packet is **enforced** here (`OutsideEligible`). `ensure_served` loads only for a holder and reports `restarted` / `cache: cold` honestly. `serve()` and `bind_run_until_dropped()` make release-on-every-path the default. The per-model two-slot semaphore is gone. | `subagents/scheduling.rs`, `subagents/mod.rs` |
| Parent/child deadlock | A child's request suspends every *holding* ancestor — recorded through the observer **before** the capacity is released — and is admitted at once. The worker additionally suspends the parent explicitly before a model-needing child, so the suspension is on the parent's record even when the parent had already released at its tool boundary. The parent's next round re-acquires (recorded as resumed) and re-binds its endpoint. | `subagents/scheduling.rs`, `subagents/worker.rs` |
| Every heavy consumer leased | Parent rounds (`context.refresh`), the parent's initial load (`drive_run`), children (worker + `child_loop` serving through `ensure_served`), OCR pages (both commands), the handoff's load and rollback reload, the orchestrator swap, and `prepare_model_for`. The lease service is created once in `lib.rs`, managed, and its observer persists `model_lease_suspended` / `model_lease_resumed` to the owner's run record. | `lib.rs`, `commands/{agent,agents,ocr,registry}.rs`, `agent_runtime/model_handoff.rs`, `subagents/child_loop.rs` |
| Round boundary (Rust) | New `agent_runtime::rounds`: `context.refresh` = lease → re-bind → compile against the **served** window → `ContextCompiled` event with the whole manifest → refuse with `round_refused` (and give the card back) on mandatory overflow or a context that cannot be compiled; `context.count` = the exact outgoing payload rendered by the served model's own template and counted by its own tokenizer (`/apply-template` + `/tokenize`, loopback-only, in the audited probe module), images accounted separately (unmeasured until the server's own usage report measures them), fit judged against window − reply − safety, projection check (every authorised block reached the wire, nothing block-shaped was introduced), `model_requested` event, refusal on overflow; `context.settle` = release + reconcile against reported usage + learn per-image cost, `model_responded` event. `tool.authorize` releases the round (a tool call means the round finished generating). Round state self-prunes to live runs. | `agent_runtime/rounds.rs`, `agent_runtime/mod.rs`, `agent_runtime/protocol.rs` (`ROUND_REFUSED`), `serving/probe.rs` (`count_rendered`) |
| Round boundary (runtime) | `RoundBoundary`: opens the round in `transformContext` *before* compaction (the compactor's own summary call must hold the card and reach the re-bound server), sends the stream to the re-bound base URL, chains `context.count` onto the provider's `onPayload` so the exact body is counted, settles on the stream's `result()`. `round_refused` fails the run with Rust's sentence and the model is not called; anything else degrades and says so. Real reserves are now sent (tool schemas, reply, framing, a 3 %/≥256 safety policy) and the run's `capability`. | `agent-runtime/src/{context-refresh,run}.ts` |
| Scope composition | Four scopes read at one cursor, each authorised on its own terms: task (precedence 0), project/workspace (1, established only), activated procedure (2, established, unexpired, `Applicability` must match the reader's capability/agent — absence is not a wildcard), preference (3, the reader's own user scope, established). Expiry parsed as an instant, not text. Each selected item records scope and precedence; each block from outside the task says what it yields to; the manifest carries a per-scope count record. `MemoryKind::{Preference, Procedure}` and `MemoryItem::applies_to` added (serde-defaulted; UI mirrors updated). | `agent_runtime/context_compiler.rs`, `agent_runtime/context_manifest.rs`, `knowledge/graph/{runtime_memory,runtime_store}.rs`, `src/services/memoryGraph.service.ts`, `src/components/graph/AgentMemoryPanel.tsx` |
| Retrieval seam | `RetrievalProvider` trait; `LexicalRetrieval` is the only shipped provider and is recorded as degraded "until P07"; a failing provider falls back to lexical and the record says so. | `agent_runtime/context_compiler.rs` |
| Budget and compaction | `Reserves.safety`; `BudgetRecord.{reserved_safety, window_source}`; exact artifact references (`id@revision sha256:<full>`) and source addresses rendered on every block; older optional material that does not fit is moved into a deterministic **digest** block (ids at revisions, graph cursor, how to recall) — never a model-written summary; the runtime compactor names the transcript range and tool-call ids its summary stands in for, and its ceiling pass now drops a tool call **together with its results** (it used to walk the cut backwards and truncate every message instead). Manifest version 3; v1/v2 manifests still verify (new fields are omitted from JSON and seal when empty — tested). | `agent_runtime/{context_compiler,context_manifest,recording}.rs`, `agent-runtime/src/compaction.ts` |
| Load failure | `serving::fallback`: failure classes read off the serving error (OOM, files missing, unsupported runtime, never ready, launch failed); the Q8/Q4 rule as a pure function (≤4.4 B → 8-bit, else 4-bit; OCR/embedding exempt; unknown width is not conformance); `base_identity` so two widths of one model are recognised as the same model; `decide()` → `retryLater` / `useAlternative` / `blocked`, **never** a lower quantisation of the failed model. Mid-task: retry later on the same model (a model change mid-task is a handoff). Pinned child job: never substituted. Fresh turn in `drive_run`: at most one alternative that passes the failed model's gates (role, classification, modalities) and the Q8/Q4 rule, recorded in the routing reasons. Mid-run failures record `model_load_failed` with the decision. | `serving/fallback.rs`, `agent_runtime/rounds.rs`, `commands/agent.rs`, `subagents/child_loop.rs` |
| Identity across hops | The child loop now sends its pinned `definitionVersion` (was `0`) and its `capability`. No KV or prompt cache is ever carried: a new server process is reported `cache: cold`. | `subagents/child_loop.rs` |
| Native gate | `scripts/p03-native-gate.mjs` (`npm run test:p03:native`) checks GPU, `llama-server`, registry and the four models' weights; reports **blocked** (exit 2) when any is missing, otherwise runs `tests/p03_native_chain.rs` (Spark → OCR → Qwen → reviewer → Spark on real servers under the production lease service, exact per-model token counts, served windows, VRAM, prior write and correction carried exactly once) and records executed/passed/failed separately. | `scripts/p03-native-gate.mjs`, `src-tauri/tests/p03_native_chain.rs`, `package.json` |
| Events | `context_compiled`, `model_lease_suspended`, `model_lease_resumed`, `model_load_failed` (observational); `model_requested` / `model_responded` are now actually written. | `agent_runtime/events/{model,machine,projection}.rs` |

No IPC command was added (172, unchanged). The three new runtime methods are
stdio RPCs between the runtime and Rust, like `context.refresh` before them.

## Contract decisions

1. **One book, capacity one.** Co-residency is a later measured optimisation
   (plan §4); a capacity above one exists only for tests and a future measured
   configuration. With nothing else in flight, admission may stop any idle
   server — it can never stop one mid-generation.
2. **A round holds the card only while it generates.** Released at the tool
   boundary, at settle and at run end. That is what makes a parent/child
   deadlock impossible by construction; ancestor suspension is the second guard.
3. **Suspension is recorded before release**, through an observer the book calls
   with the holder still in place, so a crash while the child runs still shows
   why the parent was not holding.
4. **The rendered request is counted by the served model, or it is recorded as
   not counted.** No estimate is substituted for a count; images are
   "unmeasured" until the server's own usage report measures them.
5. **Mandatory state and a rendered request that does not fit are refusals, not
   warnings.** The model is not called; the run fails with the numbers.
6. **Precedence is scope order** (task > project > procedure > preference), stated
   on the block the model reads; runtime policy and the current request sit above
   all of memory.
7. **A fallback never changes quantisation** and never switches a model under
   work already done.

## Tests

All deterministic unless marked. Rust tests drive the production handlers
(`context.refresh` / `count` / `settle`, `tool.authorize`), the real memory graph
and event log (on disk for the restart), and the real lease service; the only
fake is the model server backend.

| Property the prompt names | Test |
|---|---|
| global one-heavy-call policy, disproving the two-per-model policy | `subagents::scheduling::tests::two_rounds_on_the_same_warm_model_no_longer_run_at_once`, `every_class_of_heavy_work_shares_one_slot` (15 concurrent holders of 5 classes, peak 1), `the_in_flight_measurement_would_see_a_second_holder` |
| parent released/suspended before a capacity-dependent child; reacquire/rebind | `…::a_child_suspends_its_parent_before_taking_the_card_and_never_waits_on_it` (asserts the record was written while the parent still held), `a_parent_and_three_children_in_turn_never_overlap_and_leave_the_book_empty`, `a_grandchild_suspends_a_holding_grandparent`, `an_explicit_suspension_is_recorded_even_when_nothing_was_held`, `agent_runtime::rounds::tests::delegation_suspends_the_parent_on_the_record_and_its_next_round_resumes_it`, `a_tool_boundary_gives_the_card_back_before_the_tool_runs` |
| bounded queues, timeouts, cancellation, starvation, leaks | `the_queue_is_bounded_and_refuses_rather_than_growing`, `a_wait_that_times_out_leaves_nothing_behind`, `a_cancelled_wait_returns_promptly_and_leaves_nothing_behind`, `fresh_background_work_waits_behind_interactive_work`, `aged_background_work_is_not_starved_by_newer_interactive_work`, `a_holder_past_its_limit_is_expired_for_a_waiter_and_the_expiry_is_recorded`, `an_old_guard_cannot_release_a_newer_grant`, `a_bound_run_is_forgotten_when_its_driver_leaves_by_any_path`, `serving_one_piece_of_work_leaves_nothing_behind_either_way`, `rounds::tests::a_stopped_run_stops_waiting_for_the_card`, `round_state_for_finished_runs_is_pruned` |
| tokenizer-dependent overflow | `rounds::tests::overflow_depends_on_the_tokenizer_and_the_dense_one_is_refused` |
| huge tool output | `rounds::tests::a_huge_tool_output_is_refused_before_it_reaches_the_model`; runtime `compaction.test.ts` "drops a tool call together with its results…" |
| small-window transition | `rounds::tests::a_small_window_transition_is_rebound_recompiled_and_refused_when_it_must_be`, `a_round_takes_the_lease_and_budgets_against_the_served_window`, `scheduling::tests::a_rebind_onto_a_smaller_window_reports_the_window_it_actually_got`; runtime "fits the next projection to the smaller window the server came back with" |
| mid-task correction | `rounds::tests::a_mid_task_correction_reaches_the_very_next_round`, `context_compiler::tests::a_task_correction_outranks_an_older_preference_and_the_block_says_so` |
| cancel / OOM / load failure | `scheduling::tests::an_out_of_memory_load_is_classified_and_the_quantisation_is_untouched`, `rounds::tests::a_load_failure_is_refused_as_recoverable_and_changes_nothing_else`, `serving::fallback::tests::*` (10, incl. `a_failed_load_never_falls_back_to_a_lower_quantisation_of_the_same_model`, `a_pinned_job_is_never_substituted_even_when_an_alternative_exists`) |
| restart | `scheduling::tests::after_a_restart_nothing_is_held_and_the_resumed_run_starts_cold`, and the chain below |
| scope composition, precedence, applicability, expiry, per-scope ACL | `context_compiler::tests::{all_four_scopes_are_composed_at_one_cursor_and_recorded, a_procedure_for_another_role_is_not_applied_and_absence_is_not_a_wildcard, a_proposed_procedure_is_a_candidate_and_is_not_applied, an_expired_preference_is_not_applied_whatever_the_timestamp_spelling, another_persons_preferences_and_another_projects_knowledge_are_invisible}` |
| retrieval provider interface, honest lexical label | `…::a_provider_decides_the_ranking_and_is_recorded_as_itself`, `a_failing_provider_falls_back_to_lexical_and_says_so`, `lexical_retrieval_is_reported_as_degraded_not_as_semantic` |
| compaction with source ranges and omissions; exact artifact refs | `…::older_material_that_does_not_fit_is_named_in_a_deterministic_digest`, `a_receipt_names_its_artifact_by_exact_revision_and_hash`, `rounds::tests::a_compactions_source_range_is_recorded_without_any_text`; runtime "names the range and the tool calls a summary replaced", "names the tool calls the ceiling pass dropped" |
| manifest persists versions, true cursor, budget | `rounds::tests::a_round_takes_the_lease_and_budgets_against_the_served_window` (reads the `context_compiled` event), `context_compiler::tests::the_manifest_records_the_cursor_budget_and_hashes_and_stays_sealed` (cursor read ≠ cursor requested), `context_manifest::tests::a_version_two_manifest_with_a_budget_still_verifies_after_the_fields_grew` |
| multimodal accounting | `rounds::tests::image_cost_is_learned_from_the_server_and_then_charged`, `a_server_that_cannot_render_is_recorded_as_uncounted_not_estimated` |
| runtime boundary | `run.test.ts` "the round boundary" (5): re-bound endpoint used; refresh → count → settle order with the exact payload; refused count never reaches the model; refused refresh fails the run; an old Rust side degrades |
| **contract harness** (deterministic, *not* the native chain) | `agent_runtime::p03_chain_tests::a_prior_write_and_a_correction_survive_four_models_and_a_restart_exactly_once`: Spark → OCR → Qwen → reviewer → Spark over on-disk stores, one holder at every hop, the prior write's receipt (`art-note@1 sha256:…`) carried exactly once by all five models and after a process restart, the correction exactly once (as a mandatory constraint) by every model after it arrived, the effect ledger replaying the write rather than running it — before and after the restart — and exactly one receipt and one correction in the graph at the end |

**Found in review and fixed.** A last adversarial read of the round path found
that an error *after* the lease was taken (the session gone, a graph read
failing) left `context.refresh` without releasing the card, and — because the
runtime degrades on any error but `round_refused` — the model would then have
been called with no context at all. Such a round is now refused and gives the
card back at once:
`rounds::tests::a_round_whose_context_cannot_be_compiled_is_refused_and_gives_the_card_back`
(fails with the fix removed; so do the two overflow tests, which share the
give-back path).

**Seen failing.** Every load-bearing test was run against a mutation and failed:
capacity set to 2 (8 of the then 24 scheduler tests fail); ancestor suspension
disabled (3 fail: child/parent, grandchild, parent-and-three-children);
tool-boundary release removed together with the mandatory-overflow refusal
(3 of the then 16 round tests fail: the tool-boundary test and both overflow
tests); the runtime's endpoint swap disabled (1 fails: "sends the round to the
endpoint Rust re-bound"); artifact references dropped from rendering (the chain
harness fails at Spark's first round: receipt carried 0 times); corrections
demoted from the mandatory set (the chain harness fails: kind `neighbour`, not
`constraint`). Each mutation was reverted and the suite re-run green.

## Checks run

Raw output: `evidence/agent-system/P03/log_checks.txt`.

| Check | Result |
|---|---|
| `cargo test --lib --no-fail-fast` | **2781 passed**, 2 failed, 3 ignored. HEAD `9dd01ec` in a clean worktree on the same machine: 2720 passed, the **same 2 failed** — both assert Windows path forms (`C:/Windows`, `\\server\share`) and are pre-existing on Linux. |
| focused: scheduling / fallback / rounds / compiler / manifest / chain | 26 / 10 / 19 / 33 / 7 / 1 passed |
| `cargo test --test` agent_runtime, agent_baseline, two_runtimes, production_acceptance, production_hardening, rbac_isolation | 15 / 4 / 5 / 12 / 8 / 12 passed (agent_baseline drives the rebuilt runtime bundle) |
| `cargo test --test` subagent_model_loop_live, model_binding_handoff_live | Fail at `tests/common/mod.rs:54` (`APPDATA` is Windows-only); identical at HEAD. **Not execution coverage** — live tests need the target machine. Updated to the new lease API; `two_workers_on_one_model_take_turns_on_the_card` now asserts the new policy (it asserted the old one). |
| `cargo check --all-targets` | pass; no new warnings in changed files |
| `npm run test:integration` (the CI set, 11 targets) | 63 passed, 0 failed, 1 ignored (`seed_local_accounts`, ignored at HEAD) |
| `npm run build`, `check:bundle`, `check:offline` (the rest of CI) | pass |
| `npm run runtime:typecheck`, runtime vitest | pass; 131 files, **2301** tests (was 2291) |
| `npx tsc --noEmit -p tsconfig.json`, `npm run test:ui` | pass; 43 files, 618 tests |
| `npm run runtime:build`, bundle self-test | pass |
| `check-ipc`, `check-reachable`, `check-egress`, `check-no-lora` | pass — 172 commands, 196 modules, one chokepoint, 605 files |
| `node scripts/agent-baseline.mjs --out evidence/agent-system/P03/baseline.json` | 39 cases: 8 executed, 6 passed, **2 failed**, 31 blocked. The two failures are `fixture-01/02`: the fixture manifest's hashes were computed on the Windows working copy (CRLF) and this Linux checkout has LF — verified by hashing the CRLF form, which matches. Pre-existing and platform-specific; not regenerated here, because regenerating on Linux would break it on Windows. |
| `node scripts/p03-native-gate.mjs` | **BLOCKED** (exit 2): no NVIDIA GPU, no `llama-server`, no model registry on this machine. `evidence/agent-system/P03/native-gate.json`. |

## Measured

Nothing on the target machine. No model ran in this session.

## Unverified or blocked

| What | Why | Runnable command |
|---|---|---|
| P03 native gate (real Spark / OCR / Qwen / reviewer swaps under the lease, exact per-model counts, served windows, VRAM) | This machine has no GPU, server or weights | On the RTX 5060 machine: `ARJUN_MODELS_DIR=<app data>/com.arjun.workbench/models npm run test:p03:native` (override model ids with `ARJUN_P03_{SPARK,OCR,QWEN,REVIEWER}`) |
| The full native specialist chain with roles | Needs P05/P06/P09/P10 | P10's gate, repeated in P16 |
| Image-token cost on the real projectors | Learned only from a real server's usage report | The first vision/OCR round through the agent path on the target records it (`model_responded.imageCostLearned`) |
| `/apply-template` + `/tokenize` on each installed model's template | Needs the real servers | The native gate records `renderedTokens` / `countError` per hop |
| The installed desktop app | Not rebuilt or redeployed | `npm run tauri build`, then replace the installed binary |

## Known gaps left open (with owner)

| Gap | Why it is not closed here | Owner |
|---|---|---|
| The external-tool gateway's in-process generations (`ai_engine::scheduler`) do not take the lease | Off by default; serves external clients, not agent work; `admission::admit` already unloads the in-process model under a lease when it needs room | P14 or a gateway follow-up |
| Rust tool handlers are not interrupted at their `ToolSpec::timeout` (contract map §7) | Interrupting a side-effecting handler creates an effect nobody can account for (the reason `authorize` lets a running tool finish); a read-only-only deadline is possible but was not needed for the lease guarantees | P04/P05 |
| Historical (old-revision) graph reads | Store serves current rows; P03 records requested vs read cursor instead of pretending | P02 |
| Procedure lifecycle (candidate → evaluated → activated) | P03 applies only *established* procedures and treats a proposal as a candidate; the promotion workflow is P15 | P15 |
| Embedding/rerank retrieval | Seam in place, lexical labelled degraded | P07 |

## PS 26117 evidence matrix after P03

IDs are this plan's tracking labels (§12.1), not official steps.

| ID | P03's contribution | Evidence |
|---|---|---|
| PS-A | The exact-count call is loopback-only in the audited probe module; egress gate re-run and clean. No new network path. | `log_checks.txt` |
| PS-B | Multiple registered models share one card under one scheduler; load failure has an honest, quantisation-preserving fallback decision; the child's definition-allowed model set is enforced. Native routing demo is P05/P16. | scheduler + fallback tests |
| PS-C | Every model round re-reads corrections and receipts, fails explicitly instead of silently dropping state, and survives model swaps and restart. | rounds + chain tests |
| PS-E | OCR pages now run under the one lease; image tokens are accounted honestly. OCR quality is P06. | `commands/ocr.rs`, image test |
| PS-H | The one-heavy-call policy for an 8 GB card is implemented and tested; recoverable overload (queue bound, timeouts, OOM classification). **Target-machine measurement blocked.** | scheduler tests; `native-gate.json` |
| PS-D, F, G, I, J, K | Not addressed by P03. | — |

User-added requirements covered: model-independent memory across model change
and restart (chain harness), Q8/Q4 retention, Spark as the resumed parent.

## P03 close-out

- **Implemented** — the one lease/admission service (capacity one, classes,
  bounded queue, ageing, deadlines, cancellation, hold limit, re-entry,
  ancestor suspension recorded before release, eligible-set enforcement,
  serving and honest rebind); every heavy consumer routed through it; the
  runtime round boundary (refresh before compaction, re-bound endpoint, exact
  rendered count with multimodal accounting, settle); four-scope context
  composition with precedence, applicability, expiry and per-scope ACL; the
  retrieval seam; safety reserve, digest compaction, exact artifact references,
  whole-group ceiling drops, manifest v3 persisted per round; load-failure
  classification and fallback decisions; the native gate.
- **Wired** — `lib.rs` manages the service and its persisting observer;
  `commands::agent::runtime` hands it to the runtime's handlers; `drive_run`,
  the worker, `child_loop`, both OCR commands, the handoff, the swap and
  `prepare_model_for` take leases; the runtime's `RoundBoundary` is the stream
  function and context transform of every run.
- **Tested** — the table above; mutation-checked; Rust lib, runtime, UI and the
  related integration targets green apart from failures reproduced at HEAD.
- **Unverified or blocked** — the native gate (no GPU/server/models here); the
  full role chain (P10/P16); the installed app.
- **Remaining** — the gaps table above, each with its owner.

**Exact next step:** close P02 (receipt linkage at `subagents/worker.rs`
`event_seq: 0`; historical reads), then P04. On the target machine, run
`npm run test:p03:native` and commit `evidence/agent-system/P03/native-gate*.json`.
