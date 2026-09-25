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
| P02 | Shared memory correctness and authority | **Complete for the graph, receipts, versions, outbox, sharing and the sixth legacy store**; cutover of the other five legacy writers and graph-backed agent recall remain open — see below |
| P03 | Context compiler, GPU scheduling and continuation | Not started |
| P04 | Shared artifact, evidence and validation tools | **Complete for portable code, the production path and a real render on a Linux machine**; the render/acceptance gate on the Windows target stays open until LibreOffice is provisioned there — see below |
| P05 | Orchestrator and delegation tools | **Complete for portable code and the production path with a fixture coordinator**; the Spark-driven run and P03's parent-lease suspension are open gates — see below |
| P06 | Document & Vision Analyst with Unlimited-OCR | **Complete for portable code and the production path with deterministic OCR transport**; every real-model read and the vision probe are the open target gate — see below |
| P07 | Knowledge Retriever and local retrieval tools | **Complete for portable code and the production path, keyword retrieval measured and semantic retrieval through deterministic transport**; qualifying and measuring a real embedding model is the open target gate — see below |
| P08–P16 | — | Not started |

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

# P02 — Shared memory, provenance and graph authority

Worked on 2026-09-24, on `main` at HEAD `b9ff7fb`, uncommitted, on top of P01.
Raw output: `evidence/agent-system/P02/log_checks.txt`.

## Found before changing anything

| # | Finding | Where |
|---|---|---|
| 1 | P00 finding 2, and wider: the four mechanical worker routines wrote **no tool event at all**; the model path pinned every claim to the *first* tool the child called, on the *parent's* run, at `event_seq: 0`. | `subagents/worker.rs` |
| 2 | `admit()` accepted any positive `event_seq` with a tool and run id beside it — a number anybody can write. | `knowledge/graph/runtime_memory.rs` |
| 3 | P00 finding 1: the context compiler read the **latest** rows with a plain `snapshot` and labelled them with a cursor frozen earlier. | `agent_runtime/context_compiler.rs` |
| 4 | Items were upserted in place with no version history; supersede, tombstone and source invalidation changed `status`/`body` **without moving the revision**, so an optimistic writer could not see the change. | `runtime_store.rs` |
| 5 | `correct()` was two transactions; a crash between them left a correction and the fact it corrected both current. | `runtime_store.rs` |
| 6 | The outbox table existed and **nothing ever drained it**. | `runtime_store.rs` |
| 7 | `migrate_all` has **no production caller**, and the store `memory_api` answers from (`<app data>/memory/*.json`) was **not one of its five sources** — the sixth store its own docs predicted. | `migration.rs`, `agent_runtime/memory.rs` |
| 8 | P00 finding 8: `shared_with_task` was carried by P01 and enforced nowhere; the registry import hard-coded `false`. | `agents/store.rs` |
| 9 | `child_loop` sent `"definitionVersion": 0` to the runtime — a plausible number for one nobody had. | `subagents/child_loop.rs` |

## Implemented

| Area | Change |
|---|---|
| Receipts resolved in storage | `knowledge/graph/receipts.rs` (new): `ReceiptLedger::verify` reads the event back and requires it to exist, be `tool_succeeded`, name the same tool (either spelling) and record the same output hash. `MemoryGraph` asks before admitting; with no ledger in reach **nothing** is admitted by receipt. `Provenance::ToolReceipt` carries `output_sha256`. |
| One receipt per finding | Each mechanical action records its own `tool_succeeded` event on the **child's** run (`record_tool_receipt`, idempotent per action). On the model path `remember_outcome` now returns the event it wrote, and `ToolCallRecord` carries `eventSeq`, `outputSha256`, `toolCallId` and, for retrievals, the chunk ids that call returned — so each passage is tied to the one search that returned it. `Claim.receipt` is per claim; `Work.tool` is gone. |
| Source / measured / inferred / supplied | `Basis` computed by the store from provenance and the tool's contract (`SourceText` for evidence tools, `Measured` for others, `Inferred`, `Supplied`, `Carried`); never chosen by the writer. |
| Versions and historical reads | `agent_memory_versions` (immutable, backfilled once), one `write_revision` path every change goes through (row + version + dependency index + feed entry, one transaction), `snapshot_as_of(cursor)` and `versions_of`. A cursor ahead of the head is refused. |
| Compiler cursor (finding 1) | Reads `snapshot_as_of(frozen cursor)` or the atomic `snapshot_at`, and records the cursor actually read. |
| Correction, staleness, revocation | `correct`/`correct_at` in one transaction with an optional expected revision; `Stale` status; staleness propagated transitively through pinned dependencies and `derivedFrom`/`cites` edges on correction, tombstone, source invalidation, migration re-sync and rollback; `revoke_reader` (per-reader, propagated to derived items, new revisions so the feed drops them). |
| Dependency revalidation | `MemoryItem.depends_on` pins; at commit each pin is re-read — moved, withdrawn or missing inputs publish the result **stale** with the reason; the calculation checker pins the shared items its expressions came from. |
| Derived restrictions | At commit a derived item takes the intersection of cleared roles, the input's project and owner (two different ones are refused), a non-`Internal` classification, and every revoked reader. |
| Per-record authority | `Authority::{Graph, Legacy{store}}`; migrated copies are `Legacy`, and a graph write or correction to one is refused until its store is cut over. |
| Outbox | `deliver_pending` + `record_delivery_failure` (attempts, last error); `TaskEventLog` is the `events` consumer, writing a new `memory_published` event whose id is derived from the outbox key — a redelivery is refused as a duplicate. Delivered at start-up and after every commit (`lib.rs`). Each worker publication commits its outbox row in the same transaction as the item. |
| Sharing | `MemoryScope::Scratch{task, agent}`; a worker publishes to the task only when its pinned definition shares (`packet.shared_with_task`), otherwise to its own scratch, which it reads and its siblings do not. Import now records `sharedWithTask: true`. |
| Migration | `LegacySource::RuntimeScopedMemory` (sixth store) with `durable_items_on_disk`; `upsert_migrated` (Inserted / Unchanged / Updated / Restored); orphan retirement; `rollback_source`; `verify_runtime_scoped_memory`; a run ledger (`agent_memory_migration_runs`); the sixth store is mirrored at every start-up. |
| Service | Tauri commands `memory_graph_neighbours` (≤ 200, inside the authorised set), `memory_graph_history`, `memory_graph_correct` (operator provenance from the session, at the revision read) and `memory_graph_revoke_reader` (administrators). IPC 172 → 176, all four `frontend-pending`. |

## Migration and rollback notes

- **Schema changes are additive only.** Three new tables, three new columns; no
  row is dropped or rewritten. A database from before P02 opens, gains the tables,
  and has each existing item's *current* body backfilled as its one known version
  — earlier revisions were never stored and are not invented.
- **Restartable.** Every migrated record is addressed by
  `migrated_item_id(store, legacy_id)`; a pass interrupted part-way is simply run
  again. The run ledger shows a pass with no `finished_at`.
- **Reversible without deletion.** `rollback_source(source)` tombstones each record
  migrated from that source as a new revision and marks what was derived from them
  stale. The legacy store was never written by the migration and remains the
  authority; running the migration again restores each record (`Restored`).
- **Verified before any cutover.** `verify_runtime_scoped_memory` reports records
  missing, differing, orphaned or rolled back; `clean()` is the precondition for
  retiring that legacy writer. **No legacy writer has been retired**: every
  migrated record is `Authority::Legacy`, and the graph refuses to write it.
- **A legacy change is followed, not overwritten**: a changed value is a new
  revision (`Updated`), a vanished record is tombstoned (`retired`) — only when
  every legacy file was read, since an unreadable file is not evidence of absence.
- **Existing registry rows keep `sharedWithTask: false`** where the old import
  wrote it. That value now takes effect: those agents publish to their own scratch
  and their results say so. An administrator turns sharing on per agent on the
  Agents screen; nothing flips it automatically, because a stored `false` may be a
  person's choice.

## Tests

| Property named by the plan | Test |
|---|---|
| A publishes a real observation; B reads its version | `subagents::worker_tests::a_publishes_real_observations_each_on_its_own_receipt_and_b_reads_that_version` (production delegation path; two findings, two distinct verified receipts on the child run; B reads the version at the revision it landed) and `…::a_retrieval_is_published_as_source_text_on_its_own_search` |
| conflicting updates do not overwrite silently | `knowledge::graph::p02_tests::a_write_against_a_revision_that_moved_is_a_named_conflict` |
| correction invalidates an artifact | `…::a_correction_makes_everything_derived_from_the_corrected_fact_stale` (transitive, versions kept, feed told), `…::a_result_whose_input_moved_during_the_work_is_published_stale` |
| denied users see no hidden data or metadata | `…::a_revoked_reader_sees_nothing_of_the_item_on_any_path` (snapshot, historical read, history, neighbours, count, feed), `…::a_derived_record_inherits_the_restrictions_of_its_inputs` |
| failed outbox delivery recovers once after restart | `…::a_failed_delivery_is_recovered_exactly_once_after_a_restart` (file-backed; reopened; redelivery absorbed), `subagents::worker_tests::a_publication_reaches_the_parent_run_through_the_outbox` |
| repeat migration does not duplicate rows | `…::the_migration_is_repeatable_verifiable_and_reversible`, `…::a_changed_or_removed_legacy_record_is_found_and_followed_by_the_next_pass` |
| invalid receipts remain proposals | `subagents::worker_tests::a_receipt_the_event_log_does_not_back_stays_a_proposal` (missing event, failed call, wrong tool), `…::only_a_receipt_resolved_in_the_event_log_admits` (tampered hash, no log), `knowledge::graph::receipts::tests::*` |
| cursor honesty (P00 finding 1) | `agent_runtime::context_compiler::tests::a_context_frozen_at_a_revision_reads_that_revision_and_not_the_latest`, `…::a_read_as_of_a_cursor_is_that_cursor_and_not_the_latest_rows` |
| scratch versus task | `subagents::worker_tests::a_definition_that_does_not_share_publishes_privately` |
| older records | `knowledge::graph::runtime_memory::tests::an_item_written_before_the_p02_fields_still_reads` |

Two existing tests encoded the defects and were changed to prove the fix
instead: the compiler test that froze at 412 on a graph whose head was 1, and
`production_acceptance` step 2, which asserted that a receipt naming events 12
and 19 — never written — was "corroborated". Its journey now records real
receipts and resolves them.

**Seen failing.** With the receipt refusal turned into a pass and staleness
propagation disabled, `a_receipt_the_event_log_does_not_back_stays_a_proposal`,
`only_a_receipt_resolved_in_the_event_log_admits` and
`a_correction_makes_everything_derived_from_the_corrected_fact_stale` all failed;
restored, all pass.

## Checks run

| Check | Result |
|---|---|
| `cargo test --lib --no-fail-fast` (final) | **2735 passed, 0 failed**, 3 ignored |
| `cargo test --test production_acceptance --test production_hardening` | 12 passed · 8 passed |
| `npm run check:targets` | pass |
| `node scripts/check-ipc.mjs` | pass — 176 commands (20 frontend-pending) |

Not re-run in P02 because nothing they cover changed: the runtime TypeScript
suite, the UI suite, `check-reachable`, `check-egress`, `check-no-lora`, the
baseline harness. P02 changed no TypeScript and no network path.

## Unverified, open or deliberately left

| What | State | Owner |
|---|---|---|
| Cutover of the other five legacy sources | `migrate_all` still has no production caller; their writers are live and authoritative. Only the sixth store is mirrored at start-up. | P14 |
| Agent recall through the graph | `memory.recall_authorized` still answers from the legacy store (now mirrored in the graph). Routing it through the graph is the cutover of that store. | P03/P14 |
| Coverage of partial results | Only `tool_succeeded` events establish anything, and each passage rests on the call that returned it; there is no finer per-claim coverage model inside one successful call. | P06/P07 |
| Retrieval caches on source revocation | The graph and the compiler read current authorisation every time; the run's passage table (`RunPassages`) is not invalidated when a source is withdrawn mid-run. | P07 |
| UI for correct / history / neighbours / revoke | Commands exist and are gated; no screen calls them. The reconnect path (`reset` → fresh atomic snapshot) was read, not changed, and has no new test. | P14 |
| Commit-to-visible latency | Not measured. | P14 |
| The installed app | Not rebuilt or redeployed. | — |

## P02 close-out

- **Implemented** — verified receipts per finding; basis; immutable versions and
  historical reads; one-transaction corrections; staleness, dependency
  revalidation, derived restrictions and per-reader revocation; per-record
  authority; a working outbox; scratch versus task sharing; a restartable,
  verifiable, reversible migration including the sixth store; four bounded service
  commands.
- **Wired** — `lib.rs` gives the graph the event log as its ledger, drains the
  outbox at start and after each commit, and mirrors the runtime's memory; the
  workers and the child loop publish through `TaskMemory` with real receipts; the
  compiler reads at its cursor.
- **Tested** — the table above; mutation-checked; full lib and the two changed
  integration suites green.
- **Remaining** — the five-source cutover and graph-backed agent recall (see
  above).

**Exact next step:** P03 — begin at `agent_runtime/mod.rs` `context.refresh`
(the `FrozenScope` built near line 814), which now reads at its frozen cursor;
the served window (P00-OBS-2, `window: 0`) and the 9 039-token tool-schema floor
(P01-OBS-1) are what the context budget has to be built against.

---

# P04 — Shared artifact, evidence and validation tools

Worked on 2026-09-24 on branch `claude/hopeful-brahmagupta-1eol9d` from HEAD
`20061fc`, in a **Linux cloud container** — not the Windows host P00–P02 ran on,
and not the target machine. P03 is not started; P04 needs only P01 and P02.
Raw output: `evidence/agent-system/P04/` (index in its `README.md`).

## Found before, or while, building it

| # | Finding | Where | Now |
|---|---|---|---|
| 1 | The canonical names were `artifact.list`, `artifact.read`, `artifact.create_{approval_note,calculation_workbook,briefing_deck,chart,diagram,pdf,table}` and the DOCX-named `artifact.verify_docx` (alias `validate_artifact`), which reopened *every* kind the run produced while its name and description promised a Word check and implied readiness. | `orchestrator/tools.rs`, `agent_runtime/mod.rs` `validate` | Kept, alias and all. Its description and its answer now say it is a reopen check and that `artifact.validate` is acceptance. |
| 2 | **Every composed document and workbook was refused at the gateway**: `sections` and `sheets` were declared `Object`, the handlers deserialise lists. Found by driving `artifact.create_approval_note` with `sections` through `authorize` → `execute`; the direct-helper tests had always passed. | `orchestrator/tools.rs` | Declared `List`. The TypeScript catalogue still does not *offer* `sections`/`sheets` to a model — see "Remaining". |
| 3 | **Cross-tenant id collision**: the default artifact id was derived from the content hash alone, so a second account recording identical bytes hit the primary key — and the error told it somebody held those bytes. | `artifacts/conversation_store.rs` `record` | Derived from owner + content. |
| 4 | The classification label on `artifact.create_pdf`/`create_table` (and the composed Word/Excel paths) came from the model's arguments — contract map §7, owner P04. | `agent_runtime/{mod,artifacts}.rs` | Derived from the classifications of the passages the run retrieved; the argument is accepted for old callers and ignored. |
| 5 | Nothing checked a stored blob against its recorded hash: "immutable" was assumed. | `conversation_store.rs` `read` | Every read re-hashes; an altered blob is refused, not served. |
| 6 | The built-in PDF reader returned one text blob, so no page could be addressed. | `artifacts/pdf_validate.rs` | `page_texts`, one per page. |
| 7 | The run's memory graph (the one `context.refresh` and now the artifact links use) was opened **without** the event log as its receipt ledger, so nothing linked on a receipt could ever be admitted in the app — only in the subagents' graph. | `commands/agent.rs` | `with_receipts(state.events)`, as `lib.rs` does for the subagents. |
| 8 | `render_pages.py`'s first blank-page test used `len(pixmap.color_count())`; PyMuPDF returns an integer there, so the fallback reported every page "not blank". Caught by its own sidecar test before it shipped. | `sidecars/document_sidecar/render_pages.py` | `is_unicolor`, with a fallback that handles both return shapes. |

## Implemented

| Area | What | Files |
|---|---|---|
| Format from bytes, safe to open | `%PDF-`, OOXML main content type, `<svg` root, UTF-8 text, legacy CFB; wrong-type = claimed ≠ detected. ZIP limits: archive size, entry count, per-entry and total decompressed size, compression ratio; zip-slip names (absolute, `..`, backslash, drive, NUL) and duplicates refuse the whole package. Macros, external relationships that would be fetched (image, OLE, template, frame, external link — a hyperlink is listed, not refused) and remote Word fields (`INCLUDEPICTURE`, `INCLUDETEXT`, `LINK`, `DDE`, `DDEAUTO`, `IMPORT`) are found and block rendering. | `artifacts/package.rs` (new) |
| XML | A pull scanner with byte spans; refuses `<!DOCTYPE` (entity expansion) and any malformed tag. No new crate. | `artifacts/xml_events.rs` (new) |
| **One content/evidence contract** | `ContentModel` of located units — `p:12`, `slide:3/title`, `slide:3/body:2`, `slide:3/notes`, `sheet:Readings!B4` (with formula), `page:2`, `line:40` — each with the citations in it: `[E3]` (run evidence), `[A:art@v]`, `[M:item@rev]`, `[S:sha@loc]`. One region grammar, bounded rendering with named omissions, and a unit-level diff. Word, PowerPoint, Excel, PDF and text all read into it. | `artifacts/content.rs` (new) |
| Templates | Every template and composition structure with id, version, required/optional fields and the SHA-256 of its canonical definition; `used_by(tool, args)`. | `artifacts/templates.rs` (new) |
| **The ladder** | `fileCreated` (bytes exist and hash to the record) → `formatReopened` (from the bytes; container safe; parses; no macros) → `contentChecked` (says something; no placeholder; template's promises kept; every citation bound; no formula error) → `renderChecked` (real layout engine; no blank page; deck pages = slides) → `accepted` (all of those, and every dependency current). Each rung is `passed`, `failed`, **`unavailable`**, `notApplicable` or `notRun`, with the validator and version that decided it. Accepted is never a person's approval. | `artifacts/validation.rs` (new) |
| **Renderers** | Two pinned local adapters. **LibreOffice** (headless, per-render throwaway profile with macro execution disabled and security at its highest, staged copy of the stored bytes, killed at 120 s) lays Office files and SVG out to PDF. **PyMuPDF** via the bundled `render_pages.py` rasterises PDF pages to PNG with text and a blank flag. Qualified series recorded; any other version is *unavailable*, with the version found. Registered in `deployment::DEPENDENCIES` (`office-renderer` external, `page-rasteriser` bundled), so the deployment gate covers them. | `artifacts/render.rs` (new), `sidecars/document_sidecar/render_pages.py` (new), `deployment/mod.rs` |
| **Registry** | Additive schema on `conversation_artifacts`: stage (`recorded`/`candidate`/`final`, forward only), classification, template id/version/hash, project, published-at, graph node; new tables for dependencies (per version, with the marker that names each), effect keys (owner-scoped, first registration stands), validations and renders (each tied to the SHA-256 checked). `final` is reachable only through `promote` against an accepted validation of the same bytes. | `artifacts/conversation_store.rs` |
| **Targeted edits** | Every ZIP entry but the edited part is copied raw; inside it the one text node is spliced by byte range. Refused before anything is written: not found, ambiguous, text split across runs, formula cell, text box, notes, PDF — or a read-back diff showing any unit changed beyond the ones named. | `artifacts/edit.rs` (new) |
| **Ten tools**, end to end | `artifact.manifest`, `read_version`, `read_region`, `list_templates`, `validate`, `render`, `diff`, `resolve_evidence`, `register_version`, `edit`: `ToolName` + spec + contract (route, prerequisites `ConversationArtifacts`/`PageRenderer`, output) + `class_of`/`is_side_effecting` + runner refusal + agent-path arms + TS catalogue (closed schemas, six-clause descriptions) + `tool-names.ts` + conformance lists + UI labels; `tool-contract.json` regenerated. `validate`/`render` run on the blocking pool. | `orchestrator/{tools,contract,runner,grammar}.rs`, `agent_runtime/{mod,artifact_tools,tool_policy,planning}.rs`, `events/idempotency.rs`, `agent-runtime/src/*`, `src/services/toolNames.ts` |
| Production registration | A file a create tool writes is registered as a **candidate** with its template, the run's classification, and every citation bound to what the *run* retrieved/recalled/read — passages by document hash and page, memory items at the revision visible now, artifact versions — plus the calculations behind a calculation workbook; under the call's effect key. | `agent_runtime/{mod,artifact_tools}.rs` |
| **Receipts and graph links** | After the call's own `tool_succeeded` event is written, the version becomes an `ArtifactRef` node on that receipt (admitted when the ledger resolves it), `depends_on` its memory dependencies, `DerivedFrom` them and its base version, `Cites` artifact dependencies. A correction to a cited fact therefore reaches the artifact through P02's own staleness propagation. | `agent_runtime/artifact_tools.rs` `link_to_graph` |
| Recheck before publishing | `register_version stage=final` rechecks every dependency *now* — document still current and cleared for this reader, attachment readable, memory item unmoved and standing, artifact bytes unchanged, calculation re-evaluates the same, template definition unchanged, graph node not stale — then requires an accepted validation of these bytes, then **a person's approval** (it is `PersonBeforeEffect`). | `artifact_tools.rs` `recheck`, `register_version` |

No IPC command was added: these are model-facing tools on the existing
runtime RPC. `check:ipc` stays at 176.

## Contract decisions

1. **The format is what the bytes say.** Extension and media type are *claims*,
   compared with the bytes; disagreement fails the reopen rung.
2. **Unavailable is a state.** A missing or unqualified renderer produces
   `renderChecked: unavailable` and `accepted: unavailable` — never a skipped rung
   that reads as green, and never a publication.
3. **Accepted is the machine's last word, not a person's.** Publishing needs an
   accepted validation *and* a person; the DRAFT marking stays in the bytes.
4. **One key, one effect, first one stands.** A retry that re-rendered different
   bytes under the same effect key returns the first registration with
   `content_differed`, and publishes nothing. The per-run effect ledger
   (`events::idempotency`) absorbs a same-run replay before the registry is
   asked; the registry key is what holds across runs and restarts.
5. **A citation means what it meant when written.** `[E3]` is bound to a passage
   by the run that wrote it, at registration, and an edit carries that binding
   rather than re-guessing it in a run whose evidence table is different.
6. **Nothing takes a path.** Every tool reads the owner-scoped store, checks the
   run's conversation, and answers "absent" and "somebody else's" with the same
   words.
7. **Last in the plan, first dropped.** The family is permitted only for work on
   a deliverable (`planning::derive`: deliverable words or review/edit/publish
   words) and listed last, so a small window drops these before any producer,
   the sandbox or retrieval.

## Tests

| Property the plan names | Test |
|---|---|
| exact bytes survive another run, another model and a restart | `agent_runtime::artifact_tools_tests::a_produced_note_survives_another_run_another_model_and_a_restart` (bytes on disk = stored hash; read by a later run under another model; reopened store returns the same bytes and bindings), `artifacts::conversation_store::tests::a_blob_altered_on_disk_is_refused_rather_than_served` |
| scoped reads refuse other users' artifacts | `…::another_person_cannot_reach_an_artifact_through_any_of_the_tools` — nine tools, in Bob's conversation and in Alice's conversation id, all answered with the not-found wording and no content |
| corrupted and wrong-type files fail | `…::a_corrupted_or_mistyped_file_fails_validation_and_is_never_published`, `artifacts::validation::tests::{a_pdf_named_docx_is_a_wrong_type_failure, a_corrupted_package_fails_to_reopen_and_nothing_above_it_runs}`, `artifacts::package::tests::*` (zip-slip, bomb, macros, external fetches, remote fields), `artifacts::xml_events::tests::a_doctype_is_refused_rather_than_expanded` |
| changed evidence marks output stale | `…::a_correction_to_what_a_note_cites_makes_the_note_stale_and_unpublishable` — the graph node goes `stale` by P02's propagation; manifest, resolve_evidence and validate say why; publication is refused naming it |
| targeted edits preserve unrelated content | `…::an_edit_through_the_tool_changes_one_unit_and_copies_every_other_part` (raw ZIP entries byte-identical, diff names one change, binding and template carried, a superseded base refused), `artifacts::edit::tests::*` (including a part this product never writes surviving) |
| the same effect key does not publish twice | `…::the_same_effect_key_publishes_once_across_runs` (different bytes, same key, two runs: one version, one graph node), `artifacts::conversation_store::tests::a_repeated_effect_key_does_not_publish_a_duplicate_even_after_a_restart` |
| a real render, acceptance and publication | `…::a_sound_note_is_rendered_accepted_and_published_or_reported_unavailable` — here: LibreOffice 24.2.7.2 + PyMuPDF 1.28.2 render, accept, publish (with approval), a same-run replay and a cross-run repeat publish nothing; with LibreOffice hidden: unavailable and refused (`log_render_unavailable.txt`) |
| the model-facing path | `tests/agent_runtime.rs::a_model_reads_an_artifact_manifest_through_the_real_runtime` — the real Node bundle offers `artifact.manifest`, a fixture model calls it, and the manifest (with the artifact's hash) reaches the model's next request |
| template conformance on the offered form | `…::an_approval_note_from_the_template_is_checked_against_its_template` |
| plan gating | `…::the_artifact_family_is_offered_for_deliverable_work_only_and_last` |

**Seen failing.** With the publish-time recheck disabled, the raw copy of
untouched parts disabled (one foreign part dropped), and the effect key removed,
`a_correction_to_what_a_note_cites…`, `a_targeted_word_edit_changes_one_node…`,
`a_repeated_effect_key…` and `the_same_effect_key_publishes_once…` all failed;
restored, all pass (`log_mutation.txt`). The first mutation of the effect key
(only the lookup) did **not** fail the tool-level test: the effects table's
primary key refused the duplicate insert and rolled the version back — a second
line of defence, recorded rather than removed. The tool-level test was also
tightened so the retry renders different bytes; before that, content addressing
alone would have hidden a broken key.

## Measured

**P04-OBS-1 — the whole catalogue no longer fits an 8k window.** 43 tools; at
the smallest compression stage the catalogue is **3,902** estimated tokens
against an 8k tool budget of about **3,690** (`tool-budget.ts`'s estimator). The
ten P04 tools cost about 1,090 of that. No plan offers the whole catalogue: the
family is plan-gated and listed last, and `tool-budget.test.ts` now pins that
when the whole catalogue meets an 8k window, only P04 tools are dropped, from the
tail, and reported. P00-OBS-1 and P01-OBS-1 stand; role-scoped loading (P03) is
the remedy, not this.

**P04-OBS-2 — a real render of this product's own files.** A produced two-slide
deck lays out to exactly two non-blank pages; a produced note to one page with
745 characters of text; LibreOffice 24.2.7.2 + PyMuPDF 1.28.2 on this Linux
machine. No timing was recorded as a benchmark.

## Checks run

| Check | Result |
|---|---|
| `cargo test --lib --no-fail-fast` | **2799 passed, 2 failed**, 3 ignored. Both failures (`agent_runtime::tests::a_relative_path_that_climbs_out_is_still_refused_after_anchoring`, `skills::containment::tests::an_absolute_path_is_refused_before_it_is_joined`) assert Windows path semantics (`..\..\`, `C:`) and **fail identically at HEAD `20061fc`** in a clean worktree on this machine — not P04. |
| focused P04 suites (`log_focused_rust.txt`) | 385 passed, 1 ignored |
| `npm run test:integration` | 64 passed, 1 ignored (11 targets) |
| `npm run runtime:typecheck`, `npm run runtime:test` | pass; 131 files, **2332** tests |
| `npx tsc --noEmit -p tsconfig.json`, `npm run test:ui` | pass; 43 files, 618 tests |
| `npm run build`, `check:bundle`, `check:bundle:self`, `check:offline` | pass |
| `check-ipc`, `check-reachable`, `check-egress`, `check-no-lora`, `check-deployment` | pass — 176 commands, 196 modules, one chokepoint, 613 files, both renderers covered |
| `python -m unittest …/test_render_pages.py` | 4 passed |
| `npm run test:sidecar:document` | 116 ran, **13 errors**: `pypdf` is not installed in this container (the four new tests pass inside the run). Environment, not code. |
| `node scripts/fixture-manifest.mjs --check`, baseline `fixture-01/02` | **fail on this Linux checkout, and would at HEAD**: the manifest was generated on Windows, where these text files check out with CRLF; their committed hashes match the files only with CRLF line endings (verified). Not regenerated here — doing so would break the check on the Windows host. |
| `node scripts/agent-baseline.mjs` | 39 cases: 8 executed, 6 passed, 2 failed (the two above), 31 blocked. `art-evidence-03-artifact-hash-recorded` now names **P09**: P04 built and tested the hash the case grades; the agent whose result carries a produced file is the Document Author. |

## Unverified, open or deliberately left

| What | State | Owner / command |
|---|---|---|
| **Render and acceptance on the Windows target** | P00's inventory found no `soffice` there. Until LibreOffice is provisioned, every Office deliverable validates to `renderChecked: unavailable`, `accepted: unavailable`, and cannot be published. That is the honest reading, not a defect. | Install LibreOffice 24.2 or 24.8 (MPL-2.0) from the offline pack; set `ARJUN_SOFFICE=C:\Program Files\LibreOffice\program\soffice.com`; then `cargo test --manifest-path src-tauri/Cargo.toml --lib -- artifacts::render agent_runtime::artifact_tools_tests --nocapture` |
| PyMuPDF on the target | 1.28.0 per P00's inventory — inside the qualified series; not exercised by this phase there. | the same command |
| Composed documents/workbooks offered to a model | The gateway accepts `sections`/`sheets` now; the TypeScript catalogue does not offer them, so a model can only reach the approval-note template and the calculation workbook. | P09 / P13 |
| OS-level network isolation of LibreOffice | Not enforced. The guarantee is refusal before start (no fetchable reference, no remote field, no macro) plus the profile's macro settings. | P16 observed-egress checks |
| Render cache retention | Page images accumulate under `<app data>/artifacts/renders/`; no retention policy. | P14 |
| Editing speaker notes, text boxes, formula cells, text across runs | Refused, by design, before anything is written. | P12 / P13 if needed |
| Recalculation | A workbook's formula values are the cached ones; nothing here recalculates. | P13 |
| A foreign PDF's page text | The built-in reader parses this product's own PDFs; another writer's is reopened only by the rasteriser during render, and its reopen rung reports `unavailable` without it. | — |
| The installed app | Not rebuilt or redeployed. | — |

## P04 close-out

- **Implemented** — safe format detection and package limits; one content and
  evidence contract for Word, PowerPoint, Excel, PDF and text; versioned, hashed
  templates; a five-rung validation ladder with an honest `unavailable`; two
  pinned local render adapters; candidate/final registration with effect keys,
  per-version dependency bindings, validations and renders; byte-preserving
  targeted edits; ten tools; classification derived from the run; graph links on
  real receipts; a publish-time recheck.
- **Wired** — Rust catalogue, contract, policy, idempotency, plan, runner and
  agent-path dispatcher; the TypeScript catalogue, names, conformance and budget;
  UI labels; the deployment table; the run graph's receipt ledger in
  `commands/agent.rs`.
- **Tested** — the table above, on the production `authorize` → `execute` path
  and across the real Node process boundary; a real render here; mutation-checked.
- **Unverified or blocked** — the render/acceptance gate on the target machine;
  composed authoring forms offered to a model.
- **Remaining** — nothing else in P04's scope.

**Exact next step:** P03 — context assembly, model scheduling and durable
handoff (still not started; P05 and P06 need it). Begin at `agent_runtime/mod.rs`
`context.refresh` and the 8k tool-budget pressure recorded as P04-OBS-1: role-
scoped tool loading is what lets the artifact family, the producers and the
sandbox share a small window.

---

# P05 — The Spark orchestrator and delegation tools

Worked on 2026-09-24 on branch `claude/hopeful-brahmagupta-1eol9d` from HEAD
`fe3ff44` (P04), in the same **Linux cloud container** — not the Windows host and
not the target machine. **P03 is not started**, and §11.3 lists it as a P05
prerequisite: P05 was implemented ahead of it at the user's request, and the one
P05 requirement that needs P03 (parent-lease suspension) is recorded open rather
than imitated. Raw output: `evidence/agent-system/P05/` (index in its
`README.md`).

## Found before, or while, building it

| # | Finding | Where | Now |
|---|---|---|---|
| 1 | `agent.delegate_readonly` holds the model's loop for the whole child: nothing could ask how a child was doing or stop it, and a stopped run's child ran on until its own deadline — the manager's timeout was its only stop. | `orchestrator/runner.rs`, `subagents/manager.rs` | Jobs run in the background (`agent.delegate` + `agent.status`/`agent.cancel`); the manager races every child against its token, its id and its run in the stop table, and the run's durable ending, with one step's grace. |
| 2 | A retry of a failed child under the same idempotency key is answered "already attempted and did not succeed, so it was not started again", and the key depends only on run, role, objective and inputs — so no step could have a repair budget. | `manager.rs` `spawn`, `recall` | `Dispatch::scoped_to`: a job's key includes its plan step and attempt. The unscoped key every other caller computes is unchanged. |
| 3 | A child's id was minted inside `spawn`, so nobody could name — or stop — a child while it ran. | `manager.rs` | `Dispatch::as_child`. |
| 4 | `ChildResult::describe` calls a *completed* child's findings "partial" whenever it carries a detail note, and a completed retrieval always does (where it published). | `subagents/result.rs` | Left as is (other callers); job summaries are built from the counts. |
| 5 | The calculation checker's findings cite no passage; its grounding is its publications, admitted on the engine's receipts. `ChildResult.receipts` stays empty for the mechanical workers. | `subagents/worker.rs` | A step counts admitted publications as receipt-backed grounding. `completion::children.contract_honoured` still calls such a child uncited — recorded, not changed here. |
| 6 | The writer escalation first asked a person about a request the handler would refuse (a role asking for a tool it is not given), and the run waited for an approval nobody should give. Found as a hanging test. | `delegation::approval_needed` | A refusable request is not escalated; the handler refuses it by name. |
| 7 | The retriever's mechanical fallback ANDs every query term: a natural-language objective finds nothing, a short phrase finds the passage. | `worker.rs` `retrieve` | Recorded; P07's to improve. The tests use short phrases, as the worker tests do. |
| 8 | The artifact reviewer's fallback resolves an artifact id as a workspace path, so it cannot open a conversation artifact: an independent review of a real deliverable does not pass in this build. | `worker.rs` `review` | Recorded and pinned by a test; the Deliverable Reviewer is P10. |
| 9 | Every tool's timeout is capped at 120 s (`every_tool_has_a_time_limit`). | `orchestrator/tools.rs` | Waits capped at 100 s; a job that outlives the wait keeps running and is reported running, never abandoned. |

## Implemented

| Area | What | Files |
|---|---|---|
| **One coordinator, in the main run** | The coordinating model is the model of the chat run itself (Spark X2.5 4B Q8 in the target deployment). It plans, delegates, watches, stops and gates with five ordinary tool calls through the same gateway, plan budget, approval queue and event log as every other call. No second orchestrator; no profile holds these tools, so a child cannot delegate. | `agent_runtime/delegation.rs` (new) |
| **Typed durable plan** | Goal, constraints, operator corrections, typed steps (`direct` / `delegate` / `review`) with dependencies (acyclic), acceptance criteria, exact input references, status, outputs, receipts and a repair budget; decisions with the evidence they rest on; open questions. Patches are validated against the version the model read; strict per-operation fields. The model cannot write a status or a receipt; a completed step cannot be edited, skipped or removed, and reopens only after a later correction, keeping its receipts. | `agent_runtime/task_plan.rs` (new) |
| **Storage** | Migration `task_plans_and_jobs`: `task_plan_versions` (consecutive, append-only by trigger, body hashed and re-checked on read) and `delegated_jobs` (`queued → running → one ending`, the first ending stands). | `events/plans.rs` (new), `events/migrations.rs` |
| **Jobs** | `agent.delegate`: pending delegate step, prerequisites complete, required memory items present at their revision and usable, attempts left, fewer than 2 running for the run. Definition resolved from the registry at dispatch and recorded (id, version, origin); child's model chosen by `certification::choose` against the registry; inputs from the call and the step (`step:<id>` resolves to the producer's artifact outputs); graph-revision handoff from prerequisites' results; deadline the shorter of the call's (≤ 600 s) and the definition's; allowed tools a subset of the role's and of the run's. | `delegation.rs`, `subagents/manager.rs` |
| **Writer jobs under the existing approvals** | A job whose role writes, or whose `allowed_tools` names a write tool, is escalated to a person in the runtime's `decide` — same queue, standing approvals and events as a direct write — and the handler refuses a writer job whose call has no recorded approval. `Dispatch::writing()` has exactly this one model-facing caller; `agent.delegate_readonly` is unchanged. | `agent_runtime/mod.rs` `decide`, `delegation.rs` |
| **Settlement** | From the manager's status and the payload, never the child's words: `completed` needs a cited or receipt-backed finding or an artifact (else `partial`); artifacts must validate to `accepted` where asked; `blocked` / `refused` → blocked (a role with no worker is blocked by name before anything runs); `failed` / `timed_out` → failed; the same failure twice → blocked for no progress; attempts exhausted → blocked; a job a correction superseded → pending, uncharged. | `task_plan::job_settled`, `delegation::settle` |
| **Status, cancel, review** | `agent.status` (bounded wait, findings with citations, published ids, what was not done, the plan). `agent.cancel` (this run's jobs only). `task.request_review`: P04's ladder on every artifact output (or named version), currency of every published item, then an independent reviewer role that did not produce the work, 40 s bound; verdict recorded as a review receipt. | `delegation.rs` |
| **Stop, correction, crash** | A person's steer (`agent_steer_run`) is recorded on the plan, published to the task's memory as an operator `Correction`, and stops every running job; their steps go back to pending, uncharged. The run's cancellation and its durable ending stop its jobs. At start, `recover_interrupted_jobs` marks the last process's live jobs `interrupted` and settles their steps: a reader's failed and reopenable, a writer's blocked for a person. | `delegation.rs`, `commands/agent.rs`, `lib.rs` |
| **Shared memory and context** | After each orchestrator call, goal, constraints, the plan projection, decisions and open questions are committed to the task scope on that call's own receipt (admitted), so `context.refresh` carries them as mandatory blocks every round. Artifacts are referenced by version, never inlined. | `delegation::publish_plan`, `mod.rs` |
| **Capability discovery** | `capability.search` (the canonical tool) now also lists the delegable roles: resolved definition id/version/origin, whether a worker exists, whether it writes, output schema, model role, time limit. | `mod.rs` `capability_search`, `delegation::roles` |
| **Completion** | With a plan: `taskplan.steps_accepted`, `taskplan.no_job_running`, `taskplan.independent_review`. Without one (a simple question) they do not appear, and the orchestrator's tools are not offered (`planning::derive`). | `completion.rs`, `commands/agent.rs`, `planning.rs` |
| **Five tools, end to end** | `task.plan_update`, `agent.delegate`, `agent.status`, `agent.cancel`, `task.request_review`: `ToolName` + spec + contract + `class_of` + runner refusal + agent-path arms + TS catalogue (closed schemas, six-clause descriptions) + `tool-names.ts` + conformance + budget + UI labels; `tool-contract.json` regenerated. | `orchestrator/{tools,contract,runner}.rs`, `agent_runtime/{mod,tool_policy,planning}.rs`, `agent-runtime/src/*`, `src/services/toolNames.ts` |
| **Agents page** | "Jobs the coordinator delegated": each job with its step's status, the definition version it was pinned to, mode, status, what was not done, model and lease decision. New read-only command `agent_orchestrator_jobs`, owner-scoped (administrators see all). | `commands/agent_admin.rs`, `lib.rs`, `ipc-manifest.json`, `src/services/agentRegistry.service.ts`, `src/pages/Agents.{tsx,module.css}` |

## Contract decisions

1. **The model proposes; receipts decide.** A plan version written by the model
   can add and edit pending work; only a settled job, a claimed tool receipt
   (checked against the event log) or a review changes a step's status.
2. **A dependency waits by refusal, not by blocking.** Delegating a step whose
   prerequisites are not complete is refused with what it waits on, and costs no
   attempt; the coordinator waits with `agent.status`.
3. **An incomplete draft is never success.** A child that says "completed" with
   nothing cited is `partial`; an artifact must validate to `accepted` where the
   step asks; a replayed result from an earlier process is reported as such.
4. **Retries are bounded twice.** At most three attempts, and the same failure
   twice blocks the step even with attempts left. A correction's cancellation is
   not charged.
5. **Writer jobs are approved like writes.** Escalated per call in `decide`,
   verified again in the handler; the read-only alias never reaches it.
6. **Simple work stays simple.** The five tools are permitted only for
   deliverables, code or requests that ask for delegation; a one-search
   question is not offered them and has no plan criteria.
7. **Lease suspension is recorded, not simulated.** Each job records what
   happened to the card (`lease`); a child on another model is serialised by
   `subagents::scheduling` and the coordinator's model is not suspended until
   P03 exists.
8. **Context follows the receipts.** The plan reaches the coordinator's context
   through the graph at the version its last orchestrator call returned; a job
   that settles in the background reaches it at the next such call
   (`agent.status`), not spontaneously.

## Tests

| What the prompt names | Test |
|---|---|
| dependent two-specialist plan | `agent_runtime::delegation::tests::a_dependent_two_specialist_plan_runs_in_order_on_receipts` — the dependent step is refused first (no attempt used); the production retriever over a real index, then the production calculation checker waiting (`atLeast`) for the retriever's graph revision; both steps completed by receipt; goal, constraint and plan admitted in the task's memory; the completion verifier passes the plan criterion |
| one failed child | `…::a_failed_child_fails_its_step_then_blocks_it_and_the_run_is_reported_unfinished` — the production code worker (writer, approved) fails for want of a model runtime; retried once it fails the same way; the step is blocked for no progress with an attempt left, a third try is refused, and the verifier reports the run unfinished naming the step |
| denial of writer tools | `…::writer_jobs_need_a_person_and_never_come_through_the_read_only_alias` (the alias refuses a writing role; a rejected approval starts nothing; a writer call reaching the handler without a recorded approval is refused; `allowed_tools` cannot widen a read-only role), `…::a_writer_job_is_escalated_to_a_person_by_the_gateway_path` |
| mid-flight correction | `…::a_correction_mid_flight_stops_the_job_and_reopens_its_step_uncharged` (the running job is stopped, the step is pending with no attempt charged, the correction is in the plan and admitted in memory as a person's); `task_plan::tests::a_correction_reopens_a_completed_step_and_keeps_its_receipts` |
| duplicate delegation retry | `…::a_duplicate_delegation_starts_no_second_job` (a second call while running and after completion: one job, one child each) |
| cancel | `…::agent_cancel_stops_a_running_job_and_the_step_can_be_reopened`, `…::stopping_the_run_stops_its_running_jobs` |
| crash/resume | `…::a_restart_marks_the_running_job_interrupted_and_the_step_is_retried_fresh` (durable log on disk; the next process marks the job interrupted and fails the reader's step; reopened, attempt 2 runs under a fresh key and completes); `events::plans::tests::a_job_settles_once_and_a_restart_interrupts_only_live_ones`; `task_plan::tests::an_interrupted_writer_needs_a_person_and_an_interrupted_reader_can_be_retried` |
| incomplete capability status without fictitious success | `…::a_role_with_no_worker_is_reported_blocked_and_never_as_done` (capability search says "no worker in this build"; the job is blocked by name; no child started); `…::a_coordinators_own_deliverable_is_not_complete_until_an_independent_review_passes` (a real note, a real receipt, the P04 ladder, and a reviewer role that cannot yet read conversation artifacts: the review does not pass and the step is not complete) |
| review on published findings | `…::a_review_of_published_findings_rechecks_them_before_the_step_completes`, `…::a_review_step_settles_the_steps_it_reviewed` |
| inputs scoped like the P04 tools | `…::an_artifact_from_another_conversation_is_not_reviewed_or_read` (the same person's note from another conversation is refused as a review target and as a step input) |
| simple work spawns nothing | `…::a_simple_question_is_not_offered_the_orchestrator` |
| the plan's own rules | `task_plan::tests::*` (15: versions, cycles, strict ops, receipts only, claims checked against the log, partial, no progress, bounded attempts, corrections, interruption, one start per step, gate, round trip) |
| storage | `events::plans::tests::*` (3: consecutive versions and conflicts, append-only triggers, one ending per job) |
| **routed tool calls under the production driver** | `tests/agent_runtime.rs::a_coordinating_model_plans_delegates_and_reads_a_job_through_the_real_runtime` — a fixture coordinator's `task.plan_update` → `agent.delegate` → `agent.status` through the real Node bundle into Rust's authorize/execute; the production retriever; plan, job and child in the durable record (`log_production_driver.txt`). **Deterministic transport, not Spark.** |
| with Spark | `tests/agent_runtime.rs::spark_routes_the_orchestrator_tools_through_the_real_runtime` — `#[ignore]`d; **not run here** (no weights, no server). See "Unverified". |

**Seen failing.** With the handler's approval check, the citation rule, the
dependency check, the manager's stop race and the correction refund each
disabled in turn, the test guarding it failed; restored, it passes
(`log_mutation.txt`).

## Measured

**P05-OBS-1 — the catalogue at 48 tools.** 4,521 estimated tokens at the
smallest compression stage (5,089 at `schemaOnly`), against an 8k tool budget of
about 3,686. The five P05 tools sit before the P04 family, so when a plan
permits everything at 8k the fitter drops nine of the ten artifact tools and
keeps every orchestrator tool (`tool-budget.test.ts`). No plan permits
everything; P04-OBS-1's remedy (P03 role-scoped loading) stands.

## Checks run

| Check | Result |
|---|---|
| `cargo test --lib --no-fail-fast` | **2832 passed, 2 failed**, 3 ignored — the same two Windows-path tests that fail at `20061fc` (P04's record); `log_lib_full.txt` |
| focused suites (`log_focused_rust.txt`) | task_plan 15, delegation 15, events 127, completion 10, planning 25, tool_policy 6, orchestrator tools 28 / contract 5 / gateway 23, subagents 103; `agent_runtime::tests` 83 + the pre-existing Windows-path failure |
| `npm run test:integration` | **65 passed**, 2 ignored (the Spark test, and `seed_local_accounts` as before) |
| `npm run runtime:typecheck`, `npm run runtime:test` | pass; 131 files, **2352** tests |
| `npx tsc --noEmit`, `npm run test:ui` | pass; 43 files, 618 tests |
| `npm run build`, `runtime:build`, `check:bundle`, `check:bundle:self`, `check:offline` | pass |
| `check:ipc`, `check:reachable`, `check:egress`, `check:no-lora`, `check:deployment`, `check:targets`, `check:whitespace` | pass — **177** commands (+1), 196 modules |
| `node scripts/agent-baseline.mjs` | 39 cases: 8 executed, 6 passed, 2 failed (P04's CRLF fixture hashes), 31 blocked — no case is P05's |
| Agents page | rendered in Chromium from a throwaway entry with Tauri's `invoke` stubbed by job records the backend produced in the tests (`agents-page-jobs-panel.png`). The UI suite has no DOM environment by design; gstack (`/design-review`, `/qa`) is not installed here and was not installed. |

## Unverified, open or deliberately left

| What | State | Owner / command |
|---|---|---|
| **Spark X2.5 4B Q8 driving these tools** | Not run: no weights and no `llama-server` in this container. The production path is proven with a fixture coordinator only; nothing here says how Spark plans or whether it calls the tools well. | On the target: `llama-server -m <Spark-X2.5-4B-Q8_0.gguf> --port 8080 -c 16384 --jinja`, then `ARJUN_SPARK_URL=http://127.0.0.1:8080/v1 ARJUN_SPARK_MODEL_ID=orchestrator.spark-x2-5-4b cargo test --manifest-path src-tauri/Cargo.toml --test agent_runtime spark_ -- --ignored --nocapture` |
| **Parent-lease suspension for model-switched children** | P03 is not built. A child routed to another model is serialised on the card by `subagents::scheduling`; the coordinator's model stays loaded, and each job's `lease` says so. On an 8 GB card that is the spill P03 exists to prevent. | P03 |
| Independent review of a real deliverable | The artifact-reviewer fallback cannot open a conversation artifact, so a review of a produced note does not pass here (pinned by a test). | P10 |
| Background settlement and context | A job that settles between the coordinator's calls reaches its compiled context at its next orchestrator call, through the plan published on that call's receipt. | P03 (context refresh) |
| `completion::children.contract_honoured` | Still calls a calculation check uncited, although its findings rest on admitted receipts; the plan criterion does not. | P08 |
| Skills in the child loop | Pinned on the packet, not loaded. | P06–P13 |
| Windows target, installed app | Not run there; not rebuilt or redeployed. | P16 |

## P05 close-out

- **Implemented** — a typed, versioned, append-only plan changed only by
  validated patches and settled only by receipts; background jobs with pinned
  definitions, bounded retries, no-progress detection, concurrency limits,
  deadlines, prerequisite and memory waits; writer jobs under the existing
  approvals; status, cancel and independent review; correction, stop and crash
  propagation; plan publication to shared memory; completion from backend
  criteria.
- **Wired** — the five tools through every layer P01 lists; `decide`
  escalation; steer and startup recovery in the commands; the manager's stop
  table in `lib.rs`; `capability.search`; the Agents page and one read-only
  command.
- **Tested** — the table above, on the production `authorize` → `execute` path,
  with the production workers, and across the real Node process boundary;
  mutation-checked.
- **Unverified or blocked** — the Spark-driven run; P03's lease suspension.
- **Remaining** — nothing else in P05's scope.

**Exact next step:** P03 — context assembly, model scheduling and durable
handoff: the parent-lease suspension P05 records as open (`delegation::lease_decision`
names the seam), role-scoped tool loading for P05-OBS-1, and context refresh
for jobs that settle between rounds.
---

# P06 — The Document & Vision Analyst with Unlimited-OCR

Worked on 2026-09-24 on branch `claude/hopeful-brahmagupta-1eol9d` from HEAD
`55da242` (P05), in the same **Linux cloud container** — not the Windows host and
not the target machine. No OCR or vision weights and no `llama-server` exist
here; PyMuPDF 1.28.2 does. **P03 is not started**; the analyst reserves the card
through the existing `subagents::scheduling::ModelScheduler`, which is the shared
GPU service this build has. Raw output: `evidence/agent-system/P06/` (index in
its `README.md`).

## Found before, or while, building it

| # | Finding | Where | Now |
|---|---|---|---|
| 1 | An attachment's pages were kept as text only: no coordinates, no method, no record of which regions could not be read, no cache. `media.extract_findings` read the knowledge-base shelf and never OCR; nothing could crop, lay out or read a table. | `agent_runtime/documents.rs`, `orchestrator/runner.rs` | `crate::extraction`: evidence regions with ids, boxes in a named space, method and status; four page tools; `extract_findings` over attached documents (the shelf path kept for knowledge-base ids). |
| 2 | The OCR loop and decode-cap signals reached only UI events; a stored page could not say it was cut. | `commands/ocr.rs` | Every OCR region carries its status (`looped`, `truncated`, `malformed`, `unreadable`) and a note; a page's coverage says so. |
| 3 | The repetition guard ignores any line longer than 96 characters, so a phrase repeated along one line runs to the decode cap unflagged. | `ai_engine/ocr_repetition.rs` | `extraction::ocr::periodic_tail` catches that shape after the line guard; a register whose rows differ is not flagged (tested). The shared guard is unchanged. |
| 4 | Discovery pairs a projector only when its file is named `mmproj-*`, and grants the vision role on a projector or a name token — without anything having shown the model an image. | `registry/discovery.rs` | Explicit binding by any filename, verified from both GGUF headers; vision readiness only from a recorded image probe. Discovery's own inference is left as it was (see "Unverified"). |
| 5 | `parse_gguf_metadata` refuses a `clip` GGUF (no `{arch}.block_count`), and the projector's own keys were never read. | `ai_engine/gguf_meta.rs` | `read_gguf_scalars`, a public header reader sharing the parser. |
| 6 | A page with two typed lines (30 characters) read as a scan under a character threshold alone. Found by the typed-PDF fixture. | `extraction::service` | A page is a scan only when its text layer is thin **and** it has no text or an image covers half of it (`PageRecord::text_layer_adequate`, image coverage measured by the layout pass). |
| 7 | An OCR read that returned nothing left no region, so the page read as never read. | `extraction::ocr` | Recorded as the crop being unreadable. |
| 8 | After OCR, the layout's picture placeholder was still listed as unreadable — the answer said a page could not be read that just had been. Found reading the captured tool output. | `extraction::service` | `superseded_by_ocr`: a placeholder covered by a read OCR crop is not listed. |
| 9 | A document-extractor child is routed to the coordinator's model unless the OCR model carries a certification pack (`certification::choose` → `CHEAPER_ELIGIBLE`); none ships. | `subagents/certification.rs` | Kept: OCR is a service call inside the worker (plan §5.2) that reserves its own card. If a certified OCR model is ever routed to, the worker reads at that model's detent rather than waiting on a card it holds (`detent_for_model`). |
| 10 | `scheduler::queueing_tests::submissions_are_ordered_into_one_queue` fails intermittently under full-suite load: a live worker drains the queue while the test asserts monotonic positions. Not P06 code. | `ai_engine/scheduler.rs` | Recorded; queued as a separate task. Passes 5/5 alone. |

## Implemented

| Area | What | Files |
|---|---|---|
| **Evidence regions** | `rg-` ids derived from document, page, method, label, box and the settings read under; boxes in `pdf-points` or `image-pixels`, always named; method `embedded-text` / `embedded-table` / `ocr` / `vision-inference`; status; table cells (boxes for embedded tables only — OCR locates the table, not its cells); crop id and image hash; extractor identity; OCR cache key. One JSON store per document, written by rename. | `extraction/regions.rs` (new) |
| **Page analyser** | PyMuPDF only, through the P04 page rasteriser (`render_pages.py`): `--layout` (blocks, embedded tables with cell boxes, image coverage), `--crop` (exactly the box, clamped, 72–400 dpi or stored pixels), `--skew` (projection profile; "unmeasured" when too few dark pixels), `--probe-image`. Bounded at 45 s a call. | `sidecars/document_sidecar/render_pages.py`, `extraction/sidecar.rs` (new) |
| **Local OCR** | Unlimited-OCR through the unchanged `stream_ocr` on loopback, the card reserved through the shared `ModelScheduler`; ≤ 4 units a call, none started with under 8 s left, a unit cut by a stop or the deadline kept nowhere. Boxes mapped from the 0–999 grid through the crop onto the page. Checks: repetition (line guard, then period check), decode cap, text with no box, boxes off the grid, empty regions, missing / duplicated / out-of-order pages; skew measured and noted on each box of a skewed page. OCR `<table>` HTML parsed to cells. | `extraction/ocr.rs`, `extraction/tables.rs` (new) |
| **Cache** | Keyed by document, page, crop box and resolution, crop image hash, weights hash, projector file and size, detent, request fingerprint (sampler and prompt), parser version and coordinate convention; stores the stream's events and re-derives regions through the same checks. | `extraction/ocr.rs` |
| **Fields** | Deterministic matching in transcribed regions only (table column, key/value row, labelled line); `found` / `conflicting` / `uncertain` / `not-found`, each value with its region, page, box, method and status; never a confidence number. | `extraction/fields.rs` (new) |
| **Projector binding** | Any filename; verified from both headers (`clip`, `has_vision_encoder`, `projection_dim` = embedding width); `projector-bindings.json`, re-verified (header and size) at registry load; sets `--mmproj`, grants nothing. | `extraction/projector.rs` (new), `registry/mod.rs` |
| **Vision readiness** | A random-token probe image; ready only if the answer contains the token; recorded with the projector's measured SHA-256; void when files change; a ready model gains the vision role at load. Interpretation only from a ready model, as a `vision-inference` proposal region. | `extraction/vision.rs` (new) |
| **Service** | Authorisation (owner + conversation through `DocumentStore::get`, bytes re-hashed to the id), bounded ranges, layout, crops, OCR, interpretation, tables, findings, coverage. | `extraction/service.rs` (new) |
| **Tools** | New `document.layout_map`, `document.render_regions`, `document.ocr_regions`, `document.extract_tables`; completed `media.extract_findings` (fields, question, region ids; attached documents on the agent path, knowledge-base ids by the runner as before), `document.read_pages` (how each page was read; coverage) and `document.search` (pages no search can reach). Every P01 layer: enum, spec, contract (new prerequisites `AttachedDocument`, `LocalOcr`), class, runner refusal, plan, dispatcher, TS catalogue / names / conformance / budget, UI labels; `tool-contract.json` regenerated. | `orchestrator/{tools,contract,runner}.rs`, `agent_runtime/{mod,extraction_tools,planning,tool_policy}.rs`, `agent-runtime/src/*`, `src/services/toolNames.ts` |
| **The agent** | `document-extractor` 1.1.0: same name and id; the page tools granted; the analyst's rules in its instructions. Its worker runs the structured pass for attached documents (model loop or not), publishing clean values, fields not found, unreadable regions and coverage as tool observations on one `media.extract_findings` receipt; cut-read values as open questions; interpretations as proposals. Fields after `fields:`, a question after `question:` in the objective. | `agents/document-extractor.md`, `subagents/{worker,worker_analyst}.rs` |
| **Admin** | `document_analyst_status`, `registry_bind_projector`, `vision_probe_model` (import permission, audited); the Agents page's "What reads pages on this machine" panel; one shared `ModelScheduler` managed for workers, OCR and the probe. | `commands/extraction.rs`, `lib.rs`, `ipc-manifest.json`, `src/services/documentAnalyst.service.ts`, `src/pages/{DocumentAnalystPanel.tsx,Agents.tsx,Agents.module.css}` |

## Contract decisions

1. **Transcription and inference never share a label.** A value is reported
   only from a transcribed region; a vision model's answer is a proposal region,
   headed PROPOSAL, published without a receipt.
2. **The text layer first.** A whole page with an adequate text layer is not
   sent to OCR; an explicit crop always is.
3. **Unreadable is a finding.** An empty OCR region, a read that returned
   nothing and a smudged label are each reported with their box; a label is
   never supplied.
4. **No confidence numbers.** Checks are reported as themselves. The child
   result's required `confidence` field carries the measured page-coverage
   fraction, and says so.
5. **Bounds refuse, never trim.** 10 pages, 6 crops, 4 OCR units, 12 fields,
   2 interpretations.
6. **A cached read is re-checked.** The cache stores events; regions are
   derived again through the same checks.
7. **Binding is not seeing.** A verified projector sets `--mmproj`; only a
   passing image probe makes a model vision-ready.
8. **OCR is a service, not an agent** (plan §5.2): the extractor child runs on
   the model routing gives it; OCR reserves its own card through the shared
   scheduler, and never through a hosted service — the endpoint is checked to be
   loopback before any image is sent.
9. **Order under pressure.** The page tools are listed after every producer and
   reader and before the P04 artifact family (which stays last, as P04
   decided).

## Tests

| What the prompt names | Test (`agent_runtime::extraction_tests` unless noted) |
|---|---|
| typed PDF | `a_typed_pdf_is_read_from_its_text_layer_with_boxes_where_the_ink_is` — layout regions cover the measured ink; the embedded table cell by cell with cell boxes; fields found with method `embedded text layer`; a field absent is `NOT FOUND`; OCR refused for the page and never called; `document.read_pages` names the method and coverage |
| scan | `a_scan_is_read_by_local_ocr_with_boxes_mapped_back_onto_the_page_and_cached` — two pages OCR-read; every box within 1 pt of the ink; one request per page; the second call from the cache with no request |
| page/crop citation | `a_region_crop_is_read_and_its_boxes_land_on_the_page_not_on_the_crop` — a crop away from the origin; the region within 1 pt of the ink and the crop exactly the box asked |
| skewed page | `a_skewed_page_is_measured_and_its_boxes_say_so` — drawn at 4°, measured within 1° (4.25°); noted on every region |
| handwriting | `handwriting_is_located_in_image_pixels_and_a_scribble_is_unreadable_not_guessed` — values in image pixels within 2 px; the scribble unreadable at its box, empty |
| long repeated output | `a_long_repeated_output_is_cut_and_the_page_is_reported_incomplete` (the observed caption loop through the socket); `extraction::ocr::tests::a_long_repeated_output_of_region_lines_…`, `…a_phrase_repeated_along_one_line_…`, `…a_register_whose_rows_differ_is_not_a_loop` |
| table | `an_ocr_table_is_returned_cell_by_cell_and_two_readings_are_both_reported` — unread scan not reported as table-less; cells after OCR; two readings `CONFLICTING`, both reported |
| photo | `a_photo_is_transcribed_and_its_interpretation_is_a_labelled_proposal`; `with_no_vision_ready_model_interpretation_is_refused_by_name_and_nothing_is_guessed` |
| P&ID with an unreadable label | `a_pid_with_a_smudged_label_reports_it_unreadable_at_its_box` — within 2 px of the smudge; no transcription contains the smudged tag |
| authorisation, version, bounds | `a_document_from_another_owner_or_thread_is_not_there_and_changed_bytes_are_refused` |
| **a real parent task, committed graph changes** | `extraction_from_a_parent_task_commits_observations_and_proposals_to_the_graph` — `task.plan_update` → two `agent.delegate` jobs → the production document-extractor; graph revision 2 → 11; the tag admitted as a tool observation citing `page 1 region rg-… [box] pdf-points`, the region within 1 pt of the ink; page 2's empty read in the coverage item; the P&ID's smudged tag admitted as unreadable; the interpretation `Proposed`, not admitted |
| identity and saved settings | `agents::store::tests::the_document_extractor_upgrade_keeps_its_identity_and_its_settings` — the shipped 1.0.0 profile, configured, then 1.1.0: same id, version + 1, name, colour, sharing choice and model binding kept, page tools granted |
| projector and readiness | `extraction::projector::tests::*` (any filename binds when headers match; width mismatch refused with both numbers; text model or audio projector refused; a swapped file not applied), `extraction::vision::tests::*` (a projector alone is not ready; only a passing probe on current files grants vision; the token check) |
| sidecar | `sidecars/document_sidecar/tests/test_render_pages.py::AnalysisModeTests` (7) |
| target machine | `tests/extraction_live.rs` — three `#[ignore]`d gates; **not run here** |

**Seen failing** (`log_mutation.txt`): coordinate mapping without the crop
offset (caught by the region-crop test and a unit test — the whole-page e2e
test could not see it, which is why the region-crop test exists), loop
detection off, a proposal given a receipt, a typed page sent to OCR, an empty
region dropped, the cache never hitting, another conversation's document
readable — each caught. The worker's use of `detent_for_model` is caught only
by the helper's unit test: the path runs only when routing picks a certified OCR
model, which nothing in this build provides.

## Measured

**P06-OBS-1 — the catalogue at 52 tools.** 4,990 estimated tokens at
`minimal` (5,619 at `schemaOnly`): 95.96 per tool against the 96 Rust reserves
(`TOOL_FLOOR_TOKENS_PER_TOOL`). The page tools' schemas were trimmed to get
under it (`regionIds` folded into `regions`, `dpi` dropped from OCR, list bounds
enforced in Rust). At 8k with everything permitted the fitter drops the ten
artifact tools and the four page tools and keeps every orchestrator tool,
producer and reader. The grammar preamble for all 52 tools is 1,229 bytes; its
test bound moved from 1,200 to 1,300 with the measurement recorded. Role-scoped
loading (P03) remains the remedy.

**Skew:** the fixture page drawn at 4.0° measures 4.25° (projection profile,
quarter-degree steps).

## Checks run

| Check | Result |
|---|---|
| `cargo test --lib --no-fail-fast` | **2877 passed, 2 failed**, 3 ignored — the two Windows-path tests that fail at `20061fc`. An earlier run under the same load also failed `ai_engine::scheduler::queueing_tests::submissions_are_ordered_into_one_queue` (finding 10: a race in the test, not P06 code; 5/5 alone). `log_lib_full.txt` |
| focused suites (`log_focused_rust.txt`) | extraction 49, agents::store 20, subagents 105, orchestrator contract 5 / tools 28 / grammar 16, the catalogue test, gguf_meta 34, registry 109, delegation 15, planning 25, tool_policy 6 |
| `npm run test:integration` | **65 passed**, 2 ignored (as at P05) |
| `tests/extraction_live.rs` | 3 ignored — the target gates; **not coverage** |
| `npm run runtime:typecheck`, `npm run runtime:test` | pass; 131 files, **2368** tests |
| `npx tsc --noEmit`, `npm run test:ui`, `npm run build` | pass; 43 files, 618 tests |
| `runtime:build`, `check:bundle`, `check:bundle:self`, `check:offline` | pass |
| `check:ipc`, `check:reachable`, `check:egress`, `check:no-lora`, `check:deployment`, `check:targets`, `check:whitespace` | pass — **180** commands (+3), 198 modules |
| sidecar | `test_render_pages` 11/11; 13 errors elsewhere in the folder, all `pypdf` not installed in this container. `test_pid_engine`'s setup opens three **tracked** fixture PDFs for writing before building them, so the failure left them empty; they were restored from git before committing, and the hazard is queued separately |
| `node scripts/agent-baseline.mjs` | 39 cases: 8 executed, 6 passed, 2 failed (P04's CRLF fixture hashes), 31 blocked; `scan-01`–`03` and `fail-03` now blocked on the P06 target gate by name |
| Agents page | panel rendered in Chromium with a stubbed `invoke` (`agents-page-analyst-panel.png`). gstack (`/design-review`, `/qa`) is not installed here and was not installed. |

## Unverified, open or deliberately left

| What | State | Owner / command |
|---|---|---|
| **Unlimited-OCR reading real pages** | Not run: no weights, projector or `llama-server` here. Every OCR answer in the tests is canned; nothing here says how well the installed model reads, or whether the third-party architecture rewrite behaves on long documents — a successful load proves only a load. | On the target: `set ARJUN_APP_DATA=…\com.arjun.workbench`, then `cargo test --manifest-path src-tauri/Cargo.toml --test extraction_live -- --ignored --nocapture --test-threads=1` (field extraction and citation location scored separately; the pack's three scans; evidence JSON into `evidence/agent-system/P06/`) |
| **Gemma 4 E4B / Qwen3.5 9B vision** | Both are text-only rows on the target (P00): no projector, so no interpretation model exists there, and the analyst says so rather than guessing. | Bind a verified projector (`registry_bind_projector`, Agents page), restart, then probe (`vision_probe_model` or the live gate's probe test) |
| Upstream reference comparison | The pinned upstream Unlimited-OCR implementation was not run here (no weights, no network by policy). | Target, with a reviewed offline copy |
| `--image-max-tokens` | Still not passed to `llama-server` (P00 finding); the detents differ on the device only by decode cap. Carried in the OCR identity's request fingerprint, not fixed here. | P03 (serving) |
| Discovery's inferred vision role | `infer_roles` still grants the vision role from a projector or a name token. Interpretation ignores it (it needs a probe record); the router may not. | P14 (model administration) |
| Scan pages rendered at attach time | `commands/ocr.rs` still reads chat attachments the pre-P06 way (whole pages, text only); the analyst's regions are built on demand from the stored original. The two agree on page text; only the analyst keeps boxes. | Later cutover |
| Parent-lease suspension | P03 not built; OCR and the coordinator's model are serialised by the scheduler and admission, not by a suspended lease. | P03 |
| Windows target, installed app | Not run there; not rebuilt or redeployed. | P16 |

## P06 close-out

- **Implemented** — evidence regions and their store; layout, crops, skew and
  tables from PyMuPDF; local Unlimited-OCR over bounded crops with loop,
  truncation, malformed-span, empty-region and batch checks and a cache keyed by
  source, crop, settings and OCR version; deterministic field matching; verified
  projector binding by any filename; vision readiness only from an image probe;
  proposals kept apart from transcriptions; the extractor's structured pass and
  its P02 publication.
- **Wired** — four new tools and three completed ones through every P01 layer;
  the extractor 1.1.0 with its identity and settings kept; three admin commands
  and the Agents page panel; the registry applying bindings and readiness at
  load; one shared scheduler.
- **Tested** — the eight document kinds the prompt names, citation correctness
  against measured ink, coverage and honest uncertainty, a real parent task with
  committed graph changes, the identity migration; mutation-checked.
- **Unverified or blocked** — every real-model read and probe (the target gate).
- **Remaining** — nothing else in P06's portable scope.

**Exact next step:** run the P06 target gate on the target machine
(`tests/extraction_live.rs`, command above) and record the scores; then P03 —
context assembly, model scheduling and durable handoff (role-scoped tool loading
for P06-OBS-1, `--image-max-tokens`, parent-lease suspension).

---

# P07 — The Knowledge Retriever and local retrieval tools

Worked on 2026-09-25 on branch `claude/hopeful-brahmagupta-1eol9d` from HEAD
`dd740ab` (P06), in the same **Linux cloud container**. It is neither the
Windows host nor the target machine.

No embedding weights and no `llama-server` exist here, and nothing was
downloaded. The target has two embedding models installed and registered but
never measured:

- `Qwen3-Embedding-0.6B-Q8_0` (qwen3, SHA-256 `06507c7b…`);
- `nomic-embed-text-v2-moe.Q8_0` (nomic-bert-moe, 512 context).

`multilingual-e5-small` is not installed anywhere. Per the prompt ("unless the
existing deployment already has a suitable verified provider"), none of the
three is *verified*. So P07 pins a profile for each and ships the qualification
that verifies whichever the machine has. The semantic half stays off until that
qualification passes.

**P03 is not started.** The context compiler P07 extends is the one P02 left in
`agent_runtime/context_compiler.rs`. Raw output: `evidence/agent-system/P07/`
(index in its `README.md`).

## Found before, or while, building it

Traced from production entry points (`lib.rs`, the command layer, the runtime
dispatcher and the worker), not from comments:

| # | Finding | Where | Now |
|---|---|---|---|
| 1 | **Nothing wrote the index in production.** `ingest_collection`, `CollectionStore`, `documents::DocumentService`, `index_document`, `supersede` and `invalidate_source` had no production caller. A shipped build could only ever search an empty index. | `knowledge/ingest.rs`, `documents/`, `lib.rs` | The knowledge connector: `CollectionStore` is managed, and `knowledge_collection_save` / `_remove` / `_sync` are commands. `sync_collection` reads a folder through the real document sidecar, versions each source, withdraws what disappeared and tells the memory graph. It has a UI panel. |
| 2 | A file removed from a collection was reported "retired" but kept answering. | `knowledge/ingest.rs` | `retire_source(Withdrawn)` removes it from every search; its versions stay traceable; `invalidate_source` runs on its memory. |
| 3 | A collection's `restricted_to_roles` was stored and never enforced in search. | `knowledge/index.rs` | `Clearance::clause` is one authorisation clause used by every read: classification, plus the collection being enabled, plus the role restriction. |
| 4 | `chunk_vectors` was keyed by chunk id alone, so a second embedding model silently overwrote the first model's vectors, and a query could be compared against vectors of another model or width. | `knowledge/index.rs` | `vector_spaces` + `chunk_embeddings` keyed by space, chunk and window. A space's key includes the profile, the weights hash, the dimension, the pooling and the profile version. Search reads only the space of the embedder asking. |
| 5 | `plan_launch` had no `--embedding`, so an embedding GGUF would be served as a chat model. | `serving/mod.rs` | Embedding-only entries are launched with `--embedding --pooling <profile's>` and one slot, without the reasoning flags. |
| 6 | `CHEAPER_ELIGIBLE` included `Embedding`, and the retriever's profile says `model-role: embedding`. A certified embedding model could therefore have been routed to as the retriever's *conversational* child. | `subagents/certification.rs` | Only `DocumentOcr` is eligible, and `choose` skips embedding-only entries. The retriever's child runs on the coordinator's model; the embedding endpoint is a service called inside the worker. |
| 7 | `sanitise_fts` ANDs every token, so a natural-language question ("What stops the charge pump from cavitating?") matched nothing. | `knowledge/index.rs` | Hybrid keyword search runs an all-terms pass and then an any-term pass without stopwords. `search_authorized` keeps its all-terms contract. |
| 8 | The context compiler was lexical over memory only; nothing retrieved project knowledge per round. | `agent_runtime/context_compiler.rs` | `compile_with_knowledge`: `context.refresh` searches the run's pinned scope and compiles the hits as evidence blocks with markers and omissions. |
| 9 | The legacy `documents::DocumentStore` created a table named `documents` in the same database as the multimodal index's `documents`. Their schemas differ, so the connector failed at first use. | `documents/store.rs` | Renamed to `collection_documents`. |
| 10 | The sidecar refused `.txt`, `.md` and `.html` without a PDF engine, so a folder of SOPs in plain text could not be read. | `sidecars/document_sidecar/router.py` | Plain-text branch: form-feed pages, HTML reduced to text, and empty pages flagged for review. |

## Implemented

| Area | What | Files |
|---|---|---|
| **Embedding provider** | Pinned profiles, each with architecture, dimension, pooling, query/passage prefixes, `max_tokens` 512 and `window_chars` 900: `multilingual-e5-small` (bert, 384, mean, `query: ` / `passage: `), `nomic-embed-text-v2-moe` (768, mean, `search_query: ` / `search_document: `), `Qwen3-Embedding-0.6B` (1024, last-token, an instruction on the query, none on the passage). A profile is matched from the GGUF architecture and name; an unrecognised model has no profile and is not used. Vectors are L2-normalised by us, not trusted to the server. A passage over the window is split; a window the server refuses as too large is halved, up to 3 times, and otherwise recorded as a failure. The client refuses a non-loopback endpoint. | `knowledge/embedding.rs` |
| **Qualification** | Four checks on the machine's own endpoint: width and finiteness; determinism; a full window accepted; and labelled ordering ≥ 0.9 over 12 bundled probes (8 English, 4 Hindi questions against English passages). The record carries the identity, the measured relevant and distractor means, and a **dense floor** (their midpoint). It is stored in `retrieval/embedding-qualification.json` and is void when the weights hash, profile or dimension differ. | `knowledge/embedding.rs`, `knowledge/qualification_probes.json` |
| **Provider states** | `unavailable` (no embedding-only model) / `unqualified` (one found, not measured or failed) / `qualified`. Only `qualified` enables the semantic half. `ServedEmbeddings` serves the model named by `ARJUN_EMBEDDING_MODEL`, or the only embedding-only entry, on CPU through the existing `GpuOffloadPlan` at the profile's context. | `knowledge/provider.rs` |
| **Index identity and reindexing** | `register_space`; `passages_needing_embeddings`; `store_embeddings` refuses a passage whose text changed since it was read; failures recorded per passage. `reindex()` is resumable and cancellable, and runs in batches in the background at startup when qualified. Vectors of two models never meet. | `knowledge/index.rs`, `knowledge/service.rs` |
| **Versions, revocation, pins** | `source_versions` per logical path: added / revised / unchanged. A revision supersedes the previous version, which stays readable only to a pin taken before it. `retire_source`: withdrawn (file gone) or revoked (never readable again, even under a pin). `index_clock` gives each change a revision. A queued job is pinned to the clock when queued: later versions are invisible to it, versions superseded after the pin are still read *and flagged*, and withdrawn or revoked ones are never read. | `knowledge/index.rs`, `knowledge/service.rs` |
| **Hybrid search** | Authorised in SQL first. Keyword search (all-terms, then any-term) and dense search run over the authorised set only. Fused by reciprocal rank (k = 60), with a deterministic tie-break (fused score, then chunk id), then deduplicated by chunk and by normalised text. A dense hit below the qualification's floor casts no vote. Each hit carries a 480-character excerpt around the query's terms, its source path, version and status, page, region, method (`keyword`, `semantic`, `keyword+semantic`), scores labelled for what they are (keyword rank, cosine, fused rank score), and an evidence handle `ev:<chunk>`. Coverage reports the mode, why it is degraded, partial coverage, and no-answer explicitly. | `knowledge/hybrid.rs` |
| **Rerank** | `LexicalProximityReranker`: a bounded local reranker (at most 24 candidates) over hits already returned in this run, by marker. It is labelled as lexical proximity, not a model. | `knowledge/hybrid.rs`, `agent_runtime/retrieval_tools.rs` |
| **Graph neighbours** | `neighbourhood(start, authorised, depth ≤ 2, max_items, kinds)`: breadth-first inside the reader's authorised set, so a hidden item is neither returned nor walked through. `source_revised` marks claims on a superseded source stale and propagates. | `knowledge/graph/runtime_store.rs` |
| **Notebook scope** | A run scoped to a notebook reads it only if the person owns it (otherwise it gets the same "that notebook does not exist"). Its sources are searched by keyword and fused (`nb:` handles); a source removed from the notebook is counted, not read. | `knowledge/service.rs` |
| **Tools** | New `knowledge.hybrid_search`, `knowledge.rerank`, `knowledge.source_version` and `memory.neighbours` through every P01 layer: enum, spec, contract (Agent path; hybrid_search produces evidence), read-only class, runner refusal, plan order, dispatcher, TS catalogue, names, evidence list, conformance, budget, and UI labels. `tool-contract.json` regenerated. `source_version` says the same thing for hidden and for nonexistent. | `orchestrator/{tools,contract,runner,grammar}.rs`, `agent_runtime/{mod,retrieval_tools,planning,tool_policy}.rs`, `agent-runtime/src/*`, `src/services/toolNames.ts` |
| **The agent** | `knowledge-retriever` 1.1.0: same name and id; the six retrieval tools granted; how it runs in its instructions. Its worker always runs the deterministic retrieval, with or without a model loop. It searches the packet's pinned scope and publishes one `knowledge.hybrid_search` receipt, a coverage observation, and a fact per passage citing path, version, status and excerpt. Passages above its classification ceiling are counted, not published. A retrieval job finishes directly: no model is needed to return passages. | `agents/knowledge-retriever.md`, `subagents/worker.rs`, `subagents/packet.rs`, `agent_runtime/delegation.rs` |
| **The context compiler** | `context.refresh` searches the run's pinned scope for the round's question and records markers. It compiles up to 4 knowledge evidence blocks (method, version, status, scores, handle), with omissions when they do not fit. Project-scoped memory is read when the run has a project. The child's uncertainty now reaches the parent's delegation result. | `agent_runtime/{context_compiler,mod,retrieval,delegation}.rs` |
| **Admin** | `knowledge_collections`, `_collection_save` / `_remove` (policy permission), `_collection_sync` / `_reindex` (upload permission), `knowledge_retrieval_status`, `knowledge_embedding_qualify` (import permission). The Knowledge page shows what retrieval actually runs here and why, and lists the collections. | `commands/retrieval.rs`, `lib.rs`, `ipc-manifest.json`, `src/services/retrieval.service.ts`, `src/pages/{KnowledgeCollectionsPanel,Knowledge}.tsx` |

## Contract decisions

1. **An embedding endpoint is a service, never a child model.** It is served
   with `--embedding` on CPU, is never eligible for routing, and is called
   inside the worker.
2. **Semantic only when measured here.** Keyword retrieval is always
   available. The semantic half runs only for a model whose exact identity
   passed qualification on this machine. Every response says which mode ran
   and, when degraded, why ("keyword only because …").
3. **Filter before scoring.** Authorisation is one SQL clause applied before
   either half scores anything. Counts, coverage, versions and graph
   neighbours are computed inside the authorised set. A hidden and a
   nonexistent record read the same.
4. **Scores are labelled for their purpose.** Keyword rank, cosine and fused
   rank score are each named. None is presented as a confidence or a
   probability.
5. **A floor, not a rank.** A dense hit below the measured floor casts no
   vote. Without it, the nearest vector always "answers", and no-answer
   becomes impossible.
6. **Pins are taken when work is queued.** A later revision is invisible to
   the job. A revoked item is invisible even to a pin.
7. **Deterministic retrieval can finish a job.** Spark/Nemotron query
   planning is not used: no planner is qualified. A retrieval job returns
   its passages without a model loop.
8. **Order under pressure.** The four retrieval tools are listed after the P05
   orchestrator tools and before P06's page tools and P04's artifact family.

## Tests

| What the prompt names | Test (`agent_runtime::retrieval_tests` unless noted) |
|---|---|
| paraphrased versus literal | `a_literal_query_is_found_by_keyword_and_a_paraphrase_only_by_meaning`: the literal tag first by keyword; the paraphrase unreachable by keyword and found as `semantic` |
| multilingual | `a_hindi_question_reaches_the_english_passage_through_the_semantic_half` (the fixture transport's concept lexicon covers the Hindi terms; what a real model does is the target gate) |
| stale SOP | `a_revised_sop_answers_its_old_version_is_flagged_and_claims_on_it_go_stale`: Rev B answers as v2 current; v1 is readable only under an earlier pin, flagged superseded; graph claims on v1 go stale |
| conflicting source | `two_current_sources_that_disagree_are_both_returned_with_their_own_sources`: the vendor letter and the manual, both cited, neither chosen |
| no answer | `a_question_nothing_answers_is_said_to_have_no_answer`: no passages and `no_answer` set, even with a qualified model (the floor) |
| oversized source | `an_oversized_passage_is_embedded_in_windows_and_found_by_its_last_one`: a 2,174-character manual; the fixture server refuses over 700 characters; the answer in the last window is found |
| denied notebook | `a_run_scoped_to_someone_elses_notebook_does_not_read_it_and_says_so` |
| revoked item | `a_restricted_collection_and_a_revoked_document_are_neither_retrieved_nor_disclosed`: no hit, no count, no version, no neighbour; `source_version` reads identically for hidden and nonexistent |
| exact citations | `a_hybrid_passage_is_cited_exactly_by_marker_and_reranked_locally`; every e2e assertion checks path, version and page against the fixture |
| **another agent consumes it through the production path** | `a_retrieval_job_is_pinned_when_queued_and_its_findings_reach_the_parents_next_round`: `agent.delegate` → the production knowledge-retriever → its receipt and facts in the graph → the parent's next `context.refresh` compiles the same passage, while a revision made after queueing is invisible to the job |
| ceiling | `a_passage_above_the_retrievers_ceiling_is_counted_not_published` (added after mutation M8 went uncaught) |
| graph traversal | `memory_neighbours_walks_only_what_the_reader_may_see` |
| unqualified model | `an_unqualified_model_is_not_used_and_the_search_says_why` |
| connector | `the_connector_reads_a_real_folder_into_versioned_authorised_passages`; `knowledge::ingest::tests::a_removed_document_stops_answering_…`, `…a_revised_file_supersedes_its_previous_version` |
| P00 pack | `the_packs_sop_question_retrieves_both_sections_it_must_cite` |
| measured against labels | `labelled_retrieval_measurement` (below) |
| identity kept | `agents::store::tests::the_knowledge_retriever_upgrade_keeps_its_identity_and_its_settings` |
| serving and routing | `serving::tests::an_embedding_model_is_served_as_one_and_only_as_one`; `subagents::tests` (an embedding model is never a child's model) |
| compiler | `agent_runtime::context_compiler::knowledge_tests` (3) |
| target | `tests/retrieval_live.rs`: two `#[ignore]`d gates; **not run here** |

**Seen failing** (`log_mutation.txt`), each caught:

- the collection restriction dropped from the shared clause;
- the query prefix not applied;
- the dense floor ignored;
- the pin ignored;
- a revision not retiring its predecessor;
- an embedding model eligible as a child;
- the compiler not given the round's knowledge;
- a withdrawn file still answering;
- an embedding model served without `--embedding`.

M8, passages above the worker's ceiling being published, was **not caught** on
the first run. A test was added, and the second run catches it. Writing that
test also showed the child's uncertainty never reached the parent's result,
and that is now fixed.

## Measured

**P07-OBS-1: labelled retrieval**, 9 cases (`knowledge/testdata/retrieval/labels.json`),
10-document corpus, K = 5:

| Method | Recall@5 | MRR | No-answer | Label |
|---|---|---|---|---|
| keyword, all terms (`search_authorized`) | 0.25 | 0.25 | 1/1 | deterministic test |
| keyword, any term (hybrid without a model) | 0.75 | 0.625 | 1/1 | deterministic test |
| hybrid via fixture transport | 1.0 | 0.9375 | 1/1 | **deterministic transport — not a quality claim about any model** |

The any-term keyword row is the fallback retrieval this build actually gives
today on the target, where no model is qualified. The hybrid row shows only
that fusion, floor and windowing do what they claim when given vectors that
order correctly. How a real model orders them is `labelled_retrieval_on_the_target`.

**P07-OBS-2: the catalogue at 56 tools** is 5,350 estimated tokens at
`minimal` (95.5 per tool, under the 96 floor).

| Budget | What the fitter drops |
|---|---|
| 8k (3,686) | P04's artifact family, P06's page tools and P07's four |
| 10k | only the artifact family |
| 12k and up | nothing |

The grammar preamble is 1,317 bytes, so its bound moved from 1,300 to 1,400
with the measurement recorded. Role-scoped loading (P03) remains the remedy.

## Checks run

| Check | Result |
|---|---|
| `cargo test --lib --no-fail-fast` | **2,920 passed, 2 failed**, 3 ignored: the two Windows-path tests that fail at `20061fc`. `log_lib_full.txt` |
| focused suites (`log_focused_rust.txt`) | knowledge 369, orchestrator 222, subagents 106, agents 146, serving 45, delegation 15, context compiler 26, extraction 12, artifact tools 10, retrieval 24 |
| `npm run test:integration` | **65 passed**, 2 ignored (as at P06) |
| `tests/retrieval_live.rs` | 2 ignored: the target gates; **not coverage** |
| `npm run runtime:typecheck`, `npm run runtime:test` | pass; **2,385** tests |
| `npx tsc --noEmit`, `npm run test:ui`, `npm run build` | pass; 618 UI tests |
| `runtime:build`, `check:bundle`, `check:bundle:self`, `check:offline` | pass |
| `check:ipc`, `check:reachable`, `check:egress`, `check:no-lora`, `check:deployment`, `check:targets`, `check:whitespace` | pass; **187** commands (+7) |
| `check:lint-budget` | **fails, 53 > 40, pre-existing**: the P06 tree measures 53 too (P07 stashed); no P07 file contributes |
| sidecar | router 13/13, including 4 new plain-text tests. The same 13 `pypdf` errors as P06. The three tracked fixture PDFs truncated by `test_pid_engine`'s setup were restored from git again |
| `node scripts/agent-baseline.mjs` | 39 cases: 8 executed, 6 passed, 2 failed (P04's CRLF hashes), 31 blocked. `sop-01`/`sop-02` are now blocked on P10 by name; their retrieval half is covered by `the_packs_sop_question_…` |
| Knowledge page | the new panels rendered in Chromium with a stubbed `invoke` (`knowledge-retrieval-panel.png`). gstack is not installed here and was not installed |

## Unverified, open or deliberately left

| What | State | Owner / command |
|---|---|---|
| **A real embedding model qualifying and retrieving** | Not run: no weights or `llama-server` here. Every vector in the tests comes from a fixture lexicon. Whether Qwen3-Embedding or nomic-embed-v2 passes, what floor it measures and how it ranks the labelled corpus are unknown. | On the target: `set ARJUN_APP_DATA=…\com.arjun.workbench`, then `cargo test --manifest-path src-tauri/Cargo.toml --test retrieval_live -- --ignored --nocapture --test-threads=1` |
| `multilingual-e5-small` | Profile pinned; weights not present anywhere. Not downloaded (P00, §11.1). | An operator's reviewed, offline copy |
| Spark/Nemotron query planning | Not used: no planner is qualified for it. Deterministic retrieval finishes jobs. | After a qualified planner exists (P09/P10) |
| A model reranker | Only the lexical-proximity reranker exists; no `Rerank`-role model is wired. | Later, behind the same qualification pattern |
| Lesson retrieval | No lesson store exists yet; the compiler reads project memory and knowledge, not lessons. | P15 |
| Notebook sources | Keyword only, no dense half. | Later |
| `search_authorized` all-terms semantics | Kept for its existing callers. Only hybrid search has the any-term pass. | — |
| Windows target, installed app | Not run there; not rebuilt or redeployed. | P16 |

## P07 close-out

- **Implemented**:
  - a pinned, qualified, loopback-only embedding provider with index identity
    and resumable reindexing;
  - versioned, revocable, collection-restricted sources with run pins;
  - filter-first hybrid retrieval with a measured floor, deterministic fusion
    and dedupe;
  - a bounded local reranker and bounded graph neighbours;
  - the knowledge connector the index never had.
- **Wired**:
  - four tools through every P01 layer;
  - the retriever 1.1.0 with its identity kept, publishing receipts and cited
    facts;
  - `context.refresh` compiling pinned knowledge each round;
  - seven admin commands and the Knowledge page panels;
  - embedding models served as embedding models and never as children.
- **Tested**:
  - every case the prompt names;
  - exact citations;
  - consumption by the parent through the production path;
  - labelled measurement;
  - mutation-checked.
- **Unverified or blocked**: real-model qualification and measurement (the
  target gate).
- **Remaining**: nothing else in P07's portable scope.

**Exact next step:** P08, the Calculation Analyst & Checker. Then run the P07
target gate (`tests/retrieval_live.rs`) on the target machine and record the
qualification and measurement.
