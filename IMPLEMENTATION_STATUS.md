# Artifact generation — implementation status

Audited against the build spec's §2 capability table and §3 stuck-generation
causes. Classification is `COMPLETE` / `PARTIAL` / `NOT_IMPL`, from the tool
registry (`src-tauri/src/orchestrator/tools.rs`) and the generators themselves,
not from memory.

**A note on scope.** The spec is framed around PS 26117. The verbatim text in
[`docs/sih/ps-26117-official.md`](docs/sih/ps-26117-official.md) requires exactly
one sentence of generation: *"Output should be real deliverables, approval notes,
PPT/Word/Excel files, working code, calculations with steps shown, not just chat
replies."* Charts, tables, PDF, diagrams, ER diagrams, widgets and maps are **not
in that text**; engineering drawings appear only as *input* ("review of scanned
drawings", read via OCR). They are ARJUN's additions and are marked as such
below, because this repository has a standing record of requirements previously
invented and attributed to the problem statement.

## §3 — stuck generation (applies to every generator)

| Cause | Status | Evidence |
|---|---|---|
| 1. Unknown/mistyped tool name hangs | `COMPLETE` | Refused by the gateway naming the real tools. 4 tests, incl. `unknown_tool_tests::a_tool_nothing_knows_is_refused_at_once` |
| 2. No timeout on the generation call | **`NOT_IMPL` → fixed** | `ToolSpec::timeout` was sent over the wire as `timeoutSeconds` into a field nothing read. Now enforced in `tools.ts::withTimeout` at the single call site. 9 tests in `toolTimeout.test.ts` |
| 3. Malformed result → re-plan loop | `COMPLETE` | Bounded three ways: `Budget::max_steps` (12), `MAX_RECOVERY_ATTEMPTS = 2`, and the stall guard (4 min of silence) added earlier this session |
| 4. Stale lock / temp / zombie blocks next run | `COMPLETE` | The generators are pure in-process functions writing one file — no subprocess, temp dir or lock to leak. Model servers are enrolled in a Windows job object ("none can outlive this process") |

Cause 2 was the real one, and it was invisible: the ceiling existed, was
declared, was transmitted, and was dropped on arrival.

## §2 — capability table

| Capability | In PS 26117? | Status | Where |
|---|---|---|---|
| Charts | no | `COMPLETE` | `artifacts/chart.rs` → SVG, drawn in chat |
| Tables | no | `COMPLETE` | `xlsx.rs::write_table` + markdown in chat |
| PDF | no | `COMPLETE` | `artifacts/pdf.rs` |
| DOCX + approval notes | **yes** | `COMPLETE` | `artifacts/docx.rs` |
| XLSX | **yes** | `COMPLETE` | `artifacts/xlsx.rs` |
| PPTX | **yes** | `COMPLETE` | `artifacts/pptx.rs` |
| Calculations with steps | **yes** | `COMPLETE` | `calculation` + workbook |
| Working code | **yes** | `COMPLETE` | `sandbox.run_code` |
| Block/process/flow diagrams | no | `COMPLETE` | `artifacts/diagram.rs` |
| Engineering diagrams | input only | `COMPLETE` as output | same module, process shapes |
| Knowledge graphs | no | `COMPLETE` | `knowledge.build_graph` |
| ER diagrams | no | `PARTIAL` | `mermaidParse.ts` reads `graph LR` only; `erDiagram` falls back to source |
| Interactive widgets | no | `NOT_IMPL` | no sandboxed-iframe lane |
| Maps | no | `NOT_IMPL` | no geocoding; correctly absent in an air-gapped product |

## Divergences from the spec, and why

The spec names specific libraries. Three were evaluated and not adopted, with
reasons recorded rather than assumed:

- **`printpdf`** — added, measured at ~200 transitive crates including a GUI
  layout engine, fontconfig and crypto, then removed. This product installs air
  gapped and ships an SBOM a reviewer reads.
- **`plotters` / Chart.js / Recharts** — the spec's chart lane assumes a browser
  charting library. ARJUN renders charts as SVG produced server-side, which
  themes via `currentColor`, needs no CDN, and survives being opened outside the
  app. A CDN allowlist (§5) is moot when nothing is fetched.
- **Puppeteer / headless LibreOffice** — the spec's PDF and PPTX-thumbnail
  lanes. Neither can run in an air-gapped install, and both are the source of
  the §3 subprocess failures they then require guarding against.

Consequently §3's "kill subprocess" is satisfied by having no subprocess: the
timeout returns an error and the in-process call is abandoned, with no temp dir
or lock to reap.

**Still open, and honestly so:** §4 progressive loading (stage events), the ER
diagram lane, interactive widgets, and the §5 native-app items beyond what is
already in place. None of these is a PS 26117 requirement.

## Retrieval — the embedding half

Added after an audit of the whole feature list found the same gap counted four
times: "local embedding model", "reranker", "local embeddings" and "local vector
database" were one missing thing wearing four hats.

The audit also found something the earlier notes did not say. `Hybrid` — the
two-half search with the reciprocal-rank fusion — **had no production caller**.
The live path, `LocalToolRunner::search_hits` in `orchestrator/runner.rs`, calls
`KnowledgeIndex::search` directly, so the fusion had never run outside its own
tests. And inside `Hybrid::search` the vector half was literally
`let vector: Vec<SearchResult> = Vec::new();` after embedding the query and
discarding it — while the returned `methods` claimed `[Keyword, Vector]`.

| Piece | Status | Where |
|---|---|---|
| Vector storage, per model, per passage | `COMPLETE` | `index.rs` — `chunk_vectors`, `store_vectors` |
| Vector search, clearance bound in SQL | `COMPLETE` | `index.rs::search_vectors` |
| Resumable work list and honest coverage | `COMPLETE` | `chunks_needing_vectors`, `vector_coverage` |
| `Embedder` against a local endpoint | `COMPLETE` | `knowledge/embedding.rs` |
| Fusion actually reaching the vector half | `COMPLETE` | `hybrid.rs::search`, now async |
| `Hybrid` wired into the live search path | **`NOT_IMPL`** | `runner.rs:223` still calls `index.search` |
| An embedding model in the registry | **`NOT_IMPL`** | no `registry.json` on this machine |

Twenty tests cover the new code, including the one that matters most:
`an_uncleared_passage_is_never_compared_by_vector`. Clearance is a bound
parameter in the vector query exactly as it is in the keyword query, so a
passage above the reader's clearance is never fetched, never scored, and cannot
influence the ranking of anything else. A filter applied after scoring could not
promise that.

**What is not proven.** No embedding model is installed here, so the HTTP round
trip to a live `/embeddings` endpoint has never run. What is tested is the
refusal of every non-loopback URL, the reply parsing, the index-not-position
pairing, and the whole search path through a fake embedder. The last two rows
above are the remaining work, and neither should be described to a judge as
done.

Three decisions worth recording, all of the same kind — refusing to produce a
number rather than producing a wrong one:

- **No zero-vector fallback.** A zero vector is a valid point in the space; it
  would be stored, ranked and cited, and nothing downstream could tell it from a
  real embedding. Every failure is an error instead.
- **A batch reply is paired by its `index`, never by position.** An
  OpenAI-compatible server may answer out of order, and a positional zip would
  attach every vector to the wrong passage — invisible, because each vector is
  individually valid and search simply returns the wrong paragraph.
- **Nothing embedded yet reports `[Keyword]`, not `[Keyword, Vector]`.** That
  was the pre-existing lie, and it is the one this work most needed to stop
  telling.
