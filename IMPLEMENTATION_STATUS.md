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

## §3B — produced, then never displayed

§3A asks why generation hangs. This is the other stuck mode, and the spec is
right that it is a separate pipeline: the file is written, the run finishes, and
the reader still sees nothing. Traced with the chain §3B asks for —
`generate:done` → `present:sent` → `client:received` → `render:*` — along the
artifact path.

The first three links were sound. Both breaks are at `render`, and both are the
same fault wearing different clothes: **a wire contract no compiler checks**.
`ArtifactReport` and `ArtifactPreview` cross the Tauri IPC boundary as JSON, so
each TypeScript declaration is an unverified claim about a Rust struct. Both
claims were false. This is the failure `toolNames.ts` already warns about in its
own header — "three lists in three languages that must agree will otherwise
drift, and the drift is silent" — recurring in two places that had no such check.

### Break 1 — a briefing deck blanks the page it should appear on

| Stage | Result |
|---|---|
| `generate:done` | PPTX written. `ToolName::CreatePptx => Kind::Deck` (`agent_runtime/mod.rs:1893`) |
| `present:sent` | `ArtifactProduced` appended, projected into `RunSummary::artifacts` |
| `client:received` | `agent_task_artifacts` returns `kind: "deck"` — correct on the wire |
| `render` | **throws** |

`Kind` in `agent_runtime/artifacts.rs` has four variants: `Document`, `Workbook`,
`Deck`, `Text`. The surface declared three, and `AssistantMessageCell.tsx`
keyed its icon table on that short union:

```ts
const ARTIFACT_ICONS: Record<ArtifactReport['kind'], typeof FileText> = {
  document: FileText, workbook: FileSpreadsheet, text: FileText,
};
const Icon = ARTIFACT_ICONS[artifact.kind];   // 'deck' -> undefined
```

`<Icon />` on `undefined` is a React invariant violation, not a blank row. The
only boundary in the tree is in `AppShell.tsx`, wrapped around `<Outlet />`, so
the throw replaced **the entire page** with "Component Rendering Exception".
The deck was on disk and sound the whole time.

That is exactly the registry gap §3B calls the single most common cause. It was
worse than the `default: return null` the spec predicts: silence would have lost
one row, and this lost the conversation.

### Break 2 — every preview pane renders empty

`commands/artifact_preview.rs` serialises `{ kind, text, truncated, sizeBytes }`.
The surface declared `{ kind, mime, content | dataUrl | reason, truncated }`, and
both preview panes read the fields that are not there:

```ts
<pre>{preview.content}</pre>              // undefined for every kind
<img src={preview.dataUrl} />            // undefined for images
```

`preview.kind` matched — the seven `PreviewKind` variants camel-case exactly onto
the seven declared — which is why this looked like a working lane. Only the
payload drifted, so the spinner resolved, the pane opened, and it opened blank.
The same broken pane is duplicated in `AssistantMessageCell.tsx` and
`RunView.tsx`, so a fix applied to one would have left the other.

Rust is authoritative here: its shape is the wire format and is covered by tests
(`r.text.contains("hello world")`). The surface was corrected to match it, not
the other way round.

### What was fixed, and why in that shape

- `chat/artifactKind.ts` — kind → presentation as a **total** function. An
  unrecognised kind resolves to a generic file glyph and shows its raw name,
  the same honest fallback `labelForTool` already uses for an unknown tool. A
  lookup that cannot return `undefined` cannot repeat break 1.
- `chat/artifactPreview.ts` — one pure decision function over the real wire
  shape, so the two duplicated panes render from a single tested rule.
- `ui/InlineErrorBoundary.tsx` — §5's per-block boundary. One artifact row
  that throws now degrades to one failed row.
- Parity tests read `artifacts.rs` and `artifact_preview.rs` and fail if a Rust
  variant or field has no counterpart on the surface. A fifth `Kind` added in
  Rust now breaks a test instead of a user's screen.

Both breaks were structural and neither needed a retry, a longer timeout or a
`try/catch`. The generators were never at fault.

## §3C — the inline widget lane

Status: `COMPLETE`. It was blocked by something that had to be measured before
anything could be built, so the measurement is kept here above the result — it
is the reason the lane has the shape it has.

### What already exists

The static half of the lane is here. A `svg` fence in an assistant message is
drawn rather than printed (`Markdown.tsx`), through `svgSanitize.ts`, which is
an allowlist of elements and presentation attributes — no `script`, no `on*`,
no `href` outside a same-document `use`. Static SVG widgets are covered.

What is missing is the interactive half: model-authored HTML + CSS + JS, live in
the message.

### The blocker, measured

Two facts, in order of how much they cost.

**1. `frame-src 'none'`.** `src-tauri/tauri.conf.json` forbids every iframe in
the application today. Read from the file, not assumed.

**2. A `srcdoc` iframe inherits the embedder's CSP.** This is the expensive one,
because the obvious implementation — `srcdoc` plus `sandbox="allow-scripts"`,
with the widget's own CSP in a `<meta>` tag — quietly does not work. Measured
rather than recalled, with a positive control, in Chromium (the engine family
WebView2 is built on; not yet re-run in WebView2 itself):

| Embedder CSP | Inline script inside a sandboxed `srcdoc` frame |
|---|---|
| `script-src 'self' 'unsafe-inline'` | **ran** |
| `script-src 'self'` | **blocked** |

ARJUN ships the second. So a widget frame built with `srcdoc` cannot execute the
model's JavaScript unless `'unsafe-inline'` is added to the **main application's**
`script-src` — which would license inline script everywhere in the surface to
buy it in one iframe. That is not a trade this product should make, and it is
the opposite of what §3C is asking for.

The first probe of this returned "blocked" for both rows, which would have been
a satisfying and completely wrong result: the `<\/script>` inside the `srcdoc`
attribute never closed the tag, so nothing ran for reasons having nothing to do
with CSP. The control is what caught it. Recorded because a one-row measurement
here would have been indistinguishable from the answer.

### What this implies for the build

The widget document needs a **real origin with its own response headers**, not a
local-scheme frame that inherits the page's policy. Local schemes —
`about:blank`, `about:srcdoc`, `blob:`, `data:` — all inherit, so none of them
is a way round it. That means a custom Tauri protocol (`widget://`) serving the
assembled HTML with its own CSP, and `frame-src widget:` added to the app policy.
### What was built

- `commands/widget.rs` — the `widget://` scheme. A prepared document is held in
  memory (capped at 32, evicted oldest-first, 512 KiB each) and served with its
  own policy: `default-src 'none'`, `connect-src 'none'`, `form-action 'none'`,
  `base-uri 'none'`, `img-src data:`. **The CDN allowlist the spec asks for is
  empty, and empty is the correct value** — an air-gapped product has no CDN to
  allow, so a widget that could reach one could reach anything. A test asserts
  the policy string contains no host at all, so the boundary is checked rather
  than reviewed.
- `chat/widgetDocument.ts` — assembles the document, injects a curated set of
  theme custom properties, and carries the `sendPrompt` bridge. Also the reader
  for what a widget says back, which is strict because `sendPrompt` is the one
  capability that reaches out of the sandbox.
- `chat/WidgetFrame.tsx` — mounts the frame `sandbox="allow-scripts"` and
  deliberately without `allow-same-origin`, so it runs at an opaque origin.
  Named stages, an 8s hard timeout, and auto-fit height.
- A `widget` fence in `Markdown.tsx`, distinct from `html`, which stays source.
  The tokenizer now records whether a fence has closed, so a widget is built
  once when the model finishes writing it rather than once per keystroke.

### The timeout is cleared by `ready`, not by `onLoad`

Worth stating because it is the §3B lesson applied before the fact. `onLoad`
fires when the browser has fetched the document, which is not the same as the
reader being able to see anything — a document that throws on its first line
still fires `onLoad`. The frame posts its own `ready` after its script has run,
and only that closes the loading state. A frame that loads and then dies is a
failure with a message, not a spinner.

### Verified at runtime, not only in unit tests

The unit tests assert on the generated string, which cannot tell you whether the
bridge works. It was driven in a browser:

| Checked | Result |
|---|---|
| `ready` fires and closes the loading state | yes |
| height reported, and re-reported on layout change | `140` then `150` |
| `sendPrompt('  explain step 2  ')` arrives | `"explain step 2"`, trimmed |
| malformed messages rejected | 0 accepted |
| theme injected at build | box drew `#123456` |
| host→frame re-theme while running | box became `#abcdef` |
| parent can reach into the frame | **no** — `SecurityError`, which is the point |

That last row is the sandbox working: the probe's own attempt to call a function
inside the frame was refused by the browser.

**What is not yet verified**: the runtime check above used a `srcdoc` frame on
the dev server, so it exercised the bridge and the theming but *not* the
`widget://` response headers, which only exist inside the packaged app. The Rust
half is unit-tested and `cargo check` passes; the served-origin path has not been
exercised end to end in a running Tauri build. It should be, before this is
demonstrated.

## §2 flow/ER lane — reading the Mermaid a model actually writes

Previously `PARTIAL`, and the word understated it. `mermaidParse.ts` is a reader
for exactly one writer: the `graph LR` that `knowledge/graph/render.rs` emits,
with double-quoted labels and pipe-delimited edge labels. The two halves are
pinned to each other by a test on each side, which is a good property and the
reason that module was not widened — widening a reader-for-one-writer dissolves
the thing it exists to enforce.

The build spec's lane is different: `{mermaid_source}` **from the model**. Every
ordinary spelling fell through to a code block, so a reader who asked for a
flowchart was shown its source:

| What a model writes | Before |
|---|---|
| `flowchart TD` — the spelling Mermaid's own docs use | code block |
| `A[Login]` — unquoted label | code block |
| `A[Start] --> B{Choice}` — nodes declared in the edge line | code block |
| `A --> B` — no edge label | code block |
| `erDiagram` in any form | code block |

`mermaidDiagram.ts` is the second reader, and `MermaidGraph` tries them in
order: the writer's own grammar first, so the tool's output keeps the path built
for it (including that `(supplier)` in a label is a *type*, which no general
reader could know), then the general one.

**ER diagrams carry their cardinality.** `CUSTOMER ||--o{ ORDER : places` reads
as `places (1 → 0..n)`. Cardinality is the entire content of an ER
relationship; a picture saying only that two tables are connected would be a
different diagram. Attribute blocks are parsed and listed as small tables under
the canvas, because the canvas draws boxes and a schema without its columns is
a picture of a schema.

Both readers still fall back to the source rather than drawing something they
had to guess at. 21 tests.

## §4 — progressive loading, and where it would be dishonest

The spec asks for stage events "tied to real completed steps only — no fake
timed delays". `runProgress.ts` already carries that as a documented invariant,
stated more strictly than the spec asks:

> A step is only added when the work it names has actually started. There is no
> step for "probably loading" and no step that is inferred from elapsed time.
> … There are no percentages.

**Where §4 was needed and is now built**: the widget lane. Writing → preparing
the sandbox → loading → ready, each a real boundary, with the failure state
freezing on the stage that failed and offering a retry rather than spinning.

**Where the spec's stages are declined, with a measurement**: the file
generators. §4 lists `parse → layout → render → finalize` for PDF, `build
structure → format → finalize` for DOCX, and so on. Those stages describe a
pipeline with a headless browser and subprocess steps — Puppeteer, `soffice` —
which this application deliberately does not have; the generators are pure
in-process functions that write one file.

Measured rather than assumed: 138 artifact tests, each of which writes *and*
re-opens a real OOXML file, complete in **1.31s** — on the order of 5ms per
generation. Four stage events for a 5ms operation would be four events no
reader can ever see, and making them visible would mean inserting the timed
delay §4 explicitly forbids. The single `writing` step, carrying the tool's real
name, is the honest representation of an operation that fast.

If a generator ever grows a subprocess step, that is the moment its stages stop
being theatre and should be added.

## §6 — native rollout

- **CSP.** `frame-src 'none'` → `frame-src widget: http://widget.localhost`.
  That is the only relaxation; `script-src` stays `'self'`, which is what forced
  the `widget://` design in the first place. Both URL shapes are named because
  Windows serves a registered scheme as `http://<scheme>.localhost` and the
  other platforms use the scheme directly.
- **Component registry.** The `mermaid`, `svg` and new `widget` fences all
  dispatch from one switch in `Markdown.tsx`, and every branch has a fallback:
  an unreadable diagram renders as source, a failed widget renders as an error
  with a retry. There is no `default: return null` on this path — the §3B
  failure that started this work.
- **Error boundaries.** `ui/InlineErrorBoundary.tsx`, per artifact row in all
  three places that list artifacts. One row that cannot draw costs that row.
- **No new dependency.** The widget lane adds no crate and no npm package.
- **Version.** 0.1.0 → 0.2.0 across `package.json`, `tauri.conf.json` and
  `Cargo.toml`. There is no `CHANGELOG.md` in this repository and one was not
  invented for this; what shipped is recorded here, which is where this
  repository already tracks it.

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
| Block/process/flow diagrams | no | `COMPLETE` | `artifacts/diagram.rs`; model-written Mermaid via `mermaidDiagram.ts` |
| Engineering diagrams | input only | `COMPLETE` as output | same module, process shapes |
| Knowledge graphs | no | `COMPLETE` | `knowledge.build_graph` |
| ER diagrams | no | `COMPLETE` | `mermaidDiagram.ts` — entities, cardinality, attribute tables |
| Interactive widgets | no | `COMPLETE` | `widget://` scheme + sandboxed frame — see §3C |
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
