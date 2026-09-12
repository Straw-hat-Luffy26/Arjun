# Artifact generation: implementation report and future phase

Companion to [`artifact-generation-plan.md`](artifact-generation-plan.md), which
was the pre-implementation audit. That document said *"plan only — nothing
implemented yet."* This one records what was built, what was verified, what was
found still missing afterwards, and what is planned but deliberately **not**
built.

Written 2026-09-11.

---

## Part 1 — Phases A–I: implemented and verified

### Verification, measured

| Suite | Result |
|---|---|
| Rust `--lib` | **2159 passed**, 0 failed (2 ignored) |
| Rust integration | **9/9 binaries ok** |
| Rust baseline | 6 passed |
| agent-runtime | 2188 passed |
| frontend | 470 passed |
| Python sidecar | 107 passed |
| Gates | `egress` · `offline` · `ipc` · `targets` · `reachable` — all pass |
| Skill trust | 69 skills on disk, 69 pinned, no hash drift |

The Rust library suite began this work at 2029 tests. Every test added is a new
assertion; **no existing test was weakened, relaxed or deleted.** Three existing
tests changed, each becoming stricter — recorded in "Corrections" below.

### A — artifact tool timeouts

`create_pdf` had 15 seconds; `create_diagram`, `create_table` and `create_chart`
had 10; the three OOXML writers had 120. Those numbers came from how long each
writer took on a small fixture, so a forty-page report and a two-row table got
the same budget by accident and only the report lost.

All seven now share `ARTIFACT_RENDER_TIMEOUT` (120 s), pinned by
`every_artifact_writer_has_a_budget_fit_for_a_real_deliverable`, with a
companion test that the ceiling stays finite and inside the run deadline. This
is superseded for progress-reporting writers by phase H.

### B — a real PDF validator

`src-tauri/src/artifacts/pdf_validate.rs`, a parser rather than a substring
check. It reads back: the header, `%%EOF` (truncation), `startxref` and that it
lands on the `xref` keyword, **every** cross-reference offset and that each
lands on its own `N 0 obj`, the trailer's `/Size` and `/Root`, the catalogue,
the page tree's `/Count` against its `/Kids`, every page's `/Type` and
`/Contents`, every content stream's `/Length` against its actual bytes, the text
painted by the `Tj` operators (including octal escapes), and `/Info`.

The writer gained an `/Info` dictionary, so a produced PDF is no longer
untitled.

14 tests. Every one of these previously passed the old `"Present, N byte(s)."`
check and is now rejected: truncated file, xref offset pointing into a stream,
`startxref` missing the table, wrong `/Length`, `/Count` disagreeing with
`/Kids`, a stream that paints nothing, a renamed text file.

### C — a real SVG validator

`src-tauri/src/artifacts/svg_validate.rs`. A well-formedness scanner plus SVG
rules: root is `<svg>`, `viewBox` is four usable numbers, every internal
reference resolves (`marker-end="url(#arw)"` against the `<marker id="arw">` the
diagram writer emits), no duplicate ids, every `<text>` has text, nothing drawn
outside the viewBox, and an accessible name.

Hand-written rather than a dependency, for the reason this project already
refused a PDF crate: it ships an SBOM for an air-gapped deployment. 14 tests,
including that a stylesheet's braces and quotes are not read as markup.

### D — the repair pipeline, connected

`artifacts/production.rs` implemented compose → verify → render → re-open →
correct → revise, had 600 lines of tests, and **no production caller**. The live
`create_docx` rendered once and told the model to try again, leaving the failed
file on disk under the name the person was given.

The obstacle was real: `produce` expects to *ask* the model, and a tool handler
cannot — it is already inside the model's call. So there are two loops, in
`artifacts/live_source.rs`:

- **inner**: deterministic repairs needing no new information — a field under
  the heading rather than the key, `Recommendations` for `recommendation`,
  whitespace where text was wanted;
- **outer**: the model's own next turn, which now receives the renderer's
  objections *by field name*.

It never invents content. `a_missing_field_is_never_invented` pins that.

The accepted revision is also copied to the path the caller asked for —
`produce` writes only `note.r1.docx`, and the artifact table, the preview and
the person were all told `note.docx`.

### E — typed content models and general generators

`artifacts/doc_model.rs`: `Document`/`Section`/`Block`,
`Workbook`/`Sheet`/`Column`/`ColumnType`, `Deck`/`SlideModel`. Each validates
*and repairs in memory* before anything is written — which is what makes
"repair" a different operation from "regenerate", something the previous
architecture could not express.

New renderers, each beside the packaging it uses:

| Renderer | Adds |
|---|---|
| `docx::write_document_model` | real `Heading1..6` outline levels, bulleted and numbered lists, bordered tables with repeating header rows, page breaks, `docProps/core.xml` |
| `xlsx::write_workbook_model` | typed cells (a number is a number, not a string), live formulas, `fullCalcOnLoad`, frozen headers, column widths |
| `pptx::write_deck_model` | a composed narrative, speaker notes with correct relationships, tables on slides |
| `pdf::spec_from_document` | the same `Document` a `.docx` is made from, plus a real `Block::PageBreak` |

Validation the model enforces before rendering: heading outlines that skip a
level, ragged tables, empty sections, placeholders (`TBD`, `Lorem`, `[insert`),
Excel's own sheet-name rules, cell types against declared column types,
ISO-8601-only dates, and slide overflow.

**Both paths are live.** `create_docx` takes `template`+`content` (unchanged) or
`sections`; `create_xlsx` takes nothing (the calculation workbook, unchanged) or
`sheets`. This needed `ToolSpec::optional_arguments`, because the gateway
treated every declared argument as required.

### F — repair and regeneration for every model format

`artifacts/produce_model.rs`. One loop: validate → repair → validate → render to
a numbered revision → re-open with the format's real validator → regenerate
(bounded at 2) → accept, or fail honestly.

The deck repair is the one that matters most: `write_deck` silently
**truncated** any section past seven bullets and printed a count saying three
were dropped. The model splits it across slides instead, and
`repair_splits_an_overflowing_slide_rather_than_dropping_anything` asserts
nothing is lost.

Quality is part of acceptance, not advisory beside it: a structurally perfect
PDF of near-blank pages is refused.

`check` receives the model as well as the path — because repair can change the
shape before rendering, so a count captured beforehand describes a file that was
never written.

### G — automatic skill selection and binding

`skills/selection.rs` selects deterministically from two things known before the
run starts: the format the plan will produce, and the person's own words (not
the composed prompt — the same trap routing fell into). Scored by match count,
ties broken by name, capped at 3 for context cost.

Four new format skills — `docx-authoring`, `xlsx-authoring`, `pptx-authoring`,
`pdf-authoring` — each carrying `metadata.for-format`, the ten-section house
contract, and hash pins. 69 skills, all verified.

Selection is a *proposal*: every name goes through `SkillRegistry::load`, so the
trust list, the on-disk hash, clearance, ARJUN version and sovereignty mode are
all still checked, and `narrowing::narrow` still cannot widen a run.

Bound at run start in `commands/agent.rs`, logged per skill with the reason and
the hash, and injected into the system prompt under `--- HOUSE GUIDANCE ---`
with an explicit statement that it grants no tool.

`capability.search` now orders domain skills → format skills → imported, so a
refinery asking what the machine can do still sees `pid-reader` before
`docx-authoring`.

### H — phase-aware progress

`orchestrator/progress.rs`. Five phases with different silence windows, because
their silence means different things: `Composing` gets 4 minutes (a local model
thinks before its first token), `Rendering`/`Regenerating` 30 seconds (a writer
emits continuously), `Validating` 60, `Repairing` 10.

Budgets scale with content (`budget_for`), floored at 120 s and ceilinged at
10 minutes so no single tool can consume the 30-minute run deadline, which
remains the outer bound.

`a_task_that_keeps_moving_is_never_stopped_for_being_slow` runs 1000 advances
without a stop; `the_same_silence_means_different_things_in_different_phases`
shows 11 seconds of quiet stalls a repair and is nothing at all while composing.
Wired through `produce_model`'s loop.

### I — format-aware text files

`artifacts/text_formats.rs` — serialiser **and** parser for each of CSV, JSON,
YAML, XML, Markdown, HTML, with `write_scoped_file` now refusing content that
fails its format's check before writing.

| Format | Serialiser | Validator catches |
|---|---|---|
| CSV | RFC 4180, CRLF, no BOM, doubled quotes | ragged rows, unterminated quotes, BOM, unnamed columns |
| JSON | pretty, trailing newline | parse errors, and a bare string where structure was wanted |
| YAML | two-space, **ambiguous scalars quoted** (`yes`, `1.0`, `08:00`) | tabs, odd indentation, lines that are neither key, item nor scalar |
| XML | five predefined entities, names made valid | unclosed elements, bare `&` |
| Markdown | from `Document` | heading skips, table separators that will not render, unclosed fences, fences with no language |
| HTML | from `Document`, semantic, everything escaped | missing `lang`, missing/duplicate `h1`, `img` with no `alt`, `input` with no label |

`text_is_escaped_rather_than_becoming_markup` pins the one that matters: a
finding containing `<script>` is escaped, not rendered.

### Representative artifacts, produced and verified

`src-tauri/tests/artifact_formats_end_to_end.rs` produces **12 real artifacts**
and re-opens each with its real validator. Run with `ARJUN_ARTIFACT_OUT=<dir>`
to keep them:

```
docx 2320   pdf 2291   xlsx 1874   pptx 8496
svg (diagram) 1493   svg (chart) 1688
md 606   html 1485   csv 86   json 189   yaml 124   xml 321
```

A companion test rejects each format's characteristic corruption, so "the
validator accepted it" means something.

### End-to-end audit checklist

| Claim | Evidence |
|---|---|
| every format has a real generator | the 12-artifact test |
| every format has a real validator | same test, plus the corruption test |
| live generation uses the repair loop | `through_the_repair_loop`, `the_general_path` |
| skills actually selected/loaded/used | `skills::selection` (8), `house_guidance_reaches_the_model` |
| skills survive retries/model switches | `bound_skills_are_carried_rather_than_recomputed` |
| large tasks not killed by short limits | `a_progressing_task_outlives_the_old_wall_clock_ceilings` |
| genuine stalls still detected | `silence_past_the_phase_window_is_a_stall` |
| corrupted artifacts rejected | 28 validator tests |
| repair/regeneration works | `produce_model` (9) |
| only accepted artifacts shown | `nothing_unsound_is_ever_presented_as_finished` |
| existing workflows still work | `the_templates_this_product_shipped_with_still_produce_their_artifacts` |

### Corrections found while doing the work

- **A pre-existing flaky test.** `registry::scan` fixtures were named from
  `SystemTime::now()`, which advances in ~15 ms steps on Windows — two parallel
  tests got the same directory and one deleted the other's files. Fixed with an
  atomic counter; 3/3 clean runs.
- **Three tests became stricter,** not weaker: the capability-paging test now
  asserts a three-group ordering; the approval-note tests assert the file exists
  at the requested path *and* as a numbered revision.
- **My own earlier estimate was wrong.** The plan said the skill catalogue was
  "13.5 KB against an 8 KB cap". There is no 8 KB cap in the code, and the real
  cost was 32,184 bytes — the entire window of the smallest model served.

---

## Part 2 — remaining issues found after verification

Verified as still true after A–I.

1. **`create_chart` writes a file nothing records.** It writes `<slug>.svg` into
   the workspace, but `remember_if_produced` has no `CreateChart` arm and its
   comment still says *"it writes no file"*. The chart is therefore never in the
   run's artifact list, never re-openable, and never validated — despite a
   working SVG validator existing.
2. **ODT, ODS and ODP are not supported.** No generator, no validator, no code
   path.
3. **Layout validation is estimated, not measured.** Slide overflow is derived
   from character counts against box geometry, not from text metrics. A short
   line of wide glyphs can still overflow. PDF has no line-overflow check at
   all.
4. **Skill binding is not in the durable record.** Selection is logged and
   injected, but `TaskRecord` counts `skills_loaded` without naming them, and
   there is no typed `SkillSelected` event beside the existing `SkillLoaded`.
5. **Model selection still ignores the deliverable.** Routing reads intent and
   VRAM fit; it does not read the plan's output format, expected artifact size
   or permitted tools.
6. **The progress tracker is not connected to the agent runtime.** It governs
   `produce_model`; the Node runtime's separate 4-minute `STALL_MS` is
   unchanged, so a long *model* phase is still bounded by a different mechanism.
7. **`write_scoped_file` takes text, not a content model.** The serialisers
   exist and are tested, but the tool's argument is a string, so a model cannot
   hand over a JSON tree and have it serialised — it must produce the text and
   have it checked.
8. **The YAML, XML and HTML parsers handle a subset.** Declared honestly via
   `Check::partial`, but a file from outside this product may be valid and
   rejected.
9. **Reasoning/thinking streaming is improved, not finished.** `parseThinking`
   now handles an open `<think>` block, so a reasoning model's output streams
   instead of showing blank space. Not addressed: models whose reasoning arrives
   on `reasoning_content` still depend on `emits_reasoning` being right, and
   there is no UI state for a model that emits no reasoning at all.
10. **No cross-format consistency check.** The same `Document` becomes `.docx`,
    `.pdf`, `.md` and `.html`, and nothing asserts the four carry the same text.

---

## Part 3 — the future phase: planned only, not implemented

**Nothing in this part is built.** It is the complete plan for the remaining
work, in dependency order.

### F1 — chart and artifact persistence *(small, fixes a live defect)*

Add `ToolName::CreateChart` to `remember_if_produced` with a `Kind::Chart`
variant validated by `svg_validate`; correct the stale comment; set `written` in
`create_chart` so the path is recorded. Test: a chart appears in the run's
artifact list and is re-openable.

### F2 — OpenDocument: ODT, ODS, ODP *(large)*

A second office package format: `mimetype` stored uncompressed and first,
`META-INF/manifest.xml`, and the ODF XML vocabulary (`text:p`, `text:h`,
`table:table`, `office:spreadsheet`, `draw:page`).

- `artifacts/odf.rs` for packaging, mirroring `ooxml.rs`.
- Renderers from the **existing** models — `Document` → ODT, `Workbook` → ODS,
  `Deck` → ODP — so no new content model is needed.
- Validators: package opens, `mimetype` is first and uncompressed, manifest
  lists every part, `content.xml` parses, declared structure present.
- Acceptance: opens in LibreOffice; round-trips the same text as the OOXML
  rendering of the same model.

### F3 — measured layout validation

Replace character-count estimation with text metrics.

- Ship the base-14 AFM width tables (already implied by the PDF writer's fonts)
  and measure strings in points.
- PPTX: measure against the real placeholder geometry; split on measured
  overflow.
- PDF: detect a line whose measured width exceeds the text frame.
- DOCX: flag a table whose measured natural width exceeds the page.
- Honest limit to state: this measures *this product's* fonts and layouts, not
  what Word or PowerPoint will do with a substituted font.

### F4 — skill observability in the durable record

- A `SkillSelected` event type beside `SkillLoaded`, carrying name, reason,
  phase and hash.
- `TaskRecord` names the skills rather than counting them.
- The Approvals and task-record surfaces show which guidance a deliverable was
  produced under.
- Test: a completed run's record names the skills, with reasons.

### F5 — deliverable-aware model selection

Extend the routing decision with the plan's output format, estimated artifact
size and permitted tools. A 60-page report needs a larger context and more
output budget than a two-slide deck, and routing currently cannot see the
difference. Pair with a test that two requests differing only in deliverable
size can route differently.

### F6 — one progress architecture

Connect `orchestrator::progress` to the agent runtime so the Node loop's
`STALL_MS` and the Rust tracker are one mechanism: phase transitions crossing
the JSON-RPC boundary, heartbeats from token deltas, and a single stall policy.
Surface the phase in the chat cell so a person watching a long generation sees
`rendering · 41 s · 12 blocks` rather than a spinner.

### F7 — content models for the text formats

Give `write_scoped_file` optional typed arguments (`rows` for CSV, `value` for
JSON/YAML/XML, `sections` for Markdown/HTML) so the model hands over structure
and the serialiser guarantees the escaping — closing the gap where the model
still writes CSV quoting by hand.

### F8 — parser completeness

Either replace the subset parsers with vendored, SBOM-reviewed crates, or extend
them to cover anchors and multi-line scalars (YAML), namespaces and CDATA (XML),
and void elements and attribute quirks (HTML). Decide explicitly; the current
state is honest but limited.

### F9 — reasoning and streaming, finished

- A table test over (`emits_reasoning` × `supports_toggled_reasoning` × inline
  vs `reasoning_content`) asserting what the UI shows in each of the eight
  cases.
- A visible state for a model that emits no reasoning, so the panel's absence is
  a statement rather than a gap.
- Re-examine `thinkingLevel: "off"`'s effect on the partitioner with a live
  non-reasoning model, which this pass did not reproduce.

### F10 — cross-format consistency

One test that renders the same `Document` to every supported format, extracts
the text from each with that format's own reader, and asserts they agree. This
is the check that stops the four renderers drifting apart.

### Sequencing

F1 first (a live defect, hours). Then F4 and F6 (observability, and the thing a
person watching a slow run actually needs). Then F3 and F10 (quality that can be
measured honestly). F2, F5, F7, F8, F9 after, by whichever the deployment needs.

---

## What this report does not claim

- The artifacts are *structurally* verified by parsers written here. Whether a
  document reads well is not something a test can answer, and the 12 files in
  `ARJUN_ARTIFACT_OUT` are there to be opened rather than trusted.
- The subset parsers are stated as subsets wherever they are used.
- Part 3 is planned and unbuilt. The project is not complete while it stands.
