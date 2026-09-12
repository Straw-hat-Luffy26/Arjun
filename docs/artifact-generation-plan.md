# Artifact generation: audit and implementation plan

Status: **superseded — phases A through I are implemented.**
Written 2026-09-11 against commit `ac978f5`.

See [`artifact-generation-report.md`](artifact-generation-report.md) for what
was built, what was verified, what was found still missing afterwards, and the
future phase that is planned but deliberately not built. This document is kept
as the pre-implementation audit: it is the record of what the subsystem looked
like before the work, and several of its findings are quoted in the report.

---

## 0. What this is

An audit of how ARJUN turns a request into a file, and a plan to make every
supported format come out as something a person would put their name on.

The brief was to raise every format to industrial quality. The first job was to
find out what "every format" actually means here, because the answer is smaller
and stranger than it looks from the outside.

---

## 1. Format inventory — what is actually supported

Discovered by reading `ToolName` (`src-tauri/src/orchestrator/tools.rs`), its
dispatch arms (`src-tauri/src/agent_runtime/mod.rs`), the generators
(`src-tauri/src/artifacts/`), and the per-kind checker
(`src-tauri/src/agent_runtime/artifacts.rs`).

| Tool | Wire name | Writes | Generator | Validator |
|---|---|---|---|---|
| `create_docx` | `artifact.create_approval_note` | `.docx` | `artifacts::docx` | `check_document` — template sections present |
| `create_xlsx` | `artifact.create_calculation_workbook` | `.xlsx` | `artifacts::xlsx` | `check_workbook` — opens, live formulas counted |
| `create_pptx` | `artifact.create_briefing_deck` | `.pptx` | `artifacts::pptx` | `check_deck` — slide count, headings |
| `create_pdf` | `artifact.create_pdf` | `.pdf` | `artifacts::pdf` | **none** — "Present, N bytes" |
| `create_diagram` | `artifact.create_diagram` | `.svg` | `artifacts::diagram` | **none** — "Present, N bytes" |
| `create_table` | `artifact.create_table` | `.xlsx` | `artifacts::xlsx` | as Workbook |
| `create_chart` | `artifact.create_chart` | *nothing* | `artifacts::chart` | n/a — returned inline as SVG |
| `write_scoped_file` | `workspace.write_text` | any text file | none | **none** — "Present, N bytes" |
| `validate_artifact` | `artifact.verify_docx` | — | — | dispatches on recorded `Kind` |

**Seven producing tools. Six file formats: `.docx`, `.xlsx`, `.pptx`, `.pdf`,
`.svg`, and whatever `write_scoped_file` is handed.**

### 1.1 Formats the brief named that do not exist here

CSV, HTML, Markdown, JSON, YAML, XML, ODT, ODS, ODP have **no generator, no
tool, no validator and no code path**. They are reachable only as raw bytes
through `write_scoped_file`, which is a text writer with no notion of format:
no schema, no escaping rules, no encoding decision, no structural check.

This is the first thing the plan has to settle, and it is a scope question for
the reader rather than something to assume — see section 9.

### 1.2 The three "generators" are not generators

The wire names are the tell:

- `create_docx` maps to `artifact.create_approval_note`
- `create_xlsx` maps to `artifact.create_calculation_workbook`
- `create_pptx` maps to `artifact.create_briefing_deck`

Each produces exactly **one** document:

- `docx::template_for` recognises a single template, `approval_note`
  (`src-tauri/src/artifacts/docx.rs:62`). Any other name is refused with
  *"Available templates: approval_note."*
- `pptx` has a fixed section list:
  `BRIEFING_SECTIONS = ["Findings", "Recommendation", "Assumptions", "Evidence"]`
  (`pptx.rs:42`). Every deck is those four sections.
- `xlsx::write_workbook` takes a slice of `CalculationRecord` — it is a
  rendering of the run's calculations, not a workbook the model composes.

So "Arjun can generate a DOCX" is true only of one memo. Ask for a maintenance
procedure, an inspection report, a specification or a minutes document and the
tool refuses, because there is no template for it.

---

## 2. Architectural findings

### 2.1 The repair loop already exists, fully tested, and nothing calls it

`src-tauri/src/artifacts/production.rs` implements exactly the workflow the
brief asks for: compose, verify, render, re-open, check, feed the renderer's own
objections back as corrections, try again, up to `MAX_ATTEMPTS`, writing each
attempt as `name.r1.docx`, `name.r2.docx` so nothing is overwritten.

It has 600 lines of tests. It has **no production caller**: `produce(` appears
only inside its own test module, and the only `impl ContentSource` is a test
`Scripted`. There is no adapter from the live model to `ContentSource`.

The live path (`agent_runtime::artifacts::create_docx`) instead does a
**single** attempt: render, check, and on failure return an error string telling
the model to "produce it again" — leaving the retry to the model's discretion,
with the failed file still on disk and no revision numbering.

**This is the single largest gap, and it is a wiring gap, not a design gap.**

### 2.2 Validation is three real checks and three "the file exists"

`Kind::Pdf | Kind::Diagram | Kind::Text => report(true, "Present, N byte(s).")`
(`agent_runtime/artifacts.rs:274`).

The comment there is honest about it — claiming to have verified a PDF's layout
"would be inventing a standard this has no way to hold anything to". That was
the right call when there was no PDF structure to check. It is no longer true:
`artifacts::pdf` writes objects, an xref and a trailer, so a reader for it can
be written.

A PDF that is truncated, has a broken xref, zero pages, or no text at all
currently passes validation.

### 2.3 Skills never reach generation

`planning.rs` does not mention skills. Nothing selects a skill, nothing loads
one as part of a plan, and `skill.load` is reachable **only if the model
chooses to call it**. There is no record tying a loaded skill to a generation
phase, and nothing preserves a loaded skill across a retry or a model switch.

Sixty-five skills are installed. For artifact generation they are inert unless
the model happens to go looking.

### 2.4 Timeouts are not the problem the brief assumes, except where they are

The brief asks not to impose "a fixed 4–5 minute global timeout". What exists:

| Bound | Value | Behaviour |
|---|---|---|
| Run deadline | 30 min | hard |
| Stall guard | 4 min | **rearmed by every event** — already progress-aware |
| `MAX_TURNS` | 64 | backstop |
| `tool.authorize` | 16 min | waits for a person |
| Catalogue fetch | 30 s | hard |

The stall guard is not a 4-minute cap on work; it is 4 minutes of *silence*.
That part is already right.

**The real cap is per-tool**, and it is severe:

| Tool | Timeout |
|---|---|
| `create_docx` / `create_xlsx` / `create_pptx` | 120 s |
| `create_pdf` | **15 s** |
| `create_diagram` / `create_table` / `create_chart` | **10 s** |
| `execute_code` | 60 s |
| `validate_artifact` | 60 s |

A large PDF gets fifteen seconds. These are wall-clock ceilings on a single
tool call, with no phase awareness and no relationship to the size of the thing
being produced.

### 2.5 Model selection does not know what is being produced

Routing classifies intent from the user's words and picks a model by role and
VRAM fit. Nothing in the routing decision reads the plan's output format,
expected artifact size, or which tools the plan permits.

---

## 3. The target pipeline

One pipeline, format-specific at four named points.

```
request
  -> classify           (intent, and now: deliverable format + scale)
  -> plan               (steps, tools, budget)
  -> SELECT SKILLS      (new: format skill + domain skill, recorded)
  -> extract content    (from evidence, per format's content model)
  -> BUILD              (format-specific generator, from a content model)
  -> VALIDATE           (format-specific structural validator)
  -> INSPECT            (format-specific quality checks)
  -> repair | regenerate   (bounded, revision-numbered)
  -> VALIDATE again
  -> accept, or report failure honestly
```

The four format-specific points are: **content model**, **generator**,
**structural validator**, **quality inspector**. Everything else is shared.

### 3.1 The content model is the key idea

Today a tool takes a map of template fields or a body string, and writes it.
There is no intermediate representation, so there is nothing to check *before*
rendering and nothing to repair *without* re-rendering.

Proposal: every generator takes a typed **document model** — for DOCX a tree of
sections, headings, paragraphs, tables and lists; for XLSX sheets and columns
with declared types, formulas and formats; for PPTX an ordered slide narrative
with per-slide density limits; for PDF a paginated block stream. The model is
validated before a byte is written, and repair operates on the model rather
than on the file.

This is what makes "repair" different from "regenerate", and it is what the
current architecture cannot express.

---

## 4. Per-format plan

Each entry: content model, generator work, validator, quality checks, repair
strategy, acceptance.

### 4.1 DOCX
- **Model**: a document of sections, each with a heading level and blocks;
  a block is a paragraph, table, list, figure or page break.
- **Generator**: keep the `ooxml` packaging; replace single-template rendering
  with model rendering. Templates become *starting structures*, not the only
  permitted shape. Add heading styles and outline levels, numbered and bulleted
  lists, table styles with header rows, headers and footers with page numbers,
  document properties, and explicit page breaks.
- **Validator**: package opens; `word/document.xml` parses; declared sections
  present in order; every table well formed (equal cells per row); no empty
  required section; referenced styles exist in `styles.xml`; core properties
  set.
- **Quality**: heading hierarchy has no level skips; no section shorter than a
  sentence; no placeholder text (`TBD`, `Lorem`, angle-bracket fill-ins);
  tables have headers; a references section when the body cites.
- **Repair**: missing field means re-ask for that field only — which is what
  `production::produce` already does. Broken structure means regenerate.
- **Accept**: opens, structure matches the model, quality checks pass, every
  requested section present.

### 4.2 XLSX
- **Model**: a workbook of sheets; each sheet has named columns with a type, a
  number format and a width, plus rows, formulas, totals, freeze panes and
  validation rules.
- **Generator**: generalise beyond the calculation workbook. Add per-column
  number formats, real cell types (not everything as an inline string),
  formulas with dependency ordering, totals rows, freeze panes, column widths,
  conditional formatting where it carries meaning, workbook properties, and
  `fullCalcOnLoad` so Excel recomputes.
- **Validator**: package opens; `workbook.xml` and every sheet parse; sheet
  names unique and legal (31 characters or fewer, and none of the characters
  Excel forbids); every formula references a cell that exists; no circular
  references; declared types match cell content; shared strings table
  consistent.
- **Quality**: no orphan sheet; headers on every table; numeric columns
  formatted; a workbook claiming a calculation has at least one live formula.
- **Repair**: a bad formula is rewritten; a type mismatch recasts that column;
  structural damage regenerates.
- **Accept**: opens, recomputes without error, every requested figure present.

### 4.3 PPTX
- **Model**: a deck with a title slide and ordered sections, each carrying a
  layout, heading, bullets, an optional table or chart, and speaker notes.
- **Generator**: replace the fixed four-section briefing with a narrative the
  model composes. Add title and section slides, layout variety, speaker notes,
  tables and charts on slides, and consistent typography.
- **Validator**: package opens; slide count matches the model; every slide has
  a title placeholder; every declared relationship resolves; notes parts valid.
- **Quality**: **overflow detection** — estimate rendered height from character
  count, font size and box geometry, and fail a slide that cannot fit; bullets
  per slide within bounds; no slide with a title and nothing else; density
  consistent across the deck.
- **Repair**: an overflowing slide is split; an empty slide is dropped or
  filled.
- **Accept**: opens, no overflow, narrative covers what was asked.

### 4.4 PDF
- **Model**: metadata, pages of blocks, headers, footers, numbering.
- **Generator**: `artifacts::pdf` already writes real PDF structure and has had
  four defects fixed recently (wrapping, CP1252 encoding, code indentation,
  over-long tokens). Add explicit pagination with headers, footers and page
  numbers, margins, a document-info dictionary, tables, and a structure
  suitable for printing.
- **Validator**: **this is new work and the biggest validation gap.** Parse the
  file back: header present, xref table resolves, trailer present, `/Root` and
  `/Pages` reachable, page count matches, every page has a content stream, text
  extractable and non-empty, `/Info` populated. A truncated or xref-broken PDF
  must fail.
- **Quality**: no page with no text; page count plausible for the content;
  headings present; no text drawn outside the media box.
- **Repair**: layout overflow re-paginates; broken structure regenerates.
- **Accept**: parses, page count and text match the model, metadata set.

### 4.5 SVG (diagram and chart)
- **Model**: a drawing with a viewBox, nodes, edges and labels — already close
  to what `diagram.rs` holds internally.
- **Generator**: existing; add `title` and `desc` for accessibility and
  consistent label geometry. (An over-long tag drawn through a box edge was
  fixed recently; that wrapping work stays.)
- **Validator**: parses as XML; root is `svg` with a `viewBox`; declared node
  and edge counts present; every `href` resolves within the document; every
  label has text.
- **Quality**: no text outside the viewBox; no overlapping labels; every node
  the model asked for is drawn.
- **Repair**: relayout; structural damage regenerates.

### 4.6 `write_scoped_file` — the text formats

This is where CSV, JSON, YAML, XML, Markdown and HTML would live. Today it is
one untyped writer. The plan is to make it **format-aware by extension**, with
a content model, a serialiser and a validator per format, rather than adding
six new tools:

- **CSV** — declared columns and types, RFC 4180 quoting, UTF-8 with a stated
  BOM policy, consistent record length. Validate by re-parsing and comparing
  the row and column shape against the model.
- **JSON** — build a value tree, serialise, re-parse, and check against a
  declared schema. Refuse prose contamination.
- **YAML** — the same tree through a YAML serialiser; re-parse and compare
  against the JSON round-trip so an indentation error cannot pass.
- **XML** — element tree with escaping and namespaces; re-parse; well formed.
- **Markdown** — heading hierarchy, tables with aligned separators, fenced code
  carrying a language, link targets present.
- **HTML** — semantic elements, one `h1`, `lang` set, `alt` on images, labels
  on inputs; escape all interpolated content; re-parse and run an accessibility
  rule set.

**ODT, ODS and ODP are deliberately excluded from this plan.** They are a
second office package format — a different ZIP layout and a different XML
vocabulary — and would roughly double the OOXML surface. Recommendation: do not
add them unless somebody names a requirement. Saying so here rather than
quietly skipping it.

---

## 5. Skill architecture

1. **Format skills.** One skill per format describing what a professional
   artifact of that kind contains. Selected automatically from the plan's
   output format, not left to the model's discretion.
2. **Domain skills.** Selected by the existing capability classifier from the
   request.
3. **Binding.** The selected skills are recorded on the run, injected into the
   composition prompt, and **carried across retries and model switches** — a
   model switch currently reconstructs the request and would drop them.
4. **Observability.** A `SkillSelected` event alongside the existing
   `SkillLoaded`, carrying which skill, why (the matching signal), and which
   phase used it. `TaskRecord` already counts `skills_loaded`; it needs to name
   them.
5. **Tests.** A generation run asserts the format skill was selected, loaded,
   and present in the composition request.

---

## 6. Timeout and progress architecture

Replace per-tool wall-clock ceilings with **phase-aware progress**.

- **Phases**: composing, rendering, validating, repairing, regenerating. Each
  reports a heartbeat.
- **Progress signal**: bytes written, model tokens, blocks rendered, validation
  steps completed. A phase that is advancing is not stalled, however long it
  takes.
- **Budgets scale with the work**: a 2-slide deck and a 60-page report do not
  get the same ceiling. Derive it from the content model's size, which is known
  before rendering starts.
- **Kill only on genuine silence**: no progress signal for the phase's stall
  window.
- **Keep**: the 30-minute run deadline as an outer bound, and `MAX_TURNS`.
- **Raise immediately**: `create_pdf` at 15 s and the 10 s tools are too low for
  anything substantial and should move to the 120 s tier as an interim step
  before the phase work lands.

---

## 7. Testing strategy

1. **Golden artifacts** — one representative file per format, generated and
   validated in CI, opened by a parser that is *not* ours. The existing
   `emit_one_of_each` audit test already does this for Office formats via
   python-docx and python-pptx; extend it to PDF and SVG.
2. **Validator discrimination** — for every validator, a deliberately corrupt
   fixture it must reject: a truncated PDF, an xref pointing nowhere, a
   circular formula, an overflowing slide, a CSV with a short row.
3. **Repair loop** — a `ContentSource` that fails on attempt 1 and succeeds on
   attempt 2; assert revisions are numbered and the failed one is kept.
4. **Regeneration bound** — a source that never succeeds; assert it stops and
   reports failure rather than looping or returning the broken file.
5. **Skill binding** — assert selection, loading, and survival across a model
   switch.
6. **Large-task timeout** — a generator that takes four minutes while reporting
   progress must not be killed; one that reports nothing must be.

No existing test is to be weakened. Section 2 records several that currently
pass because the validator is "Present, N bytes" — those become real
assertions, which makes them stricter, not looser.

---

## 8. Phasing

| Phase | Work | Risk |
|---|---|---|
| A | Raise the PDF, diagram, table and chart tool timeouts off 10–15 s | trivial |
| B | Real PDF validator: parse back, xref, pages, text, metadata | low |
| C | Real SVG validator | low |
| D | Wire `production::produce` into the live DOCX path — the loop already exists and is tested; this is an adapter from the model to `ContentSource` | medium |
| E | Content models and generalised generators, format by format: DOCX, XLSX, PPTX, PDF | large |
| F | Extend the repair loop to XLSX, PPTX and PDF once they have content models | medium |
| G | Format skills, automatic selection, binding across retries | medium |
| H | Phase-aware progress and scaled budgets | medium |
| I | Format-aware `write_scoped_file`: CSV, JSON, YAML, XML, Markdown, HTML | large |

A through D are independently shippable and each fixes something currently
broken. E is the bulk of the work and is where "industrial-grade" actually
lives.

---

## 9. Open questions for the reader

These change the shape of the work and are decisions rather than details:

1. **Which of CSV, JSON, YAML, XML, Markdown and HTML are actually wanted?**
   None exists today. Each is real work. The brief listed them; the codebase
   has never had them.
2. **ODT, ODS, ODP** — recommend excluding (section 4.6). Confirm.
3. **Template strategy for DOCX.** Generalise to a free document model, or add
   a fixed set of named templates (procedure, inspection report, specification,
   minutes)? The first is more capable; the second is more checkable, and
   checkability is what this product's evidence story rests on.
4. **Does `create_chart` need to write a file?** It currently returns SVG
   inline and records no artifact, so it cannot be validated or re-opened.

---

## 10. What this plan does not claim

- Nothing here is implemented. The findings are verified against the code; the
  design is not.
- The effort in phase E is large and is not a single session's work.
- "Industrial-grade" is asserted per format by the acceptance criteria in
  section 4, not by a global claim. Where a check cannot be written honestly —
  visual balance, for instance — it is listed as a quality heuristic and not as
  a validation gate, because a gate that cannot fail is worse than no gate.
