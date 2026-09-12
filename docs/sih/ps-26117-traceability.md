# PS 26117 — traceability for main-chat notebook research

What this change set contributes to the official problem statement, and what it
does not.

**Read [`ps-26117-official.md`](ps-26117-official.md) first.** Every quotation
below is copied from it. That file also records what the problem statement does
*not* say; nothing here may add to the requirements it lists.

This table covers the notebook-research work only. It is not a completion claim
for PS 26117, and notebook Q&A working is not evidence that the problem
statement has been met — most of what the statement asks for is served by other
parts of ARJUN, and several rows below are still open.

## What the statement asks, and where this change set lands

| PS text (verbatim) | Contribution | State |
| --- | --- | --- |
| "ground itself in the organization's own manuals, SOPs and past correspondence through a local knowledge base connector, again with nothing going external" | `/notebook` in the main composer scopes a question to selected sources; retrieval resolves the scope in Rust against the store for the signed-in owner; every answer records an evidence manifest | Implemented, tested |
| "scanned PDFs, handwritten notes, engineering drawings, photographs, read through on device OCR and vision models" | Per-source readiness distinguishes `needsVision` from `failed`, and reports `visionUnavailable` when no local vision model is loaded, rather than calling an image-only page empty | Implemented, tested. The OCR/vision pipeline itself predates this change set |
| "internal document search" | Notebook retrieval over selected sources, owner-scoped and membership-checked before ranking | Implemented; ranking is **keyword only** — see Limitations |
| "spreadsheet work" | The `source-spreadsheet` skill requires reading the complete relevant range and computing deterministically, and forbids reporting a cached formula value as recomputed | Skill authored and selected. The deterministic calculation path is `calculation.evaluate_with_units`, which predates this change set |
| "Output should be real deliverables, approval notes, PPT/Word/Excel files … not just chat replies" | An answer's evidence footer offers "Save to notebook", bound to the manifest's own notebook; authoring skills are still selected alongside reading skills in the same turn | Wired. The end-to-end "ask a source question, then request a cited approval note in the same chat" demonstration has **not been run** |
| "show, through logs or a visible network monitor, that no external calls are made at any point" | Every new skill declares `network: none`; the egress gate passes; no new outbound host is introduced | Static checks pass; **no live network observation was run** |
| "support multiple open weight models at once and automatically pick the right one for a given task" | Unchanged by this work. Skill selection now also reads *input* modality, which is a separate decision from model routing | Not affected |

## What this change set fixed that the statement implies but does not name

The problem statement does not enumerate failure modes. These were found in
ARJUN's own code, and are recorded here because they made the grounding claim
above false in practice:

- `notebook.add_source` wrote a notebook membership row without the document
  store access record, so a document filed into a notebook from chat was listed
  and could never be read. Fixed; both ingestion paths now produce readable
  sources, with a regression test for each.
- "Missing", "corrupt", "this notebook was never given access" and "no text on
  the page" were one message on screen with four different repairs. They are now
  four states, each with its own reason and, where one exists, a repair.
- An empty source list meant "every source", so a selection that failed to load
  widened a question to the whole notebook. `All`, `Subset` and `None` are now
  three values that cannot be mistaken for one another.

## Limitations, stated plainly

- **Retrieval is keyword ranking.** `RetrievalMode::Keyword` is what runs, what
  every manifest records, and what the evidence footer shows. The hybrid and
  semantic paths exist in `knowledge::hybrid` and are not wired into the
  notebook path. Nothing here may be described as semantic or hybrid search.
- **No evaluation has been run.** There is no measured comparison of the
  previous and current pipelines on retrieval recall, answer correctness,
  citation correctness, unsupported-claim rate or latency. The evaluation set
  described in the brief has not been built.
- **Legacy Office files are detected, not converted.** A `.doc`, `.xls` or
  `.ppt` is identified by reading its OLE2 directory — not by its extension and
  not by scanning bytes — and refused by name with `reason: missing-parser`. An
  encrypted modern file in the same container is reported as
  `password-protected`, and a container whose directory names nothing
  recognised is reported as `unidentified-container` rather than being called a
  pre-2007 document. No converter ships. The `legacy-office-conversion` skill
  describes the controlled procedure for when one is installed; it does not
  create one.
- **Skills are selected and loaded; their effect on answers is unmeasured.**
  Tests prove the bodies reach the model with their trust hashes. Whether they
  improve answers is an evaluation question, and the evaluation has not run.
- **No demonstration on real local models.** Every test here runs against real
  stores and real files, and none of them runs a model.
