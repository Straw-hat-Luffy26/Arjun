# P09 evidence: the Document Author and Word authoring tools

Produced on 2026-09-25 and 2026-09-30 in a Linux cloud container, which is
neither the Windows host that produced P00–P02 nor the target machine. Branch
`claude/hopeful-brahmagupta-1eol9d`, from HEAD `8d591c2` (P08).

**No model runs anywhere in this evidence.** The production `document-author`
worker ran its no-model path, which writes only what the task's memory holds
and declares a gap for everything else. The renderers are real: LibreOffice
24.2.7.2 laid out every page and PyMuPDF 1.28.2 rasterised it, on this
machine. Nothing was downloaded or installed.

| File | What it is | Label |
|---|---|---|
| `log_document_e2e.txt` | `agent_runtime::document_tests` with `--nocapture`, one thread (8 tests), every call through the runtime's own `authorize` → `execute`. Shows the composed note and its checks, the refusals (missing field and section, uncited claim, unsupported figure), the declared gaps and their open questions, the 60-row table over four pages, the stale-patch refusals and the one-section patch with its byte-identity report, the compatible legacy path, and the delegated author writing v1, finding it stale after a person's correction, and repairing it as v2 and v3 through its own tool port, with the graph edges printed. | deterministic test, real render |
| `artifacts/` | The versions those tests wrote, and every rendered page of each version's latest render (PNG) with the render record (renderers, fonts, font files and versions). `note-v1.docx` is the cited approval note (two pages); `patch-before.docx` / `patch-after.docx` are one exact-version section patch; `delegated-v1.docx` / `delegated-repaired.docx` are the author's first version and its repair after the correction; `long-table.docx` runs a 60-row table over four pages, its header repeated on each. | deterministic test, real render |
| `log_engine.txt` | `artifacts::authoring`, `artifacts::section_patch`, `artifacts::document_checks` and `subagents::worker::author` with `--nocapture`, including the real render of a long table over pages and the same render with its header repeat removed (caught). | deterministic test, real render |
| `log_focused_rust.txt` | Focused suites: artifacts, agent_runtime, subagents, orchestrator, calculation and agents. | deterministic test |
| `log_lib_full.txt` | `cargo test --lib --no-fail-fast`: 3,006 passed, 2 failed (the two Windows-path tests P06–P08 also recorded), 3 ignored. | deterministic test |
| `log_integration.txt` | `npm run test:integration`: 65 passed, 2 ignored. | deterministic transport |
| `log_runtime.txt` | `npm run runtime:typecheck` and `npm run runtime:test`: 2,422 tests, including the catalogue conformance and the tool budget measured at 65 tools. | deterministic test |
| `log_ui.txt` | `npx tsc --noEmit`, `npm run test:ui` (618), `npm run build` (failed at the egress gate on a P09 mistake; the re-run after the fix passes, appended). | deterministic test |
| `log_gates.txt` | Every repository gate. `check:egress` and `check:deployment` failed on two P09 mistakes and pass after the fixes (re-runs appended). `check:lint-budget` fails at 53 against 40, as at P06–P08. `check:generated` compares against the committed SBOM and passes once the regenerated one is committed. | gates |
| `log_mutation.txt` | Twelve mutations, each applied, the named tests run, and the file restored. The first run missed two (weak tests, both strengthened); the re-run of those two is appended: 12 of 12 caught. | mutation check |
| `baseline.json`, `log_baseline.txt` | `node scripts/agent-baseline.mjs`: 39 cases, 8 executed, 6 passed, 2 failed (P04's CRLF hashes), 31 blocked; the four document cases now name P10's model run. | harness |

**Open, with what unblocks it.** Authoring by a qualified model (Qwen3.5 9B
Q4 is registered on the target, not qualified for document synthesis); the
independent model review, which is P10; and rendering on the Windows target,
where LibreOffice is not provisioned (P04's gate), so the render checks will
report *unavailable* there and a version will not be accepted. On the target:

```text
cargo test --manifest-path src-tauri/Cargo.toml --lib agent_runtime::document_tests -- --nocapture --test-threads=1
```
