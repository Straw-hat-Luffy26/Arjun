# P06 evidence — the Document & Vision Analyst with Unlimited-OCR

Produced on 2026-09-24 in a Linux cloud container (not the Windows host that
produced P00–P02, and not the target machine). Branch
`claude/hopeful-brahmagupta-1eol9d`, from HEAD `55da242` (P05). **No OCR or
vision weights and no `llama-server` exist here**; Python 3.11 and PyMuPDF
1.28.2 do. Nothing was downloaded or installed for P06.

| File | What it is | Label |
|---|---|---|
| `log_extraction_e2e.txt` | `agent_runtime::extraction_tests` with `--nocapture`: eight labelled documents drawn by `sidecars/document_sidecar/tests/p06_fixtures.py` (typed PDF, scan, skewed page, handwriting, long repeated output, scanned table, JPEG photo, P&ID with a smudged tag), read through the runtime's own `authorize` → `execute`; the real page analyser; `stream_ocr` over a real loopback socket; the region store, OCR cache, event log and memory graph; the production document-extractor dispatched by `agent.delegate`. Includes the tool outputs a model sees and the graph items committed by two extraction jobs. | **deterministic transport** — the OCR and vision *answers* are canned by a fixture server in Unlimited-OCR's own output format, keyed by the image's SHA-256. No model read anything. |
| `log_focused_rust.txt` | `extraction::*` (regions, OCR executor and checks, tables, fields, projector, vision), the extractor profile upgrade, `subagents`, `orchestrator::{contract,tools,grammar}`, the catalogue test, `gguf_meta`, `registry`, delegation, planning, tool policy | deterministic test |
| `log_lib_full.txt` | `cargo test --lib --no-fail-fast` (see the ledger for the counts and the known failures) | deterministic test |
| `log_integration.txt` | `npm run test:integration`: 65 passed, 2 ignored (the Spark test and `seed_local_accounts`, as at P05) | deterministic transport |
| `log_mutation.txt` | Eight mutations (coordinate mapping, loop detection, proposals given a receipt, a typed page sent to OCR, an empty OCR region dropped, the cache never hitting, another conversation's document readable, the worker ignoring the OCR model it holds), each applied, its test run, the file restored. Run 1 found M1 uncaught end to end — the whole-page crops start at the page origin — and a region-crop test was added; M8 remains caught only by a unit test (see the ledger). | mutation check |
| `log_live_ignored.txt` | `tests/extraction_live.rs`: three target-machine gates, all **ignored here** and not counted as coverage | blocked |
| `log_checks.txt` | Runtime and UI suites, typecheck, build, bundle, offline, IPC (180 commands), reachability, egress, LoRA, deployment, all targets, whitespace | deterministic test and gates |
| `log_sidecar.txt` | `python -m unittest discover -s sidecars/document_sidecar/tests`: `test_render_pages` 11/11; 13 errors in `test_attachment_extract` and `test_pid_engine`, all from `pypdf` not being installed in this container (neither uses the changed script) | deterministic test; environment gap. Its setup truncated three tracked `_fixtures/*.pdf`; restored from git, not committed |
| `baseline.json`, `log_baseline.txt` | `node scripts/agent-baseline.mjs --out …/P06/baseline.json`: 39 cases, 8 executed, 6 passed, 2 failed (P04's CRLF fixture hashes), 31 blocked. `scan-01`–`03` and `fail-03` are now blocked on the **target gate** by name, not on "implement P06" | harness |
| `agents-page-analyst-panel.png` | The Agents page's new "What reads pages on this machine" panel, rendered in Chromium from a throwaway entry with Tauri's `invoke` stubbed by a status shaped like the target machine's P00 inventory (the two Unlimited-OCR rows sharing the patched F16 projector; no other model with a projector) | UI render with stubbed data |

**Blocked, with the command that unblocks it.** On the target machine:

```text
set ARJUN_APP_DATA=C:\Users\<you>\AppData\Local\com.arjun.workbench
cargo test --manifest-path src-tauri/Cargo.toml --test extraction_live -- --ignored --nocapture --test-threads=1
```

That reads the eight labelled documents and the P00 pack's three degraded
scans through the production OCR path on the installed Unlimited-OCR Q6_K and
its patched projector, scores field extraction and citation location separately
against the measured ink, records every loop, cap and unreadable region, and
probes every model with an image projector with a random-token image. It writes
`target-ocr-*.json`, `target-pack-scans-*.json` and `target-vision-probe-*.json`
into this folder, with the measured SHA-256 of the weights and projector, their
GGUF headers and the `llama-server --version` line. Not run here.
