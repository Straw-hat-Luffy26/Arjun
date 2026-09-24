# P04 evidence — shared artifact, evidence and validation tools

Produced on 2026-09-24 in a Linux cloud container (not the Windows host that
produced P00–P02, and not the target machine). Branch
`claude/hopeful-brahmagupta-1eol9d`, from HEAD `20061fc`.

| File | What it is | Label |
|---|---|---|
| `renderer-inventory.json` | What `artifacts::render::probe()` found here: LibreOffice 24.2.7.2 and PyMuPDF 1.28.2, both in their qualified series | measured on this machine |
| `log_render_available.txt` | The real adapters rendering a produced deck (2 slides → 2 pages) and a produced note validated to **accepted** through the production path, with page handles | real renderer, deterministic inputs |
| `log_render_unavailable.txt` | The same two tests with LibreOffice hidden from `PATH`: render **unavailable**, acceptance **unavailable**, publication refused | real adapter probe, honest-unavailable branch |
| `log_focused_rust.txt` | Every test under `artifacts::`, `agent_runtime::artifact_tools_tests`, `agent_runtime::artifact_carryover_tests`, `deployment::`, `orchestrator::contract`, `orchestrator::tools` — 385 passed, 1 ignored (pre-existing) | deterministic test |
| `log_integration.txt` | `npm run test:integration` (CI's list): 64 passed, 1 ignored; includes `a_model_reads_an_artifact_manifest_through_the_real_runtime` (real Node bundle over stdio, fixture model) | deterministic transport |
| `log_lib_full.txt` | `cargo test --lib`: 2799 passed, 2 failed, 3 ignored — both failures reproduced at HEAD `20061fc` in a clean worktree (Windows path semantics on Linux; not P04) | deterministic test |
| `log_mutation.txt` | Four guarded tests failing with their fix disabled, passing restored | mutation check |
| `log_checks.txt` | Runtime and UI suites, typecheck, IPC / reachability / egress / LoRA / deployment / bundle / offline gates, fixture manifest, sidecar tests | deterministic test and gates |
| `baseline.json`, `log_baseline.txt` | `node scripts/agent-baseline.mjs`: 39 cases, 8 executed, 6 passed, 2 failed (fixture hashes: CRLF, see ledger), 31 blocked | harness |

Environment provisioning done for verification only, and not a repository
dependency change: `apt-get install libgtk-3-dev libwebkit2gtk-4.1-dev
libsoup-3.0-dev libayatana-appindicator3-dev librsvg2-dev libxdo-dev` (Tauri's
Linux build), `apt-get install --no-install-recommends libreoffice-writer
libreoffice-impress libreoffice-calc` (only `libreoffice-core` was present,
which cannot load a document), `pip install pymupdf` (1.28.2; the target
machine has 1.28.0 for the document sidecar).
