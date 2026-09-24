# P05 evidence — the Spark orchestrator and delegation tools

Produced on 2026-09-24 in a Linux cloud container (not the Windows host that
produced P00–P02, and not the target machine). Branch
`claude/hopeful-brahmagupta-1eol9d`, from HEAD `fe3ff44` (P04). No Spark weights
and no `llama-server` exist here.

| File | What it is | Label |
|---|---|---|
| `log_production_driver.txt` | `tests/agent_runtime.rs::a_coordinating_model_plans_delegates_and_reads_a_job_through_the_real_runtime` with `--nocapture`: a scripted coordinator's `task.plan_update` → `agent.delegate` → `agent.status`, routed through the real Node runtime into Rust's authorize/execute; the production retriever's child; the plan at v3 and the job record from the durable store | deterministic transport (fixture model, **not Spark**) |
| `log_focused_rust.txt` | Every test under `agent_runtime::{task_plan, delegation, events, completion, planning, tool_policy, tests}`, `orchestrator::{tools, contract, gateway}`, `subagents` | deterministic test |
| `log_lib_full.txt` | `cargo test --lib --no-fail-fast`: 2832 passed, 2 failed (the Windows-path tests that fail at `20061fc`), 3 ignored | deterministic test |
| `log_integration.txt` | `npm run test:integration`: 65 passed, 2 ignored — the Spark-driven P05 test (needs a Spark server) and `seed_local_accounts` (ignored as before) | deterministic transport |
| `log_mutation.txt` | Five guards (writer approval in the handler, the citation rule, the dependency check, the manager's stop race, the correction refund) each disabled: its test fails; restored: passes | mutation check |
| `log_checks.txt` | Runtime and UI suites, typecheck, build, bundle, offline, IPC (177 commands), reachability, egress, LoRA, deployment, targets, whitespace | deterministic test and gates |
| `baseline.json`, `log_baseline.txt` | `node scripts/agent-baseline.mjs`: 39 cases, 8 executed, 6 passed, 2 failed (P04's CRLF fixture hashes), 31 blocked; no case is P05's | harness |
| `agents-page-jobs-panel.png` | The Agents page's new jobs panel rendered in Chromium from a throwaway entry, Tauri's `invoke` stubbed with job records shaped as the backend produced them in the tests (one completed retrieval, one role with no worker) | UI render with stubbed data |

**Blocked, with the command that unblocks it:** the Spark-driven run of the same
journey, `tests/agent_runtime.rs::spark_routes_the_orchestrator_tools_through_the_real_runtime`
(`#[ignore]`d): start `llama-server -m <Spark-X2.5-4B-Q8_0.gguf> --port 8080 -c 16384 --jinja`,
then `ARJUN_SPARK_URL=http://127.0.0.1:8080/v1 ARJUN_SPARK_MODEL_ID=orchestrator.spark-x2-5-4b
cargo test --manifest-path src-tauri/Cargo.toml --test agent_runtime spark_ -- --ignored --nocapture`.
Not run here, and not counted as coverage.

Environment provisioning carried over from P04 (LibreOffice writer/impress/calc,
PyMuPDF, Tauri's Linux build libraries); nothing new was installed for P05.
