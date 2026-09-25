# P08 evidence: the Calculation Analyst & Checker and numerical tools

Produced on 2026-09-25 in a Linux cloud container. This is neither the
Windows host that produced P00–P02 nor the target machine. Branch
`claude/hopeful-brahmagupta-1eol9d`, from HEAD `c3d803f` (P07).

**No model runs anywhere in this evidence, by design.** The engine computes
every number. The independent checker recomputes from the cited sources, and
checking a record never starts a model. The prompt's formulation models
(Nemotron Nano 4B Q8, Qwen3.5 9B Q4) are unqualified on the target (P00), so
formulation by a model is an open gate. Nothing was downloaded or installed.

| File | What it is | Label |
|---|---|---|
| `log_calculation_e2e.txt` | `agent_runtime::calculation_tests` with `--nocapture` (4 tests), through the runtime's own `authorize` → `execute`. The main test runs as follows. Two sourced memory facts are evaluated into a record, which is published on its receipt. The production calculation-checker, dispatched by `agent.delegate_readonly`, re-reads both facts and VERIFIES. An approval note (run `r`) and a briefing deck (run `r-deck`) both cite `[C:calc-…]` with the exact display. A person corrects the reading. The record is unchanged, and its memory item, the verification and both deliverables go stale, with `artifact.resolve_evidence` naming the calculation. A second check reports STALE and recomputes 12.22 % as a new record citing the correction; checking that record VERIFIES. The other tests cover refusal, provisional/unresolved, the other families and the workbook. The captured tool outputs are what a model sees. | deterministic test |
| `log_known_answers.txt` | `calculation::tests` with `--nocapture`. The P00 pack's six calculation cases are run with the pack's inputs against the pack's expected answers (read from `fixtures/agent-system/v1`), and the rendered records are printed. Also: unit conversions, pressure/temperature mismatch, boundary and domain refusals, rounding, uncertainty, a changed source value, the solver families, compare with and without a versioned standard, and sensitivity. | deterministic test |
| `log_focused_rust.txt` | Focused suites: `calculation`, `orchestrator`, `subagents`, `agents`, `artifacts`, the P08 end-to-end tests, delegation, artifact tools and retrieval. | deterministic test |
| `log_lib_full.txt` | `cargo test --lib --no-fail-fast`: 2,982 passed, 2 failed (the two Windows-path tests P06 and P07 also recorded), 3 ignored. | deterministic test |
| `log_integration.txt` | `npm run test:integration`: 65 passed, 2 ignored (as at P07). | deterministic transport |
| `log_runtime.txt` | `npm run runtime:typecheck` and `npm run runtime:test`: 2,402 tests, including the catalogue conformance and the tool budget measured at 60 tools. | deterministic test |
| `log_ui.txt` | `npx tsc --noEmit`, `npm run test:ui` (618), `npm run build`. | deterministic test |
| `log_gates.txt` | runtime build, bundle, offline, egress, LoRA, deployment, IPC, reachability, targets, whitespace: all pass. `check:lint-budget` fails at 53 against 40. That is the same count as at P06 and P07, and no P08 code adds to it. `check:generated` compares against the committed SBOM and passes once the regenerated one is committed. | gates |
| `log_mutation.txt` | Twelve mutations, each applied, the `calculation` tests run, and the file restored. All twelve are caught. The first form of M12 did not compile, and the log says so. | mutation check |
| `baseline.json`, `log_baseline.txt` | `node scripts/agent-baseline.mjs`: 39 cases, 8 executed, 6 passed, 2 failed (P04's CRLF hashes), 31 blocked. `calc-01` … `calc-06` are now blocked on P10 by name, and their engine half is `pack_calc_*`. | harness |

**Open, with what unblocks it.** Formulation of a calculation by a model
(which equation, which basis, which inputs) is what the P00 cases grade as an
*agent's* answer. It needs a qualified formulation model and the first
complete journey (P10). The engine half is already checked against the pack's
own expected answers:

```text
cargo test --manifest-path src-tauri/Cargo.toml --lib calculation::tests::pack_calc -- --nocapture
```
