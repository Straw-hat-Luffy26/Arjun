# P07 evidence — the Knowledge Retriever and local retrieval tools

Produced on 2026-09-25 in a Linux cloud container, not the Windows host that
produced P00–P02 and not the target machine. Branch
`claude/hopeful-brahmagupta-1eol9d`, from HEAD `dd740ab` (P06).

**No embedding weights and no `llama-server` exist here.** The target machine has
two embedding models installed and registered but never measured
(`Qwen3-Embedding-0.6B-Q8_0` and `nomic-embed-text-v2-moe.Q8_0`, both
`discovered` in P00's inventory). `multilingual-e5-small` is not installed
anywhere. Nothing was downloaded or installed for P07.

The document sidecar runs here with no PDF engine (`pypdf` and Docling are
absent), so the labelled corpus is Markdown, read through the sidecar's new
plain-text branch.

| File | What it is | Label |
|---|---|---|
| `log_retrieval_e2e.txt` | `agent_runtime::retrieval_tests` with `--nocapture` (16 tests). The real document sidecar reads a folder of labelled Markdown into the real index through `sync_collection`. Also exercised: version history, collection access and memory invalidation; `knowledge.hybrid_search`, `knowledge.source_version`, `knowledge.rerank` and `memory.neighbours` through the runtime's own `authorize` → `execute`; the production retriever dispatched by `agent.delegate` with its queued scope pinned; and the parent's next round compiled by `context.refresh`. Includes the tool outputs a model sees. | **deterministic test** for keyword, connector, versions, access, pins, worker and compiler. **Deterministic transport** for everything semantic: real client, provider, windowing, qualification, fusion and floor against a fixture `/v1/embeddings` server. Its vectors come from a concept lexicon in the test file; no model embedded anything. |
| `measurement-fixture-transport.json` | Recall@5, MRR and no-answer correctness over the 9 labelled cases in `src-tauri/src/knowledge/testdata/retrieval/labels.json`, per method. | The two keyword rows are **deterministic test**, measured on this machine. The hybrid row is **deterministic transport** and is **not** a quality claim about any model. |
| `log_focused_rust.txt` | `knowledge` 369, `orchestrator` 222, `subagents` 106, `agents` 146, `serving` 45, delegation 15, context compiler 26, runtime 83 (+ the Windows-path failure), extraction 12, artifact tools 10, retrieval 24. | deterministic test |
| `log_lib_full.txt` | `cargo test --lib --no-fail-fast`: 2,920 passed, 2 failed (the two Windows-path tests P06 also recorded), 3 ignored. | deterministic test |
| `log_integration.txt` | `npm run test:integration`: 65 passed, 2 ignored (the same two as P06). | deterministic transport |
| `log_mutation.txt` | Ten mutations, each applied, its tests run, the file restored. Run 1 found M8 uncaught (a passage above the retriever's classification ceiling published into shared memory); a test was added and run 2 shows it caught. All ten are caught. | mutation check |
| `log_live_ignored.txt` | `tests/retrieval_live.rs`: two target-machine gates, **ignored here** and not counted as coverage. | blocked |
| `log_gates.txt`, `log_checks.txt` | Egress, LoRA, deployment, offline, runtime audit and typecheck, IPC (187 commands), reachability, targets, whitespace; runtime build, SBOM, bundle, runtime tests (2,385), UI tests (618), frontend typecheck and build. `check:lint-budget` fails at 53 against 40 **with the P06 tree as well** (measured by stashing P07); no P07 file contributes. `check:generated` compares against the committed SBOM and passes once the regenerated one is committed. | deterministic test and gates |
| `log_sidecar.txt` | `python -m unittest discover -s sidecars/document_sidecar/tests`: 127 run, 13 errors, all from `pypdf` not being installed here (the same 13 as P06). The router's four new plain-text tests pass. The suite's setup truncated three tracked `_fixtures/*.pdf`; they were restored from git and not committed. | deterministic test; environment gap |
| `baseline.json`, `log_baseline.txt` | `node scripts/agent-baseline.mjs --out …/P07/baseline.json`: 39 cases, 8 executed, 6 passed, 2 failed (P04's CRLF fixture hashes), 31 blocked. `sop-01` and `sop-02` now name what they wait on: a model run through the task driver (P10). Their retrieval half, both governing passages of Revision D, is checked in `retrieval_tests.rs`. | harness |
| `knowledge-retrieval-panel.png` | The Knowledge page's new "Retrieval on this machine" and "Collections" panels, rendered in Chromium from a throwaway entry with Tauri's `invoke` stubbed. The stubbed status is shaped like the target machine's P00 inventory: Qwen3-Embedding installed, profile-matched, unqualified, so retrieval is keyword-only and says why. | UI render with stubbed data |

**Blocked, with the command that unblocks it.** On the target machine:

```text
set ARJUN_APP_DATA=C:\Users\<you>\AppData\Roaming\com.arjun.workbench
cargo test --manifest-path src-tauri/Cargo.toml --test retrieval_live -- --ignored --nocapture --test-threads=1
```

The first gate serves each installed embedding-only model: CPU only,
`--embedding`, with the pooling from its pinned profile. It measures each model
on the bundled labelled probes, including Hindi questions against English
passages. It writes each record where the application reads it, so a model that
passes becomes the semantic half, and also writes
`target-embedding-qualification-<model>.json` here.

The second gate indexes the labelled corpus, embeds it with the qualified model,
and writes `target-retrieval-measurement.json`: Recall@5, MRR and no-answer
correctness for keyword-only and hybrid retrieval.
