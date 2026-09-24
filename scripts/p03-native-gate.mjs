#!/usr/bin/env node
/**
 * P03 native gate — Spark → OCR → Qwen → reviewer → Spark on real servers.
 *
 * Checks every prerequisite first, and only then runs
 * `src-tauri/tests/p03_native_chain.rs`. The report says which of three things
 * happened, and the exit code agrees:
 *
 *   executed and passed  → exit 0
 *   executed and failed  → exit 1
 *   blocked              → exit 2   (a prerequisite is missing; nothing ran)
 *
 * A blocked gate is never a pass. An exit code of zero from a test runner that
 * skipped its only test is exactly the evidence this repository refuses to
 * publish, so the prerequisites are checked here, by name, and a missing one is
 * reported as missing.
 *
 * Inputs (all optional; defaults are the P00 inventory's registry ids):
 *
 *   ARJUN_MODELS_DIR     the directory holding registry.json and the weights
 *   ARJUN_P03_SPARK      default Spark-X2.5-4B-Q8_0
 *   ARJUN_P03_OCR        default unlimited-ocr-q6-k
 *   ARJUN_P03_QWEN       default Qwen3.5-9B-Q4_K_S
 *   ARJUN_P03_REVIEWER   default NVIDIA-Nemotron3-Nano-4B-Q4_K_M
 *   ARJUN_LLAMA_SERVER   the llama-server binary, when it is not on PATH
 *
 * Usage: node scripts/p03-native-gate.mjs [--out evidence/agent-system/P03/native-gate.json]
 */

import { execFileSync, spawnSync } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { homedir, hostname, platform } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const argOut = process.argv.indexOf('--out');
const OUT = resolve(
  ROOT,
  argOut >= 0 ? process.argv[argOut + 1] : 'evidence/agent-system/P03/native-gate.json',
);

const MODELS = {
  spark: process.env.ARJUN_P03_SPARK ?? 'Spark-X2.5-4B-Q8_0',
  ocr: process.env.ARJUN_P03_OCR ?? 'unlimited-ocr-q6-k',
  qwen: process.env.ARJUN_P03_QWEN ?? 'Qwen3.5-9B-Q4_K_S',
  reviewer: process.env.ARJUN_P03_REVIEWER ?? 'NVIDIA-Nemotron3-Nano-4B-Q4_K_M',
};

/** Where this product keeps its models, per platform, unless told. */
function defaultModelsDir() {
  if (process.env.ARJUN_MODELS_DIR) return process.env.ARJUN_MODELS_DIR;
  const identity = 'com.arjun.workbench';
  if (platform() === 'win32' && process.env.APPDATA) return join(process.env.APPDATA, identity, 'models');
  if (platform() === 'darwin') return join(homedir(), 'Library', 'Application Support', identity, 'models');
  return join(process.env.XDG_DATA_HOME ?? join(homedir(), '.local', 'share'), identity, 'models');
}

function run(command, args) {
  try {
    return execFileSync(command, args, { encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'], timeout: 20_000 });
  } catch {
    return null;
  }
}

const missing = [];
const found = {};

// The card.
const gpu = run('nvidia-smi', ['--query-gpu=name,memory.total', '--format=csv,noheader']);
if (gpu) {
  found.gpu = gpu.trim();
} else {
  missing.push('an NVIDIA GPU with nvidia-smi on PATH (the target is an RTX 5060, 8 GB)');
}

// The server binary.
const llama = process.env.ARJUN_LLAMA_SERVER ?? 'llama-server';
const version = run(llama, ['--version']);
if (version !== null) {
  found.llamaServer = version.split('\n').find((line) => /version|build/i.test(line))?.trim() ?? llama;
} else {
  missing.push(`llama-server (tried ${llama}; set ARJUN_LLAMA_SERVER or put it on PATH)`);
}

// The registry and the four models' weights.
const modelsDir = defaultModelsDir();
found.modelsDir = modelsDir;
const registryFile = join(modelsDir, 'registry.json');
let entries = [];
if (existsSync(registryFile)) {
  try {
    const parsed = JSON.parse(readFileSync(registryFile, 'utf8'));
    entries = Array.isArray(parsed) ? parsed : (parsed.models ?? []);
  } catch (error) {
    missing.push(`a readable registry at ${registryFile} (${error.message})`);
  }
} else {
  missing.push(`a model registry at ${registryFile} (set ARJUN_MODELS_DIR)`);
}
for (const [role, id] of Object.entries(MODELS)) {
  const entry = entries.find((candidate) => candidate.id === id);
  if (!entry) {
    if (entries.length > 0) missing.push(`registry entry ${id} for the ${role} hop`);
    continue;
  }
  const weights = resolve(modelsDir, entry.path ?? '');
  if (!existsSync(weights)) missing.push(`weights for ${id} at ${weights}`);
  if (entry.projector && !existsSync(resolve(modelsDir, entry.projector))) {
    missing.push(`the projector for ${id} at ${resolve(modelsDir, entry.projector)}`);
  }
  found[role] = { id, quantization: entry.quantization ?? null };
}

// The toolchain that builds the test.
if (run('cargo', ['--version']) === null) missing.push('cargo');

const report = {
  gate: 'P03 native: Spark -> OCR -> Qwen -> reviewer -> Spark',
  checkedAt: new Date().toISOString(),
  host: hostname(),
  platform: platform(),
  models: MODELS,
  found,
  missing,
  command:
    'node scripts/p03-native-gate.mjs  (runs: cargo test --manifest-path src-tauri/Cargo.toml ' +
    '--test p03_native_chain -- --ignored --nocapture --test-threads=1)',
  note:
    'This is the serving, scheduling and context half of the native chain. The specialist roles ' +
    'are built in P06/P09/P10; the full native chain is P10\'s gate, repeated in P16.',
};

mkdirSync(dirname(OUT), { recursive: true });

if (missing.length > 0) {
  report.status = 'blocked';
  writeFileSync(OUT, `${JSON.stringify(report, null, 2)}\n`);
  console.error(`P03 native gate: BLOCKED — ${missing.length} prerequisite(s) missing:`);
  for (const line of missing) console.error(`  - ${line}`);
  console.error(`Report: ${OUT}`);
  process.exit(2);
}

const hopsReport = OUT.replace(/\.json$/, '.hops.json');
const result = spawnSync(
  'cargo',
  [
    'test',
    '--manifest-path',
    join(ROOT, 'src-tauri', 'Cargo.toml'),
    '--test',
    'p03_native_chain',
    '--',
    '--ignored',
    '--nocapture',
    '--test-threads=1',
  ],
  {
    cwd: ROOT,
    encoding: 'utf8',
    env: {
      ...process.env,
      ARJUN_MODELS_DIR: modelsDir,
      ARJUN_P03_SPARK: MODELS.spark,
      ARJUN_P03_OCR: MODELS.ocr,
      ARJUN_P03_QWEN: MODELS.qwen,
      ARJUN_P03_REVIEWER: MODELS.reviewer,
      ARJUN_P03_REPORT: hopsReport,
    },
  },
);
const output = `${result.stdout ?? ''}${result.stderr ?? ''}`;
// "1 passed" and nothing else: a run that selected no test is not a pass.
const selected = /test result: \w+\. (\d+) passed; (\d+) failed/.exec(output);
const passed = selected ? Number(selected[1]) : 0;
const failed = selected ? Number(selected[2]) : 0;
report.status = result.status === 0 && passed === 1 && failed === 0 ? 'passed' : 'failed';
report.executed = { exitCode: result.status, passed, failed, hopsReport };
report.outputTail = output.split('\n').slice(-40);
writeFileSync(OUT, `${JSON.stringify(report, null, 2)}\n`);
console.log(`P03 native gate: ${report.status.toUpperCase()} — report ${OUT}`);
process.exit(report.status === 'passed' ? 0 : 1);
