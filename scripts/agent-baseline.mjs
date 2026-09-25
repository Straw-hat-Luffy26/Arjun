#!/usr/bin/env node
/**
 * The agent-system baseline harness (plan P00).
 *
 *   node scripts/agent-baseline.mjs [--out <path>]
 *
 * ## What it reports, and why the five states are separate
 *
 *   executed  cases that actually ran
 *   passed    ran, and met their check
 *   failed    ran, and did not
 *   skipped   not selected by this invocation
 *   blocked   a prerequisite this machine does not have
 *
 * Collapsing `blocked` into `passed` is the failure this file exists to
 * prevent, and it is not hypothetical: a suite that reports "0 failed" after
 * selecting no cases is the shape of every green board that was hiding
 * something. So `executed` is counted independently of the exit code, a group
 * that selected nothing is reported as broken rather than clean, and every
 * blocked case carries the exact command that would unblock it.
 *
 * ## The held-out key
 *
 * Before anything runs, the harness checks that no path under the fixture
 * pack's `expected/` directory resolves inside a run workspace root. If one
 * did, every graded case would be grading against a key the agent could have
 * read, and the whole run is reported BLOCKED rather than producing numbers
 * that mean nothing.
 *
 * ## What is real here and what is not
 *
 * The driver group starts the production `AgentRuntime` child. Three of its
 * cases never reach a model and say `deterministic-transport`; the plan permits
 * that for contract tests on condition they are marked, and they are. Most of
 * the agent case groups are BLOCKED at P00 because the agents they grade do not
 * exist yet — which is the true state of this system on the day the harness was
 * written, and recording it as anything else would be the fabrication the plan
 * forbids.
 */

import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { dirname, join, relative, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const PACK = join(ROOT, 'fixtures', 'agent-system', 'v1');
const EXPECTED_DIR = join(PACK, 'expected');
const ARGV = process.argv.slice(2);
const OUT = (() => {
  const at = ARGV.indexOf('--out');
  return at >= 0 && ARGV[at + 1]
    ? ARGV[at + 1]
    : join(ROOT, 'evidence', 'agent-system', 'P00', 'baseline.json');
})();

const PASSED = 'passed';
const FAILED = 'failed';
const BLOCKED = 'blocked';
const SKIPPED = 'skipped';

/** Every case this run produced, in the order it produced them. */
const cases = [];
/** Every command actually run, so the evidence names what was done. */
const commands = [];

function record(entry) {
  if (![PASSED, FAILED, BLOCKED, SKIPPED].includes(entry.status)) {
    throw new Error(`case ${entry.id} has no valid status`);
  }
  cases.push(entry);
  return entry;
}

function run(command, args, options = {}) {
  const started = Date.now();
  const result = spawnSync(command, args, {
    cwd: ROOT,
    encoding: 'utf8',
    timeout: options.timeoutMs ?? 900_000,
    env: { ...process.env, ...(options.env ?? {}) },
    windowsHide: true,
  });
  const entry = {
    command: `${command} ${args.join(' ')}`,
    exitCode: result.status,
    durationMs: Date.now() - started,
    toolchainAbsent: result.error?.code === 'ENOENT',
  };
  commands.push(entry);
  return {
    ...entry,
    ok: !result.error && result.status === 0,
    stdout: result.stdout ?? '',
    stderr: result.stderr ?? '',
  };
}

const sha256 = (path) => createHash('sha256').update(readFileSync(path)).digest('hex');
const rel = (path) => relative(ROOT, path).split(sep).join('/');

// ── Group 1: the held-out key really is held out ────────────────────────────

/**
 * Every directory a run may read or write.
 *
 * `Workspace::create` joins `<app data>/runs/<run_id>`, so the roots are the
 * `runs` directories of the app-data identities this machine carries, plus the
 * temp directory the Rust targets build their workspaces under. A key inside
 * any of them is a key the agent can read.
 */
function workspaceRoots() {
  const roots = [];
  for (const base of [process.env.APPDATA, process.env.LOCALAPPDATA]) {
    if (base) roots.push(join(base, 'com.arjun.workbench', 'runs'));
  }
  if (process.env.TEMP) roots.push(process.env.TEMP);
  return roots;
}

function checkHeldOut() {
  const key = resolve(EXPECTED_DIR);

  if (!existsSync(key)) {
    return record({
      id: 'heldout-01-key-present',
      group: 'held-out',
      kind: 'precondition',
      status: BLOCKED,
      detail: `the grading key directory ${rel(EXPECTED_DIR)} does not exist, so nothing can be graded`,
      unblockCommand: 'Restore fixtures/agent-system/v1/expected/ from version control.',
    });
  }

  const roots = workspaceRoots();
  const breached = roots.filter((root) => {
    const r = resolve(root);
    return key === r || key.startsWith(r + sep);
  });

  return record({
    id: 'heldout-01-key-outside-every-workspace',
    group: 'held-out',
    kind: 'precondition',
    status: breached.length ? FAILED : PASSED,
    detail: breached.length
      ? `the grading key resolves inside a run workspace root: ${breached.join(', ')}. ` +
        'Every graded case would be grading against a key the agent could read.'
      : `the grading key at ${rel(EXPECTED_DIR)} is outside all ${roots.length} workspace root(s) checked`,
    checkedRoots: roots,
  });
}

// ── Group 2: the fixture inputs are the ones that were graded ───────────────

function checkFixtures() {
  const manifestPath = join(PACK, 'manifest.json');
  if (!existsSync(manifestPath)) {
    record({
      id: 'fixture-01-manifest-present',
      group: 'fixtures',
      kind: 'precondition',
      status: BLOCKED,
      detail: 'the fixture manifest does not exist, so no input can be shown to be unchanged',
      unblockCommand: 'node scripts/fixture-manifest.mjs',
    });
    return null;
  }

  const check = run('node', ['scripts/fixture-manifest.mjs', '--check']);
  record({
    id: 'fixture-01-manifest-current',
    group: 'fixtures',
    kind: 'precondition',
    status: check.ok ? PASSED : FAILED,
    detail: check.ok
      ? check.stdout.trim()
      : `the manifest is stale or unreadable: ${(check.stderr || check.stdout).trim()}`,
    command: check.command,
  });

  // Independently of the manifest's own check, re-hash every input a case
  // depends on. The manifest check proves the manifest matches the files; this
  // proves the files this run will read are the ones the manifest names, which
  // is the claim the scores rest on.
  const manifest = JSON.parse(readFileSync(manifestPath, 'utf8'));
  const inputs = [...manifest.files, ...manifest.reused_in_place]
    .filter((file) => file.role !== 'expected.held-out');

  const mismatched = [];
  const missing = [];
  for (const file of inputs) {
    const path = join(ROOT, file.path);
    if (!existsSync(path)) { missing.push(file.path); continue; }
    if (sha256(path) !== file.sha256) mismatched.push(file.path);
  }

  record({
    id: 'fixture-02-inputs-match-their-hashes',
    group: 'fixtures',
    kind: 'precondition',
    status: missing.length || mismatched.length ? FAILED : PASSED,
    detail: missing.length || mismatched.length
      ? `missing: ${missing.join(', ') || 'none'}; changed since the manifest: ${mismatched.join(', ') || 'none'}`
      : `all ${inputs.length} graded input(s) match the sha-256 the manifest records`,
    inputsChecked: inputs.length,
  });

  return manifest;
}

// ── Group 3: the production task driver ─────────────────────────────────────

const MARKER = 'ARJUN_BASELINE_CASE';

function runDriverGroup() {
  const cargo = run('cargo', [
    'test', '--manifest-path', 'src-tauri/Cargo.toml',
    '--test', 'agent_baseline', '--', '--nocapture', '--test-threads=1',
  ], { timeoutMs: 1_800_000 });

  if (cargo.toolchainAbsent) {
    return record({
      id: 'driver-00-toolchain',
      group: 'driver',
      kind: 'precondition',
      status: BLOCKED,
      detail: 'cargo is not on PATH, so the production task driver was not started',
      unblockCommand: 'Install the Rust toolchain and put cargo on PATH.',
    });
  }

  const output = `${cargo.stdout}\n${cargo.stderr}`;
  const lines = output.split(/\r?\n/).filter((line) => line.includes(MARKER));

  // An exit code of zero with no case lines is the failure mode §11.1 names by
  // name. It is reported as a broken group, never as a clean one.
  if (lines.length === 0) {
    return record({
      id: 'driver-00-selected-nothing',
      group: 'driver',
      kind: 'precondition',
      status: FAILED,
      detail:
        `the driver target produced no case records (cargo exit ${cargo.exitCode}). ` +
        'Zero selected cases is not a pass.',
      command: cargo.command,
    });
  }

  for (const line of lines) {
    const payload = line.slice(line.indexOf(MARKER) + MARKER.length).trim();
    let parsed;
    try {
      parsed = JSON.parse(payload);
    } catch (error) {
      record({
        id: 'driver-parse-error',
        group: 'driver',
        kind: 'precondition',
        status: FAILED,
        detail: `a case record did not parse as JSON: ${String(error?.message ?? error)}`,
      });
      continue;
    }
    record({
      id: parsed.id,
      group: 'driver',
      kind: parsed.kind,
      status: parsed.status,
      detail: parsed.detail,
      runId: parsed.runId,
      agentId: parsed.agentId,
      modelId: parsed.modelId,
      trace: parsed.trace,
      artifactHashes: parsed.artifactHashes,
      unblockCommand: parsed.unblockCommand,
      driver: parsed.driver,
      command: cargo.command,
    });
  }
  return null;
}

// ── Group 4: the coding fixture grades in both directions ───────────────────

function runCodingGroup() {
  const tests = join(EXPECTED_DIR, 'hidden-tests', 'test_thickness.py');
  const reference = join(EXPECTED_DIR, 'hidden-tests', 'reference_thickness.py');
  const wrong = join(PACK, 'sources', 'code', 'failing-example.py');

  if (!existsSync(tests)) {
    return record({
      id: 'coding-00-tests-present',
      group: 'coding',
      kind: 'precondition',
      status: BLOCKED,
      detail: 'the hidden tests are not on this machine',
      unblockCommand: 'Restore fixtures/agent-system/v1/expected/hidden-tests/ from version control.',
    });
  }

  const selected = (output) => {
    const match = output.match(/(\d+) passed, (\d+) failed, (\d+) selected/);
    return match ? { passed: +match[1], failed: +match[2], selected: +match[3] } : null;
  };

  // A grading key has to do two things, and a suite that only ever fails does
  // one of them. The reference must pass; otherwise the tests are
  // unsatisfiable and every submission fails for a reason that is not about
  // the submission.
  const good = run('python', [rel(tests), rel(reference)]);
  const goodCounts = selected(good.stdout);
  record({
    id: 'coding-01-reference-passes-the-hidden-tests',
    group: 'coding',
    kind: 'deterministic-fixture',
    status: good.toolchainAbsent
      ? BLOCKED
      : (good.ok && goodCounts && goodCounts.selected > 0 ? PASSED : FAILED),
    detail: good.toolchainAbsent
      ? 'python is not on PATH'
      : `${goodCounts?.selected ?? 0} case(s) selected, ${goodCounts?.passed ?? 0} passed. ` +
        'The grading key is satisfiable.',
    selectedCases: goodCounts?.selected ?? 0,
    unblockCommand: good.toolchainAbsent ? 'Install Python 3.11 and put it on PATH.' : null,
    command: good.command,
  });

  // And the deliberately wrong implementation must fail. A suite that passes
  // it grades nothing, and would report a broken program as working — plan §10
  // journey 5, inverted.
  const bad = run('python', [rel(tests), rel(wrong)]);
  const badCounts = selected(bad.stdout);
  record({
    id: 'coding-02-wrong-implementation-is-reported-as-failed',
    group: 'coding',
    kind: 'deterministic-fixture',
    status: bad.toolchainAbsent
      ? BLOCKED
      : (!bad.ok && badCounts && badCounts.failed > 0 ? PASSED : FAILED),
    detail: bad.toolchainAbsent
      ? 'python is not on PATH'
      : `${badCounts?.failed ?? 0} of ${badCounts?.selected ?? 0} case(s) failed, as required. ` +
        'A harness that passed this file would report a broken program as working.',
    selectedCases: badCounts?.selected ?? 0,
    unblockCommand: bad.toolchainAbsent ? 'Install Python 3.11 and put it on PATH.' : null,
    command: bad.command,
  });

  return null;
}

// ── Group 5: the cases whose agents do not exist yet ────────────────────────

/**
 * Every fixture case that needs an agent this build has not got.
 *
 * Listed individually rather than as one line, because "22 cases blocked on
 * P02-P13" is a fact somebody can act on and "agents not built" is not. Each
 * names the phase that delivers it, so the ledger's remaining work and this
 * report cannot drift apart.
 */
function blockedOnUnbuiltAgents() {
  const pending = [
    ['calc-01-wall-loss-mm', 'P08', 'Calculation Analyst & Checker'],
    ['calc-02-mixed-units-inch', 'P08', 'Calculation Analyst & Checker'],
    ['calc-03-percent-of-what', 'P08', 'Calculation Analyst & Checker'],
    ['calc-04-replacement-window', 'P08', 'Calculation Analyst & Checker'],
    ['calc-05-corrosion-rate', 'P08', 'Calculation Analyst & Checker'],
    ['calc-06-division-by-zero', 'P08', 'Calculation Analyst & Checker'],
    ['art-docx-02-required-sections', 'P09', 'Document Author and Word authoring tools'],
    ['art-docx-03-names-the-revision', 'P09', 'Document Author and Word authoring tools'],
    ['art-docx-04-no-invented-internal-inspection', 'P09', 'Document Author and Word authoring tools'],
    ['code-01-thickness-helpers', 'P11', 'Coding & Testing Agent and real sandbox tools'],
    ['art-pptx-02-every-slide-has-a-title', 'P12', 'Presentation Creator'],
    ['art-pptx-03-no-invented-remaining-life', 'P12', 'Presentation Creator'],
    ['art-xlsx-02-live-formulas', 'P13', 'Spreadsheet Analyst'],
    ['art-xlsx-03-recalculates', 'P13', 'Spreadsheet Analyst, and a recalculation engine'],
    ['art-evidence-01-citations-resolve', 'P02', 'shared memory provenance and authority'],
    ['art-evidence-02-receipt-is-real', 'P02', 'real receipt provenance (plan §3 finding 2)'],
    // P04 built what this grades against -- every registered version carries
    // its sha-256, `artifact.manifest` returns it, and a read re-hashes the
    // stored bytes (src-tauri/src/agent_runtime/artifact_tools_tests.rs). What
    // is still missing is the agent whose *result* carries a produced file.
    ['art-evidence-03-artifact-hash-recorded', 'P09', 'a writer agent whose result carries the artifact version (the P04 tools that record and verify the hash exist)'],
  ];

  // P06 built the analyst (the document-extractor's structured pass over
  // attached scans: layout, local OCR, checks, fields, publication). What these
  // three cases still need is the model itself: the installed Unlimited-OCR
  // weights, projector and llama-server, which only the target machine has.
  // Blocked on that, by name, with the command that runs the gate there.
  for (const id of ['scan-01-governing-reading', 'scan-02-pitting-present', 'scan-03-no-internal-inspection']) {
    record({
      id,
      group: 'agent-cases',
      kind: 'fixture-case',
      status: BLOCKED,
      detail:
        'the Document & Vision Analyst exists (P06); answering needs the local Unlimited-OCR model, ' +
        'which this machine does not have. The deterministic pipeline is covered by ' +
        'src-tauri/src/agent_runtime/extraction_tests.rs; the real read is the target gate.',
      blockedOnPhase: 'P06-target',
      unblockCommand:
        'On the target machine: set ARJUN_APP_DATA to the app data directory, then ' +
        'cargo test --manifest-path src-tauri/Cargo.toml --test extraction_live -- --ignored --nocapture --test-threads=1',
    });
  }

  // P07 built the Knowledge Retriever: the connector, hybrid retrieval and
  // versioned citations. Retrieving the two passages sop-01 must cite (section
  // 3 and the 3.1 carve-out of Revision D) is deterministic and is checked on
  // this pack's own file by
  // src-tauri/src/agent_runtime/retrieval_tests.rs::the_packs_sop_question_retrieves_both_sections_it_must_cite.
  // What these two cases grade is the *answer* -- 9.0 mm because pitting
  // engages 3.1, and the conflict stated rather than silently resolved -- which
  // a model composes. This harness does not run a model through the task
  // driver; the first complete journey (P10) does.
  for (const id of ['sop-01-applicable-minimum', 'sop-02-conflict-surfaced']) {
    record({
      id,
      group: 'agent-cases',
      kind: 'fixture-case',
      status: BLOCKED,
      detail:
        'the Knowledge Retriever exists (P07) and retrieves both governing passages of Revision D ' +
        '(retrieval_tests.rs); grading the answer needs a model run through the production task ' +
        'driver, which this harness does not perform.',
      blockedOnPhase: 'P10',
      unblockCommand:
        'Implement P10 (first complete journey with a real model), then re-run this harness; ' +
        'on the target machine qualify an embedding model first with ' +
        'cargo test --manifest-path src-tauri/Cargo.toml --test retrieval_live -- --ignored --nocapture --test-threads=1',
    });
  }

  for (const [id, phase, what] of pending) {
    record({
      id,
      group: 'agent-cases',
      kind: 'fixture-case',
      status: BLOCKED,
      detail: `the agent that would answer this does not exist in this build yet: ${what}`,
      blockedOnPhase: phase,
      unblockCommand: `Implement ${phase} of docs/plans/2026-09-20-agent-system-build-plan.md, then re-run this harness.`,
    });
  }

  // The failure cases are blocked for the same reason, and named individually
  // so the count in this report matches the count in the pack.
  const path = join(PACK, 'failures', 'failure-cases.json');
  if (!existsSync(path)) {
    return record({
      id: 'failure-cases-00-present',
      group: 'failure-cases',
      kind: 'precondition',
      status: BLOCKED,
      detail: 'the failure-case definitions are not on this machine',
      unblockCommand: 'Restore fixtures/agent-system/v1/failures/ from version control.',
    });
  }
  const failures = JSON.parse(readFileSync(path, 'utf8'));
  // Failure cases whose agent now exists and whose remaining blocker is a model
  // on the target machine. Named with that blocker rather than "implement the
  // owning phase", which would no longer be true.
  const onTarget = {
    'fail-03-unreadable-scan-region':
      'the analyst reports an unreadable region with its page and box and never guesses or blanks ' +
      'it (P06; deterministic cases in src-tauri/src/agent_runtime/extraction_tests.rs); reading ' +
      'page-03 itself needs the local Unlimited-OCR model, which this machine does not have',
  };
  for (const failure of failures.cases) {
    if (onTarget[failure.id]) {
      record({
        id: failure.id,
        group: 'failure-cases',
        kind: 'fixture-case',
        status: BLOCKED,
        detail: onTarget[failure.id],
        blockedOnPhase: 'P06-target',
        unblockCommand:
          'On the target machine: set ARJUN_APP_DATA to the app data directory, then ' +
          'cargo test --manifest-path src-tauri/Cargo.toml --test extraction_live -- --ignored --nocapture --test-threads=1',
      });
      continue;
    }
    record({
      id: failure.id,
      group: 'failure-cases',
      kind: 'fixture-case',
      status: BLOCKED,
      detail:
        `defined and not yet exercisable: the correct outcome is '${failure.correct_failure}', ` +
        'and no agent exists to produce it',
      unblockCommand: 'Implement the owning phase, then re-run this harness.',
    });
  }
  return null;
}

// ── Report ──────────────────────────────────────────────────────────────────

checkHeldOut();
checkFixtures();
runDriverGroup();
runCodingGroup();
blockedOnUnbuiltAgents();

const tally = (status) => cases.filter((entry) => entry.status === status).length;
const executed = cases.filter((entry) => entry.status === PASSED || entry.status === FAILED).length;

const report = {
  schema: 'arjun.agent-baseline/1',
  phase: 'P00',
  generated_at: new Date().toISOString(),
  reading_rule:
    'executed counts only cases that ran. blocked is a prerequisite this machine does not have and is never a ' +
    'pass. A group that selected nothing is reported as failed, not as clean.',
  totals: {
    cases: cases.length,
    executed,
    passed: tally(PASSED),
    failed: tally(FAILED),
    blocked: tally(BLOCKED),
    skipped: tally(SKIPPED),
  },
  by_kind: cases.reduce((acc, entry) => {
    acc[entry.kind] = (acc[entry.kind] ?? 0) + 1;
    return acc;
  }, {}),
  deterministic_fixtures_used: cases
    .filter((entry) => entry.kind === 'deterministic-transport' || entry.kind === 'deterministic-fixture')
    .map((entry) => entry.id),
  commands,
  cases,
};

mkdirSync(dirname(OUT), { recursive: true });
writeFileSync(OUT, `${JSON.stringify(report, null, 2)}\n`, 'utf8');

const { totals } = report;
process.stdout.write(
  '\nagent-system baseline — P00\n' +
  `  cases      ${totals.cases}\n` +
  `  executed   ${totals.executed}\n` +
  `  passed     ${totals.passed}\n` +
  `  failed     ${totals.failed}\n` +
  `  blocked    ${totals.blocked}\n` +
  `  skipped    ${totals.skipped}\n` +
  `\n  written to ${rel(OUT)}\n`);

if (totals.blocked) {
  process.stdout.write('\nblocked, and what would unblock each:\n');
  const byCommand = new Map();
  for (const entry of cases.filter((c) => c.status === BLOCKED)) {
    const key = entry.unblockCommand ?? '(no command recorded)';
    byCommand.set(key, (byCommand.get(key) ?? 0) + 1);
  }
  for (const [command, count] of byCommand) {
    process.stdout.write(`  ${String(count).padStart(3)}x  ${command}\n`);
  }
}

// A failure is a failure. Blocked cases do not fail the run — they are the
// honest state of a machine without the prerequisite — but they are printed
// above so nobody reads a zero exit as completeness.
process.exit(totals.failed > 0 ? 1 : 0);
