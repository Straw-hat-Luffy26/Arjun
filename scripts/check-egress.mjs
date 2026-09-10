#!/usr/bin/env node
/**
 * Egress gate — fails the build if ARJUN grows a way to reach the network that
 * does not go through the broker.
 *
 * PS 26117 asks for proof that no external call is made at any point. Proving
 * that by inspection at demo time is not possible; proving it about the *source*
 * is. Two rules do the work:
 *
 *   1. Exactly one module may construct an outbound HTTP client. Anything else
 *      that does is a second, unaudited chokepoint.
 *   2. Every external host that appears anywhere in the tree must be on a list
 *      that a human has agreed to.
 *
 * This is the cheapest and strongest of the five proof layers in the plan,
 * because it shows there is no code that *could* call out — a stronger claim
 * than a packet capture showing that nothing did.
 *
 * A line may opt out with a trailing `arjun-egress-ok: <reason>` comment. The
 * reason is required, so every exemption is self-documenting in review.
 */

import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join, relative, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = join(fileURLToPath(new URL('.', import.meta.url)), '..');

/** The one file permitted to construct an outbound client. */
const BROKER_FILE = join('src-tauri', 'src', 'sovereignty', 'broker.rs');

/**
 * Files that construct a client which can only ever reach loopback. These are
 * not egress, but the pattern is indistinguishable from egress by inspection,
 * so each is listed with a reason rather than silently pattern-matched away.
 *
 * A reviewer should confirm the URL really is loopback before adding one.
 */
const LOOPBACK_CLIENT_FILES = new Map([
  [
    'src-tauri/src/system_analyzer/ai_runtime_collector.rs',
    'probes http://127.0.0.1:11434 to detect a local Ollama install',
  ],
  [
    'src-tauri/src/commands/ocr.rs',
    'streams one page to the llama-server ARJUN itself started; calls '
      + 'serving::probe::check_loopback on the endpoint and refuses to send the '
      + 'document at all if the base URL is not loopback',
  ],
  [
    'src-tauri/src/serving/probe.rs',
    'probes local inference servers (llama-server, vLLM, SGLang); refuses any URL '
      + 'whose host does not parse as a loopback address before the client is built',
  ],
  [
    'src-tauri/src/knowledge/embedding.rs',
    'asks a local embedding model for one vector per passage; calls '
      + 'serving::probe::check_loopback on the base URL and returns an error '
      + 'before the client is constructed if the host is not loopback, so the '
      + 'client cannot be built for a remote endpoint at all',
  ],
  [
    'src-tauri/src/ai_engine/vision_bridge.rs',
    'speaks the OpenAI vision schema to a local vLLM / llama.cpp server; the constructor '
      + 'parses base_url and refuses any host that is not localhost, 127.0.0.1 or ::1 '
      + 'before the client is built, so the client can only ever address loopback',
  ],
]);

/**
 * Files whose host literals are a *ledger of what is refused*, not destinations.
 *
 * The bundle gate has to name the vendor endpoints it looks for, and its
 * self-test has to reintroduce them to prove the gate catches them. Those hosts
 * appear in the source for the same reason a deny-list appears in a firewall
 * config: to be matched against, never to be dialled.
 *
 * Listed rather than pattern-matched away, and only for rule 2 — the
 * one-chokepoint rule still applies to these files, so neither may build a
 * client. A reviewer should confirm a file really is a ledger before adding one.
 */
const HOST_LEDGER_FILES = new Map([
  [
    'scripts/check-bundle.mjs',
    'the reviewed ledger of vendor hosts the shipped bundle may contain, and the exclusion list it scans for',
  ],
  [
    'scripts/check-bundle.test.mjs',
    'reintroduces each excluded host into a throwaway copy to prove the bundle gate catches it',
  ],
]);

/** Directories never worth scanning. */
const SKIP_DIRS = new Set([
  'node_modules', 'target', 'dist', '.git', 'gen', 'vendor', '__pycache__', 'scratch',
]);

const SCAN_EXTENSIONS = new Set(['.rs', '.ts', '.tsx', '.js', '.jsx', '.mjs', '.html', '.css', '.json']);

/**
 * Hosts allowed to appear as literals. Each needs a reason: this list is the
 * product's entire external surface, and it should be uncomfortable to extend.
 */
const ALLOWED_HOSTS = new Map([
  ['huggingface.co',            'model catalog and weights, Provisioning mode only'],
  ['cdn-lfs.huggingface.co',    'weight blobs for the above'],
  ['cdn-lfs-us-1.huggingface.co', 'weight blobs for the above'],
  ['127.0.0.1',                 'loopback'],
  ['localhost',                 'loopback'],
  ['ipc.localhost',             'Tauri IPC, never leaves the process'],
  ['asset.localhost',           'Tauri asset protocol, local files'],
  ['widget.localhost',          'Tauri custom protocol for sandboxed widgets, served in-process'],
  ['tauri.localhost',           'Tauri webview origin'],
  ['schema.tauri.app',          'JSON-schema reference string, never fetched'],
  ['www.w3.org',                'SVG/XML namespace identifier, never fetched'],
  ['react.dev',                 'React error-message doc link, never fetched'],
  ['reactrouter.com',           'React Router error-message doc link, never fetched'],
  ['example.invalid',           'the egress canary target, which must never resolve'],
]);

/** Ways to construct an outbound HTTP client, or fire a one-shot request. */
const CLIENT_PATTERNS = [
  /reqwest::Client::(?:new|builder)\s*\(/,
  /reqwest::blocking::Client::(?:new|builder)\s*\(/,
  /reqwest::(?:blocking::)?get\s*\(/,
  /\bfetch\s*\(/,
  /new\s+XMLHttpRequest\b/,
  /new\s+WebSocket\b/,
];

// A trailing dot is sentence punctuation, not part of the host.
const URL_PATTERN = /\bhttps?:\/\/([a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)*)/g;
const OPT_OUT = /arjun-egress-ok:\s*\S+/;

/**
 * An OOXML namespace URI is a name, not an address.
 *
 * A .docx, .xlsx and .pptx identify their own schemas with
 * `schemas.openxmlformats.org/...` URIs (scheme omitted here so this comment
 * does not itself read as a call), written into the file as
 * `xmlns="..."` or as a relationship `Type="..."`. Nothing resolves them —
 * they are compared as strings by Word, Excel and PowerPoint. Adding the host
 * to ALLOWED_HOSTS would be the wrong fix, because that list says "we may call
 * this", and we may not.
 *
 * So the exemption is structural rather than by host: a URL is exempt only when
 * it sits immediately after `xmlns=`, `xmlns:prefix=` or `Type=`. A genuine
 * fetch of the same host anywhere else still fails the gate.
 */
const NAMESPACE_ATTRIBUTE = /(?:xmlns(?::[A-Za-z0-9_-]+)?|Type)="$/;

function isNamespaceDeclaration(line, index) {
  return NAMESPACE_ATTRIBUTE.test(line.slice(0, index).replace(/\\/g, ""));
}

/**
 * Rust test modules are exempt. Negative fixtures have to name hosts that must
 * be refused (`evil.test`), and asserting on them is the point — flagging those
 * would push people to delete the very tests that prove the gate works.
 *
 * Simplification: everything from the first `#[cfg(test)]` to end-of-file is
 * treated as test code. That holds for this codebase, where tests sit in a
 * trailing `mod tests`, and it fails safe — a stray `#[cfg(test)]` mid-file
 * would hide later lines, so the marker is required to be at column zero.
 */
function testRegionStart(lines) {
  const idx = lines.findIndex(l => l.startsWith('#[cfg(test)]'));
  return idx === -1 ? Number.POSITIVE_INFINITY : idx;
}

const failures = [];

function walk(dir) {
  for (const entry of readdirSync(dir)) {
    if (SKIP_DIRS.has(entry)) continue;
    const full = join(dir, entry);
    const stat = statSync(full);
    if (stat.isDirectory()) {
      walk(full);
    } else if (SCAN_EXTENSIONS.has(entry.slice(entry.lastIndexOf('.')))) {
      inspect(full);
    }
  }
}

function inspect(file) {
  // Normalise to forward slashes so the lookups below behave the same on Windows.
  const rel = relative(ROOT, file).split(sep).join('/');
  const isBroker = rel === BROKER_FILE.split(sep).join('/');
  const loopbackOnly = LOOPBACK_CLIENT_FILES.has(rel);
  // Rule 2 only. These files must still not build a client.
  const hostLedger = HOST_LEDGER_FILES.has(rel);
  const lines = readFileSync(file, 'utf8').split(/\r?\n/);
  const testsFrom = testRegionStart(lines);

  lines.forEach((line, i) => {
    if (OPT_OUT.test(line)) return;
    if (i >= testsFrom) return;
    const at = `${rel}:${i + 1}`;

    // Rule 1 — only the broker may build an outbound client.
    if (!isBroker && !loopbackOnly) {
      for (const pattern of CLIENT_PATTERNS) {
        if (pattern.test(line)) {
          failures.push({
            at,
            rule: 'second chokepoint',
            detail: `constructs an HTTP client outside the broker: ${line.trim()}`,
          });
          break;
        }
      }
    }

    // Rule 2 — every host literal must be on the agreed list.
    if (hostLedger) return;
    for (const match of line.matchAll(URL_PATTERN)) {
      if (isNamespaceDeclaration(line, match.index)) continue;
      const host = match[1].toLowerCase();
      if (!ALLOWED_HOSTS.has(host)) {
        failures.push({
          at,
          rule: 'unapproved host',
          detail: `${host} is not on the allowlist in scripts/check-egress.mjs`,
        });
      }
    }
  });
}

// `agent-runtime/src` is ARJUN's own runtime code and is held to the same rule
// as the rest. Its `vendor/` and `dist/` are skipped by SKIP_DIRS and covered
// instead by `agent-runtime/scripts/audit-vendor.mjs` (what the vendored copy
// may contain) and `scripts/check-bundle.mjs` (what actually ships) — the one
// chokepoint rule cannot apply to a vendored HTTP transport whose entire job is
// to be one.
for (const dir of ['src', 'src-tauri/src', 'scripts', 'sidecars', 'agent-runtime/src']) {
  try {
    walk(join(ROOT, dir));
  } catch {
    // A directory that does not exist yet is not a failure.
  }
}
for (const file of ['index.html', 'src-tauri/tauri.conf.json', 'package.json']) {
  try {
    inspect(join(ROOT, file));
  } catch {
    // Same.
  }
}

if (failures.length === 0) {
  console.log('egress gate: pass — one chokepoint, no unapproved hosts');
  process.exit(0);
}

console.error(`egress gate: FAIL — ${failures.length} finding(s)\n`);
for (const f of failures) {
  console.error(`  ${f.at}\n    [${f.rule}] ${f.detail}\n`);
}
console.error(
  'Route the call through src-tauri/src/sovereignty/broker.rs, add the host to\n' +
  'ALLOWED_HOSTS with a reason, or annotate the line `arjun-egress-ok: <reason>`.',
);
process.exit(1);
