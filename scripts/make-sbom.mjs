#!/usr/bin/env node
/**
 * Software bill of materials for the whole workbench.
 *
 * ## Why this exists
 *
 * PS 26117 asks for a system a public-sector organisation can actually deploy,
 * and the question a PSU's procurement asks first is not "does it work" but
 * "what is in it, and under what licence". A product whose answer is "a
 * thousand npm packages and some crates" does not get installed.
 *
 * There is a second reason, specific to this build. ARJUN vendors a pruned copy
 * of a large upstream project. The claim that the pruning removed every cloud
 * provider is only as good as the ability to enumerate what remains — so the
 * SBOM and `scripts/check-bundle.mjs` answer the same question from opposite
 * ends: what was declared, and what actually shipped.
 *
 * ## What it covers, and what it deliberately does not
 *
 * Three ecosystems, because ARJUN is three runtimes: Rust crates from
 * `Cargo.lock`, Node packages from both lockfiles, and the vendored OpenClaw
 * copy recorded as one component pinned to a commit.
 *
 * Python is **not** covered. The document sidecar's dependencies are installed
 * by the deployment, not by this repository, and inventing an entry for
 * something whose version is decided elsewhere would make the document less
 * trustworthy rather than more complete. The gap is stated in the output.
 *
 * ## Format
 *
 * CycloneDX 1.5 JSON, which is what most procurement tooling reads, plus a
 * short Markdown summary a person can actually skim. Generated rather than
 * maintained: a hand-written inventory is wrong the day after it is written.
 */

import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = join(fileURLToPath(new URL('.', import.meta.url)), '..');
const OUT_DIR = join(ROOT, 'evidence');
const OUT_JSON = join(OUT_DIR, 'sbom.cdx.json');
const OUT_MD = join(OUT_DIR, 'sbom.md');

/** Reads JSON, or returns undefined rather than throwing on an absent file. */
function readJson(path) {
  return existsSync(path) ? JSON.parse(readFileSync(path, 'utf8')) : undefined;
}

const components = [];
const notes = [];

// --- Rust ------------------------------------------------------------------
// Cargo.lock is TOML. Parsed with a narrow line reader rather than a
// dependency: the file's `[[package]]` blocks are a fixed shape, and adding a
// TOML parser to generate an inventory of dependencies is its own small irony.
const cargoLock = join(ROOT, 'src-tauri', 'Cargo.lock');
if (existsSync(cargoLock)) {
  let current;
  for (const line of readFileSync(cargoLock, 'utf8').split('\n')) {
    const trimmed = line.trim();
    if (trimmed === '[[package]]') {
      if (current?.name) components.push(current);
      current = { type: 'library', ecosystem: 'cargo' };
      continue;
    }
    if (!current) continue;
    const match = /^(name|version|source|checksum)\s*=\s*"(.*)"$/.exec(trimmed);
    if (match) current[match[1]] = match[2];
  }
  if (current?.name) components.push(current);
} else {
  notes.push('Cargo.lock was not found, so no Rust crates are listed.');
}

// --- Node ------------------------------------------------------------------
for (const [label, lockPath] of [
  ['frontend', join(ROOT, 'package-lock.json')],
  ['agent-runtime', join(ROOT, 'agent-runtime', 'package-lock.json')],
]) {
  const lock = readJson(lockPath);
  if (!lock) {
    notes.push(`${lockPath} was not found, so its Node packages are not listed.`);
    continue;
  }
  for (const [path, entry] of Object.entries(lock.packages ?? {})) {
    // The root entry describes this repository, not a dependency.
    if (!path) continue;
    const name = entry.name ?? path.split('node_modules/').pop();
    if (!name) continue;
    // Vendored workspace members are ARJUN's own copy of OpenClaw, recorded
    // below as one pinned component. npm records them twice — once at their
    // real path and once as a link under `node_modules/@openclaw/*` — and both
    // forms have to be skipped or the inventory lists ten private packages with
    // no version, which is worse than not listing them.
    if (path.includes('vendor/openclaw') || name.startsWith('@openclaw/')) continue;
    // A dependency with no resolved version cannot be reproduced, so it is
    // reported as a gap rather than written into the document as "unknown".
    if (!entry.version) {
      notes.push(`${name} (${label}) has no version in ${lockPath}, so it is not listed.`);
      continue;
    }
    components.push({
      type: 'library',
      ecosystem: 'npm',
      name,
      version: entry.version,
      license: typeof entry.license === 'string' ? entry.license : undefined,
      checksum: entry.integrity,
      scope: label,
    });
  }
}

// --- The vendored copy -----------------------------------------------------
// One component, not many. It is a pruned snapshot of a single upstream project
// at a single commit, and describing it as its constituent packages would imply
// they can be updated independently. They cannot: they move when the pin moves.
const vendorReadme = join(ROOT, 'agent-runtime', 'vendor', 'README.md');
if (existsSync(vendorReadme)) {
  const text = readFileSync(vendorReadme, 'utf8');
  const commit = /`([0-9a-f]{40})`/.exec(text)?.[1];
  components.push({
    type: 'library',
    ecosystem: 'vendored',
    name: 'openclaw/openclaw (pruned)',
    version: commit ?? 'unpinned',
    license: 'MIT',
    source: 'https://github.com/openclaw/openclaw', // arjun-egress-ok: provenance text in the SBOM; nothing dereferences it
  });
  if (!commit) {
    notes.push(
      'The vendored OpenClaw copy has no commit recorded in agent-runtime/vendor/README.md, ' +
        'so it cannot be reproduced exactly. Pin it.',
    );
  }
} else {
  notes.push('agent-runtime/vendor/README.md was not found, so the vendored copy is unrecorded.');
}

// --- The shipped artifact --------------------------------------------------
// Hashed so the SBOM and the thing it describes can be tied together. Without
// this the document says what was declared; with it, what was built.
const bundle = join(ROOT, 'agent-runtime', 'dist', 'arjun-agent-runtime.mjs');
let bundleDigest;
if (existsSync(bundle)) {
  bundleDigest = createHash('sha256').update(readFileSync(bundle)).digest('hex');
} else {
  notes.push('The agent runtime bundle was not built, so the SBOM does not name what shipped.');
}

// Python is out of scope, and saying so is part of the document.
notes.push(
  "The document sidecar's Python dependencies are installed by the deployment rather than by " +
    'this repository, so they are not listed here. They belong in the deployment record.',
);
notes.push(
  'The same is true of the graph sidecar (sidecars/graph_sidecar), stated separately because ' +
    'the dependency is far heavier: it pins torch and transformers in its own ' +
    'requirements.txt. An air-gapped install must vendor those wheels; nothing in this ' +
    'repository fetches them.',
);
// The licence is a deployment blocker, not a footnote, so it is in the document
// rather than only on the registry entry.
notes.push(
  'The relation model that sidecar loads, Babelscape/rebel-large, is CC BY-NC-SA 4.0 - ' +
    'non-commercial. It is not redistributed here and is not in the installer; a deployment ' +
    'downloads it deliberately, and ModelEntry::license_allowed gates its use. A commercial ' +
    'distribution of ARJUN cannot ship or rely on it.',
);

// --- Write -----------------------------------------------------------------
mkdirSync(OUT_DIR, { recursive: true });

const cyclonedx = {
  bomFormat: 'CycloneDX',
  specVersion: '1.5',
  version: 1,
  metadata: {
    // No timestamp: a document that differs on every run cannot be diffed, and
    // the interesting question is whether the inventory changed, not when it
    // was printed. The bundle hash is what ties it to a build.
    component: {
      type: 'application',
      name: 'ARJUN',
      version: readJson(join(ROOT, 'package.json'))?.version ?? '0.0.0',
      description: 'Sovereign local-model industrial workbench (SIH PS 26117)',
      ...(bundleDigest
        ? { hashes: [{ alg: 'SHA-256', content: bundleDigest }] }
        : {}),
    },
  },
  components: components
    .map((component) => ({
      type: component.type,
      name: component.name,
      version: component.version ?? 'unknown',
      ...(component.license ? { licenses: [{ license: { id: component.license } }] } : {}),
      ...(component.checksum && component.ecosystem === 'cargo'
        ? { hashes: [{ alg: 'SHA-256', content: component.checksum }] }
        : {}),
      purl:
        component.ecosystem === 'cargo'
          ? `pkg:cargo/${component.name}@${component.version}`
          : component.ecosystem === 'npm'
            ? `pkg:npm/${component.name}@${component.version}`
            : `pkg:generic/${component.name}@${component.version}`,
    }))
    .sort((a, b) => a.purl.localeCompare(b.purl)),
};

writeFileSync(OUT_JSON, `${JSON.stringify(cyclonedx, null, 2)}\n`);

const byEcosystem = new Map();
for (const component of components) {
  byEcosystem.set(component.ecosystem, (byEcosystem.get(component.ecosystem) ?? 0) + 1);
}

const markdown = [
  '# ARJUN — software bill of materials',
  '',
  'Generated by `scripts/make-sbom.mjs`. Do not edit: a hand-maintained inventory is wrong',
  'the day after it is written.',
  '',
  '| Ecosystem | Components |',
  '| --- | ---: |',
  ...[...byEcosystem.entries()].map(([name, count]) => `| ${name} | ${count} |`),
  `| **total** | **${components.length}** |`,
  '',
  '## What shipped',
  '',
  bundleDigest
    ? `Agent runtime bundle \`agent-runtime/dist/arjun-agent-runtime.mjs\`\n\nSHA-256 \`${bundleDigest}\``
    : 'The agent runtime bundle was not built when this was generated.',
  '',
  '## Gaps, stated rather than hidden',
  '',
  ...notes.map((note) => `- ${note}`),
  '',
  '## Related evidence',
  '',
  '- `scripts/check-bundle.mjs` — what is actually in the shipped runtime',
  '- `scripts/check-egress.mjs` — one outbound chokepoint in the source',
  '- `scripts/check-offline-build.mjs` — the build needs no network',
  '- `agent-runtime/scripts/audit-vendor.mjs` — what the vendored copy may contain',
  '',
].join('\n');

writeFileSync(OUT_MD, markdown);

console.log(
  `sbom: wrote ${components.length} component(s) to evidence/sbom.cdx.json and evidence/sbom.md`,
);
for (const note of notes) console.log(`  note: ${note}`);
