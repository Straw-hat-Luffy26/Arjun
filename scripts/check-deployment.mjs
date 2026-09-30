#!/usr/bin/env node
/**
 * Deployment gate — proves the installer contains what the code will go
 * looking for.
 *
 * The failure this exists to catch has a specific shape, and it is invisible to
 * every other check in this repository. A developer adds a sidecar, spawns it
 * with a path computed from the working directory, and tests it. It works — on
 * their machine, where the working directory is the repository and the script
 * is a sibling of the binary. It keeps working in CI, for the same reason. It
 * fails on the first machine that has only the installer, because the file was
 * never in `bundle.resources` and nobody noticed, because nothing ever asked.
 *
 * That is exactly what happened here: `tauri.conf.json` shipped the agent
 * runtime, the skills and the agents, and not one of the three Python
 * sidecars. The unit tests passed. The build passed. A clean Windows
 * installation could not read a PDF.
 *
 * ## What it checks
 *
 * 1. Every dependency the Rust table declares `Bundled` is covered by a
 *    `bundle.resources` entry, so the installer carries it.
 * 2. Every declared bundle path actually exists in the repository, so a typo
 *    in the table cannot pass check 1 by naming a file nobody ships.
 * 3. Every literal program name spawned anywhere in `src-tauri/src` is either
 *    in the table or in the reviewed ledger below.
 *
 * ## The ledger, and why it is a ledger and not a ban
 *
 * `check-bundle.mjs` next door makes the same argument about vendor hostnames,
 * and the reasoning carries over. Some spawns are not ARJUN's to ship:
 * `powershell` and `taskkill` are Windows itself, `open`/`xdg-open`/`explorer`
 * are the platform's "reveal in file manager", and `nvidia-smi`, `rocm-smi`
 * and `ollama` are optional probes whose absence is a supported answer rather
 * than a failure. Banning them outright would mean lying about what the code
 * does. Enumerating them means a *new* one has to be looked at by a person,
 * which is the property worth having.
 */

import { existsSync, readFileSync, readdirSync, statSync } from 'node:fs';
import { join, posix, relative, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = join(fileURLToPath(new URL('.', import.meta.url)), '..');
const TABLE = join(ROOT, 'src-tauri', 'src', 'deployment', 'mod.rs');
const CONFIG = join(ROOT, 'src-tauri', 'tauri.conf.json');
const RUST_SRC = join(ROOT, 'src-tauri', 'src');

/**
 * Program names that may be spawned without appearing in the dependency table.
 *
 * Each entry says why. A name that is not here and not in the table fails the
 * gate — see the module comment for the reasoning.
 */
const LEDGER = new Map([
  ['powershell', 'Windows itself; the system probes shell out to it'],
  ['taskkill', 'Windows itself; kills a child that outlived its timeout'],
  ['explorer', 'Windows "reveal in file manager"'],
  ['open', 'macOS "reveal in Finder"'],
  ['xdg-open', 'Linux "reveal in file manager"'],
  ['nvidia-smi', 'optional GPU probe; absence is a supported answer'],
  ['rocm-smi', 'optional GPU probe; absence is a supported answer'],
  ['ollama', 'optional external runtime probe; absence is a supported answer'],
  ['podman', 'optional sandbox runtime; absence is reported, not fatal'],
  ['docker', 'optional sandbox runtime; absence is reported, not fatal'],
  ['fc-list', 'optional font listing (fontconfig), used only to record which font files a render drew with; absent on Windows, where the versions are reported unrecorded, never passed'],
]);

/** Bundle paths that are build outputs, absent until their build step runs. */
const BUILD_OUTPUTS = new Set(['agent-runtime/dist/arjun-agent-runtime.mjs']);

const failures = [];
const notes = [];

function fail(message) {
  failures.push(message);
}

/**
 * Pulls the dependency table out of the Rust source.
 *
 * Parsing Rust with a regex is normally a bad idea; it is defensible here
 * because the target is a `const` array of struct literals with one field per
 * line, and because a parse that finds nothing is treated as a failure rather
 * than as an empty pass. A refactor that reshapes the table breaks this loudly
 * instead of quietly approving everything.
 */
function readDependencies(source) {
  const start = source.indexOf('pub const DEPENDENCIES');
  if (start === -1) {
    fail(`could not find DEPENDENCIES in ${relative(ROOT, TABLE)}`);
    return [];
  }
  const table = source.slice(start, source.indexOf('\n];', start));
  const entries = [];
  for (const block of table.split(/Dependency\s*\{/).slice(1)) {
    const field = (name) => {
      const match = block.match(new RegExp(`${name}:\\s*([^\\n]+?),\\s*\\n`));
      return match ? match[1].trim() : null;
    };
    const unquote = (raw) => {
      if (!raw || raw === 'None') return null;
      const inner = raw.startsWith('Some(') ? raw.slice(5, -1) : raw;
      const match = inner.match(/^"(.*)"$/s);
      return match ? match[1] : null;
    };
    const id = unquote(field('id'));
    if (!id) continue;
    entries.push({
      id,
      packaging: (field('packaging') ?? '').replace('Packaging::', ''),
      bundlePath: unquote(field('bundle_path')),
      program: unquote(field('program')),
    });
  }
  if (entries.length === 0) {
    fail(
      `parsed zero dependencies from ${relative(ROOT, TABLE)}; the table shape changed ` +
        'and this gate can no longer read it',
    );
  }
  return entries;
}

/**
 * Whether a `bundle.resources` entry covers a repo-relative path.
 *
 * Tauri resource entries are written relative to `src-tauri/`, so they begin
 * `../`. A trailing slash means the whole directory ships, which is what makes
 * a single `../sidecars/document_sidecar/` entry cover both `main.py` and
 * `attachment_extract.py`.
 */
function covers(resource, bundlePath) {
  const normalised = resource.replace(/^\.\.\//, '').split(sep).join(posix.sep);
  if (normalised.endsWith('/')) return bundlePath.startsWith(normalised);
  return normalised === bundlePath;
}

/**
 * Drops whole-line comments before the spawn scan.
 *
 * Without this, a doc comment that *mentions* `Command::new("python")` — and
 * `deployment/mod.rs` has one, explaining what it replaced — reads as a spawn.
 * That would be a gate reporting something it did not observe, which is the
 * failure mode this repository cares about most.
 *
 * Only lines that are entirely a comment are removed. A trailing `//` after
 * code is left alone deliberately: stripping it means deciding whether a `//`
 * sits inside a string literal, and a URL in a string is far more common in
 * this tree than a spawn hidden behind an end-of-line comment.
 */
function withoutCommentLines(source) {
  return source
    .split('\n')
    .filter((line) => !line.trimStart().startsWith('//'))
    .join('\n');
}

/** Every `.rs` file under `src-tauri/src`. */
function rustFiles(dir) {
  const found = [];
  for (const name of readdirSync(dir)) {
    const full = join(dir, name);
    if (statSync(full).isDirectory()) found.push(...rustFiles(full));
    else if (name.endsWith('.rs')) found.push(full);
  }
  return found;
}

// ---------------------------------------------------------------------------

if (!existsSync(TABLE)) {
  fail(`${relative(ROOT, TABLE)} is missing`);
} else if (!existsSync(CONFIG)) {
  fail(`${relative(ROOT, CONFIG)} is missing`);
} else {
  const dependencies = readDependencies(readFileSync(TABLE, 'utf8'));
  const config = JSON.parse(readFileSync(CONFIG, 'utf8'));
  const resources = config.bundle?.resources ?? [];

  if (resources.length === 0) {
    fail('tauri.conf.json declares no bundle.resources at all');
  }

  // 1 + 2. Bundled dependencies must ship, and must exist.
  for (const dep of dependencies) {
    if (dep.packaging !== 'Bundled') continue;
    if (!dep.bundlePath) {
      fail(`${dep.id} is Bundled but declares no bundle_path`);
      continue;
    }

    if (resources.some((resource) => covers(resource, dep.bundlePath))) {
      notes.push(`${dep.id} ships as ${dep.bundlePath}`);
    } else {
      fail(
        `${dep.id} is declared Bundled but no bundle.resources entry covers ` +
          `"${dep.bundlePath}". The installer will not contain it, and the app will ` +
          'fall back to the checkout — which exists only on a developer machine. Add ' +
          `"../${dep.bundlePath}" (or its directory) to bundle.resources in tauri.conf.json.`,
      );
    }

    if (!BUILD_OUTPUTS.has(dep.bundlePath) && !existsSync(join(ROOT, dep.bundlePath))) {
      fail(
        `${dep.id} declares bundle_path "${dep.bundlePath}", which is not in the ` +
          'repository. Either the path is a typo or the file was deleted.',
      );
    }
  }

  // 3. Every literal spawn is accounted for.
  const declared = new Set(dependencies.map((dep) => dep.program).filter(Boolean));
  const spawnPattern = /(?:create_hidden_command|Command::new)\("([^"]+)"\)/g;
  const spawned = new Set();

  for (const file of rustFiles(RUST_SRC)) {
    const source = withoutCommentLines(readFileSync(file, 'utf8'));
    for (const [, program] of source.matchAll(spawnPattern)) {
      spawned.add(program);
      if (declared.has(program) || LEDGER.has(program)) continue;
      fail(
        `${relative(ROOT, file)} spawns "${program}", which is neither in the ` +
          'dependency table nor in this gate\'s reviewed ledger. Add it to DEPENDENCIES ' +
          'in src-tauri/src/deployment/mod.rs so the preflight can report it, or to ' +
          'LEDGER here with the reason it is not ARJUN\'s to ship.',
      );
    }
  }

  // A declared program that no literal spawn names is the *good* state: it
  // means the call site goes through `deployment::program()` and honours the
  // operator's override. Recorded so a reviewer can see which ones do.
  for (const dep of dependencies) {
    if (dep.program && !spawned.has(dep.program)) {
      notes.push(`${dep.id} is spawned through deployment::program(), not a bare literal`);
    }
  }
}

// ---------------------------------------------------------------------------

for (const note of notes) console.log(`  ok  ${note}`);

if (failures.length > 0) {
  console.error(`\nDeployment gate FAILED with ${failures.length} problem(s):\n`);
  for (const failure of failures) console.error(`  x  ${failure}\n`);
  console.error(
    'Each of these means an installed build would look for something the installer ' +
      'does not contain.\n',
  );
  process.exit(1);
}

console.log('\nDeployment gate passed: everything the code resolves, the installer ships.');
