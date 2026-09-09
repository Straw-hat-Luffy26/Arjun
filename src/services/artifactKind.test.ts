/**
 * The surface can draw every kind of file Rust can produce.
 *
 * ## What these tests are guarding
 *
 * A produced briefing deck crashed the page it was meant to appear on. Rust's
 * `Kind` has four variants; the surface declared three and keyed its icon table
 * on that union, so `ARTIFACT_ICONS['deck']` was `undefined` and `<Icon />` on
 * `undefined` throws. The nearest boundary was the one around the whole page.
 *
 * Nothing could have caught it at compile time: the value arrives from Rust as
 * JSON, so the TypeScript union was an unverified claim about another language.
 * These tests verify the claim by reading the Rust source, which is the only
 * place the answer actually lives.
 */
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';

import {
  ARTIFACT_GLYPHS,
  ARTIFACT_KINDS,
  artifactPresentation,
} from './artifactKind';

const ARTIFACTS_RS = 'src-tauri/src/agent_runtime/artifacts.rs';

/** `Document` -> `document`, `XlsxSheet` -> `xlsxSheet`: serde's camelCase. */
function camel(variant: string): string {
  return variant.charAt(0).toLowerCase() + variant.slice(1);
}

/** The variants of `pub enum Kind` as Rust declares them. */
function rustKinds(): string[] {
  const source = readFileSync(ARTIFACTS_RS, 'utf8');
  const block = source.match(/pub enum Kind \{([\s\S]*?)\n\}/);
  if (!block) throw new Error(`could not find 'pub enum Kind' in ${ARTIFACTS_RS}`);
  return block[1]
    .split('\n')
    .map(line => line.trim())
    .filter(line => /^[A-Z][A-Za-z0-9]*,$/.test(line))
    .map(line => line.slice(0, -1));
}

/** `Kind::label()` in Rust, as a map from wire name to label. */
function rustLabels(): Map<string, string> {
  const source = readFileSync(ARTIFACTS_RS, 'utf8');
  const labels = new Map<string, string>();
  for (const [, variant, label] of source.matchAll(/Kind::(\w+) => "([^"]+)"/g)) {
    labels.set(camel(variant), label);
  }
  return labels;
}

describe('artifact kinds: the surface can draw everything Rust produces', () => {
  it('finds the four kinds in the Rust source', () => {
    // If this fails the parser below is wrong, and every other test in this
    // file would pass for the wrong reason.
    expect(rustKinds()).toEqual(['Document', 'Workbook', 'Deck', 'Text']);
  });

  it('knows every kind Rust can send', () => {
    // The defect, as one assertion. Before the fix `deck` was not in the list
    // and this failed.
    for (const variant of rustKinds()) {
      const kind = camel(variant);
      expect(ARTIFACT_KINDS, kind).toContain(kind);
      expect(artifactPresentation(kind).known, kind).toBe(true);
    }
  });

  it('gives every kind a glyph the icon table has', () => {
    // The crash was a lookup returning undefined. This is that lookup.
    for (const variant of rustKinds()) {
      const { glyph } = artifactPresentation(camel(variant));
      expect(ARTIFACT_GLYPHS, camel(variant)).toContain(glyph);
    }
  });

  it('calls a file what Rust calls it', () => {
    // Two languages describing the same file to the same person should not
    // choose different words for it.
    const labels = rustLabels();
    expect(labels.size).toBe(rustKinds().length);
    for (const [kind, label] of labels) {
      expect(artifactPresentation(kind).label, kind).toBe(label);
    }
  });

  it('draws a kind from a newer backend instead of throwing', () => {
    // The whole point of the shape. An older surface handed a kind that did
    // not exist when it was built must still render a row.
    const unknown = artifactPresentation('hologram');
    expect(unknown.known).toBe(false);
    expect(ARTIFACT_GLYPHS).toContain(unknown.glyph);
    // Honest rather than pretty, the same way `labelForTool` shows the raw
    // name of a tool it does not recognise.
    expect(unknown.label).toBe('hologram');
  });

  it('never returns undefined, for any string at all', () => {
    // Including the ones that break naive lookups: a missing field arriving as
    // an empty string, and the prototype keys an object-literal table answers.
    for (const kind of ['', 'constructor', 'toString', '__proto__', 'deck ', 'DECK']) {
      const presentation = artifactPresentation(kind);
      expect(presentation, kind).toBeDefined();
      expect(ARTIFACT_GLYPHS, kind).toContain(presentation.glyph);
      expect(presentation.label.length, kind).toBeGreaterThan(0);
    }
  });

  it('does not claim to know a kind Rust cannot send', () => {
    // Drift in the other direction: a kind listed here and removed from Rust
    // would be dead code claiming to be supported.
    const fromRust = new Set(rustKinds().map(camel));
    for (const kind of ARTIFACT_KINDS) {
      expect(fromRust, kind).toContain(kind);
    }
  });
});
