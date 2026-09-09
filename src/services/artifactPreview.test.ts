/**
 * A preview pane draws what the backend actually sent.
 *
 * ## What these tests are guarding
 *
 * `artifact_preview.rs` sends `{ kind, text, truncated, sizeBytes }`. The
 * surface declared `{ kind, mime, content | dataUrl | reason, truncated }` and
 * read `preview.content` and `preview.dataUrl` — names that have never been on
 * the wire. Every preview therefore opened blank: the call succeeded, the
 * spinner cleared, and an empty `<pre>` appeared.
 *
 * The `kind` strings matched, which is why the lane looked healthy. Only the
 * payload had drifted, and only a test that reads the Rust struct can tell.
 */
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';

import {
  PREVIEW_KINDS,
  previewDisplay,
  type ArtifactPreview,
  type PreviewKind,
} from './artifactPreview';

const PREVIEW_RS = 'src-tauri/src/commands/artifact_preview.rs';

function camel(variant: string): string {
  return variant.charAt(0).toLowerCase() + variant.slice(1);
}

/** The variants of `pub enum PreviewKind`, as serde spells them on the wire. */
function rustPreviewKinds(): string[] {
  const source = readFileSync(PREVIEW_RS, 'utf8');
  const block = source.match(/pub enum PreviewKind \{([\s\S]*?)\n\}/);
  if (!block) throw new Error(`could not find 'pub enum PreviewKind' in ${PREVIEW_RS}`);
  return block[1]
    .split('\n')
    .map(line => line.trim())
    .filter(line => /^[A-Z][A-Za-z0-9]*,$/.test(line))
    .map(line => camel(line.slice(0, -1)));
}

/** The fields of `pub struct ArtifactPreview`, as serde spells them. */
function rustPreviewFields(): string[] {
  const source = readFileSync(PREVIEW_RS, 'utf8');
  const block = source.match(/pub struct ArtifactPreview \{([\s\S]*?)\n\}/);
  if (!block) throw new Error(`could not find 'pub struct ArtifactPreview' in ${PREVIEW_RS}`);
  return [...block[1].matchAll(/pub (\w+):/g)].map(match =>
    match[1].replace(/_([a-z])/g, (_, c: string) => c.toUpperCase()),
  );
}

function preview(over: Partial<ArtifactPreview> = {}): ArtifactPreview {
  return { kind: 'text', text: 'hello world', truncated: false, sizeBytes: 11, ...over };
}

describe('artifact previews: the surface reads the fields Rust sends', () => {
  it('declares exactly the fields the Rust struct has', () => {
    // The defect, as one assertion. `content`, `dataUrl`, `reason` and `mime`
    // were read; none of them is here.
    const fromRust = rustPreviewFields().sort();
    expect(fromRust).toEqual(['kind', 'sizeBytes', 'text', 'truncated']);

    // And the module's own type is built from those names: a body read from
    // anywhere else would be undefined at runtime.
    const sample = preview();
    for (const field of fromRust) {
      expect(Object.keys(sample), field).toContain(field);
    }
  });

  it('declares exactly the kinds the Rust enum has', () => {
    expect([...PREVIEW_KINDS].sort()).toEqual(rustPreviewKinds().sort());
  });

  it('draws the body of every kind Rust can send', () => {
    // Not one of them may come back empty-handed for a preview that has text.
    for (const kind of rustPreviewKinds()) {
      if (kind === 'unsupported' || kind === 'image') continue;
      const display = previewDisplay(preview({ kind: kind as PreviewKind }));
      expect(display.layout, kind).toBe('body');
      if (display.layout === 'body') {
        expect(display.body, kind).toBe('hello world');
      }
    }
  });

  it('draws an image from the data URL Rust puts in text', () => {
    const display = previewDisplay(
      preview({ kind: 'image', text: 'data:image/png;base64,iVBORw0KGgo=' }),
    );
    expect(display.layout).toBe('image');
    if (display.layout === 'image') {
      expect(display.src).toBe('data:image/png;base64,iVBORw0KGgo=');
    }
  });

  it('refuses to draw an image that is not a data URL', () => {
    // An <img> with an empty or relative src resolves against the page and
    // draws nothing — the same blank box this module exists to remove.
    const display = previewDisplay(preview({ kind: 'image', text: '' }));
    expect(display.layout).toBe('notice');
  });

  it('says so in words when a file previews to nothing', () => {
    // An empty <pre> and a failed preview look identical to a reader, so the
    // empty case is never drawn as an empty box.
    const display = previewDisplay(preview({ text: '   ' }));
    expect(display.layout).toBe('notice');
    if (display.layout === 'notice') {
      expect(display.message).toMatch(/no text/i);
    }
  });

  it('explains an unsupported format instead of showing an empty pane', () => {
    // Rust sends an empty `text` for this kind, so the old pane rendered a
    // blank box with no explanation.
    const display = previewDisplay(preview({ kind: 'unsupported', text: '' }));
    expect(display.layout).toBe('notice');
    if (display.layout === 'notice') {
      expect(display.message).toMatch(/file manager/i);
    }
  });

  it('mentions truncation only when the backend reported it', () => {
    const cut = previewDisplay(preview({ truncated: true }));
    expect(cut.layout === 'body' && cut.note).toMatch(/truncated/i);

    const whole = previewDisplay(preview({ truncated: false }));
    expect(whole.layout === 'body' && whole.note).toBeNull();
  });

  it('draws extracted and tabular bodies monospaced, prose not', () => {
    // A .docx body and a sheet rendered as a table lose their meaning in a
    // proportional font; markdown and slide lists read as prose.
    for (const kind of ['text', 'docxBody', 'xlsxFirstSheet'] as const) {
      const display = previewDisplay(preview({ kind }));
      expect(display.layout === 'body' && display.mono, kind).toBe(true);
    }
    for (const kind of ['markdown', 'pptxSlideList'] as const) {
      const display = previewDisplay(preview({ kind }));
      expect(display.layout === 'body' && display.mono, kind).toBe(false);
    }
  });
});
