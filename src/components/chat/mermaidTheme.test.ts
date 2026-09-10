import { describe, expect, it } from 'vitest';

import {
  mermaidConfig,
  mermaidTheme,
  readDiagramTokens,
  type DiagramTokens,
} from './mermaidTheme';

/**
 * What these tests are pinning.
 *
 * Both of the defects covered here were found by rendering diagrams in a
 * browser rather than by reading the code, and both were silent in the way that
 * matters: the first stopped every diagram on the page from drawing at all, and
 * the second drew all of them in colours nobody could see. Neither would have
 * shown up in a type check.
 *
 * The suite is pure — vitest runs here with `environment: 'node'` — so it covers
 * the mapping from tokens to Mermaid variables and nothing about rendering.
 * `sanitizeDiagramSvg` needs a `DOMParser` and so is not testable in this
 * harness; it is exercised against real Mermaid output in a browser instead.
 */

/** The dark theme's tokens, exactly as `index.css` declares them. */
const DARK: DiagramTokens = {
  background: '#000000',
  surface: '#0B0B0B',
  surfaceRaised: '#141414',
  textPrimary: 'rgba(250, 250, 250, 0.92)',
  textSecondary: 'rgba(212, 212, 216, 0.72)',
  border: '#262626',
  accent: '#FAFAFA',
  accentForeground: '#000000',
  fontFamily: 'Segoe UI, sans-serif',
  fontMono: 'Cascadia Mono, monospace',
};

/** The light theme's, likewise. */
const LIGHT: DiagramTokens = {
  background: '#FFFFFF',
  surface: '#FAFAFA',
  surfaceRaised: '#F4F4F4',
  textPrimary: '#0A0A0A',
  textSecondary: '#5C5C5C',
  border: '#E4E4E4',
  accent: '#0A0A0A',
  accentForeground: '#FFFFFF',
  fontFamily: 'Segoe UI, sans-serif',
  fontMono: 'Cascadia Mono, monospace',
};

const HEX = /^#[0-9a-f]{6}$/;

function channels(hex: string): [number, number, number] {
  return [
    parseInt(hex.slice(1, 3), 16),
    parseInt(hex.slice(3, 5), 16),
    parseInt(hex.slice(5, 7), 16),
  ];
}

/** Perceived lightness, near enough to compare two greys by. */
function luminance(hex: string): number {
  const [r, g, b] = channels(hex);
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

/** Every variable that names a colour, rather than a font or a number. */
function colourValues(variables: Record<string, string>): [string, string][] {
  return Object.entries(variables).filter(
    ([name]) => name !== 'fontFamily' && name !== 'fontSize' && name !== 'pieOpacity',
  );
}

describe('mermaidTheme', () => {
  /**
   * The defect: shading a token with `color-mix(in srgb, …)` and letting the
   * browser resolve it. Mermaid does not hand these to the stylesheet
   * untouched — it parses each one with `khroma` to derive further shades, and
   * throws `Unsupported color format` on anything that is not a plain colour,
   * taking every diagram on the page down with it.
   */
  it.each([
    ['dark', DARK],
    ['light', LIGHT],
  ])('gives Mermaid plain hex in the %s theme, never a CSS function', (_name, tokens) => {
    for (const [variable, value] of colourValues(mermaidTheme(tokens))) {
      expect(value, `${variable} must be plain hex`).toMatch(HEX);
    }
  });

  /**
   * The palette is monochrome by rule; see the note at `index.css:117`. Mermaid's
   * `base` theme is not, and a single variable left unset comes back coloured —
   * one unset `noteBkgColor` is a yellow sticky note in the middle of an
   * otherwise greyscale sequence diagram.
   */
  it.each([
    ['dark', DARK],
    ['light', LIGHT],
  ])('stays monochrome in the %s theme', (_name, tokens) => {
    /*
      Near-neutral rather than exactly grey, and the tolerance is not slack:
      `--text-secondary` is declared `rgba(212, 212, 216, 0.72)`, so the design
      token itself carries four steps of blue. Asserting a zero spread would be
      asserting something untrue about `index.css`, and the honest check is that
      nothing here *adds* a hue — a derived colour that drifted would show up as
      a spread far wider than the tokens it came from.
    */
    for (const [variable, value] of colourValues(mermaidTheme(tokens))) {
      const rgb = channels(value);
      const spread = Math.max(...rgb) - Math.min(...rgb);
      expect(spread, `${variable} (${value}) carries a hue`).toBeLessThanOrEqual(6);
    }
  });

  /**
   * The second defect. Mapping node fill to `--bg-tertiary` and the outline to
   * `--border-default` looks obviously right and is unreadable: in the dark
   * theme the fill would be nine steps out of 255 from its ground, which on a
   * one-pixel outline is a flowchart of invisible boxes.
   */
  it.each([
    ['dark', DARK],
    ['light', LIGHT],
  ])('separates node, outline and ground in the %s theme', (_name, tokens) => {
    const variables = mermaidTheme(tokens);
    const ground = luminance(variables.background);
    const fill = luminance(variables.mainBkg);
    const stroke = luminance(variables.nodeBorder);
    const line = luminance(variables.lineColor);

    expect(Math.abs(fill - ground)).toBeGreaterThan(8);
    expect(Math.abs(stroke - fill)).toBeGreaterThan(20);
    // An edge has to be findable against the ground it crosses.
    expect(Math.abs(line - ground)).toBeGreaterThan(60);
  });

  it.each([
    ['dark', DARK],
    ['light', LIGHT],
  ])('keeps label text legible on its node in the %s theme', (_name, tokens) => {
    const variables = mermaidTheme(tokens);
    expect(
      Math.abs(luminance(variables.nodeTextColor) - luminance(variables.mainBkg)),
    ).toBeGreaterThan(100);
  });

  /**
   * Mermaid reads exactly `pie1` through `pie12` and falls back to its own
   * categorical ramp for any left unset, so eleven greys and a magenta is the
   * failure mode. Neighbours are checked apart because a pie is the one place
   * here where a reader separates two regions with no word attached.
   */
  it('sets all twelve pie slices, each distinguishable from its neighbour', () => {
    const variables = mermaidTheme(DARK);
    const slices = Array.from({ length: 12 }, (_, index) => variables[`pie${index + 1}`]);

    for (const [index, slice] of slices.entries()) {
      expect(slice, `pie${index + 1} unset`).toMatch(HEX);
    }
    for (let index = 1; index < slices.length; index += 1) {
      expect(
        Math.abs(luminance(slices[index]) - luminance(slices[index - 1])),
        `pie${index} and pie${index + 1} are too close`,
      ).toBeGreaterThan(40);
    }
  });

  it('flattens a translucent token against the diagram ground', () => {
    // `rgba(250, 250, 250, 0.92)` over `#000000` is 230, not 250: alpha inside
    // an SVG would otherwise resolve against whatever sits behind each part of
    // the drawing, rendering one token as two different greys.
    expect(mermaidTheme(DARK).nodeTextColor).toBe('#e6e6e6');
  });

  it('carries the theme fonts through rather than Mermaid’s own', () => {
    expect(mermaidTheme(DARK).fontFamily).toBe(DARK.fontFamily);
  });
});

describe('mermaidConfig', () => {
  /**
   * With HTML labels on, every node label is a `<foreignObject>` holding real
   * `<div>` elements, and `sanitizeDiagramSvg` would have to admit arbitrary
   * HTML into a chat message for a flowchart to draw at all.
   *
   * The top-level flag is the one that works. Setting only the per-diagram flags
   * was measured and left `<foreignObject>` in flowchart, ER, state and class
   * output; both spellings are asserted so neither is dropped later.
   */
  it('turns HTML labels off at every level Mermaid reads them', () => {
    const config = mermaidConfig(DARK);
    expect(config.htmlLabels).toBe(false);
    expect(config.flowchart.htmlLabels).toBe(false);
    expect(config.er.htmlLabels).toBe(false);
    expect(config.state.htmlLabels).toBe(false);
    expect(config.class.htmlLabels).toBe(false);
  });

  it('renders nothing on its own and paints no error graphic', () => {
    const config = mermaidConfig(DARK);
    // `startOnLoad` would have Mermaid scan the document itself, outside React.
    expect(config.startOnLoad).toBe(false);
    // Otherwise a syntax error puts Mermaid's "bomb" graphic into
    // `document.body`, where React will never remove it.
    expect(config.suppressErrorRendering).toBe(true);
    expect(config.securityLevel).toBe('strict');
  });

  it('overrides the base theme rather than picking a named one', () => {
    // Every named Mermaid theme is coloured; `base` exists to be overridden.
    expect(mermaidConfig(DARK).theme).toBe('base');
  });
});

describe('readDiagramTokens', () => {
  it('falls back to a drawable palette with no document to read', () => {
    // vitest runs with `environment: 'node'`, so there is no `getComputedStyle`
    // here — the same position a diagram is in before the theme has been
    // applied. Empty strings would have Mermaid paint black on black, which on
    // screen is indistinguishable from a diagram that failed to render.
    const tokens = readDiagramTokens(null);
    for (const [name, value] of Object.entries(tokens)) {
      expect(value, `${name} is empty`).not.toBe('');
    }
    expect(mermaidTheme(tokens).background).toMatch(HEX);
  });
});
