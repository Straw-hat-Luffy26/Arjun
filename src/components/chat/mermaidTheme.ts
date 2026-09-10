/**
 * Dressing Mermaid in ARJUN's palette.
 *
 * ## Why this is not just `theme: 'dark'`
 *
 * Mermaid ships named themes — `default`, `dark`, `forest`, `neutral` — and
 * every one of them is coloured. `default` draws flowchart nodes in lavender on
 * cream; a `pie` chart gets a twelve-hue categorical ramp. Dropping any of them
 * into this application would put more hues on one screen than the rest of the
 * product uses in total.
 *
 * The palette here is monochrome by rule (see the note at `index.css:117`):
 * status is carried by luminance, and only three accent hues exist anywhere,
 * each always paired with a word. A rainbow pie chart is not a small
 * inconsistency against that — it is the one rule the design has.
 *
 * So this uses Mermaid's `base` theme, which exists precisely to be overridden,
 * and derives every variable from the CSS custom properties the rest of the
 * surface already styles with. A diagram then follows the light/dark switch for
 * free, because the tokens it was built from are the ones that changed.
 *
 * ## Why the tokens are read rather than copied
 *
 * `--bg-primary` and friends are declared under `[data-theme="dark"]` and
 * `[data-theme="light"]` in `index.css`, so their values only exist at runtime,
 * on the live document. Hard-coding a copy here would be a second palette that
 * drifts silently from the first — the failure `toolNames.ts` and
 * `artifactKind.ts` both have headers about.
 *
 * [`readDiagramTokens`] does the reading and is the only part that needs a DOM.
 * [`mermaidTheme`] and [`mermaidConfig`] are pure, so vitest — which runs here
 * with `environment: 'node'` — can test the mapping without one.
 */

/** The tokens a diagram is drawn from, as read off the live document. */
export interface DiagramTokens {
  background: string;
  surface: string;
  surfaceRaised: string;
  textPrimary: string;
  textSecondary: string;
  border: string;
  accent: string;
  accentForeground: string;
  fontFamily: string;
  fontMono: string;
}

/**
 * Values used when a token is missing from the document.
 *
 * Not a second palette: these are the dark theme's own values, and they exist
 * for the one case where `getComputedStyle` returns nothing — a diagram
 * rendered before the theme has been applied, or in a harness with no
 * stylesheet. Without them Mermaid receives empty strings and paints
 * black-on-black, which on screen is indistinguishable from a diagram that
 * failed to render at all.
 */
const FALLBACK: DiagramTokens = {
  background: '#0B0B0B',
  surface: '#141414',
  surfaceRaised: '#1F1F1F',
  textPrimary: 'rgba(250, 250, 250, 0.92)',
  textSecondary: 'rgba(212, 212, 216, 0.72)',
  border: '#262626',
  accent: '#FAFAFA',
  accentForeground: '#000000',
  fontFamily: 'system-ui, -apple-system, sans-serif',
  fontMono: 'Consolas, monospace',
};

/** Reads the palette off an element, falling back per token rather than wholesale. */
export function readDiagramTokens(element: Element | null): DiagramTokens {
  if (!element || typeof getComputedStyle !== 'function') return FALLBACK;

  const style = getComputedStyle(element);
  const read = (name: string, fallback: string): string => {
    const value = style.getPropertyValue(name).trim();
    return value || fallback;
  };

  return {
    background: read('--bg-primary', FALLBACK.background),
    surface: read('--bg-secondary', FALLBACK.surface),
    surfaceRaised: read('--bg-tertiary', FALLBACK.surfaceRaised),
    textPrimary: read('--text-primary', FALLBACK.textPrimary),
    textSecondary: read('--text-secondary', FALLBACK.textSecondary),
    border: read('--border-default', FALLBACK.border),
    accent: read('--accent-primary', FALLBACK.accent),
    accentForeground: read('--accent-foreground', FALLBACK.accentForeground),
    fontFamily: read('--font-ui', FALLBACK.fontFamily),
    fontMono: read('--font-mono', FALLBACK.fontMono),
  };
}

/* ------------------------------------------------------------------ *
 * Colour
 * ------------------------------------------------------------------ */

/**
 * Why the colours are computed here rather than left to CSS.
 *
 * The obvious way to shade a token is `color-mix(in srgb, …)` and let the
 * browser resolve it — Mermaid writes these values into a stylesheet, so the
 * engine that already parses every CSS colour form would do the work.
 *
 * It does not get that far. Mermaid does not pass `themeVariables` through to
 * the stylesheet untouched; it parses each one with `khroma` to *derive* further
 * shades — a lighter node fill, a darker border. Handed a `color-mix()`
 * expression it throws `Unsupported color format` before a single diagram is
 * laid out, taking every diagram on the page with it. Measured, not assumed.
 *
 * So the tokens are parsed and blended here, and Mermaid is handed plain
 * six-digit hex, which is the one format nothing has an opinion about.
 */
interface Rgb {
  r: number;
  g: number;
  b: number;
  a: number;
}

const HEX = /^#([0-9a-f]{3,8})$/i;
const FUNCTIONAL = /^rgba?\(([^)]+)\)$/i;

/** Parses the CSS colour forms `index.css` actually declares. Null for anything else. */
function parseColour(value: string): Rgb | null {
  const text = value.trim();

  const hex = text.match(HEX);
  if (hex) {
    const digits = hex[1];
    const expand = (pair: string) => parseInt(pair.length === 1 ? pair + pair : pair, 16);
    if (digits.length === 3 || digits.length === 4) {
      return {
        r: expand(digits[0]),
        g: expand(digits[1]),
        b: expand(digits[2]),
        a: digits.length === 4 ? expand(digits[3]) / 255 : 1,
      };
    }
    if (digits.length === 6 || digits.length === 8) {
      return {
        r: expand(digits.slice(0, 2)),
        g: expand(digits.slice(2, 4)),
        b: expand(digits.slice(4, 6)),
        a: digits.length === 8 ? expand(digits.slice(6, 8)) / 255 : 1,
      };
    }
    return null;
  }

  const functional = text.match(FUNCTIONAL);
  if (functional) {
    // Both spellings: `rgb(1, 2, 3)` and the newer `rgb(1 2 3 / 50%)`.
    const parts = functional[1].split('/');
    const channels = parts[0].trim().split(/[\s,]+/).filter(Boolean).map(Number);
    if (channels.length < 3 || channels.some(Number.isNaN)) return null;
    const alphaText = (parts[1] ?? String(channels[3] ?? 1)).trim();
    const alpha = alphaText.endsWith('%')
      ? Number(alphaText.slice(0, -1)) / 100
      : Number(alphaText);
    return {
      r: channels[0],
      g: channels[1],
      b: channels[2],
      a: Number.isNaN(alpha) ? 1 : alpha,
    };
  }

  return null;
}

function toHex(colour: Rgb): string {
  const channel = (value: number) =>
    Math.max(0, Math.min(255, Math.round(value)))
      .toString(16)
      .padStart(2, '0');
  return `#${channel(colour.r)}${channel(colour.g)}${channel(colour.b)}`;
}

/**
 * Composites a translucent colour onto an opaque one.
 *
 * The dark theme states its text as `rgba(250, 250, 250, 0.92)`, and alpha
 * inside an SVG resolves against whatever happens to sit behind that part of
 * the drawing — a node fill in one place, the page in another, so the same
 * token would render as two different greys. Flattened once against the
 * diagram's own ground, it is one grey everywhere.
 */
function flatten(colour: Rgb, over: Rgb): Rgb {
  const a = Math.max(0, Math.min(1, colour.a));
  return {
    r: colour.r * a + over.r * (1 - a),
    g: colour.g * a + over.g * (1 - a),
    b: colour.b * a + over.b * (1 - a),
    a: 1,
  };
}

/** A colour `ratio` of the way from `background` toward `foreground`. */
function blend(foreground: Rgb, background: Rgb, ratio: number): Rgb {
  const t = Math.max(0, Math.min(1, ratio));
  return {
    r: foreground.r * t + background.r * (1 - t),
    g: foreground.g * t + background.g * (1 - t),
    b: foreground.b * t + background.b * (1 - t),
    a: 1,
  };
}

/** Every token as an opaque colour, resolved against the diagram's own ground. */
interface Palette {
  background: Rgb;
  surface: Rgb;
  surfaceRaised: Rgb;
  textPrimary: Rgb;
  textSecondary: Rgb;
  border: Rgb;
  accent: Rgb;
  accentForeground: Rgb;
}

/**
 * Resolves the raw token strings into a palette.
 *
 * A token that cannot be parsed falls back to the same token's dark-theme
 * value rather than to black: an unparseable `--border-default` should produce
 * a visible border, not an invisible one.
 */
function palette(tokens: DiagramTokens): Palette {
  const ground =
    parseColour(tokens.background) ??
    parseColour(FALLBACK.background) ??
    { r: 0, g: 0, b: 0, a: 1 };

  const resolve = (value: string, fallback: string): Rgb => {
    const parsed = parseColour(value) ?? parseColour(fallback);
    return flatten(parsed ?? ground, ground);
  };

  return {
    background: ground,
    surface: resolve(tokens.surface, FALLBACK.surface),
    surfaceRaised: resolve(tokens.surfaceRaised, FALLBACK.surfaceRaised),
    textPrimary: resolve(tokens.textPrimary, FALLBACK.textPrimary),
    textSecondary: resolve(tokens.textSecondary, FALLBACK.textSecondary),
    border: resolve(tokens.border, FALLBACK.border),
    accent: resolve(tokens.accent, FALLBACK.accent),
    accentForeground: resolve(tokens.accentForeground, FALLBACK.accentForeground),
  };
}

/**
 * The twelve slice colours a pie or git graph is drawn with.
 *
 * Mermaid reads exactly `pie1` through `pie12` and falls back to its own
 * categorical ramp for any left unset, so setting eleven of them would produce
 * ten greys and a magenta. They are generated as a luminance ramp between the
 * ground and the text colour rather than listed, so the series stays monochrome
 * in both themes.
 *
 * The ramp alternates rather than descending smoothly. A pie chart is the one
 * place in this product where a reader has to tell adjacent regions apart with
 * no word attached, and neighbouring slices drawn one step apart on a
 * twelve-step gradient are not distinguishable; at opposite ends of it they are.
 */
function sliceRamp(colours: Palette, ground: Rgb): string[] {
  // Ordered so that no two neighbours land within 0.2 of each other on the
  // ramp. The obvious descending order fails that immediately, and an earlier
  // hand-picked alternation still put two slices 0.18 apart — about nine steps
  // of grey, which side by side read as one region.
  const steps = [0.95, 0.3, 0.8, 0.15, 0.65, 0.45, 0.9, 0.25, 0.75, 0.1, 0.6, 0.4];
  return steps.map(step => toHex(blend(colours.textPrimary, ground, step)));
}

/**
 * Mermaid's `themeVariables`, built from the application's own tokens.
 *
 * Every family Mermaid names is set explicitly. An unset variable does not
 * inherit from the ones near it — it falls back to the `base` theme's own
 * colour, which is how one unset `noteBkgColor` puts a pale yellow sticky note
 * in the middle of an otherwise greyscale sequence diagram.
 */
export function mermaidTheme(tokens: DiagramTokens): Record<string, string> {
  const colours = palette(tokens);

  /*
    The ground a diagram is painted on is `--bg-secondary`, because that is what
    `.mdDiagram` in `ChatSurface.module.css` fills the figure with. Mermaid is
    told the same thing so its own edge-label backing matches the figure instead
    of punching darker holes in it.
  */
  const ground = colours.surface;

  /*
    Node fills and outlines are *derived* from the distance between the text
    colour and the ground, rather than taken from `--bg-tertiary` and
    `--border-default` directly.

    Not a stylistic preference — the direct mapping was tried and is
    unreadable. In the dark theme `--bg-secondary` is `#0B0B0B` and
    `--bg-tertiary` is `#141414`: a nine-step difference out of 255, which on
    screen is a flowchart of invisible boxes. Those tokens are sized for large
    surfaces like a sidebar against a page, where nine steps of separation over
    several hundred pixels reads fine. A node outline is one pixel wide.

    Deriving the steps from the text-to-ground contrast instead makes the
    separation hold in both themes automatically, because it is computed from
    the one pair of colours the theme guarantees are far apart.
  */
  const step = (ratio: number) => toHex(blend(colours.textPrimary, ground, ratio));

  const backdrop = toHex(ground);
  const nodeFill = step(0.07);
  const nodeRaised = step(0.13);
  const nodeStroke = step(0.3);
  const line = step(0.55);

  const textPrimary = toHex(colours.textPrimary);
  const textSecondary = toHex(colours.textSecondary);
  const accent = toHex(colours.accent);

  const variables: Record<string, string> = {
    background: backdrop,
    fontFamily: tokens.fontFamily,
    fontSize: '13px',

    /* Nodes, and the text and outline on them. */
    primaryColor: nodeFill,
    primaryTextColor: textPrimary,
    primaryBorderColor: nodeStroke,
    secondaryColor: nodeRaised,
    secondaryTextColor: textPrimary,
    secondaryBorderColor: nodeStroke,
    tertiaryColor: backdrop,
    tertiaryTextColor: textSecondary,
    tertiaryBorderColor: nodeStroke,
    mainBkg: nodeFill,
    nodeBorder: nodeStroke,
    nodeTextColor: textPrimary,
    titleColor: textPrimary,
    textColor: textPrimary,

    /* Edges. The label sits on the diagram's own ground so a line passing
       behind it does not read as passing through the word. */
    lineColor: line,
    arrowheadColor: line,
    edgeLabelBackground: backdrop,
    labelColor: textPrimary,
    labelTextColor: textPrimary,

    /* Subgraphs: a shade *below* a node, so a node reads as sitting on the
       cluster rather than the cluster as another node. */
    clusterBkg: step(0.03),
    clusterBorder: nodeStroke,

    /* Notes, in every diagram type that has them. */
    noteBkgColor: nodeRaised,
    noteTextColor: textPrimary,
    noteBorderColor: nodeStroke,

    /* Sequence diagrams. */
    actorBkg: nodeFill,
    actorBorder: nodeStroke,
    actorTextColor: textPrimary,
    actorLineColor: nodeStroke,
    signalColor: line,
    signalTextColor: textPrimary,
    labelBoxBkgColor: nodeFill,
    labelBoxBorderColor: nodeStroke,
    loopTextColor: textPrimary,
    activationBkgColor: nodeRaised,
    activationBorderColor: nodeStroke,
    sequenceNumberColor: toHex(colours.accentForeground),

    /* State diagrams. */
    transitionColor: line,
    transitionLabelColor: textPrimary,
    stateLabelColor: textPrimary,
    stateBkg: nodeFill,
    labelBackgroundColor: backdrop,
    compositeBackground: step(0.03),
    compositeBorder: nodeStroke,
    compositeTitleBackground: nodeRaised,
    altBackground: nodeRaised,
    /* The filled dot a state machine starts and ends on. It is solid by
       convention, so it takes the text colour rather than a surface one. */
    innerEndBackground: textPrimary,
    specialStateColor: textPrimary,

    /* Class diagrams. */
    classText: textPrimary,

    /* ER diagrams. The two row colours are what make an attribute list readable
       as rows; left unset they come back as pastel blue. */
    attributeBackgroundColorOdd: nodeFill,
    attributeBackgroundColorEven: nodeRaised,

    /* Pie charts. The slices are the ramp below; these are the parts around them. */
    pieTitleTextColor: textPrimary,
    /* Drawn on top of a slice, so it must contrast with the ramp rather than
       with the page — the ramp runs from near-ground to near-text, and the
       ground end of it is where the label sits legibly. */
    pieSectionTextColor: backdrop,
    pieLegendTextColor: textPrimary,
    pieStrokeColor: backdrop,
    pieOuterStrokeColor: nodeStroke,
    pieOpacity: '1',

    /* Git graphs and Gantt charts, so neither falls back to a coloured ramp. */
    gitBranchLabel0: backdrop,
    gitBranchLabel1: textPrimary,
    taskBkgColor: nodeFill,
    taskTextColor: textPrimary,
    taskTextOutsideColor: textPrimary,
    taskBorderColor: nodeStroke,
    activeTaskBkgColor: nodeRaised,
    activeTaskBorderColor: accent,
    doneTaskBkgColor: step(0.04),
    doneTaskBorderColor: nodeStroke,
    critBorderColor: accent,
    critBkgColor: nodeRaised,
    gridColor: nodeStroke,
    sectionBkgColor: nodeFill,
    sectionBkgColor2: nodeRaised,
    altSectionBkgColor: backdrop,
  };

  sliceRamp(colours, ground).forEach((colour, index) => {
    variables[`pie${index + 1}`] = colour;
    if (index < 8) variables[`git${index}`] = colour;
  });

  return variables;
}

/**
 * The whole `mermaid.initialize` configuration.
 *
 * ## `htmlLabels: false` is a security decision, not a styling one
 *
 * With HTML labels on — Mermaid's default — every node label is wrapped in a
 * `<foreignObject>` holding real `<div>`, `<p>` and `<span>` elements. That is
 * HTML smuggled inside the SVG, and `sanitizeDiagramSvg` would have to admit
 * arbitrary HTML into the message surface for a flowchart to draw at all.
 *
 * Off, every label is an ordinary `<text>`/`<tspan>` pair. This was measured
 * rather than assumed: with the flag set only per-diagram
 * (`flowchart.htmlLabels`), `<foreignObject>` still appeared in flowchart, ER,
 * state and class output; only the top-level flag removed it from all four.
 * Both spellings are set, because a per-diagram value overrides the top-level
 * one wherever Mermaid reads it.
 *
 * ## `suppressErrorRendering`
 *
 * On a syntax error Mermaid otherwise paints its own "bomb" graphic straight
 * into `document.body` — outside React's tree, where nothing will ever remove
 * it. Suppressed, `render` simply throws and the caller shows the source
 * instead, which is the fallback this surface already uses everywhere else.
 */
export function mermaidConfig(tokens: DiagramTokens) {
  return {
    startOnLoad: false,
    // Mermaid runs its own output through DOMPurify at this level. It is not
    // the only defence — `sanitizeDiagramSvg` re-checks against an allowlist
    // before anything reaches the page — but there is no reason to turn it off.
    securityLevel: 'strict' as const,
    suppressErrorRendering: true,
    theme: 'base' as const,
    themeVariables: mermaidTheme(tokens),
    fontFamily: tokens.fontFamily,
    altFontFamily: tokens.fontMono,
    htmlLabels: false,
    flowchart: { htmlLabels: false, useMaxWidth: true, curve: 'basis' as const },
    er: { htmlLabels: false, useMaxWidth: true },
    state: { htmlLabels: false, useMaxWidth: true },
    class: { htmlLabels: false, useMaxWidth: true },
    sequence: { useMaxWidth: true },
    gantt: { useMaxWidth: true },
    pie: { useMaxWidth: true },
  };
}
