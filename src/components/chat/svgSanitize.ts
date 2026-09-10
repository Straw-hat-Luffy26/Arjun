/**
 * Makes an SVG safe to put into the page.
 *
 * ARJUN draws charts as SVG because an SVG is text: it renders inline, follows
 * the theme, and scales without going soft. The cost of that choice is that an
 * SVG is also a document with a script model, and the fence this parses came
 * out of a language model. A model that has read a hostile document can be
 * induced to emit `<script>` or `onload=` inside a diagram, and putting that
 * into the page unexamined would be the injection this whole product exists to
 * avoid.
 *
 * So the rule here is an allowlist, not a blocklist: only shapes, text and
 * presentation attributes survive. Anything not named is dropped. A blocklist
 * of "dangerous" tags fails the day somebody finds one nobody listed.
 *
 * This is defence in depth rather than the only defence. ARJUN's own charts are
 * written by `artifacts/chart.rs`, which escapes every label it is given and
 * cannot emit a tag at all. This exists for the SVG that did *not* come from
 * there.
 */

/** Elements that can appear in a chart or diagram. Everything else is dropped. */
const ALLOWED_ELEMENTS = new Set([
  'svg',
  'g',
  'defs',
  'title',
  'desc',
  'style',
  'rect',
  'circle',
  'ellipse',
  'line',
  'polyline',
  'polygon',
  'path',
  'text',
  'tspan',
  'marker',
  'lineargradient',
  'radialgradient',
  'stop',
  'clippath',
  'use',
]);

/**
 * Attributes that only affect how something is drawn.
 *
 * No `href`/`xlink:href` on anything but `use`, no `filter`, and nothing
 * beginning `on`. `use` keeps a same-document `href` because a marker
 * definition is referenced that way and cannot reach outside the document.
 */
const ALLOWED_ATTRIBUTES = new Set([
  'viewbox',
  'width',
  'height',
  'x',
  'y',
  'x1',
  'y1',
  'x2',
  'y2',
  'cx',
  'cy',
  'r',
  'rx',
  'ry',
  'd',
  'points',
  'fill',
  'fill-opacity',
  'fill-rule',
  'stroke',
  'stroke-width',
  'stroke-opacity',
  'stroke-dasharray',
  'stroke-linecap',
  'stroke-linejoin',
  'transform',
  // Without these a diagram loses its arrowheads. `diagram.rs` draws every edge
  // as `marker-end="url(#arw)"`, referring to a `<marker>` defined in the same
  // document — which `defs` and `marker` are already allowed to carry. Stripping
  // the reference left the definition in place and nothing pointing at it, so a
  // flowchart rendered as undirected lines: the same picture, saying something
  // different.
  'marker-end',
  'marker-start',
  'marker-mid',
  'opacity',
  'class',
  'text-anchor',
  'dominant-baseline',
  'alignment-baseline',
  'font-size',
  'font-family',
  'font-weight',
  'letter-spacing',
  'offset',
  'stop-color',
  'stop-opacity',
  'gradientunits',
  'markerwidth',
  'markerheight',
  'markerunits',
  'orient',
  'refx',
  'refy',
  'preserveaspectratio',
  'role',
  'aria-label',
  'id',
  'xmlns',
]);

/** A `url(...)` reference that leaves the document, or a script URL. */
function valueIsSafe(name: string, value: string): boolean {
  const lowered = value.toLowerCase();
  if (lowered.includes('javascript:') || lowered.includes('data:text/html')) return false;
  // `url(#id)` is a same-document reference and fine; `url(http...)` is not.
  if (lowered.includes('url(') && !/url\(\s*['"]?#/.test(lowered)) return false;
  if (name === 'href' || name === 'xlink:href') return lowered.startsWith('#');
  return true;
}

/**
 * Returns sanitised SVG markup, or null when the input is not an SVG at all.
 *
 * Parsed with the browser's own XML parser rather than a regular expression:
 * the shapes a regex misses on markup like `<scr<script>ipt>` are exactly the
 * ones an attacker reaches for.
 */
export function sanitizeSvg(source: string): string | null {
  const text = source.trim();
  if (!text.startsWith('<svg') && !text.startsWith('<?xml')) return null;

  const parsed = new DOMParser().parseFromString(text, 'image/svg+xml');
  if (parsed.getElementsByTagName('parsererror').length > 0) return null;

  const root = parsed.documentElement;
  if (!root || root.localName.toLowerCase() !== 'svg') return null;

  const walk = (node: Element): boolean => {
    const name = node.localName.toLowerCase();
    if (!ALLOWED_ELEMENTS.has(name)) return false;

    for (const attribute of [...node.attributes]) {
      const attributeName = attribute.name.toLowerCase();
      const allowed =
        ALLOWED_ATTRIBUTES.has(attributeName) ||
        (name === 'use' && (attributeName === 'href' || attributeName === 'xlink:href'));
      if (!allowed || !valueIsSafe(attributeName, attribute.value)) {
        node.removeAttribute(attribute.name);
      }
    }

    for (const child of [...node.children]) {
      if (!walk(child)) child.remove();
    }
    return true;
  };

  if (!walk(root)) return null;

  // A stylesheet inside the drawing can carry a `url()` out of the document,
  // so its text is checked as a whole rather than trusted for being in a
  // `<style>` the allowlist happens to permit.
  for (const style of [...root.querySelectorAll('style')]) {
    const css = style.textContent ?? '';
    if (!valueIsSafe('style', css) || /@import|expression\(/i.test(css)) style.remove();
  }

  // Always sized by its container, never by an attribute the source chose: a
  // chart that declares `height="8000"` would push the conversation off screen.
  root.removeAttribute('width');
  root.removeAttribute('height');
  if (!root.getAttribute('viewBox')) return null;

  return new XMLSerializer().serializeToString(root);
}

/* ------------------------------------------------------------------ *
 * Diagrams rendered by Mermaid
 * ------------------------------------------------------------------ */

/**
 * The same job for a different provenance.
 *
 * ## Why this is not [`sanitizeSvg`] with a longer list
 *
 * The two inputs are not the same kind of thing, and merging them would loosen
 * the tighter one to suit the looser.
 *
 * [`sanitizeSvg`] reads an `svg` fence: **markup a language model wrote by
 * hand**. Every element and attribute in it is a choice the model made, so the
 * list is as short as a chart can be drawn with, and `style` is deliberately
 * not on it — a model that writes `style="background:url(...)"` is exactly the
 * case that function exists to stop.
 *
 * This reads the output of `mermaid.render`: **markup our own bundled library
 * generated** from model-written *text*. The model chose the words in the
 * labels, not the tags around them, and Mermaid has already escaped those and
 * run the result through DOMPurify. What arrives here is a drawing built by a
 * known program, and the shapes that program uses are not negotiable — a
 * `<filter>` for node shadows, a `<symbol>` in sequence diagrams, `style`
 * attributes carrying per-node fills.
 *
 * So this is still an allowlist and still drops everything it does not name. It
 * names a different set, and the set was **measured** rather than guessed:
 * flowchart, `graph LR`, ER, sequence, state, class and pie output was rendered
 * and its elements and attributes enumerated. Nothing appears here that was not
 * seen coming out of Mermaid.
 *
 * The two things it will not admit at any width are the two that matter:
 * `<script>` and `<foreignObject>`. The second is why `mermaidConfig` sets
 * `htmlLabels: false` — with HTML labels on, every node carries a `<div>` inside
 * the SVG, and admitting those would mean admitting arbitrary HTML into a chat
 * message.
 */
const DIAGRAM_ELEMENTS = new Set([
  ...ALLOWED_ELEMENTS,
  // Drop shadows on nodes: Mermaid emits `<filter><feDropShadow/></filter>` in
  // flowchart, ER, state and class output.
  'filter',
  'fedropshadow',
  // Sequence diagrams define an actor glyph once and `<use>` it.
  'symbol',
]);

/**
 * Attributes Mermaid draws with, on top of the ones a hand-written chart needs.
 *
 * `style` is the significant addition. Mermaid puts per-node fills and strokes
 * in it, and without it a diagram renders as unfilled outlines. It is not
 * admitted blindly: `valueIsSafe` rejects `javascript:`, `data:text/html` and
 * any `url()` not pointing inside this same document, and it runs on this
 * attribute exactly as it does on every other.
 */
const DIAGRAM_ATTRIBUTES = new Set([
  ...ALLOWED_ATTRIBUTES,
  'style',
  'dx',
  'dy',
  'clip-rule',
  'font-style',
  // `<feDropShadow>` parameters.
  'flood-color',
  'flood-opacity',
  'stddeviation',
  // Mermaid names its markers and gradients; `<use>` and `url(#…)` refer back.
  'name',
  // Announced to a screen reader as "flowchart", "sequence diagram" and so on.
  'aria-roledescription',
]);

/**
 * True for the bookkeeping attributes Mermaid hangs on its own nodes —
 * `data-id`, `data-edge`, `data-points` and the rest.
 *
 * Allowed as a family rather than enumerated, because the set changes between
 * Mermaid releases and a diagram should not lose its shape on an upgrade. They
 * are inert: `data-*` has no meaning to the browser beyond `dataset`, and
 * nothing in this application reads them.
 */
function isDataAttribute(name: string): boolean {
  return name.startsWith('data-');
}

/**
 * Sanitises SVG produced by `mermaid.render`.
 *
 * Returns null when the markup is not a well-formed SVG carrying a `viewBox`,
 * which is the same "show the source instead" signal [`sanitizeSvg`] gives.
 */
export function sanitizeDiagramSvg(source: string): string | null {
  const text = source.trim();
  if (!text.startsWith('<svg') && !text.startsWith('<?xml')) return null;

  const parsed = new DOMParser().parseFromString(text, 'image/svg+xml');
  if (parsed.getElementsByTagName('parsererror').length > 0) return null;

  const root = parsed.documentElement;
  if (!root || root.localName.toLowerCase() !== 'svg') return null;

  const walk = (node: Element): boolean => {
    const name = node.localName.toLowerCase();
    if (!DIAGRAM_ELEMENTS.has(name)) return false;

    for (const attribute of [...node.attributes]) {
      const attributeName = attribute.name.toLowerCase();
      const allowed =
        DIAGRAM_ATTRIBUTES.has(attributeName) ||
        isDataAttribute(attributeName) ||
        ((name === 'use' || name === 'symbol') &&
          (attributeName === 'href' || attributeName === 'xlink:href'));
      if (!allowed || !valueIsSafe(attributeName, attribute.value)) {
        node.removeAttribute(attribute.name);
      }
    }

    for (const child of [...node.children]) {
      if (!walk(child)) child.remove();
    }
    return true;
  };

  if (!walk(root)) return null;

  // Mermaid's stylesheet is generated from `themeVariables` and carries no
  // `url()`, but it is checked on the same terms as any other: a stylesheet is
  // the one place inside a drawing that can still reach out of it.
  for (const style of [...root.querySelectorAll('style')]) {
    const css = style.textContent ?? '';
    if (!valueIsSafe('style', css) || /@import|expression\(/i.test(css)) style.remove();
  }

  root.removeAttribute('width');
  root.removeAttribute('height');
  // Mermaid writes `style="max-width: 332px"` on the root, which would pin a
  // diagram to whatever width its layout engine happened to measure and leave
  // the rest of the figure empty. The container sizes it instead.
  root.removeAttribute('style');
  if (!root.getAttribute('viewBox')) return null;

  return new XMLSerializer().serializeToString(root);
}
