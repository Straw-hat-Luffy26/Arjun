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
