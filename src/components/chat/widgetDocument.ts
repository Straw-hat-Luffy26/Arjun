/**
 * Assembling the document a widget frame loads, and reading what it says back.
 *
 * ## The two halves of the boundary
 *
 * A widget is HTML, CSS and JS a language model wrote, running live in a chat
 * message. Everything that makes that safe lives in two places, and this module
 * is the client half:
 *
 * - `commands/widget.rs` serves the document from the `widget://` scheme with
 *   `default-src 'none'` — no network in any form. It is a separate origin
 *   because a `srcdoc` frame inherits the application's CSP, which was measured
 *   with a control rather than assumed.
 * - The frame is mounted `sandbox="allow-scripts"` and *without*
 *   `allow-same-origin`, so it runs at an opaque origin with no reach into the
 *   parent document, and without `allow-popups` or `allow-top-navigation`.
 *
 * What is left for this module is everything that has to be decided rather than
 * enforced: which theme tokens a widget may see, what a widget is allowed to
 * say to the application, and how the frame reports that it is alive.
 *
 * ## Why the reply is validated rather than trusted
 *
 * A widget can post anything into the parent. `sendPrompt` turns a click inside
 * a model-written document into a real chat turn, which is the one capability
 * here that reaches back out — so the shape, the type and the length of every
 * message are checked before any of it is believed, and anything unrecognised
 * is dropped rather than passed along.
 *
 * The frame's origin is `"null"` (that is what an opaque origin reports), so
 * the origin cannot be used to tell one frame from another. The *source* can:
 * the component compares `event.source` against its own `contentWindow`, and
 * this module never sees a message that failed that check.
 */

/* ------------------------------------------------------------------ *
 * Theme
 * ------------------------------------------------------------------ */

/**
 * The custom properties a widget is given.
 *
 * A curated list rather than every token the application defines. Two reasons:
 * a widget that could read all of them would couple itself to internals that
 * move, and a short list is one a model can be told about and actually use.
 * These are the names the rest of the surface styles with, so a widget that
 * uses them looks like it belongs and follows the light/dark switch without
 * being told.
 */
export const WIDGET_THEME_TOKENS = [
  '--bg-primary',
  '--bg-secondary',
  '--bg-tertiary',
  '--text-primary',
  '--text-secondary',
  '--text-tertiary',
  '--accent-primary',
  '--border-default',
  '--radius-sm',
  '--radius-md',
  '--radius-lg',
] as const;

export type WidgetThemeToken = (typeof WIDGET_THEME_TOKENS)[number];

/** Token name to value, as read from the host document. */
export type WidgetTheme = Partial<Record<WidgetThemeToken, string>>;

/**
 * A CSS value safe to write into a stylesheet the frame will parse.
 *
 * These values are read from the host's own computed style, so they are not
 * attacker-controlled — but they are interpolated into a `<style>` block, and a
 * value containing `}` or `<` would end the rule or the element. Dropped rather
 * than escaped: a token that looks like that is a bug in the theme, not a value
 * worth rescuing.
 */
function isSafeCssValue(value: string): boolean {
  return value.length > 0 && value.length <= 120 && !/[<>{};]/.test(value);
}

/** The `:root` block that themes a widget, from the host's live values. */
export function themeBlock(theme: WidgetTheme): string {
  const declarations = WIDGET_THEME_TOKENS.flatMap(token => {
    const value = theme[token]?.trim();
    if (!value || !isSafeCssValue(value)) return [];
    return [`  ${token}: ${value};`];
  });
  return `:root {\n${declarations.join('\n')}\n}`;
}

/* ------------------------------------------------------------------ *
 * The document
 * ------------------------------------------------------------------ */

/** The name a widget's messages carry, so nothing else is mistaken for one. */
export const WIDGET_SOURCE = 'arjun-widget';

/** The name the application's own messages into a frame carry. */
export const HOST_SOURCE = 'arjun-host';

/** Longest prompt a widget may raise. A click becomes a sentence, not an essay. */
export const MAX_PROMPT_LENGTH = 2000;

/** Tallest a widget may ask to be, in CSS pixels. */
export const MAX_WIDGET_HEIGHT = 4000;

/**
 * The bridge every widget document carries.
 *
 * Deliberately small, and written in the JavaScript of ten years ago so it runs
 * whatever the WebView. It gives the widget exactly three things:
 *
 * - `sendPrompt(text)` — the one way back out, and the reason the message
 *   reader on the other side is strict.
 * - a `ready` signal. The frame reporting that it has finished is what closes
 *   the loading state; without it the component could only guess, and guessing
 *   is how a spinner ends up running forever.
 * - its own height, remeasured on every layout change, so the frame fits its
 *   content instead of being given an arbitrary box.
 */
const BRIDGE = `
(function () {
  var post = function (type, extra) {
    var message = { source: ${JSON.stringify(WIDGET_SOURCE)}, type: type };
    if (extra) { for (var key in extra) { message[key] = extra[key]; } }
    try { parent.postMessage(message, '*'); } catch (error) { /* nothing to do */ }
  };

  window.sendPrompt = function (text) {
    if (typeof text !== 'string') return;
    var trimmed = text.trim();
    if (!trimmed) return;
    post('prompt', { text: trimmed.slice(0, ${MAX_PROMPT_LENGTH}) });
  };

  var lastHeight = -1;
  var report = function () {
    var height = Math.ceil(document.documentElement.scrollHeight);
    if (height !== lastHeight) { lastHeight = height; post('height', { height: height }); }
  };

  if (window.ResizeObserver) {
    new ResizeObserver(report).observe(document.documentElement);
  } else {
    setInterval(report, 500);
  }

  // Re-themed whenever the application switches, so a widget written months ago
  // follows a theme it was never told about.
  window.addEventListener('message', function (event) {
    var data = event.data;
    if (!data || data.source !== ${JSON.stringify(HOST_SOURCE)} || data.type !== 'theme') return;
    if (typeof data.css !== 'string') return;
    var style = document.getElementById('arjun-theme');
    if (style) { style.textContent = data.css; report(); }
  });

  var announce = function () { post('ready', {}); report(); };
  if (document.readyState === 'complete') { announce(); }
  else { window.addEventListener('load', announce); }
})();
`;

/**
 * The reset a widget starts from.
 *
 * Enough that a widget which sets no styles at all still reads as part of the
 * application, and no more: a widget's own CSS should win, so this sets only
 * what would otherwise be a browser default nobody wanted — the 8px body
 * margin, and text that does not follow the theme.
 */
const BASE_STYLE = `
  html, body { margin: 0; padding: 0; background: transparent; }
  body {
    color: var(--text-primary, #111);
    font: 13px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif;
    overflow-x: hidden;
  }
  * { box-sizing: border-box; }
`;

/**
 * Builds the complete document for a widget.
 *
 * The model's markup goes into the body untouched. It is not sanitised here,
 * and that is deliberate rather than overlooked: this document is served to a
 * separate opaque origin with no network and no reach into the application, so
 * script inside it is the point. `svgSanitize` allowlists because an SVG fence
 * is injected into the application's *own* DOM; nothing here is.
 */
export function buildWidgetDocument(body: string, theme: WidgetTheme = {}): string {
  return [
    '<!doctype html>',
    '<html>',
    '<head>',
    '<meta charset="utf-8">',
    '<meta name="viewport" content="width=device-width, initial-scale=1">',
    `<style id="arjun-theme">${themeBlock(theme)}</style>`,
    `<style>${BASE_STYLE}</style>`,
    '</head>',
    '<body>',
    body,
    `<script>${BRIDGE}</script>`,
    '</body>',
    '</html>',
  ].join('\n');
}

/**
 * Where a prepared widget lives.
 *
 * Windows serves a registered scheme as `http://<scheme>.localhost`; the other
 * platforms use the scheme directly. Both are named in the application's
 * `frame-src`, and both are produced here rather than at the call site so there
 * is one place that knows.
 */
export function widgetUrl(id: string, platform: string): string {
  return /Windows/i.test(platform)
    ? `http://widget.localhost/${id}`
    : `widget://localhost/${id}`;
}

/* ------------------------------------------------------------------ *
 * What a widget may say back
 * ------------------------------------------------------------------ */

export type WidgetMessage =
  | { type: 'ready' }
  | { type: 'height'; height: number }
  | { type: 'prompt'; text: string };

/**
 * Reads a message from a widget, or returns `null`.
 *
 * Total, and strict in both directions: anything that is not one of the three
 * known shapes is dropped, and a known shape carrying an unusable value — a
 * height of `NaN`, a prompt of whitespace, a number where a string belongs — is
 * dropped too rather than passed on half-checked.
 *
 * The caller has already established that the message came from its own frame.
 * This decides whether what the frame said means anything.
 */
export function readWidgetMessage(data: unknown): WidgetMessage | null {
  if (typeof data !== 'object' || data === null) return null;
  const message = data as Record<string, unknown>;
  if (message.source !== WIDGET_SOURCE) return null;

  switch (message.type) {
    case 'ready':
      return { type: 'ready' };

    case 'height': {
      const height = message.height;
      if (typeof height !== 'number' || !Number.isFinite(height)) return null;
      if (height <= 0) return null;
      // Clamped rather than refused: a widget that measures itself absurdly
      // tall is still worth showing, just not allowed to take the whole
      // scrollback with it.
      return { type: 'height', height: Math.min(Math.ceil(height), MAX_WIDGET_HEIGHT) };
    }

    case 'prompt': {
      const text = message.text;
      if (typeof text !== 'string') return null;
      const trimmed = text.trim();
      if (!trimmed) return null;
      return { type: 'prompt', text: trimmed.slice(0, MAX_PROMPT_LENGTH) };
    }

    default:
      return null;
  }
}
