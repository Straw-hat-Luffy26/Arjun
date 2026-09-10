/**
 * The widget boundary, from the client side.
 *
 * A widget is code a language model wrote, running live in a message. Two
 * things keep that safe, and only one of them is testable here: the served CSP
 * and the iframe sandbox are enforced by the browser and asserted on the Rust
 * side, while every decision the surface makes for itself — what a widget may
 * see of the theme, and what it is allowed to say back — is in this module and
 * is asserted here.
 *
 * The message reader gets the most attention because `sendPrompt` is the one
 * capability that reaches back out of the sandbox. Everything else a widget
 * does stays inside it.
 */
import { describe, expect, it } from 'vitest';

import {
  HOST_SOURCE,
  MAX_PROMPT_LENGTH,
  MAX_WIDGET_HEIGHT,
  WIDGET_SOURCE,
  WIDGET_THEME_TOKENS,
  buildWidgetDocument,
  readWidgetMessage,
  themeBlock,
  widgetUrl,
  type WidgetTheme,
} from './widgetDocument';

const widget = (fields: Record<string, unknown>) => ({ source: WIDGET_SOURCE, ...fields });

describe('the document a widget is served', () => {
  it('carries the model markup and the bridge', () => {
    const document = buildWidgetDocument('<p id="hello">hi</p>');

    expect(document).toContain('<p id="hello">hi</p>');
    expect(document).toContain('window.sendPrompt');
    expect(document).toMatch(/^<!doctype html>/);
  });

  it('does not alter the markup the model wrote', () => {
    // Deliberate. This document is served to a separate opaque origin with no
    // network and no reach into the application, so script inside it is the
    // feature. `svgSanitize` allowlists because an SVG fence goes into the
    // application's own DOM; nothing here does.
    const body = '<script>let x = 1 < 2 && 3 > 2;</script><canvas id="c"></canvas>';
    expect(buildWidgetDocument(body)).toContain(body);
  });

  it('injects the theme so a widget follows light and dark without being told', () => {
    const document = buildWidgetDocument('<p>x</p>', {
      '--bg-primary': '#000000',
      '--text-primary': 'rgba(250, 250, 250, 0.92)',
    });

    expect(document).toContain('--bg-primary: #000000;');
    expect(document).toContain('--text-primary: rgba(250, 250, 250, 0.92);');
  });

  it('forwards only the tokens on the list', () => {
    // A widget that could read every token the application defines would couple
    // itself to internals that move.
    const block = themeBlock({
      '--bg-primary': '#fff',
      '--secret-internal': 'nope',
    } as WidgetTheme);

    expect(block).toContain('--bg-primary');
    expect(block).not.toContain('--secret-internal');
  });

  it('drops a token value that would break out of the style block', () => {
    // These come from the host's own computed style rather than from a model,
    // but they are interpolated into a <style> element, and a value carrying
    // "}" or "<" would end the rule or the element.
    const block = themeBlock({ '--bg-primary': '#fff} body{display:none' });
    expect(block).not.toContain('display:none');

    const script = themeBlock({ '--text-primary': '</style><script>alert(1)</script>' });
    expect(script).not.toContain('<script>');
  });

  it('names every listed token in the block it builds', () => {
    const everything = Object.fromEntries(
      WIDGET_THEME_TOKENS.map(token => [token, 'red']),
    ) as WidgetTheme;

    const block = themeBlock(everything);
    for (const token of WIDGET_THEME_TOKENS) {
      expect(block, token).toContain(token);
    }
  });
});

describe('where a widget is loaded from', () => {
  it('uses the shape the platform actually serves', () => {
    // One registration, two URL shapes. Both are named in the application's
    // frame-src, and getting this wrong is a frame that silently never loads.
    expect(widgetUrl('abc', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)')).toBe(
      'http://widget.localhost/abc',
    );
    expect(widgetUrl('abc', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0)')).toBe(
      'widget://localhost/abc',
    );
  });
});

describe('what a widget is allowed to say back', () => {
  it('reads the three messages the bridge sends', () => {
    expect(readWidgetMessage(widget({ type: 'ready' }))).toEqual({ type: 'ready' });
    expect(readWidgetMessage(widget({ type: 'height', height: 240 }))).toEqual({
      type: 'height',
      height: 240,
    });
    expect(readWidgetMessage(widget({ type: 'prompt', text: 'explain step 2' }))).toEqual({
      type: 'prompt',
      text: 'explain step 2',
    });
  });

  it('ignores a message that does not claim to be from a widget', () => {
    // The page receives messages from anything that can reach it. Only the ones
    // that say what they are, and are then checked, are believed.
    expect(readWidgetMessage({ type: 'prompt', text: 'hi' })).toBeNull();
    expect(readWidgetMessage({ source: 'somewhere-else', type: 'ready' })).toBeNull();
    expect(readWidgetMessage({ source: HOST_SOURCE, type: 'theme', css: '' })).toBeNull();
  });

  it('ignores anything that is not a message at all', () => {
    for (const value of [null, undefined, 'ready', 42, [], true]) {
      expect(readWidgetMessage(value), String(value)).toBeNull();
    }
  });

  it('ignores a type this build does not know', () => {
    // A widget from a newer build, or one probing for a handler.
    // The assertion is that this message is dropped, so the URL is never read,
    // let alone fetched.
    const probe = { type: 'navigate', url: 'https://example.com' }; // arjun-egress-ok: never fetched
    expect(readWidgetMessage(widget(probe))).toBeNull();
    expect(readWidgetMessage(widget({ type: 'constructor' }))).toBeNull();
    expect(readWidgetMessage(widget({}))).toBeNull();
  });

  it('refuses a height that is not a usable number', () => {
    for (const height of [NaN, Infinity, -1, 0, '300', null]) {
      expect(readWidgetMessage(widget({ type: 'height', height })), String(height)).toBeNull();
    }
  });

  it('clamps a height rather than letting one widget take the scrollback', () => {
    const message = readWidgetMessage(widget({ type: 'height', height: 999999 }));
    expect(message).toEqual({ type: 'height', height: MAX_WIDGET_HEIGHT });
  });

  it('refuses a prompt that is not text, or is only whitespace', () => {
    // `sendPrompt` starts a real chat turn. An empty one would be a turn the
    // person did not ask for and cannot read.
    for (const text of ['', '   ', '\n\t', 42, null, undefined, { toString: () => 'hi' }]) {
      expect(readWidgetMessage(widget({ type: 'prompt', text })), String(text)).toBeNull();
    }
  });

  it('caps a prompt so a widget cannot post an essay', () => {
    const message = readWidgetMessage(widget({ type: 'prompt', text: 'x'.repeat(9999) }));
    expect(message).toEqual({ type: 'prompt', text: 'x'.repeat(MAX_PROMPT_LENGTH) });
  });

  it('trims a prompt to what would have been typed', () => {
    expect(readWidgetMessage(widget({ type: 'prompt', text: '  why?  ' }))).toEqual({
      type: 'prompt',
      text: 'why?',
    });
  });
});
