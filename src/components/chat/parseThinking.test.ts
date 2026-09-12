import { describe, expect, it } from 'vitest';

import { parseThinking } from './parseThinking';

describe('parseThinking', () => {
  it('gives the reasoning as it arrives, before the block has closed', () => {
    // The case that was broken. Every frame of a reasoning pass looks like
    // this, and the old matcher returned nothing for all of them — so the
    // panel stayed empty and the raw tag went to Markdown, which drew
    // nothing. Twelve seconds of blank space on a real model.
    const parsed = parseThinking('<think>The user said hi. I should');

    expect(parsed.open).toBe(true);
    expect(parsed.reasoning).toBe('The user said hi. I should');
    expect(parsed.answer).toBe('');
    expect(parsed.sawTag).toBe(true);
  });

  it('never hands a raw reasoning tag to the renderer', () => {
    // Markdown swallows `<think>` as an unclosed HTML tag, so a caller that
    // renders this verbatim shows an empty bubble rather than the text.
    for (const content of [
      '<think>half a thought',
      '<think>a whole one</think>the answer',
      'preamble <think>mid-message',
    ]) {
      const { answer } = parseThinking(content);
      expect(answer).not.toContain('<think');
      expect(answer).not.toContain('</think');
    }
  });

  it('splits a closed block into the thought and the answer', () => {
    const parsed = parseThinking('<think>weighing it up</think>Hello! How can I help?');

    expect(parsed.open).toBe(false);
    expect(parsed.reasoning).toBe('weighing it up');
    expect(parsed.answer).toBe('Hello! How can I help?');
  });

  it('streams the answer once the block has closed', () => {
    // The frame after the closer carries the first characters of the reply.
    // The old code produced this correctly and the new code must not lose it.
    const parsed = parseThinking('<think>done</think>Hel');
    expect(parsed.answer).toBe('Hel');
  });

  it('leaves an ordinary message alone', () => {
    const parsed = parseThinking('Just an answer, no reasoning at all.');

    expect(parsed.sawTag).toBe(false);
    expect(parsed.open).toBe(false);
    expect(parsed.reasoning).toBe('');
    expect(parsed.answer).toBe('Just an answer, no reasoning at all.');
  });

  it('keeps text the model wrote before it started reasoning', () => {
    const parsed = parseThinking('One moment. <think>checking');

    expect(parsed.answer).toBe('One moment.');
    expect(parsed.reasoning).toBe('checking');
  });

  it('accepts the other openers a local model may use', () => {
    for (const tag of ['think', 'thinking', 'thought', 'reasoning']) {
      const parsed = parseThinking(`<${tag}>inner</${tag}>outer`);
      expect(parsed.reasoning, tag).toBe('inner');
      expect(parsed.answer, tag).toBe('outer');
    }
  });

  it('does not pair an opener with a different tag closer', () => {
    // `</thinking>` does not close `<think>`. Treating it as a closer would
    // put the closing tag itself into the answer.
    const parsed = parseThinking('<think>still going</thinking>');
    expect(parsed.open).toBe(true);
    expect(parsed.answer).toBe('');
  });

  it('treats a closing tag with no opener as ordinary text', () => {
    // Nothing to split. Rendering the buffer unchanged is the honest fallback;
    // inventing a reasoning block from a stray closer is not.
    const parsed = parseThinking('an answer</think>');
    expect(parsed.sawTag).toBe(false);
    expect(parsed.answer).toBe('an answer</think>');
  });
});
