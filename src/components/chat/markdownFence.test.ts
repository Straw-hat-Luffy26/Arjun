import { describe, expect, it } from 'vitest';

import { isClosingFence, openingFenceLanguage } from './markdownFence';

/** Three backticks, written this way so the fixtures below stay readable. */
const F = '```';

describe('openingFenceLanguage: the language tags models actually write', () => {
  it('accepts the plain alphanumeric tags that always worked', () => {
    expect(openingFenceLanguage(`${F}cpp`)).toBe('cpp');
    expect(openingFenceLanguage(`${F}python`)).toBe('python');
    expect(openingFenceLanguage(`${F}ts`)).toBe('ts');
    expect(openingFenceLanguage(`${F}sql`)).toBe('sql');
  });

  /**
   * The defect. `\w` excludes `+`, `#` and `-`, so a model that answered a C++
   * question by opening its block with `c++` had the whole answer rendered as
   * prose — and the closing delimiter read as the *opening* of a block that
   * then swallowed everything after it.
   */
  it('accepts the punctuated language names that were silently rejected', () => {
    expect(openingFenceLanguage(`${F}c++`)).toBe('c++');
    expect(openingFenceLanguage(`${F}c#`)).toBe('c#');
    expect(openingFenceLanguage(`${F}f#`)).toBe('f#');
    expect(openingFenceLanguage(`${F}objective-c`)).toBe('objective-c');
    expect(openingFenceLanguage(`${F}asp.net`)).toBe('asp.net');
  });

  it('reads a bare fence as a fence that named no language', () => {
    expect(openingFenceLanguage(F)).toBeUndefined();
    expect(openingFenceLanguage(`${F}   `)).toBeUndefined();
  });

  /**
   * `null` and `undefined` mean different things here, and a caller that
   * treated them alike would reopen the swallowing bug.
   */
  it('separates "not a fence" from "a fence with no language"', () => {
    expect(openingFenceLanguage('Here is some code:')).toBeNull();
    expect(openingFenceLanguage('`inline`')).toBeNull();
    expect(openingFenceLanguage(`text before ${F}`)).toBeNull();
    // A tag with a space in it is prose that happens to start with backticks.
    expect(openingFenceLanguage(`${F}c++ and more prose`)).toBeNull();
  });

  it('tolerates trailing whitespace, which models emit constantly', () => {
    expect(openingFenceLanguage(`${F}c++  `)).toBe('c++');
  });
});

describe('isClosingFence', () => {
  it('closes on a bare fence, with or without trailing space', () => {
    expect(isClosingFence(F)).toBe(true);
    expect(isClosingFence(`${F}  `)).toBe(true);
  });

  it('does not close on a fence that opens a language', () => {
    expect(isClosingFence(`${F}c++`)).toBe(false);
    expect(isClosingFence(`${F}cpp`)).toBe(false);
  });

  it('does not close on a line of code that merely contains backticks', () => {
    expect(isClosingFence(`std::cout << "${F}" << std::endl;`)).toBe(false);
  });
});

/**
 * The property the reader actually experiences: a block that is still being
 * written is still a block.
 *
 * These walk the sequence of prefixes a stream produces, because that is the
 * state the renderer is in for the whole time the model is writing — and the
 * state in which the old pattern failed for `c++`.
 */
describe('a fence recognised while the answer is still arriving', () => {
  const answer = [
    "Here's a singly linked list in C++:",
    '',
    `${F}c++`,
    'struct Node {',
    '    int data;',
    '    Node* next;',
    '};',
    F,
    'That is the whole structure.',
  ].join('\n');

  /** Every prefix of the answer, as the stream delivers them. */
  function prefixes(text: string): string[] {
    return Array.from({ length: text.length + 1 }, (_, n) => text.slice(0, n));
  }

  it('opens the block the moment the fence line is complete, before any code', () => {
    const justTheFence = answer.slice(0, answer.indexOf(`${F}c++`) + 6);
    const lines = justTheFence.split('\n');
    expect(openingFenceLanguage(lines[lines.length - 1])).toBe('c++');
  });

  /**
   * Walking every prefix is the point: at no stage may a partially delivered
   * answer contain a line that reads as an opening fence when it is really the
   * close, which is how the old pattern turned the trailing prose into code.
   */
  it('never mistakes the closing fence for the opening of another block', () => {
    for (const prefix of prefixes(answer)) {
      const lines = prefix.split('\n');
      let open = false;
      let openedTwice = false;
      for (const line of lines) {
        if (!open) {
          if (openingFenceLanguage(line) !== null) open = true;
        } else if (isClosingFence(line)) {
          open = false;
        } else if (openingFenceLanguage(line) !== null) {
          openedTwice = true;
        }
      }
      expect(openedTwice).toBe(false);
    }
  });

  it('leaves the prose after the block as prose once the block has closed', () => {
    const lines = answer.split('\n');
    let open = false;
    const code: string[] = [];
    const prose: string[] = [];
    for (const line of lines) {
      if (!open && openingFenceLanguage(line) !== null) {
        open = true;
        continue;
      }
      if (open && isClosingFence(line)) {
        open = false;
        continue;
      }
      (open ? code : prose).push(line);
    }
    expect(code).toEqual(['struct Node {', '    int data;', '    Node* next;', '};']);
    expect(prose).toContain('That is the whole structure.');
  });
});
