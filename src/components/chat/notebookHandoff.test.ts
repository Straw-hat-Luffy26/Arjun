/**
 * The hand-off from the Notebooks page to the main composer.
 *
 * The bug these are written against: the hand-off used to be a `CustomEvent`,
 * and the chat surface is a route. While the Notebooks page is on screen the
 * surface is not mounted, so it had no listener, so the event reached nothing
 * — and the person arrived at an empty composer having pressed a button that
 * appeared to do nothing at all.
 */
import { afterEach, describe, expect, it } from 'vitest';
import {
  clearPendingHandoff,
  handOffNotebook,
  isHandoff,
  takePendingHandoff,
  type NotebookHandoff,
} from './notebookHandoff';

const SHA = 'a'.repeat(64);

const handoff = (over: Partial<NotebookHandoff> = {}): NotebookHandoff => ({
  notebookId: 'nb-a',
  notebookName: 'Unit Four',
  sourceSha256s: null,
  prompt: null,
  ...over,
});

afterEach(() => clearPendingHandoff());

describe('surviving the navigation', () => {
  it('is still there when the surface mounts later', () => {
    handOffNotebook(handoff({ prompt: 'What do my sources say about PV-2201?' }));

    // Nothing was listening at dispatch time; the surface mounts now.
    const found = takePendingHandoff();
    expect(found?.notebookId).toBe('nb-a');
    expect(found?.prompt).toBe('What do my sources say about PV-2201?');
  });

  /** Consumed once. Returning to the chat must not re-attach it. */
  it('is cleared by reading it', () => {
    handOffNotebook(handoff());
    expect(takePendingHandoff()).not.toBeNull();
    expect(takePendingHandoff()).toBeNull();
  });

  it('is replaced rather than queued when a second one is made', () => {
    handOffNotebook(handoff({ notebookId: 'nb-a' }));
    handOffNotebook(handoff({ notebookId: 'nb-b', notebookName: 'Procurement' }));

    expect(takePendingHandoff()?.notebookId).toBe('nb-b');
    expect(takePendingHandoff()).toBeNull();
  });

  /**
   * The value half must work with no DOM at all, which is what this suite runs
   * in. A hand-off that threw reaching for `window` would be lost *and* would
   * take its caller with it.
   */
  it('does not need a window to survive', () => {
    expect(typeof globalThis.window).toBe('undefined');
    handOffNotebook(handoff({ notebookName: 'Unit Four' }));
    expect(takePendingHandoff()?.notebookName).toBe('Unit Four');
  });

  it('also fires an event, for a surface that is already on screen', () => {
    const listeners: ((event: unknown) => void)[] = [];
    let seen: unknown = null;
    // A minimal stand-in, so the event half is exercised without pulling a
    // whole DOM into a suite of pure-logic tests.
    (globalThis as Record<string, unknown>).window = {
      dispatchEvent: (event: { detail?: unknown }) => {
        listeners.forEach(listen => listen(event));
        return true;
      },
    };
    (globalThis as Record<string, unknown>).CustomEvent = class {
      detail: unknown;
      constructor(_type: string, init?: { detail?: unknown }) {
        this.detail = init?.detail;
      }
    };
    listeners.push(event => {
      seen = (event as { detail?: unknown }).detail;
    });

    try {
      handOffNotebook(handoff({ notebookName: 'Unit Four' }));
      expect(isHandoff(seen)).toBe(true);
    } finally {
      delete (globalThis as Record<string, unknown>).window;
      delete (globalThis as Record<string, unknown>).CustomEvent;
    }
  });
});

describe('the selection it carries', () => {
  /**
   * `null` and `[]` are different. `null` is "the page narrowed nothing", which
   * becomes `all`; an empty list would be "everything was deselected", and
   * reading that as `all` is the mistake the selection type exists to prevent.
   */
  it('keeps "narrowed nothing" distinct from "narrowed to nothing"', () => {
    handOffNotebook(handoff({ sourceSha256s: null }));
    expect(takePendingHandoff()?.sourceSha256s).toBeNull();

    handOffNotebook(handoff({ sourceSha256s: [] }));
    expect(takePendingHandoff()?.sourceSha256s).toEqual([]);
  });

  it('carries a narrowed selection through unchanged', () => {
    handOffNotebook(handoff({ sourceSha256s: [SHA] }));
    expect(takePendingHandoff()?.sourceSha256s).toEqual([SHA]);
  });
});

describe('validating what arrives on the event', () => {
  it('rejects anything that is not a hand-off', () => {
    for (const value of [null, undefined, 'nb-a', 42, {}, { notebookId: '' }]) {
      expect(isHandoff(value)).toBe(false);
    }
  });

  it('accepts a real one', () => {
    expect(isHandoff(handoff())).toBe(true);
  });
});
