/**
 * The `/notebook` command and the frozen turn scope.
 *
 * Every test here is a regression for a way this was, or could quietly become,
 * wrong: a draft cleared by choosing a notebook, a command word sent to the
 * model, an empty selection read as "everything", and — the case the whole
 * freeze design exists for — a queued question answered from the notebook that
 * happened to be selected when the queue drained rather than the one it was
 * asked against.
 */
import { describe, expect, it } from 'vitest';
import {
  caretAfterConsume,
  consumeCommand,
  filterNotebooks,
  findNotebookCommand,
  freezeScope,
  moveHighlight,
  sameScope,
  type NotebookScope,
} from './notebookScope';
import { allSources, noSources, someSources } from '../../services/agent.service';

const SHA_A = 'a'.repeat(64);
const SHA_B = 'b'.repeat(64);
const SHA_C = 'c'.repeat(64);

describe('finding the command', () => {
  it('opens on the bare command', () => {
    const span = findNotebookCommand('/notebook', 9);
    expect(span).not.toBeNull();
    expect(span!.complete).toBe(true);
    expect(span!.query).toBe('');
  });

  it('opens while the word is still being typed', () => {
    const span = findNotebookCommand('/note', 5);
    expect(span).not.toBeNull();
    expect(span!.complete).toBe(false);
    expect(span!.query).toBe('');
  });

  it('opens after a question that is already drafted', () => {
    const text = 'What is the rated duty of P-101? /notebook';
    const span = findNotebookCommand(text, text.length);
    expect(span).not.toBeNull();
    expect(span!.start).toBe(text.indexOf('/notebook'));
    expect(span!.end).toBe(text.length);
  });

  it('carries what was typed after the command as the search', () => {
    const text = '/notebook unit four';
    const span = findNotebookCommand(text, text.length);
    expect(span!.query).toBe('unit four');
  });

  /** A slash in the middle of a word is a path, not a command. */
  it('does not fire on a slash inside a word', () => {
    const text = 'see docs/notebook.md for the layout';
    expect(findNotebookCommand(text, text.length)).toBeNull();
    expect(findNotebookCommand(text, text.indexOf('.md'))).toBeNull();
  });

  it('does not fire on another command', () => {
    expect(findNotebookCommand('/notes', 6)).toBeNull();
    expect(findNotebookCommand('/nothing at all', 15)).toBeNull();
  });

  it('closes once the caret moves off the command', () => {
    const text = '/notebook\nand then a second line';
    expect(findNotebookCommand(text, text.length)).toBeNull();
    expect(findNotebookCommand(text, 9)).not.toBeNull();
  });

  it('does not let a command span a newline', () => {
    const text = '/notebook\nsecond';
    const span = findNotebookCommand(text, 9)!;
    expect(text.slice(span.start, span.end)).toBe('/notebook');
  });
});

describe('consuming the command', () => {
  /** The requirement: the draft survives choosing a notebook. */
  it('keeps the question already being drafted', () => {
    const text = 'What is the rated duty of P-101? /notebook';
    const span = findNotebookCommand(text, text.length)!;
    expect(consumeCommand(text, span)).toBe('What is the rated duty of P-101?');
  });

  it('removes the command from the middle without leaving a double space', () => {
    const text = 'before /notebook';
    const span = findNotebookCommand(text, text.length)!;
    expect(consumeCommand(text, span)).toBe('before');
  });

  /** The command word must never be sent to the model. */
  it('leaves no trace of the command token', () => {
    for (const text of ['/notebook', 'ask this /notebook', '/notebook unit four']) {
      const span = findNotebookCommand(text, text.indexOf('/notebook') + 9)!;
      expect(consumeCommand(text, span)).not.toContain('/notebook');
    }
  });

  it('puts the caret where the command was', () => {
    const text = 'What is this? /notebook';
    const span = findNotebookCommand(text, text.length)!;
    expect(caretAfterConsume(span)).toBe(text.indexOf('/notebook'));
  });

  it('keeps a second line intact', () => {
    const text = '/notebook\nsecond line';
    const span = findNotebookCommand(text, 9)!;
    expect(consumeCommand(text, span)).toBe('\nsecond line');
  });
});

describe('filtering the chooser', () => {
  const notebooks = [
    { id: '1', name: 'Procurement', documentCount: 3 },
    { id: '2', name: 'Unit Four', documentCount: 8 },
    { id: '3', name: 'Unit Four Handover', documentCount: 2 },
    { id: '4', name: 'Spare parts for Unit Four', documentCount: 1 },
  ];

  it('returns everything for an empty query', () => {
    expect(filterNotebooks(notebooks, '')).toHaveLength(4);
    expect(filterNotebooks(notebooks, '   ')).toHaveLength(4);
  });

  it('puts a prefix match ahead of a contains match', () => {
    const found = filterNotebooks(notebooks, 'unit');
    expect(found.map(n => n.name)).toEqual([
      'Unit Four',
      'Unit Four Handover',
      'Spare parts for Unit Four',
    ]);
  });

  it('ignores case', () => {
    expect(filterNotebooks(notebooks, 'PROCUREMENT')).toHaveLength(1);
  });

  it('returns nothing when nothing matches, rather than everything', () => {
    expect(filterNotebooks(notebooks, 'zzzz')).toHaveLength(0);
  });
});

describe('keyboard navigation', () => {
  it('wraps at both ends', () => {
    expect(moveHighlight(0, -1, 3)).toBe(2);
    expect(moveHighlight(2, 1, 3)).toBe(0);
    expect(moveHighlight(0, 1, 3)).toBe(1);
  });

  it('selects nothing in an empty list', () => {
    expect(moveHighlight(0, 1, 0)).toBe(-1);
  });
});

describe('freezing a turn scope', () => {
  const scope = (selection: NotebookScope['selection']): NotebookScope => ({
    notebookId: 'nb-a',
    notebookName: 'Unit Four',
    documentCount: 3,
    selection,
  });

  it('sends nothing at all when no notebook is attached', () => {
    expect(freezeScope(null, [SHA_A])).toBeUndefined();
  });

  /**
   * "All" is resolved to the sources that existed at freeze time. A source
   * added while a question waits in the queue must not join a turn that was
   * scoped before it existed.
   */
  it('resolves "all" to the sources that existed when it was frozen', () => {
    const frozen = freezeScope(scope(allSources()), [SHA_A, SHA_B])!;
    expect(frozen.sources).toEqual(someSources([SHA_A, SHA_B]));

    const laterWithAnExtraSource = freezeScope(scope(allSources()), [SHA_A, SHA_B, SHA_C])!;
    expect(sameScope(frozen, laterWithAnExtraSource)).toBe(false);
  });

  it('carries a subset through unchanged', () => {
    const frozen = freezeScope(scope(someSources([SHA_B])), [SHA_A, SHA_B])!;
    expect(frozen.sources).toEqual(someSources([SHA_B]));
  });

  /** The distinction the whole selection type exists to preserve. */
  it('keeps "none" as none rather than turning it into everything', () => {
    const frozen = freezeScope(scope(noSources()), [SHA_A, SHA_B])!;
    expect(frozen.sources).toEqual(noSources());
    expect(frozen.sources.mode).not.toBe('all');
  });

  it('never produces an empty subset, which the backend refuses', () => {
    const frozen = freezeScope(scope(allSources()), [])!;
    expect(frozen.sources).toEqual(allSources());
  });

  it('always states a selection', () => {
    for (const selection of [allSources(), someSources([SHA_A]), noSources()]) {
      expect(freezeScope(scope(selection), [SHA_A])!.sources).toBeDefined();
    }
  });
});

describe('the queued-turn guarantee', () => {
  /**
   * The critical case. A question is typed against notebook A and queued
   * behind a run. The chip is then switched to B. When the queue drains, the
   * waiting question must still be answered from A — and the next question
   * typed afterwards from B.
   */
  it('a queued turn keeps the notebook it was asked against', () => {
    const a: NotebookScope = {
      notebookId: 'nb-a',
      notebookName: 'Unit Four',
      documentCount: 2,
      selection: allSources(),
    };
    const b: NotebookScope = {
      notebookId: 'nb-b',
      notebookName: 'Procurement',
      documentCount: 1,
      selection: allSources(),
    };

    // Typed and queued against A.
    const queued = freezeScope(a, [SHA_A, SHA_B]);
    // The person switches the chip to B while it waits.
    const next = freezeScope(b, [SHA_C]);

    expect(queued!.notebookId).toBe('nb-a');
    expect(queued!.sources).toEqual(someSources([SHA_A, SHA_B]));
    expect(next!.notebookId).toBe('nb-b');
    expect(sameScope(queued, next)).toBe(false);
  });

  /**
   * The frozen value is a copy. Changing the live selection afterwards must
   * not reach into a turn that is already waiting — a shared array here would
   * mean deselecting a source silently rewrote a question already asked.
   */
  it('freezing copies rather than aliasing the live selection', () => {
    const live = [SHA_A, SHA_B];
    const scope: NotebookScope = {
      notebookId: 'nb-a',
      notebookName: 'Unit Four',
      documentCount: 2,
      selection: someSources(live),
    };

    const queued = freezeScope(scope, [])!;
    live.push(SHA_C);

    expect(queued.sources).toEqual(someSources([SHA_A, SHA_B]));
  });

  it('two turns frozen against the same scope compare equal', () => {
    const scope: NotebookScope = {
      notebookId: 'nb-a',
      notebookName: 'Unit Four',
      documentCount: 2,
      selection: someSources([SHA_B, SHA_A]),
    };
    expect(sameScope(freezeScope(scope, []), freezeScope(scope, []))).toBe(true);
  });

  it('order within a subset does not make two scopes differ', () => {
    expect(
      sameScope(
        { notebookId: 'nb', sources: someSources([SHA_A, SHA_B]) },
        { notebookId: 'nb', sources: someSources([SHA_B, SHA_A]) },
      ),
    ).toBe(true);
  });

  it('a notebook with no chip differs from one with a chip', () => {
    expect(sameScope(undefined, { notebookId: 'nb', sources: allSources() })).toBe(false);
    expect(sameScope(undefined, undefined)).toBe(true);
  });
});
