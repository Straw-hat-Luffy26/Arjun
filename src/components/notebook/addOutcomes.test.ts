/**
 * The bug this pins: "already here" reported as a failure.
 *
 * A document the notebook already holds comes back `added: false` *and* with a
 * problem sentence, exactly like a file that could not be read. A tally that
 * counts problems rather than missing content addresses tells somebody their
 * document failed when nothing went wrong, and sends them off to re-read a scan
 * that is already indexed.
 */
import { describe, expect, it } from 'vitest';
import { problemRows, summariseAdds, tallyAdds } from './addOutcomes';
import type { AddedDocument } from '../../services/notebook.service';

function outcome(over: Partial<AddedDocument> = {}): AddedDocument {
  return {
    name: 'pump.pdf',
    sha256: 'sha-1',
    pages: 3,
    added: true,
    problem: null,
    ...over,
  };
}

const addedFile = outcome();
const alreadyHeld = outcome({
  name: 'contract.pdf',
  sha256: 'sha-2',
  added: false,
  problem: 'This notebook already has that document.',
});
const unreadable = outcome({
  name: 'scan.png',
  sha256: null,
  pages: 0,
  added: false,
  problem: 'the document could not be read',
});

describe('tallyAdds', () => {
  it('separates a document already held from one that could not be read', () => {
    expect(tallyAdds([addedFile, alreadyHeld, unreadable])).toEqual({
      added: 1,
      already: 1,
      failed: 1,
    });
  });

  it('does not count an already-held document as a failure', () => {
    // Both carry a problem sentence; only one of them is a failure.
    expect(tallyAdds([alreadyHeld]).failed).toBe(0);
    expect(tallyAdds([alreadyHeld]).already).toBe(1);
  });

  it('counts an empty batch as nothing at all', () => {
    expect(tallyAdds([])).toEqual({ added: 0, already: 0, failed: 0 });
  });
});

describe('summariseAdds', () => {
  it('names all three outcomes when all three happened', () => {
    expect(summariseAdds([addedFile, alreadyHeld, unreadable])).toBe(
      '1 added · 1 already here · 1 could not be read',
    );
  });

  it('omits the counts that are zero', () => {
    expect(summariseAdds([addedFile])).toBe('1 added');
    expect(summariseAdds([alreadyHeld])).toBe('1 already here');
  });

  it('says so plainly when there was nothing to add', () => {
    expect(summariseAdds([])).toBe('Nothing to add.');
  });
});

describe('problemRows', () => {
  it('keeps every row carrying a sentence, whatever kind', () => {
    expect(problemRows([addedFile, alreadyHeld, unreadable]).map((row) => row.name)).toEqual([
      'contract.pdf',
      'scan.png',
    ]);
  });

  it('is empty when everything went in cleanly', () => {
    expect(problemRows([addedFile])).toEqual([]);
  });
});
