/**
 * What these pin: a typing pass reported as a success when it mostly failed.
 *
 * The verification gate drops any claim whose quote is not in the passage it
 * cited. If the screen reports only what survived, a run where the model
 * fabricated four fifths of its output looks identical to a small, clean graph —
 * and the person reading it has no reason to suspect the model is wrong for the
 * job. The drop counts and both sides of every ratio are the product here.
 */
import { describe, expect, it } from 'vitest';
import {
  describeCoverage,
  describeRun,
  describeVerification,
  totalDropped,
  yieldIsPoor,
} from './typingOutcome';
import type { TypingOutcome } from '../../services/notebook.service';

function outcome(over: Partial<TypingOutcome> = {}): TypingOutcome {
  return {
    documentsTotal: 3,
    documentsTyped: 3,
    documentsSkipped: 0,
    documentsFailed: 0,
    proposedNodes: 40,
    proposedEdges: 10,
    keptNodes: 30,
    keptEdges: 8,
    droppedUnknownType: 2,
    droppedUnknownTerm: 3,
    droppedUncited: 1,
    droppedMisquoted: 6,
    typedTerms: 30,
    totalTerms: 412,
    problems: [],
    ...over,
  };
}

describe('describeCoverage', () => {
  it('always gives both sides of the ratio', () => {
    expect(describeCoverage(outcome())).toBe('30 of 412 terms typed');
  });

  it('does not imply a graph exists when none does', () => {
    expect(describeCoverage(outcome({ totalTerms: 0, typedTerms: 0 }))).toBe(
      'Nothing to type yet.',
    );
  });
});

describe('describeVerification', () => {
  it('names how many claims were rejected and why', () => {
    const sentence = describeVerification(outcome());
    expect(sentence).toContain('38 of 50 claims verified');
    expect(sentence).toContain('6 could not be found in the passage cited');
    expect(sentence).toContain('3 named a term not in the graph');
  });

  it('does not invent reasons that did not occur', () => {
    const sentence = describeVerification(
      outcome({
        droppedUnknownType: 0,
        droppedUnknownTerm: 0,
        droppedUncited: 0,
        droppedMisquoted: 0,
        keptNodes: 40,
        keptEdges: 10,
      }),
    );
    expect(sentence).toBe('50 of 50 claims verified.');
  });

  it('says nothing at all when the model proposed nothing', () => {
    // A row of zeroes reads as a malfunction; silence is the honest output.
    expect(
      describeVerification(
        outcome({
          proposedNodes: 0,
          proposedEdges: 0,
          keptNodes: 0,
          keptEdges: 0,
          droppedUnknownType: 0,
          droppedUnknownTerm: 0,
          droppedUncited: 0,
          droppedMisquoted: 0,
        }),
      ),
    ).toBeNull();
  });
});

describe('totalDropped', () => {
  it('counts every rejection reason', () => {
    expect(totalDropped(outcome())).toBe(12);
  });
});

describe('describeRun', () => {
  it('separates typed, already-typed and failed', () => {
    expect(
      describeRun(outcome({ documentsTyped: 1, documentsSkipped: 1, documentsFailed: 1 })),
    ).toBe('1 read · 1 already typed · 1 failed');
  });

  it('omits the counts that are zero', () => {
    expect(
      describeRun(outcome({ documentsTyped: 2, documentsSkipped: 0, documentsFailed: 0 })),
    ).toBe('2 read');
  });

  it('says so plainly when there was nothing to do', () => {
    expect(
      describeRun(outcome({ documentsTyped: 0, documentsSkipped: 0, documentsFailed: 0 })),
    ).toBe('Nothing to do.');
  });
});

describe('yieldIsPoor', () => {
  it('flags a run where most of what the model proposed was rejected', () => {
    expect(yieldIsPoor(outcome({ keptNodes: 5, keptEdges: 1 }))).toBe(true);
  });

  it('does not flag a healthy run', () => {
    expect(yieldIsPoor(outcome())).toBe(false);
  });

  it('does not flag a sample too small to mean anything', () => {
    expect(
      yieldIsPoor(outcome({ proposedNodes: 4, proposedEdges: 0, keptNodes: 1, keptEdges: 0 })),
    ).toBe(false);
  });
});
