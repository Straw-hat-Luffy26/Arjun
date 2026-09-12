import { describe, expect, it } from 'vitest';
import { describeClaim, describeEdge } from './edgeLabel';
import type { GraphEdge } from '../../services/notebook.service';

/**
 * What the interface says a link means.
 *
 * The two behaviours worth pinning are the two that were wrong in storage: a
 * direction that must not flip to suit the reader's viewpoint, and a
 * disagreement that must not be resolved by picking a winner.
 */

function cooccurrence(over: Partial<GraphEdge> = {}): GraphEdge {
  return {
    source: 'a',
    target: 'b',
    kind: 'cooccurrence',
    weight: 3,
    relation: null,
    ...over,
  };
}

const claim = {
  id: 'a1',
  subject: 'pv-2201',
  subjectLabel: 'PV-2201',
  predicate: 'supplied by',
  object: 'northern valve company',
  objectLabel: 'Northern Valve Company',
  provenance: 'model' as const,
  status: 'proposed' as const,
  directionCertain: true,
  stale: false,
  evidenceCount: 1,
};

describe('describeEdge', () => {
  it('says only what was observed when nothing has named the link', () => {
    expect(describeEdge(cooccurrence())).toBe('appears with, in 3 passages');
  });

  it('uses a settled label when one claim can stand for the link', () => {
    expect(describeEdge(cooccurrence({ relation: 'supplied by' }))).toBe(
      'supplied by, in 3 passages',
    );
  });

  it('names a disagreement rather than showing one of the two claims', () => {
    const described = describeEdge(
      cooccurrence({ relation: null, contested: true, assertions: [claim] }),
    );
    expect(described).toContain('sources disagree');
    // And it does not assert either reading.
    expect(described).not.toContain('supplied by');
  });

  it('describes a file-to-term link from whichever end is being read', () => {
    const edge: GraphEdge = {
      source: 'doc',
      target: 'term',
      kind: 'appearsIn',
      weight: 2,
      relation: null,
    };
    expect(describeEdge(edge, 'doc')).toBe('contains, in 2 passages');
    expect(describeEdge(edge, 'term')).toBe('appears in, in 2 passages');
  });
});

describe('describeClaim', () => {
  it('reads subject-first whichever node the reader came from', () => {
    // The regression: reversing the words to suit the viewpoint is how the
    // direction was lost in storage, and it would be just as wrong here.
    expect(describeClaim(claim)).toBe(
      'PV-2201 → supplied by → Northern Valve Company (proposed, unreviewed)',
    );
  });

  it('marks a claim whose direction could not be recovered', () => {
    expect(describeClaim({ ...claim, directionCertain: false })).toContain(
      'direction unverified',
    );
  });

  it('separates a confirmed claim from an unreviewed one', () => {
    expect(describeClaim({ ...claim, status: 'accepted' })).toContain('confirmed');
    expect(describeClaim(claim)).toContain('unreviewed');
  });

  it('marks a claim a person wrote as unverified, not as evidence', () => {
    expect(describeClaim({ ...claim, provenance: 'user', status: 'proposed' })).toContain(
      'yours, unverified',
    );
  });

  it('says when a claim was refused', () => {
    expect(describeClaim({ ...claim, status: 'rejected' })).toContain('rejected');
  });
});
