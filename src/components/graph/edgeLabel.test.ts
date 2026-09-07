/**
 * What these pin: a co-occurrence presented as a relation.
 *
 * The statistical pass observes that two terms shared a passage and nothing
 * more. If the interface renders that as a bare line, or borrows a verb for it,
 * the graph is asserting something no pass ever verified — the same class of
 * failure as publishing a benchmark nobody ran. These tests hold the wording to
 * what was actually observed, and hold the typed case to showing the model's
 * relation only when there is one.
 */
import { describe, expect, it } from 'vitest';
import { describeEdge } from './edgeLabel';
import type { GraphEdge } from '../../services/notebook.service';

function edge(over: Partial<GraphEdge> = {}): GraphEdge {
  return {
    source: 'a',
    target: 'b',
    kind: 'cooccurrence',
    weight: 3,
    relation: null,
    ...over,
  };
}

describe('describeEdge', () => {
  it('says only that two terms appeared together, and in how many passages', () => {
    expect(describeEdge(edge())).toBe('appears with, in 3 passages');
  });

  it('never drops the count to a bare "appears with"', () => {
    // The count is the whole evidential content of a statistical edge. Without
    // it the line says two terms are related, which is not what was observed.
    expect(describeEdge(edge({ weight: 1 }))).toBe('appears with, in 1 passage');
  });

  it('names a relation once the typing pass has verified one', () => {
    expect(describeEdge(edge({ relation: 'supplies' }))).toBe('supplies, in 3 passages');
  });

  it('describes a file membership as membership, not as a relation', () => {
    // Read from the file, which is where the arrow starts.
    expect(describeEdge(edge({ kind: 'appearsIn', weight: 2 }))).toBe('contains, in 2 passages');
  });

  it('reads the same membership from the term end without reversing the fact', () => {
    // The arrow points file → term. Standing at the term, the same edge is
    // "appears in" — not a second relation, the same one from the other end.
    expect(describeEdge(edge({ kind: 'appearsIn', weight: 2 }), 'b')).toBe(
      'appears in, in 2 passages',
    );
    expect(describeEdge(edge({ kind: 'appearsIn', weight: 2 }), 'a')).toBe(
      'contains, in 2 passages',
    );
  });

  it('does not let a stray relation retype a file membership', () => {
    // A membership edge is a fact of the corpus and is never typed. If a
    // relation ever arrived on one it would be a bug upstream, and borrowing it
    // here would put a model's word on a link the model never saw.
    expect(describeEdge(edge({ kind: 'appearsIn', relation: 'supplies', weight: 2 }))).toBe(
      'contains, in 2 passages',
    );
  });

  it('never points a co-occurrence in a direction', () => {
    // Two terms sharing a passage have no direction, so which end the reader
    // stands on cannot change what the edge says — and nothing downstream may
    // draw it with an arrowhead.
    expect(describeEdge(edge(), 'a')).toBe(describeEdge(edge(), 'b'));
  });
});
