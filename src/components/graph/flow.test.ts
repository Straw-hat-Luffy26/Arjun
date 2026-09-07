/**
 * What these pin: a chain that looks connected when it is not.
 *
 * The flow view exists to answer "how are these documents linked", and the
 * honest answer is sometimes "they are not". Two files with nothing in common
 * must produce a visible gap and a sentence that says so — a picture that
 * quietly bridged them, or dropped the pair, would report the opposite of what
 * the extractor found. The rest holds the middle column to terms genuinely in
 * *both* neighbours, since a term from one file alone says nothing about how
 * two files connect.
 */
import { describe, expect, it } from 'vitest';
import { buildFlow, describeFlow } from './flow';
import type { GraphEdge, GraphNode } from '../../services/notebook.service';

function file(id: string, label: string): GraphNode {
  return {
    id,
    label,
    kind: 'document',
    nodeType: null,
    occurrences: 0,
    degree: 0,
    documentSha256: `sha-${id}`,
    documentCount: 0,
  };
}

function term(id: string, label: string, occurrences = 2): GraphNode {
  return {
    id,
    label,
    kind: 'term',
    nodeType: null,
    occurrences,
    degree: 2,
    documentSha256: null,
    documentCount: 2,
  };
}

function inFile(fileId: string, termId: string, weight = 1): GraphEdge {
  return { source: fileId, target: termId, kind: 'appearsIn', weight, relation: null };
}

const SUPPLIERS = file('f1', 'suppliers.pdf');
const CUSTOMERS = file('f2', 'customers.pdf');
const PRODUCTS = file('f3', 'products.pdf');
const VALVE = term('t1', 'Northern Valve', 5);
const UNIT = term('t2', 'Unit Four', 3);
const GASKET = term('t3', 'Gasket', 4);

describe('buildFlow', () => {
  it('alternates file, shared terms, file', () => {
    const flow = buildFlow(
      [SUPPLIERS, CUSTOMERS],
      [SUPPLIERS, CUSTOMERS, VALVE],
      [inFile('f1', 't1', 5), inFile('f2', 't1', 2)],
    );
    expect(flow.layers).toHaveLength(3);
    expect(flow.layers[0].map((n) => n.label)).toEqual(['suppliers.pdf']);
    expect(flow.layers[1].map((n) => n.label)).toEqual(['Northern Valve']);
    expect(flow.layers[2].map((n) => n.label)).toEqual(['customers.pdf']);
    // Each shared term is joined to the file on either side of it.
    expect(flow.links).toEqual([
      { source: 'f1', target: 't1', weight: 5 },
      { source: 't1', target: 'f2', weight: 2 },
    ]);
  });

  it('puts a term in the middle only when it is in both neighbours', () => {
    // Gasket is in suppliers alone. It says nothing about how suppliers and
    // customers connect, so it does not appear between them.
    const flow = buildFlow(
      [SUPPLIERS, CUSTOMERS],
      [SUPPLIERS, CUSTOMERS, VALVE, GASKET],
      [inFile('f1', 't1'), inFile('f2', 't1'), inFile('f1', 't3')],
    );
    expect(flow.layers[1].map((n) => n.label)).toEqual(['Northern Valve']);
    expect(flow.sharedTerms).toBe(1);
  });

  it('keeps an empty layer, and reports the gap, when two files share nothing', () => {
    const flow = buildFlow(
      [SUPPLIERS, CUSTOMERS],
      [SUPPLIERS, CUSTOMERS, VALVE, GASKET],
      [inFile('f1', 't1'), inFile('f2', 't3')],
    );
    expect(flow.layers).toHaveLength(3);
    expect(flow.layers[1]).toEqual([]);
    expect(flow.gaps).toEqual([{ from: 'suppliers.pdf', to: 'customers.pdf' }]);
    expect(describeFlow(flow)).toContain('share no terms at all');
  });

  it('chains three files through two separate middles', () => {
    const flow = buildFlow(
      [SUPPLIERS, CUSTOMERS, PRODUCTS],
      [SUPPLIERS, CUSTOMERS, PRODUCTS, VALVE, UNIT, GASKET],
      [
        inFile('f1', 't1'),
        inFile('f2', 't1'),
        inFile('f1', 't2'),
        inFile('f2', 't2'),
        inFile('f2', 't3'),
        inFile('f3', 't3'),
      ],
    );
    expect(flow.layers.map((layer) => layer.length)).toEqual([1, 2, 1, 1, 1]);
    // Busiest first: Northern Valve at 5 occurrences outranks Unit Four at 3.
    expect(flow.layers[1].map((n) => n.label)).toEqual(['Northern Valve', 'Unit Four']);
    expect(flow.layers[3].map((n) => n.label)).toEqual(['Gasket']);
    expect(flow.sharedTerms).toBe(3);
    expect(describeFlow(flow)).toBe('3 files, joined by 3 terms.');
  });

  it('never routes the chain through a co-occurrence', () => {
    // Two terms sitting in the same passage is not a route from one file to
    // another. Only membership edges may carry the flow.
    const cooccurrence: GraphEdge = {
      source: 't1',
      target: 't3',
      kind: 'cooccurrence',
      weight: 9,
      relation: null,
    };
    const flow = buildFlow(
      [SUPPLIERS, CUSTOMERS],
      [SUPPLIERS, CUSTOMERS, VALVE, GASKET],
      [inFile('f1', 't1'), inFile('f2', 't3'), cooccurrence],
    );
    expect(flow.layers[1]).toEqual([]);
    expect(flow.links).toEqual([]);
  });

  it('says what to do when only one file is picked', () => {
    const flow = buildFlow([SUPPLIERS], [SUPPLIERS], []);
    expect(flow.layers).toHaveLength(1);
    expect(describeFlow(flow)).toBe('Pick a second file to trace what they share.');
  });

  it('has nothing to draw for no files', () => {
    const flow = buildFlow([], [], []);
    expect(flow.layers).toEqual([]);
    expect(flow.sharedTerms).toBe(0);
  });

  it('names both the join and the gap when a chain is only partly connected', () => {
    const flow = buildFlow(
      [SUPPLIERS, CUSTOMERS, PRODUCTS],
      [SUPPLIERS, CUSTOMERS, PRODUCTS, VALVE],
      [inFile('f1', 't1'), inFile('f2', 't1')],
    );
    expect(describeFlow(flow)).toBe(
      '3 files, joined by 1 term. Nothing links customers.pdf and products.pdf.',
    );
  });
});
