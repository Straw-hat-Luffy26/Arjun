/**
 * Reading the Mermaid a model actually writes.
 *
 * The cases here are not invented: each one is a spelling a model reaches for
 * routinely and that the previous reader rejected, sending a diagram to the
 * screen as a code block. `flowchart` instead of `graph`, unquoted labels,
 * nodes declared inside the edge line, edges with no label, and `erDiagram` in
 * any form at all.
 */
import { describe, expect, it } from 'vitest';

import { isSupportedDiagram, parseDiagram } from './mermaidDiagram';

describe('flowcharts as a model writes them', () => {
  it('reads the spelling Mermaid documents and the old reader refused', () => {
    // `flowchart TD` with nodes declared inline and one unlabelled edge — the
    // single most common shape of a model-written diagram.
    const parsed = parseDiagram(['flowchart TD', '  A[Start] --> B{Approved?}'].join('\n'));

    expect(parsed).not.toBeNull();
    expect(parsed!.kind).toBe('flowchart');
    expect(parsed!.nodes.map(n => [n.id, n.label, n.nodeType])).toEqual([
      ['A', 'Start', 'process'],
      ['B', 'Approved?', 'decision'],
    ]);
    expect(parsed!.edges).toHaveLength(1);
    expect(parsed!.edges[0]).toMatchObject({ source: 'A', target: 'B', relation: null });
  });

  it('reads a chain declared on one line', () => {
    const parsed = parseDiagram(['graph LR', '  A --> B --> C'].join('\n'))!;
    expect(parsed.nodes.map(n => n.id)).toEqual(['A', 'B', 'C']);
    expect(parsed.edges.map(e => [e.source, e.target])).toEqual([
      ['A', 'B'],
      ['B', 'C'],
    ]);
  });

  it('reads an edge label in both spellings', () => {
    const piped = parseDiagram(['flowchart LR', '  A -->|yes| B'].join('\n'))!;
    const inline = parseDiagram(['flowchart LR', '  A -- yes --> B'].join('\n'))!;

    expect(piped.edges[0].relation).toBe('yes');
    expect(inline.edges[0].relation).toBe('yes');
  });

  it('keeps a dashed link unnamed, whatever its label said', () => {
    // The same rule the writer's own reader enforces: a dotted line is not a
    // stated relationship, so no label of one is carried through.
    const parsed = parseDiagram(['flowchart LR', '  A -.->|maybe| B'].join('\n'))!;
    expect(parsed.edges[0].relation).toBeNull();
  });

  it('keeps a label that contains brackets of its own', () => {
    // `A[Cost (net)]` is legal, and a pattern that stopped at the first ")"
    // would cut the label in half.
    const parsed = parseDiagram(['flowchart LR', '  A[Cost (net)] --> B[Total]'].join('\n'))!;
    expect(parsed.nodes[0].label).toBe('Cost (net)');
    expect(parsed.nodes[1].label).toBe('Total');
  });

  it('distinguishes the node shapes a flowchart uses', () => {
    const parsed = parseDiagram(
      [
        'flowchart TD',
        '  A([Begin]) --> C((Hub))',
        '  C --> D{{Prepare}}',
        '  E[(Store)] --> F[[Sub]]',
      ].join('\n'),
    )!;

    const types = new Map(parsed.nodes.map(n => [n.id, n.nodeType]));
    expect(types.get('A')).toBe('terminal');
    expect(types.get('C')).toBe('circle');
    expect(types.get('D')).toBe('hexagon');
    expect(types.get('E')).toBe('store');
    expect(types.get('F')).toBe('subroutine');
  });

  it('strips the markup Mermaid permits inside a label', () => {
    const parsed = parseDiagram(['flowchart LR', '  A["Order<br/>placed"] --> B'].join('\n'))!;
    // Not passed through: the canvas draws text, and a box reading "<br/>"
    // looks to a reader like the app is broken.
    expect(parsed.nodes[0].label).toBe('Order placed');
  });

  it('skips styling lines instead of failing on them', () => {
    const parsed = parseDiagram(
      [
        'flowchart TD',
        '  classDef warn fill:#f00',
        '  subgraph Cluster',
        '  A[One] --> B[Two]',
        '  end',
        '  style A fill:#0f0',
        '  click A callback',
      ].join('\n'),
    )!;

    expect(parsed.nodes.map(n => n.id)).toEqual(['A', 'B']);
    expect(parsed.edges).toHaveLength(1);
  });

  it('draws what it can from a reply cut off mid-diagram', () => {
    const parsed = parseDiagram(['flowchart LR', '  A[One] --> B[Two]', '  B --> '].join('\n'))!;
    expect(parsed.edges).toHaveLength(1);
    expect(parsed.nodes.map(n => n.id)).toEqual(['A', 'B']);
  });

  it('gives degree from the links actually drawn', () => {
    const parsed = parseDiagram(['flowchart LR', '  A --> B', '  B --> C'].join('\n'))!;
    expect(parsed.nodes.find(n => n.id === 'B')!.degree).toBe(2);
  });

  it('does not invent an occurrence count', () => {
    const parsed = parseDiagram(['flowchart LR', '  A --> B'].join('\n'))!;
    expect(parsed.nodes.every(n => n.occurrences === 0)).toBe(true);
  });
});

describe('ER diagrams', () => {
  const SCHEMA = [
    'erDiagram',
    '    CUSTOMER ||--o{ ORDER : places',
    '    ORDER ||--|{ LINE_ITEM : contains',
    '    CUSTOMER {',
    '        string name',
    '        string custNumber PK',
    '        int age FK "years"',
    '    }',
  ].join('\n');

  it('reads entities and relationships', () => {
    const parsed = parseDiagram(SCHEMA)!;
    expect(parsed.kind).toBe('er');
    expect(parsed.nodes.map(n => n.id)).toEqual(['CUSTOMER', 'ORDER', 'LINE_ITEM']);
    expect(parsed.nodes.every(n => n.nodeType === 'entity')).toBe(true);
    expect(parsed.edges).toHaveLength(2);
  });

  it('carries the cardinality, not just the fact of a link', () => {
    // Cardinality is the entire content of an ER relationship. A picture that
    // said only "these two are connected" would be a different diagram.
    const parsed = parseDiagram(SCHEMA)!;
    expect(parsed.edges[0].relation).toBe('places (1 → 0..n)');
    expect(parsed.edges[1].relation).toBe('contains (1 → 1..n)');
  });

  it('reads every cardinality marker Mermaid defines', () => {
    const parsed = parseDiagram(
      [
        'erDiagram',
        '    A |o--o| B : optional',
        '    C }o..o{ D : many',
        '    E }|--|{ F : atLeastOne',
      ].join('\n'),
    )!;

    expect(parsed.edges.map(e => e.relation)).toEqual([
      'optional (0..1 → 0..1)',
      'many (0..n → 0..n)',
      'atLeastOne (1..n → 1..n)',
    ]);
  });

  it('reads an entity block, including key markers and comments', () => {
    const parsed = parseDiagram(SCHEMA)!;
    const customer = parsed.entities.find(e => e.id === 'CUSTOMER')!;

    expect(customer.attributes).toEqual([
      { type: 'string', name: 'name', key: null },
      { type: 'string', name: 'custNumber', key: 'PK' },
      { type: 'int', name: 'age', key: 'FK' },
    ]);
  });

  it('lists an entity that has no attribute block', () => {
    // Declared only by a relationship. It is still an entity and still drawn.
    const parsed = parseDiagram(SCHEMA)!;
    expect(parsed.entities.find(e => e.id === 'ORDER')!.attributes).toEqual([]);
  });

  it('keeps an entity whose block a truncated reply never closed', () => {
    const cut = ['erDiagram', '    CUSTOMER {', '        string name'].join('\n');
    const parsed = parseDiagram(cut)!;
    expect(parsed.entities.find(e => e.id === 'CUSTOMER')!.attributes).toHaveLength(1);
  });

  it('reads a quoted relationship label', () => {
    const parsed = parseDiagram(['erDiagram', '    A ||--|| B : "is paid by"'].join('\n'))!;
    expect(parsed.edges[0].relation).toBe('is paid by (1 → 1)');
  });
});

describe('what it refuses to draw', () => {
  it('returns null for a diagram type it does not read', () => {
    // Better the source than a wrong picture: a sequence diagram drawn as a
    // force graph would misrepresent it entirely.
    expect(parseDiagram('sequenceDiagram\n  A->>B: hi')).toBeNull();
    expect(parseDiagram('pie title Nope')).toBeNull();
    expect(parseDiagram('gantt\n  title X')).toBeNull();
  });

  it('returns null for a header with no body', () => {
    expect(parseDiagram('flowchart TD')).toBeNull();
    expect(parseDiagram('erDiagram')).toBeNull();
    expect(parseDiagram('')).toBeNull();
  });

  it('agrees with itself about what it supports', () => {
    // `isSupportedDiagram` gates the attempt; disagreeing with `parseDiagram`
    // would mean either a silent fallback or a promise it cannot keep.
    const supported = [
      'flowchart TD\n A --> B',
      'erDiagram\n A ||--|| B : x',
      'graph LR\n A --> B',
    ];
    for (const source of supported) {
      expect(isSupportedDiagram(source), source).toBe(true);
      expect(parseDiagram(source), source).not.toBeNull();
    }
    for (const source of ['sequenceDiagram\n A->>B: hi', 'pie title Nope']) {
      expect(isSupportedDiagram(source), source).toBe(false);
    }
  });
});
