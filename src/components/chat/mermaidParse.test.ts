import { describe, expect, it } from 'vitest';

import { looksLikeOurGraph, parseMermaidGraph } from './mermaidParse';

/**
 * Exactly what `render_mermaid` writes, character for character.
 *
 * Pinned rather than paraphrased: this module is a reader for one writer, and a
 * reader tested against a hand-typed approximation of its writer passes right up
 * until the writer changes a space.
 *
 * The writer asserts this exact string from its own side, in
 * `knowledge/graph/render.rs`, in the test named
 * `the_emitted_format_is_exactly_what_the_chat_surface_parses`. Change one and
 * the other fails, which is the point: two halves of one contract, checked
 * against one artifact rather than against each other's descriptions.
 */
const REAL_OUTPUT = [
  'graph LR',
  '  n0["Acme Pumps Ltd (supplier)"]',
  '  n1["PV-2201 (equipment)"]',
  '  n2["Refining Division"]',
  '  n0 -->|"manufacturer"| n1',
  '  n1 -.->|"together in 3"| n2',
].join('\n');

describe('parseMermaidGraph', () => {
  it('reads nodes, splitting the type off the label', () => {
    const parsed = parseMermaidGraph(REAL_OUTPUT);

    expect(parsed).not.toBeNull();
    expect(parsed!.nodes.map((n) => [n.label, n.nodeType])).toEqual([
      ['Acme Pumps Ltd', 'supplier'],
      ['PV-2201', 'equipment'],
      ['Refining Division', null],
    ]);
  });

  /**
   * The distinction the whole feature rests on. A solid arrow is a relation the
   * documents stated; a dashed one means only that two terms shared a passage.
   * Carrying "together in 3" through as a relation would put a co-occurrence on
   * the canvas dressed as a fact.
   */
  it('keeps a named relation and refuses to invent one for a dashed link', () => {
    const parsed = parseMermaidGraph(REAL_OUTPUT)!;

    const named = parsed.edges.find((e) => e.source === 'n0')!;
    expect(named.relation).toBe('manufacturer');

    const cooccurrence = parsed.edges.find((e) => e.source === 'n1')!;
    expect(cooccurrence.relation).toBeNull();
    expect(cooccurrence.weight).toBe(3);
  });

  it('carries through the count of links the writer left out', () => {
    const truncated = [REAL_OUTPUT, '  %% 12 more link(s) not drawn'].join('\n');
    expect(parseMermaidGraph(truncated)!.omitted).toBe(12);
  });

  it('drops an edge whose ends were lost to truncation, and counts it', () => {
    const cut = ['graph LR', '  n0["Acme Pumps Ltd"]', '  n0 -->|"manufacturer"| n9'].join('\n');

    const parsed = parseMermaidGraph(cut)!;
    expect(parsed.edges).toHaveLength(0);
    expect(parsed.omitted).toBe(1);
    expect(parsed.nodes).toHaveLength(1);
  });

  it('gives degree from the links actually drawn', () => {
    const parsed = parseMermaidGraph(REAL_OUTPUT)!;
    expect(parsed.nodes.find((n) => n.id === 'n1')!.degree).toBe(2);
    expect(parsed.nodes.find((n) => n.id === 'n0')!.degree).toBe(1);
  });

  /**
   * Occurrences are not in the diagram. Reporting anything but zero would put a
   * number on screen that looks measured and was not - the failure mode
   * CLAUDE.md calls out by name.
   */
  it('does not invent an occurrence count', () => {
    const parsed = parseMermaidGraph(REAL_OUTPUT)!;
    expect(parsed.nodes.every((n) => n.occurrences === 0)).toBe(true);
  });

  it('returns null for a diagram it does not write', () => {
    expect(parseMermaidGraph('sequenceDiagram\n  A->>B: hi')).toBeNull();
    expect(parseMermaidGraph('graph LR')).toBeNull();
    expect(parseMermaidGraph('')).toBeNull();
  });

  it('recognises only graph declarations as ours', () => {
    expect(looksLikeOurGraph('graph LR\n  n0["a"]')).toBe(true);
    expect(looksLikeOurGraph('graph TD\n  n0["a"]')).toBe(true);
    expect(looksLikeOurGraph('pie title Nope')).toBe(false);
    expect(looksLikeOurGraph('flowchart LR\n  a --> b')).toBe(false);
  });

  it('reads the empty diagram the writer emits for an empty selection', () => {
    const parsed = parseMermaidGraph('graph LR\n  empty["nothing to draw"]')!;
    expect(parsed.nodes).toHaveLength(1);
    expect(parsed.nodes[0].label).toBe('nothing to draw');
    expect(parsed.edges).toHaveLength(0);
  });
});
