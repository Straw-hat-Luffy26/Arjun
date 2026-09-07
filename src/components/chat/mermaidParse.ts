/**
 * Reading back the Mermaid this application writes.
 *
 * `knowledge.build_graph` answers a chat turn with a `graph LR` block, and the
 * model passes it through into its reply. Left as text it renders as a fenced
 * code block: the reader gets the source of a picture instead of the picture.
 *
 * ## Why parse Mermaid instead of drawing from the tool result directly
 *
 * The diagram arrives inside the assistant's message, not beside it. By the time
 * it reaches the transcript it is prose the model composed, and the structured
 * result that produced it is two layers back. Parsing what is actually on screen
 * means the picture always matches the words around it - including when the
 * model draws only part of what it was given.
 *
 * ## Why not the mermaid library
 *
 * `src/components/graph` already renders graphs, hand-rolled, on a canvas: a
 * force simulation, hit-testing, theme-aware colours, node types drawn as
 * shapes. Adding a rendering library to draw the same thing a second way would
 * mean two graph looks in one application, plus a megabyte of dependency and an
 * SBOM entry, to display output this repository generates itself.
 *
 * So this reads only the grammar `render_mermaid` emits, and hands the result to
 * the canvas that already exists. It is a reader for one writer, and says so
 * rather than pretending to be a Mermaid parser - anything it does not
 * recognise is skipped, and a block that yields no nodes is reported as
 * unparsed so the caller can fall back to showing the source.
 */

import type { GraphEdge, GraphNode } from '../../services/notebook.service';

/** A node declaration: `  n0["Acme Pumps Ltd (supplier)"]` */
const NODE = /^\s*([A-Za-z][\w-]*)\[\s*"([^"]*)"\s*\]\s*$/;

/**
 * An edge, solid or dashed, with a quoted label.
 *
 * `  n0 -->|"manufacturer"| n1` is a relation read out of the documents.
 * `  n0 -.->|"together in 3"| n1` is co-occurrence and nothing more. The two
 * are kept apart all the way to the canvas, because collapsing them would let a
 * shared passage read as a stated fact.
 */
const EDGE = /^\s*([A-Za-z][\w-]*)\s*(-->|-\.->)\s*\|\s*"([^"]*)"\s*\|\s*([A-Za-z][\w-]*)\s*$/;

/** `(supplier)` at the end of a label, as `render_mermaid` writes the type. */
const TYPE_SUFFIX = /^(.*?)\s*\(([^()]+)\)$/;

/** How a co-occurrence label states its weight: `together in 3`. */
const WEIGHT = /^together in (\d+)$/;

export interface ParsedMermaid {
  nodes: GraphNode[];
  edges: GraphEdge[];
  /** Links the writer said it left out, so the view can say so too. */
  omitted: number;
}

/**
 * Whether a fenced block is one of ours.
 *
 * Checked before parsing so a `mermaid` fence written by hand - a sequence
 * diagram, a flowchart with a syntax this does not read - falls through to the
 * code block rather than rendering as an empty canvas.
 */
export function looksLikeOurGraph(source: string): boolean {
  return /^\s*graph\s+(LR|TD|RL|BT)\b/.test(source);
}

/**
 * Parses a `graph LR` block into what the canvas draws.
 *
 * Returns `null` when nothing recognisable was found. That is not the same as
 * an empty graph: the caller shows the source text instead, which is the honest
 * outcome for a diagram this cannot read.
 */
export function parseMermaidGraph(source: string): ParsedMermaid | null {
  if (!looksLikeOurGraph(source)) return null;

  const labels = new Map<string, string>();
  const types = new Map<string, string | null>();
  const order: string[] = [];
  const edges: GraphEdge[] = [];
  let omitted = 0;

  for (const line of source.split('\n')) {
    const comment = line.match(/^\s*%%\s*(\d+) more link/);
    if (comment) {
      omitted = Number(comment[1]);
      continue;
    }

    const node = line.match(NODE);
    if (node) {
      const [, id, raw] = node;
      const typed = raw.match(TYPE_SUFFIX);
      labels.set(id, typed ? typed[1] : raw);
      types.set(id, typed ? typed[2] : null);
      if (!order.includes(id)) order.push(id);
      continue;
    }

    const edge = line.match(EDGE);
    if (edge) {
      const [, from, arrow, label, to] = edge;
      const dashed = arrow === '-.->';
      const weight = label.match(WEIGHT);
      edges.push({
        source: from,
        target: to,
        kind: 'cooccurrence',
        weight: weight ? Number(weight[1]) : 1,
        // A dashed link carries no relation, whatever its label said. The label
        // is the weight in words, and passing it on as a relation would put
        // "together in 3" on the canvas where a named fact belongs.
        relation: dashed ? null : label,
      });
    }
  }

  if (order.length === 0) return null;

  const degree = (id: string) =>
    edges.filter((edge) => edge.source === id || edge.target === id).length;

  const nodes: GraphNode[] = order.map((id) => ({
    id,
    label: labels.get(id) ?? id,
    kind: 'term',
    nodeType: types.get(id) ?? null,
    // The diagram does not carry occurrence counts, and inventing one would put
    // a measured-looking number on screen that nothing measured. Degree is what
    // this view actually knows, and it is what sizes the node.
    occurrences: 0,
    degree: degree(id),
    documentSha256: null,
    documentCount: 0,
  }));

  // Edges whose ends are not both declared. The writer does not emit these, but
  // this reads a model's copy of what the writer emitted, and a copy can be
  // truncated mid-block.
  const known = new Set(order);
  const whole = edges.filter((edge) => known.has(edge.source) && known.has(edge.target));

  return { nodes, edges: whole, omitted: omitted + (edges.length - whole.length) };
}
