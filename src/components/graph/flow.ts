/**
 * The flow between chosen files: what one shares with the next.
 *
 * The force graph answers "what is in this notebook". It does not answer "how
 * do these three documents connect to each other", because the path from one
 * file to another runs through terms scattered across the picture and tangled
 * with everything else. This lays that path out flat: a file, then the terms it
 * has in common with the next file, then that file, and so on.
 *
 * ## The order is the reader's, not the corpus's
 *
 * Nothing in the extraction says a supplier list comes before a product list.
 * There is no sequence in the data to discover, so the left-to-right order here
 * is exactly the order the person picked the files in, and means nothing more
 * than that. Deriving an order and presenting it as the documents' own would be
 * an invented finding, which is the failure this repository has rules against.
 *
 * ## An empty layer is the answer, not a missing one
 *
 * Two adjacent files that share no terms produce an empty middle layer, and it
 * is returned rather than collapsed. "These two documents have nothing in
 * common" is a real and useful result; a picture that quietly joined them up,
 * or silently dropped the gap, would report the opposite of what was found.
 *
 * Pure, so vitest — which runs here with `environment: 'node'` and no DOM — can
 * test it. Positions are layer and slot indices; turning those into pixels is
 * the canvas's job.
 */
import type { GraphEdge, GraphNode } from '../../services/notebook.service';

export interface FlowNode {
  id: string;
  label: string;
  kind: 'term' | 'document';
  /** Column, left to right. Even layers are files, odd layers are terms. */
  layer: number;
  /** Row within the column. */
  slot: number;
  /** How many nodes share this column, so the caller can space them. */
  slots: number;
}

export interface FlowLink {
  source: string;
  target: string;
  /** Passages of the file that contained the term. A count, not a score. */
  weight: number;
}

/** Two adjacent files with nothing between them. */
export interface FlowGap {
  from: string;
  to: string;
}

export interface FlowView {
  layers: FlowNode[][];
  links: FlowLink[];
  /** Adjacent pairs that share no term, so the screen can say so in words. */
  gaps: FlowGap[];
  /** Distinct terms doing the joining, across every pair. */
  sharedTerms: number;
}

/**
 * Lays out the chain for the chosen files.
 *
 * `files` is in the order the reader picked them. Only `appearsIn` edges are
 * consulted: a term is in a file because the extractor put it there, never
 * because two terms happened to sit near each other.
 */
export function buildFlow(
  files: readonly GraphNode[],
  nodes: readonly GraphNode[],
  edges: readonly GraphEdge[],
): FlowView {
  const layers: FlowNode[][] = [];
  const links: FlowLink[] = [];
  const gaps: FlowGap[] = [];
  const shared = new Set<string>();

  if (files.length === 0) return { layers, links, gaps, sharedTerms: 0 };

  const byId = new Map(nodes.map((node) => [node.id, node]));

  // Which terms each file contributed, and with what weight. Built from the
  // directed membership edges only.
  const termsOf = new Map<string, Map<string, number>>();
  for (const edge of edges) {
    if (edge.kind !== 'appearsIn') continue;
    const forFile = termsOf.get(edge.source) ?? new Map<string, number>();
    forFile.set(edge.target, edge.weight);
    termsOf.set(edge.source, forFile);
  }

  const fileLayer = (file: GraphNode, layer: number): FlowNode[] => [
    { id: file.id, label: file.label, kind: 'document', layer, slot: 0, slots: 1 },
  ];

  layers.push(fileLayer(files[0], 0));

  for (let i = 0; i + 1 < files.length; i += 1) {
    const left = files[i];
    const right = files[i + 1];
    const leftTerms = termsOf.get(left.id) ?? new Map<string, number>();
    const rightTerms = termsOf.get(right.id) ?? new Map<string, number>();

    // In both, and only in both. A term in one file alone says nothing about
    // how these two are connected, and putting it in the middle column would
    // read as though it did.
    const between = [...leftTerms.keys()]
      .filter((id) => rightTerms.has(id))
      .map((id) => byId.get(id))
      .filter((node): node is GraphNode => node !== undefined)
      // Busiest first, then alphabetically: a stable order, and the one a
      // reader wants when the column is longer than the screen.
      .sort((a, b) => b.occurrences - a.occurrences || a.label.localeCompare(b.label));

    const layer = layers.length;
    layers.push(
      between.map((node, slot) => ({
        id: node.id,
        label: node.label,
        kind: 'term' as const,
        layer,
        slot,
        slots: between.length,
      })),
    );

    for (const node of between) {
      shared.add(node.id);
      links.push({ source: left.id, target: node.id, weight: leftTerms.get(node.id) ?? 1 });
      links.push({ source: node.id, target: right.id, weight: rightTerms.get(node.id) ?? 1 });
    }
    if (between.length === 0) gaps.push({ from: left.label, to: right.label });

    layers.push(fileLayer(right, layers.length));
  }

  return { layers, links, gaps, sharedTerms: shared.size };
}

/** One sentence saying what the chain found. Every number is counted. */
export function describeFlow(flow: FlowView): string {
  const files = flow.layers.filter((layer) => layer[0]?.kind === 'document').length;
  if (files < 2) return 'Pick a second file to trace what they share.';
  const terms = `${flow.sharedTerms} ${flow.sharedTerms === 1 ? 'term' : 'terms'}`;
  if (flow.gaps.length === 0) return `${files} files, joined by ${terms}.`;
  const pairs = flow.gaps.map((gap) => `${gap.from} and ${gap.to}`).join('; ');
  // The gap is the finding, so it leads rather than trailing as a caveat.
  return flow.sharedTerms === 0
    ? `These files share no terms at all — nothing links ${pairs}.`
    : `${files} files, joined by ${terms}. Nothing links ${pairs}.`;
}
