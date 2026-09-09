/**
 * Reading the Mermaid a *model* writes, as opposed to the Mermaid this
 * application writes.
 *
 * ## Why this is separate from `mermaidParse.ts`
 *
 * That module is a reader for one writer. It parses exactly the `graph LR` that
 * `knowledge/graph/render.rs` emits, and the two halves are pinned to one
 * another by a test on each side. Widening it to accept everything a language
 * model might type would dissolve the contract it exists to enforce: the point
 * of a reader for one writer is that it fails when the writer changes.
 *
 * So the writer's grammar keeps its exact reader, and this handles the other
 * source of diagrams — a model asked for a flowchart or an ER diagram, writing
 * ordinary Mermaid into its reply.
 *
 * ## The gap this closes
 *
 * The build spec's flow/ER lane is `{mermaid_source}` from the model, rendered
 * client-side. What was actually shipped read `graph LR` with double-quoted
 * labels and a pipe-delimited edge label, and nothing else. Everything a model
 * ordinarily writes fell through to a code block:
 *
 * - `flowchart TD` — the spelling Mermaid's own documentation uses throughout,
 *   and the one a model reaches for first.
 * - `A[Login]` — an unquoted label. Quoting is optional in Mermaid, and models
 *   mostly do not bother.
 * - `A[Start] --> B{Choice}` — nodes declared inside the edge line, which is
 *   how flowcharts are almost always written.
 * - `A --> B` — an edge with no label at all.
 * - `erDiagram` — not read in any form.
 *
 * The old reader was not wrong; it was answering a narrower question than the
 * one the product now asks.
 *
 * ## What it deliberately does not do
 *
 * This is not a Mermaid implementation. It reads flowcharts and ER diagrams,
 * skips the lines it does not recognise (`style`, `classDef`, `click`,
 * `subgraph`), and returns `null` when a source yields no nodes so the caller
 * can fall back to showing the source. A diagram drawn wrong is worse than a
 * diagram shown as the text it came from, so anything ambiguous falls back.
 */

import type { GraphEdge, GraphNode } from '../../services/notebook.service';

/** Which of the two grammars a source turned out to be. */
export type MermaidDiagramKind = 'flowchart' | 'er';

/** One column of an ER entity. The canvas draws boxes, so these go in a table. */
export interface EntityAttribute {
  type: string;
  name: string;
  /** `PK`, `FK`, `UK`, or null. Mermaid's key markers, carried through as-is. */
  key: string | null;
}

export interface EntityBlock {
  id: string;
  label: string;
  attributes: EntityAttribute[];
}

export interface ParsedDiagram {
  kind: MermaidDiagramKind;
  nodes: GraphNode[];
  edges: GraphEdge[];
  /** Links dropped because an endpoint was never declared. */
  omitted: number;
  /** ER attribute blocks. Empty for a flowchart. */
  entities: EntityBlock[];
}

/* ------------------------------------------------------------------ *
 * Shared
 * ------------------------------------------------------------------ */

/** Lines that are styling or structure, not content. Skipped, never failed on. */
const IGNORED =
  /^\s*(%%|style\b|classDef\b|class\b|click\b|linkStyle\b|subgraph\b|end\b|direction\b)/;

function unquote(text: string): string {
  const trimmed = text.trim();
  if (trimmed.length >= 2 && trimmed.startsWith('"') && trimmed.endsWith('"')) {
    return trimmed.slice(1, -1).trim();
  }
  return trimmed;
}

/**
 * Mermaid allows a literal `<br>` and HTML entities inside labels. Turned into
 * plain text rather than passed through: the canvas draws text, not markup, and
 * a box reading `Order<br/>placed` looks like a rendering bug to the reader.
 */
function cleanLabel(text: string): string {
  return unquote(text)
    .replace(/<br\s*\/?>/gi, ' ')
    .replace(/&quot;/g, '"')
    .replace(/&amp;/g, '&')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/\s+/g, ' ')
    .trim();
}

/**
 * Assembles the parse result, computing degree from the edges actually kept.
 *
 * Occurrence counts stay zero for the same reason `mermaidParse` keeps them
 * zero: the diagram does not carry them, and a plausible-looking number that
 * nothing measured is the failure this repository has a standing rule about.
 */
function assemble(
  kind: MermaidDiagramKind,
  order: string[],
  labels: Map<string, string>,
  types: Map<string, string | null>,
  rawEdges: GraphEdge[],
  entities: EntityBlock[],
): ParsedDiagram | null {
  if (order.length === 0) return null;

  const known = new Set(order);
  const edges = rawEdges.filter(edge => known.has(edge.source) && known.has(edge.target));

  const degree = (id: string) =>
    edges.filter(edge => edge.source === id || edge.target === id).length;

  const nodes: GraphNode[] = order.map(id => ({
    id,
    label: labels.get(id) ?? id,
    kind: 'term',
    nodeType: types.get(id) ?? null,
    occurrences: 0,
    degree: degree(id),
    documentSha256: null,
    documentCount: 0,
  }));

  return { kind, nodes, edges, omitted: rawEdges.length - edges.length, entities };
}

/* ------------------------------------------------------------------ *
 * Flowcharts
 * ------------------------------------------------------------------ */

const FLOW_HEADER = /^\s*(?:graph|flowchart)\s+(?:LR|RL|TD|TB|BT)\b/;

/**
 * Node shapes, longest delimiter first so `[[` is never read as `[`.
 *
 * The shape becomes the node's type, which is what gives a decision rhombus a
 * different glyph from a process box on the canvas.
 */
const SHAPES: ReadonlyArray<{ open: string; close: string; type: string }> = [
  { open: '[[', close: ']]', type: 'subroutine' },
  { open: '[(', close: ')]', type: 'store' },
  { open: '([', close: '])', type: 'terminal' },
  { open: '((', close: '))', type: 'circle' },
  { open: '{{', close: '}}', type: 'hexagon' },
  { open: '[', close: ']', type: 'process' },
  { open: '(', close: ')', type: 'rounded' },
  { open: '{', close: '}', type: 'decision' },
  { open: '>', close: ']', type: 'flag' },
];

interface FoundNode {
  id: string;
  label: string;
  type: string;
}

/**
 * Pulls every node declaration out of a line, returning what was found and the
 * line with each declaration reduced to its bare id.
 *
 * Scanned rather than matched with one regular expression because a label may
 * contain the delimiters of another shape — `A[Cost (net)]` is legal Mermaid,
 * and a pattern that stopped at the first `)` would take the label apart.
 */
function extractNodes(line: string): { found: FoundNode[]; stripped: string } {
  const found: FoundNode[] = [];
  let stripped = '';
  let i = 0;

  while (i < line.length) {
    const rest = line.slice(i);
    const idMatch = rest.match(/^[A-Za-z_][\w-]*/);
    if (!idMatch) {
      stripped += line[i];
      i += 1;
      continue;
    }

    const id = idMatch[0];
    const afterId = i + id.length;
    const shape = SHAPES.find(candidate => line.startsWith(candidate.open, afterId));
    if (!shape) {
      stripped += id;
      i = afterId;
      continue;
    }

    const from = afterId + shape.open.length;
    const close = line.indexOf(shape.close, from);
    if (close === -1) {
      // An unterminated shape: left alone rather than guessing where the label
      // was meant to end.
      stripped += id;
      i = afterId;
      continue;
    }

    found.push({ id, label: cleanLabel(line.slice(from, close)), type: shape.type });
    stripped += id;
    i = close + shape.close.length;
  }

  return { found, stripped };
}

/** Rewrites `A -- yes --> B` into the other spelling, `A -->|yes| B`. */
function normaliseInlineLabels(line: string): string {
  return line
    .replace(/--\s+([^->|][^>|]*?)\s+(-->|---)/g, '$2|$1|')
    .replace(/-\.\s+([^.|][^|]*?)\s+\.->/g, '-.->|$1|')
    .replace(/==\s+([^=|][^|]*?)\s+==>/g, '==>|$1|');
}

const ARROWS = ['-.->', '-.-', '-->', '---', '==>', '===', '--x', '--o'] as const;

/** A dashed or dotted link. Kept apart from a solid one, as elsewhere. */
const DASHED: ReadonlySet<string> = new Set(['-.->', '-.-']);

interface Link {
  from: string;
  to: string;
  label: string | null;
  dashed: boolean;
}

/**
 * Reads one edge line, which may be a chain: `A --> B --> C`.
 *
 * Tokenised rather than matched whole, because a chain has no fixed length and
 * a regular expression for "two or more of these" would either miss the third
 * link or accept nonsense between them.
 */
function readLinks(stripped: string): Link[] {
  const links: Link[] = [];
  let cursor = 0;
  let previous: string | null = null;
  let arrow: string | null = null;
  let label: string | null = null;

  while (cursor < stripped.length) {
    const rest = stripped.slice(cursor);

    if (/^\s/.test(rest)) {
      cursor += 1;
      continue;
    }

    const foundArrow = ARROWS.find(candidate => rest.startsWith(candidate));
    if (foundArrow) {
      arrow = foundArrow;
      label = null;
      cursor += foundArrow.length;
      continue;
    }

    if (rest.startsWith('|')) {
      const close = stripped.indexOf('|', cursor + 1);
      if (close === -1) break;
      label = cleanLabel(stripped.slice(cursor + 1, close));
      cursor = close + 1;
      continue;
    }

    const idMatch = rest.match(/^[A-Za-z_][\w-]*/);
    if (!idMatch) break;

    const id = idMatch[0];
    if (previous !== null && arrow !== null) {
      links.push({ from: previous, to: id, label: label || null, dashed: DASHED.has(arrow) });
    }
    previous = id;
    arrow = null;
    label = null;
    cursor += id.length;
  }

  return links;
}

function parseFlowchart(source: string): ParsedDiagram | null {
  const labels = new Map<string, string>();
  const types = new Map<string, string | null>();
  const order: string[] = [];
  const edges: GraphEdge[] = [];

  const see = (id: string) => {
    if (!order.includes(id)) order.push(id);
  };

  for (const raw of source.split('\n').slice(1)) {
    if (!raw.trim() || IGNORED.test(raw)) continue;

    const line = normaliseInlineLabels(raw);
    const { found, stripped } = extractNodes(line);

    for (const node of found) {
      see(node.id);
      // A later declaration wins: a model that names a node twice usually
      // spelled the fuller label the second time.
      if (node.label) labels.set(node.id, node.label);
      types.set(node.id, node.type);
    }

    for (const link of readLinks(stripped)) {
      see(link.from);
      see(link.to);
      edges.push({
        source: link.from,
        target: link.to,
        kind: 'cooccurrence',
        weight: 1,
        relation: link.dashed ? null : link.label,
      });
    }
  }

  return assemble('flowchart', order, labels, types, edges, []);
}

/* ------------------------------------------------------------------ *
 * ER diagrams
 * ------------------------------------------------------------------ */

const ER_HEADER = /^\s*erDiagram\b/;

/**
 * Cardinality as Mermaid spells it, on each side of the line.
 *
 * The markers mirror around the line: `}o--o{` is "zero or more" at both ends.
 * Read into a compact form a reader can take in at a glance, because
 * cardinality is the entire content of an ER relationship — dropping it would
 * leave a picture saying two tables are connected and nothing about how.
 */
const LEFT_CARDINALITY: Readonly<Record<string, string>> = {
  '||': '1',
  '|o': '0..1',
  'o|': '0..1',
  '}o': '0..n',
  '}|': '1..n',
};

const RIGHT_CARDINALITY: Readonly<Record<string, string>> = {
  '||': '1',
  'o|': '0..1',
  '|o': '0..1',
  'o{': '0..n',
  '|{': '1..n',
};

const ER_RELATION =
  /^([A-Za-z_][\w-]*)\s+(\|\||\|o|o\||\}o|\}\|)(--|\.\.)(\|\||o\||\|o|o\{|\|\{)\s+([A-Za-z_][\w-]*)\s*:\s*(.+?)$/;

const ER_BLOCK_OPEN = /^([A-Za-z_][\w-]*)\s*\{$/;
const ER_ATTRIBUTE = /^([A-Za-z_][\w<>,()[\]-]*)\s+([A-Za-z_][\w-]*)\s*(PK|FK|UK)?\s*(?:"[^"]*")?$/;

function parseErDiagram(source: string): ParsedDiagram | null {
  const labels = new Map<string, string>();
  const types = new Map<string, string | null>();
  const order: string[] = [];
  const edges: GraphEdge[] = [];
  const entities = new Map<string, EntityBlock>();

  const see = (id: string) => {
    if (!order.includes(id)) order.push(id);
    if (!labels.has(id)) labels.set(id, id);
    types.set(id, 'entity');
  };

  let open: EntityBlock | null = null;

  for (const raw of source.split('\n').slice(1)) {
    const line = raw.trim();
    if (!line || line.startsWith('%%')) continue;

    if (open) {
      if (line === '}') {
        entities.set(open.id, open);
        open = null;
        continue;
      }
      const attribute = line.match(ER_ATTRIBUTE);
      if (attribute) {
        open.attributes.push({
          type: attribute[1],
          name: attribute[2],
          key: attribute[3] ?? null,
        });
      }
      continue;
    }

    const blockOpen = line.match(ER_BLOCK_OPEN);
    if (blockOpen) {
      see(blockOpen[1]);
      open = { id: blockOpen[1], label: blockOpen[1], attributes: [] };
      continue;
    }

    const relation = line.match(ER_RELATION);
    if (relation) {
      const [, left, leftCard, , rightCard, right, rawLabel] = relation;
      see(left);
      see(right);
      const label = cleanLabel(rawLabel);
      const from = LEFT_CARDINALITY[leftCard] ?? '?';
      const to = RIGHT_CARDINALITY[rightCard] ?? '?';
      edges.push({
        source: left,
        target: right,
        kind: 'cooccurrence',
        weight: 1,
        // Label and cardinality together, because either alone misdescribes
        // the relationship.
        relation: label ? `${label} (${from} → ${to})` : `${from} → ${to}`,
      });
    }
  }

  // A block left unclosed by a truncated reply still describes an entity.
  if (open) entities.set(open.id, open);

  return assemble(
    'er',
    order,
    labels,
    types,
    edges,
    order.map(id => entities.get(id) ?? { id, label: id, attributes: [] }),
  );
}

/* ------------------------------------------------------------------ *
 * Entry point
 * ------------------------------------------------------------------ */

/** Whether this source is written in a grammar this module reads at all. */
export function isSupportedDiagram(source: string): boolean {
  return FLOW_HEADER.test(source) || ER_HEADER.test(source);
}

/**
 * Parses a model-written Mermaid block.
 *
 * `null` means "show the source instead" — an unrecognised diagram type, or one
 * whose body yielded no nodes. Both are cases where drawing something would be
 * a guess.
 */
export function parseDiagram(source: string): ParsedDiagram | null {
  if (ER_HEADER.test(source)) return parseErDiagram(source);
  if (FLOW_HEADER.test(source)) return parseFlowchart(source);
  return null;
}
