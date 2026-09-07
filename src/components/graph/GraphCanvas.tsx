import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { radiusFor } from './layout';
import { GraphSimulation, type SimNode } from './simulation';
import type { GraphEdge, GraphNode } from '../../services/notebook.service';

/**
 * The graph, drawn on a canvas and run as a physics simulation.
 *
 * ## It moves
 *
 * Nodes are placed by a force simulation that runs frame by frame and settles,
 * rather than a layout solved once and painted — see `./simulation.ts` for the
 * model and why it is written rather than installed. A node can be picked up
 * and dropped, and the graph rearranges around it and comes back to rest. That
 * is not decoration: a static picture of a graph gives no way to pull a cluster
 * apart and see what is inside it, and pulling things apart is most of what
 * looking at a graph is for.
 *
 * ## Canvas, not SVG
 *
 * SVG is one DOM element per node, and browsers begin to struggle past a few
 * hundred of them — every frame re-lays-out the document. Canvas is one element
 * and holds a steady frame rate into the low thousands, comfortably past
 * anything a notebook of documents produces. (Obsidian, which has the same
 * problem at a larger size, moved from SVG to a WebGL renderer for the same
 * reason; canvas is the middle rung and enough here.)
 *
 * ## Type is never carried by colour alone
 *
 * The palette is monochrome by rule (see the note at index.css:117): status is
 * luminance, and only three accent hues exist in the whole product. Eight node
 * types cannot be told apart by shade on a greyscale ramp, and would be
 * invisible to anyone who cannot separate them anyway. So a term's type is
 * drawn as a **shape** — circle, square, diamond, triangle, hexagon — a file is
 * a named box, a link's kind is solid or dashed, and every one of them is named
 * in the legend. Colour carries selection and focus only, which are binary and
 * can afford to be.
 *
 * ## Untyped is a state, not a gap
 *
 * Before the typing pass runs, every node is a term with no type. Those draw as
 * plain circles and the inspector says "not yet typed". They are not drawn as a
 * guessed type, and not hidden.
 */

/** How a node type is drawn. Shape first; the label always accompanies it. */
type Glyph = 'circle' | 'square' | 'diamond' | 'triangle' | 'hexagon';

/** A file's drawn box, kept so hit-testing uses the geometry actually painted. */
interface Box {
  x: number;
  y: number;
  width: number;
  height: number;
}

/** Height of a file's box. Fixed; only the width follows the name. */
const FILE_HEIGHT = 22;

/**
 * Below this many nodes, every node is labelled.
 *
 * Chosen against the canvas rather than by taste: at 320px tall, which is what
 * a chat reply gives it, a label is 11px and needs roughly 14px of vertical
 * room with its node. Two dozen of them spread over that area do not overlap;
 * many more begin to.
 */
const LABEL_EVERY_NODE_BELOW = 24;

/** A file name long enough to be a paragraph is trimmed to fit its box. */
function fileLabel(label: string): string {
  return label.length > 24 ? `${label.slice(0, 23)}…` : label;
}

function roundedRect(context: CanvasRenderingContext2D, box: Box, radius: number) {
  const r = Math.min(radius, box.height / 2, box.width / 2);
  context.beginPath();
  context.moveTo(box.x + r, box.y);
  context.arcTo(box.x + box.width, box.y, box.x + box.width, box.y + box.height, r);
  context.arcTo(box.x + box.width, box.y + box.height, box.x, box.y + box.height, r);
  context.arcTo(box.x, box.y + box.height, box.x, box.y, r);
  context.arcTo(box.x, box.y, box.x + box.width, box.y, r);
  context.closePath();
}

const GLYPH_FOR_TYPE: Record<string, Glyph> = {
  supplier: 'square',
  equipment: 'diamond',
  department: 'triangle',
  contract: 'hexagon',
  risk: 'triangle',
  person: 'circle',
  site: 'square',
  document: 'hexagon',
};

export interface GraphCanvasProps {
  nodes: GraphNode[];
  edges: GraphEdge[];
  /** The node the local view is centred on, drawn with a ring. */
  focusId?: string | null;
  /** Nodes the person has picked for import. */
  selectedIds?: ReadonlySet<string>;
  onSelect?: (node: GraphNode) => void;
  onFocus?: (node: GraphNode) => void;
}

function glyphPath(
  context: CanvasRenderingContext2D,
  glyph: Glyph,
  x: number,
  y: number,
  r: number,
) {
  context.beginPath();
  switch (glyph) {
    case 'square':
      context.rect(x - r, y - r, r * 2, r * 2);
      break;
    case 'diamond':
      context.moveTo(x, y - r);
      context.lineTo(x + r, y);
      context.lineTo(x, y + r);
      context.lineTo(x - r, y);
      context.closePath();
      break;
    case 'triangle':
      context.moveTo(x, y - r);
      context.lineTo(x + r, y + r * 0.8);
      context.lineTo(x - r, y + r * 0.8);
      context.closePath();
      break;
    case 'hexagon':
      for (let i = 0; i < 6; i += 1) {
        const angle = (Math.PI / 3) * i - Math.PI / 6;
        const px = x + r * Math.cos(angle);
        const py = y + r * Math.sin(angle);
        if (i === 0) context.moveTo(px, py);
        else context.lineTo(px, py);
      }
      context.closePath();
      break;
    default:
      context.arc(x, y, r, 0, Math.PI * 2);
  }
}

/**
 * Width a file's box will be drawn at, measured with the font it will use.
 *
 * Measured rather than estimated, because this width becomes the node's
 * collision radius: guess it short and the simulation lets two names overlap,
 * guess it long and the graph is needlessly sparse.
 */
function measureFileWidth(context: CanvasRenderingContext2D, label: string): number {
  context.font = '11px system-ui, sans-serif';
  return context.measureText(fileLabel(label)).width + 18;
}

/**
 * A filled head at the far end of a line, stopped short of the node it points
 * at so it touches the outline rather than disappearing under it.
 *
 * Only the directed edge gets one. A co-occurrence is symmetric — two terms
 * were in the same passage, and neither came first — so an arrowhead there
 * would draw a claim about order that nothing in the corpus supports.
 */
function arrowHead(
  context: CanvasRenderingContext2D,
  from: { x: number; y: number },
  to: { x: number; y: number },
  clearance: number,
  size = 7,
) {
  const angle = Math.atan2(to.y - from.y, to.x - from.x);
  const tipX = to.x - Math.cos(angle) * clearance;
  const tipY = to.y - Math.sin(angle) * clearance;
  const spread = 0.42;
  context.beginPath();
  context.moveTo(tipX, tipY);
  context.lineTo(tipX - Math.cos(angle - spread) * size, tipY - Math.sin(angle - spread) * size);
  context.lineTo(tipX - Math.cos(angle + spread) * size, tipY - Math.sin(angle + spread) * size);
  context.closePath();
  context.fill();
}

export const GraphCanvas: React.FC<GraphCanvasProps> = ({
  nodes,
  edges,
  focusId,
  selectedIds,
  onSelect,
  onFocus,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  // Where each file was actually painted. Its width follows the measured name,
  // so hit-testing cannot recompute it without risking a box that disagrees
  // with the one on screen — clicking a file you can see and missing it is
  // exactly the kind of small wrongness that makes a canvas feel broken.
  const fileBoxes = useRef<Map<string, Box>>(new Map());
  const [size, setSize] = useState({ width: 800, height: 560 });
  const [hovered, setHovered] = useState<string | null>(null);

  const simulation = useRef<GraphSimulation | null>(null);
  const frame = useRef<number | null>(null);
  const drawRef = useRef<() => void>(() => {});
  // Pan and zoom live in a ref, not in state: they change on every pointer move
  // and the frame loop reads them directly, so putting them through React would
  // re-render the component sixty times a second for nothing.
  const viewRef = useRef({ scale: 1, x: 0, y: 0 });
  const gesture = useRef<
    | { kind: 'pan'; x: number; y: number; moved: boolean }
    | { kind: 'drag'; id: string; moved: boolean }
    | null
  >(null);

  const byId = useMemo(() => new Map(nodes.map((node) => [node.id, node])), [nodes]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const parent = canvas.parentElement;
    if (!parent) return;
    const observer = new ResizeObserver(([entry]) => {
      const width = Math.max(200, entry.contentRect.width);
      const height = Math.max(200, entry.contentRect.height);
      // Only a real change. A new object with the same numbers would rebuild
      // the simulation and throw away an arrangement the person is looking at.
      setSize((current) =>
        current.width === width && current.height === height ? current : { width, height },
      );
    });
    observer.observe(parent);
    return () => observer.disconnect();
  }, []);

  /** Screen pixels to simulation space, undoing the pan and zoom. */
  const toWorld = useCallback(
    (px: number, py: number) => {
      const { scale, x, y } = viewRef.current;
      return {
        x: (px - x - size.width / 2) / scale + size.width / 2,
        y: (py - y - size.height / 2) / scale + size.height / 2,
      };
    },
    [size],
  );

  /** Simulation space to screen pixels. */
  const toScreen = useCallback(
    (point: { x: number; y: number }) => {
      const { scale, x, y } = viewRef.current;
      return {
        x: (point.x - size.width / 2) * scale + size.width / 2 + x,
        y: (point.y - size.height / 2) * scale + size.height / 2 + y,
        scale,
      };
    },
    [size],
  );

  /**
   * Runs the loop until the simulation is at rest.
   *
   * Idle costs nothing: once alpha falls below its minimum the loop stops and
   * the last frame stays on screen. Anything that changes the picture without
   * moving a node — a hover, a new selection — asks for one more frame.
   */
  const ensureLoop = useCallback(() => {
    if (frame.current !== null) return;
    const tick = () => {
      frame.current = null;
      const running = simulation.current?.step() ?? false;
      drawRef.current();
      if (running) frame.current = requestAnimationFrame(tick);
    };
    frame.current = requestAnimationFrame(tick);
  }, []);

  // A new graph, or a resized canvas, is a new simulation. Sizes are measured
  // from the font the labels will actually be drawn in, so the collision radii
  // match the boxes on screen.
  useEffect(() => {
    const context = canvasRef.current?.getContext('2d');
    if (!context) return;
    simulation.current = new GraphSimulation(
      nodes.map((node) => ({
        id: node.id,
        radius:
          node.kind === 'document'
            ? Math.max(FILE_HEIGHT, measureFileWidth(context, node.label)) / 2 + 6
            : radiusFor(node.degree) + 10,
      })),
      edges.map((edge) => ({ source: edge.source, target: edge.target, weight: edge.weight })),
      { width: size.width, height: size.height },
    );
    ensureLoop();
  }, [nodes, edges, size, ensureLoop]);

  useEffect(
    () => () => {
      if (frame.current !== null) cancelAnimationFrame(frame.current);
      frame.current = null;
    },
    [],
  );

  const nodeAt = useCallback(
    (clientX: number, clientY: number): GraphNode | null => {
      const canvas = canvasRef.current;
      const sim = simulation.current;
      if (!canvas || !sim) return null;
      const rect = canvas.getBoundingClientRect();
      const px = clientX - rect.left;
      const py = clientY - rect.top;

      // Files are painted over the terms, so they are picked first.
      for (let i = nodes.length - 1; i >= 0; i -= 1) {
        if (nodes[i].kind !== 'document') continue;
        const box = fileBoxes.current.get(nodes[i].id);
        if (!box) continue;
        if (px >= box.x && px <= box.x + box.width && py >= box.y && py <= box.y + box.height) {
          return nodes[i];
        }
      }

      for (let i = nodes.length - 1; i >= 0; i -= 1) {
        if (nodes[i].kind === 'document') continue;
        const position = sim.find(nodes[i].id);
        if (!position) continue;
        const { x, y, scale } = toScreen(position);
        const r = Math.max(6, radiusFor(nodes[i].degree) * scale);
        if (Math.hypot(px - x, py - y) <= r + 3) return nodes[i];
      }
      return null;
    },
    [nodes, toScreen],
  );

  /**
   * Zoom, on a native listener and only with a modifier held.
   *
   * The panel sits in a column that scrolls, and a canvas this tall is most of
   * what the pointer is over on the way down it. A plain wheel that zoomed
   * would take the page's scroll away wherever the cursor happened to land, so
   * a plain wheel is left alone and scrolls the page as it does everywhere
   * else; ctrl (or cmd) plus wheel zooms, which is also what a trackpad pinch
   * sends.
   *
   * Registered here rather than as `onWheel` because React attaches wheel
   * listeners passively: `preventDefault` from a React handler is ignored with
   * a console warning, and without it ctrl+wheel would zoom the whole window
   * instead of the graph.
   */
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const onWheel = (event: WheelEvent) => {
      if (!event.ctrlKey && !event.metaKey) return;
      event.preventDefault();
      const current = viewRef.current;
      viewRef.current = {
        ...current,
        scale: Math.min(6, Math.max(0.3, current.scale * (event.deltaY < 0 ? 1.1 : 0.9))),
      };
      ensureLoop();
    };
    canvas.addEventListener('wheel', onWheel, { passive: false });
    return () => canvas.removeEventListener('wheel', onWheel);
  }, [ensureLoop]);

  // The draw, rebuilt on every render and read by the frame loop through a ref,
  // so the loop itself never has to be torn down and restarted.
  drawRef.current = () => {
    const canvas = canvasRef.current;
    const sim = simulation.current;
    if (!canvas || !sim) return;
    const context = canvas.getContext('2d');
    if (!context) return;

    const ratio = window.devicePixelRatio || 1;
    canvas.width = size.width * ratio;
    canvas.height = size.height * ratio;
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, size.width, size.height);

    // Colours are read from the live theme rather than hardcoded, so the canvas
    // follows the light/dark switch like every other surface.
    const styles = getComputedStyle(canvas);
    const ink = styles.getPropertyValue('--text-primary').trim() || '#fafafa';
    const faint = styles.getPropertyValue('--text-tertiary').trim() || '#888888';
    const line = styles.getPropertyValue('--border-default').trim() || '#333333';
    const ground = styles.getPropertyValue('--bg-primary').trim() || '#000000';
    const accent = styles.getPropertyValue('--accent-primary').trim() || '#ffffff';

    const at = (id: string): SimNode | undefined => sim.find(id);

    // Two kinds of line, told apart by dash and by head rather than by shade.
    // An unbroken line with no head is a co-occurrence between terms: symmetric,
    // so it points nowhere. A dashed line with a head runs from a file to a term
    // found in it, which is a real direction — the file produced the term, and
    // the reverse is not a separate fact. Both are named in the legend and
    // spelled out in the inspector, so neither has to be read off the drawing.
    for (const edge of edges) {
      const a = at(edge.source);
      const b = at(edge.target);
      if (!a || !b) continue;
      const from = toScreen(a);
      const to = toScreen(b);
      const membership = edge.kind === 'appearsIn';
      const dimmed = hovered && edge.source !== hovered && edge.target !== hovered;
      context.strokeStyle = line;
      context.setLineDash(membership ? [3, 4] : []);
      // Thickness is the passage count, dampened — the difference between 2 and
      // 20 should be visible without 20 becoming a bar across the screen.
      context.lineWidth = membership ? 1 : Math.min(4, 0.6 + Math.log(edge.weight + 1) * 0.7);
      context.globalAlpha = dimmed ? 0.15 : membership ? 0.45 : 0.7;
      context.beginPath();
      context.moveTo(from.x, from.y);
      context.lineTo(to.x, to.y);
      context.stroke();

      if (membership) {
        // Stopped at the target's outline. A term is a circle, so its drawn
        // radius is the clearance; a file is a box, and half its height is the
        // closest a head should come without overlapping the name inside it.
        const target = byId.get(edge.target);
        const clearance =
          target?.kind === 'document'
            ? FILE_HEIGHT / 2 + 4
            : Math.max(4, radiusFor(target?.degree ?? 0) * to.scale) + 4;
        context.setLineDash([]);
        context.fillStyle = line;
        arrowHead(context, from, to, clearance);
      }
    }
    context.setLineDash([]);
    context.globalAlpha = 1;

    for (const node of nodes) {
      if (node.kind === 'document') continue;
      const position = at(node.id);
      if (!position) continue;
      const { x, y, scale } = toScreen(position);
      const r = Math.max(4, radiusFor(node.degree) * scale);
      const isSelected = selectedIds?.has(node.id) ?? false;
      const isFocus = node.id === focusId;
      const glyph = node.nodeType ? (GLYPH_FOR_TYPE[node.nodeType] ?? 'circle') : 'circle';

      glyphPath(context, glyph, x, y, r);
      context.fillStyle = isSelected || isFocus ? accent : ground;
      context.fill();
      context.lineWidth = isFocus ? 2.5 : 1.2;
      context.strokeStyle = isSelected || isFocus ? accent : faint;
      context.stroke();

      // A ring, so the focus is distinguishable from a selection without
      // depending on a colour difference.
      if (isFocus) {
        context.beginPath();
        context.arc(x, y, r + 5, 0, Math.PI * 2);
        context.strokeStyle = accent;
        context.lineWidth = 1;
        context.globalAlpha = 0.5;
        context.stroke();
        context.globalAlpha = 1;
      }

      // Labels only where they can be read: on bigger nodes, the focus, the
      // selection, and whatever is under the cursor. A label on every node at
      // any zoom is an unreadable smear.
      //
      // Except when there is nothing to smear. `radiusFor` gives a degree-1
      // node 6.2, so `r > 7` needs degree 2 or more - and a graph drawn in a
      // chat reply is usually half a dozen terms in a line, every one of them
      // degree 1. The rule that keeps a 300-node picture readable was leaving
      // the small ones as unlabelled circles, which is not a graph at all: the
      // reader cannot tell which term is which.
      //
      // The bound is on the node count rather than the radius because that is
      // the thing that decides whether labels collide.
      const roomForEveryLabel = nodes.length <= LABEL_EVERY_NODE_BELOW;
      if (r > 7 || roomForEveryLabel || isFocus || isSelected || node.id === hovered) {
        context.fillStyle = ink;
        context.font = '11px system-ui, sans-serif';
        context.textAlign = 'center';
        context.textBaseline = 'top';
        const text = node.label.length > 28 ? `${node.label.slice(0, 27)}…` : node.label;
        context.fillText(text, x, y + r + 3);
      }
    }

    // Files last, so they sit above the terms and their names stay readable —
    // and always named, because an unlabelled file node is worth nothing. The
    // shape is the difference that carries: a box with a name in it, against a
    // circle with a name beneath it.
    fileBoxes.current.clear();
    context.font = '11px system-ui, sans-serif';
    context.textAlign = 'center';
    context.textBaseline = 'middle';
    for (const node of nodes) {
      if (node.kind !== 'document') continue;
      const position = at(node.id);
      if (!position) continue;
      const { x, y } = toScreen(position);
      const isSelected = selectedIds?.has(node.id) ?? false;
      const isFocus = node.id === focusId;
      const text = fileLabel(node.label);
      const width = context.measureText(text).width + 18;
      const box: Box = { x: x - width / 2, y: y - FILE_HEIGHT / 2, width, height: FILE_HEIGHT };
      fileBoxes.current.set(node.id, box);

      context.globalAlpha = hovered && node.id !== hovered ? 0.85 : 1;
      roundedRect(context, box, 6);
      context.fillStyle = isSelected || isFocus ? accent : ground;
      context.fill();
      context.lineWidth = isFocus ? 2.5 : 1.2;
      // A file with no links contributed nothing to the graph. Drawn dimmer
      // rather than hidden: "this document produced no terms" is a finding, and
      // omitting it would read as a notebook with fewer files in it.
      context.strokeStyle = isSelected || isFocus ? accent : node.degree === 0 ? line : ink;
      context.stroke();
      context.fillStyle = isSelected || isFocus ? ground : node.degree === 0 ? faint : ink;
      context.fillText(text, x, y + 0.5);
      context.globalAlpha = 1;
    }

    // The legend. Shape and dash are the only things telling the four kinds
    // apart, so the key to them has to be on the picture itself.
    const hasFiles = nodes.some((node) => node.kind === 'document');
    const key: Array<'term' | 'file' | 'with' | 'in'> = hasFiles
      ? ['term', 'file', 'with', 'in']
      : ['term', 'with'];
    const LABELS = { term: 'term', file: 'file', with: 'appears with', in: 'file → term' } as const;
    context.font = '10px system-ui, sans-serif';
    context.textAlign = 'left';
    context.textBaseline = 'middle';
    const rowHeight = 15;
    const legendWidth =
      26 + Math.max(...key.map((entry) => context.measureText(LABELS[entry]).width));
    const legend: Box = {
      x: size.width - legendWidth - 10,
      y: size.height - key.length * rowHeight - 14,
      width: legendWidth + 4,
      height: key.length * rowHeight + 8,
    };
    context.globalAlpha = 0.92;
    roundedRect(context, legend, 4);
    context.fillStyle = ground;
    context.fill();
    context.globalAlpha = 1;
    context.strokeStyle = line;
    context.lineWidth = 1;
    context.stroke();

    key.forEach((entry, index) => {
      const y = legend.y + 4 + rowHeight * index + rowHeight / 2;
      const x = legend.x + 6;
      context.strokeStyle = faint;
      context.fillStyle = ground;
      context.lineWidth = 1.2;
      if (entry === 'term') {
        context.beginPath();
        context.arc(x + 6, y, 4.5, 0, Math.PI * 2);
        context.fill();
        context.stroke();
      } else if (entry === 'file') {
        context.strokeStyle = ink;
        roundedRect(context, { x, y: y - 5, width: 13, height: 10 }, 3);
        context.fill();
        context.stroke();
      } else {
        context.strokeStyle = line;
        context.setLineDash(entry === 'in' ? [3, 4] : []);
        context.beginPath();
        context.moveTo(x, y);
        context.lineTo(x + (entry === 'in' ? 9 : 13), y);
        context.stroke();
        context.setLineDash([]);
        // The head is half the key: it is what says this line has a direction
        // and the plain one above it does not.
        if (entry === 'in') {
          context.fillStyle = line;
          arrowHead(context, { x, y }, { x: x + 14, y }, 0, 5);
        }
      }
      context.fillStyle = faint;
      context.fillText(LABELS[entry], x + 20, y + 0.5);
    });
  };

  // Anything that changes the picture without moving a node still needs a
  // frame: a hover, a new selection, a changed focus.
  useEffect(() => {
    ensureLoop();
  });

  return (
    <canvas
      ref={canvasRef}
      style={{
        width: '100%',
        height: '100%',
        display: 'block',
        cursor: gesture.current?.kind === 'drag' ? 'grabbing' : hovered ? 'grab' : 'default',
      }}
      onMouseDown={(event) => {
        const node = nodeAt(event.clientX, event.clientY);
        const sim = simulation.current;
        if (node && sim) {
          // Picking a node up pins it and reheats the simulation, so the rest
          // of the graph gets out of the way while it is held.
          const held = sim.find(node.id);
          if (held) {
            held.fx = held.x;
            held.fy = held.y;
          }
          sim.reheat();
          gesture.current = { kind: 'drag', id: node.id, moved: false };
          ensureLoop();
          return;
        }
        gesture.current = { kind: 'pan', x: event.clientX, y: event.clientY, moved: false };
      }}
      onMouseMove={(event) => {
        const active = gesture.current;
        if (active?.kind === 'drag') {
          const canvas = canvasRef.current;
          const sim = simulation.current;
          if (!canvas || !sim) return;
          const rect = canvas.getBoundingClientRect();
          const point = toWorld(event.clientX - rect.left, event.clientY - rect.top);
          const held = sim.find(active.id);
          if (held) {
            held.fx = point.x;
            held.fy = point.y;
          }
          active.moved = true;
          sim.reheat();
          ensureLoop();
          return;
        }
        if (active?.kind === 'pan') {
          const deltaX = event.clientX - active.x;
          const deltaY = event.clientY - active.y;
          if (Math.abs(deltaX) > 2 || Math.abs(deltaY) > 2) active.moved = true;
          active.x = event.clientX;
          active.y = event.clientY;
          const current = viewRef.current;
          viewRef.current = { ...current, x: current.x + deltaX, y: current.y + deltaY };
          ensureLoop();
          return;
        }
        setHovered(nodeAt(event.clientX, event.clientY)?.id ?? null);
      }}
      onMouseUp={(event) => {
        const active = gesture.current;
        gesture.current = null;
        const sim = simulation.current;
        if (active?.kind === 'drag' && sim) {
          // Dropped nodes are let go rather than left pinned. A graph that
          // quietly accumulated pins would stop being a function of its data.
          const held = sim.find(active.id);
          if (held) {
            held.fx = null;
            held.fy = null;
          }
          sim.cool();
          ensureLoop();
          if (!active.moved) {
            const node = byId.get(active.id);
            if (node) onSelect?.(node);
          }
          return;
        }
        // A click that panned the canvas is not a click on a node.
        if (active?.moved) return;
        const node = nodeAt(event.clientX, event.clientY);
        if (node) onSelect?.(node);
      }}
      onMouseLeave={() => {
        const active = gesture.current;
        gesture.current = null;
        const sim = simulation.current;
        if (active?.kind === 'drag' && sim) {
          const held = sim.find(active.id);
          if (held) {
            held.fx = null;
            held.fy = null;
          }
          sim.cool();
          ensureLoop();
        }
        setHovered(null);
      }}
      onDoubleClick={(event) => {
        const node = nodeAt(event.clientX, event.clientY);
        if (node) onFocus?.(node);
      }}
    />
  );
};

export default GraphCanvas;
