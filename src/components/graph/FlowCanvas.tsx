import React, { useCallback, useEffect, useRef, useState } from 'react';
import type { FlowView } from './flow';

/**
 * The flow between chosen files, drawn as layered columns.
 *
 * No physics. The force graph in `./GraphCanvas.tsx` answers "what is in this
 * notebook" and needs a simulation to find an arrangement; this answers "how do
 * these particular files connect", and the arrangement is already decided by
 * `./flow.ts` — a file, the terms it shares with the next file, that file, and
 * so on. Running a simulation over a layout that is a chain by construction
 * would only make it wobble.
 *
 * The look is the layered one a person means by "like a neural network":
 * columns of nodes, curved links between adjacent columns, everything flowing
 * one way. The one-way appearance is honest here — every link drawn is a
 * membership edge, which genuinely has a direction. No co-occurrence is drawn,
 * because a route from one file to another through "these two terms sat in the
 * same passage" is not a route the extractor ever observed.
 *
 * ## A column with nothing in it stays a column
 *
 * Two adjacent files that share no terms leave an empty gap, labelled in place.
 * Closing it up would let the eye read a chain of connected documents where the
 * finding is the opposite.
 */

export interface FlowCanvasProps {
  flow: FlowView;
  selectedId?: string | null;
  onSelect?: (id: string) => void;
}

const FILE_HEIGHT = 24;
const TERM_RADIUS = 7;
/** Nothing is drawn closer than this to an edge of the canvas. */
const PAD = 18;

interface Placed {
  id: string;
  label: string;
  kind: 'term' | 'document';
  x: number;
  y: number;
  /** Half-width of the hit box; for a term this is its circle. */
  halfWidth: number;
  halfHeight: number;
}

function roundedRect(
  context: CanvasRenderingContext2D,
  x: number,
  y: number,
  width: number,
  height: number,
  radius: number,
) {
  const r = Math.min(radius, height / 2, width / 2);
  context.beginPath();
  context.moveTo(x + r, y);
  context.arcTo(x + width, y, x + width, y + height, r);
  context.arcTo(x + width, y + height, x, y + height, r);
  context.arcTo(x, y + height, x, y, r);
  context.arcTo(x, y, x + width, y, r);
  context.closePath();
}

/** Trims a label to the room its column actually has. */
function fit(context: CanvasRenderingContext2D, label: string, maxWidth: number): string {
  if (context.measureText(label).width <= maxWidth) return label;
  let text = label;
  while (text.length > 1 && context.measureText(`${text}…`).width > maxWidth) {
    text = text.slice(0, -1);
  }
  return `${text}…`;
}

export const FlowCanvas: React.FC<FlowCanvasProps> = ({ flow, selectedId, onSelect }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const placed = useRef<Placed[]>([]);
  const [size, setSize] = useState({ width: 800, height: 440 });
  const [hovered, setHovered] = useState<string | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const parent = canvas?.parentElement;
    if (!parent) return;
    const observer = new ResizeObserver(([entry]) => {
      const width = Math.max(200, entry.contentRect.width);
      const height = Math.max(200, entry.contentRect.height);
      setSize((current) =>
        current.width === width && current.height === height ? current : { width, height },
      );
    });
    observer.observe(parent);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const context = canvas.getContext('2d');
    if (!context) return;

    const ratio = window.devicePixelRatio || 1;
    canvas.width = size.width * ratio;
    canvas.height = size.height * ratio;
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, size.width, size.height);

    const styles = getComputedStyle(canvas);
    const ink = styles.getPropertyValue('--text-primary').trim() || '#fafafa';
    const faint = styles.getPropertyValue('--text-tertiary').trim() || '#888888';
    const line = styles.getPropertyValue('--border-default').trim() || '#333333';
    const ground = styles.getPropertyValue('--bg-primary').trim() || '#000000';
    const accent = styles.getPropertyValue('--accent-primary').trim() || '#ffffff';

    placed.current = [];
    if (flow.layers.length === 0) return;

    const columns = flow.layers.length;
    const columnWidth = (size.width - PAD * 2) / columns;
    const labelWidth = Math.max(40, columnWidth - 16);

    context.font = '11px system-ui, sans-serif';
    context.textAlign = 'center';
    context.textBaseline = 'middle';

    // Place everything first, so the links can be drawn underneath the nodes.
    const positions = new Map<string, Placed>();
    flow.layers.forEach((layer, index) => {
      const x = PAD + columnWidth * index + columnWidth / 2;
      layer.forEach((node) => {
        // Evenly spread down the column, centred as a group.
        const step = (size.height - PAD * 2) / (node.slots + 1);
        const y = PAD + step * (node.slot + 1);
        const label = fit(context, node.label, labelWidth);
        const entry: Placed =
          node.kind === 'document'
            ? {
                id: node.id,
                label,
                kind: 'document',
                x,
                y,
                halfWidth: context.measureText(label).width / 2 + 9,
                halfHeight: FILE_HEIGHT / 2,
              }
            : {
                id: node.id,
                label,
                kind: 'term',
                x,
                y,
                halfWidth: TERM_RADIUS,
                halfHeight: TERM_RADIUS,
              };
        positions.set(node.id, entry);
        placed.current.push(entry);
      });
    });

    // Links: a horizontal cubic, which is what gives the layered picture its
    // read as flow rather than as a lattice of straight lines.
    for (const link of flow.links) {
      const from = positions.get(link.source);
      const to = positions.get(link.target);
      if (!from || !to) continue;
      const startX = from.x + from.halfWidth;
      const endX = to.x - to.halfWidth - 6;
      const bend = Math.max(18, (endX - startX) / 2);
      const lit = hovered === link.source || hovered === link.target;
      const chosen = selectedId === link.source || selectedId === link.target;

      context.strokeStyle = chosen || lit ? accent : line;
      context.globalAlpha = chosen || lit ? 0.9 : 0.55;
      context.lineWidth = Math.min(3.5, 0.8 + Math.log(link.weight + 1) * 0.7);
      context.beginPath();
      context.moveTo(startX, from.y);
      context.bezierCurveTo(startX + bend, from.y, endX - bend, to.y, endX, to.y);
      context.stroke();

      // A head, because a membership edge has a real direction: the file
      // produced the term. Pointing right, which is the curve's tangent where
      // it lands.
      context.fillStyle = chosen || lit ? accent : line;
      context.beginPath();
      context.moveTo(endX + 6, to.y);
      context.lineTo(endX, to.y - 4);
      context.lineTo(endX, to.y + 4);
      context.closePath();
      context.fill();
    }
    context.globalAlpha = 1;

    // An empty middle column is a finding. Say so where the terms would be.
    flow.layers.forEach((layer, index) => {
      if (layer.length > 0 || index % 2 === 0) return;
      context.fillStyle = faint;
      context.font = '10px system-ui, sans-serif';
      context.fillText(
        'nothing shared',
        PAD + columnWidth * index + columnWidth / 2,
        size.height / 2,
      );
      context.font = '11px system-ui, sans-serif';
    });

    for (const node of placed.current) {
      const isSelected = node.id === selectedId;
      const isHovered = node.id === hovered;
      if (node.kind === 'document') {
        roundedRect(
          context,
          node.x - node.halfWidth,
          node.y - node.halfHeight,
          node.halfWidth * 2,
          node.halfHeight * 2,
          6,
        );
        context.fillStyle = isSelected ? accent : ground;
        context.fill();
        context.lineWidth = isSelected || isHovered ? 2 : 1.2;
        context.strokeStyle = isSelected ? accent : ink;
        context.stroke();
        context.fillStyle = isSelected ? ground : ink;
        context.fillText(node.label, node.x, node.y + 0.5);
      } else {
        context.beginPath();
        context.arc(node.x, node.y, TERM_RADIUS, 0, Math.PI * 2);
        context.fillStyle = isSelected ? accent : ground;
        context.fill();
        context.lineWidth = isSelected || isHovered ? 2 : 1.2;
        context.strokeStyle = isSelected ? accent : faint;
        context.stroke();
        // Beneath the circle, as in the force graph: the shape says what kind
        // of thing it is, and the word says which one.
        context.fillStyle = isSelected || isHovered ? ink : faint;
        context.textBaseline = 'top';
        context.fillText(node.label, node.x, node.y + TERM_RADIUS + 4);
        context.textBaseline = 'middle';
      }
    }
  }, [flow, size, hovered, selectedId]);

  const hit = useCallback((clientX: number, clientY: number): Placed | null => {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    const px = clientX - rect.left;
    const py = clientY - rect.top;
    for (const node of placed.current) {
      // A term's label is part of what a person aims at, so the strip below the
      // circle counts as the term.
      const padY = node.kind === 'term' ? 16 : 0;
      const reach = Math.max(node.halfWidth, 30);
      if (
        px >= node.x - reach &&
        px <= node.x + reach &&
        py >= node.y - node.halfHeight &&
        py <= node.y + node.halfHeight + padY
      ) {
        return node;
      }
    }
    return null;
  }, []);

  return (
    <canvas
      ref={canvasRef}
      style={{
        width: '100%',
        height: '100%',
        display: 'block',
        cursor: hovered ? 'pointer' : 'default',
      }}
      onMouseMove={(event) => setHovered(hit(event.clientX, event.clientY)?.id ?? null)}
      onMouseLeave={() => setHovered(null)}
      onClick={(event) => {
        const node = hit(event.clientX, event.clientY);
        if (node) onSelect?.(node.id);
      }}
    />
  );
};

export default FlowCanvas;
