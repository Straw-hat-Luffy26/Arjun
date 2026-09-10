import React from 'react';

import { CodeBlock } from './CodeBlock';
import { parseMermaidGraph } from './mermaidParse';
import { parseDiagram } from './mermaidDiagram';
import { mermaidConfig, readDiagramTokens } from './mermaidTheme';
import { sanitizeDiagramSvg } from './svgSanitize';
import { useTheme } from '../../contexts/ThemeContext';
import styles from './ChatSurface.module.css';

/**
 * A `mermaid` fence in an assistant message, drawn as the diagram it describes.
 *
 * ## What changed, and why the previous drawing was the wrong answer
 *
 * This used to hand every fence to `GraphCanvas` — the force simulation that
 * draws knowledge graphs. That is the right picture for "what is connected to
 * what" across terms nobody chose, and the wrong one for a diagram somebody
 * asked for by name. `artifacts/diagram.rs` says it plainly in its own header:
 * a block diagram has an author's order, inlet before pump before header, and a
 * force layout discards exactly that, arranging by attraction instead of by
 * intent. Asking for a flowchart and receiving a floating cloud of labelled
 * circles "is not a different style of the same answer; it is a different
 * answer."
 *
 * So `flowchart TD; A --> B --> C` now renders as a flowchart that reads
 * downward, because Mermaid lays it out by rank. The force canvas is untouched
 * and still draws the notebook graph in `NotebookGraphPanel`, which is what it
 * was built for.
 *
 * ## Why Mermaid rather than a layout written here
 *
 * Because the grammar is Mermaid's. Everything a model writes into a `mermaid`
 * fence — sequence diagrams, state charts, class diagrams, Gantt, pie, and the
 * dozen flowchart node shapes — is defined by that project, and a reader for it
 * written here would be a permanent chase after a specification somebody else
 * moves. The library is bundled at build time and reaches the network at no
 * point; `check-egress.mjs` still holds, because there is still exactly one
 * module in this repository that can construct an outbound client.
 *
 * ## Three ways this can end, and none of them is a blank space
 *
 * 1. Mermaid draws it. The SVG goes through `sanitizeDiagramSvg` first.
 * 2. Mermaid refuses it — a diagram type it does not know, a truncated reply,
 *    a syntax error. The source is shown as a code block, with a line naming
 *    what the diagram was *going to be* when either of the two pure readers can
 *    still tell (`parseDiagram`, `parseMermaidGraph`), because "Diagram, 6
 *    nodes — could not be drawn" tells a reader more than the source alone.
 * 3. The fence is still streaming. The source is shown until it closes, which
 *    is what a half-written diagram honestly is.
 */

/**
 * Mermaid renders one diagram at a time.
 *
 * `mermaid.render` mounts a temporary element, measures text in it, and tears
 * it down. Two of those overlapping in one tick measure each other's element,
 * and a message carrying three diagrams would draw two of them at the wrong
 * size. Serialised through one chain rather than fixed with a lock, because
 * ordering is all that is needed and a chain cannot deadlock.
 */
let renderQueue: Promise<unknown> = Promise.resolve();

function enqueue<T>(work: () => Promise<T>): Promise<T> {
  const next = renderQueue.then(work, work);
  // A failed render must not poison the queue for every diagram after it.
  renderQueue = next.catch(() => undefined);
  return next;
}

/**
 * The library, loaded on first use.
 *
 * Mermaid is by a wide margin the largest dependency in this surface, and most
 * sessions never show a diagram. Imported dynamically so it lands in its own
 * chunk and the application starts without paying for it; the promise is cached
 * so a message carrying several fences loads it once.
 */
let mermaidModule: Promise<(typeof import('mermaid'))['default']> | null = null;

function loadMermaid() {
  if (!mermaidModule) {
    mermaidModule = import('mermaid').then(module => module.default);
  }
  return mermaidModule;
}

/** What a diagram was going to be, for the line shown when it cannot be drawn. */
function describeUndrawn(source: string): string | null {
  const parsed = parseMermaidGraph(source) ?? parseDiagram(source);
  if (!parsed) return null;

  const count = parsed.nodes.length;
  if (count === 0) return null;

  const kind =
    'kind' in parsed && parsed.kind === 'er'
      ? `Entity diagram, ${count} ${count === 1 ? 'entity' : 'entities'}`
      : `Diagram, ${count} ${count === 1 ? 'node' : 'nodes'}`;
  return `${kind} — could not be drawn, so the source is shown.`;
}

/**
 * The caveat a knowledge graph carries and a flowchart does not.
 *
 * In `knowledge.build_graph` output an unlabelled link means two terms appeared
 * in the same passage and nothing more. A reader who takes that for a stated
 * relationship has been misled by the picture, so the picture says so. A model's
 * own flowchart makes no such claim — an unlabelled arrow there means only that
 * the model drew it unlabelled — so nothing is asserted about one.
 */
function knowledgeGraphNote(source: string): string | null {
  const parsed = parseMermaidGraph(source);
  if (!parsed) return null;

  const unnamed = parsed.edges.filter(edge => edge.relation === null).length;
  const terms = `${parsed.nodes.length} ${parsed.nodes.length === 1 ? 'term' : 'terms'}`;
  const parts = [terms];
  if (unnamed > 0) {
    parts.push(
      `${unnamed} unnamed ${unnamed === 1 ? 'link' : 'links'} (shared passages only)`,
    );
  }
  if (parsed.omitted > 0) parts.push(`${parsed.omitted} more not drawn`);
  return parts.join(', ');
}

/**
 * The width Mermaid laid the diagram out at, from its `viewBox`.
 *
 * `sanitizeDiagramSvg` strips the root `width` and the inline `max-width`
 * Mermaid writes, so that a diagram is sized by the column rather than by
 * whatever its layout engine happened to measure. Stripping both alone,
 * however, lets a small diagram *grow*: a three-node graph laid out at 647px
 * was being stretched to 1214px to fill the chat column, which does not add
 * detail — it just renders every stroke and label at twice the intended size
 * and soft.
 *
 * So the natural width comes back as a number and becomes a ceiling in CSS. A
 * diagram shrinks to fit a narrow column and never scales past the size it was
 * drawn for.
 */
function intrinsicWidth(svg: string): number | null {
  const viewBox = svg.match(/viewBox="([^"]+)"/)?.[1];
  if (!viewBox) return null;
  const parts = viewBox.trim().split(/[\s,]+/).map(Number);
  if (parts.length !== 4 || parts.some(Number.isNaN)) return null;
  const width = parts[2];
  return width > 0 ? Math.round(width) : null;
}

type Drawing =
  | { state: 'pending' }
  | { state: 'drawn'; svg: string; width: number | null }
  | { state: 'failed' };

export function MermaidGraph({
  source,
  complete = true,
}: {
  source: string;
  /**
   * False while the fence is still streaming.
   *
   * A diagram is not attempted until its fence closes. Mermaid parses the whole
   * source or none of it, so a half-written one produces an error per token and
   * nothing on screen; the source is shown meanwhile, which is what a
   * half-written diagram is.
   */
  complete?: boolean;
}) {
  const { resolvedTheme } = useTheme();
  const [drawing, setDrawing] = React.useState<Drawing>({ state: 'pending' });

  // React's own id carries delimiters — `«r0»` in React 19 — and Mermaid writes
  // this straight into a CSS selector in the stylesheet it generates. Stripped
  // to word characters so the rule it scopes actually matches the diagram.
  const reactId = React.useId().replace(/[^a-zA-Z0-9]/g, '');
  const id = `arjun-diagram-${reactId}`;

  React.useEffect(() => {
    if (!complete) {
      setDrawing({ state: 'pending' });
      return;
    }

    let live = true;
    setDrawing({ state: 'pending' });

    void enqueue(async () => {
      try {
        const mermaid = await loadMermaid();
        if (!live) return;

        // Re-applied per render rather than once at startup: `initialize` is
        // global, and the theme it bakes in is read off the document, which
        // changes when the person switches between light and dark.
        mermaid.initialize(
          mermaidConfig(readDiagramTokens(document.documentElement)) as Parameters<
            typeof mermaid.initialize
          >[0],
        );

        // Asked before rendering so an unreadable diagram takes the fallback
        // path rather than throwing through the queue.
        const readable = await mermaid.parse(source, { suppressErrors: true });
        if (!live) return;
        if (!readable) {
          setDrawing({ state: 'failed' });
          return;
        }

        const { svg } = await mermaid.render(id, source);
        if (!live) return;

        const safe = sanitizeDiagramSvg(svg);
        setDrawing(
          safe
            ? { state: 'drawn', svg: safe, width: intrinsicWidth(safe) }
            : { state: 'failed' },
        );
      } catch {
        // Every failure lands here as the same thing: no picture, so show the
        // source. The reason is not surfaced because none of them is actionable
        // by the reader — a diagram type Mermaid does not know and a reply cut
        // off mid-line look identical from here.
        if (live) setDrawing({ state: 'failed' });
      }
    });

    return () => {
      live = false;
    };
  }, [source, complete, id, resolvedTheme]);

  const note = React.useMemo(() => knowledgeGraphNote(source), [source]);

  if (drawing.state === 'drawn') {
    return (
      <figure
        className={styles.mdDiagram}
        // The ceiling described on `intrinsicWidth`, handed to CSS rather than
        // applied here so the stylesheet keeps every rule about how a diagram
        // is sized in one place.
        style={
          drawing.width
            ? ({ '--diagram-width': `${drawing.width}px` } as React.CSSProperties)
            : undefined
        }
      >
        <div
          className={styles.mdDiagramSurface}
          // Sanitised immediately above by `sanitizeDiagramSvg`, an allowlist
          // that drops every element and attribute it does not name — scripts,
          // event handlers and `<foreignObject>` among them.
          dangerouslySetInnerHTML={{ __html: drawing.svg }}
        />
        {note && <figcaption className={styles.mdDiagramCaption}>{note}</figcaption>}
      </figure>
    );
  }

  // Pending and failed both show the source. They are not the same state, but
  // they have the same honest rendering: the diagram as text, because there is
  // no picture yet or there will not be one.
  const undrawn = drawing.state === 'failed' ? describeUndrawn(source) : null;

  return (
    <>
      <CodeBlock code={source} lang="mermaid" />
      {undrawn && <p className={styles.mdDiagramCaption}>{undrawn}</p>}
    </>
  );
}
