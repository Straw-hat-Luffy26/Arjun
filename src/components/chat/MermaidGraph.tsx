import React from 'react';

import { GraphCanvas } from '../graph/GraphCanvas';
import { CodeBlock } from './CodeBlock';
import { parseMermaidGraph } from './mermaidParse';
import styles from './ChatSurface.module.css';

/**
 * A `mermaid` fence in an assistant message, drawn rather than printed.
 *
 * The graph tool answers with a `graph LR` block and the model passes it into
 * its reply. Without this the reader gets diagram source in a code block, which
 * is the one form a graph is least useful in.
 *
 * ## It falls back rather than failing
 *
 * A fence this cannot read - a sequence diagram, a flowchart somebody typed by
 * hand, a block the model truncated mid-line - renders as the code block it
 * would have been. Showing the source is a worse answer than showing a picture
 * and a much better one than showing nothing, and it makes a parser bug visible
 * instead of silent.
 */
export function MermaidGraph({ source }: { source: string }) {
  const parsed = React.useMemo(() => parseMermaidGraph(source), [source]);

  if (!parsed) {
    return <CodeBlock code={source} lang="mermaid" />;
  }

  const named = parsed.edges.filter((edge) => edge.relation !== null).length;
  const dashed = parsed.edges.length - named;

  return (
    <figure className={styles.mdGraph}>
      {/*
        The canvas carries `height: 100%` inline, and an inline style beats a
        class rule, so the height cannot be given to it from the stylesheet. It
        needs a parent with a definite height instead: without one the
        percentage resolves against nothing, the ResizeObserver measures the
        canvas's own grown size, writes a larger backing store, and measures
        that in turn. It ran to 2968px here before the browser stopped painting
        an over-large surface altogether and drew a blank white rectangle.
      */}
      <div className={styles.mdGraphCanvas}>
        <GraphCanvas nodes={parsed.nodes} edges={parsed.edges} />
      </div>
      <figcaption className={styles.mdGraphCaption}>
        {parsed.nodes.length} {parsed.nodes.length === 1 ? 'term' : 'terms'}
        {named > 0 && `, ${named} named ${named === 1 ? 'relation' : 'relations'}`}
        {/*
          Said in the caption, not left to the line style alone. A dashed line
          means two terms shared a passage and nothing more; a reader who takes
          it for a stated relationship has been misled by the picture, so the
          picture carries the correction.
        */}
        {dashed > 0 &&
          `, ${dashed} unnamed ${dashed === 1 ? 'link' : 'links'} (shared passages only)`}
        {parsed.omitted > 0 && ` · ${parsed.omitted} more not drawn`}
      </figcaption>
    </figure>
  );
}
