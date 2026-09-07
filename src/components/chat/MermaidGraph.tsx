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
      <GraphCanvas nodes={parsed.nodes} edges={parsed.edges} />
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
