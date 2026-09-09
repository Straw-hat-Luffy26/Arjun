import React from 'react';

import { GraphCanvas } from '../graph/GraphCanvas';
import { CodeBlock } from './CodeBlock';
import { parseMermaidGraph } from './mermaidParse';
import { parseDiagram, type EntityBlock } from './mermaidDiagram';
import styles from './ChatSurface.module.css';

/**
 * A `mermaid` fence in an assistant message, drawn rather than printed.
 *
 * ## Two readers, in order
 *
 * A fence can come from either of two writers, and they are not the same.
 *
 * `knowledge.build_graph` answers a tool call with a `graph LR` block in an
 * exact format, and `mermaidParse` is a reader written for that one writer —
 * pinned to it by a test on each side. It runs first, so the tool's own output
 * always takes the path built for it, including the part no general reader
 * would know: that `(supplier)` at the end of a label is a *type*, not part of
 * the name.
 *
 * Everything else is a model writing ordinary Mermaid into its reply, which
 * `mermaidDiagram` reads. That was the missing half: `flowchart TD`, unquoted
 * labels, `A[Start] --> B{Choice}` and `erDiagram` all fell through to a code
 * block, so a reader who asked for a flowchart got its source instead.
 *
 * ## It falls back rather than failing
 *
 * A fence neither reader recognises — a sequence diagram, a Gantt chart, a
 * block the model truncated mid-line — renders as the code block it would have
 * been. Showing the source is a worse answer than showing a picture and a much
 * better one than showing nothing, and it makes a parser gap visible instead of
 * silent.
 */
export function MermaidGraph({ source }: { source: string }) {
  // The writer's own grammar first, then the general one. Ordered, not merged:
  // the two readers disagree about what a label means, and the specific one is
  // right about its own writer.
  const fromWriter = React.useMemo(() => parseMermaidGraph(source), [source]);
  const fromModel = React.useMemo(
    () => (fromWriter ? null : parseDiagram(source)),
    [source, fromWriter],
  );

  const parsed = fromWriter ?? fromModel;
  if (!parsed) {
    return <CodeBlock code={source} lang="mermaid" />;
  }

  const isSchema = fromModel?.kind === 'er';
  const entities = isSchema ? fromModel.entities : [];
  const named = parsed.edges.filter(edge => edge.relation !== null).length;
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

      {/*
        An ER diagram's columns are the diagram. The canvas draws boxes and
        lines, so the attributes are listed beneath it rather than dropped — a
        schema shown without its columns is a picture of a schema.
      */}
      {entities.length > 0 && <EntityTables entities={entities} />}

      <figcaption className={styles.mdGraphCaption}>
        {countOf(parsed.nodes.length, isSchema)}
        {named > 0 && `, ${named} named ${named === 1 ? 'relation' : 'relations'}`}
        {/*
          Said in the caption, not left to the line style alone. In a knowledge
          graph a dashed line means two terms shared a passage and nothing more;
          a reader who takes it for a stated relationship has been misled by the
          picture, so the picture carries the correction. A flowchart's dotted
          arrow means only that the model drew it dotted, so the parenthetical
          is not claimed there.
        */}
        {dashed > 0 &&
          `, ${dashed} unnamed ${dashed === 1 ? 'link' : 'links'}${
            fromWriter ? ' (shared passages only)' : ''
          }`}
        {parsed.omitted > 0 && ` · ${parsed.omitted} more not drawn`}
      </figcaption>
    </figure>
  );
}

/** Names what is actually drawn: entities in a schema, terms in a graph. */
function countOf(count: number, isSchema: boolean): string {
  if (isSchema) return `${count} ${count === 1 ? 'entity' : 'entities'}`;
  return `${count} ${count === 1 ? 'term' : 'terms'}`;
}

function EntityTables({ entities }: { entities: EntityBlock[] }) {
  const withColumns = entities.filter(entity => entity.attributes.length > 0);
  if (withColumns.length === 0) return null;

  return (
    <div className={styles.mdEntityList}>
      {withColumns.map(entity => (
        <table key={entity.id} className={styles.mdEntityTable}>
          <caption className={styles.mdEntityName}>{entity.label}</caption>
          <tbody>
            {entity.attributes.map(attribute => (
              <tr key={`${entity.id}-${attribute.name}`}>
                <td className={styles.mdEntityType}>{attribute.type}</td>
                <td className={styles.mdEntityField}>{attribute.name}</td>
                <td className={styles.mdEntityKey}>{attribute.key ?? ''}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ))}
    </div>
  );
}
