import React from 'react';
import { CodeBlock } from './CodeBlock';
import { isClosingFence, openingFenceLanguage } from './markdownFence';
import { MermaidGraph } from './MermaidGraph';
import { WidgetFrame } from './WidgetFrame';
import { sanitizeSvg } from './svgSanitize';
import styles from './ChatSurface.module.css';

/**
 * Minimal Markdown renderer for the chat surface.
 *
 * Why hand-rolled:
 * - The Arjun project has an egress gate that forbids adding a runtime
 *   markdown library at this layer (the only chokepoint is the broker).
 * - A model response typically only needs: paragraphs, inline code,
 *   fenced code blocks, bold, italic, unordered lists, headings, and
 *   links. That's a small, well-bounded set we can render correctly
 *   without pulling in a parser.
 *
 * Streaming-safe: rendering is a pure function of `content`, so each
 * new chunk produces a deterministic, valid React tree.
 */

function escapeHtml(s: string): string {
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

/**
 * Apply inline markdown transforms to a single line of text.
 * The order of operations matters: code spans first (to protect their
 * contents from later transforms), then bold, then italic, then links.
 */
function renderInline(line: string, keyPrefix: string): React.ReactNode[] {
  const out: React.ReactNode[] = [];
  let i = 0;
  let key = 0;
  let buf = '';

  const flushBuf = () => {
    if (buf) {
      out.push(<span key={`${keyPrefix}-t-${key++}`}>{buf}</span>);
      buf = '';
    }
  };

  while (i < line.length) {
    // Inline code: `...`
    if (line[i] === '`') {
      const end = line.indexOf('`', i + 1);
      if (end !== -1) {
        flushBuf();
        out.push(
          <code key={`${keyPrefix}-c-${key++}`} className={styles.mdCode}>
            {line.slice(i + 1, end)}
          </code>,
        );
        i = end + 1;
        continue;
      }
    }
    // Bold: **...**
    if (line[i] === '*' && line[i + 1] === '*') {
      const end = line.indexOf('**', i + 2);
      if (end !== -1) {
        flushBuf();
        out.push(
          <strong key={`${keyPrefix}-b-${key++}`}>{line.slice(i + 2, end)}</strong>,
        );
        i = end + 2;
        continue;
      }
    }
    // Italic: *...* (single asterisk, but not double)
    if (line[i] === '*' && line[i + 1] !== '*' && (i === 0 || line[i - 1] !== '*')) {
      const end = findUnescaped(line, '*', i + 1);
      if (end !== -1 && end > i + 1) {
        flushBuf();
        out.push(
          <em key={`${keyPrefix}-i-${key++}`}>{line.slice(i + 1, end)}</em>,
        );
        i = end + 1;
        continue;
      }
    }
    // Link: [text](url)
    if (line[i] === '[') {
      const labelEnd = line.indexOf(']', i + 1);
      if (labelEnd !== -1 && line[labelEnd + 1] === '(') {
        const urlEnd = line.indexOf(')', labelEnd + 2);
        if (urlEnd !== -1) {
          const label = line.slice(i + 1, labelEnd);
          const url = line.slice(labelEnd + 2, urlEnd);
          flushBuf();
          out.push(
            <a
              key={`${keyPrefix}-a-${key++}`}
              href={url}
              target="_blank"
              rel="noreferrer"
              className={styles.mdLink}
            >
              {label}
            </a>,
          );
          i = urlEnd + 1;
          continue;
        }
      }
    }
    buf += line[i];
    i += 1;
  }
  flushBuf();
  return out;
}

function findUnescaped(s: string, ch: string, start: number): number {
  for (let i = start; i < s.length; i += 1) {
    if (s[i] === ch) return i;
  }
  return -1;
}

/** How a table column is aligned, from its delimiter row. */
type ColumnAlign = 'left' | 'center' | 'right';

interface Block {
  kind: 'p' | 'h' | 'ul' | 'ol' | 'code' | 'blockquote' | 'table';
  level?: number;
  lang?: string;
  text: string;
  /**
   * `code` only: whether the closing fence has arrived.
   *
   * Code streams, so a block is emitted before it is finished. That is right
   * for source — the reader watches it arrive — and wrong for a widget, which
   * would otherwise be built and run once per keystroke, half-written.
   */
  closed?: boolean;
  /** `table` only: the header cells. */
  header?: string[];
  /** `table` only: the body rows, already split into cells. */
  rows?: string[][];
  /** `table` only: one entry per column. */
  align?: ColumnAlign[];
}

/** Splits one table line into cells, tolerating the optional outer pipes. */
function tableCells(line: string): string[] {
  let s = line.trim();
  if (s.startsWith('|')) s = s.slice(1);
  if (s.endsWith('|')) s = s.slice(0, -1);
  return s.split('|').map(c => c.trim());
}

/**
 * Whether this line is a table's delimiter row (`| :--- | ---: |`).
 *
 * The delimiter is what makes a table a table: a line of pipes on its own is
 * far more often prose. Requiring it is why a sentence containing a vertical
 * bar does not become a one-column table.
 */
function tableAlignments(line: string): ColumnAlign[] | null {
  if (!line.includes('|')) return null;
  const cells = tableCells(line);
  if (cells.length === 0) return null;
  const out: ColumnAlign[] = [];
  for (const cell of cells) {
    if (!/^:?-{1,}:?$/.test(cell)) return null;
    const left = cell.startsWith(':');
    const right = cell.endsWith(':');
    out.push(left && right ? 'center' : right ? 'right' : 'left');
  }
  return out;
}

/**
 * Tokenize the input into a sequence of blocks. The tokenizer is
 * intentionally minimal � it recognises the most common shapes a model
 * emits in chat and folds everything else into a paragraph.
 */
function tokenize(input: string): Block[] {
  const lines = input.split('\n');
  const blocks: Block[] = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];

    // Fenced code block. `null` means this line is not a fence at all;
    // `undefined` means a fence that named no language. See `markdownFence`.
    const lang = openingFenceLanguage(line);
    if (lang !== null) {
      const body: string[] = [];
      i += 1;
      while (i < lines.length && !isClosingFence(lines[i])) {
        body.push(lines[i]);
        i += 1;
      }
      const closed = i < lines.length;
      if (closed) i += 1; // skip closing fence
      // Emitted even when the closing fence has not arrived, which is what
      // makes code stream. A block that waited for its close would leave the
      // reader watching a blank space for the whole time the model spends
      // writing the code — the part of an answer that takes longest.
      blocks.push({ kind: 'code', lang, text: body.join('\n'), closed });
      continue;
    }

    // Pipe table: a header row, a delimiter row, then body rows.
    //
    // Checked before the paragraph fallback, which is what used to swallow
    // these: a model asked to lay out a work order emits a markdown table,
    // and without this the reader got a wall of `| Item | Qty | Ref |` with
    // the alignment colons still in it.
    if (line.includes('|') && i + 1 < lines.length) {
      const align = tableAlignments(lines[i + 1]);
      if (align) {
        const header = tableCells(line);
        const rows: string[][] = [];
        i += 2;
        while (i < lines.length && lines[i].includes('|') && lines[i].trim() !== '') {
          const cells = tableCells(lines[i]);
          // Ragged rows are padded rather than dropped. A row with a cell
          // missing is still data somebody wrote down.
          while (cells.length < header.length) cells.push('');
          rows.push(cells.slice(0, Math.max(header.length, 1)));
          i += 1;
        }
        blocks.push({ kind: 'table', text: '', header, rows, align });
        continue;
      }
    }

    // Heading: # ... ######
    const heading = line.match(/^(#{1,6})\s+(.+)$/);
    if (heading) {
      blocks.push({ kind: 'h', level: heading[1].length, text: heading[2] });
      i += 1;
      continue;
    }

    // Unordered list
    if (/^[\s]*[-*]\s+/.test(line)) {
      const items: string[] = [];
      while (i < lines.length && /^[\s]*[-*]\s+/.test(lines[i])) {
        items.push(lines[i].replace(/^[\s]*[-*]\s+/, ''));
        i += 1;
      }
      blocks.push({ kind: 'ul', text: items.join('\n') });
      continue;
    }

    // Ordered list
    if (/^[\s]*\d+\.\s+/.test(line)) {
      const items: string[] = [];
      while (i < lines.length && /^[\s]*\d+\.\s+/.test(lines[i])) {
        items.push(lines[i].replace(/^[\s]*\d+\.\s+/, ''));
        i += 1;
      }
      blocks.push({ kind: 'ol', text: items.join('\n') });
      continue;
    }

    // Blockquote: > ...
    if (/^>\s+/.test(line)) {
      const body: string[] = [];
      while (i < lines.length && /^>\s+/.test(lines[i])) {
        body.push(lines[i].replace(/^>\s+/, ''));
        i += 1;
      }
      blocks.push({ kind: 'blockquote', text: body.join('\n') });
      continue;
    }

    // Blank line: skip
    if (line.trim() === '') {
      i += 1;
      continue;
    }

    // Paragraph: collect consecutive non-blank, non-special lines.
    const para: string[] = [line];
    i += 1;
    while (
      i < lines.length &&
      lines[i].trim() !== '' &&
      !/^```/.test(lines[i]) &&
      !/^#{1,6}\s/.test(lines[i]) &&
      !/^[\s]*[-*]\s+/.test(lines[i]) &&
      !/^[\s]*\d+\.\s+/.test(lines[i]) &&
      !/^>\s+/.test(lines[i]) &&
      // A table starting on the next line ends this paragraph, or its header
      // row would be absorbed into the prose above it.
      !(i + 1 < lines.length && lines[i].includes('|') && tableAlignments(lines[i + 1]))
    ) {
      para.push(lines[i]);
      i += 1;
    }
    blocks.push({ kind: 'p', text: para.join('\n') });
  }

  return blocks;
}

/**
 * A data table, rendered so it can actually be read.
 *
 * The scroll container is the load-bearing part: a wide table inside a chat
 * column either wraps into unreadable slivers or pushes the whole page
 * sideways. It scrolls in its own box instead, and the page never does.
 */
function MarkdownTable({
  header,
  rows,
  align,
  renderCell,
}: {
  header: string[];
  rows: string[][];
  align: ColumnAlign[];
  renderCell: (text: string, key: string) => React.ReactNode;
}) {
  const alignFor = (i: number) => align[i] ?? 'left';
  return (
    <div className="my-4 overflow-x-auto rounded-lg border border-line">
      <table className="w-full border-collapse text-[13px]">
        <thead>
          <tr>
            {header.map((cell, i) => (
              <th
                key={`th-${i}`}
                style={{ textAlign: alignFor(i) }}
                className="border-b border-line bg-surface-hover/50 px-[14px] py-[10px]
                           font-semibold text-ink"
              >
                {renderCell(cell, `th-i-${i}`)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, r) => (
            <tr key={`tr-${r}`} className="odd:bg-transparent even:bg-surface-raised/40">
              {row.map((cell, c) => (
                <td
                  key={`td-${r}-${c}`}
                  style={{ textAlign: alignFor(c) }}
                  className="border-b border-line/60 px-[14px] py-[10px] align-top
                             text-ink-muted last:border-r-0"
                >
                  {renderCell(cell, `td-i-${r}-${c}`)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function Markdown({
  content,
  onPrompt,
}: {
  content: string;
  /** Raised when a widget in this message calls `sendPrompt`. */
  onPrompt?: (text: string) => void;
}) {
  const blocks = React.useMemo(() => tokenize(content), [content]);

  return (
    <div className={styles.markdown}>
      {blocks.map((block, idx) => {
        const key = `b-${idx}`;
        switch (block.kind) {
          case 'h': {
            const level = Math.min(Math.max(block.level ?? 1, 1), 6);
            // Narrowed to the six heading tags rather than every intrinsic
            // element. `keyof JSX.IntrinsicElements` stopped being callable
            // once react-three-fiber augmented that interface with three.js
            // elements, and the wider type was never what was meant here.
            const Tag = `h${level}` as 'h1' | 'h2' | 'h3' | 'h4' | 'h5' | 'h6';
            return (
              <Tag key={key} className={styles.mdHeading} data-level={level}>
                {renderInline(block.text, `${key}-i`)}
              </Tag>
            );
          }
          case 'p':
            return (
              <p key={key} className={styles.mdParagraph}>
                {renderInline(block.text, `${key}-i`)}
              </p>
            );
          case 'code':
            // A mermaid fence is a picture the model was handed, not source it
            // wrote. Drawn rather than printed; `MermaidGraph` falls back to a
            // code block for any diagram it cannot read.
            if (block.lang === 'mermaid') {
              return <MermaidGraph key={key} source={block.text} />;
            }
            // An SVG fence is a picture too - a chart, or a diagram the model
            // was handed. Drawn rather than printed, and sanitised first: this
            // markup came from a model, and an SVG has a script model.
            if (block.lang === 'svg') {
              return <InlineSvg key={key} source={block.text} />;
            }
            // A widget fence is a page the model wrote to be *run*, not read.
            // Sandboxed, served its own origin and its own policy; see
            // `WidgetFrame`. Deliberately not `html`, which stays source.
            if (block.lang === 'widget') {
              return (
                <WidgetFrame
                  key={key}
                  html={block.text}
                  complete={block.closed ?? false}
                  onPrompt={onPrompt}
                />
              );
            }
            return <CodeBlock key={key} code={block.text} lang={block.lang} />;
          case 'table':
            return (
              <MarkdownTable
                key={key}
                header={block.header ?? []}
                rows={block.rows ?? []}
                align={block.align ?? []}
                renderCell={(text, cellKey) => renderInline(text, cellKey)}
              />
            );
          case 'ul':
            return (
              <ul key={key} className={styles.mdList}>
                {block.text.split('\n').map((item, i) => (
                  <li key={`${key}-li-${i}`}>{renderInline(item, `${key}-i-${i}`)}</li>
                ))}
              </ul>
            );
          case 'ol':
            return (
              <ol key={key} className={styles.mdList}>
                {block.text.split('\n').map((item, i) => (
                  <li key={`${key}-li-${i}`}>{renderInline(item, `${key}-i-${i}`)}</li>
                ))}
              </ol>
            );
          case 'blockquote':
            return (
              <blockquote key={key} className={styles.mdBlockquote}>
                {renderInline(block.text, `${key}-i`)}
              </blockquote>
            );
        }
        return null;
      })}
    </div>
  );
}

// Suppress unused warning for escapeHtml (kept available for future raw-html
// support without re-importing it).
void escapeHtml;

/**
 * An `svg` fence, drawn.
 *
 * Falls back to the source, exactly as `MermaidGraph` does. A fence that does
 * not survive sanitising is shown as text rather than silently dropped: a
 * reader seeing nothing cannot tell a refused drawing from a model that never
 * produced one, and the difference matters when the reason is that something
 * tried to put a script in it.
 */
function InlineSvg({ source }: { source: string }) {
  const safe = React.useMemo(() => sanitizeSvg(source), [source]);

  if (!safe) {
    return <CodeBlock code={source} lang="svg" />;
  }

  return (
    <figure
      className={styles.mdSvg}
      // Sanitised immediately above by an allowlist that drops every element
      // and attribute it does not name, including scripts and event handlers.
      dangerouslySetInnerHTML={{ __html: safe }}
    />
  );
}
