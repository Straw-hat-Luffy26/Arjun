import React from 'react';
import { AlertTriangle, Check, FileText, Trash2 } from 'lucide-react';
import type { NotebookDocument } from '../../services/notebook.service';
import styles from './workspace.module.css';

/**
 * The notebook's sources, and which ones the next question will use.
 *
 * ## Two different actions, deliberately separated
 *
 * Including a source and opening a source are different intentions, and a list
 * where clicking a row did both would make one of them an accident. The
 * checkbox governs what the next question reads; the name opens the reader.
 * Neither is implied by the other: a person often reads a document they have
 * excluded, to decide whether to include it.
 *
 * ## Why "0 selected" is not the same as "all"
 *
 * An empty selection is shown as *every source*, which is what the backend does
 * with an empty list — but the header says so in words rather than leaving a
 * row of unticked boxes to imply nothing will be read. The one reading a person
 * must never be left with is that their question silently used less than they
 * thought.
 */
interface SourceListProps {
  sources: NotebookDocument[];
  /** Content addresses the next question will use. Empty means all of them. */
  selected: Set<string>;
  onToggle: (documentSha256: string) => void;
  onSelectAll: () => void;
  onSelectNone: () => void;
  /** Opens the reader on this source. */
  onOpen: (source: NotebookDocument) => void;
  onRemove: (source: NotebookDocument) => void;
  /** Sources whose extracted text could not be read, by content address. */
  unreadable: Set<string>;
  /** The source the reader currently has open. */
  openSha: string | null;
  busy?: boolean;
}

export const SourceList: React.FC<SourceListProps> = ({
  sources,
  selected,
  onToggle,
  onSelectAll,
  onSelectNone,
  onOpen,
  onRemove,
  unreadable,
  openSha,
  busy,
}) => {
  const usingAll = selected.size === 0;
  const count = usingAll ? sources.length : selected.size;

  if (sources.length === 0) {
    return (
      <div className={styles.empty}>
        <FileText size={22} />
        <p>No sources yet.</p>
        <p className={styles.emptyHint}>
          Add documents to this notebook. Everything you ask it will be answered from
          them and cited back to the page it came from.
        </p>
      </div>
    );
  }

  return (
    <div className={styles.sourceList}>
      <div className={styles.sourceListHead}>
        <span className={styles.sectionLabel}>
          {count} of {sources.length} will be read
        </span>
        <div className={styles.sourceListActions}>
          <button type="button" className={styles.linkAction} onClick={onSelectAll}>
            All
          </button>
          <button
            type="button"
            className={styles.linkAction}
            onClick={onSelectNone}
            disabled={usingAll}
          >
            Reset
          </button>
        </div>
      </div>

      {usingAll && (
        <p className={styles.sourceListNote}>
          Nothing is narrowed, so every source in this notebook is read. Tick sources to
          answer from only those.
        </p>
      )}

      <ul className={styles.sourceRows}>
        {sources.map((source) => {
          const included = usingAll || selected.has(source.documentSha256);
          const broken = unreadable.has(source.documentSha256);
          return (
            <li
              key={source.documentSha256}
              className={styles.sourceRow}
              data-open={source.documentSha256 === openSha || undefined}
            >
              <button
                type="button"
                role="checkbox"
                aria-checked={included}
                aria-label={`Use ${source.documentName} for questions`}
                className={styles.sourceCheck}
                data-checked={included || undefined}
                onClick={() => onToggle(source.documentSha256)}
                disabled={busy}
              >
                {included && <Check size={11} strokeWidth={3} />}
              </button>

              <button
                type="button"
                className={styles.sourceName}
                title={`Open ${source.documentName}`}
                onClick={() => onOpen(source)}
              >
                <span className={styles.sourceNameText}>{source.documentName}</span>
                {broken && (
                  <span className={styles.sourceProblem}>
                    <AlertTriangle size={11} /> text missing
                  </span>
                )}
              </button>

              <button
                type="button"
                className={styles.sourceRemove}
                title={`Take ${source.documentName} out of this notebook`}
                aria-label={`Take ${source.documentName} out of this notebook`}
                onClick={() => onRemove(source)}
                disabled={busy}
              >
                <Trash2 size={12} />
              </button>
            </li>
          );
        })}
      </ul>
    </div>
  );
};
