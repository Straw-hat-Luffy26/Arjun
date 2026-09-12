import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { AlertTriangle, ChevronLeft, ChevronRight, X } from 'lucide-react';
import { Spinner } from '../ui';
import {
  notebookResearchService,
  type ResolvedCitation,
  type SourcePage,
} from '../../services/notebookResearch.service';
import styles from './workspace.module.css';

/**
 * Reading one source, and landing on the passage a citation points at.
 *
 * ## What it shows, and what it says it is showing
 *
 * ARJUN stores the *extracted text* of a document, page by page — that is what
 * every answer was built from and what a citation resolves against. So that is
 * what this shows, and the header names the extraction (`pdf-text`,
 * `pdf-scan`, `docx`) rather than implying the original file is on screen.
 * Rendering the PDF beside text that came out of OCR would be the more
 * impressive screen and the more misleading one: the two can differ, and it is
 * the text that was read.
 *
 * ## Landing on a citation
 *
 * A citation carries the passage as it was *used*, from the turn's manifest,
 * and separately the page as it reads *now*. When the quote is still on the
 * page verbatim it is highlighted. When it is not — the document was re-read,
 * or removed — the quote is shown above the page with a sentence saying which
 * of those happened. It is never quietly matched to something nearby.
 */
interface SourceReaderProps {
  notebookId: string;
  documentSha256: string;
  documentName: string;
  /** A citation to land on, when the reader was opened from one. */
  citation?: ResolvedCitation | null;
  onClose: () => void;
}

/** A page's text, split around the passage to highlight. */
function splitForHighlight(
  text: string,
  start: number | null,
  length: number | null,
): { before: string; match: string; after: string } | null {
  if (start === null || length === null || length <= 0) return null;
  const characters = Array.from(text);
  if (start < 0 || start + length > characters.length) return null;
  return {
    before: characters.slice(0, start).join(''),
    match: characters.slice(start, start + length).join(''),
    after: characters.slice(start + length).join(''),
  };
}

export const SourceReader: React.FC<SourceReaderProps> = ({
  notebookId,
  documentSha256,
  documentName,
  citation,
  onClose,
}) => {
  const [page, setPage] = useState(citation?.page ?? 1);
  const [content, setContent] = useState<SourcePage | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // A new citation moves the reader to its page.
  useEffect(() => {
    if (citation) setPage(citation.page);
  }, [citation]);

  useEffect(() => {
    let live = true;
    setLoading(true);
    setError(null);
    notebookResearchService
      .sourcePage(notebookId, documentSha256, page)
      .then((read) => {
        if (live) setContent(read);
      })
      .catch((err) => {
        if (live) setError(String(err));
      })
      .finally(() => {
        if (live) setLoading(false);
      });
    return () => {
      live = false;
    };
  }, [notebookId, documentSha256, page]);

  const step = useCallback(
    (by: number) => {
      setPage((current) => {
        const next = current + by;
        const last = content?.totalPages ?? current;
        return Math.min(Math.max(next, 1), Math.max(last, 1));
      });
    },
    [content?.totalPages],
  );

  // Only highlight on the page the citation is actually on: paging away from it
  // must not carry the highlight along to unrelated text.
  const highlight = useMemo(() => {
    if (!citation || !content) return null;
    if (citation.page !== content.page) return null;
    if (citation.documentSha256 !== content.documentSha256) return null;
    return splitForHighlight(
      content.text,
      citation.highlightStart,
      citation.highlightLength,
    );
  }, [citation, content]);

  const showQuoteAsUsed =
    citation !== null &&
    citation !== undefined &&
    (citation.state !== 'available' || highlight === null);

  return (
    <section className={styles.reader} aria-label={`Reading ${documentName}`}>
      <header className={styles.readerHead}>
        <div className={styles.readerTitleBlock}>
          <h3 className={styles.readerTitle}>{documentName}</h3>
          {content && (
            <p className={styles.readerMeta}>
              Page {content.page} of {content.totalPages} · extracted text (
              {content.extractionKind})
            </p>
          )}
        </div>
        <button
          type="button"
          className={styles.iconAction}
          onClick={onClose}
          aria-label="Close the reader"
        >
          <X size={14} />
        </button>
      </header>

      {citation && citation.problem && (
        <p className={styles.readerWarning}>
          <AlertTriangle size={13} /> {citation.problem}
        </p>
      )}

      {content?.sourceTruncated && (
        <p className={styles.readerWarning}>
          <AlertTriangle size={13} /> This file was cut short when it was first read, so
          pages past the end of what is stored do not exist here at all.
        </p>
      )}

      {showQuoteAsUsed && citation && (
        <blockquote className={styles.quoteAsUsed}>
          <span className={styles.sectionLabel}>The passage as the answer used it</span>
          {citation.quote}
        </blockquote>
      )}

      {error && (
        <p className={styles.error}>
          <AlertTriangle size={13} /> {error}
        </p>
      )}

      <div className={styles.readerBody}>
        {loading && !content ? (
          <div className={styles.readerLoading}>
            <Spinner />
          </div>
        ) : content && content.text.trim() ? (
          <pre className={styles.readerText}>
            {highlight ? (
              <>
                {highlight.before}
                <mark className={styles.readerMark}>{highlight.match}</mark>
                {highlight.after}
              </>
            ) : (
              content.text
            )}
          </pre>
        ) : (
          <p className={styles.emptyHint}>
            No text was extracted from this page. It may be a blank page, or an image the
            reader could not recognise.
          </p>
        )}
      </div>

      <footer className={styles.readerFoot}>
        <button
          type="button"
          className={styles.pageStep}
          onClick={() => step(-1)}
          disabled={loading || page <= 1}
          aria-label="Previous page"
        >
          <ChevronLeft size={14} />
        </button>
        <span className={styles.pageNumber}>
          {content ? `${content.page} / ${content.totalPages}` : '—'}
        </span>
        <button
          type="button"
          className={styles.pageStep}
          onClick={() => step(1)}
          disabled={loading || (content ? page >= content.totalPages : true)}
          aria-label="Next page"
        >
          <ChevronRight size={14} />
        </button>
      </footer>
    </section>
  );
};
