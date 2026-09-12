import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { AlertTriangle, BookmarkPlus, BookOpen, ExternalLink, X } from 'lucide-react';
import {
  notebookResearchService,
  type EvidenceManifest,
  type ResolvedCitation,
} from '../../services/notebookResearch.service';
import { Markdown } from './Markdown';
import styles from './ChatSurface.module.css';

/**
 * Citations, and the evidence behind them, wherever an answer is shown.
 *
 * Extracted from the Notebooks page's own chat, where all of this used to
 * live. That mattered as soon as the question moved into the main composer: an
 * answer rendered in main chat would have shown `[E1]` as three characters of
 * literal text, and the manifest that makes it resolvable would never have
 * been asked for.
 *
 * ## What a marker is allowed to do
 *
 * Resolve against the manifest recorded for *that message*, and nothing else.
 * The marker is not an index into anything this file holds — it is looked up in
 * Rust, against the run that produced the answer, for the signed-in owner. A
 * marker with no entry says so rather than silently doing nothing, because a
 * citation that quietly fails to open is indistinguishable from one that was
 * never real.
 */

/** Splits an answer into text and the `[En]` markers inside it. */
export function withCitations(
  text: string,
  onMarker: (marker: number) => void,
  known?: ReadonlySet<number>,
): React.ReactNode[] {
  const parts: React.ReactNode[] = [];
  const pattern = /\[E(\d+)\]/g;
  let last = 0;
  let match: RegExpExecArray | null;
  let key = 0;
  while ((match = pattern.exec(text)) !== null) {
    if (match.index > last) {
      parts.push(<Markdown key={`t${key++}`} content={text.slice(last, match.index)} />);
    }
    const marker = Number(match[1]);
    // A marker the manifest does not carry is drawn inert and says why on
    // hover, rather than as a button that does nothing when pressed.
    const resolvable = !known || known.has(marker);
    parts.push(
      <button
        key={`c${key++}`}
        type="button"
        className={styles.citation}
        data-unresolved={!resolvable || undefined}
        disabled={!resolvable}
        title={
          resolvable
            ? `Open the passage behind [E${marker}]`
            : `No evidence was recorded for [E${marker}] in this answer.`
        }
        onClick={() => onMarker(marker)}
      >
        E{marker}
      </button>,
    );
    last = match.index + match[0].length;
  }
  if (last < text.length) {
    parts.push(<Markdown key={`t${key++}`} content={text.slice(last)} />);
  }
  return parts;
}

/** Whether an answer has anything worth asking for a manifest about. */
export function hasCitations(text: string): boolean {
  return /\[E\d+\]/.test(text);
}

/**
 * The manifest for one answer, fetched once.
 *
 * Three outcomes, not two: not asked yet, asked and there is none, and asked
 * and here it is. A turn that was never scoped to a notebook has no manifest
 * and that is not a failure — but nor is it the same as a manifest whose write
 * failed, which is why the problem is reported separately.
 */
export function useTurnEvidence(
  conversationId: string | null,
  messageId: string | null,
  enabled: boolean,
): { manifest: EvidenceManifest | null; problem: string | null; loading: boolean } {
  const [manifest, setManifest] = useState<EvidenceManifest | null>(null);
  const [problem, setProblem] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!enabled || !conversationId || !messageId) {
      setManifest(null);
      setProblem(null);
      return;
    }
    let live = true;
    setLoading(true);
    setProblem(null);
    notebookResearchService
      .turnEvidence(conversationId, messageId)
      .then(found => {
        if (live) setManifest(found);
      })
      .catch(error => {
        if (!live) return;
        setManifest(null);
        // Said out loud. An answer carrying `[E1]` whose manifest cannot be
        // read is an answer that looks citable and is not, and that is the one
        // thing this must never present as ordinary.
        setProblem(
          error instanceof Error
            ? error.message
            : 'The evidence behind this answer could not be read.',
        );
      })
      .finally(() => {
        if (live) setLoading(false);
      });
    return () => {
      live = false;
    };
  }, [conversationId, messageId, enabled]);

  return { manifest, problem, loading };
}

export interface EvidenceFooterProps {
  manifest: EvidenceManifest;
  onOpenNotebook?: (notebookId: string) => void;
}

/**
 * What an answer was actually built on, under the answer.
 *
 * Names the sources that were used and the ones that were not, rather than
 * counting them. "Three of your eight sources were not used" is the difference
 * between an answer that read the collection and one that read part of it, and
 * a count cannot say which three.
 */
export const EvidenceFooter: React.FC<EvidenceFooterProps> = ({
  manifest,
  onOpenNotebook,
}) => {
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState<string | null>(null);
  const [saveProblem, setSaveProblem] = useState<string | null>(null);

  /**
   * Saves this answer into the notebook its evidence came from.
   *
   * `manifest.notebookId`, deliberately -- not whichever notebook the chip
   * happens to name now. Somebody who asks Unit Four a question, switches the
   * chip to Procurement, then scrolls up and saves the earlier answer must get
   * it filed against Unit Four, because that is the notebook whose sources it
   * cites. Reading the live selection here would file a cited answer under a
   * notebook that has none of its evidence in it.
   */
  const save = useCallback(async () => {
    const title = window.prompt('Save this answer as:', 'Saved answer');
    if (!title) return;
    setSaving(true);
    setSaveProblem(null);
    try {
      const note = await notebookResearchService.saveAnswer({
        notebookId: manifest.notebookId,
        conversationId: manifest.conversationId,
        messageId: manifest.messageId,
        title,
        kind: 'answer',
      });
      setSaved(note.title);
    } catch (error) {
      setSaveProblem(
        error instanceof Error ? error.message : 'That answer could not be saved.',
      );
    } finally {
      setSaving(false);
    }
  }, [manifest.notebookId, manifest.conversationId, manifest.messageId]);

  const used = useMemo(() => {
    const names = new Map<string, number>();
    for (const entry of manifest.entries) {
      names.set(entry.documentName, (names.get(entry.documentName) ?? 0) + 1);
    }
    return [...names.entries()];
  }, [manifest.entries]);

  return (
    <div className={styles.evidenceFooter}>
      <div className={styles.evidenceFooterHead}>
        <BookOpen size={12} aria-hidden="true" />
        <span>
          {manifest.entries.length}{' '}
          {manifest.entries.length === 1 ? 'passage' : 'passages'} from {used.length}{' '}
          {used.length === 1 ? 'source' : 'sources'}
        </span>
        {/* The mode is reported, never implied. A keyword scan described as
          * semantic search is the claim this repository has a standing rule
          * against, and the manifest is where the truth of it is recorded. */}
        <span className={styles.evidenceMode}>{manifest.retrievalMode} search</span>
        <button
          type="button"
          className={styles.evidenceLink}
          onClick={() => void save()}
          disabled={saving || saved !== null}
        >
          <BookmarkPlus size={11} aria-hidden="true" />
          {saved ? `Saved as “${saved}”` : saving ? 'Saving…' : 'Save to notebook'}
        </button>
        {onOpenNotebook && (
          <button
            type="button"
            className={styles.evidenceLink}
            onClick={() => onOpenNotebook(manifest.notebookId)}
          >
            <ExternalLink size={11} aria-hidden="true" /> Open notebook
          </button>
        )}
      </div>

      {saveProblem && (
        <p className={styles.evidenceProblem} role="alert">
          {saveProblem}
        </p>
      )}

      {used.length > 0 && (
        <ul className={styles.evidenceUsed}>
          {used.map(([name, count]) => (
            <li key={name}>
              {name} <span className={styles.evidenceCount}>×{count}</span>
            </li>
          ))}
        </ul>
      )}

      {manifest.sourcesUnused.length > 0 && (
        <p className={styles.evidenceUnused}>
          Not used: {manifest.sourcesUnused.join(', ')}
        </p>
      )}

      {manifest.limitations.length > 0 && (
        <ul className={styles.evidenceLimitations}>
          {manifest.limitations.map((limitation, i) => (
            <li key={i}>
              <AlertTriangle size={11} aria-hidden="true" /> {limitation}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
};

export interface CitationReaderProps {
  citation: ResolvedCitation;
  onClose: () => void;
}

/**
 * The passage behind one marker, opened where the answer is.
 *
 * Shows the quote as the answer used it *and* the page as it reads now, which
 * are not always the same text. A source re-read at a better OCR stop moves
 * underneath a citation written a month ago, and the honest thing is to show
 * both rather than to pick one and present it as the source.
 */
export const CitationReader: React.FC<CitationReaderProps> = ({ citation, onClose }) => {
  // Escape closes it, like every other overlay in the app.
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  const highlighted = useMemo(() => {
    if (
      citation.pageText === null ||
      citation.highlightStart === null ||
      citation.highlightLength === null
    ) {
      return null;
    }
    const start = citation.highlightStart;
    const end = start + citation.highlightLength;
    return {
      before: citation.pageText.slice(0, start),
      match: citation.pageText.slice(start, end),
      after: citation.pageText.slice(end),
    };
  }, [citation]);

  return (
    <div className={styles.citationOverlay} role="dialog" aria-modal="true">
      <div className={styles.citationBackdrop} onClick={onClose} />
      <div className={styles.citationPanel}>
        <div className={styles.citationHead}>
          <h3 className={styles.citationTitle}>
            [E{citation.marker}] {citation.documentName}
            <span className={styles.citationWhere}>
              page {citation.page}
              {citation.sectionPath.length > 0
                ? ` · ${citation.sectionPath.join(' › ')}`
                : ''}
            </span>
          </h3>
          <button
            type="button"
            className={styles.inspectorClose}
            onClick={onClose}
            aria-label="Close"
          >
            <X size={14} />
          </button>
        </div>

        {citation.problem && (
          <p className={styles.notebookChooserProblem} role="alert">
            <AlertTriangle size={12} aria-hidden="true" /> {citation.problem}
          </p>
        )}

        <section className={styles.citationSection}>
          <h4 className={styles.citationSectionTitle}>As the answer used it</h4>
          <blockquote className={styles.citationQuote}>{citation.quote}</blockquote>
        </section>

        {highlighted && (
          <section className={styles.citationSection}>
            <h4 className={styles.citationSectionTitle}>
              {citation.state === 'sourceChanged'
                ? 'The page as it reads now — this source has been re-read since'
                : 'On the page'}
            </h4>
            <p className={styles.citationPage}>
              {highlighted.before}
              <mark className={styles.citationMark}>{highlighted.match}</mark>
              {highlighted.after}
            </p>
          </section>
        )}
      </div>
    </div>
  );
};

/**
 * Opens a citation for one message, holding the reader and its failures.
 *
 * A hook rather than a component, so the caller decides where the overlay is
 * mounted and so "which marker is open" does not have to be lifted into a
 * message list that has no other use for it.
 */
export function useCitationReader(conversationId: string | null) {
  const [citation, setCitation] = useState<ResolvedCitation | null>(null);
  const [problem, setProblem] = useState<string | null>(null);
  const [opening, setOpening] = useState(false);
  // Which request is the current one. Two markers clicked in quick succession
  // are two calls, and without this the slower one wins whichever was clicked
  // first -- so the panel opens on the citation somebody has already moved on
  // from.
  const latest = useRef(0);

  const open = useCallback(
    async (messageId: string, marker: number) => {
      if (!conversationId) return;
      const request = ++latest.current;
      setProblem(null);
      setOpening(true);
      try {
        const resolved = await notebookResearchService.openCitation(
          conversationId,
          messageId,
          marker,
        );
        if (latest.current === request) setCitation(resolved);
      } catch (error) {
        if (latest.current !== request) return;
        setProblem(
          error instanceof Error ? error.message : `[E${marker}] could not be opened.`,
        );
      } finally {
        if (latest.current === request) setOpening(false);
      }
    },
    [conversationId],
  );

  return {
    citation,
    problem,
    opening,
    open,
    close: useCallback(() => setCitation(null), []),
    dismissProblem: useCallback(() => setProblem(null), []),
  };
}
