import React, { useCallback, useEffect, useState } from 'react';
import { AlertTriangle, ArrowRight, Check, Plus, Repeat, Trash2, X } from 'lucide-react';
import { Button, Spinner } from '../ui';
import {
  notebookResearchService,
  type Assertion,
} from '../../services/notebookResearch.service';
import styles from './workspace.module.css';

/**
 * Reviewing what the extraction passes claimed about the sources.
 *
 * ## Three standings, never merged
 *
 * A relationship a model proposed, one a person confirmed, and one a person
 * wrote themselves are different kinds of claim and are drawn differently. The
 * distinction is not cosmetic: a proposed claim has evidence nobody has checked,
 * and a hand-written one has no evidence at all. An interface that showed them
 * alike would let somebody act on the weakest as though it were the strongest.
 *
 * ## Direction is a first-class thing to get wrong
 *
 * Every row reads *subject → object* in the order the extractor produced it, and
 * "Swap" writes the reverse as the person's own corrected claim. A claim
 * migrated from the old storage — which could not express direction — is marked
 * unverified rather than being drawn with a confident arrow.
 *
 * ## Rejection keeps the row
 *
 * Rejecting records the refusal instead of deleting the claim, so the next
 * extraction pass does not propose the same wrong thing with nothing to say it
 * was already refused. Only a claim a person wrote themselves can be deleted.
 */
interface RelationshipReviewProps {
  notebookId: string;
  /** Narrows the list to one source, when the workspace has one scoped. */
  documentSha256: string | null;
  /** Takes a claim into the notebook's chat as the question's focus. */
  onAskAbout: (assertion: Assertion) => void;
  /** Opens a source at the passage behind a claim. */
  onOpenEvidence: (documentSha256: string, page: number) => void;
  /** Raised after any change, so the graph can re-read. */
  onChanged: () => void;
}

function standingOf(
  assertion: Assertion,
): { label: string; tone: 'strong' | 'weak' | 'refused' } {
  if (assertion.status === 'rejected') return { label: 'Rejected', tone: 'refused' };
  if (assertion.provenance === 'user' && assertion.evidence.length === 0) {
    return { label: 'Yours — no source evidence', tone: 'weak' };
  }
  if (assertion.status === 'accepted') return { label: 'Confirmed by you', tone: 'strong' };
  return { label: 'Proposed by a model', tone: 'weak' };
}

export const RelationshipReview: React.FC<RelationshipReviewProps> = ({
  notebookId,
  documentSha256,
  onAskAbout,
  onOpenEvidence,
  onChanged,
}) => {
  const [rows, setRows] = useState<Assertion[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [includeRejected, setIncludeRejected] = useState(false);
  const [busyId, setBusyId] = useState<string | null>(null);
  const [adding, setAdding] = useState(false);
  const [draft, setDraft] = useState({ subject: '', predicate: '', object: '' });

  const reload = useCallback(async () => {
    setError(null);
    try {
      setRows(
        await notebookResearchService.assertions(notebookId, {
          documentSha256,
          includeRejected,
        }),
      );
    } catch (err) {
      setError(String(err));
    } finally {
      setLoading(false);
    }
  }, [notebookId, documentSha256, includeRejected]);

  useEffect(() => {
    setLoading(true);
    void reload();
  }, [reload]);

  const act = useCallback(
    async (id: string, work: () => Promise<unknown>) => {
      setBusyId(id);
      setError(null);
      try {
        await work();
        await reload();
        onChanged();
      } catch (err) {
        setError(String(err));
      } finally {
        setBusyId(null);
      }
    },
    [onChanged, reload],
  );

  const addByHand = useCallback(async () => {
    const subject = draft.subject.trim().toLowerCase();
    const object = draft.object.trim().toLowerCase();
    const predicate = draft.predicate.trim();
    if (!subject || !object || !predicate) return;
    setError(null);
    try {
      await notebookResearchService.createAssertion({
        notebookId,
        subject,
        predicate,
        object,
      });
      setDraft({ subject: '', predicate: '', object: '' });
      setAdding(false);
      await reload();
      onChanged();
    } catch (err) {
      setError(String(err));
    }
  }, [draft, notebookId, onChanged, reload]);

  return (
    <section className={styles.review} aria-label="Relationship review">
      <header className={styles.notesHead}>
        <span className={styles.sectionLabel}>
          Relationships {documentSha256 && '· this source only'}
        </span>
        <div className={styles.notesActions}>
          <label className={styles.toggleLabel}>
            <input
              type="checkbox"
              checked={includeRejected}
              onChange={(event) => setIncludeRejected(event.target.checked)}
            />
            show rejected
          </label>
          <Button size="sm" variant="ghost" onClick={() => setAdding((on) => !on)}>
            <Plus size={13} /> Add
          </Button>
        </div>
      </header>

      {adding && (
        <>
          <div className={styles.addRow}>
            <input
              className={styles.addInput}
              placeholder="subject"
              value={draft.subject}
              onChange={(event) => setDraft((d) => ({ ...d, subject: event.target.value }))}
              aria-label="Subject term"
            />
            <ArrowRight size={13} className={styles.addArrow} />
            <input
              className={styles.addInput}
              placeholder="relationship"
              value={draft.predicate}
              onChange={(event) =>
                setDraft((d) => ({ ...d, predicate: event.target.value }))
              }
              aria-label="Relationship"
            />
            <ArrowRight size={13} className={styles.addArrow} />
            <input
              className={styles.addInput}
              placeholder="object"
              value={draft.object}
              onChange={(event) => setDraft((d) => ({ ...d, object: event.target.value }))}
              aria-label="Object term"
            />
            <Button size="sm" onClick={() => void addByHand()}>
              Add
            </Button>
          </div>
          <p className={styles.sourceListNote}>
            A relationship you write has no passage behind it. It is kept as yours, shown
            as unverified, and any answer that uses it is told the same.
          </p>
        </>
      )}

      {error && (
        <p className={styles.error}>
          <AlertTriangle size={13} /> {error}
        </p>
      )}

      {loading ? (
        <div className={styles.readerLoading}>
          <Spinner />
        </div>
      ) : rows.length === 0 ? (
        <div className={styles.empty}>
          <p>No relationships yet.</p>
          <p className={styles.emptyHint}>
            Build the graph and run the relationship pass from the Graph tab, or add one
            yourself. A relationship names what a link between two terms means.
          </p>
        </div>
      ) : (
        <ul className={styles.reviewList}>
          {rows.map((assertion) => {
            const standing = standingOf(assertion);
            const busy = busyId === assertion.id;
            return (
              <li key={assertion.id} className={styles.reviewRow} data-tone={standing.tone}>
                <div className={styles.claim}>
                  <span className={styles.claimTerm}>{assertion.subjectLabel}</span>
                  <span className={styles.claimPredicate}>
                    <ArrowRight size={12} /> {assertion.predicate}
                  </span>
                  <span className={styles.claimTerm}>{assertion.objectLabel}</span>
                </div>

                <div className={styles.claimMeta}>
                  <span className={styles.scopeChip} data-tone={standing.tone}>
                    {standing.label}
                  </span>
                  {!assertion.directionCertain && (
                    <span className={styles.scopeProblem}>
                      <AlertTriangle size={12} /> direction unverified — recovered from
                      storage that could not record it
                    </span>
                  )}
                  {assertion.stale && (
                    <span className={styles.scopeProblem}>
                      <AlertTriangle size={12} /> its source is no longer in this notebook
                    </span>
                  )}
                  {assertion.note && (
                    <span className={styles.answerMeta}>“{assertion.note}”</span>
                  )}
                </div>

                {assertion.evidence.length > 0 && (
                  <ul className={styles.evidenceList}>
                    {assertion.evidence.map((row) => (
                      <li key={row.chunkId}>
                        <button
                          type="button"
                          className={styles.evidenceRow}
                          onClick={() => onOpenEvidence(row.documentSha256, row.page)}
                        >
                          <span className={styles.evidenceName}>page {row.page}</span>
                          {row.quote && (
                            <span className={styles.evidenceQuote}>{row.quote}</span>
                          )}
                        </button>
                      </li>
                    ))}
                  </ul>
                )}

                <div className={styles.reviewActions}>
                  {assertion.status !== 'accepted' && (
                    <button
                      type="button"
                      className={styles.linkAction}
                      disabled={busy}
                      onClick={() =>
                        void act(assertion.id, () =>
                          notebookResearchService.reviewAssertion(
                            notebookId,
                            assertion.id,
                            'accepted',
                          ),
                        )
                      }
                    >
                      <Check size={12} /> Accept
                    </button>
                  )}
                  {assertion.status !== 'rejected' && (
                    <button
                      type="button"
                      className={styles.linkAction}
                      disabled={busy}
                      onClick={() =>
                        void act(assertion.id, () =>
                          notebookResearchService.reviewAssertion(
                            notebookId,
                            assertion.id,
                            'rejected',
                          ),
                        )
                      }
                    >
                      <X size={12} /> Reject
                    </button>
                  )}
                  <button
                    type="button"
                    className={styles.linkAction}
                    disabled={busy}
                    title="Record it the other way round, as your own corrected claim"
                    onClick={() =>
                      void act(assertion.id, () =>
                        notebookResearchService.correctAssertion({
                          notebookId,
                          assertionId: assertion.id,
                          subject: assertion.object,
                          predicate: assertion.predicate,
                          object: assertion.subject,
                          note: 'direction corrected by hand',
                        }),
                      )
                    }
                  >
                    <Repeat size={12} /> Swap
                  </button>
                  <button
                    type="button"
                    className={styles.linkAction}
                    disabled={busy}
                    onClick={() => onAskAbout(assertion)}
                  >
                    Ask about this
                  </button>
                  {assertion.provenance === 'user' && (
                    <button
                      type="button"
                      className={styles.linkAction}
                      disabled={busy}
                      onClick={() =>
                        void act(assertion.id, () =>
                          notebookResearchService.deleteAssertion(notebookId, assertion.id),
                        )
                      }
                    >
                      <Trash2 size={12} /> Delete
                    </button>
                  )}
                </div>
              </li>
            );
          })}
        </ul>
      )}
    </section>
  );
};
