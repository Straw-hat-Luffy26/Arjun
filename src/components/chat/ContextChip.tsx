import React, { useEffect, useMemo, useState } from 'react';
import { ChevronDown, ChevronUp, Database, Pin, X } from 'lucide-react';
import { useContextLedger } from '../run/runAdopt';
import { useConversation } from '../run/useConversation';
import {
  explainLedger,
  fitted,
  ledgerRows,
} from '../run/context-ledger';
import {
  driftSummary,
  entityRows,
  firstToGo,
  hasUnmeasuredTurns,
  type EntityRow,
} from '../run/context-entities';
import { agentService, type CompactionRecord } from '../../services/agent.service';
import styles from './ChatSurface.module.css';

/**
 * How a row's size reads.
 *
 * A document still being read shows "reading" rather than a number. It has a
 * real place in the next turn and an unknown size, and a zero beside it would
 * read as "this is free" — the opposite of what is about to be true.
 */
function tokenLabel(row: EntityRow): string {
  if (row.tokens === null) return 'reading…';
  // The tilde is the whole contract with the reader: it marks a figure nothing
  // has confirmed. Dropping it on a measured row and keeping it on an estimated
  // one is the only way a person can tell which totals to trust.
  return row.measured ? row.tokens.toLocaleString() : `~${row.tokens.toLocaleString()}`;
}

/**
 * The compact context meter that lives in the right side of the
 * composer.
 *
 * Re-uses the same shape as the legacy `ContextPanel` (a chip +
 * popover with a per-section breakdown) so a person switching between
 * workbench and chat sees the same numbers. Three states, distinguished
 * by luminance rather than colour:
 *  - **ok** (< 70% of window) — quiet grey
 *  - **tight** (70-90%) — amber
 *  - **critical** (≥ 90%) — red
 */
export function ContextChip() {
  const { conversation, activeRunId, activeMessageId } = useConversation();
  const latestRunId = useMemo(() => {
    if (activeRunId) return activeRunId;
    if (!conversation || conversation.runs.length === 0) return null;
    return conversation.runs[conversation.runs.length - 1].runId;
  }, [activeRunId, conversation]);

  // Two identities, because a turn has two. `latestRunId` moves from the
  // composer's correlation id to the run's own the moment `plan_ready` lands;
  // `activeMessageId` is reserved before the turn starts and never changes. The
  // ledger belongs to the run and the attachment costs belong to the turn — see
  // `useContextLedger`, where the split is explained.
  const { ledger, compactions, attachments, status, error } = useContextLedger(
    latestRunId,
    activeMessageId,
  );
  const [open, setOpen] = useState(false);
  /**
   * Rows the person has protected from eviction.
   *
   * Mirrors what the conversation holds: the pins are persisted by
   * `agent_pin_context` and read back below, and this is the copy the panel
   * draws from so a press is answered immediately rather than after a round
   * trip.
   */
  const [pinned, setPinned] = useState<ReadonlySet<string>>(new Set());

  /**
   * The stored pins, re-read whenever the conversation changes.
   *
   * Without this a pin survived only as long as the tab stayed open: it was
   * written to the conversation and never read back, so reopening the thread
   * showed every row unpinned while the backend went on protecting them. The
   * panel and the compactor would then disagree about what was being kept, and
   * the panel is the half a person believes.
   */
  const conversationId = conversation?.id ?? null;
  /**
   * The stored set as a *value*, not as an array reference.
   *
   * `conversation` is replaced on every streaming persist — several times a
   * second while a turn is running — and each replacement brings a new
   * `pinnedContext` array with identical contents. An effect depending on that
   * reference therefore re-ran constantly, and each run called `setPinned` with
   * the stored set.
   *
   * That is not merely wasteful: it fights the person. A pin press is optimistic
   * — the panel fills the icon immediately and the write goes out — so a reset
   * landing in the round trip pops the pin back off under their cursor, and
   * then it fills again when the write returns. Keying on the contents means
   * the effect fires when the pins actually change and at no other time.
   */
  const storedPinKey = JSON.stringify(conversation?.pinnedContext ?? []);
  useEffect(() => {
    setPinned(new Set(JSON.parse(storedPinKey) as string[]));
    // Keyed on the conversation as well, so switching threads resets the set
    // rather than carrying one conversation's pins into another's panel.
  }, [conversationId, storedPinKey]);

  const rows = useMemo(() => {
    const merged = entityRows(ledger, attachments);
    return merged.map(row => (pinned.has(row.id) ? { ...row, pinned: true } : row));
  }, [ledger, attachments, pinned]);

  const nextToGo = useMemo(() => firstToGo(rows), [rows]);
  const drift = useMemo(() => driftSummary(ledger), [ledger]);
  const unmeasured = useMemo(() => hasUnmeasuredTurns(ledger), [ledger]);

  /** Said out loud when a pin was pressed and nothing took it. */
  const [pinProblem, setPinProblem] = useState<string | null>(null);

  /**
   * Protect a row from eviction, or stop protecting it.
   *
   * ## Why this leaves the component
   *
   * It used to be four lines that toggled a `Set` in this file's own state, and
   * nothing else. The icon darkened, the "what goes first" line moved on to the
   * next row, and the compactor — which is in another process and had never
   * heard of any of this — cleared the pinned document on its next pass exactly
   * as if the button had never been pressed.
   *
   * That is the worst shape a control can have. Somebody who pinned the drawing
   * they were working from, watched the meter fill, and carried on had every
   * reason to believe the drawing was safe. A control that did nothing at all
   * would at least have left them looking for another way.
   *
   * The local state stays, because the panel has to answer the press
   * immediately and the round trip is not instant. What is new is that the set
   * is *sent*, and that a pin the runtime did not take says so rather than
   * looking exactly like one it did.
   *
   * The whole set goes every time, not a delta: unpinning matters as much as
   * pinning, and a call that could only add would make this a decision nobody
   * could take back.
   */
  const togglePin = (id: string) => {
    const next = new Set(pinned);
    if (next.has(id)) next.delete(id);
    else next.add(id);
    setPinned(next);
    setPinProblem(null);

    // A pin belongs to the conversation, not to whatever run happens to be in
    // flight, so it is stored even when nothing is running — that is what makes
    // it survive the turn it was pressed in. The run id is passed too, and is
    // what decides whether it also takes effect *now*.
    if (!conversationId) {
      setPinProblem('There is no conversation open to keep anything in.');
      return;
    }
    void agentService
      .pinContext(conversationId, activeRunId, [...next])
      .then(outcome => {
        if (!outcome.stored) {
          // The write is the half that matters, so a failure here means the pin
          // did not happen at all. Rolled back rather than left drawn: a filled
          // pin over an unprotected row is the lie this control exists to stop
          // telling.
          setPinned(pinned);
          setPinProblem(
            'That could not be kept — this conversation is not available to ' +
              'write to. Nothing is being protected.',
          );
        }
        // `appliedToRun === false` is not reported. It means no run was in
        // flight, which is the ordinary case for a pin pressed between turns,
        // and the pin is stored and will be honoured by the next one.
      })
      .catch((cause: unknown) => {
        setPinned(pinned);
        setPinProblem(
          `That could not be kept: ${
            cause instanceof Error ? cause.message : String(cause)
          }. Nothing is being protected.`,
        );
      });
  };

  const lastCompaction: CompactionRecord | null =
    compactions.length > 0 ? compactions[compactions.length - 1] : null;

  // A run whose documents are still being read has no ledger yet but does have
  // rows worth showing — that is the whole of the first turn, and a meter that
  // stays idle through it is blank exactly while somebody is watching it.
  if ((!ledger || ledger.committed === 0) && rows.length === 0) {
    // Expands, because it used to only pretend to.
    //
    // This branch returned the button alone while still toggling `open`, and
    // the card that reads `open` lives past the guard below — so clicking the
    // idle chip flipped a flag that nothing rendered. It looked frozen, which
    // is worse than looking disabled: a control that does nothing when pressed
    // reads as a broken application rather than as an empty state.
    return (
      <div className={styles.contextChipWrap}>
        <button
          type="button"
          className={styles.contextChipIdle}
          onClick={() => setOpen(o => !o)}
          aria-expanded={open}
          title="Context usage"
          data-state={status === 'failed' ? 'critical' : 'ok'}
        >
          <Database size={11} />
          {/* Three states, three labels. They used to be one — "No context
              yet" — which read the same whether the reading was on its way,
              genuinely absent, or unreadable. Those call for waiting, sending a
              message, and reporting a fault respectively, and a person could
              not tell which they were looking at. */}
          <span>
            {status === 'loading'
              ? 'Reading context…'
              : status === 'failed'
                ? 'Context unavailable'
                : 'No context yet'}
          </span>
        </button>
        {open && (
          <div className={styles.contextCard} role="dialog" aria-label="Context breakdown">
            <div className={styles.contextCardHeader}>
              <span>Context</span>
              <button
                type="button"
                className={styles.contextCardClose}
                onClick={() => setOpen(false)}
                aria-label="Close"
              >
                <X size={13} />
              </button>
            </div>
            {status === 'failed' ? (
              <p className={styles.contextWillNotFit}>
                The stored reading for this run could not be read
                {error ? `: ${error}` : '.'} The run itself is unaffected; this
                panel cannot say what its window holds.
              </p>
            ) : status === 'loading' ? (
              <p className={styles.contextEmptyNote}>
                Looking for this run&rsquo;s reading&hellip;
              </p>
            ) : (
              <p className={styles.contextEmptyNote}>
                Nothing has been measured yet. The window is itemised from the
                first model call of a turn, so this fills in once you send a
                message — and stays filled for the rest of the conversation.
              </p>
            )}
          </div>
        )}
      </div>
    );
  }

  // Past the guard the ledger may still be absent — documents can be read
  // before the run has made a model call. The chip then shows the rows it has
  // and no percentage, because a percentage of an unknown window is not a
  // number anybody can act on.
  const pct = ledger
    ? Math.min(100, Math.round((ledger.occupied / Math.max(1, ledger.window)) * 100))
    : null;
  const diagnosis = ledger ? explainLedger(ledger) : null;
  const willFit = ledger ? fitted(ledger) : null;

  return (
    <div className={styles.contextChipWrap}>
      <button
        type="button"
        className={styles.contextChip}
        onClick={() => setOpen(o => !o)}
        aria-expanded={open}
        title="Context usage"
        data-state={
          pct === null ? 'ok' : pct >= 90 ? 'critical' : pct >= 70 ? 'tight' : 'ok'
        }
      >
        <Database size={11} />
        <span>
          {/* Raw counts as well as the percentage: a share tells somebody how
              worried to be, and the count is what they need to compare against
              a document they are about to attach. */}
          {ledger && pct !== null ? (
            <>
              {pct}% ·{' '}
              {ledger.occupied.toLocaleString()}
              {ledger.window > 0 && ` / ${ledger.window.toLocaleString()}`}
            </>
          ) : (
            'Reading documents…'
          )}
        </span>
        {compactions.length > 0 && (
          <span className={styles.contextCompactCount}>×{compactions.length}</span>
        )}
        {open ? <ChevronUp size={10} /> : <ChevronDown size={10} />}
      </button>
      {open && (
        <div className={styles.contextCard} role="dialog" aria-label="Context breakdown">
          <div className={styles.contextCardHeader}>
            <strong>Context</strong>
            <button
              type="button"
              className={styles.contextCardClose}
              onClick={() => setOpen(false)}
              aria-label="Close"
            >
              <X size={12} />
            </button>
          </div>
          {/* Only when there is something to say. `explainLedger` returns null
              rather than a hedge, and printing "unclear" in its place would
              occupy the line a real diagnosis needs. */}
          {diagnosis && <p className={styles.contextDiagnosis}>{diagnosis}</p>}
          {willFit === false && (
            <p className={styles.contextWillNotFit}>
              The next turn would not fit in this model&rsquo;s window.
            </p>
          )}
          {/* What the compactor takes first, so the person moves the right
              thing. `null` is its own message: nothing can be reclaimed, so the
              next turn fails rather than degrades — which is the one case here
              worth interrupting somebody for. */}
          {willFit !== null &&
            (nextToGo ? (
              <p className={styles.contextCompactionLine}>
                If the window fills, <strong>{nextToGo.label}</strong> goes first.
              </p>
            ) : (
              <p className={styles.contextWillNotFit}>
                Nothing here can be reclaimed — everything is either structural or
                pinned. The next turn will fail rather than shorten.
              </p>
            ))}
          {/* The estimate-against-actual line. Absent when no call has reported
              usage, because "drift unknown" is not worth a line. */}
          {/* A pin that did not land. Loud, because the whole point of the
              control is that a person can rely on it — and a pin they believe
              took effect and did not is worse than no pin at all. */}
          {pinProblem && <p className={styles.contextWillNotFit}>{pinProblem}</p>}
          {drift && <p className={styles.contextCompactionLine}>{drift}</p>}
          {unmeasured && (
            <p className={styles.contextCompactionLine}>
              Some turns reported no usage, so part of this total is estimated
              rather than confirmed.
            </p>
          )}
          {/* Loud, because it means the rows below do not explain the bar
              above. Silent in normal operation. */}
          {(ledger?.itemisationErrors?.length ?? 0) > 0 && (
            <p className={styles.contextWillNotFit}>
              These rows do not add up to the totals they describe
              ({ledger?.itemisationErrors?.map(e => e.section).join(', ')}). Treat
              the breakdown as unreliable.
            </p>
          )}
          {compactions.length > 0 && (
            <p className={styles.contextCompactionLine}>
              {compactions.length} compaction
              {compactions.length === 1 ? '' : 's'} so far
              {lastCompaction && (
                <>
                  {' '}
                  · last: {lastCompaction.tokensBefore.toLocaleString()} →{' '}
                  {lastCompaction.tokensAfter.toLocaleString()} tokens
                </>
              )}
            </p>
          )}
          {/* The itemisation when there is one, the section breakdown when
              there is not. A record written before entities existed still has
              sections, and falling back to them beats an empty list under a
              full bar. */}
          <ul className={styles.contextList}>
            {rows.length > 0
              ? rows.map(row => (
                  <li
                    key={row.id}
                    className={
                      row.status === 'dropped' || row.status === 'summarised'
                        ? `${styles.contextRow} ${styles.contextRowReserved}`
                        : styles.contextRow
                    }
                    title={row.note ?? undefined}
                  >
                    <span className={styles.contextRowLabel}>
                      {row.label}
                      {/* A document that entered in part is the one thing on
                          this panel that changes how much an answer can be
                          trusted, so it is marked on the row itself and not
                          only in a tooltip. */}
                      {row.note && ' ⚠'}
                    </span>
                    <span className={styles.contextRowBar}>
                      <span
                        className={styles.contextRowFill}
                        style={{ width: `${Math.round(row.share * 100)}%` }}
                      />
                    </span>
                    <span className={styles.contextRowTokens}>{tokenLabel(row)}</span>
                    {row.evictable && (
                      <button
                        type="button"
                        className={styles.contextCardClose}
                        onClick={() => togglePin(row.id)}
                        aria-pressed={row.pinned}
                        aria-label={
                          row.pinned
                            ? `Allow ${row.label} to be dropped`
                            : `Keep ${row.label} when the window fills`
                        }
                        title={
                          row.pinned
                            ? 'Pinned — kept when the window fills'
                            : 'Keep this when the window fills'
                        }
                      >
                        <Pin
                          size={10}
                          style={{ opacity: row.pinned ? 1 : 0.35 }}
                        />
                      </button>
                    )}
                  </li>
                ))
              : ledger &&
                ledgerRows(ledger).map(row => (
                  <li
                    key={row.section}
                    className={
                      row.committedNotOccupied
                        ? `${styles.contextRow} ${styles.contextRowReserved}`
                        : styles.contextRow
                    }
                  >
                    <span className={styles.contextRowLabel}>{row.label}</span>
                    <span className={styles.contextRowBar}>
                      <span
                        className={styles.contextRowFill}
                        style={{ width: `${Math.round(row.share * 100)}%` }}
                      />
                    </span>
                    <span className={styles.contextRowTokens}>
                      {row.tokens.toLocaleString()}
                    </span>
                  </li>
                ))}
          </ul>
        </div>
      )}
    </div>
  );
}
