/**
 * Turning a context ledger into the sentence somebody actually needs.
 *
 * ## Why this is not just a table
 *
 * The backend hands over eight token counts and a window size. Drawn as a
 * table, that tells an operator nothing they can act on — they do not know what
 * a healthy `toolSchema` figure looks like, and they should not have to.
 *
 * What they need is the answer to one question: *why did this run keep losing
 * its history?* There are only a few real answers, and each has a different
 * remedy:
 *
 * - **The tool schemas dominate.** Too many tools are offered at once. Fix the
 *   catalogue; the run itself is fine.
 * - **Evidence dominates.** Whole documents are reaching the window instead of
 *   references. Fix retrieval.
 * - **The transcript dominates.** The task is genuinely long. Route it to a
 *   model with a larger window.
 * - **The notes dominate.** The bound is wrong, which is a defect here.
 *
 * So this module ranks the sections and names the largest, and the screen shows
 * that sentence above the numbers rather than instead of them.
 *
 * Kept apart from the component and free of JSX so it can be tested without a
 * DOM implementation this repository does not vendor — the same split
 * `recovery.ts` already uses.
 */

import type {
  AgentEvent,
  CompactionRecord,
  ContextLedgerRecord,
} from '../../services/agent.service';

/**
 * Folds one compaction into the list, keyed by its ordinal.
 *
 * A run's compactions are numbered 1..n by the runtime, so the ordinal is the
 * identity. Merging on it is what lets the live event and the stored record
 * describe the same pass without producing two rows for it — which is what an
 * append-only list would do the moment a run finished while its panel was open.
 *
 * Sorted, because the two sources do not arrive in order: the stored record is
 * fetched once and may land after several live events.
 */
export function mergeCompaction(
  current: CompactionRecord[],
  arriving: CompactionRecord,
): CompactionRecord[] {
  const at = current.findIndex(held => held.ordinal === arriving.ordinal);
  if (at === -1) {
    return [...current, arriving].sort((a, b) => a.ordinal - b.ordinal);
  }
  const next = current.slice();
  next[at] = arriving;
  return next;
}

/**
 * Builds a compaction record from a live event, or `null` if it cannot.
 *
 * ## Why this can fail
 *
 * The runtime has always sent the whole record; the wire type declared three of
 * its seven fields. So nothing could be built from a live event, and the
 * compaction count in the meter came only from the record written *after* the
 * run — which is the one time nobody is watching it.
 *
 * A frame genuinely missing its ordinal is skipped rather than given one: an
 * invented ordinal merges with somebody else's row, and a wrong row is worse
 * than a missing one.
 */
export function liveCompaction(
  event: Extract<AgentEvent, { type: 'context_compacted' }>,
  now: () => string = () => new Date().toISOString(),
): CompactionRecord | null {
  if (typeof event.ordinal !== 'number' || !event.ledger) return null;
  return {
    ordinal: event.ordinal,
    // Stamped by the runtime, which is the side that knows when it happened.
    // A missing one is filled from this clock — a worse answer, but one field
    // of a row whose numbers are all real.
    at: event.at ?? now(),
    tokensBefore: event.tokensBefore,
    tokensAfter: event.tokensAfter,
    messagesSummarised: event.messagesSummarised,
    refinedExistingSummary: event.refinedExistingSummary ?? false,
    toolResultsCleared: event.toolResultsCleared ?? 0,
    ledger: event.ledger,
  };
}

/** The sections, in the order they are shown. */
export const LEDGER_SECTIONS = [
  'system',
  'skill',
  'toolSchema',
  'evidence',
  'notes',
  'transcript',
  'compaction',
  'reserve',
] as const;

export type LedgerSection = (typeof LEDGER_SECTIONS)[number];

/** What each section is, in the words an operator would use. */
const SECTION_LABELS: Record<LedgerSection, string> = {
  system: 'System prompt',
  skill: 'Skill guidance',
  toolSchema: 'Tool definitions',
  evidence: 'Retrieved evidence',
  notes: 'Working notes',
  transcript: 'Conversation',
  compaction: 'Summary of earlier turns',
  reserve: 'Reserved for the reply',
};

/** One row of the ledger, ready to draw. */
export interface LedgerRow {
  section: LedgerSection;
  label: string;
  tokens: number;
  /**
   * Share of the committed total, 0–1. Zero when nothing is committed, rather
   * than `NaN` — a bar of width `NaN` renders as a full bar, which would read
   * as "this section filled the window".
   */
  share: number;
  /**
   * True for `reserve`, which is committed by policy rather than grown by the
   * run. Drawn differently, because an operator reading it as occupied space
   * will try to reclaim it, and reserving less is the one change guaranteed to
   * make the run fail.
   */
  committedNotOccupied: boolean;
}

export function ledgerRows(ledger: ContextLedgerRecord): LedgerRow[] {
  const total = Math.max(1, ledger.committed);
  return LEDGER_SECTIONS.map(section => ({
    section,
    label: SECTION_LABELS[section],
    tokens: ledger[section],
    share: ledger.committed > 0 ? ledger[section] / total : 0,
    committedNotOccupied: section === 'reserve',
  }));
}

/**
 * The section that grew most, excluding the reserve.
 *
 * `null` when nothing was measured, which happens on a run that never
 * compacted and on a record written before the ledger existed. Both are cases
 * where saying nothing is right and inventing a diagnosis is not.
 */
export function largestSection(ledger: ContextLedgerRecord): LedgerRow | null {
  const ranked = ledgerRows(ledger)
    .filter(row => !row.committedNotOccupied && row.tokens > 0)
    .sort((a, b) => b.tokens - a.tokens);
  return ranked[0] ?? null;
}

/**
 * What filled the window, in one sentence.
 *
 * Returns `null` rather than a hedge when there is nothing to say. A line that
 * reads "the context was mostly unclear" is worse than no line, because it
 * occupies the space where a real diagnosis would have gone.
 */
export function explainLedger(ledger: ContextLedgerRecord): string | null {
  const largest = largestSection(ledger);
  if (!largest || ledger.committed === 0) return null;

  const share = Math.round(largest.share * 100);
  const of = ledger.window > 0 ? ` of a ${ledger.window.toLocaleString()}-token window` : '';
  return `${largest.label} took the most room — ${largest.tokens.toLocaleString()} tokens, ${share}% of the ${ledger.committed.toLocaleString()} committed${of}.`;
}

/**
 * Whether the next turn would have fitted.
 *
 * `null` when the window is unknown. Not `false`: an unknown window is not
 * evidence that something did not fit, and a screen that says "did not fit"
 * about a run that completed would be plainly wrong.
 */
export function fitted(ledger: ContextLedgerRecord): boolean | null {
  if (ledger.window <= 0) return null;
  return ledger.headroom >= 0;
}

/** One compaction, described the way somebody reviewing the run would say it. */
export function describeCompaction(record: CompactionRecord): string {
  const reclaimed = record.tokensBefore - record.tokensAfter;
  const start = `Compaction ${record.ordinal} replaced ${record.messagesSummarised} message(s)`;
  const saved =
    reclaimed > 0
      ? ` and reclaimed ${reclaimed.toLocaleString()} tokens`
      : // Non-positive happens when the summary is no smaller than what it
        // replaced. Worth saying plainly: it means the run has reached the point
        // where compaction has stopped buying anything.
        ' and reclaimed nothing — the summary was no smaller than the turns it replaced';
  const refined = record.refinedExistingSummary ? ', refining the summary already held' : '';
  const cleared =
    record.toolResultsCleared > 0
      ? `. ${record.toolResultsCleared} raw tool result(s) had already been replaced by evidence references.`
      : '.';
  return `${start}${saved}${refined}${cleared}`;
}

/**
 * The warning to show above a run's compaction list, if any.
 *
 * Two things are worth interrupting somebody for, and nothing else is:
 *
 * - **A compaction that started a new summary rather than refining one.** That
 *   means the earlier half of the run is described twice or not at all, and any
 *   answer resting on it should be re-checked.
 * - **Compaction that has stopped reclaiming anything.** The task no longer
 *   fits the model it was given, and further passes only lose history.
 */
export function compactionWarning(records: CompactionRecord[]): string | null {
  const restarted = records.some(
    (record, index) => index > 0 && !record.refinedExistingSummary,
  );
  if (restarted) {
    return 'One of these compactions started a new summary instead of extending the previous one, so part of this run’s earlier history may be described twice or not at all. Treat anything answered after that point as needing a second look.';
  }
  const unproductive = records.filter(record => record.tokensBefore - record.tokensAfter <= 0);
  if (unproductive.length > 0) {
    return 'Compaction stopped reclaiming room in this run. The task is larger than the routed model’s window, and further summarising only loses history — route work like this to a model with a larger window.';
  }
  return null;
}
