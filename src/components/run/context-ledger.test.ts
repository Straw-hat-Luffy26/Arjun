/**
 * What the context ledger tells somebody reviewing a run.
 *
 * The failures guarded here are all failures of the *explanation*, not of the
 * arithmetic: a screen that names the reserve as the thing that filled the
 * window, or that claims a run did not fit when nobody recorded a window, sends
 * an operator to change the one setting that will break the next run.
 */

import { describe, expect, it } from 'vitest';
import type { CompactionRecord, ContextLedgerRecord } from '../../services/agent.service';
import {
  LEDGER_SECTIONS,
  liveCompaction,
  mergeCompaction,
  compactionWarning,
  describeCompaction,
  explainLedger,
  fitted,
  largestSection,
  ledgerRows,
} from './context-ledger';

function ledger(over: Partial<ContextLedgerRecord> = {}): ContextLedgerRecord {
  const base: ContextLedgerRecord = {
    system: 400,
    skill: 0,
    toolSchema: 1_200,
    evidence: 300,
    notes: 150,
    transcript: 2_000,
    compaction: 250,
    reserve: 1_600,
    occupied: 4_300,
    committed: 5_900,
    window: 8_192,
    headroom: 2_292,
  };
  return { ...base, ...over };
}

function compaction(over: Partial<CompactionRecord> = {}): CompactionRecord {
  const base: CompactionRecord = {
    ordinal: 1,
    at: '2026-08-28T09:15:00+00:00',
    tokensBefore: 7_400,
    tokensAfter: 3_100,
    messagesSummarised: 24,
    refinedExistingSummary: false,
    toolResultsCleared: 0,
    ledger: ledger(),
  };
  return { ...base, ...over };
}

describe('the ledger rows', () => {
  it('shows every section, including the ones that are empty', () => {
    // A missing row is indistinguishable from a section that was never
    // measured, and the reader cannot tell which they are looking at.
    const rows = ledgerRows(ledger({ skill: 0 }));

    expect(rows.map(row => row.section)).toEqual([...LEDGER_SECTIONS]);
    expect(rows.find(row => row.section === 'skill')?.tokens).toBe(0);
  });

  it('marks the reserve as committed rather than occupied', () => {
    // Drawn the same as the rest, an operator reads it as space to reclaim —
    // and reserving less is the one change guaranteed to break the next run.
    const reserve = ledgerRows(ledger()).find(row => row.section === 'reserve');

    expect(reserve?.committedNotOccupied).toBe(true);
  });

  it('gives a share of zero rather than NaN when nothing was committed', () => {
    // A bar of width NaN renders full, which reads as "this filled the window".
    const rows = ledgerRows(
      ledger({
        system: 0,
        skill: 0,
        toolSchema: 0,
        evidence: 0,
        notes: 0,
        transcript: 0,
        compaction: 0,
        reserve: 0,
        occupied: 0,
        committed: 0,
      }),
    );

    for (const row of rows) {
      expect(Number.isFinite(row.share)).toBe(true);
      expect(row.share).toBe(0);
    }
  });
});

describe('naming what filled the window', () => {
  it('names the section that actually grew', () => {
    const explanation = explainLedger(ledger({ toolSchema: 5_000, committed: 9_000 }));

    expect(explanation).toContain('Tool definitions');
    expect(explanation).toContain('5,000');
  });

  it('never blames the reserve, whose only remedy would break the run', () => {
    // The reserve is set from the window by policy. On a small window it is
    // routinely the largest single number, and naming it sends the operator to
    // reduce the one thing that must not be reduced.
    const largest = largestSection(ledger({ reserve: 100_000 }));

    expect(largest?.section).not.toBe('reserve');
    expect(largest?.section).toBe('transcript');
  });

  it('says nothing rather than hedging when nothing was measured', () => {
    const empty = ledger({
      system: 0,
      skill: 0,
      toolSchema: 0,
      evidence: 0,
      notes: 0,
      transcript: 0,
      compaction: 0,
      reserve: 0,
      occupied: 0,
      committed: 0,
    });

    expect(explainLedger(empty)).toBeNull();
    expect(largestSection(empty)).toBeNull();
  });

  it('leaves the window out of the sentence when nobody recorded one', () => {
    const explanation = explainLedger(ledger({ window: 0 }));

    expect(explanation).not.toContain('window');
  });
});

describe('whether the next turn would have fitted', () => {
  it('is true with headroom and false without', () => {
    expect(fitted(ledger({ headroom: 2_292 }))).toBe(true);
    expect(fitted(ledger({ headroom: -40 }))).toBe(false);
  });

  it('declines to answer at all when the window is unknown', () => {
    // Not false. An unknown window is not evidence that something did not fit,
    // and "did not fit" on a run that completed is plainly wrong.
    expect(fitted(ledger({ window: 0 }))).toBeNull();
  });
});

describe('describing one compaction', () => {
  it('says how much was replaced and how much came back', () => {
    const described = describeCompaction(compaction());

    expect(described).toContain('24 message(s)');
    expect(described).toContain('4,300');
  });

  it('says plainly when a pass reclaimed nothing', () => {
    // The signal that the task no longer fits the model. Reported as "reclaimed
    // 0 tokens" it reads as a rounding detail rather than as the problem.
    const described = describeCompaction(compaction({ tokensBefore: 7_400, tokensAfter: 7_400 }));

    expect(described).toContain('reclaimed nothing');
  });

  it('distinguishes refining the existing summary from starting a new one', () => {
    expect(describeCompaction(compaction({ ordinal: 2, refinedExistingSummary: true }))).toContain(
      'refining the summary already held',
    );
    expect(
      describeCompaction(compaction({ ordinal: 2, refinedExistingSummary: false })),
    ).not.toContain('refining');
  });

  it('mentions cleared tool results only when there were some', () => {
    expect(describeCompaction(compaction({ toolResultsCleared: 3 }))).toContain(
      '3 raw tool result(s)',
    );
    expect(describeCompaction(compaction({ toolResultsCleared: 0 }))).not.toContain(
      'raw tool result',
    );
  });
});

describe('the warning above the compaction list', () => {
  it('warns when a later pass started a new summary instead of refining one', () => {
    // The failure that costs the reader trust in the answer: part of the run's
    // history is then described twice or not at all.
    const warning = compactionWarning([
      compaction({ ordinal: 1, refinedExistingSummary: false }),
      compaction({ ordinal: 2, refinedExistingSummary: false }),
    ]);

    expect(warning).toContain('started a new summary');
  });

  it('does not warn about the first pass, which has no summary to refine', () => {
    expect(
      compactionWarning([compaction({ ordinal: 1, refinedExistingSummary: false })]),
    ).toBeNull();
  });

  it('warns when compaction has stopped reclaiming room', () => {
    const warning = compactionWarning([
      compaction({ ordinal: 1, refinedExistingSummary: false }),
      compaction({
        ordinal: 2,
        refinedExistingSummary: true,
        tokensBefore: 7_000,
        tokensAfter: 7_000,
      }),
    ]);

    expect(warning).toContain('larger than the routed model');
  });

  it('stays quiet on a healthy run', () => {
    expect(
      compactionWarning([
        compaction({ ordinal: 1, refinedExistingSummary: false }),
        compaction({ ordinal: 2, refinedExistingSummary: true }),
      ]),
    ).toBeNull();
  });

  it('stays quiet when nothing compacted at all', () => {
    expect(compactionWarning([])).toBeNull();
  });
});

/**
 * The two sources of a run's compaction list, and why they must merge.
 *
 * A compaction reaches the panel twice: live, as the runtime performs it, and
 * again in the record written when the run ends. Appending both produced two
 * rows for one pass the moment a run finished while its panel was open, so the
 * chip read "x4 compactions" for a run that compacted twice — and that count is
 * the number an operator uses to decide whether to route the task to a bigger
 * model.
 */
describe('folding compactions from two sources', () => {
  const ledgerAt = (occupied: number): ContextLedgerRecord =>
    ({
      window: 32_000,
      occupied,
      committed: occupied,
      sections: {},
    }) as unknown as ContextLedgerRecord;

  const record = (ordinal: number, tokensAfter = 1_000): CompactionRecord => ({
    ordinal,
    at: '2026-09-06T03:00:00.000Z',
    tokensBefore: 8_000,
    tokensAfter,
    messagesSummarised: 4,
    refinedExistingSummary: ordinal > 1,
    toolResultsCleared: 0,
    ledger: ledgerAt(tokensAfter),
  });

  it('adds a compaction it has not seen', () => {
    expect(mergeCompaction([], record(1))).toHaveLength(1);
  });

  it('replaces rather than duplicates the same ordinal', () => {
    const held = mergeCompaction([], record(1, 1_000));
    const merged = mergeCompaction(held, record(1, 2_000));
    expect(merged).toHaveLength(1);
    // The later description of the same pass wins, which is what lets the
    // stored record correct a live event that was missing a field.
    expect(merged[0].tokensAfter).toBe(2_000);
  });

  it('keeps the list in ordinal order however the rows arrive', () => {
    let held = mergeCompaction([], record(3));
    held = mergeCompaction(held, record(1));
    held = mergeCompaction(held, record(2));
    expect(held.map(r => r.ordinal)).toEqual([1, 2, 3]);
  });

  it('folding a stored list into live rows leaves one row per pass', () => {
    const live = [record(1), record(2)].reduce(mergeCompaction, [] as CompactionRecord[]);
    const withStored = [record(1), record(2)].reduce(mergeCompaction, live);
    expect(withStored).toHaveLength(2);
  });
});

/**
 * Building a compaction row from the live event.
 *
 * The runtime has always sent the whole record; the wire type declared three of
 * its fields, so nothing could be built and the count only ever came from the
 * record written after the run — the one time nobody is watching it.
 */
describe('reading a compaction off the live channel', () => {
  const full = {
    type: 'context_compacted' as const,
    tokensBefore: 9_000,
    tokensAfter: 2_500,
    messagesSummarised: 6,
    ordinal: 2,
    at: '2026-09-06T03:05:00.000Z',
    refinedExistingSummary: true,
    toolResultsCleared: 3,
    ledger: { window: 32_000, occupied: 2_500, committed: 2_500, sections: {} },
  };

  it('carries every number the event holds', () => {
    const built = liveCompaction(full as never);
    expect(built).not.toBeNull();
    expect(built!.ordinal).toBe(2);
    expect(built!.tokensBefore).toBe(9_000);
    expect(built!.tokensAfter).toBe(2_500);
    expect(built!.messagesSummarised).toBe(6);
    expect(built!.refinedExistingSummary).toBe(true);
    expect(built!.toolResultsCleared).toBe(3);
  });

  it('uses the runtime timestamp, not the reader clock', () => {
    const built = liveCompaction(full as never, () => 'LOCAL-CLOCK');
    expect(built!.at).toBe('2026-09-06T03:05:00.000Z');
  });

  it('falls back to the local clock only when the event carries no time', () => {
    const { at: _dropped, ...withoutTime } = full;
    const built = liveCompaction(withoutTime as never, () => 'LOCAL-CLOCK');
    expect(built!.at).toBe('LOCAL-CLOCK');
  });

  /**
   * An invented ordinal merges with somebody else's row, so a frame without one
   * is skipped. A wrong row is worse than a missing one.
   */
  it('refuses to build a row it would have to invent an ordinal for', () => {
    const { ordinal: _dropped, ...withoutOrdinal } = full;
    expect(liveCompaction(withoutOrdinal as never)).toBeNull();
  });

  it('refuses to build a row with no ledger to show', () => {
    const { ledger: _dropped, ...withoutLedger } = full;
    expect(liveCompaction(withoutLedger as never)).toBeNull();
  });
});
