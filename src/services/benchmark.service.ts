/**
 * Service for the System Health page's benchmark section.
 *
 * The page renders the most recent measured `run_benchmark` result, and
 * nothing else. There is no synthetic row: the `synthetic_benchmark` command
 * that used to supply one — a fixed 38 tok/s stamped with the current time —
 * was removed, because the page rendered it in the same grid as measured
 * values under the words "Last measured".
 *
 * A machine that has recorded no benchmark has no benchmark, and the page says
 * so.
 */

import { getBackendService } from './api';

export interface BenchmarkRow {
  modelId: string;
  promptTokens: number;
  replyTokens: number;
  ttftMs: number;
  totalMs: number;
  tokensPerSecond: number;
  vramPeakMib: number;
  accuracyPct: number;
  at: string;
  hardwareTier: string;
}

export const benchmarkService = {
  /** Returns the most recent measured rows, newest first. */
  recent(limit?: number): Promise<BenchmarkRow[]> {
    return getBackendService().invoke<BenchmarkRow[]>('recent_benchmarks', {
      limit: limit ?? 5,
    });
  },
};
