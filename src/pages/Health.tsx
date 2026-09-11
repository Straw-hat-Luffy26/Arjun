import React, { useCallback, useEffect, useRef, useState } from 'react';
import { Activity, AlertTriangle, CircleCheck, CircleHelp, Cpu, Gauge, RefreshCw } from 'lucide-react';
import { healthService, type HealthItem, type HealthSnapshot, type Reading } from '../services/health.service';
import { benchmarkService, type BenchmarkRow } from '../services/benchmark.service';
import styles from './Health.module.css';

/**
 * How often the panel re-reads. Every source is local — a GPU query, a COUNT
 * against the on-disk index, an in-memory event log, the OS socket table — so
 * this costs nothing outside this machine, which is the point.
 */
const POLL_INTERVAL_MS = 5000;

const ICONS: Record<Reading, React.ReactNode> = {
  ok: <CircleCheck size={16} />,
  attention: <AlertTriangle size={16} />,
  unknown: <CircleHelp size={16} />,
};

/* Unknown gets its own treatment rather than borrowing "ok"'s. The whole
 * purpose of the state is that it cannot be mistaken for health. */
const STATE_CLASS: Record<Reading, string> = {
  ok: styles.stateOk,
  attention: styles.stateAttention,
  unknown: styles.stateUnknown,
};

const STATE_LABEL: Record<Reading, string> = {
  ok: 'Checked',
  attention: 'Needs attention',
  unknown: 'Could not check',
};

const Card = ({ item }: { item: HealthItem }) => (
  <article className={`${styles.card} ${STATE_CLASS[item.state]}`}>
    <header className={styles.cardHead}>
      <span className={styles.cardName}>{item.name}</span>
      <span className={styles.cardState} title={STATE_LABEL[item.state]}>
        {ICONS[item.state]}
      </span>
    </header>
    <p className={styles.cardValue}>{item.value}</p>
    <p className={styles.cardNote}>{item.note}</p>
  </article>
);

export const Health = () => {
  const [snapshot, setSnapshot] = useState<HealthSnapshot | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const [benchmark, setBenchmark] = useState<BenchmarkRow | null>(null);
  const [benchmarkError, setBenchmarkError] = useState<string | null>(null);

  const mountedRef = useRef(true);

  const refresh = useCallback(async () => {
    try {
      const next = await healthService.snapshot();
      if (!mountedRef.current) return;
      setSnapshot(next);
      setError(null);
    } catch (e) {
      if (!mountedRef.current) return;
      // A panel that silently stopped updating would be the worst outcome
      // here — it would look like steady health.
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  /**
   * Reads the most recent measured row, and nothing else.
   *
   * There used to be a second `try` here that fell through to
   * `benchmarkService.synthetic()` whenever no real run had been recorded. It
   * returned a fixed 38 tok/s stamped with the current time, and the grid below
   * rendered it exactly like a measurement, under the words "Last measured".
   * Both the command and the fallback are gone: a machine that has benchmarked
   * nothing shows the empty state, which is the true answer.
   */
  const refreshBenchmark = useCallback(async () => {
    try {
      const rows = await benchmarkService.recent(1);
      if (!mountedRef.current) return;
      setBenchmark(rows.length > 0 ? rows[0] : null);
      setBenchmarkError(null);
    } catch (e) {
      if (!mountedRef.current) return;
      // A read that failed is not "no benchmark" — it is not knowing, and the
      // section says which of the two it is rather than showing the empty
      // state for both.
      setBenchmark(null);
      setBenchmarkError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    mountedRef.current = true;
    void refresh();
    void refreshBenchmark();
    const timer = window.setInterval(() => void refresh(), POLL_INTERVAL_MS);
    return () => {
      mountedRef.current = false;
      window.clearInterval(timer);
    };
  }, [refresh, refreshBenchmark]);

  const onRefresh = async () => {
    setRefreshing(true);
    await refresh();
    setRefreshing(false);
  };

  const attention = snapshot?.items.filter((i) => i.state === 'attention') ?? [];
  const unknown = snapshot?.items.filter((i) => i.state === 'unknown') ?? [];

  return (
    <div className={styles.page}>
      <header className={styles.header}>
        <h1 className={styles.title}>Health</h1>
        <p className={styles.subtitle}>
          What this machine can currently do, read from this machine. No check here contacts
          anything outside it — no version check, no licence check, no status beacon. Each of
          those would appear in ARJUN&rsquo;s own network monitor a moment after it fired.
        </p>
      </header>

      {error && (
        <div className={styles.error} role="alert">
          <AlertTriangle size={18} />
          <div>
            <strong>The panel could not be read.</strong>
            <p>{error}</p>
          </div>
        </div>
      )}

      <div className={styles.summary}>
        <div className={styles.summaryMain}>
          <Activity size={18} />
          <div>
            <strong>
              {attention.length === 0 && unknown.length === 0
                ? 'Everything checked is in order'
                : [
                    attention.length > 0 && `${attention.length} needing attention`,
                    unknown.length > 0 && `${unknown.length} could not be checked`,
                  ]
                    .filter(Boolean)
                    .join(' · ')}
            </strong>
            <p>
              {snapshot
                ? `Read ${new Date(snapshot.takenAt).toLocaleTimeString()} · ${snapshot.externalCallsMade} external calls made`
                : 'Reading…'}
            </p>
          </div>
        </div>
        <button
          type="button"
          className={styles.refresh}
          onClick={() => void onRefresh()}
          disabled={refreshing}
        >
          <RefreshCw size={15} className={refreshing ? styles.spinning : undefined} />
          Refresh
        </button>
      </div>

      <section className={styles.benchSection}>
        <header className={styles.benchHeader}>
          <Gauge size={18} />
          <h2>Performance</h2>
          <button
            type="button"
            className={styles.benchButton}
            onClick={() => void refreshBenchmark()}
          >
            <RefreshCw size={14} /> Refresh
          </button>
        </header>
        {benchmark ? (
          <div className={styles.benchGrid}>
            <div className={styles.benchStat}>
              <span className={styles.benchLabel}>Tokens / sec</span>
              <span className={styles.benchValue}>{benchmark.tokensPerSecond.toFixed(1)}</span>
              <span className={styles.benchNote}>
                {benchmark.replyTokens}-token reply
              </span>
            </div>
            <div className={styles.benchStat}>
              <span className={styles.benchLabel}>Time to first token</span>
              <span className={styles.benchValue}>
                {benchmark.ttftMs > 0 ? `${benchmark.ttftMs} ms` : 'not recorded'}
              </span>
              <span className={styles.benchNote}>
                {benchmark.ttftMs > 0
                  ? 'measured from prompt to first received token'
                  : 'this run carried no first-token measurement'}
              </span>
            </div>
            <div className={styles.benchStat}>
              <span className={styles.benchLabel}>VRAM peak</span>
              <span className={styles.benchValue}>{benchmark.vramPeakMib} MiB</span>
              <span className={styles.benchNote}>
                nvidia-smi read at the end of the run
              </span>
            </div>
            <div className={styles.benchStat}>
              <span className={styles.benchLabel}>Accuracy on demo tasks</span>
              <span className={styles.benchValue}>{benchmark.accuracyPct.toFixed(0)}%</span>
              <span className={styles.benchNote}>
                hand-graded on tag ID, calculation, policy compliance
              </span>
            </div>
          </div>
        ) : benchmarkError ? (
          <p className={styles.benchEmpty}>
            <AlertTriangle size={14} /> The benchmark history could not be read, so
            nothing is claimed about performance on this machine: {benchmarkError}
          </p>
        ) : (
          <p className={styles.benchEmpty}>
            <Cpu size={14} /> Nothing has been measured on this machine yet. Run a
            benchmark and the row will appear here &mdash; until then this section
            stays empty rather than showing a figure from somewhere else.
          </p>
        )}
        {benchmark && (
          <p className={styles.benchMeta}>
            Last measured {new Date(benchmark.at).toLocaleString()} · hardware
            tier <code>{benchmark.hardwareTier}</code> · model{' '}
            <code>{benchmark.modelId}</code>
          </p>
        )}
      </section>

      <div className={styles.grid}>
        {snapshot?.items.map((item) => (
          <Card key={item.name} item={item} />
        ))}
      </div>
    </div>
  );
};
